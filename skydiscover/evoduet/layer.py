"""Gate retrieval, run observed search, and credit evidence with evaluated outcomes."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional, Union

from skydiscover.evoduet.analysis import KnowledgeAnalyzer
from skydiscover.evoduet.generation import SolutionModelPool, active_solution_model
from skydiscover.evoduet.history import verbalize_history
from skydiscover.evoduet.lookup_provenance import snapshot_lookup_selection
from skydiscover.evoduet.query_construction import QueryConstruction
from skydiscover.evoduet.retrieval import (
    SearchResult,
    SearchStore,
    TavilyRetrieval,
)
from skydiscover.evoduet.retrieval_gating import (
    AlwaysRetrievalGating,
    GateDecision,
    GateResult,
    HeuristicRetrievalGating,
    LLMRetrievalGating,
)
from skydiscover.evoduet.scoring import finite_score
from skydiscover.evoduet.search import ObservedSearch
from skydiscover.evoduet.search_context import render_search_experience
from skydiscover.evoduet.selective_generation import record_gate_action
from skydiscover.evoduet.stagnation_gating import EvoXStagnationRetrievalGating
from skydiscover.evoduet.trace import (
    iteration_dir,
    make_step,
    persist_query_trace,
    record_gate_decision,
    record_query_evolution,
)
from skydiscover.utils.metrics import get_score, is_numeric_metric

logger = logging.getLogger("skydiscover.evoduet")
EVODUET_STATE_VERSION = 1


class EvoDuet(ObservedSearch):
    """Observed web search for the supported discovery presets."""

    def __init__(self, config, output_dir=None):
        self.config = config
        self.output_dir = output_dir
        wk = config.evoduet
        if wk.retrieval_gating_backend_type == "stagnation" and config.max_parallel_iterations != 1:
            raise ValueError(
                "stagnation retrieval gating requires max_parallel_iterations=1; "
                "parallel solution candidates within one iteration are supported"
            )
        self.solution_model_pool = SolutionModelPool(config.llm.models, wk)
        self.gating = self._select_gating_backend()
        self.query_construction = QueryConstruction(self.solution_model_pool)
        self.retrieval = self._select_retrieval_backend()
        self.analyzer = KnowledgeAnalyzer(self.solution_model_pool._generate, wk)
        self.query_optimization_history: list[dict] = []
        self._pending_query_optimizations: dict[int, dict] = {}
        selection = wk.search_selection
        self.search_store = SearchStore(
            output_dir=output_dir,
            search_selection_policy=selection.policy,
            search_selection_num=selection.num or wk.top_k_for_retrieval,
            search_selection_criterion=selection.criterion,
            documents_per_entry=wk.documents_per_entry,
            max_document_chars=wk.max_document_chars,
        )
        if output_dir:
            _save_evoduet_config(wk, output_dir)

    def preflight(self) -> None:
        """Validate the external retrieval backend before model calls are spent."""
        check = getattr(self.retrieval, "preflight", None)
        if callable(check):
            check()

    def _select_retrieval_backend(self):
        return TavilyRetrieval(self.config)

    def _select_gating_backend(self):
        backend = self.config.evoduet.retrieval_gating_backend_type
        if backend == "llm":
            return LLMRetrievalGating(self.config, llm=self.solution_model_pool)
        if backend == "stagnation":
            return EvoXStagnationRetrievalGating(self.config)
        if backend == "heuristic":
            return HeuristicRetrievalGating(self.config)
        if backend == "always":
            return AlwaysRetrievalGating(self.config)
        raise ValueError(f"Unsupported retrieval gating type: {backend!r}")

    def _selected_search_context(self):
        return render_search_experience(self.search_store, self.query_optimization_history)

    def _persist_trace(self, trace):
        persist_query_trace(self.output_dir, trace)

    async def run(self, *, solution_model=None, **kwargs) -> List[SearchResult]:
        """Use one solution model for this iteration without changing other tasks."""
        token = active_solution_model.set(solution_model)
        try:
            return await self._run(**kwargs)
        finally:
            active_solution_model.reset(token)

    async def _run(
        self,
        *,
        parent,
        history,
        population=None,
        iteration=None,
        task_context="",
        target_score=None,
    ):
        # Every gate precedes population analysis. LLM gating sees only current
        # program, native history and recorded search experiences; it writes knowledge
        # analysis as part of its own response.
        # Standard controllers call WKL before generating the next solution; the
        # population therefore contains outcomes completed before this iteration.
        population = list(population or [])
        history_text = verbalize_history(history)
        stagnation = isinstance(self.gating, EvoXStagnationRetrievalGating)
        search_database = None if stagnation else self._selected_search_context()
        gate_request = dict(
            parent=parent,
            history=history_text,
            search_store=search_database,
            iteration=iteration,
        )
        if stagnation:
            gate_request["best_score"] = _population_best_score(population)
        decide = getattr(self.gating, "decide_with_details", None) or self.gating.decide
        outcome = await decide(**gate_request)
        gate_result = outcome if isinstance(outcome, GateResult) else GateResult(outcome)
        record_gate_action(self, gate_result.decision, iteration)
        gate_knowledge = gate_result.knowledge_state_analysis[
            : self.config.evoduet.analysis_max_chars
        ]
        document_ids = (
            gate_result.search_document_ids if gate_result.decision == GateDecision.LOOK_UP else ()
        )
        details = {
            "knowledge_state_analysis": gate_knowledge,
            "reasoning": gate_result.reasoning,
            "search_document_ids": list(document_ids),
        }
        if stagnation:
            details.update(backend="stagnation", stagnation=self.gating.diagnostics)
        record_gate_decision(
            self.output_dir, iteration, gate_result.decision.value, details=details
        )
        if gate_result.decision == GateDecision.NO_OP:
            self._snapshot_search_db(iteration)
            return []

        target_id, fallback_score = _program_target(parent)
        target = dict(
            target_program_id=target_id,
            target_score=target_score if target_score is not None else fallback_score,
        )
        if gate_result.decision == GateDecision.LOOK_UP:
            documents = self._lookup_documents(document_ids, iteration=iteration, **target)
            self._snapshot_search_db(iteration)
            return documents

        if search_database is None:
            search_database = self._selected_search_context()
        context = await self.analyzer.analyze(
            population,
            search_database,
            iteration,
            task_context=task_context,
            parent=parent,
            history=history_text,
        )
        if gate_knowledge:
            context = replace(context, knowledge_state_analysis=gate_knowledge)
        details["knowledge_state_analysis"] = context.knowledge_state_analysis
        details["population_state_analysis"] = context.population_analysis
        record_gate_decision(
            self.output_dir, iteration, gate_result.decision.value, details=details
        )
        request = dict(
            parent=parent,
            history=history_text,
            iteration=iteration,
            task_context=task_context,
            **target,
        )
        documents = await self._retrieve_with_query(**request, context=context)
        self._snapshot_search_db(iteration)
        return documents

    def _lookup_documents(
        self, document_ids, *, iteration, target_program_id, target_score
    ) -> List[SearchResult]:
        """Reuse stored bodies and record a new evidence event for the current parent."""
        stored = self.search_store.lookup_documents(document_ids)
        selection = snapshot_lookup_selection(self.search_store, stored)
        # Preserve every stored document field on the new evidence event, whose
        # evaluation credit belongs to the current parent. Prompt conversion is
        # applied to separate copies so the recorded snippet remains intact.
        record = self.search_store.add(
            None,
            copy.deepcopy(stored),
            iteration=iteration,
            target_program_id=target_program_id,
            target_score=target_score,
            operation="look-up",
            lookup_provenance=selection["document_provenance"],
        )
        documents = copy.deepcopy(record.search_results)
        for document in documents:
            document.content = document.raw_content or document.content
        step = make_step(0, "look-up", None, documents)
        step.update(selection, search_record_id=record.id)
        resolved_search_ids = {document.search_document_id for document in documents}
        resolved_ids = resolved_search_ids | {
            document.id
            for document in self.search_store.documents()
            if document.search_document_id in resolved_search_ids
        }
        step["requested_doc_ids"] = list(document_ids)
        step["missing_doc_ids"] = [value for value in document_ids if value not in resolved_ids]
        record_query_evolution(
            self.output_dir,
            iteration,
            decision="look-up",
            mode="look-up",
            steps=[step],
        )
        record = self.search_store.records[-1]
        trace = {
            "iteration": iteration,
            "parent_id": target_program_id,
            "parent_score": finite_score(target_score),
            "operation": "look-up",
            "mode": "observed_search",
            "outcome_scope": "selected_evidence",
            "estimated_child_score": None,
            "generation_condition": self._generation_condition(),
            "rounds": [],
        }
        self.query_optimization_history.append(trace)
        # Shared write-back requires a numeric baseline. Missing lookup scores
        # still remain faithfully stored by SearchStore, without a forecast.
        if trace["parent_score"] is None:
            trace.update(search_record_id=record.id, status="invalid_parent_score")
            self._persist_trace(trace)
        else:
            self._stage_trace(trace, record)
        return documents

    def state_dict(self) -> dict[str, Any]:
        """Return the layer's learned state for the program checkpoint."""
        state = {
            "schema_version": EVODUET_STATE_VERSION,
            "store": self.search_store.state_dict(),
            "query_optimization_history": copy.deepcopy(
                getattr(self, "query_optimization_history", [])
            ),
        }
        if isinstance(getattr(self, "gating", None), EvoXStagnationRetrievalGating):
            state["stagnation_gating"] = self.gating.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the exact search memory captured by a program checkpoint."""
        if (
            not isinstance(state, dict)
            or type(state.get("schema_version")) is not int
            or state["schema_version"] != EVODUET_STATE_VERSION
        ):
            raise ValueError("unsupported EvoDuet checkpoint format; start a new run")
        required = {"schema_version", "store", "query_optimization_history"}
        if not required <= state.keys() or state.keys() - required - {"stagnation_gating"}:
            raise ValueError("invalid EvoDuet checkpoint fields")
        history = state["query_optimization_history"]
        if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
            raise ValueError("invalid EvoDuet query optimization history")
        history = copy.deepcopy(history)
        pending = {}
        for record in history:
            if record.get("status") == "awaiting_outcome":
                iteration = record.get("iteration")
                if type(iteration) is not int or iteration < 0:
                    raise ValueError("invalid pending EvoDuet query optimization")
                pending[iteration] = record
        restored_gate = None
        if isinstance(getattr(self, "gating", None), EvoXStagnationRetrievalGating):
            if "stagnation_gating" not in state:
                raise ValueError("checkpoint is missing stagnation retrieval gating state")
            # Validate before changing the current store or gate.
            restored_gate = EvoXStagnationRetrievalGating(self.config)
            restored_gate.load_state_dict(state["stagnation_gating"])
        self.search_store.load_state_dict(state["store"])
        self.query_optimization_history = history
        self._pending_query_optimizations = pending
        if restored_gate is not None:
            self.gating = restored_gate

    def record_usage(
        self,
        *,
        iteration: int,
        score: Optional[float] = None,
        feedback: Optional[str] = None,
        result_program_id: str = "",
    ) -> None:
        """Credit the exact search record staged when this child was requested."""
        try:
            self.search_store.record_usage(
                iteration,
                score=score,
                feedback=feedback,
                result_program_id=result_program_id,
            )
        except Exception:
            logger.exception("EvoDuet: record_usage failed")
        trace = getattr(self, "_pending_query_optimizations", {}).pop(iteration, None)
        if trace is not None:
            child_score = finite_score(score)
            improvement = (
                finite_score(child_score - trace["parent_score"])
                if child_score is not None
                else None
            )
            trace.update(
                status="evaluated" if improvement is not None else "not_evaluated",
                result_program_id=result_program_id,
                result_score=child_score,
                actual_improvement=improvement,
                feedback=feedback,
            )
            self._persist_trace(trace)
        self._snapshot_search_db(iteration)

    def staged_search_record_id(self, iteration: int) -> Optional[str]:
        return self.search_store.staged_record_id(iteration)

    def discard_usage(self, iteration: int) -> None:
        """Forget staged evidence when prompt augmentation did not complete."""
        discard = getattr(self.search_store, "discard_usage", None)
        if callable(discard):
            discard(iteration)
        trace = getattr(self, "_pending_query_optimizations", {}).pop(iteration, None)
        if trace is not None:
            trace["status"] = "discarded"
            self._persist_trace(trace)

    def _snapshot_search_db(self, iteration) -> None:
        directory = iteration_dir(self.output_dir, iteration)
        if directory is None:
            return
        try:
            self.search_store.snapshot(directory)
        except Exception:
            logger.exception("EvoDuet: search-db snapshot failed at iteration %s", iteration)


def _population_best_score(population):
    """Use the full retained population's fitness, not the sampled parent's score.

    The controller attaches the database's higher-is-better fitness proxy as
    evoduet_score; it equals combined_score for ordinary scalar runs.
    Plain Program callers may supply combined_score directly.
    """
    scores = []
    for program in population:
        metrics = (
            program.get("metrics", {})
            if isinstance(program, Mapping)
            else getattr(program, "metrics", {})
        )
        if not isinstance(metrics, Mapping):
            continue
        for field in ("evoduet_score", "combined_score"):
            value = metrics.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            try:
                score = float(value)
            except (ValueError, OverflowError):
                continue
            if math.isfinite(score):
                scores.append(score)
                break
    return max(scores, default=None)


def _program_target(program) -> tuple[str, Optional[float]]:
    metrics = getattr(program, "metrics", None) or {}
    score = None
    # get_score defaults to zero for non-score metadata. That is not an observed
    # evaluator baseline, especially when query optimization compares against it.
    combined = metrics.get("combined_score")
    has_numeric_score = any(is_numeric_metric(value) for value in metrics.values())
    if isinstance(combined, str):
        try:
            has_numeric_score = math.isfinite(float(combined)) or has_numeric_score
        except (ValueError, OverflowError):
            pass
    if has_numeric_score and not isinstance(combined, bool):
        try:
            candidate = float(get_score(metrics))
            score = candidate if math.isfinite(candidate) else None
        except (TypeError, ValueError, OverflowError):
            pass
    return getattr(program, "id", "") or "", score


def _document_body(document, max_chars: int) -> str:
    text = getattr(document, "content", "") or getattr(document, "raw_content", "") or ""
    return str(text).replace("```", "~~~")[:max_chars]


def _redact_secrets(obj) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "api_key" and value:
                obj[key] = "***"
            else:
                _redact_secrets(value)
    elif isinstance(obj, list):
        for item in obj:
            _redact_secrets(item)


def _save_evoduet_config(evoduet_config, output_dir) -> None:
    data = dataclasses.asdict(evoduet_config)
    _redact_secrets(data)
    path = Path(output_dir) / "evoduet" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _web_block(index: int, document, max_document_chars: int) -> str:
    """One retrieved page: where it came from, then what it says."""
    return (
        f"## Web Document {index}\n"
        f"Title: {document.title or document.url or document.id}\n"
        f"URL: {document.url or '(not provided)'}\n"
        f"Content: {_document_body(document, max_document_chars)}"
    )


EVIDENCE_HEADER = "# Helpful Knowledge"


def format_documents(
    documents: Union[str, List[SearchResult]],
    max_document_chars: int = 10_000,
) -> str:
    """Render bounded, fence-neutralized documents as the mutation prompt's evidence block."""
    if not documents:
        return ""
    if isinstance(documents, str):
        return documents.replace("```", "~~~")
    blocks = [
        _web_block(index, document, max_document_chars)
        for index, document in enumerate(documents, 1)
    ]
    return f"{EVIDENCE_HEADER}\n\n" + "\n\n".join(blocks)
