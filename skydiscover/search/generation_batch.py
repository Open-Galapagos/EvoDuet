"""Generate N candidates for one frozen prompt, evaluate them, and select one.

The controller keeps the single-candidate path unchanged. This module owns only
the opt-in batch, with process-local limits shared across concurrent iterations.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from skydiscover import solution_confidence
from skydiscover.evaluation import EvaluationResult
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.label_scoring import finite_json
from skydiscover.llm.response_metadata import json_safe, reasoning_content
from skydiscover.search.base_database import Program
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import parse_full_rewrite

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    index: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response: LLMResponse | None = None
    solution: str | None = None
    changes: str | None = None
    image_path: str | None = None
    evaluation: EvaluationResult | None = None
    confidence: dict | None = None
    score: float | None = None
    status: str = "pending"
    error: str | None = None
    calls: dict = field(default_factory=lambda: {"calls": []})
    searches: dict = field(default_factory=lambda: {"tavily_searches": []})

    def ranking_program(self) -> Program:
        return Program(
            id=self.id,
            solution=self.solution or "",
            metrics=self.evaluation.metrics if self.evaluation is not None else {},
        )

    def metadata(self, selected: bool) -> dict:
        return finite_json(
            json_safe(
                {
                    "sample_index": self.index,
                    "program_id": self.id,
                    "selected": selected,
                    "status": self.status,
                    "error": self.error,
                    "score": self.score,
                    "solution": self.solution,
                    "llm_response": self.response.text if self.response is not None else None,
                    "metrics": self.evaluation.metrics if self.evaluation is not None else {},
                    "artifacts": self.evaluation.artifacts if self.evaluation is not None else {},
                    "image_path": self.image_path,
                    # Full billable records are stored once, on the iteration result.
                    "llm_call_ids": [call.get("llm_call_id") for call in self.calls["calls"]],
                    "solution_confidence_record_id": (self.confidence or {}).get("record_id"),
                }
            )
        )


@dataclass
class GenerationBatch:
    prompt: dict
    attempt: int
    candidates: list[Candidate]
    winner: Candidate | None = None
    generation_time: float = 0.0
    evaluation_time: float = 0.0

    def metadata(self) -> dict:
        return {
            "attempt": self.attempt,
            "mode": "parallel_requests",
            "num_generations": len(self.candidates),
            "num_evaluated": sum(c.evaluation is not None for c in self.candidates),
            "num_valid": sum(c.status == "evaluated" for c in self.candidates),
            "selected_index": self.winner.index if self.winner is not None else None,
            "prompt": copy.deepcopy(self.prompt),
            "candidates": [c.metadata(c is self.winner) for c in self.candidates],
        }

    def failed_attempts(self, parent: Program) -> list[dict]:
        return [
            {
                "solution": c.solution or (c.response.text if c.response is not None else ""),
                "metrics": c.evaluation.metrics if c.evaluation is not None else {},
                "metadata": {
                    "changes": c.changes,
                    "parent_metrics": parent.metrics,
                    "error": c.error,
                    "attempt_number": self.attempt,
                    "sample_index": c.index,
                },
            }
            for c in self.candidates
        ]


async def _gather_and_drain(coroutines) -> None:
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class CandidateBatchRunner:
    def __init__(self, config):
        self.count = config.num_generations
        self.generation_slots = asyncio.Semaphore(config.max_parallel_generations or self.count)
        self.evaluation_slots = asyncio.Semaphore(config.max_parallel_evaluations)

    async def run(
        self,
        controller,
        *,
        prompt: dict,
        parent: Program | None,
        iteration: int,
        attempt: int,
        retry_times: int,
        confidence_template: dict | None,
        llm_reasoning: dict,
        web_search_results: dict,
    ) -> GenerationBatch:
        batch = GenerationBatch(
            copy.deepcopy(prompt), attempt, [Candidate(i) for i in range(self.count)]
        )
        parent_id = parent.id if parent is not None else None
        parent_solution = parent.solution if parent is not None else None
        # Bind in the caller's context before create_task: every child inherits
        # the same model, including when EvoDuet is disabled.
        controller.llms.reserve_model(iteration)
        for candidate in batch.candidates:
            if confidence_template is not None:
                if candidate.index == 0:
                    candidate.confidence = confidence_template
                else:
                    candidate.confidence = copy.deepcopy(confidence_template)
                    candidate.confidence["record_id"] = uuid.uuid4().hex
                    controller.database.solution_confidence_records.append(candidate.confidence)
                candidate.confidence.update(
                    sample_index=candidate.index,
                    num_generations=self.count,
                    program_id=candidate.id,
                )

        async def generate(candidate: Candidate) -> None:
            try:
                async with self.generation_slots:
                    candidate.status = "generating"
                    candidate.response = await controller._call_llm(
                        batch.prompt["system"],
                        batch.prompt["user"],
                        llm_context={
                            "iteration": iteration,
                            "attempt": attempt,
                            "phase": "generation",
                            "source_program_id": parent_id,
                        },
                        reasoning_result_sink=candidate.calls["calls"],
                        web_search_result_sink=candidate.searches["tavily_searches"],
                    )
                    if not isinstance(candidate.response, LLMResponse):
                        candidate.response = None
                        raise TypeError("Each generation must return one LLMResponse")
                    controller._merge_llm_reasoning(
                        candidate.calls, candidate.response.llm_reasoning
                    )
                    controller._merge_web_search_results(
                        candidate.searches, candidate.response.web_search_results
                    )
                    if not candidate.response.text:
                        raise ValueError("Empty LLM response")
                    candidate.status = "generated"
            except Exception as exc:
                candidate.status = "generation_error"
                candidate.error = str(exc)
                solution_confidence.abort(candidate.confidence, candidate.status, candidate.error)

        async def evaluate(candidate: Candidate) -> None:
            try:
                async with self.evaluation_slots:
                    candidate.status = "evaluating"
                    await controller._assess_solution_confidence(
                        candidate.confidence, candidate.response, candidate.solution, candidate.id
                    )
                    candidate.evaluation = await controller.evaluator.evaluate_program(
                        candidate.solution, candidate.id
                    )
                    if not isinstance(candidate.evaluation, EvaluationResult):
                        candidate.evaluation = None
                        raise TypeError("Evaluator must return an EvaluationResult")
                    controller._merge_llm_reasoning(
                        candidate.calls, candidate.evaluation.llm_reasoning
                    )
                    controller._finish_solution_confidence(
                        candidate.confidence, candidate.evaluation
                    )
                    metrics = candidate.evaluation.metrics
                    if isinstance(metrics.get("image_path"), str):
                        candidate.image_path = metrics.pop("image_path")
                    candidate.error = controller._evaluation_failure_reason(
                        metrics, candidate.evaluation.artifacts
                    )
                    if candidate.error is None:
                        score = float(
                            controller.database.get_selection_score(candidate.ranking_program())
                        )
                        if not metrics or not math.isfinite(score):
                            candidate.error = "Evaluation returned no finite score"
                        else:
                            candidate.score = score
                    candidate.status = "evaluation_error" if candidate.error else "evaluated"
            except Exception as exc:
                candidate.status = "evaluation_error"
                candidate.error = str(exc)
                solution_confidence.abort(candidate.confidence, candidate.status, candidate.error)

        try:
            started = time.monotonic()
            await _gather_and_drain(generate(c) for c in batch.candidates)
            batch.generation_time = time.monotonic() - started
            for candidate in batch.candidates:
                if candidate.status != "generated":
                    continue
                try:
                    if parent is None:
                        candidate.solution = parse_full_rewrite(
                            candidate.response.text, controller.config.language
                        )
                        candidate.changes = "Generated from scratch"
                    else:
                        candidate.solution, candidate.changes, candidate.error = (
                            controller._parse_llm_response(
                                candidate.response.text,
                                parent_solution,
                                iteration,
                                attempt,
                                retry_times,
                            )
                        )
                    if not candidate.solution:
                        candidate.error = candidate.error or "No valid solution in response"
                    elif len(candidate.solution) > controller.config.max_solution_length:
                        candidate.error = f"Solution exceeds maximum length {controller.config.max_solution_length}"
                    if candidate.error:
                        raise ValueError(candidate.error)
                    candidate.status = "parsed"
                except Exception as exc:
                    candidate.status = "parse_error"
                    candidate.error = str(exc)
                    solution_confidence.abort(
                        candidate.confidence, candidate.status, candidate.error
                    )

            started = time.monotonic()
            await _gather_and_drain(evaluate(c) for c in batch.candidates if c.status == "parsed")
            batch.evaluation_time = time.monotonic() - started
            for candidate in batch.candidates:
                if candidate.status != "evaluated":
                    continue
                if batch.winner is None or controller.database._is_better(
                    candidate.ranking_program(), batch.winner.ranking_program()
                ):
                    batch.winner = candidate
            logger.info(
                "Iteration %s attempt %s: evaluated %s/%s candidates, selected sample %s (score=%s)",
                iteration,
                attempt,
                sum(c.evaluation is not None for c in batch.candidates),
                self.count,
                batch.winner.index if batch.winner is not None else None,
                batch.winner.score if batch.winner is not None else None,
            )
            return batch
        finally:
            # Also retain costs from cancelled/failed calls, after every sibling
            # has drained. Candidate IDs survive the controller's owner binding.
            for candidate in batch.candidates:
                if candidate.status in {
                    "pending",
                    "generating",
                    "generated",
                    "parsed",
                    "evaluating",
                }:
                    candidate.status = "cancelled"
                    candidate.error = "Candidate batch interrupted"
                    solution_confidence.abort(candidate.confidence, "cancelled", candidate.error)
                for records in (candidate.calls["calls"], candidate.searches["tavily_searches"]):
                    for record in records:
                        record.update(
                            candidate_program_id=candidate.id,
                            sample_index=candidate.index,
                            candidate_status=candidate.status,
                            candidate_selected=candidate is batch.winner,
                        )
                controller._merge_llm_reasoning(llm_reasoning, candidate.calls)
                controller._merge_web_search_results(web_search_results, candidate.searches)

    @staticmethod
    def result(
        controller,
        batches: list[GenerationBatch],
        *,
        parent: Program | None,
        iteration: int,
        iteration_start: float,
        llm_reasoning: dict,
        web_search_results: dict,
        context_program_ids: list | None = None,
        parent_info: tuple | None = None,
        context_info: list | None = None,
    ) -> SerializableResult:
        last = batches[-1]
        # An all-failed batch is archived by the existing failure path, never
        # inserted into the population. Its metadata retains every failed sample.
        candidate = last.winner or next(
            (c for c in reversed(last.candidates) if c.solution), last.candidates[0]
        )
        error = (
            None
            if last.winner is not None
            else f"All {len(last.candidates)} generation candidates failed after {last.attempt} attempts"
        )
        metadata: dict[str, Any] = {
            "changes": candidate.changes,
            "generation_batches": [batch.metadata() for batch in batches],
        }
        if parent is not None:
            metadata["parent_metrics"] = parent.metrics
        if candidate.image_path:
            metadata["image_path"] = candidate.image_path
        if controller.evoduet is not None:
            record_id = controller.evoduet.staged_search_record_id(iteration)
            if record_id:
                metadata["evoduet_search_record_id"] = record_id
        association = "generated_program" if error is None else "failed_generation_attempt"
        parent_id = parent.id if parent is not None else None
        metrics = (
            candidate.evaluation.metrics if candidate.evaluation is not None else {"error": 0.0}
        )
        if error is not None:
            # Failed scores such as NaN still need a readable JSON archive.
            metrics = finite_json(json_safe(metrics))
        child = Program(
            id=candidate.id,
            solution=candidate.solution or "",
            language=controller.config.language,
            parent_id=parent_id,
            other_context_ids=context_program_ids or [],
            parent_info=parent_info,
            context_info=context_info,
            metrics=metrics,
            iteration_found=iteration,
            metadata=metadata,
            artifacts=candidate.evaluation.artifacts if candidate.evaluation is not None else {},
            solution_confidence=copy.deepcopy(candidate.confidence or {}),
            llm_response=candidate.response.text if candidate.response is not None else None,
            llm_reasoning=controller._bind_llm_reasoning(
                llm_reasoning,
                program_id=candidate.id,
                source_program_id=parent_id,
                association=association,
                successful_attempt=last.attempt,
            ),
            llm_reasoning_content=reasoning_content(candidate.calls["calls"]),
            web_search_results=controller._bind_web_search_results(
                web_search_results,
                program_id=candidate.id,
                source_program_id=parent_id,
                association=association,
                successful_attempt=last.attempt,
            ),
        )
        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent_id,
            other_context_ids=context_program_ids or [],
            iteration=iteration,
            error=error,
            attempts_used=last.attempt,
            prompt=last.prompt,
            llm_response=child.llm_response,
            iteration_time=time.time() - iteration_start,
            llm_generation_time=sum(batch.generation_time for batch in batches),
            eval_time=sum(batch.evaluation_time for batch in batches),
            llm_reasoning=child.llm_reasoning,
            llm_reasoning_content=child.llm_reasoning_content,
            web_search_results=child.web_search_results,
        )
