"""Adapt queries to observed evidence while keeping the parent and prior scores fixed."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from skydiscover.evoduet.scoring import finite_score
from skydiscover.evoduet.search_context import _number
from skydiscover.evoduet.trace import make_step, record_query_evolution

logger = logging.getLogger("skydiscover.evoduet")
_PROMPTS = Path(__file__).parent / "prompts"


async def _gather_calls(calls):
    """Preserve sample order and drain sibling calls on cancellation."""
    tasks = [asyncio.create_task(call) for call in calls]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _render(name, fields):
    # One substitution pass: braces inside program/document data stay literal.
    template = (_PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    return re.sub(r"\{(\w+)\}", lambda m: str(fields.get(m[1], m[0])), template)


def _document_forecasts(rounds):
    """Collect only validated document forecasts from this inner loop."""
    forecasts = {}
    for row in rounds:
        if row.get("prediction_error"):
            continue
        for prediction in row.get("document_predictions", []):
            score = finite_score(prediction.get("estimated_child_score"))
            if score is not None:
                forecasts[prediction["evidence_ref"]] = score
    return forecasts


def _search_context(rounds, documents, max_chars, *, parent_score):
    """Render available documents; unscored documents have no score line."""
    forecasts = _document_forecasts(rounds)
    blocks = []
    for reference, (query, document) in documents.items():
        lines = [f"[[Observed Document: {reference}]]", f"doc_id: {reference}"]
        if reference in forecasts:
            lines.append(
                f"estimated_child_score: {_number(parent_score)} -> {_number(forecasts[reference])}"
            )
        lines.extend(
            (
                f"source_query: {json.dumps(query.text, ensure_ascii=False)}",
                f"title: {document.title}",
                f"url: {document.url}",
                "content:",
                str(document.raw_content or document.content or "")[:max_chars],
                "[[/Observed Document]]",
            )
        )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) or "(none)"


def _prediction(text, *, available_refs, top_k, previous_scores=None):
    """Keep valid new scores and rank them with the fixed existing forecasts.

    Invalid rows never overwrite an existing score or discard an unrelated
    valid prediction. Unscored documents cannot enter the selected pool.
    """

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate prediction field: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonstandard JSON constant: {value}")

    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    result = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("evidence prediction must be one JSON object")
    knowledge_state = result.get("knowledge_state_analysis")
    if not isinstance(knowledge_state, str) or not knowledge_state.strip():
        raise ValueError("prediction requires nonempty knowledge_state_analysis")
    previous_scores = previous_scores or {}
    forecasts = {
        ref: finite_score(previous_scores[ref]) for ref in available_refs if ref in previous_scores
    }
    if any(score is None for score in forecasts.values()):
        raise ValueError("existing document forecasts must be finite")
    new_refs = [ref for ref in available_refs if ref not in forecasts]
    rows = result.get("document_predictions")
    if not isinstance(rows, list):
        raise ValueError("document_predictions must be a list")
    available = set(new_refs)
    reference_counts = Counter(
        row["evidence_ref"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("evidence_ref"), str)
        and row["evidence_ref"] in available
    )
    new_scores, dropped = {}, []
    for index, row in enumerate(rows):
        reference = row.get("evidence_ref") if isinstance(row, dict) else None
        if not isinstance(row, dict):
            reason = "not_an_object"
        elif not isinstance(reference, str):
            reason = "invalid_reference"
        elif reference in previous_scores:
            reason = "already_scored"
        elif reference not in available:
            reason = "not_in_prediction_targets"
        elif reference_counts[reference] > 1:
            # An ambiguous ID cannot choose its score by row order or maximum.
            # Drop all its occurrences, while preserving other documents.
            reason = "duplicate_reference"
        else:
            value = finite_score(row.get("estimated_child_score"))
            reason = "invalid_score" if value is None else None
        if reason:
            dropped.append(
                {
                    "index": index,
                    "evidence_ref": reference if isinstance(reference, str) else None,
                    "reason": reason,
                }
            )
        else:
            new_scores[reference] = value
    forecasts.update(new_scores)
    unscored = [ref for ref in new_refs if ref not in new_scores]
    if not new_scores:
        status = "no_valid_predictions"
    elif dropped or unscored:
        status = "partial"
    else:
        status = "scored"
    # Stable sort keeps candidate input order for ties, independent of JSON row order.
    refs = sorted(
        (ref for ref in available_refs if ref in forecasts), key=lambda ref: -forecasts[ref]
    )[:top_k]
    return {
        "document_predictions": [
            {"evidence_ref": ref, "estimated_child_score": new_scores[ref]}
            for ref in new_refs
            if ref in new_scores
        ],
        "selected_evidence_refs": refs,
        "knowledge_state_analysis": knowledge_state,
        "assessment_status": status,
        "dropped_predictions": dropped,
        "unscored_document_refs": unscored,
    }


class ObservedSearch:
    """Query, retrieve, and score documents before the outer solution attempt."""

    def _generation_condition(self):
        """Execution metadata for logs, not an input to the prediction model."""
        count = self.config.num_generations
        return {
            "num_generations": count,
            "score_target": "single_child" if count == 1 else "best_valid_child_of_n_attempts",
        }

    def _stage_trace(self, trace, record):
        trace.update(search_record_id=record.id, status="awaiting_outcome")
        if record.search_results and trace["iteration"] is not None:
            self._pending_query_optimizations[trace["iteration"]] = trace
        else:
            # No document reached the solution prompt, so this record cannot
            # receive an evidence-attributed outcome or a pending forecast.
            trace["status"] = "no_evidence"
        self._persist_trace(trace)

    async def _search_round(self, prompt, *, iteration, number, count, row, trace):
        """Sample queries, then search each distinct valid query concurrently."""
        samples = [{"query_id": i, "search_attempted": False} for i in range(1, count + 1)]
        row["query_samples"] = samples

        async def construct(sample):
            label = "wk-query" if count == 1 else f"wk-query-sample-r{number}-{sample['query_id']}"
            try:
                text = await self.query_construction._generate(
                    "", prompt, label=label, iteration=iteration
                )
                sample["response"] = text
                query = self.query_construction._parse_query_sample(text)
                sample.update(query=query.text, constructed_query=asdict(query))
                return query
            except Exception as exc:
                sample["query_error"] = str(exc)
                logger.warning(
                    "EvoDuet query round %s sample %s failed: %s", number, sample["query_id"], exc
                )
                return None

        proposed = await _gather_calls(construct(sample) for sample in samples)
        seen, unique = {}, []
        for sample, query in zip(samples, proposed):
            if query is None:
                continue
            key = " ".join(query.text.split())
            if key in seen:
                sample["duplicate_of"] = seen[key]
                continue
            seen[key] = sample["query_id"]
            unique.append((sample, query))
        row["queries"] = [query.text for _, query in unique]
        if count == 1:
            # Preserve the single-query trace fields used by older readers.
            for key in ("query", "query_error"):
                if key in samples[0]:
                    row[key] = samples[0][key]

        async def search(sample, query):
            sample["search_attempted"] = row["search_attempted"] = True
            trace["search_attempts"] += 1
            limit = self.config.evoduet.tavily_retrieval.max_results
            try:
                found = await self.retrieval.retrieve(query, iteration=iteration, max_results=limit)
                documents = [replace(doc, query=doc.query or query.text) for doc in found[:limit]]
            except Exception as exc:
                documents = []
                sample["search_error"] = str(exc)
                logger.warning(
                    "EvoDuet search round %s sample %s failed: %s", number, sample["query_id"], exc
                )
            sample["documents"] = [asdict(doc) for doc in documents]
            return query, documents

        batches = await _gather_calls(search(sample, query) for sample, query in unique)
        if count == 1 and "search_error" in samples[0]:
            row["search_error"] = samples[0]["search_error"]
        return batches

    async def _retrieve_with_query(
        self, *, parent, history, context, iteration, target_program_id, target_score, task_context
    ):
        wk = self.config.evoduet
        parent_score = finite_score(target_score)
        if parent_score is None:
            raise ValueError("EvoDuet retrieval requires a finite actual parent evaluator score")
        # The inner loop must never move its parent or borrow outcomes from a
        # concurrently completed outer iteration.
        parent, context = copy.deepcopy(parent), copy.deepcopy(context)
        budget = wk.query_optimization_max_rounds
        query_count = wk.query_optimization_queries_per_round
        condition = self._generation_condition()
        knowledge_state = context.knowledge_state_analysis
        trace = {
            "iteration": iteration,
            "parent_id": target_program_id,
            "parent_score": parent_score,
            "mode": "observed_search",
            "operation": "retrieve",
            "max_rounds": budget,
            "queries_per_round": query_count,
            "search_budget": budget * query_count,
            "search_attempts": 0,
            "generation_condition": condition,
            "prediction_target": "individual_documents",
            "initial_knowledge_state_analysis": knowledge_state,
            "knowledge_state_analysis": knowledge_state,
            "rounds": [],
            "outcome_scope": "selected_evidence",
            "estimated_child_score": None,
            "status": "optimizing",
            "selection_method": "top_k_predicted_document_scores",
            "prompt_templates": {
                "initial_query": "query_generation.txt",
                "evolving_query": "query_refinement.txt",
                "prediction": "evidence_scoring.txt",
            },
        }
        self.query_optimization_history.append(trace)
        # Only selected documents live in pool. The identity-to-reference map
        # preserves ID meanings across pruning; full rejected bodies live in trace.
        pool, identities, searched = {}, {}, []
        steps = []
        try:
            # Each round has at most query_count constructions and searches.
            # Failed/duplicate samples are not replenished with extra calls.
            for number in range(1, budget + 1):
                fields = {
                    "current_program": parent.solution,
                    "parent_score": parent_score,
                    "evolutionary_history": history,
                    "search_database": context.search_database,
                    "population_state_analysis": context.population_analysis,
                    "knowledge_state_analysis": knowledge_state,
                    "search_context": _search_context(
                        trace["rounds"], pool, wk.max_document_chars, parent_score=parent_score
                    ),
                }
                row = {
                    "round": number,
                    "search_attempted": False,
                    "knowledge_state_analysis": knowledge_state,
                }
                trace["rounds"].append(row)
                name = "query_generation" if number == 1 else "query_refinement"
                batches = await self._search_round(
                    _render(name, fields),
                    iteration=iteration,
                    number=number,
                    count=query_count,
                    row=row,
                    trace=trace,
                )
                if not batches:
                    self._persist_trace(trace)
                    continue
                observed = [(query, doc) for query, docs in batches for doc in docs]
                documents = [doc for _, doc in observed]
                for query, docs in batches:
                    searched.append(query)
                    step = make_step(len(steps) + 1, "observed_search", query, docs)
                    step["round"] = number
                    steps.append(step)
                row["documents"] = [asdict(doc) for doc in documents]
                candidates = dict(pool)
                document_refs = []
                for query, doc in observed:
                    body_hash = hashlib.sha256(
                        str(doc.raw_content or doc.content or "").encode("utf-8")
                    ).hexdigest()
                    identity = (doc.url or doc.id, doc.title, body_hash)
                    if identity not in identities:
                        identities[identity] = f"evidence_{len(identities) + 1}"
                    reference = identities[identity]
                    document_refs.append(reference)
                    candidates.setdefault(reference, (query, doc))
                row.update(
                    document_evidence_refs=document_refs,
                    candidate_evidence_refs=list(candidates),
                    candidate_count=len(candidates),
                    pool_size=len(pool),
                )
                if not candidates:
                    row["search_status"] = "no_usable_documents"
                    self._persist_trace(trace)
                    continue
                previous_scores = _document_forecasts(trace["rounds"])
                new_refs = [ref for ref in candidates if ref not in previous_scores]
                row["new_document_refs"] = new_refs
                if not new_refs:
                    refs = sorted(candidates, key=lambda ref: -previous_scores[ref])[
                        : wk.search_result_top_k
                    ]
                    pool = {ref: candidates[ref] for ref in refs}
                    row.update(
                        document_predictions=[],
                        selected_evidence_refs=refs,
                        selection_status="retained_previous",
                        assessment_status="no_new_documents",
                        pool_size=len(pool),
                    )
                    self._persist_trace(trace)
                    continue
                fields.update(
                    search_context=_search_context(
                        trace["rounds"],
                        candidates,
                        wk.max_document_chars,
                        parent_score=parent_score,
                    ),
                )
                try:
                    text = await self.solution_model_pool._generate(
                        "",
                        _render("evidence_scoring", fields),
                        label="wk-query-prediction",
                        iteration=iteration,
                    )
                    prediction = _prediction(
                        text,
                        available_refs=list(candidates),
                        top_k=wk.search_result_top_k,
                        previous_scores=previous_scores,
                    )
                    # Commit only individually validated scores. Partial rounds
                    # have no prediction_error, so their accepted forecasts also
                    # survive subsequent rounds and checkpoint restoration.
                    pool = {ref: candidates[ref] for ref in prediction["selected_evidence_refs"]}
                    row.update(prediction)
                    if prediction["assessment_status"] == "scored":
                        knowledge_state = prediction["knowledge_state_analysis"]
                        row["knowledge_state_status"] = "updated"
                    else:
                        # The prose can contain the same ID confusion as the
                        # filtered scores. Preserve it for inspection, without
                        # feeding it into the next query as committed knowledge.
                        row["proposed_knowledge_state_analysis"] = prediction[
                            "knowledge_state_analysis"
                        ]
                        row["knowledge_state_analysis"] = knowledge_state
                        row["knowledge_state_status"] = "retained_previous"
                        if prediction["assessment_status"] == "no_valid_predictions":
                            row["selection_status"] = "retained_previous" if pool else "unavailable"
                        logger.warning(
                            "EvoDuet evidence prediction round %s: accepted %s new scores, "
                            "dropped %s rows, %s documents remain unscored; "
                            "previous knowledge state retained",
                            number,
                            len(prediction["document_predictions"]),
                            len(prediction["dropped_predictions"]),
                            len(prediction["unscored_document_refs"]),
                        )
                    trace["knowledge_state_analysis"] = knowledge_state
                except Exception as exc:
                    row["assessment_status"] = "failed"
                    row["knowledge_state_status"] = "retained_previous"
                    row["prediction_error"] = str(exc)
                    row["estimated_child_score"] = None
                    row["selected_evidence_refs"] = list(pool)
                    row["selection_status"] = "retained_previous" if pool else "unavailable"
                    logger.warning(
                        "EvoDuet evidence selection/prediction round %s failed: %s", number, exc
                    )
                row["pool_size"] = len(pool)
                self._persist_trace(trace)

            documents = [doc for _, doc in pool.values()]
            lead = next(iter(pool.values()))[0] if pool else (searched[0] if searched else None)
            record = self.search_store.add(
                lead,
                copy.deepcopy(documents),
                iteration=iteration,
                target_program_id=target_program_id,
                target_score=parent_score,
                source_queries=[query.text for query in searched],
            )
            forecasts = _document_forecasts(trace["rounds"])
            trace.update(
                estimated_child_score=None,
                estimated_improvement=None,
                knowledge_state_analysis=knowledge_state,
                selected_evidence_refs=list(pool),
                evidence=[
                    {
                        "pool_ref": reference,
                        "id": doc.id,
                        "search_document_id": doc.search_document_id,
                        "estimated_child_score": forecasts[reference],
                        "url": doc.url,
                        "title": doc.title,
                    }
                    for reference, doc in zip(pool, record.search_results)
                ],
                stop_reason=(
                    "search_budget_exhausted"
                    if trace["search_attempts"] == trace["search_budget"]
                    else "round_limit_reached"
                ),
            )
            self._stage_trace(trace, record)
            record_query_evolution(
                self.output_dir,
                iteration,
                decision="retrieve",
                mode="observed_search",
                steps=steps + [make_step(0, "selection", lead, record.search_results)],
            )
            # Keep provider snippets in the database. Only prompt copies use raw
            # content, matching both the forecast and the existing lookup path.
            return [
                replace(doc, content=doc.raw_content or doc.content)
                for doc in copy.deepcopy(record.search_results)
            ]
        except BaseException as exc:
            trace.update(status="failed", error=str(exc))
            self._persist_trace(trace)
            raise
