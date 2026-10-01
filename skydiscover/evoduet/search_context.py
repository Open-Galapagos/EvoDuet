"""Render search evidence together with its recorded experimental outcomes."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from skydiscover.evoduet.retrieval.base_retrieval import (
        SearchRecord,
        SearchResult,
        SearchStore,
    )


def _identifier(document: "SearchResult") -> str:
    return document.search_document_id or document.id


def _number(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if math.isfinite(value):
                return str(value)
        except OverflowError:
            pass
    return "unknown"


def _text(value: Any) -> str:
    """Quote metadata without allowing line breaks to become new field labels."""
    return json.dumps(value, ensure_ascii=False)


def _selected_experiences(store: "SearchStore") -> list[tuple["SearchRecord", list[int]]]:
    if store.search_selection_policy != "top_k":
        return [
            (record, list(range(min(len(record.search_results), store.documents_per_entry))))
            for record in store.select_records()
        ]

    # Legacy top_k limits individual document occurrences, not whole records.
    # Match exact stored objects so a later reuse of the same document ID cannot
    # accidentally receive the original retrieval's outcome (or vice versa).
    owners: dict[int, deque] = defaultdict(deque)
    for record in store.records:
        for index, document in enumerate(record.search_results):
            owners[id(document)].append((record, index))
    grouped: dict[int, tuple["SearchRecord", list[int]]] = {}
    for document in store.select_store():
        if not owners[id(document)]:
            continue
        record, index = owners[id(document)].popleft()
        grouped.setdefault(id(record), (record, []))[1].append(index)
    return list(grouped.values())


def _searched_queries(record: "SearchRecord") -> list[str]:
    """Retain executed searches, including those without a selected document."""
    if record.operation == "look-up":
        return []
    sources = record.source_queries or (
        [record.query.text] if record.query and record.query.text else []
    )
    if not sources:
        sources = [document.query for document in record.search_results]
    return list(
        dict.fromkeys(
            query.strip() for query in sources if isinstance(query, str) and query.strip()
        )
    )


def _render_document(
    store: "SearchStore", record: "SearchRecord", index: int, query_refs: dict[str, str]
) -> str:
    document = record.search_results[index]
    identifier = _identifier(document)
    lines = [
        f"[[Search Document: {identifier}]]",
        f"search_document_id: {_text(identifier)}",
    ]
    source = document.query.strip() if isinstance(document.query, str) else ""
    if record.operation == "look-up":
        # This is historical provenance, not a new query or forecast for reuse.
        lines.append(f"source_query: {_text(source) if source else 'unknown'}")
    else:
        reference = query_refs.get(source)
        if not source and len(query_refs) == 1:
            reference = next(iter(query_refs.values()))
        lines.append(f"source_query_ref: {reference or 'unknown'}")
        if source and reference is None:
            # Preserve inconsistent/legacy provenance without assigning by rank
            # or pretending that an unrelated forecast belongs to this query.
            lines.append(f"source_query: {_text(source)}")
    lines.extend((f"title: {document.title or ''}", f"url: {document.url or ''}"))
    # Keep the historical body associated with this experience. Substituting a
    # newer version by ID would falsely attach old outcomes to different evidence.
    body = str(document.raw_content or document.content or "")[: store.max_document_chars]
    lines.extend(("content:", body, "[[/Search Document]]"))
    return "\n".join(lines)


def render_search_experience(store, query_optimization_history=()):
    experiences = _selected_experiences(store)
    if not experiences:
        return "(empty — no prior search experience)"
    traces = {
        trace.get("search_record_id"): trace
        for trace in query_optimization_history
        if isinstance(trace, dict) and trace.get("search_record_id")
    }
    blocks = []
    for number, (record, indices) in enumerate(experiences, 1):
        trace = traces.get(record.id, {})
        when = "pre-loop" if record.iteration is None else f"iteration {record.iteration}"
        lines = [f"[Search Experience {number}] {when}", f"operation: {record.operation}"]
        refs = {query: f"q{i}" for i, query in enumerate(_searched_queries(record), 1)}
        lines.append("\nsearched_queries:" if refs else "\nsearched_queries: []")
        for query, ref in refs.items():
            lines.extend((f"- query_ref: {ref}", f"  query: {_text(query)}"))
        if trace.get("prediction_target") != "individual_documents":
            lines.append(
                "\nestimated_child_score (selected evidence together): "
                + _number(trace.get("estimated_child_score"))
            )
        if not record.impacts:
            lines.extend(
                (
                    "\nactual outcome: unknown (no evaluated outcome recorded)",
                    f"parent_score -> child_score: {_number(record.retrieved_for_score)} -> unknown",
                )
            )
        for impact in record.impacts:
            when = f"iteration {_number(impact.iteration)}; " if len(record.impacts) > 1 else ""
            lines.extend(
                (
                    f"\nactual outcome ({when}combined evidence):",
                    f"parent_score -> child_score: {_number(impact.target_score)} -> {_number(impact.result_score)}",
                )
            )
            if impact.feedback and impact.feedback.strip():
                lines.append(f"feedback: {_text(impact.feedback[:store.max_document_chars])}")
        lines.append(
            f"used_document_ids: {_text([_identifier(doc) for doc in record.search_results])}"
        )
        if len(indices) < len(record.search_results):
            lines.append(
                f"documents shown: {len(indices)}/{len(record.search_results)}; "
                "forecasts and outcomes include the omitted documents too"
            )
        if not indices:
            lines.append("documents: (none)")
        forecasts = {
            item.get("search_document_id"): item.get("estimated_child_score")
            for item in trace.get("evidence", [])
        }
        for index in indices:
            document = _render_document(store, record, index, refs)
            score = _number(forecasts.get(_identifier(record.search_results[index])))
            if score != "unknown":
                heading, _, body = document.partition("\n")
                document = (
                    f"{heading}\nestimated_child_score: "
                    f"{_number(record.retrieved_for_score)} -> {score}\n{body}"
                )
            lines.append("\n" + document)
        blocks.append("\n".join(lines))
    notes = (
        "Document-level estimated_child_score predicts a child using that document alone. "
        "Legacy evidence-set forecasts use the listed documents together. Neither is a measured outcome. "
        "Actual outcomes belong to the whole attempt, not a comparison against no retrieval. "
        "Missing scores are unknown. Query refs are local to an experience, not document IDs. "
        "A document's source_query_ref identifies its retained retrieval provenance; duplicates "
        "can also occur in other searches. For look-up, source_query is historical provenance. "
        "Bodies are historical; look-up fetches the latest stored version of each ID."
    )
    return notes + "\n\n" + "\n\n".join(blocks)
