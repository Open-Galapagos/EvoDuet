"""Capture complete stored evidence for a lookup without changing prompt rendering."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from skydiscover.evoduet.retrieval.base_retrieval import SearchResult, SearchStore


def snapshot_lookup_selection(
    store: "SearchStore", documents: Iterable["SearchResult"]
) -> dict[str, Any]:
    """Snapshot selected documents and their complete, deduplicated source records.

    Call before converting stored bodies for prompt injection or adding the new
    lookup record. Document order follows the selection; source records follow
    database insertion order. All nested values are copied by ``asdict`` so later
    usage credit cannot change the recorded selection. The persistent provenance
    contains only IDs and iterations; full source records belong in the external
    iteration trace and therefore never recursively contain prior snapshots.
    """
    selected = list(documents)
    provenance = []
    source_record_ids = set()
    for document in selected:
        identifier = document.search_document_id

        def matches(candidate: "SearchResult") -> bool:
            if identifier:
                return candidate.search_document_id == identifier
            return candidate is document or bool(document.id and candidate.id == document.id)

        matching_records = [
            record
            for record in store.records
            if any(matches(candidate) for candidate in record.search_results)
        ]
        retrieval_records = [record for record in matching_records if record.operation != "look-up"]
        first = retrieval_records[0] if retrieval_records else None
        latest = retrieval_records[-1] if retrieval_records else None
        selected_source = next(
            (
                record
                for record in reversed(matching_records)
                if any(candidate is document for candidate in record.search_results)
            ),
            None,
        )
        record_ids = list(dict.fromkeys(record.id for record in matching_records))
        source_record_ids.update(record_ids)
        provenance.append(
            {
                "search_document_id": identifier or document.id,
                "first_retrieved_record_id": first.id if first is not None else None,
                "first_retrieved_iteration": first.iteration if first is not None else None,
                "latest_retrieved_record_id": latest.id if latest is not None else None,
                "latest_retrieved_iteration": latest.iteration if latest is not None else None,
                "selected_source_record_id": (
                    selected_source.id if selected_source is not None else None
                ),
                "selected_source_iteration": (
                    selected_source.iteration if selected_source is not None else None
                ),
                "source_record_ids": record_ids,
            }
        )

    source_records = []
    for record in store.records:
        if record.id in source_record_ids:
            source_records.append(asdict(record))
            source_record_ids.remove(record.id)
    return {
        "documents": [asdict(document) for document in selected],
        "document_provenance": provenance,
        "source_records": source_records,
    }
