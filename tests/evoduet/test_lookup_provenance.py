"""Lookup audit snapshots preserve source history without recursive DB records."""

import copy
import json
from dataclasses import asdict

import pytest

from skydiscover.evoduet.lookup_provenance import snapshot_lookup_selection
from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval.base_retrieval import (
    SearchImpact,
    SearchRecord,
    SearchResult,
    SearchResultAssessment,
    SearchStore,
)


def _document(name="page", *, iteration=4, **kwargs):
    return SearchResult(
        id=f"provider-{name}",
        url=f"https://example.test/{name}",
        raw_content=f"RAW {name}\nfull body",
        content=f"SNIPPET {name}",
        title=f"Title {name}",
        iteration=iteration,
        **kwargs,
    )


def test_snapshot_preserves_complete_records_cosearched_documents_and_selected_order():
    first = _document(
        source="tavily",
        published_date="2024-01-02",
        query="original query",
        keywords=["keyword"],
        resources=["paper"],
        query_type="method",
        rank=2,
        relevance_score=0.8,
        favicon="https://example.test/favicon.ico",
        images=["https://example.test/diagram.png"],
        metadata={"nested": {"original": [1, 2]}},
        retrieval_iteration=4,
        visit_count=3,
        evolution_iterations=[4, 5, 6],
        evolution_scores=[0.4, 0.5, 0.6],
        evolution_feedbacks=["original feedback"],
    )
    second = _document("second")
    cosearched = _document("unselected")
    record = SearchRecord(
        query=ConstructedQuery(query="original query", keywords=["keyword"], visit_count=3),
        search_results=[first, second, cosearched],
        iteration=4,
        retrieved_for_program_id="original-parent",
        retrieved_for_score=0.3,
        assessments=[
            SearchResultAssessment(
                document_ref=first.id,
                relevance=4,
                novelty=3,
                helpfulness=5,
                estimated_score=0.7,
                estimated_improvement_score=0.4,
                reason="original assessment",
            )
        ],
        relevance=4,
        novelty=3,
        helpfulness=5,
        estimated_score=0.7,
        estimated_improvement_score=0.4,
        impacts=[
            SearchImpact(
                iteration=4,
                target_program_id="original-parent",
                target_score=0.3,
                result_program_id="original-child",
                result_score=0.5,
                delta=0.2,
                feedback="joint feedback",
            )
        ],
        source_queries=["original query", "pooled query"],
    )
    unrelated = SearchRecord(search_results=[_document("unrelated")])
    store = SearchStore(records=[record, unrelated])
    before = copy.deepcopy(store.state_dict())

    snapshot = snapshot_lookup_selection(store, [second, first])

    assert snapshot["documents"] == [asdict(second), asdict(first)]
    assert snapshot["source_records"] == [asdict(record)]
    assert len(snapshot["source_records"][0]["search_results"]) == 3
    assert [entry["search_document_id"] for entry in snapshot["document_provenance"]] == [
        second.search_document_id,
        first.search_document_id,
    ]
    assert store.state_dict() == before

    frozen = copy.deepcopy(snapshot)
    first.metadata["nested"]["original"].append(3)
    first.content = "changed content"
    record.query.keywords.append("changed")
    record.impacts[0].feedback = "later feedback"
    record.assessments[0].reason = "changed assessment"
    assert snapshot == frozen


def test_first_search_refetch_and_selected_lookup_are_distinct_in_insertion_order():
    first = SearchRecord(search_results=[_document(iteration=12)], iteration=12)
    refetched = SearchRecord(search_results=[_document(iteration=3)], iteration=3)
    store = SearchStore(records=[first, refetched])
    identifier = first.search_results[0].search_document_id
    initial_snapshot = snapshot_lookup_selection(store, store.lookup_documents([identifier]))
    reused = copy.deepcopy(refetched.search_results[0])
    reused.content = reused.raw_content
    lookup = store.add(
        None,
        [reused],
        iteration=15,
        operation="look-up",
        lookup_provenance=initial_snapshot["document_provenance"],
    )

    snapshot = snapshot_lookup_selection(store, store.lookup_documents([identifier]))

    assert snapshot["document_provenance"] == [
        {
            "search_document_id": identifier,
            "first_retrieved_record_id": first.id,
            "first_retrieved_iteration": 12,
            "latest_retrieved_record_id": refetched.id,
            "latest_retrieved_iteration": 3,
            "selected_source_record_id": lookup.id,
            "selected_source_iteration": 15,
            "source_record_ids": [first.id, refetched.id, lookup.id],
        }
    ]
    assert snapshot["documents"] == [asdict(reused)]
    assert snapshot["source_records"] == [asdict(record) for record in store.records]
    assert snapshot["source_records"][1]["search_results"][0]["content"] == "SNIPPET page"


def test_grounding_none_is_preserved_with_known_first_record():
    grounding = SearchRecord(search_results=[_document(iteration=0)], iteration=None)
    later = SearchRecord(search_results=[_document(iteration=9)], iteration=9)
    store = SearchStore(records=[grounding, later])

    snapshot = snapshot_lookup_selection(store, [later.search_results[0]])

    provenance = snapshot["document_provenance"][0]
    assert provenance["first_retrieved_record_id"] == grounding.id
    assert provenance["first_retrieved_iteration"] is None
    assert provenance["latest_retrieved_record_id"] == later.id
    assert provenance["latest_retrieved_iteration"] == 9


def test_repeated_lookup_records_link_history_without_nested_record_snapshots():
    original = SearchRecord(search_results=[_document()], iteration=4)
    store = SearchStore(records=[original])
    identifier = original.search_results[0].search_document_id
    for iteration in range(5, 15):
        selected = store.lookup_documents([identifier])
        snapshot = snapshot_lookup_selection(store, selected)
        assert len(snapshot["source_records"]) == len(store.records)
        added = store.add(
            None,
            copy.deepcopy(selected),
            iteration=iteration,
            operation="look-up",
            lookup_provenance=snapshot["document_provenance"],
        )
        # Only source identity and iteration metadata enter the database. Keeping
        # complete snapshots exclusively in the trace avoids recursive histories.
        assert set(added.lookup_provenance[0]) == {
            "search_document_id",
            "first_retrieved_record_id",
            "first_retrieved_iteration",
            "latest_retrieved_record_id",
            "latest_retrieved_iteration",
            "selected_source_record_id",
            "selected_source_iteration",
            "source_record_ids",
        }
        assert added.lookup_provenance[0]["first_retrieved_record_id"] == original.id
        assert added.lookup_provenance[0]["source_record_ids"] == [
            record.id for record in store.records[:-1]
        ]
        assert json.dumps(asdict(added)).count('"raw_content"') == 1
        snapshot["document_provenance"][0]["source_record_ids"].append("external mutation")
        assert "external mutation" not in added.lookup_provenance[0]["source_record_ids"]


def test_provenance_roundtrips_checkpoint_and_is_written_to_jsonl(tmp_path):
    store = SearchStore(output_dir=tmp_path)
    original = store.add(None, [_document()], iteration=4)
    selected = store.lookup_documents([original.search_results[0].search_document_id])
    snapshot = snapshot_lookup_selection(store, selected)
    lookup = store.add(
        None,
        copy.deepcopy(selected),
        iteration=7,
        target_program_id="lookup-parent",
        target_score=0.4,
        operation="look-up",
        lookup_provenance=snapshot["document_provenance"],
    )
    expected = json.loads(json.dumps(store.state_dict()))
    restored = SearchStore()
    restored.load_state_dict(expected)
    assert restored.state_dict() == expected
    rows = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert rows[-1]["lookup_provenance"] == lookup.lookup_provenance

    before = copy.deepcopy(restored.records[0])
    restored.record_usage(7, score=0.6, result_program_id="lookup-child")
    assert asdict(restored.records[0]) == asdict(before)
    assert restored.records[-1].impacts[0].target_program_id == "lookup-parent"
    assert restored.records[-1].lookup_provenance == lookup.lookup_provenance

    before = restored.state_dict()
    legacy = copy.deepcopy(before)
    legacy["records"][0].pop("lookup_provenance")
    with pytest.raises(ValueError, match="SearchRecord checkpoint fields"):
        restored.load_state_dict(legacy)
    assert restored.state_dict() == before


def test_lookup_only_history_does_not_invent_an_initial_search():
    record = SearchRecord(search_results=[_document()], operation="look-up", iteration=6)
    store = SearchStore(records=[record])

    provenance = snapshot_lookup_selection(store, record.search_results)["document_provenance"][0]

    assert provenance["first_retrieved_record_id"] is None
    assert provenance["first_retrieved_iteration"] is None
    assert provenance["latest_retrieved_record_id"] is None
    assert provenance["latest_retrieved_iteration"] is None
    assert provenance["selected_source_record_id"] == record.id
    assert provenance["selected_source_iteration"] == 6


def test_empty_selection_has_no_documents_or_source_records():
    store = SearchStore(records=[SearchRecord(search_results=[_document()])])
    assert snapshot_lookup_selection(store, []) == {
        "documents": [],
        "document_provenance": [],
        "source_records": [],
    }
