"""Search context preserves the relationship between evidence and outcomes."""

import copy
import json
from unittest.mock import Mock

import pytest

from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval.base_retrieval import (
    SearchImpact,
    SearchRecord,
    SearchResult,
    SearchResultAssessment,
    SearchStore,
)
from skydiscover.evoduet.search_context import render_search_experience


def document(identifier, *, raw="body", **kwargs):
    return SearchResult(id=identifier, raw_content=raw, content="snippet", **kwargs)


def rendered_ids(text):
    return [
        json.loads(line.split(":", 1)[1])
        for line in text.splitlines()
        if line.startswith("search_document_id:")
    ]


def query_block(reference, query):
    return f"- query_ref: {reference}\n" f"  query: {json.dumps(query, ensure_ascii=False)}"


def test_recency_retains_ten_records_and_three_documents_without_mutation(monkeypatch):
    records = [
        SearchRecord(
            iteration=index,
            search_results=[document(f"{index}-{offset}") for offset in range(4)],
        )
        for index in range(12)
    ]
    store = SearchStore(
        records=records,
        search_selection_policy="recency",
        search_selection_num=10,
        documents_per_entry=3,
    )
    before = copy.deepcopy(store.state_dict())
    monkeypatch.setattr(store, "lookup_documents", Mock(side_effect=AssertionError("no lookup")))
    monkeypatch.setattr(store, "_persist", Mock(side_effect=AssertionError("no persistence")))

    text = render_search_experience(store)

    assert rendered_ids(text) == [
        doc.search_document_id
        for record in reversed(records[2:])
        for doc in record.search_results[:3]
    ]
    assert text.count("[Search Experience ") == 10
    assert text.count("[[Search Document: ") == 30
    assert "[Search Experience 1] iteration 11" in text
    assert "[Search Experience 10] iteration 2" in text
    assert "documents shown: 3/4" in text
    for record in records[2:]:
        assert (
            "used_document_ids: "
            + json.dumps([doc.search_document_id for doc in record.search_results])
            in text
        )
    assert store.state_dict() == before


@pytest.mark.parametrize("impacts", [[], [SearchImpact(iteration=3, feedback="unevaluated")]])
def test_unevaluated_outcomes_remain_unknown_not_zero(impacts):
    store = SearchStore(
        records=[SearchRecord(search_results=[document("doc")], impacts=impacts)],
        search_selection_policy="full",
    )

    text = render_search_experience(store)

    assert "parent_score -> child_score: unknown -> unknown" in text
    assert "delta:" not in text
    if impacts:
        assert 'feedback: "unevaluated"' in text
    else:
        assert "feedback:" not in text


def test_nonfinite_scores_are_unknown_but_available_partial_outcomes_are_preserved():
    record = SearchRecord(
        retrieved_for_score=float("nan"),
        estimated_score=float("inf"),
        impacts=[SearchImpact(iteration=4, result_score=0.6, delta=float("nan"))],
    )
    text = render_search_experience(SearchStore(records=[record], search_selection_policy="full"))

    assert "parent_score -> child_score: unknown -> 0.6" in text
    assert "delta:" not in text
    assert "feedback:" not in text
    assert "used_document_ids: []" in text
    assert "documents: (none)" in text


def test_reused_ids_keep_their_historical_body_without_lookup_provenance_dump():
    first = document(
        "provider", raw="ORIGINAL BODY", url="https://evidence.test", query="original search"
    )
    latest = document(
        "provider", raw="LATER BODY", url="https://evidence.test", query="later search"
    )
    original = SearchRecord(id="sr_original", iteration=2, search_results=[first])
    lookup = SearchRecord(
        id="sr_lookup",
        iteration=7,
        operation="look-up",
        search_results=[latest],
        source_queries=["reuse instruction is not a fresh search"],
    )
    store = SearchStore(records=[original, lookup], search_selection_policy="recency")
    lookup.lookup_provenance = [
        {
            "search_document_id": latest.search_document_id,
            "first_retrieved_record_id": original.id,
            "first_retrieved_iteration": 2,
            "selected_source_record_id": original.id,
            "source_record_ids": [original.id],
            "unexpected_recursive_snapshot": {"raw_content": "DO NOT DUMP"},
        }
    ]

    before = copy.deepcopy(store.state_dict())
    text = render_search_experience(store)

    assert rendered_ids(text) == [latest.search_document_id, first.search_document_id]
    assert latest.search_document_id == first.search_document_id
    lookup_text, original_text = text.split("[Search Experience 2]")
    assert "LATER BODY" in lookup_text and "ORIGINAL BODY" not in lookup_text
    assert "ORIGINAL BODY" in original_text and "LATER BODY" not in original_text
    assert "operation: look-up" in lookup_text
    assert "searched_queries: []" in lookup_text
    assert 'source_query: "later search"' in lookup_text
    assert "source_query_ref:" not in lookup_text
    assert "estimated_child_score:" not in lookup_text
    assert "reuse instruction is not a fresh search" not in text
    assert "sr_original" not in text
    assert "first_retrieved_iteration:" not in text
    assert "DO NOT DUMP" not in text
    assert "latest stored version" in text
    assert store.state_dict() == before


def test_top_k_selects_individual_occurrences_with_owning_record_outcomes(monkeypatch):
    old = document("reused", raw="OLD", relevance_score=0.5)
    new = document("reused", raw="NEW", relevance_score=0.9)
    extra = document("extra", raw="EXTRA", relevance_score=0.8)
    original = SearchRecord(id="sr_original", search_results=[old])
    latest = SearchRecord(
        id="sr_latest",
        search_results=[new, extra],
        impacts=[SearchImpact(iteration=10, target_score=0.2, result_score=0.4, delta=0.2)],
    )
    store = SearchStore(
        records=[original, latest],
        search_selection_policy="top_k",
        search_selection_num=2,
        documents_per_entry=1,
    )
    monkeypatch.setattr(
        store, "select_records", Mock(side_effect=AssertionError("not record selection"))
    )

    text = render_search_experience(store)

    assert rendered_ids(text) == [new.search_document_id, extra.search_document_id]
    assert text.count("[Search Experience ") == 1
    assert (
        "content:\nNEW\n[[/Search Document]]" in text
        and "content:\nEXTRA\n[[/Search Document]]" in text
        and "content:\nOLD\n[[/Search Document]]" not in text
    )
    assert "parent_score -> child_score: 0.2 -> 0.4" in text


def test_document_assessments_are_hidden_but_preserved_in_storage():
    first, second = document("first"), document("second")
    record = SearchRecord(
        search_results=[first, second],
        assessments=[
            SearchResultAssessment(
                document_ref="second", helpfulness=0.2, reason="second rationale"
            ),
            SearchResultAssessment(document_ref="first", helpfulness=0.8, reason="first rationale"),
        ],
    )
    store = SearchStore(records=[record], search_selection_policy="full")
    before = copy.deepcopy(store.state_dict())

    text = render_search_experience(store)

    for hidden in ("helpfulness", "first rationale", "second rationale", "prediction reasoning"):
        assert hidden not in text
    assert store.state_dict() == before


def test_raw_and_fallback_bodies_preserve_formatting_and_per_document_limits():
    raw = "### Methods\n\n```python\nx = 1\n```\nRAW TAIL"
    fallback = "fallback\n\n$$x^2$$\nCONTENT TAIL"
    first = document("raw", raw=raw)
    second = document("fallback", raw="")
    second.content = fallback
    store = SearchStore(
        records=[SearchRecord(search_results=[first, second])],
        search_selection_policy="full",
        max_document_chars=26,
    )

    text = render_search_experience(store)

    assert f"content:\n{raw[:26]}\n[[/Search Document]]\n\n[[Search Document:" in text
    assert text.endswith(f"content:\n{fallback[:26]}\n[[/Search Document]]")
    assert "snippet" not in text and "~~~" not in text


def test_empty_store_has_no_invented_experience():
    assert render_search_experience(SearchStore()) == "(empty — no prior search experience)"


def test_pending_outcome_preserves_known_parent_and_omits_absent_feedback():
    record = SearchRecord(retrieved_for_score=0.7, search_results=[document("pending")])

    text = render_search_experience(SearchStore(records=[record], search_selection_policy="full"))

    assert "parent_score -> child_score: 0.7 -> unknown" in text
    assert "feedback:" not in text


def test_three_winning_documents_can_share_q1_without_dropping_other_executed_queries():
    queries = ["method A", "method B", "method C"]
    record = SearchRecord(
        source_queries=queries,
        search_results=[document(f"winner-{index}", query=queries[0]) for index in range(3)],
    )

    text = render_search_experience(SearchStore(records=[record], search_selection_policy="full"))

    for index, query in enumerate(queries, 1):
        assert query_block(f"q{index}", query) in text
    assert text.count("source_query_ref: q1") == 3
    assert "source_query_ref: q2" not in text
    assert "source_query_ref: q3" not in text


@pytest.mark.parametrize(
    "queries, document_query, expected_ref",
    [
        (["only query"], None, "q1"),
        (["method A", "method B"], None, "unknown"),
        (["method A", "method B"], "unmatched query", "unknown"),
    ],
)
def test_query_source_fallback_does_not_invent_a_mapping(queries, document_query, expected_ref):
    record = SearchRecord(
        source_queries=queries,
        search_results=[document("doc", query=document_query)],
    )

    text = render_search_experience(SearchStore(records=[record], search_selection_policy="full"))

    block = text.split("[[Search Document:", 1)[1]
    assert f"source_query_ref: {expected_ref}" in block
    if document_query == "unmatched query":
        assert 'source_query: "unmatched query"' in block
        assert query_block("q3", "unmatched query") not in text


def test_query_list_falls_back_to_record_query_before_document_queries():
    record = SearchRecord(
        query=ConstructedQuery(query="record query"),
        search_results=[document("doc", query="document query")],
    )

    text = render_search_experience(SearchStore(records=[record], search_selection_policy="full"))

    assert query_block("q1", "record query") in text
    assert query_block("q2", "document query") not in text
    block = text.split("[[Search Document:", 1)[1]
    assert "source_query_ref: unknown" in block
    assert 'source_query: "document query"' in block


def test_document_queries_fallback_deduplicates_text_when_record_queries_are_unavailable():
    record = SearchRecord(
        search_results=[
            document("first", query="method B"),
            document("second", query="method A"),
            document("third", query="method B"),
        ],
    )

    text = render_search_experience(SearchStore(records=[record], search_selection_policy="full"))

    assert query_block("q1", "method B") in text
    assert query_block("q2", "method A") in text
    assert text.count("- query_ref:") == 2
    blocks = text.split("[[Search Document:")[1:]
    for block, ref in zip(blocks, ["q1", "q2", "q1"]):
        assert f"source_query_ref: {ref}" in block
