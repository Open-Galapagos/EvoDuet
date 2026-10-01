"""Tests for D_search memory selection, retrieval credit, and persistence."""

import copy
import json

import pytest

from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval.base_retrieval import (
    SEARCH_DB_FILE,
    SEARCH_DB_SUBDIR,
    SearchRecord,
    SearchResult,
    SearchStore,
)


def _doc(i: int, score, **kw):
    return SearchResult(
        raw_content=f"raw{i}", content=f"c{i}", id=f"d{i}", relevance_score=score, **kw
    )


def _store(**kw) -> SearchStore:
    # record 0: results with relevance 0.9 and 0.3 ; record 1: a result with 0.7
    records = [
        SearchRecord(query=None, search_results=[_doc(1, 0.9), _doc(2, 0.3)]),
        SearchRecord(query=None, search_results=[_doc(3, 0.7)]),
    ]
    return SearchStore(records=records, **kw)


# ── selection policy (select_store applies it on the fly) ────────────────────
def test_select_store_full_returns_all():
    assert [d.id for d in _store(search_selection_policy="full").select_store()] == [
        "d1",
        "d2",
        "d3",
    ]


def test_select_store_top_k_by_relevance():
    docs = _store(search_selection_policy="top_k", search_selection_num=2).select_store()
    assert [d.id for d in docs] == ["d1", "d3"]  # 0.9, 0.7 — d2 (0.3) dropped


def test_select_store_top_k_none_returns_all_but_ranked():
    docs = _store(search_selection_policy="top_k", search_selection_num=None).select_store()
    assert {d.id for d in docs} == {"d1", "d2", "d3"}
    assert docs[0].id == "d1"  # highest relevance first


def test_select_store_top_k_treats_missing_score_as_lowest():
    s = SearchStore(
        records=[SearchRecord(search_results=[_doc(1, None), _doc(2, 0.5)])],
        search_selection_policy="top_k",
        search_selection_num=1,
    )
    assert [d.id for d in s.select_store()] == ["d2"]  # the scored one wins


def test_select_store_top_k_ranks_by_configured_criterion():
    # criterion=visit_count ranks by a different field than relevance_score
    records = [
        SearchRecord(
            search_results=[
                _doc(1, 0.9, visit_count=1),
                _doc(2, 0.2, visit_count=5),
            ]
        ),
    ]
    s = SearchStore(
        records=records,
        search_selection_policy="top_k",
        search_selection_num=1,
        search_selection_criterion="visit_count",
    )
    assert [d.id for d in s.select_store()] == ["d2"]  # highest visit_count wins


def test_select_store_recency_keeps_newest_first():
    # insertion order is d1, d2 (record 0) then d3 (record 1); d3 is newest.
    docs = _store(search_selection_policy="recency", search_selection_num=2).select_store()
    assert [d.id for d in docs] == ["d3", "d2"]  # last 2 added, newest-first (ignores relevance)


def test_select_store_recency_none_returns_all_newest_first():
    docs = _store(search_selection_policy="recency", search_selection_num=None).select_store()
    assert [d.id for d in docs] == ["d3", "d2", "d1"]  # all, reverse insertion order


def test_select_store_empty_returns_empty():
    assert SearchStore().select_store() == []


# ── verbalize (dumb renderer of the given results) ───────────────────────────


# ── verbalize via a configured template ──────────────────────────────────────


# ── usage attribution (visit_count + score/feedback history) ─────────────────
def test_record_usage_updates_only_the_fresh_retrieval_results_and_query():
    q = ConstructedQuery(query="q", id="q1")
    d1, d2 = _doc(1, 0.9), _doc(2, 0.3)
    s = SearchStore(records=[SearchRecord(search_results=[d2])])
    s.add(q, [d1], iteration=10)
    s.record_usage(10, score=0.98, feedback="good")
    assert d1.visit_count == 1 and d1.evolution_iterations == [10]
    assert d1.evolution_scores == [0.98] and d1.evolution_feedbacks == []
    assert q.visit_count == 1 and q.evolution_scores == [0.98]  # query updated too
    assert d2.visit_count == 0  # prior memory was not retrieved for this child


def test_record_usage_attributes_out_of_order_iterations_to_their_retrievals():
    s = SearchStore()
    first = s.add(None, [_doc(1, 0.9)], iteration=5)
    second = s.add(None, [_doc(2, 0.7)], iteration=6)
    s.record_usage(6, score=0.7, feedback="ok")
    s.record_usage(5, score=0.5)
    d1, d2 = first.search_results[0], second.search_results[0]
    assert d1.visit_count == d2.visit_count == 1
    assert d1.evolution_iterations == [5] and d2.evolution_iterations == [6]
    assert d1.evolution_scores == [0.5] and d2.evolution_scores == [0.7]
    assert d1.evolution_feedbacks == d2.evolution_feedbacks == []
    assert first.impacts[0].feedback is None and second.impacts[0].feedback == "ok"


def test_analysis_memory_selection_does_not_stage_retrieval_credit():
    from skydiscover.evoduet.search_context import render_search_experience

    s = _store(search_selection_policy="full")
    selected = s.select_store()
    assert selected[0].raw_content in render_search_experience(s)
    s.record_usage(7, score=0.9)
    assert s._pending == {}
    assert all(document.visit_count == 0 for document in selected)
    assert all(record.impacts == [] for record in s.records)


def test_record_usage_for_retrieve_add():
    q = ConstructedQuery(query="q", id="q1")
    s = SearchStore()
    s.add(q, [_doc(1, 0.9)], iteration=8)
    s.record_usage(8, score=0.4)
    assert s.records[0].search_results[0].visit_count == 1
    assert q.visit_count == 1 and q.evolution_iterations == [8]


def test_record_usage_noop_when_nothing_injected():
    records = [SearchRecord(query=None, search_results=[_doc(1, 0.9)])]
    s = SearchStore(records=records)
    s.record_usage(99, score=0.5)  # iter 99 never injected
    assert records[0].search_results[0].visit_count == 0


# ── persistence ─────────────────────────────────────────────────────────────
def test_add_writes_record(tmp_path):
    s = SearchStore(output_dir=str(tmp_path))
    s.add(None, [_doc(1, 0.9)])
    path = tmp_path / SEARCH_DB_SUBDIR / SEARCH_DB_FILE
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["search_results"][0]["id"] == "d1"


def test_checkpoint_round_trips_persisted_store(tmp_path):
    src = SearchStore(output_dir=str(tmp_path))
    q = ConstructedQuery(query="circle packing", keywords=["packing"])
    src.add(q, [_doc(1, 0.9, title="T1", url="http://x"), _doc(2, 0.3)])
    loaded = SearchStore(search_selection_policy="full")
    loaded.load_state_dict(json.loads(json.dumps(src.state_dict())))
    assert len(loaded.records) == 1
    docs = loaded.select_store()
    assert [d.id for d in docs] == ["d1", "d2"]
    assert docs[0].title == "T1" and docs[0].relevance_score == 0.9
    assert loaded.records[0].query.query == "circle packing"  # query reconstructed too


def test_jsonl_seed_input_is_removed(tmp_path):
    with pytest.raises(TypeError, match="rag_file_path"):
        SearchStore(rag_file_path=tmp_path / "seed.jsonl")


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_pending",
        "missing_counter",
        "invalid_record",
        "missing_record_id",
        "documents_alias",
        "query_text_alias",
        "provider_score_alias",
        "assessment_alias",
        "missing_source_queries",
        "missing_document_id",
        "blank_document_id",
        "summary",
        "invalid_pending",
        "missing_pending_record",
        "bad_result_index",
        "missing_target",
    ],
)
def test_old_or_incomplete_store_state_fails_without_changing_memory_or_disk(tmp_path, invalid):
    store = SearchStore(output_dir=tmp_path)
    store.add(ConstructedQuery(query="kept"), [_doc(1, 0.5)], iteration=3, target_score=0.2)
    before = store.state_dict()
    disk_before = store.path.read_bytes()
    state = copy.deepcopy(before)
    record = state["records"][0]
    document = record["search_results"][0]
    pending = state["pending"]["3"]
    if invalid == "missing_pending":
        state.pop("pending")
    elif invalid == "missing_counter":
        state.pop("next_document_id")
    elif invalid == "invalid_record":
        state["records"].append(None)
    elif invalid == "missing_record_id":
        record.pop("id")
    elif invalid == "documents_alias":
        record["documents"] = record.pop("search_results")
    elif invalid == "query_text_alias":
        record["query"]["text"] = record["query"].pop("query")
    elif invalid == "provider_score_alias":
        document["score"] = document.pop("relevance_score")
    elif invalid == "assessment_alias":
        record["document_assessments"] = record.pop("assessments")
    elif invalid == "missing_source_queries":
        record.pop("source_queries")
    elif invalid == "missing_document_id":
        document.pop("search_document_id")
    elif invalid == "blank_document_id":
        document["search_document_id"] = ""
    elif invalid == "summary":
        record["summary"] = "old summary"
    elif invalid == "invalid_pending":
        state["pending"]["bad iteration"] = pending
    elif invalid == "missing_pending_record":
        pending["records"][0]["record_id"] = "missing"
    elif invalid == "bad_result_index":
        pending["records"][0]["result_indices"] = [5]
    elif invalid == "missing_target":
        pending.pop("target_score")

    with pytest.raises(ValueError):
        store.load_state_dict(state)
    assert store.state_dict() == before
    assert store.path.read_bytes() == disk_before


def test_snapshot_writes_records_to_checkpoint_dir(tmp_path):
    s = SearchStore()
    s.add(ConstructedQuery(query="q"), [_doc(1, 0.9), _doc(2, 0.3)])
    out = s.snapshot(tmp_path / "checkpoint_5")
    assert out == tmp_path / "checkpoint_5" / SEARCH_DB_FILE
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1  # one record
    assert [d["id"] for d in json.loads(lines[0])["search_results"]] == ["d1", "d2"]


def test_usage_is_persisted(tmp_path):
    s = SearchStore(output_dir=str(tmp_path), search_selection_policy="full")
    s.add(None, [_doc(1, 0.9)], iteration=3)
    s.record_usage(3, score=0.7, feedback="fb")
    path = tmp_path / SEARCH_DB_SUBDIR / SEARCH_DB_FILE
    sr = json.loads(path.read_text().strip().splitlines()[-1])["search_results"][0]
    assert sr["visit_count"] == 1
    assert sr["evolution_scores"] == [0.7] and sr["evolution_feedbacks"] == []
