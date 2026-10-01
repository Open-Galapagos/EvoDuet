"""Focused tests for record-level search assessment, selection, and credit."""

import json
import math

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


def _doc(name: str) -> SearchResult:
    return SearchResult(
        raw_content=f"raw {name}",
        content=f"summary {name}",
        id=name,
        title=f"Document {name}",
        url=f"https://example.org/{name}",
    )


def _rated(query: str, delta: float) -> SearchRecord:
    return SearchRecord(
        query=ConstructedQuery(query=query),
        search_results=[_doc(query)],
        impacts=[SearchImpact(delta=delta)],
    )


def test_add_and_usage_credit_the_exact_staged_target():
    store = SearchStore()
    query = ConstructedQuery(query="credited direction")
    document = _doc("d1")
    assessment = SearchResultAssessment(
        document_ref=document.id,
        relevance=0.9,
        novelty=0.6,
        estimated_improvement_score=0.3,
        estimated_score=0.6,
        reason="useful and distinct",
    )

    record = store.add(
        query,
        [document],
        iteration=4,
        target_program_id="parent",
        target_score=0.5,
        assessments=[assessment],
    )
    assert record.id.startswith("sr_")
    assert record.retrieved_for_program_id == "parent"
    assert record.relevance == pytest.approx(0.9)
    assert record.estimated_score == pytest.approx(0.6)

    # Credit uses the staged value, not mutable record/population state observed later.
    record.retrieved_for_score = -100.0
    store.record_usage(4, score=0.8, feedback="better", result_program_id="child")

    impact = record.impacts[0]
    assert impact.target_program_id == "parent" and impact.result_program_id == "child"
    assert impact.target_score == pytest.approx(0.5)
    assert impact.result_score == pytest.approx(0.8)
    assert impact.delta == pytest.approx(0.3)
    assert impact.delta > 0 and impact.feedback == "better"
    assert document.visit_count == query.visit_count == 1
    assert document.evolution_scores == query.evolution_scores == [0.8]


def test_empty_retrieval_is_recorded_but_never_credited():
    store = SearchStore()
    record = store.add(
        ConstructedQuery(query="empty"),
        [],
        iteration=2,
        target_program_id="parent",
        target_score=0.4,
    )

    assert record.iteration == 2 and 2 not in store._pending
    store.record_usage(2, score=0.9, result_program_id="child")
    assert record.impacts == []


def test_non_finite_outcomes_never_enter_usage_history():
    store = SearchStore()
    query = ConstructedQuery(query="unstable")
    document = _doc("unstable")
    record = store.add(query, [document], iteration=3, target_score=0.4)

    store.record_usage(3, score=math.nan)

    assert record.impacts[0].result_score is None
    assert query.evolution_scores == document.evolution_scores == []


def test_discard_usage_forgets_evidence_that_never_reached_the_prompt():
    store = SearchStore()
    query = ConstructedQuery(query="not injected")
    document = _doc("not-injected")
    record = store.add(query, [document], iteration=2, target_score=0.4)

    store.discard_usage(2)
    store.record_usage(2, score=0.9, result_program_id="child")

    assert record.impacts == []
    assert query.visit_count == document.visit_count == 0


def test_negative_outcome_credits_the_fresh_retrieval_without_changing_prior_records():
    first, second = _doc("first"), _doc("second")
    prior = SearchRecord(query=ConstructedQuery(query="q1"), search_results=[first])
    store = SearchStore(records=[prior])

    record = store.add(
        ConstructedQuery(query="q2"),
        [second],
        iteration=9,
        target_program_id="p",
        target_score=0.7,
    )
    store.record_usage(9, score=0.6, result_program_id="c")

    assert prior.impacts == [] and first.visit_count == 0
    assert record.impacts[0].delta == pytest.approx(-0.1)
    assert record.impacts[0].delta < 0
    assert second.visit_count == record.query.visit_count == 1


@pytest.mark.parametrize("resume", [False, True])
def test_joint_retrieval_outcome_never_credits_the_representative_query(resume):
    store = SearchStore()
    record = store.add(
        ConstructedQuery(query="algorithm reference"),
        [_doc("algorithm"), _doc("implementation")],
        source_queries=["algorithm reference", "implementation reference"],
        iteration=4,
        target_program_id="parent",
        target_score=0.7,
    )
    if resume:
        restored = SearchStore()
        restored.load_state_dict(json.loads(json.dumps(store.state_dict())))
        store, record = restored, restored.records[0]

    assert store._pending[4]["queries"] == []
    store.record_usage(4, score=0.6, result_program_id="child")

    assert record.delta == pytest.approx(-0.1)
    assert record.query.visit_count == 0
    assert record.query.evolution_scores == []
    assert all(document.visit_count == 1 for document in record.search_results)
    rendered = render_search_experience(store)
    assert "actual outcome (combined evidence):" in rendered
    assert 'query: "algorithm reference"' in rendered
    assert 'query: "implementation reference"' in rendered


def test_joint_retrieval_source_queries_survive_checkpoint_round_trip(tmp_path):
    store = SearchStore(output_dir=tmp_path)
    record = store.add(
        ConstructedQuery(query="first"),
        [_doc("saved")],
        source_queries=["first", "second"],
        iteration=2,
        target_score=0.3,
    )
    store.record_usage(2, score=0.4)

    restored = SearchStore()
    restored.load_state_dict(json.loads(json.dumps(store.state_dict())))

    assert restored.records[0] == record
    assert restored.records[0].source_queries == ["first", "second"]
    assert restored.records[0].query.visit_count == 0


@pytest.mark.parametrize("source_queries", [None, ["single query"]])
def test_single_query_checkpoint_keeps_query_credit(source_queries):
    store = SearchStore()
    store.add(
        ConstructedQuery(query="single query"),
        [_doc("single")],
        source_queries=source_queries,
        iteration=5,
        target_score=0.2,
    )
    state = json.loads(json.dumps(store.state_dict()))
    restored = SearchStore()
    restored.load_state_dict(state)

    restored.record_usage(5, score=0.3)

    record = restored.records[0]
    assert record.query.visit_count == 1 and record.query.evolution_scores == [0.3]
    assert render_search_experience(restored).count("- query_ref:") == 1


def test_record_delta_and_delta_selection_use_only_measured_outcomes():
    great = _rated("great", 0.5)
    middling = _rated("middling", 0.05)
    awful = _rated("awful", -0.4)
    fresh = SearchRecord(query=ConstructedQuery(query="fresh"))
    newest = SearchRecord(query=ConstructedQuery(query="newest"))
    store = SearchStore(
        records=[great, middling, awful, fresh, newest],
        search_selection_policy="delta",
        search_selection_num=4,
    )

    assert [r.query.query for r in store.select_records()] == [
        "great",
        "middling",
        "awful",
        "newest",
    ]
    mixed = SearchRecord(
        impacts=[
            SearchImpact(delta=0.4),
            SearchImpact(delta=None),
            SearchImpact(delta=-0.2),
        ]
    )
    assert mixed.delta == pytest.approx(0.1)


def test_estimated_score_selection_falls_back_to_recent_unrated_records():
    low = SearchRecord(query=ConstructedQuery(query="low"), estimated_score=0.2)
    high = SearchRecord(query=ConstructedQuery(query="high"), estimated_score=0.8)
    older = SearchRecord(query=ConstructedQuery(query="older unrated"))
    newer = SearchRecord(query=ConstructedQuery(query="newer unrated"))
    store = SearchStore(
        records=[low, high, older, newer],
        search_selection_policy="estimated_score",
        search_selection_num=3,
    )

    assert [r.query.query for r in store.select_records()] == [
        "high",
        "low",
        "newer unrated",
    ]


def test_assessments_and_impacts_round_trip_with_stable_record_id(tmp_path):
    store = SearchStore(output_dir=tmp_path)
    record = store.add(
        ConstructedQuery(query="persisted"),
        [_doc("saved")],
        iteration=3,
        target_program_id="p",
        target_score=0.25,
        assessments=[
            SearchResultAssessment(
                document_ref="saved",
                relevance=0.8,
                novelty=0.7,
                estimated_score=0.75,
            )
        ],
    )
    store.record_usage(3, score=0.5, feedback="worked", result_program_id="c")

    restored = SearchStore()
    restored.load_state_dict(json.loads(json.dumps(store.state_dict())))
    loaded = restored.records[0]
    assert loaded.id == record.id
    assert loaded.assessments == record.assessments
    assert loaded.impacts == record.impacts
    assert loaded.impacts[0].delta > 0
    assert loaded.delta == pytest.approx(0.25)


def test_partial_document_assessment_does_not_create_partial_record_means():
    store = SearchStore()
    record = store.add(
        ConstructedQuery(query="partially rated"),
        [_doc("partial")],
        assessments=[SearchResultAssessment(relevance=0.9, novelty=0.8)],
    )

    assert record.relevance is None
    assert record.novelty is None
    assert record.helpfulness is None
    assert record.estimated_improvement_score is None
    assert record.estimated_score is None


def test_checkpoint_assessments_preserve_helpfulness_and_legacy_improvement():
    assessments = [
        SearchResultAssessment(
            document_ref="new",
            relevance=0.8,
            novelty=0.7,
            helpfulness=0.9,
        ),
        SearchResultAssessment(
            document_ref="legacy",
            relevance=0.6,
            novelty=0.3,
            estimated_improvement_score=-0.6,
            estimated_score=0.1,
        ),
    ]
    store = SearchStore()
    record = store.add(None, [_doc("new"), _doc("legacy")], assessments=assessments)

    assert record.relevance == pytest.approx(0.7)
    assert record.novelty == pytest.approx(0.5)
    assert record.helpfulness == pytest.approx(0.9)
    assert record.estimated_improvement_score == pytest.approx(-0.6)
    assert record.estimated_score == pytest.approx(0.1)
    assert record.delta is None


def test_incomplete_helpfulness_batch_does_not_create_partial_record_means():
    store = SearchStore()
    record = store.add(
        None,
        [_doc("complete"), _doc("partial")],
        assessments=[
            SearchResultAssessment(
                relevance=0.8,
                novelty=0.7,
                helpfulness=0.9,
            ),
            SearchResultAssessment(relevance=0.9, helpfulness=0.7),
        ],
    )

    assert record.relevance is None
    assert record.novelty is None
    assert record.helpfulness is None
    assert record.estimated_improvement_score is None
    assert record.estimated_score is None


def test_legacy_assessment_without_aggregate_score_remains_incomplete():
    record = SearchStore().add(
        None,
        [_doc("legacy partial")],
        assessments=[
            SearchResultAssessment(relevance=0.9, novelty=0.8, estimated_improvement_score=0.3)
        ],
    )

    assert record.relevance is None
    assert record.novelty is None
    assert record.helpfulness is None
    assert record.estimated_improvement_score is None
    assert record.estimated_score is None


def test_ranked_assessments_checkpoint_round_trip_without_aggregate_score(tmp_path):
    store = SearchStore(output_dir=tmp_path)
    record = store.add(
        ConstructedQuery(query="helpful documents"),
        [_doc("first"), _doc("second")],
        iteration=7,
        target_score=0.5,
        assessments=[
            SearchResultAssessment(
                document_ref="first",
                relevance=0.8,
                novelty=0.4,
                helpfulness=0.9,
            ),
            SearchResultAssessment(
                document_ref="second",
                relevance=0.6,
                novelty=0.6,
                helpfulness=0.5,
            ),
        ],
    )
    store.record_usage(7, score=0.4, result_program_id="child")
    restored = SearchStore()
    restored.load_state_dict(json.loads(json.dumps(store.state_dict())))

    loaded = restored.records[0]
    assert loaded == record
    assert [document.id for document in loaded.search_results] == ["first", "second"]
    assert [assessment.document_ref for assessment in loaded.assessments] == ["first", "second"]
    assert loaded.relevance == pytest.approx(0.7)
    assert loaded.novelty == pytest.approx(0.5)
    assert loaded.helpfulness == pytest.approx(0.7)
    assert loaded.estimated_improvement_score is None
    assert loaded.estimated_score is None
    assert all(assessment.estimated_score is None for assessment in loaded.assessments)
    assert loaded.delta == pytest.approx(-0.1)
    rendered = render_search_experience(restored)
    assert "helpfulness" not in rendered  # Old ratings remain stored, not used as forecasts.
    assert "actual outcome (combined evidence):" in rendered


def test_explicit_helpfulness_aggregate_is_preserved():
    record = SearchStore().add(None, [_doc("explicit")], helpfulness=0.75)

    assert record.helpfulness == pytest.approx(0.75)
    assert record.estimated_improvement_score is None


def test_record_feedback_is_bounded_and_not_duplicated_per_document():
    store = SearchStore(max_document_chars=8)
    query = ConstructedQuery(query="q")
    document = _doc("bounded")
    record = store.add(query, [document], iteration=2)

    store.record_usage(2, feedback="```abcdefghijk")

    assert record.impacts[0].feedback == "~~~abcde"
    assert document.evolution_feedbacks == []
    assert query.evolution_feedbacks == []
