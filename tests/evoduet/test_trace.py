"""Tests for the EvoDuet iteration-level trace writers."""

import json

from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval.base_retrieval import SearchResult
from skydiscover.evoduet.trace import (
    make_step,
    record_gate_decision,
    record_query_evolution,
)


def _doc(i):
    return SearchResult(raw_content="rc", content="c", id=f"d{i}", title=f"T{i}")


def test_make_step_structured_and_agentic():
    s = make_step(
        1,
        "reflect",
        ConstructedQuery(query="q", keywords=["k"], query_type="algo"),
        [_doc(1), _doc(2)],
    )
    assert s == {
        "step": 1,
        "kind": "reflect",
        "query": "q",
        "keywords": ["k"],
        "query_type": "algo",
        "num_docs": 2,
        "doc_ids": ["d1", "d2"],
        "doc_titles": ["T1", "T2"],
        "search_document_ids": [],  # Candidate documents have not entered the DB yet.
    }
    a = make_step(0, "agentic", None, [_doc(1)])
    assert a["query"] is None and a["num_docs"] == 1


def test_record_gate_decision_per_iteration_dir(tmp_path):
    for it, dec in [(0, "retrieve"), (1, "no-op"), (2, "retrieve")]:
        record_gate_decision(str(tmp_path), it, dec)
    for it, dec in [(0, "retrieve"), (1, "no-op"), (2, "retrieve")]:
        p = tmp_path / "evoduet" / f"checkpoint_{it}" / "gate_decision.json"
        assert json.loads(p.read_text()) == {"iteration": it, "decision": dec}


def test_record_query_evolution_per_iteration_dir(tmp_path):
    steps = [
        make_step(1, "candidate", ConstructedQuery(query="q0"), [_doc(1)]),
        make_step(2, "candidate", ConstructedQuery(query="q1"), [_doc(2)]),
        make_step(0, "selection", ConstructedQuery(query="q1"), [_doc(2)]),
    ]
    record_query_evolution(
        str(tmp_path),
        7,
        decision="retrieve",
        mode="pooled",
        steps=steps,
    )
    payload = json.loads(
        (tmp_path / "evoduet" / "checkpoint_7" / "query_evolution.json").read_text()
    )
    assert payload["iteration"] == 7 and payload["mode"] == "pooled"
    assert payload["num_steps"] == 3 and payload["final_query"] == "q1"
    assert payload["committed_doc_ids"] == ["d2"]
    assert [s["query"] for s in payload["steps"]] == ["q0", "q1", "q1"]


def test_record_query_evolution_grounding_dir(tmp_path):
    record_query_evolution(
        str(tmp_path),
        "grounding",
        decision="online_rag",
        mode="structured",
        steps=[make_step(0, "construct", ConstructedQuery(query="g"), [_doc(1)])],
    )
    assert (tmp_path / "evoduet" / "checkpoint_grounding" / "query_evolution.json").exists()


def test_writers_noop_without_output_dir(tmp_path):
    # None output_dir -> silently do nothing, never raise
    record_gate_decision(None, 0, "retrieve")
    record_query_evolution(None, 0, decision="retrieve", mode="structured", steps=[])
