"""Observed-search tests: no network, solution generation, or evaluator calls."""

import asyncio
import copy
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skydiscover.config import Config, EvoDuetConfig, LLMModelConfig, SearchSelectionConfig
from skydiscover.evoduet.layer import EvoDuet, format_documents
from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval import SearchResult
from skydiscover.evoduet.search import _prediction, _search_context
from skydiscover.search.base_database import Program


def document(n):
    return SearchResult(
        id=f"provider_{n}",
        title=f"Title {n}",
        url=f"https://example.test/{n}",
        content=f"SNIPPET_{n}",
        raw_content=f"RAW_{n}\n### heading\n```python\n{{current_program}}\n```",
        metadata={"full_provider_field": {"n": n}},
        iteration=20,
        retrieval_iteration=20,
    )


def knowledge_input(prompt):
    header = "[Current Knowledge State: What You Know and What You Still Do Not Know]\n"
    assert prompt.count(header) == 1
    return prompt.split(header, 1)[1].split("\n\n", 1)[0]


def document_blocks(prompt):
    return dict(
        re.findall(
            r"^\[\[Observed Document: (evidence_\d+)\]\]\n(.*?)^\[\[/Observed Document\]\]",
            prompt,
            re.M | re.S,
        )
    )


@pytest.mark.parametrize("score", [0.312, 0.0, -0.01])
def test_document_context_uses_latest_forecast_and_omits_missing_score(score):
    query = ConstructedQuery(query="source query")
    documents = {
        "evidence_2": (query, document(2)),
        "evidence_4": (query, document(4)),
    }
    rounds = [
        {
            "document_predictions": [
                {"evidence_ref": "evidence_2", "estimated_child_score": 0.31},
                {"evidence_ref": "evidence_3", "estimated_child_score": 0.4},
            ],
            "estimated_child_score": 0.8,
        },
        {
            "document_predictions": [
                {"evidence_ref": "evidence_2", "estimated_child_score": score}
            ],
            "estimated_child_score": 0.9,
        },
        {
            "prediction_error": "invalid response",
            "document_predictions": [{"evidence_ref": "evidence_2", "estimated_child_score": 0.99}],
        },
    ]
    original = copy.deepcopy((rounds, documents))
    rendered = _search_context(rounds, documents, 10000, parent_score=0.3)
    blocks = document_blocks(rendered)
    assert list(blocks) == ["evidence_2", "evidence_4"]
    assert blocks["evidence_2"].startswith(
        f"doc_id: evidence_2\nestimated_child_score: 0.3 -> {score}\n"
    )
    assert blocks["evidence_4"].startswith('doc_id: evidence_4\nsource_query: "source query"\n')
    assert "estimated_child_score:" not in blocks["evidence_4"]
    assert rendered.startswith("[[Observed Document:")
    assert all(f"-> {score}" not in rendered for score in (0.8, 0.9, 0.99))
    assert "evidence_3" not in rendered and "invalid response" not in rendered
    assert all('source_query: "source query"' in block for block in blocks.values())
    assert "content:\n" + document(2).raw_content in blocks["evidence_2"]
    assert "SNIPPET" not in rendered
    assert (rounds, documents) == original
    assert _search_context(rounds, {}, 10000, parent_score=0.3) == "(none)"


class Model:
    def __init__(self, decision="retrieve"):
        self.decision = decision
        self.events, self.calls = [], []
        self.queries = self.predictions = self.searches = 0
        self.fail_searches, self.fail_predictions, self.fail_queries = set(), set(), set()
        self.prediction_overrides = {}
        self.query_labels = []
        self.results_per_search = 1
        self.after_search = None
        self.retrieve = AsyncMock(side_effect=self.search)

    async def search(self, query, **kwargs):
        self.events.append("search")
        self.searches += 1
        assert kwargs["max_results"] == 5
        if self.after_search:
            self.after_search()
        if self.searches in self.fail_searches:
            raise RuntimeError("search unavailable")
        if self.results_per_search == 1:
            return [document(self.searches)]
        return [document(self.searches * 10 + i) for i in range(1, self.results_per_search + 1)]

    async def generate(self, system, messages, *, llm_context, **kwargs):
        phase = llm_context["phase"]
        if phase == "wk-query" or phase.startswith("wk-query-sample-"):
            self.query_labels.append(phase)
            phase = "wk-query"
        prompt = messages[0]["content"]
        self.calls.append((phase, system, prompt))
        self.events.append(phase)
        if phase == "wk-retrieval-gate":
            response = json.dumps(
                {
                    "decision": self.decision,
                    "knowledge_state_analysis": "INITIAL_GAP",
                    "reasoning": "choose evidence",
                    "search_document_ids": ["doc_000001"],
                }
            )
        elif phase == "wk-population-analysis":
            response = "POPULATION_REPORT"
        elif phase == "wk-knowledge-state-analysis":
            response = "HEURISTIC_INITIAL_KNOWLEDGE"
        elif phase == "wk-query":
            self.queries += 1
            response = (
                "bad query"
                if self.queries in self.fail_queries
                else json.dumps({"query": f"QUERY_{self.queries}"})
            )
        elif phase == "wk-query-prediction":
            self.predictions += 1
            refs = [
                ref
                for ref, block in document_blocks(prompt).items()
                if "estimated_child_score:" not in block
            ]
            response = (
                "bad prediction"
                if self.predictions in self.fail_predictions
                else json.dumps(
                    {
                        "document_predictions": [
                            {
                                "evidence_ref": ref,
                                "estimated_child_score": 0.4
                                + int(ref.removeprefix("evidence_")) / 100,
                            }
                            for ref in reversed(refs)
                        ],
                        "knowledge_state_analysis": f"KNOWN_DETAIL_{self.predictions}; MISSING_DETAIL_{self.predictions}",
                    }
                )
            )
            response = self.prediction_overrides.get(self.predictions, response)
        else:
            raise AssertionError(f"Unexpected inner-loop call: {phase}")
        return SimpleNamespace(text=response)

    def prompts(self, phase):
        return [prompt for label, _, prompt in self.calls if label == phase]


def layer_for(monkeypatch, tmp_path, model=None, n=1, **overrides):
    model = model or Model()
    config = Config()
    config.num_generations = n
    config.llm.models = [
        LLMModelConfig(
            name="local-test-model",
            max_tokens=32768,
            init_client=lambda _: SimpleNamespace(generate=model.generate),
        )
    ]
    values = dict(
        enabled=True,
        retrieval_gating_backend_type="llm",
        retrieval_gating_prompt_template_name="retrieval_gating",
        population_state_prompt_template_name="population_analysis",
        knowledge_state_analysis_prompt_template_name="knowledge_analysis",
        query_optimization_max_rounds=3,
        query_optimization_queries_per_round=1,
        search_result_top_k=2,
        search_selection=SearchSelectionConfig(policy="recency", num=10),
        documents_per_entry=3,
    )
    values.update(overrides)
    config.evoduet = EvoDuetConfig(**values)
    config.evoduet.tavily_retrieval.max_results = 5
    monkeypatch.setattr(
        EvoDuet,
        "_select_retrieval_backend",
        lambda self: SimpleNamespace(retrieve=model.retrieve),
    )
    layer = EvoDuet(config, output_dir=tmp_path)
    return layer, model


def parent_program():
    return Program(
        id="parent",
        solution="def main(): return 42  # FIXED_PARENT",
        metrics={"combined_score": 0.4},
    )


def run(layer, parent=None):
    parent = parent or parent_program()
    return asyncio.run(
        layer.run(parent=parent, population=[parent], history="NATIVE_HISTORY", iteration=20)
    )


def test_actual_search_order_cumulative_evidence_and_feedback(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path)
    old = layer.search_store.add(
        ConstructedQuery(query="OLD_QUERY"), [document(99)], iteration=1, target_score=0.2
    )
    layer.search_store.record_usage(1, score=0.3, result_program_id="older_child")
    snapshot = layer._selected_search_context()
    documents = run(layer)
    assert (
        model.events
        == ["wk-retrieval-gate", "wk-population-analysis"]
        + ["wk-query", "search", "wk-query-prediction"] * 3
    )
    assert model.searches == 3 and [doc.id for doc in documents] == ["provider_3", "provider_2"]
    queries, predictions = model.prompts("wk-query"), model.prompts("wk-query-prediction")
    expected_states = [
        "INITIAL_GAP",
        "KNOWN_DETAIL_1; MISSING_DETAIL_1",
        "KNOWN_DETAIL_2; MISSING_DETAIL_2",
    ]
    assert [knowledge_input(prompt) for prompt in queries] == expected_states
    assert [knowledge_input(prompt) for prompt in predictions] == expected_states
    assert all("INITIAL_GAP" not in prompt for prompt in queries[1:] + predictions[1:])
    for i in range(1, 3):
        assert f"RAW_{i}" in queries[i] and f"MISSING_DETAIL_{i}" in queries[i]
        assert f"estimated_child_score: 0.4 -> {0.4 + i / 100}" in queries[i]
    for prompt in queries + predictions:
        assert snapshot in prompt and "FIXED_PARENT" in prompt and "NATIVE_HISTORY" in prompt
    assert not hasattr(layer, "reranking")
    assert not hasattr(layer, "query_policy")
    assert not hasattr(layer, "policy_optimizer")
    assert "wk-reranking" not in model.events
    assert "RAW_1" in predictions[-1]  # the joint call must see every candidate
    assert "RAW_2" in predictions[-1] and "RAW_3" in predictions[-1]
    assert "RAW_1\n" not in format_documents(documents, 10000)
    assert "SNIPPET_3" not in predictions[-1]
    assert "{current_program}" in predictions[-1]  # literal braces in document data
    assert "{observed_documents}" not in predictions[-1]
    assert all("{search_context}" not in prompt for prompt in queries + predictions)
    # Only current document blocks are supplied, with individual prior forecasts.
    blocks = document_blocks(predictions[-1])
    assert list(blocks) == ["evidence_2", "evidence_1", "evidence_3"]
    for n in (1, 2):
        assert f"estimated_child_score: 0.4 -> {0.4 + n / 100}" in blocks[f"evidence_{n}"]
    assert "estimated_child_score:" not in blocks["evidence_3"]
    assert all(f'source_query: "QUERY_{n}"' in blocks[f"evidence_{n}"] for n in (1, 2, 3))
    assert all("-> 0.52" not in block for block in blocks.values())
    assert "Searches and document selections" not in predictions[-1]
    assert all(predictions[-1].count(f"RAW_{n}\n") == 1 for n in (1, 2, 3))
    assert "{top_k}" not in predictions[-1]
    assert all(
        "[Search Budget]" not in prompt
        and "one web search per round" not in prompt
        and "{search_budget}" not in prompt
        and "{round}" not in prompt
        for prompt in queries + predictions
    )
    assert all(
        system == "" for phase, system, _ in model.calls if phase != "wk-population-analysis"
    )
    population = next(
        (system, prompt)
        for phase, system, prompt in model.calls
        if phase == "wk-population-analysis"
    )
    expected = (
        Path(__file__).resolve().parents[2] / "skydiscover/evoduet/prompts/population_analysis.txt"
    ).read_text()
    assert population[0] == expected and population[1].startswith("Population Statistics:")
    assert all(pool.models_cfg[0].max_tokens == 32768 for pool in layer.solution_model_pool.pools)
    record = layer.search_store.records[-1]
    assert len(layer.search_store.records) == 2 and old.impacts[0].result_score == 0.3
    assert record.source_queries == ["QUERY_1", "QUERY_2", "QUERY_3"]
    assert not record.impacts
    assert record.assessments == []
    assert record.search_results[0].content == "SNIPPET_3"
    assert documents[0].content == document(3).raw_content
    assert "RAW_3" in format_documents(documents, 10000)
    trace = layer.query_optimization_history[-1]
    assert trace["initial_knowledge_state_analysis"] == "INITIAL_GAP"
    assert trace["knowledge_state_analysis"] == "KNOWN_DETAIL_3; MISSING_DETAIL_3"
    assert [row["knowledge_state_analysis"] for row in trace["rounds"]] == [
        f"KNOWN_DETAIL_{n}; MISSING_DETAIL_{n}" for n in (1, 2, 3)
    ]
    assert trace["status"] == "awaiting_outcome" and trace["estimated_child_score"] is None
    assert [len(row["document_predictions"]) for row in trace["rounds"]] == [1, 1, 1]
    assert trace["search_record_id"] == record.id
    assert [row["candidate_count"] for row in trace["rounds"]] == [1, 2, 3]
    assert [row["pool_size"] for row in trace["rounds"]] == [1, 2, 2]
    assert trace["rounds"][0]["documents"][0]["metadata"] == {"full_provider_field": {"n": 1}}
    assert [entry["search_document_id"] for entry in trace["evidence"]] == [
        doc.search_document_id for doc in documents
    ]


@pytest.mark.parametrize("n", [1, 8])
def test_writeback_checkpoint_and_evidence_set_rendering(monkeypatch, tmp_path, n):
    layer, model = layer_for(monkeypatch, tmp_path, n=n)
    docs = run(layer)
    assert all(
        "num_generations" not in prompt
        and "generation_condition" not in prompt
        and "Solution Generation Condition" not in prompt
        and "best valid child" not in prompt
        for _, _, prompt in model.calls
    )
    state = copy.deepcopy(layer.state_dict())
    restored, _ = layer_for(monkeypatch, tmp_path / "restored", n=n)
    restored.load_state_dict(state)
    restored.record_usage(
        iteration=20, score=0.45, result_program_id="child", feedback="actual feedback"
    )
    trace = restored.query_optimization_history[-1]
    record = restored.search_store.records[-1]
    assert trace["generation_condition"]["num_generations"] == n
    assert trace["prediction_target"] == "individual_documents"
    assert trace["estimated_child_score"] is None and trace["result_score"] == 0.45
    assert trace["actual_improvement"] == pytest.approx(0.05)
    assert record.impacts[0].target_score == 0.4 and record.impacts[0].result_score == 0.45
    assert all(len(doc.evolution_scores) == 1 for doc in record.search_results)
    rendered = restored._selected_search_context()
    assert "estimated_child_score (selected evidence together)" not in rendered
    assert rendered.count("estimated_child_score:") == 2
    assert f"estimated_child_score: 0.4 -> {0.4 + 3 / 100}" in rendered
    assert "num_generations" not in rendered and "score_target" not in rendered
    assert "generation condition" not in rendered and "0.4 -> 0.45" in rendered
    assert "source_query_ref: q3" in rendered and "source_query_ref: q2" in rendered
    assert "RAW_1" not in rendered
    assert "[[Search Document: " + docs[0].search_document_id + "]]" in rendered
    assert "[[/Search Document]]" in rendered and "### heading" in rendered
    persisted = json.loads(
        (tmp_path / "restored/evoduet/checkpoint_20/query_optimization.json").read_text()
    )
    assert persisted == trace


def test_parent_and_past_database_are_frozen_across_awaits(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path)
    parent = parent_program()

    def mutate_external_state():
        parent.solution = "EXTERNAL_MUTATION"
        layer.search_store.add(ConstructedQuery(query="CONCURRENT_DB_WRITE"), [], iteration=500)

    model.after_search = mutate_external_state
    run(layer, parent)
    for prompt in model.prompts("wk-query") + model.prompts("wk-query-prediction"):
        assert "FIXED_PARENT" in prompt
        assert "EXTERNAL_MUTATION" not in prompt and "CONCURRENT_DB_WRITE" not in prompt


@pytest.mark.parametrize("decision", ["no-op", "look-up"])
def test_noop_lookup_skip_inner_loop_and_preserve_provenance(monkeypatch, tmp_path, decision):
    layer, model = layer_for(monkeypatch, tmp_path, Model(decision), n=8)
    original = layer.search_store.add(
        ConstructedQuery(query="ORIGINAL_QUERY"), [document(1)], iteration=2
    )
    result = run(layer)
    assert model.events == ["wk-retrieval-gate"]
    model.retrieve.assert_not_awaited()
    if decision == "no-op":
        assert not result and not layer.query_optimization_history
    else:
        assert result[0].search_document_id == original.search_results[0].search_document_id
        record = layer.search_store.records[-1]
        assert record.operation == "look-up" and record.lookup_provenance
        assert record.search_results[0].metadata == original.search_results[0].metadata
        layer.record_usage(iteration=20, score=0.5, result_program_id="lookup_child")
        assert record.impacts[0].result_score == 0.5 and not original.impacts
        assert layer.query_optimization_history[-1]["generation_condition"]["num_generations"] == 8


def test_failures_stay_bounded_and_preserve_last_evidence(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path)
    model.fail_searches = {2, 3}
    model.fail_predictions = {3}
    docs = run(layer)
    assert [doc.id for doc in docs] == ["provider_1"]
    assert model.searches == 3
    trace = layer.query_optimization_history[-1]
    assert trace["estimated_child_score"] is None  # no stale successful forecast
    assert "search_error" in trace["rounds"][1]
    assert trace["rounds"][2]["selection_status"] == "retained_previous"
    assert trace["rounds"][2]["assessment_status"] == "no_new_documents"
    assert model.predictions == 1
    assert trace["rounds"][1]["search_error"] == "search unavailable"
    assert "search unavailable" not in model.prompts("wk-query")[2]


def test_failed_new_document_response_preserves_last_selection(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path)
    model.fail_predictions = {3}
    docs = run(layer)
    assert [doc.id for doc in docs] == ["provider_2", "provider_1"]
    assert layer.query_optimization_history[-1]["estimated_child_score"] is None


@pytest.mark.parametrize("failure", ["query", "search", "prediction"])
def test_all_failed_attempts_have_no_fabricated_evidence_outcome(monkeypatch, tmp_path, failure):
    layer, model = layer_for(monkeypatch, tmp_path)
    if failure == "query":
        model.fail_queries = {1, 2, 3}
    elif failure == "search":
        model.fail_searches = {1, 2, 3}
    else:
        model.fail_predictions = {1, 2, 3}
    assert run(layer) == []
    trace = layer.query_optimization_history[-1]
    assert trace["status"] == "no_evidence"
    assert trace["search_attempts"] == (0 if failure == "query" else 3)
    layer.record_usage(iteration=20, score=0.9, result_program_id="child_without_evidence")
    assert not layer.search_store.records[-1].impacts
    assert "result_score" not in trace and trace["estimated_child_score"] is None


def test_duplicates_accumulate_once_and_record_all_observations(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path)
    model.retrieve.side_effect = None
    model.retrieve.return_value = [document(1)]
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    assert len(docs) == 1 and docs[0].query == "QUERY_1"
    assert model.predictions == 1
    assert [row["pool_size"] for row in trace["rounds"]] == [1, 1, 1]
    assert [row["documents"][0]["query"] for row in trace["rounds"]] == [
        "QUERY_1",
        "QUERY_2",
        "QUERY_3",
    ]


def test_only_score_top_k_survives_into_next_query_and_candidate_set(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path, search_result_top_k=3)
    model.results_per_search = 5
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    assert [row["candidate_count"] for row in trace["rounds"]] == [5, 8, 8]
    assert [row["pool_size"] for row in trace["rounds"]] == [3, 3, 3]
    assert trace["rounds"][1]["candidate_evidence_refs"] == [
        "evidence_5",
        "evidence_4",
        "evidence_3",
        "evidence_6",
        "evidence_7",
        "evidence_8",
        "evidence_9",
        "evidence_10",
    ]
    queries, predictions = model.prompts("wk-query"), model.prompts("wk-query-prediction")
    assert all(f"RAW_{n}\n" in queries[1] for n in (13, 14, 15))
    assert all(f"RAW_{n}\n" not in queries[1] for n in (11, 12))
    assert all(f"RAW_{n}\n" in queries[2] for n in (23, 24, 25))
    assert all(f"RAW_{n}\n" not in queries[2] for n in (11, 12, 13, 14, 15, 21, 22))
    assert "RAW_11\n" not in predictions[1] and "RAW_21\n" not in predictions[2]
    assert all('"document_predictions"' not in prompt for prompt in queries)
    assert [doc.id for doc in docs] == ["provider_35", "provider_34", "provider_33"]
    assert trace["selected_evidence_refs"] == ["evidence_15", "evidence_14", "evidence_13"]
    assert [item["pool_ref"] for item in trace["evidence"]] == trace["selected_evidence_refs"]
    # Pruning affects only active evidence, never the diagnostic search records.
    assert len(trace["rounds"][0]["documents"]) == 5
    assert trace["rounds"][0]["documents"][0]["raw_content"] == document(11).raw_content
    assert [len(row["document_predictions"]) for row in trace["rounds"]] == [5, 5, 5]
    assert not layer.search_store.records[-1].impacts


def test_rejected_document_returns_only_if_searched_again_with_stable_reference(
    monkeypatch, tmp_path
):
    layer, model = layer_for(monkeypatch, tmp_path, search_result_top_k=1)
    model.retrieve.side_effect = [
        [document(1), document(2)],
        [document(3)],
        [document(1), document(4)],
    ]
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    assert trace["rounds"][1]["candidate_evidence_refs"] == ["evidence_2", "evidence_3"]
    assert trace["rounds"][2]["candidate_evidence_refs"] == [
        "evidence_3",
        "evidence_1",
        "evidence_4",
    ]
    assert trace["rounds"][2]["document_evidence_refs"] == ["evidence_1", "evidence_4"]
    assert docs[0].id == "provider_4"


def test_failed_prediction_does_not_admit_new_documents_into_local_pool(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path, search_result_top_k=3)
    model.results_per_search = 5
    model.fail_predictions = {2}
    run(layer)
    trace = layer.query_optimization_history[-1]
    assert trace["rounds"][1]["pool_size"] == 3
    assert (
        trace["rounds"][1]["selected_evidence_refs"] == trace["rounds"][0]["selected_evidence_refs"]
    )
    query = model.prompts("wk-query")[2]
    assert knowledge_input(query) == "KNOWN_DETAIL_1; MISSING_DETAIL_1"
    assert knowledge_input(model.prompts("wk-query-prediction")[2]) == knowledge_input(query)
    assert trace["rounds"][1]["knowledge_state_analysis"] == knowledge_input(query)
    assert all(f"RAW_{n}\n" in query for n in (13, 14, 15))
    assert all(f"RAW_{n}\n" not in query for n in (21, 22, 23, 24, 25))
    assert trace["rounds"][2]["candidate_count"] == 8


def test_search_without_evidence_preserves_initial_knowledge(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path)
    model.retrieve.side_effect = [[], RuntimeError("search unavailable"), [document(1)]]
    run(layer)
    queries = model.prompts("wk-query")
    assert [knowledge_input(prompt) for prompt in queries] == ["INITIAL_GAP"] * 3
    assert knowledge_input(model.prompts("wk-query-prediction")[0]) == "INITIAL_GAP"
    trace = layer.query_optimization_history[-1]
    assert all(row["search_status"] == "no_usable_documents" for row in trace["rounds"][:2])
    assert trace["knowledge_state_analysis"] == "KNOWN_DETAIL_1; MISSING_DETAIL_1"


@pytest.mark.parametrize("state", [None, "", " \n\t", [], {}])
def test_invalid_knowledge_state_does_not_replace_evidence_or_prior_analysis(
    monkeypatch, tmp_path, state
):
    layer, model = layer_for(monkeypatch, tmp_path)
    model.prediction_overrides[2] = json.dumps(
        {
            "document_predictions": [
                {"evidence_ref": ref, "estimated_child_score": 0.9}
                for ref in ("evidence_1", "evidence_2")
            ],
            "estimated_child_score": 0.9,
            "reason": "forecast",
            "knowledge_state_analysis": state,
        }
    )
    run(layer)
    trace = layer.query_optimization_history[-1]
    assert trace["rounds"][1]["selected_evidence_refs"] == ["evidence_1"]
    assert trace["rounds"][1]["knowledge_state_analysis"] == "KNOWN_DETAIL_1; MISSING_DETAIL_1"
    assert knowledge_input(model.prompts("wk-query")[2]) == "KNOWN_DETAIL_1; MISSING_DETAIL_1"
    assert "RAW_2\n" not in model.prompts("wk-query")[2]


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"estimated_child_score":true,"reason":"","knowledge_state_analysis":""}',
        '{"estimated_child_score":NaN,"reason":"","knowledge_state_analysis":""}',
        '{"estimated_child_score":0.5}',
        "null",
    ],
)
def test_invalid_evidence_predictions_are_unknown(text):
    with pytest.raises(ValueError):
        _prediction(text, available_refs=["evidence_1"], top_k=1)


def test_evidence_prediction_accepts_negative_absolute_score_and_rejects_duplicates():
    result = _prediction(
        '```json\n{"document_predictions": [{"evidence_ref": "evidence_1", "estimated_child_score": -10.6}], "estimated_child_score": -10.5, "reason": "risk", "knowledge_state_analysis": "details"}\n```',
        available_refs=["evidence_1"],
        top_k=1,
    )
    assert result["document_predictions"][0]["estimated_child_score"] == -10.6
    assert "estimated_child_score" not in result  # no aggregate forecast
    with pytest.raises(ValueError, match="duplicate"):
        _prediction(
            '{"estimated_child_score": 0.2, "estimated_child_score": 0.9, "reason": "", "knowledge_state_analysis": ""}',
            available_refs=["evidence_1"],
            top_k=1,
        )


@pytest.mark.parametrize(
    "rows",
    [None, "evidence_1", {}, 1],
)
def test_prediction_still_requires_a_document_predictions_list(rows):
    text = json.dumps(
        {
            "document_predictions": rows,
            "estimated_child_score": 0.5,
            "reason": "forecast",
            "knowledge_state_analysis": "missing",
        }
    )
    with pytest.raises(ValueError):
        _prediction(text, available_refs=["evidence_1", "evidence_2"], top_k=1)


@pytest.mark.parametrize("invalid_score", [None, True, "0.6", 10**400])
def test_prediction_filters_bad_rows_without_losing_valid_new_scores(invalid_score):
    rows = [
        {"evidence_ref": "evidence_3", "estimated_child_score": -0.2},
        {"evidence_ref": "evidence_1", "estimated_child_score": 999},
        {"evidence_ref": "evidence_999", "estimated_child_score": 0.9},
        {"evidence_ref": ["evidence_2"], "estimated_child_score": 0.9},
        "not an object",
        {"evidence_ref": "evidence_2", "estimated_child_score": invalid_score},
        {"evidence_ref": "evidence_4", "estimated_child_score": 0.0},
    ]
    previous = {"evidence_1": -0.1}
    result = _prediction(
        json.dumps({"document_predictions": rows, "knowledge_state_analysis": "proposed"}),
        available_refs=["evidence_1", "evidence_2", "evidence_4", "evidence_3"],
        top_k=4,
        previous_scores=previous,
    )
    assert previous == {"evidence_1": -0.1}
    assert result["document_predictions"] == [
        {"evidence_ref": "evidence_4", "estimated_child_score": 0.0},
        {"evidence_ref": "evidence_3", "estimated_child_score": -0.2},
    ]
    assert result["selected_evidence_refs"] == ["evidence_4", "evidence_1", "evidence_3"]
    assert result["assessment_status"] == "partial"
    assert result["unscored_document_refs"] == ["evidence_2"]
    assert result["dropped_predictions"] == [
        {"index": 1, "evidence_ref": "evidence_1", "reason": "already_scored"},
        {"index": 2, "evidence_ref": "evidence_999", "reason": "not_in_prediction_targets"},
        {"index": 3, "evidence_ref": None, "reason": "invalid_reference"},
        {"index": 4, "evidence_ref": None, "reason": "not_an_object"},
        {"index": 5, "evidence_ref": "evidence_2", "reason": "invalid_score"},
    ]
    assert result["knowledge_state_analysis"] == "proposed"
    assert "prediction_error" not in result


@pytest.mark.parametrize("first_score", [0.7, None])
def test_duplicate_new_references_drop_all_occurrences_and_preserve_other_scores(first_score):
    result = _prediction(
        json.dumps(
            {
                "document_predictions": [
                    {"evidence_ref": "evidence_2", "estimated_child_score": first_score},
                    {"evidence_ref": "evidence_3", "estimated_child_score": 0.6},
                    {"evidence_ref": "evidence_2", "estimated_child_score": 0.9},
                ],
                "knowledge_state_analysis": "ambiguous duplicate",
            }
        ),
        available_refs=["evidence_1", "evidence_2", "evidence_3"],
        previous_scores={"evidence_1": 0.8},
        top_k=3,
    )
    assert result["document_predictions"] == [
        {"evidence_ref": "evidence_3", "estimated_child_score": 0.6}
    ]
    assert result["selected_evidence_refs"] == ["evidence_1", "evidence_3"]
    assert result["unscored_document_refs"] == ["evidence_2"]
    assert result["assessment_status"] == "partial"
    assert result["dropped_predictions"] == [
        {"index": index, "evidence_ref": "evidence_2", "reason": "duplicate_reference"}
        for index in (0, 2)
    ]


def test_ac1_response_preserves_existing_scores_and_salvages_evidence_12_through_15():
    score = 0.6595576276601904
    previous = {"evidence_10": 0.75, "evidence_4": 0.68, "evidence_1": 0.7}
    result = _prediction(
        json.dumps(
            {
                "document_predictions": [
                    {"evidence_ref": f"evidence_{n}", "estimated_child_score": score}
                    for n in (1, 12, 13, 14, 15)
                ],
                "knowledge_state_analysis": "Evidence_1 was newly assessed.",
            }
        ),
        available_refs=list(previous) + [f"evidence_{n}" for n in range(11, 16)],
        previous_scores=previous,
        top_k=4,
    )
    assert previous == {"evidence_10": 0.75, "evidence_4": 0.68, "evidence_1": 0.7}
    assert result["document_predictions"] == [
        {"evidence_ref": f"evidence_{n}", "estimated_child_score": score} for n in range(12, 16)
    ]
    assert result["selected_evidence_refs"] == [
        "evidence_10",
        "evidence_1",
        "evidence_4",
        "evidence_12",
    ]
    assert result["unscored_document_refs"] == ["evidence_11"]
    assert result["dropped_predictions"] == [
        {"index": 0, "evidence_ref": "evidence_1", "reason": "already_scored"}
    ]
    assert result["assessment_status"] == "partial"


def test_top_k_uses_numeric_scores_not_output_order_or_model_selected_refs():
    text = json.dumps(
        {
            "document_predictions": [
                {"evidence_ref": "evidence_9", "estimated_child_score": 0.1},
                {"evidence_ref": "evidence_2", "estimated_child_score": 0.9},
                {"evidence_ref": "evidence_7", "estimated_child_score": 0.9},
            ],
            "selected_evidence_refs": ["evidence_9"],
            "estimated_child_score": 0.8,
            "reason": "joint",
            "knowledge_state_analysis": "gap",
        }
    )
    result = _prediction(text, available_refs=["evidence_7", "evidence_9", "evidence_2"], top_k=2)
    assert result["selected_evidence_refs"] == ["evidence_7", "evidence_2"]
    assert "estimated_child_score" not in result


def test_new_scores_compete_with_fixed_existing_scores_without_modifying_them():
    previous = {"evidence_1": 0.8, "evidence_2": 0.0}
    response = {
        "document_predictions": [
            {"evidence_ref": "evidence_4", "estimated_child_score": 0.6},
            {"evidence_ref": "evidence_3", "estimated_child_score": 0.5},
        ],
        "knowledge_state_analysis": "New documents clarify one issue; another is unresolved.",
    }
    result = _prediction(
        json.dumps(response),
        available_refs=["evidence_1", "evidence_2", "evidence_3", "evidence_4"],
        top_k=2,
        previous_scores=previous,
    )
    assert previous == {"evidence_1": 0.8, "evidence_2": 0.0}
    assert result["selected_evidence_refs"] == ["evidence_1", "evidence_4"]
    assert [row["evidence_ref"] for row in result["document_predictions"]] == [
        "evidence_3",
        "evidence_4",
    ]
    assert result["knowledge_state_analysis"] == response["knowledge_state_analysis"]
    assert result["assessment_status"] == "scored"
    assert result["dropped_predictions"] == result["unscored_document_refs"] == []


@pytest.mark.parametrize("refs", [["evidence_1"], [], ["evidence_1", "evidence_2"]])
def test_prediction_drops_rescoring_old_documents_and_excludes_missing_new_scores(refs):
    response = {
        "document_predictions": [
            {"evidence_ref": ref, "estimated_child_score": 0.9} for ref in refs
        ],
        "knowledge_state_analysis": "updated",
    }
    result = _prediction(
        json.dumps(response),
        available_refs=["evidence_1", "evidence_2"],
        top_k=1,
        previous_scores={"evidence_1": 0.0},
    )
    if "evidence_2" in refs:
        assert result["selected_evidence_refs"] == ["evidence_2"]
        assert result["document_predictions"] == [
            {"evidence_ref": "evidence_2", "estimated_child_score": 0.9}
        ]
        assert result["assessment_status"] == "partial"
        assert result["unscored_document_refs"] == []
    else:
        assert result["selected_evidence_refs"] == ["evidence_1"]
        assert result["document_predictions"] == []
        assert result["assessment_status"] == "no_valid_predictions"
        assert result["unscored_document_refs"] == ["evidence_2"]
    assert "prediction_error" not in result


def test_invalid_selection_never_changes_stored_evidence_or_attributes_its_score(
    monkeypatch, tmp_path
):
    layer, model = layer_for(monkeypatch, tmp_path)
    model.prediction_overrides[3] = json.dumps(
        {
            "document_predictions": [
                {"evidence_ref": ref, "estimated_child_score": 0.9}
                for ref in ["evidence_1", "evidence_2", "evidence_999"]
            ],
            "estimated_child_score": 999,
            "reason": "invalid selection",
            "knowledge_state_analysis": "invalid gap",
        }
    )
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    assert [doc.id for doc in docs] == ["provider_2", "provider_1"]
    assert trace["estimated_child_score"] is None
    assert trace["knowledge_state_analysis"] == "KNOWN_DETAIL_2; MISSING_DETAIL_2"
    assert trace["rounds"][2]["selected_evidence_refs"] == ["evidence_2", "evidence_1"]
    assert trace["rounds"][2]["selection_status"] == "retained_previous"
    assert trace["rounds"][2]["assessment_status"] == "no_valid_predictions"
    assert trace["rounds"][2]["unscored_document_refs"] == ["evidence_3"]
    assert "prediction_error" not in trace["rounds"][2]
    assert "reranking" not in trace["prompt_templates"]
    assert all("assessments" not in row for row in trace["rounds"])
    layer.record_usage(iteration=20, score=0.45, result_program_id="child")
    assert [doc.id for doc in layer.search_store.records[-1].search_results] == [
        "provider_2",
        "provider_1",
    ]
    assert trace["result_score"] == 0.45 and trace["estimated_child_score"] is None


def test_partial_prediction_reuses_scores_and_retains_knowledge_through_checkpoint(
    monkeypatch, tmp_path
):
    layer, model = layer_for(monkeypatch, tmp_path)
    model.retrieve.side_effect = [
        [document(1)],
        [document(2), document(3)],
        [document(4)],
    ]
    model.prediction_overrides[2] = json.dumps(
        {
            "document_predictions": [
                {"evidence_ref": "evidence_1", "estimated_child_score": 100},
                {"evidence_ref": "evidence_2", "estimated_child_score": 0.9},
            ],
            "knowledge_state_analysis": "CONFUSED_PARTIAL_ANALYSIS",
        }
    )
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    partial = trace["rounds"][1]
    assert [doc.id for doc in docs] == ["provider_2", "provider_4"]
    assert partial["assessment_status"] == "partial"
    assert partial["document_predictions"] == [
        {"evidence_ref": "evidence_2", "estimated_child_score": 0.9}
    ]
    assert partial["selected_evidence_refs"] == ["evidence_2", "evidence_1"]
    assert partial["unscored_document_refs"] == ["evidence_3"]
    assert partial["knowledge_state_status"] == "retained_previous"
    assert partial["knowledge_state_analysis"] == "KNOWN_DETAIL_1; MISSING_DETAIL_1"
    assert partial["proposed_knowledge_state_analysis"] == "CONFUSED_PARTIAL_ANALYSIS"
    assert "prediction_error" not in partial
    for prompt in (model.prompts("wk-query")[2], model.prompts("wk-query-prediction")[2]):
        assert knowledge_input(prompt) == "KNOWN_DETAIL_1; MISSING_DETAIL_1"
        assert "CONFUSED_PARTIAL_ANALYSIS" not in prompt
        blocks = document_blocks(prompt)
        assert "estimated_child_score: 0.4 -> 0.9" in blocks["evidence_2"]
        assert f"estimated_child_score: 0.4 -> {0.4 + 1 / 100}" in blocks["evidence_1"]
        assert "evidence_3" not in blocks
    assert trace["rounds"][2]["new_document_refs"] == ["evidence_4"]
    assert trace["rounds"][2]["knowledge_state_status"] == "updated"
    assert trace["knowledge_state_analysis"] == "KNOWN_DETAIL_3; MISSING_DETAIL_3"
    assert [entry["estimated_child_score"] for entry in trace["evidence"]] == [0.9, 0.44]
    persisted = json.loads((tmp_path / "evoduet/checkpoint_20/query_optimization.json").read_text())
    assert persisted == trace

    restored, _ = layer_for(monkeypatch, tmp_path / "restored_partial")
    restored.load_state_dict(copy.deepcopy(layer.state_dict()))
    restored.record_usage(iteration=20, score=0.47, result_program_id="actual_child")
    restored_trace = restored.query_optimization_history[-1]
    record = restored.search_store.records[-1]
    assert restored_trace["rounds"][1] == partial
    assert restored_trace["estimated_child_score"] is None
    assert restored_trace["result_score"] == 0.47
    assert [doc.id for doc in record.search_results] == ["provider_2", "provider_4"]
    assert record.impacts[0].result_program_id == "actual_child"
    assert all(len(doc.evolution_scores) == 1 for doc in record.search_results)
    rendered = restored._selected_search_context()
    assert "estimated_child_score: 0.4 -> 0.9" in rendered
    assert "RAW_3\n" not in rendered and "CONFUSED_PARTIAL_ANALYSIS" not in rendered


def test_omitted_document_can_receive_a_score_when_retrieved_again(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path, query_optimization_max_rounds=2)
    model.retrieve.side_effect = [[document(1), document(2)], [document(1), document(2)]]
    model.prediction_overrides[1] = json.dumps(
        {
            "document_predictions": [{"evidence_ref": "evidence_1", "estimated_child_score": 0.8}],
            "knowledge_state_analysis": "INCOMPLETE_KNOWLEDGE",
        }
    )
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    assert [doc.id for doc in docs] == ["provider_1", "provider_2"]
    assert trace["rounds"][0]["assessment_status"] == "partial"
    assert trace["rounds"][0]["unscored_document_refs"] == ["evidence_2"]
    assert trace["rounds"][1]["new_document_refs"] == ["evidence_2"]
    assert trace["rounds"][1]["assessment_status"] == "scored"
    prompt = model.prompts("wk-query-prediction")[1]
    blocks = document_blocks(prompt)
    assert "estimated_child_score: 0.4 -> 0.8" in blocks["evidence_1"]
    assert "estimated_child_score:" not in blocks["evidence_2"]
    assert knowledge_input(prompt) == "INITIAL_GAP"
    assert [entry["estimated_child_score"] for entry in trace["evidence"]] == [0.8, 0.4 + 2 / 100]


@pytest.mark.parametrize("existing_pool", [False, True])
@pytest.mark.parametrize("empty_response", [False, True])
def test_no_valid_predictions_preserve_prior_pool_and_knowledge(
    monkeypatch, tmp_path, existing_pool, empty_response
):
    rounds = 2 if existing_pool else 1
    layer, model = layer_for(monkeypatch, tmp_path, query_optimization_max_rounds=rounds)
    rows = (
        []
        if empty_response
        else [
            {"evidence_ref": "evidence_999", "estimated_child_score": 0.99},
            {"evidence_ref": "evidence_1", "estimated_child_score": True},
        ]
    )
    model.prediction_overrides[rounds] = json.dumps(
        {"document_predictions": rows, "knowledge_state_analysis": "UNSUPPORTED_ANALYSIS"}
    )
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    row = trace["rounds"][-1]
    expected_state = "KNOWN_DETAIL_1; MISSING_DETAIL_1" if existing_pool else "INITIAL_GAP"
    assert [doc.id for doc in docs] == (["provider_1"] if existing_pool else [])
    assert row["assessment_status"] == "no_valid_predictions"
    assert row["document_predictions"] == []
    assert row["unscored_document_refs"] == [f"evidence_{rounds}"]
    assert row["selection_status"] == ("retained_previous" if existing_pool else "unavailable")
    assert row["knowledge_state_status"] == "retained_previous"
    assert row["proposed_knowledge_state_analysis"] == "UNSUPPORTED_ANALYSIS"
    assert row["knowledge_state_analysis"] == trace["knowledge_state_analysis"] == expected_state
    assert "prediction_error" not in row
    assert trace["status"] == ("awaiting_outcome" if existing_pool else "no_evidence")
    layer.record_usage(iteration=20, score=0.45, result_program_id="child")
    assert bool(layer.search_store.records[-1].impacts) == existing_pool
    assert ("result_score" in trace) == existing_pool


@pytest.mark.parametrize(
    "overrides",
    [
        {"query_optimization_queries_per_round": 0},
        {"query_optimization_top_k": 3},
        {"summarize_documents": True},
        {"retrieval_policy": "prompt"},
    ],
)
def test_unsupported_modes_fail_before_model_calls(monkeypatch, tmp_path, overrides):
    model = Model()
    with pytest.raises((ValueError, TypeError)):
        layer_for(monkeypatch, tmp_path, model, **overrides)
    assert not model.events


@pytest.mark.parametrize("top_k", [3, 5])
def test_multi_query_rounds_score_all_new_documents_and_keep_global_top_k(
    monkeypatch, tmp_path, top_k
):
    layer, model = layer_for(
        monkeypatch, tmp_path, query_optimization_queries_per_round=3, search_result_top_k=top_k
    )
    model.results_per_search = 5
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    assert (
        model.events
        == ["wk-retrieval-gate", "wk-population-analysis"]
        + (["wk-query"] * 3 + ["search"] * 3 + ["wk-query-prediction"]) * 3
    )
    assert model.queries == model.searches == 9 and model.predictions == 3
    assert trace["search_budget"] == trace["search_attempts"] == 9
    assert trace["max_rounds"] == trace["queries_per_round"] == 3
    assert trace["stop_reason"] == "search_budget_exhausted"
    assert [row["candidate_count"] for row in trace["rounds"]] == [15, 15 + top_k, 15 + top_k]
    assert [len(row["document_predictions"]) for row in trace["rounds"]] == [15, 15, 15]
    assert [row["pool_size"] for row in trace["rounds"]] == [top_k] * 3
    queries = model.prompts("wk-query")
    for index in range(3):
        assert queries[index * 3] == queries[index * 3 + 1] == queries[index * 3 + 2]
    assert knowledge_input(queries[3]) == "KNOWN_DETAIL_1; MISSING_DETAIL_1"
    assert knowledge_input(queries[6]) == "KNOWN_DETAIL_2; MISSING_DETAIL_2"
    assert len(set(model.query_labels)) == 9
    assert [doc.id for doc in docs] == [f"provider_{i}" for i in range(95, 95 - top_k, -1)]
    record = layer.search_store.records[-1]
    assert record.source_queries == [f"QUERY_{i}" for i in range(1, 10)]
    assert all(doc.query == "QUERY_9" for doc in docs)
    assert len(record.search_results) == top_k and not record.impacts
    assert len(layer.search_store.records) == 1
    assert "source_query_ref: q9" in layer._selected_search_context()
    layer.record_usage(iteration=20, score=0.48, result_program_id="actual_child")
    assert record.impacts[0].result_score == 0.48
    assert trace["result_score"] == 0.48


def test_stagnation_preset_runs_v7_search_only_after_heuristic_trigger(monkeypatch, tmp_path):
    layer, model = layer_for(
        monkeypatch, tmp_path, retrieval_gating_backend_type="stagnation", search_result_top_k=3
    )
    layer.gating.switch_interval = 1
    assert run(layer) == []  # first population observation establishes the baseline
    assert not model.events
    parent = parent_program()
    docs = asyncio.run(
        layer.run(parent=parent, population=[parent], history="NATIVE_HISTORY", iteration=21)
    )
    assert (
        model.events
        == ["wk-population-analysis", "wk-knowledge-state-analysis"]
        + ["wk-query", "search", "wk-query-prediction"] * 3
    )
    assert model.searches == 3 and len(docs) == 3
    assert knowledge_input(model.prompts("wk-query")[0]) == "HEURISTIC_INITIAL_KNOWLEDGE"
    assert (
        layer.query_optimization_history[-1]["initial_knowledge_state_analysis"]
        == "HEURISTIC_INITIAL_KNOWLEDGE"
    )


def test_multi_query_partial_failures_preserve_successes_and_provenance(monkeypatch, tmp_path):
    layer, model = layer_for(
        monkeypatch,
        tmp_path,
        query_optimization_queries_per_round=3,
        query_optimization_max_rounds=1,
    )
    model.fail_queries = {1}
    model.fail_searches = {1}
    docs = run(layer)
    trace = layer.query_optimization_history[-1]
    samples = trace["rounds"][0]["query_samples"]
    assert "query_error" in samples[0] and not samples[0]["search_attempted"]
    assert samples[1]["search_error"] == "search unavailable"
    assert samples[2]["documents"][0]["query"] == "QUERY_3"
    assert model.queries == 3 and model.searches == 2 and model.predictions == 1
    assert len(docs) == 1 and docs[0].query == "QUERY_3"
    assert layer.search_store.records[-1].source_queries == ["QUERY_2", "QUERY_3"]
    assert trace["stop_reason"] == "round_limit_reached"


def test_multi_query_duplicates_are_searched_once_per_round(monkeypatch, tmp_path):
    layer, model = layer_for(
        monkeypatch,
        tmp_path,
        query_optimization_queries_per_round=3,
        query_optimization_max_rounds=1,
    )
    layer.query_construction._generate = AsyncMock(
        side_effect=['{"query":"same query"}', '{"query":"same query"}', '{"query":"other query"}']
    )
    model.retrieve.side_effect = None
    model.retrieve.return_value = [document(1)]
    docs = run(layer)
    row = layer.query_optimization_history[-1]["rounds"][0]
    assert layer.query_construction._generate.await_count == 3
    assert model.retrieve.await_count == 2 and model.predictions == 1
    assert row["query_samples"][1]["duplicate_of"] == 1
    assert not row["query_samples"][1]["search_attempted"]
    assert row["candidate_count"] == len(row["document_predictions"]) == 1
    assert len(row["documents"]) == 2  # preserve both retrieval observations
    assert len(docs) == 1 and docs[0].query == "same query"
    assert layer.search_store.records[-1].source_queries == ["same query", "other query"]


@pytest.mark.parametrize("failure", ["query", "search", "prediction"])
def test_multi_query_failures_do_not_exceed_the_budget(monkeypatch, tmp_path, failure):
    layer, model = layer_for(monkeypatch, tmp_path, query_optimization_queries_per_round=3)
    attribute = {
        "query": "fail_queries",
        "search": "fail_searches",
        "prediction": "fail_predictions",
    }[failure]
    setattr(model, attribute, set(range(1, 10)))
    assert run(layer) == []
    trace = layer.query_optimization_history[-1]
    assert model.queries == 9
    assert trace["search_attempts"] == (0 if failure == "query" else 9)
    assert len(trace["rounds"]) == 3 and trace["status"] == "no_evidence"


def test_multi_query_calls_overlap_and_completion_order_does_not_change_selection(
    monkeypatch, tmp_path
):
    class ConcurrentModel(Model):
        async def generate(self, system, messages, *, llm_context, **kwargs):
            response = await super().generate(system, messages, llm_context=llm_context, **kwargs)
            if llm_context["phase"].startswith("wk-query-sample-"):
                if self.queries == 3:
                    self.queries_ready.set()
                await asyncio.wait_for(self.queries_ready.wait(), timeout=2)
            return response

        async def search(self, query, **kwargs):
            docs = await super().search(query, **kwargs)
            index = self.searches
            if index < 3:
                await asyncio.wait_for(self.completed[index].wait(), timeout=2)
            self.completion_order.append(index)
            self.completed[index - 1].set()
            return docs

    model = ConcurrentModel()
    model.queries_ready = asyncio.Event()
    model.completed = [asyncio.Event() for _ in range(3)]
    model.completion_order = []
    layer, _ = layer_for(
        monkeypatch,
        tmp_path,
        model,
        query_optimization_queries_per_round=3,
        query_optimization_max_rounds=1,
    )
    docs = run(layer)
    assert model.completion_order == [3, 2, 1]
    assert [doc.query for doc in docs] == ["QUERY_3", "QUERY_2"]
    row = layer.query_optimization_history[-1]["rounds"][0]
    assert [doc["query"] for doc in row["documents"]] == ["QUERY_1", "QUERY_2", "QUERY_3"]
    assert row["selected_evidence_refs"] == ["evidence_3", "evidence_2"]


def test_cancelling_multi_query_iteration_drains_all_samples(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path, query_optimization_queries_per_round=3)

    async def exercise():
        started, stopped = [], []
        ready, blocked = asyncio.Event(), asyncio.Event()

        async def wait_for_cancellation(system, prompt, *, label, iteration):
            started.append(label)
            if len(started) == 3:
                ready.set()
            try:
                await blocked.wait()
            finally:
                stopped.append(label)

        layer.query_construction._generate = wait_for_cancellation
        parent = parent_program()
        task = asyncio.create_task(
            layer.run(parent=parent, population=[parent], history="NATIVE_HISTORY", iteration=20)
        )
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert len(stopped) == 3 and set(started) == set(stopped)

    asyncio.run(exercise())
    model.retrieve.assert_not_awaited()
    assert layer.query_optimization_history[-1]["status"] == "failed"


def test_invalid_parent_score_does_not_search(monkeypatch, tmp_path):
    layer, model = layer_for(monkeypatch, tmp_path)
    parent = parent_program()
    parent.metrics = {}
    with pytest.raises(ValueError, match="finite actual parent"):
        run(layer, parent)
    model.retrieve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("diff", [False, True])
@pytest.mark.parametrize("action,count", [("no-op", 1), ("retrieve", 8), ("look-up", 8)])
async def test_outer_controller_selects_one_and_credits_final_evidence(
    monkeypatch, tmp_path, diff, action, count
):
    from skydiscover.evoduet.selective_generation import selective_generation_scope

    path = Path(__file__).resolve().parents[1] / "search/test_generation_batch.py"
    spec = importlib.util.spec_from_file_location("evoduet_batch_fixture", path)
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    controller = batch._controller(tmp_path, n=8)
    controller.config.diff_based_generation = diff
    layer, model = layer_for(monkeypatch, tmp_path, Model(action), n=8)
    layer.search_store.add(ConstructedQuery(query="old query"), [document(99)], iteration=1)
    controller.evoduet = layer

    async def knowledge(parent, history, iteration):
        docs = await layer.run(
            parent=parent,
            population=list(controller.database.programs.values()),
            history=history,
            iteration=iteration,
        )
        # Inner completion must precede every solution/evaluator call.
        assert not controller.test_state.requests
        controller.evaluator.evaluate_program.assert_not_awaited()
        assert all(not record.impacts for record in layer.search_store.records)
        return format_documents(docs)

    async def generate(index, request):
        if diff:
            return f"<<<<<<< SEARCH\nvalue = -10\n=======\nvalue = {index}\n>>>>>>> REPLACE"
        return f"```python\nvalue = {index}\n```"

    controller._maybe_run_evoduet = AsyncMock(side_effect=knowledge)
    controller.test_state.generate = generate
    with selective_generation_scope():
        result = await controller._run_iteration(20)
        controller._process_iteration_result(result, 20, verbose=False)
    assert result.error is None
    assert len(controller.test_state.requests) == count
    assert controller.evaluator.evaluate_program.await_count == count
    assert result.child_program_dict["solution"] == f"value = {count - 1}"
    controller.database.add.assert_called_once()
    metadata = result.child_program_dict["metadata"]["selective_generation"]
    assert metadata == {
        "gate_decision": action,
        "configured_num_generations": 8,
        "num_generations": count,
    }
    requests = controller.test_state.requests
    assert all(request["messages"] == requests[0]["messages"] for request in requests)
    assert model.searches == (3 if action == "retrieve" else 0)
    if action != "no-op":
        record = layer.search_store.records[-1]
        assert len(record.impacts) == 1
        assert record.impacts[0].target_score == -10
        assert record.impacts[0].result_score == 7
        trace = layer.query_optimization_history[-1]
        assert trace["generation_condition"]["num_generations"] == 8
        assert trace["result_score"] == 7 and trace["actual_improvement"] == 17
        if action == "retrieve":
            assert "num_generations" not in model.prompts("wk-query-prediction")[-1]
            assert "RAW_3" in requests[0]["messages"][0]["content"]
            assert len(trace["rounds"]) == 3
            assert all("result_score" not in row for row in trace["rounds"])
        else:
            assert "RAW_99" in requests[0]["messages"][0]["content"]
    else:
        assert not layer.query_optimization_history
