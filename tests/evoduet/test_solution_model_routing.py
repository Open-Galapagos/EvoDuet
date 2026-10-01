"""All retrieval stages use the pinned solution model across concurrent iterations."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skydiscover.config import EvoDuetConfig, LLMModelConfig
from skydiscover.evoduet.generation import active_solution_model
from skydiscover.evoduet.layer import EvoDuet, format_documents
from skydiscover.evoduet.retrieval import SearchResult
from skydiscover.llm.base import LLMResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("query_budget", [1024, 32768])
async def test_all_stages_use_pinned_solution_model_in_concurrent_iterations(
    monkeypatch, query_budget
):
    calls, prediction_rounds, active_queries, peak_queries = [], {}, {}, {}

    def init_client(model):
        async def generate(system, messages, *, llm_context, n=1, max_tokens=None):
            iteration, phase = llm_context["iteration"], llm_context["phase"]
            calls.append((iteration, phase, model, system, messages))
            is_query = phase.startswith("wk-query-sample-")
            assert max_tokens == (query_budget if is_query else None)
            assert n == 1
            assert bool(system) == (phase == "wk-population-analysis")
            assert len(messages) == 1 and messages[0]["role"] == "user"
            if is_query:
                active_queries[iteration] = active_queries.get(iteration, 0) + 1
                peak_queries[iteration] = max(
                    peak_queries.get(iteration, 0), active_queries[iteration]
                )
            await asyncio.sleep(0)
            if phase == "wk-retrieval-gate":
                response = json.dumps(
                    {"decision": "retrieve", "knowledge_state_analysis": "Remaining gap."}
                )
            elif phase == "wk-population-analysis":
                response = "Known constraints; unresolved implementation gap."
            elif is_query:
                active_queries[iteration] -= 1
                response = json.dumps({"query": f"q-{iteration}-{phase}"})
            elif phase == "wk-query-prediction":
                number = prediction_rounds.get(iteration, 0)
                prediction_rounds[iteration] = number + 1
                response = json.dumps(
                    {
                        "knowledge_state_analysis": "New evidence addresses the gap.",
                        "document_predictions": [
                            {
                                "evidence_ref": f"evidence_{index}",
                                "estimated_child_score": 0.6 + index * 0.01,
                            }
                            for index in (number * 2 + 1, number * 2 + 2)
                        ],
                    }
                )
            else:
                raise AssertionError(f"Unexpected phase {phase}")
            return LLMResponse(text=response)

        return SimpleNamespace(generate=generate)

    models = [
        LLMModelConfig(
            name=f"solution-{index}",
            api_base=f"https://solution-{index}.test/v1",
            api_key=f"test-key-{index}",
            max_tokens=8192 * (index + 1),
            temperature=0.2 + index * 0.3,
            reasoning_effort="high",
            tools=["tavily"],
            tool_choice="auto",
            max_tool_rounds=4,
            init_client=init_client,
        )
        for index in range(2)
    ]
    config = SimpleNamespace(
        llm=SimpleNamespace(models=models),
        num_generations=1,
        evoduet=EvoDuetConfig(
            enabled=True,
            query_optimization_max_rounds=2,
            query_optimization_queries_per_round=2,
            search_result_top_k=2,
            query_construction_max_tokens=query_budget,
            retrieval_gating_backend_type="llm",
        ),
    )

    async def retrieve(query, **kwargs):
        return [
            SearchResult(
                id=query.text,
                title=query.text,
                url=f"https://evidence.test/{query.text}",
                content="Provider snippet",
                raw_content=f"Evidence for {query.text}",
            )
        ]

    monkeypatch.setattr(
        EvoDuet, "_select_retrieval_backend", lambda self: SimpleNamespace(retrieve=retrieve)
    )
    layer = EvoDuet(config)
    parent = SimpleNamespace(
        id="parent", solution="def solve(): return 1", metrics={"combined_score": 0.5}
    )

    async def run(iteration, model):
        documents = await layer.run(
            solution_model=model,
            parent=parent,
            population=[parent],
            history="Earlier solutions",
            iteration=iteration,
            task_context="THIS MUST NOT BECOME A SYSTEM MESSAGE",
        )
        assert len(documents) == 2
        assert all(f"q-{iteration}-" in document.content for document in documents)
        assert all(document.content in format_documents(documents) for document in documents)
        assert active_solution_model.get() is None

    await asyncio.gather(run(10, models[0]), run(11, models[1]))
    expected_phases = {
        "wk-population-analysis",
        "wk-retrieval-gate",
        "wk-query-prediction",
        "wk-query-sample-r1-1",
        "wk-query-sample-r1-2",
        "wk-query-sample-r2-1",
        "wk-query-sample-r2-2",
    }
    for iteration, expected in ((10, models[0]), (11, models[1])):
        records = [row for row in calls if row[0] == iteration]
        assert {row[1] for row in records} == expected_phases
        for _, _, actual, _, _ in records:
            for field in (
                "name",
                "api_base",
                "api_key",
                "max_tokens",
                "temperature",
                "reasoning_effort",
            ):
                assert getattr(actual, field) == getattr(expected, field)
            assert actual.tools == [] and actual.tool_choice is None and actual.max_tool_rounds == 0
        prompts = [row[4][0]["content"] for row in records if row[1].startswith("wk-query-sample-")]
        assert len(prompts) == 4 and peak_queries[iteration] == 2
        assert prompts[0] == prompts[1] and prompts[2] == prompts[3]
        assert "[[Observed Document:" not in prompts[0]
        assert "[[Observed Document:" in prompts[2]
        assert f"q-{21 - iteration}-" not in prompts[2]
    assert all(model.tools == ["tavily"] and model.max_tool_rounds == 4 for model in models)


@pytest.mark.asyncio
async def test_solution_model_scope_resets_after_failure():
    layer = EvoDuet.__new__(EvoDuet)
    layer._run = AsyncMock(side_effect=RuntimeError("failed call"))
    with pytest.raises(RuntimeError, match="failed call"):
        await layer.run(solution_model=object())
    assert active_solution_model.get() is None
