"""Exercise EvoX's real control flow with v7 and offline model/evaluator doubles."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skydiscover.config import (
    Config,
    EvoDuetConfig,
    EvoxDatabaseConfig,
    LLMConfig,
    LLMModelConfig,
    SearchConfig,
    SearchSelectionConfig,
)
from skydiscover.evaluation import EvaluationResult
from skydiscover.evoduet.layer import EvoDuet
from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.llm.base import LLMResponse
from skydiscover.search.default_discovery_controller import DiscoveryControllerInput
from skydiscover.search.evox.controller import CoEvolutionController
from skydiscover.search.registry import create_database, get_program
from skydiscover.search.route import get_discovery_controller
from tests.evoduet.test_layer import Model, document

ROOT = Path(__file__).resolve().parents[2]


def _controller(monkeypatch, tmp_path, *, decision="retrieve", diff=False):
    model = Model(decision)
    state = SimpleNamespace(
        generations=[], evaluations=[], meta_generations=[], events=model.events
    )
    config = Config(language="python", diff_based_generation=diff)
    config.context_builder.system_message = "Improve the synthetic value program."
    config.checkpoint_interval = 1
    config.max_parallel_iterations = 1
    config.llm = LLMConfig(models=[LLMModelConfig(name="offline-solution")])
    config.search = SearchConfig(
        type="evox",
        database=EvoxDatabaseConfig(db_path=None, auto_generate_variation_operators=False),
        output_dir=str(tmp_path / "search"),
        switch_interval=1,
        share_llm=False,
    )
    config.search.database.config_path = str(ROOT / "tests/evoduet/fixtures/evox.yaml")
    config.evoduet = EvoDuetConfig(
        enabled=True,
        retrieval_gating_backend_type="llm",
        retrieval_gating_prompt_template_name="retrieval_gating",
        population_state_prompt_template_name="population_analysis",
        query_optimization_max_rounds=3,
        query_optimization_queries_per_round=1,
        search_result_top_k=3,
        search_selection=SearchSelectionConfig(policy="recency", num=10),
    )
    config.evoduet.tavily_retrieval.max_results = 5
    seed_strategy = Path(config.search.database.database_file_path).read_text()
    evolved_strategy = seed_strategy.replace(
        "    def sample(", "    integration_marker = 'evox-v7'\n\n    def sample(", 1
    )

    class Backend:
        def __init__(self, cfg):
            self.model = cfg.name

        async def generate(self, system, messages, **kwargs):
            phase = (kwargs.get("llm_context") or {}).get("phase", "")
            if phase.startswith("wk-"):
                assert self.model == "offline-solution"
                response = await model.generate(system, messages, **kwargs)
                return LLMResponse(text=response.text)
            if phase != "generation":
                # EvoX availability checks and guide summaries use this pool too.
                return LLMResponse(text="Offline guide summary.")
            if self.model == "deepseek-v4-flash":
                state.meta_generations.append(copy.deepcopy(messages))
                return LLMResponse(text=f"```python\n{evolved_strategy}\n```")
            state.events.append("solution-generation")
            state.generations.append(copy.deepcopy(messages))
            if diff:
                return LLMResponse(
                    text="<<<<<<< SEARCH\nvalue = 0\n=======\nvalue = 1\n>>>>>>> REPLACE"
                )
            return LLMResponse(text="```python\nvalue = 1\n```")

    def evaluator_factory(evaluator_config, **kwargs):
        is_meta = evaluator_config.evaluation_file.endswith("search_strategy_evaluator.py")

        async def evaluate(solution, program_id):
            if not is_meta:
                state.events.append("solution-evaluation")
                state.evaluations.append(solution)
            return EvaluationResult(
                metrics={"combined_score": 1.0 if is_meta else 0.6},
                artifacts={"feedback": "offline evaluator outcome"},
            )

        return SimpleNamespace(
            evaluate_program=AsyncMock(side_effect=evaluate), llm_judge=None, close=lambda: None
        )

    def forbid_network(*args, **kwargs):
        raise AssertionError("No external calls in EvoX/v7 integration tests")

    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "https://offline-meta.test")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "offline-key")
    monkeypatch.setattr("socket.socket.connect", forbid_network)
    monkeypatch.setattr("skydiscover.llm.llm_pool.OpenAILLM", Backend)
    monkeypatch.setattr(
        "skydiscover.search.default_discovery_controller.create_evaluator", evaluator_factory
    )
    monkeypatch.setattr("skydiscover.evoduet.EvoDuet", EvoDuet)
    monkeypatch.setattr(
        EvoDuet,
        "_select_retrieval_backend",
        lambda self: SimpleNamespace(retrieve=model.retrieve),
    )
    database = create_database("evox", config.search.database)
    database.add(get_program(config, "value = 0", "parent", {"combined_score": 0.4}, 0), 0)
    database.initial_program_id = "parent"
    database.initial_program_score = 0.4
    controller = get_discovery_controller(
        DiscoveryControllerInput(
            config=config,
            evaluation_file=str(tmp_path / "synthetic_evaluator.py"),
            database=database,
            output_dir=str(tmp_path),
        )
    )
    assert isinstance(controller, CoEvolutionController)
    assert isinstance(controller.evoduet, EvoDuet)
    assert controller.search_controller.evoduet is None
    return controller, model, state


@pytest.mark.asyncio
@pytest.mark.parametrize("diff", [False, True])
@pytest.mark.parametrize("decision", ["retrieve", "look-up", "no-op"])
async def test_evox_v7_runs_search_before_generation_and_credits_actual_child(
    monkeypatch, tmp_path, diff, decision
):
    controller, model, state = _controller(monkeypatch, tmp_path, decision=decision, diff=diff)
    layer = controller.evoduet
    if decision == "look-up":
        layer.search_store.add(ConstructedQuery(query="stored query"), [document(99)], iteration=0)
    try:
        best = await controller.run_discovery(start_iteration=1, max_iterations=1)
        assert best.solution == "value = 1"
        assert len(state.generations) == len(state.evaluations) == 1
        assert not state.meta_generations
        assert model.searches == (3 if decision == "retrieve" else 0)
        prompt = state.generations[0][0]["content"]
        if decision == "no-op":
            assert "RAW_" not in prompt
            assert not layer.query_optimization_history
            return
        assert ("RAW_3" if decision == "retrieve" else "RAW_99") in prompt
        assert state.events.index("solution-generation") < state.events.index("solution-evaluation")
        if decision == "retrieve":
            assert max(i for i, event in enumerate(state.events) if event == "search") < (
                state.events.index("solution-generation")
            )
        record = layer.search_store.records[-1]
        assert len(record.impacts) == 1
        assert record.impacts[0].target_score == 0.4
        assert record.impacts[0].result_score == 0.6
        assert record.impacts[0].result_program_id == best.id
        trace = layer.query_optimization_history[-1]
        assert trace["result_score"] == 0.6
        assert trace["actual_improvement"] == pytest.approx(0.2)
        assert trace["generation_condition"]["num_generations"] == 1
    finally:
        controller.close()
        controller.search_controller.close()


@pytest.mark.asyncio
async def test_evox_strategy_switch_and_checkpoint_preserve_v7_search_memory(monkeypatch, tmp_path):
    controller, model, state = _controller(monkeypatch, tmp_path / "first")
    checkpoints = []

    def checkpoint(iteration):
        checkpoints.append((iteration, copy.deepcopy(controller.evoduet_checkpoint_state())))

    try:
        await controller.run_discovery(1, 3, checkpoint_callback=checkpoint)
        assert len(state.generations) == len(state.evaluations) == 3
        assert len(state.meta_generations) == 1
        assert model.searches == 9
        assert controller.database.integration_marker == "evox-v7"
        assert len(controller.evoduet.query_optimization_history) == 3
        # Capture while the changed strategy still awaits its delayed window score.
        iteration, saved = next(
            row for row in checkpoints if row[1]["evox"]["pending_search_result"]
        )
        assert iteration == 2
        saved = json.loads(json.dumps(saved))
        assert saved["format"] == "evox-v1"
        assert saved["search"] is None
        assert saved["solution"]["schema_version"] == 1
        assert len(saved["solution"]["query_optimization_history"]) == 2
        resumed, resumed_model, resumed_state = _controller(monkeypatch, tmp_path / "resumed")
        try:
            resumed.load_evoduet_checkpoint_state(saved)
            assert resumed.database.integration_marker == "evox-v7"
            assert resumed._pending_search_result is not None
            assert len(resumed.evoduet.search_store.records) == 2
            assert resumed.evoduet.state_dict() == saved["solution"]
            await resumed.run_discovery(3, 1)
            assert len(resumed_state.generations) == len(resumed_state.evaluations) == 1
            assert resumed_model.searches == 3
            assert not resumed_state.meta_generations
            assert resumed._pending_search_result is None
            assert len(resumed.evoduet.query_optimization_history) == 3
            assert all(len(row.impacts) == 1 for row in resumed.evoduet.search_store.records)
            assert len(resumed.search_controller.database.programs) == 2
        finally:
            resumed.close()
            resumed.search_controller.close()
    finally:
        controller.close()
        controller.search_controller.close()
