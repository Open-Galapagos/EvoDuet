"""Offline v7 integration with the registered SimpleTES search scaffolds."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skydiscover.config import (
    AdaEvolveDatabaseConfig,
    BeamSearchDatabaseConfig,
    BestOfNDatabaseConfig,
    Config,
    DatabaseConfig,
    EvoDuetConfig,
    LLMConfig,
    LLMModelConfig,
    SearchConfig,
    SearchSelectionConfig,
)
from skydiscover.evaluation import EvaluationResult
from skydiscover.evoduet.layer import EvoDuet
from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.llm.base import LLMResponse
from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.registry import create_database, get_program
from skydiscover.search.route import get_discovery_controller
from tests.evoduet.test_layer import Model, document

DATABASE_CONFIGS = {
    "best_of_n": BestOfNDatabaseConfig,
    "topk": DatabaseConfig,
    "adaevolve": AdaEvolveDatabaseConfig,
    "beam_search": BeamSearchDatabaseConfig,
}


def _controller(monkeypatch, tmp_path, scaffold, *, decision="retrieve", diff=False):
    model = Model(decision)
    state = SimpleNamespace(generations=[], evaluations=[], events=model.events)
    config = Config(language="python", diff_based_generation=diff)
    config.context_builder.system_message = "Improve the synthetic value program."
    config.checkpoint_interval = 1
    config.max_parallel_iterations = 1
    config.llm = LLMConfig(models=[LLMModelConfig(name="offline-solution")])
    config.search = SearchConfig(
        type=scaffold,
        database=DATABASE_CONFIGS[scaffold](db_path=None),
    )
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

    class Backend:
        def __init__(self, cfg):
            self.model = cfg.name

        async def generate(self, system, messages, **kwargs):
            phase = (kwargs.get("llm_context") or {}).get("phase", "")
            if phase.startswith("wk-"):
                response = await model.generate(system, messages, **kwargs)
                return LLMResponse(text=response.text)
            if phase != "generation":
                return LLMResponse(text="Offline guide summary.")
            state.events.append("solution-generation")
            state.generations.append(copy.deepcopy(messages))
            if diff:
                return LLMResponse(
                    text="<<<<<<< SEARCH\nvalue = 0\n=======\nvalue = 1\n>>>>>>> REPLACE"
                )
            return LLMResponse(text="```python\nvalue = 1\n```")

    def evaluator_factory(evaluator_config, **kwargs):
        async def evaluate(solution, program_id):
            state.events.append("solution-evaluation")
            state.evaluations.append(solution)
            return EvaluationResult(
                metrics={"combined_score": 0.6},
                artifacts={"feedback": "offline evaluator outcome"},
            )

        return SimpleNamespace(
            evaluate_program=AsyncMock(side_effect=evaluate), llm_judge=None, close=lambda: None
        )

    def forbid_network(*args, **kwargs):
        raise AssertionError("No external calls in scaffold/v7 integration tests")

    monkeypatch.setattr("socket.socket.connect", forbid_network)
    monkeypatch.setattr("socket.socket.connect_ex", forbid_network)
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
    # Keep AdaEvolve's normal defaults, but never ask a live guide for paradigms.
    monkeypatch.setattr(
        "skydiscover.search.adaevolve.controller.ParadigmGenerator.generate",
        AsyncMock(return_value=[]),
    )
    database = create_database(scaffold, config.search.database)
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
    expected_class = AdaEvolveController if scaffold == "adaevolve" else DiscoveryController
    assert type(controller) is expected_class
    assert isinstance(controller.evoduet, EvoDuet)
    return controller, model, state


@pytest.mark.asyncio
@pytest.mark.parametrize("scaffold", DATABASE_CONFIGS)
@pytest.mark.parametrize("decision", ["retrieve", "look-up", "no-op"])
@pytest.mark.parametrize("diff", [False, True])
async def test_scaffold_v7_injects_evidence_before_generation_and_credits_child(
    monkeypatch, tmp_path, scaffold, decision, diff
):
    controller, model, state = _controller(
        monkeypatch, tmp_path, scaffold, decision=decision, diff=diff
    )
    layer = controller.evoduet
    if decision == "look-up":
        layer.search_store.add(ConstructedQuery(query="stored query"), [document(99)], iteration=0)
    try:
        best = await controller.run_discovery(start_iteration=1, max_iterations=1)
        assert best.solution == "value = 1"
        assert len(state.generations) == len(state.evaluations) == 1
        assert model.searches == (3 if decision == "retrieve" else 0)
        assert state.events.index("wk-retrieval-gate") < state.events.index("solution-generation")
        assert state.events.index("solution-generation") < state.events.index("solution-evaluation")
        prompt = state.generations[0][0]["content"]
        if decision == "no-op":
            assert "RAW_" not in prompt
            assert not layer.query_optimization_history
            assert not layer.search_store.records
            return
        assert ("RAW_3" if decision == "retrieve" else "RAW_99") in prompt
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


@pytest.mark.asyncio
@pytest.mark.parametrize("scaffold", DATABASE_CONFIGS)
async def test_scaffold_v7_checkpoint_restores_evidence_and_evaluated_outcome(
    monkeypatch, tmp_path, scaffold
):
    controller, _, state = _controller(monkeypatch, tmp_path / "first", scaffold)
    checkpoints = []

    def checkpoint(iteration):
        checkpoints.append((iteration, copy.deepcopy(controller.evoduet_checkpoint_state())))

    try:
        await controller.run_discovery(1, 1, checkpoint_callback=checkpoint)
        assert len(state.evaluations) == 1
        assert len(checkpoints) == 1
        iteration, saved = checkpoints[0]
        assert iteration == 1
        saved = json.loads(json.dumps(saved))
        assert saved["schema_version"] == 1
        assert len(saved["query_optimization_history"]) == 1
        resumed, resumed_model, resumed_state = _controller(
            monkeypatch, tmp_path / "resumed", scaffold, decision="look-up"
        )
        try:
            resumed.load_evoduet_checkpoint_state(saved)
            assert resumed.evoduet.state_dict() == saved
            first_record = resumed.evoduet.search_store.records[0]
            assert len(first_record.impacts) == 1
            assert first_record.impacts[0].result_score == 0.6
            # Exercise the restored search memory through a new controller step.
            await resumed.run_discovery(2, 1)
            assert len(resumed_state.generations) == len(resumed_state.evaluations) == 1
            assert resumed_model.searches == 0
            assert "RAW_3" in resumed_state.generations[0][0]["content"]
            # Lookup creates its own credited record and preserves source history.
            assert len(first_record.impacts) == 1
            assert len(resumed.evoduet.search_store.records) == 2
            lookup_record = resumed.evoduet.search_store.records[-1]
            assert lookup_record.operation == "look-up"
            assert lookup_record.lookup_provenance
            assert len(lookup_record.impacts) == 1
            assert lookup_record.impacts[0].iteration == 2
            assert lookup_record.impacts[0].result_score == 0.6
            assert len(resumed.evoduet.query_optimization_history) == 2
        finally:
            resumed.close()
    finally:
        controller.close()
