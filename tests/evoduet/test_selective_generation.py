"""Exercise gate-dependent generation through the existing solution controller."""

import asyncio
import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skydiscover.config import Config, DatabaseConfig, LLMConfig, LLMModelConfig
from skydiscover.evoduet.retrieval_gating import GateDecision
from skydiscover.evoduet.selective_generation import (
    record_gate_action,
    selective_generation_scope,
)
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.topk.database import TopKDatabase


def _fixture_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TESTS = Path(__file__).resolve().parents[1]
_batch = _fixture_module("selective_batch_fixture", _TESTS / "search/test_generation_batch.py")


def _gate(controller, action="retrieve", evidence="EVIDENCE"):
    layer = SimpleNamespace(
        staged_search_record_id=Mock(return_value=None),
        record_usage=Mock(),
        discard_usage=Mock(),
    )
    controller.evoduet = layer

    async def run(parent, history, iteration):
        decision = action(iteration) if callable(action) else action
        record_gate_action(layer, GateDecision(decision), iteration)
        return evidence

    controller._maybe_run_evoduet = AsyncMock(side_effect=run)
    return layer


def _assert_metadata(result, action, count, configured=3):
    assert result.error is None
    metadata = result.child_program_dict["metadata"]
    assert metadata["selective_generation"] == {
        "gate_decision": action,
        "configured_num_generations": configured,
        "num_generations": count,
    }
    assert ("generation_batches" in metadata) is (count > 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retrieve", "look-up"])
async def test_empty_documents_do_not_change_the_gate_action(tmp_path, action):
    controller = _batch._controller(tmp_path)
    _gate(controller, action, evidence="")
    with selective_generation_scope():
        result = await controller._run_iteration(1)
    _assert_metadata(result, action, 3)
    assert len(controller.test_state.requests) == 3
    assert controller.evaluator.evaluate_program.await_count == 3


@pytest.mark.asyncio
async def test_sequential_parallel_sequential_reuses_runner_without_changing_config(tmp_path):
    controller = _batch._controller(tmp_path)
    original_config = controller.config
    before = original_config.to_dict()
    actions = {1: "no-op", 2: "retrieve", 3: "no-op", 4: "look-up"}
    _gate(controller, actions.__getitem__)
    with selective_generation_scope():
        first = await controller._run_iteration(1)
        assert not hasattr(controller, "_generation_batch_runner")
        second = await controller._run_iteration(2)
        runner = controller._generation_batch_runner
        third = await controller._run_iteration(3)
        fourth = await controller._run_iteration(4)
        assert controller._generation_batch_runner is runner
        assert runner.count == 3
        assert original_config.to_dict() == before
        assert controller.config.to_dict() == before
        assert controller.config.num_generations == 3
    for result, action, count in (
        (first, "no-op", 1),
        (second, "retrieve", 3),
        (third, "no-op", 1),
        (fourth, "look-up", 3),
    ):
        _assert_metadata(result, action, count)
    counts = Counter(
        request["llm_context"]["iteration"] for request in controller.test_state.requests
    )
    assert counts == {1: 1, 2: 3, 3: 1, 4: 3}


@pytest.mark.asyncio
@pytest.mark.parametrize("action,count", [("no-op", 1), ("retrieve", 3), ("look-up", 3)])
async def test_retries_keep_action_and_do_not_repeat_the_gate(tmp_path, action, count):
    controller = _batch._controller(tmp_path)
    controller.config.diff_based_generation = True
    _gate(controller, action)

    async def generate(index, request):
        old = "value = absent" if index < count else "value = -10"
        return f"<<<<<<< SEARCH\n{old}\n=======\nvalue = {index}\n>>>>>>> REPLACE"

    controller.test_state.generate = generate
    with selective_generation_scope():
        result = await controller._run_iteration(1, retry_times=2)
    _assert_metadata(result, action, count)
    assert len(controller.test_state.requests) == 2 * count
    assert controller.evaluator.evaluate_program.await_count == count
    assert result.attempts_used == 2
    controller._maybe_run_evoduet.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_iterations_keep_independent_gate_actions(tmp_path):
    controller = _batch._controller(tmp_path)
    layer = _gate(controller)
    parallel_decided = asyncio.Event()
    sequential_decided = asyncio.Event()

    async def knowledge(parent, history, iteration):
        if iteration == 1:
            record_gate_action(layer, GateDecision.RETRIEVE, iteration)
            parallel_decided.set()
            await sequential_decided.wait()
        else:
            await parallel_decided.wait()
            record_gate_action(layer, GateDecision.NO_OP, iteration)
            sequential_decided.set()
        return ""

    controller._maybe_run_evoduet = AsyncMock(side_effect=knowledge)
    with selective_generation_scope():
        parallel, sequential = await asyncio.wait_for(
            asyncio.gather(controller._run_iteration(1), controller._run_iteration(2)), 5
        )
    _assert_metadata(parallel, "retrieve", 3)
    _assert_metadata(sequential, "no-op", 1)
    counts = Counter(
        request["llm_context"]["iteration"] for request in controller.test_state.requests
    )
    assert counts == {1: 3, 2: 1}


@pytest.mark.asyncio
async def test_cancellation_resets_iteration_state_and_keeps_next_no_op_sequential(tmp_path):
    controller = _batch._controller(tmp_path)
    _gate(controller, lambda iteration: "retrieve" if iteration == 1 else "no-op")
    started = asyncio.Event()

    async def generate(index, request):
        started.set()
        await asyncio.Future()

    controller.test_state.generate = generate
    with selective_generation_scope():
        task = asyncio.create_task(controller._run_iteration(1))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert controller.config.num_generations == 3
        controller.test_state.generate = None
        result = await controller._run_iteration(2)
    _assert_metadata(result, "no-op", 1)
    assert (
        sum(request["llm_context"]["iteration"] == 2 for request in controller.test_state.requests)
        == 1
    )


@pytest.mark.asyncio
async def test_failure_before_solution_generation_cannot_leak_action(tmp_path):
    controller = _batch._controller(tmp_path)
    layer = _gate(controller)

    async def knowledge(parent, history, iteration):
        if iteration == 1:
            record_gate_action(layer, GateDecision.RETRIEVE, iteration)
            raise RuntimeError("pre-generation failure")
        record_gate_action(layer, GateDecision.NO_OP, iteration)
        return ""

    controller._maybe_run_evoduet = AsyncMock(side_effect=knowledge)
    with selective_generation_scope():
        failed = await controller._run_iteration(1)
        assert "pre-generation failure" in failed.error
        assert controller.config.num_generations == 3
        result = await controller._run_iteration(2)
    _assert_metadata(result, "no-op", 1)
    assert len(controller.test_state.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", [False, True])
async def test_from_scratch_without_gate_remains_single_generation(tmp_path, direct):
    controller = _batch._controller(tmp_path, parent=False)
    with selective_generation_scope():
        method = controller._run_from_scratch_iteration if direct else controller._run_iteration
        result = await method(1)
    _assert_metadata(result, None, 1)
    assert len(controller.test_state.requests) == 1
    assert controller.evaluator.evaluate_program.await_count == 1
    controller._maybe_run_evoduet.assert_not_awaited()


@pytest.mark.asyncio
async def test_one_generation_setting_remains_one_even_for_retrieve(tmp_path):
    controller = _batch._controller(tmp_path, n=1)
    _gate(controller)
    with selective_generation_scope():
        result = await controller._run_iteration(1)
    _assert_metadata(result, "retrieve", 1, configured=1)
    assert len(controller.test_state.requests) == 1


def test_scope_restores_standard_controller_methods_on_error():
    names = ("__init__", "_run_iteration", "_run_from_scratch_iteration")
    before = {name: getattr(DiscoveryController, name) for name in names}
    with pytest.raises(RuntimeError, match="scope failure"):
        with selective_generation_scope():
            raise RuntimeError("scope failure")
    assert {name: getattr(DiscoveryController, name) for name in names} == before


@pytest.mark.asyncio
async def test_real_constructor_preserves_config_and_keeps_one_admitted_child(tmp_path):
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "from pathlib import Path\n"
        "def evaluate(path):\n"
        "    return {'combined_score': float(Path(path).read_text().split('=')[1])}\n"
    )
    state = SimpleNamespace(requests=[], generate=None)
    backend = _batch._Backend("local-test", state)
    config = Config(num_generations=3, language="python", diff_based_generation=False)
    config.llm = LLMConfig(
        models=[LLMModelConfig(name="local-test", init_client=lambda cfg: backend)]
    )
    config.evaluator.cascade_evaluation = False
    database = TopKDatabase("topk", DatabaseConfig(db_path=None))
    database.add(Program(id="parent", solution="value = -10", metrics={"combined_score": -10}))
    with selective_generation_scope():
        controller = DiscoveryController(
            DiscoveryControllerInput(config, str(evaluator), database, output_dir=str(tmp_path))
        )
        try:
            _gate(controller, lambda iteration: "retrieve" if iteration == 1 else "no-op")
            for iteration, count in ((1, 3), (2, 1)):
                result = await controller._run_iteration(iteration)
                _assert_metadata(result, "retrieve" if iteration == 1 else "no-op", count)
                controller._process_iteration_result(result, iteration, verbose=False)
            assert config.num_generations == controller.config.num_generations == 3
            assert len(state.requests) == 4
            assert len(database.programs) == 3
        finally:
            controller.close()
