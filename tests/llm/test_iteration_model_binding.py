"""EvoDuet and solution generation share an iteration's sampled model."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.agentic_generator import AgenticGenerator
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.default_discovery_controller import DiscoveryController


def _pool():
    backends = []
    for name in ("model-a", "model-b"):

        async def generate(system, messages, *, name=name, **kwargs):
            samples = [LLMResponse(text=name) for _ in range(kwargs.get("n", 1))]
            return samples if kwargs.get("n", 1) > 1 else samples[0]

        backends.append(
            SimpleNamespace(
                model=name,
                client=object(),
                _use_responses_api=True,
                generate=AsyncMock(side_effect=generate),
            )
        )
    configs = [
        LLMModelConfig(name=backend.model, init_client=lambda cfg, backend=backend: backend)
        for backend in backends
    ]
    return LLMPool(configs)


def test_reserve_returns_config_of_sampled_model_and_reuses_it():
    pool = _pool()
    pool._sample_model = Mock(return_value=pool.models[1])

    assert pool.reserve_model(8) is pool.models_cfg[1]
    assert pool.reserve_model(8) is pool.models_cfg[1]
    assert pool.get_model_for_context({"iteration": 8}) is pool.models[1]
    pool._sample_model.assert_called_once()


@pytest.mark.asyncio
async def test_unreserved_generation_preserves_per_call_sampling():
    pool = _pool()
    pool._sample_model = Mock(side_effect=pool.models)

    first = await pool.generate("", [], llm_context={"iteration": 3})
    second = await pool.generate("", [], llm_context={"iteration": 3})

    assert first.generation_model is pool.models[0]
    assert second.generation_model is pool.models[1]
    assert pool._sample_model.call_count == 2


@pytest.mark.asyncio
async def test_reserved_model_survives_multiple_samples_and_retries():
    pool = _pool()
    pool._sample_model = Mock(return_value=pool.models[1])
    pool.reserve_model(4)
    pool._sample_model.side_effect = AssertionError("unexpected resampling")

    samples = await pool.generate("", [], n=3, llm_context={"iteration": 4, "attempt": 1})
    retry = await pool.generate("", [], llm_context={"iteration": 4, "attempt": 2})

    assert len(samples) == 3
    assert all(sample.generation_model is pool.models[1] for sample in samples)
    assert retry.generation_model is pool.models[1]


def test_binding_only_affects_matching_iteration_and_releases_explicitly():
    pool = _pool()
    pool._sample_model = Mock(side_effect=[pool.models[1], pool.models[0], pool.models[0]])
    pool.reserve_model(6)

    assert pool.get_model_for_context({"iteration": 7}) is pool.models[0]
    pool.release_model(7)
    assert pool.get_model_for_context({"iteration": 6}) is pool.models[1]
    pool.release_model(6)
    assert pool.get_model_for_context({"iteration": 6}) is pool.models[0]


def test_next_iteration_replaces_binding_instead_of_growing_a_map():
    pool = _pool()
    pool._sample_model = Mock(side_effect=[pool.models[1], pool.models[0], pool.models[1]])

    assert pool.reserve_model(10) is pool.models_cfg[1]
    assert pool.reserve_model(11) is pool.models_cfg[0]
    assert pool.get_model_for_context({"iteration": 11}) is pool.models[0]
    assert pool.get_model_for_context({"iteration": 10}) is pool.models[1]
    assert pool._sample_model.call_count == 3


@pytest.mark.asyncio
async def test_concurrent_iterations_keep_independent_choices_and_leave_caller_unbound():
    pool = _pool()
    pool._sample_model = Mock(side_effect=[pool.models[1], pool.models[0], pool.models[0]])
    ready = [asyncio.Event(), asyncio.Event()]

    async def iteration(index):
        config = pool.reserve_model(index)
        ready[index].set()
        await ready[1 - index].wait()
        first = await pool.generate("", [], llm_context={"iteration": index, "attempt": 1})
        await asyncio.sleep(0)
        second = await pool.generate("", [], llm_context={"iteration": index, "attempt": 2})
        assert first.generation_model is second.generation_model
        assert first.text == config.name
        return first.text

    assert await asyncio.gather(iteration(0), iteration(1)) == ["model-b", "model-a"]
    # A child task's binding never mutates its caller's asynchronous context.
    assert pool.reserve_model(0) is pool.models_cfg[0]
    assert pool._sample_model.call_count == 3


@pytest.mark.asyncio
async def test_child_tasks_inherit_iteration_choice():
    pool = _pool()
    pool._sample_model = Mock(return_value=pool.models[1])
    pool.reserve_model(9)
    result = await asyncio.create_task(pool.generate("", [], llm_context={"iteration": 9}))

    assert result.generation_model is pool.models[1]
    pool._sample_model.assert_called_once()


@pytest.mark.asyncio
async def test_agentic_tool_steps_use_the_reserved_solution_model():
    pool = _pool()
    pool._sample_model = Mock(return_value=pool.models[1])
    pool.reserve_model(12)
    generator = AgenticGenerator.__new__(AgenticGenerator)
    generator.llm_pool = pool
    generator._call_llm_responses = AsyncMock(return_value={"content": "child", "tool_calls": []})

    for step in range(2):
        await generator._call_llm("task", [], llm_context={"iteration": 12}, agent_step=step)

    assert generator._call_llm_responses.await_count == 2
    assert all(
        call.args[0] is pool.models[1] for call in generator._call_llm_responses.await_args_list
    )
    pool._sample_model.assert_called_once()


def _controller(pool, *, failing_retrieval=False):
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.llms = pool
    controller.agentic_generator = None
    controller.config = SimpleNamespace(
        evoduet=SimpleNamespace(
            max_document_chars=10_000,
        )
    )
    controller.evoduet = SimpleNamespace(
        run=AsyncMock(return_value=[]),
        discard_usage=Mock(),
    )
    if failing_retrieval:
        controller.evoduet.run.side_effect = RuntimeError("retrieval failed")
    controller._preflight_evoduet = Mock()
    controller._evoduet_population = Mock(return_value=[])
    controller._evoduet_task_context = Mock(return_value="task")
    controller._evoduet_program_score = Mock(return_value=0.5)
    return controller


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_retrieval", [False, True])
async def test_controller_passes_actual_model_to_wkl_and_reuses_it_for_solution(failing_retrieval):
    pool = _pool()
    pool._sample_model = Mock(return_value=pool.models[1])
    controller = _controller(pool, failing_retrieval=failing_retrieval)

    assert await controller._maybe_run_evoduet(SimpleNamespace(), "history", 15) == ""
    assert controller.evoduet.run.await_args.kwargs["solution_model"] is pool.models_cfg[1]
    first = await controller._call_llm("task", "mutate", llm_context={"iteration": 15})
    retry = await controller._call_llm("task", "retry", llm_context={"iteration": 15, "attempt": 2})

    assert first.generation_model is retry.generation_model is pool.models[1]
    pool._sample_model.assert_called_once()


@pytest.mark.asyncio
async def test_default_iteration_releases_choice_when_it_finishes():
    pool = _pool()
    pool._sample_model = Mock(side_effect=[pool.models[1], pool.models[0]])
    controller = _controller(pool)
    controller.database = SimpleNamespace(programs={})
    controller._run_from_scratch_iteration = AsyncMock(return_value="completed")
    pool.reserve_model(16)

    assert await controller._run_iteration(16) == "completed"
    assert pool.get_model_for_context({"iteration": 16}) is pool.models[0]


@pytest.mark.asyncio
async def test_disabled_evoduet_does_not_reserve_a_model():
    pool = _pool()
    pool._sample_model = Mock()
    controller = _controller(pool)
    controller.evoduet = None

    assert await controller._maybe_run_evoduet(SimpleNamespace(), "history", 17) == ""
    pool._sample_model.assert_not_called()
