"""Regression tests for evaluator failures in the discovery loop."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from skydiscover.evaluation import EvaluationResult
from skydiscover.llm.base import LLMResponse
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryController


def _controller(evaluation_results):
    parent = Program(
        id="parent",
        solution="seed",
        metrics={"combined_score": 0.5},
    )
    database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **_kwargs: (parent, []),
        add=Mock(),
        log_prompt=Mock(),
        _evolution_archive={},
        unattached_web_search_results={},
        unattached_llm_reasoning={},
    )
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = database
    controller.num_context_programs = 0
    controller.config = SimpleNamespace(
        language="python", max_solution_length=10_000, diff_based_generation=False
    )
    controller.feedback_reader = None
    controller.evoduet = None
    controller._maybe_run_evoduet = AsyncMock(return_value="")
    controller._build_prompt = Mock(return_value={"system": "system", "user": "prompt"})
    controller._call_llm = AsyncMock(return_value=LLMResponse(text="mutation"))
    controller._parse_llm_response = Mock(return_value=("child", "change", None))
    controller.evaluator = SimpleNamespace(
        evaluate_program=AsyncMock(side_effect=evaluation_results)
    )
    return controller


def test_negative_error_sentinel_is_retried_then_success_can_be_returned():
    controller = _controller(
        [
            EvaluationResult(
                metrics={
                    "combined_score": -100.0,
                    "error": "Kernel failed correctness check or did not compile",
                }
            ),
            EvaluationResult(metrics={"combined_score": 0.7}),
        ]
    )

    result = asyncio.run(controller._run_iteration(1, retry_times=2))

    assert result.error is None
    assert result.attempts_used == 2
    assert result.child_program_dict["metrics"] == {"combined_score": 0.7}
    assert controller.evaluator.evaluate_program.await_count == 2
    second_prompt_errors = controller._build_prompt.call_args_list[1].kwargs["failed_attempts"]
    assert second_prompt_errors[0]["metrics"]["combined_score"] == -100.0


def test_negative_error_sentinel_is_archived_without_entering_the_population():
    controller = _controller(
        [
            EvaluationResult(
                metrics={
                    "combined_score": -100.0,
                    "error": "Kernel failed correctness check or did not compile",
                }
            )
        ]
    )

    result = asyncio.run(controller._run_iteration(1, retry_times=1))
    controller._process_iteration_result(result, 1, verbose=False)

    assert result.error == (
        "Evaluator failed after 1 attempts: " "Kernel failed correctness check or did not compile"
    )
    controller.database.add.assert_not_called()
    child_id = result.child_program_dict["id"]
    archived = controller.database._evolution_archive[child_id]
    assert archived.solution == "child"
    assert archived.metrics == result.child_program_dict["metrics"]
    assert archived.metadata["evaluation_status"] == "failed"
    assert archived.metadata["error"] == result.error
    assert list(controller.database.programs) == ["parent"]
    controller.database.log_prompt.assert_called_once_with(
        template_key="full_rewrite_user_message",
        program_id=child_id,
        prompt=result.prompt,
        responses=[result.llm_response],
    )
