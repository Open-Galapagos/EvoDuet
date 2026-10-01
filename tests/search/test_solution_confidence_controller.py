"""Exercise confidence timing, retry retention and disabled-loop compatibility."""

import asyncio
import json
import multiprocessing
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skydiscover.config import Config, SolutionConfidenceConfig
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.evaluation import EvaluationResult
from skydiscover.llm.base import LLMResponse
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.evoduet import format_documents
from skydiscover.evoduet.retrieval.base_retrieval import SearchResult


def controller_fixture(results, *, enabled=True):
    parent = Program(id="parent", solution="parent code", metrics={"combined_score": 0.5})
    database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **kwargs: (parent, []),
        solution_confidence_records=[],
        unattached_web_search_results={},
        unattached_llm_reasoning={},
        write_evolution_trace=Mock(),
        log_status=Mock(),
    )
    settings = SolutionConfidenceConfig(enabled=enabled)
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = database
    controller.output_dir = "/tmp/confidence-test-output"
    controller.evaluation_file = "/benchmark/evaluator.py"
    controller.num_context_programs = 0
    controller.config = SimpleNamespace(
        language="python",
        max_solution_length=10000,
        solution_confidence=settings,
        checkpoint_interval=2,
        search=SimpleNamespace(database=SimpleNamespace(random_seed=42)),
    )
    controller.feedback_reader = None
    controller.evoduet = None
    controller._maybe_run_evoduet = AsyncMock(return_value="ORACLE evidence")

    def build_prompt(**kwargs):
        capture = kwargs.get("confidence_context")
        if capture is not None:
            capture.update(
                evolutionary_history="Earlier improvement: score 0.4",
                parent_program="parent code",
            )
        return {
            "system": "Original task",
            "user": "Parent code and history\n# Task\nImprove it",
        }

    controller._build_prompt = Mock(side_effect=build_prompt)
    model = SimpleNamespace(
        score_labels=AsyncMock(
            return_value={
                "status": "success",
                "confidence": 0.8,
                "model": "same-generator",
                "logprob_true": -0.2231435513142097,
                "logprob_false": -1.6094379124341003,
                "label_probability_mass": 1.0,
            }
        ),
        score_verbalized=AsyncMock(
            return_value={
                "status": "success",
                "confidence": 0.6,
                "model": "same-generator",
                "raw_text": "0.6",
            }
        ),
    )
    response = LLMResponse(text="mutation")
    response.generation_model = model
    response.generation_context = {
        "api": "chat_completions",
        "system_message": "Original task",
        "messages": [
            {"role": "user", "content": "Parent code and history ORACLE evidence"},
            {"role": "tool", "content": "Tool evidence before candidate"},
        ],
    }
    controller._call_llm = AsyncMock(return_value=response)
    controller._parse_llm_response = Mock(return_value=("candidate code", "change", None))
    controller.evaluator = SimpleNamespace(evaluate_program=AsyncMock(side_effect=results))
    return controller, model, response


def test_same_model_scores_before_evaluator_with_history_parent_document_and_candidate():
    controller, model, _ = controller_fixture([])
    events = []

    async def score(system, messages, **kwargs):
        events.append("assess")
        assert controller.evaluator.evaluate_program.await_count == 0
        content = messages[0]["content"]
        assert "Earlier improvement: score 0.4" in content
        assert "parent code" in content
        assert "[Web Document]\nORACLE evidence" in content
        assert "Tool evidence before candidate" not in content
        assert "candidate code" in content
        assert "0.987654321" not in content
        return {"status": "success", "confidence": 0.8}

    async def evaluate(candidate, program_id):
        events.append("evaluate")
        assert model.score_labels.await_count == 1
        assert model.score_verbalized.await_count == 1
        # Even a changed parent object cannot alter the comparison snapshot.
        controller.database.programs["parent"].metrics["combined_score"] = 100.0
        return EvaluationResult(metrics={"combined_score": 0.987654321})

    model.score_labels.side_effect = score
    controller.evaluator.evaluate_program.side_effect = evaluate
    result = asyncio.run(controller._run_iteration(7))
    assert result.error is None
    assert events == ["assess", "evaluate"]
    record = controller.database.solution_confidence_records[0]
    assert record["parent_score"] == 0.5
    assert (record["iteration"], record["attempt"], record["seed"]) == (7, 1, 42)
    assert record["y_t"] == 1
    assert record["brier_score"] == pytest.approx(0.04)
    assert "ORACLE evidence" in record["generation_prompt"]["user"]
    assert record["web_document"] == "ORACLE evidence"
    assert record["prompt_template"] == "solution_confidence_user_web.txt"
    assert record["prompt_template_verbalized"] == "solution_confidence_verbalized_user_web.txt"
    assert record["confidence_verbalized"] == 0.6
    assert record["brier_score_verbalized"] == pytest.approx(0.16)
    assert record["status"] == "completed"
    assert record["prompt_version"] == "parent-improvement-oracle-v1"
    assert "Tool evidence before candidate" in json.dumps(record["generation_context"])
    assert result.child_program_dict["metrics"] == {"combined_score": 0.987654321}
    assert result.child_program_dict["solution_confidence"] == record
    assert result.child_program_dict["solution_confidence"] is not record
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("slow_method", ["score_labels", "score_verbalized"])
def test_evaluator_waits_for_both_confidence_requests(slow_method):
    async def scenario():
        controller, model, _ = controller_fixture(
            [EvaluationResult(metrics={"combined_score": 0.7})]
        )
        slow_started = asyncio.Event()
        fast_completed = asyncio.Event()
        release_slow = asyncio.Event()

        async def slow(*args, **kwargs):
            slow_started.set()
            await release_slow.wait()
            return {"status": "success", "confidence": 0.8}

        async def fast(*args, **kwargs):
            fast_completed.set()
            return {"status": "success", "confidence": 0.6}

        fast_method = "score_verbalized" if slow_method == "score_labels" else "score_labels"
        getattr(model, slow_method).side_effect = slow
        getattr(model, fast_method).side_effect = fast
        iteration = asyncio.create_task(controller._run_iteration(1))
        try:
            await asyncio.wait_for(
                asyncio.gather(slow_started.wait(), fast_completed.wait()), timeout=1.0
            )
            controller.evaluator.evaluate_program.assert_not_awaited()
        finally:
            release_slow.set()
            result = await asyncio.wait_for(iteration, timeout=1.0)
        assert result.error is None
        controller.evaluator.evaluate_program.assert_awaited_once()
        assert controller.database.solution_confidence_records[0]["status"] == "completed"

    asyncio.run(scenario())


def test_confidence_forwards_provider_settings_before_evaluation():
    controller, model, _ = controller_fixture([EvaluationResult(metrics={"combined_score": 0.7})])
    controller.config.solution_confidence.top_logprobs = 5
    controller.config.solution_confidence.openrouter_provider = "alibaba"

    result = asyncio.run(controller._run_iteration(1))

    assert result.error is None
    assert model.score_labels.call_args.kwargs == {
        "timeout": controller.config.solution_confidence.timeout,
        "top_logprobs": 5,
        "openrouter_provider": "alibaba",
    }
    assert model.score_verbalized.call_args.kwargs == {
        "timeout": controller.config.solution_confidence.timeout,
        "openrouter_provider": "alibaba",
    }
    record = controller.database.solution_confidence_records[0]
    assert record["confidence"] == 0.8
    assert record["y_t"] == 1
    assert record["brier_score"] == pytest.approx(0.04)


def test_default_mode_measures_confidence_without_oracle_context():
    controller, model, response = controller_fixture(
        [EvaluationResult(metrics={"combined_score": 0.7})]
    )
    controller._maybe_run_evoduet = AsyncMock(return_value=None)
    response.generation_context = {
        "api": "chat_completions",
        "system_message": "Original task",
        "messages": [{"role": "user", "content": "Parent code and history\n# Task\nImprove it"}],
    }

    async def score(system, messages, **kwargs):
        assert controller.evaluator.evaluate_program.await_count == 0
        content = messages[0]["content"]
        assert "Earlier improvement: score 0.4" in content
        assert "parent code" in content
        assert "candidate code" in content
        assert "ORACLE evidence" not in content
        assert "Tool evidence before candidate" not in content
        return {"status": "success", "confidence": 0.8}

    model.score_labels.side_effect = score
    result = asyncio.run(controller._run_iteration(1))

    assert result.error is None
    model.score_labels.assert_awaited_once()
    record = controller.database.solution_confidence_records[0]
    assert record["generation_context"] == response.generation_context
    assert "ORACLE evidence" not in record["generation_prompt"]["user"]
    assert record["web_document"] == ""
    assert record["prompt_template"] == "solution_confidence_user.txt"
    assert "[Web Document]" not in record["assessment_prompt"]["user"]
    assert "[Web Document]" not in record["assessment_prompt_verbalized"]["user"]
    assert record["prompt_template_verbalized"] == "solution_confidence_verbalized_user.txt"
    assert record["confidence_verbalized"] == 0.6
    assert record["brier_score_verbalized"] == pytest.approx(0.16)
    assert record["confidence"] == 0.8
    assert record["y_t"] == 1
    assert record["brier_score"] == pytest.approx(0.04)
    assert result.child_program_dict["solution_confidence"] == record


@pytest.mark.parametrize("oracle", ["", "ORACLE evidence"])
def test_real_builder_captures_history_and_parent_once_before_generation(oracle):
    controller, model, response = controller_fixture([])
    controller.config = Config()
    controller.config.solution_confidence.enabled = True
    controller.context_builder = DefaultContextBuilder(controller.config)
    controller._prompt_context = {}
    controller._build_prompt = DiscoveryController._build_prompt.__get__(controller)
    controller._maybe_run_evoduet.return_value = oracle
    parent = controller.database.programs["parent"]
    previous = Program(
        id="previous",
        solution="previous code",
        metrics={"combined_score": 0.4},
        metadata={"changes": "historical change before generation"},
    )
    reference = Program(
        id="reference", solution="reference code before generation", metrics={"combined_score": 0.3}
    )
    controller.database.sample = lambda **kwargs: (parent, [reference])
    controller.database.get_statistics = Mock(return_value={"previous_programs": [previous]})
    for name in (
        "_format_previous_attempts",
        "_format_other_context_programs",
        "_format_current_program",
    ):
        setattr(
            controller.context_builder,
            name,
            Mock(wraps=getattr(controller.context_builder, name)),
        )

    async def generate(system, user, **kwargs):
        assert "historical change before generation" in user
        assert "reference code before generation" in user
        parent.solution = "parent source changed while generation awaited"
        previous.metadata["changes"] = "later historical change"
        reference.solution = "later reference source"
        return response

    async def score(system, messages, **kwargs):
        content = messages[0]["content"]
        assert controller.evaluator.evaluate_program.await_count == 0
        assert "historical change before generation" in content
        assert "reference code before generation" in content
        assert "parent code" in content
        assert "candidate code" in content
        assert "later historical change" not in content
        assert "later reference source" not in content
        assert "parent source changed" not in content
        assert ("[Web Document]\nORACLE evidence" in content) == bool(oracle)
        assert "Tool evidence before candidate" not in content
        assert "0.87654321" not in content
        return {"status": "success", "confidence": 0.8}

    controller._call_llm.side_effect = generate
    model.score_labels.side_effect = score
    controller.evaluator.evaluate_program.side_effect = [
        EvaluationResult(metrics={"combined_score": 0.87654321})
    ]
    result = asyncio.run(controller._run_iteration(7))

    assert result.error is None
    controller.database.get_statistics.assert_called_once_with()
    controller.context_builder._format_previous_attempts.assert_called_once()
    controller.context_builder._format_other_context_programs.assert_called_once()
    controller.context_builder._format_current_program.assert_called_once()
    record = controller.database.solution_confidence_records[0]
    assert record["evolutionary_history"] in record["generation_prompt"]["user"]
    assert record["parent_program"] in record["generation_prompt"]["user"]
    assert ("ORACLE evidence" in record["generation_prompt"]["user"]) == bool(oracle)
    assert record["confidence"] == 0.8
    assert record["y_t"] == 1
    assert record["brier_score"] == pytest.approx(0.04)


@pytest.mark.parametrize("score,label,brier", [(0.8, 1, 0.04), (0.5, 0, 0.64), (0.2, 0, 0.64)])
def test_parent_improvement_outcomes(score, label, brier):
    controller, _, _ = controller_fixture([EvaluationResult(metrics={"combined_score": score})])
    result = asyncio.run(controller._run_iteration(1))
    assert result.error is None
    record = controller.database.solution_confidence_records[0]
    assert record["y_t"] == label
    assert record["brier_score"] == pytest.approx(brier)


def test_failed_parse_and_invalid_candidate_retries_all_remain_in_ledger():
    controller, model, _ = controller_fixture(
        [
            EvaluationResult(metrics={"combined_score": 0, "validity": 0}),
            EvaluationResult(metrics={"combined_score": 0.7}),
        ]
    )
    controller._parse_llm_response.side_effect = [
        (None, None, "bad diff"),
        ("bad candidate", "change", None),
        ("good candidate", "change", None),
    ]
    result = asyncio.run(controller._run_iteration(4, retry_times=3))
    assert result.error is None
    records = controller.database.solution_confidence_records
    assert [record["attempt"] for record in records] == [1, 2, 3]
    assert [record["y_t"] for record in records] == [None, 0, 1]
    assert records[0]["assessment_status"] == "not_measured"
    assert records[0]["status"] == "parse_error"
    assert records[1]["brier_score"] == pytest.approx(0.64)
    assert all(record["web_document"] == "ORACLE evidence" for record in records)
    assert model.score_labels.await_count == 2
    assert model.score_verbalized.await_count == 2
    assert result.child_program_dict["solution_confidence"] == records[-1]


def test_assessment_reuses_exact_bounded_document_despite_later_source_changes():
    controller, model, response = controller_fixture(
        [EvaluationResult(metrics={"combined_score": 0.7})]
    )
    source = SearchResult(
        id="doc-1",
        title="Source document",
        url="https://example.org/document",
        content="A bound is {'score': 0.7}. " + "x" * 200 + "UNSEEN_SOURCE_TAIL",
        raw_content="The raw source must not be loaded again.",
    )
    injected = format_documents([source], max_document_chars=50)
    controller._maybe_run_evoduet.return_value = injected

    async def generate(system, user, **kwargs):
        assert injected in user
        source.content = "LATER_SOURCE_CHANGE"
        controller._maybe_run_evoduet.return_value = "LATER_SELECTED_DOCUMENT"
        return response

    async def score(system, messages, **kwargs):
        content = messages[0]["content"]
        assert f"[Web Document]\n{injected}" in content
        assert "UNSEEN_SOURCE_TAIL" not in content
        assert "LATER_SOURCE_CHANGE" not in content
        assert "LATER_SELECTED_DOCUMENT" not in content
        assert "raw source must not be loaded" not in content
        assert controller.evaluator.evaluate_program.await_count == 0
        return {"status": "success", "confidence": 0.8}

    controller._call_llm.side_effect = generate
    model.score_labels.side_effect = score
    result = asyncio.run(controller._run_iteration(1))

    assert result.error is None
    controller._maybe_run_evoduet.assert_awaited_once()
    record = controller.database.solution_confidence_records[0]
    assert record["web_document"] == injected
    assert result.child_program_dict["solution_confidence"]["web_document"] == injected
    assert record["brier_score"] == pytest.approx(0.04)


def test_missing_logprobs_preserve_observation_without_fabricated_confidence():
    controller, model, _ = controller_fixture([EvaluationResult(metrics={"combined_score": 0.7})])
    model.score_labels.return_value = {"status": "missing_logprobs", "confidence": None}
    result = asyncio.run(controller._run_iteration(1))
    assert result.error is None
    record = controller.database.solution_confidence_records[0]
    assert record["confidence"] is None
    assert record["confidence_verbalized"] == 0.6
    assert record["brier_score_verbalized"] == pytest.approx(0.16)
    assert record["status"] == "partial"
    assert record["y_t"] == 1
    assert record["brier_score"] is None
    assert record["assessment_status"] == "missing_logprobs"


def test_disabled_does_not_measure_or_change_metrics():
    controller, model, _ = controller_fixture(
        [EvaluationResult(metrics={"combined_score": 0.7})], enabled=False
    )
    result = asyncio.run(controller._run_iteration(1))
    assert result.error is None
    model.score_labels.assert_not_awaited()
    model.score_verbalized.assert_not_awaited()
    assert controller.database.solution_confidence_records == []
    assert result.child_program_dict["solution_confidence"] == {}
    assert result.child_program_dict["metrics"] == {"combined_score": 0.7}


def test_evaluator_exception_retains_prediction_but_not_an_observed_label():
    controller, _, _ = controller_fixture([RuntimeError("worker crashed")])
    result = asyncio.run(controller._run_iteration(1))
    assert result.error == "worker crashed"
    record = controller.database.solution_confidence_records[0]
    assert record["confidence"] == 0.8
    assert record["y_t"] is None
    assert record["brier_score"] is None
    assert record["status"] == "iteration_error"


def test_cancellation_retains_attempt_and_propagates():
    controller, _, _ = controller_fixture([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller._run_iteration(1))
    record = controller.database.solution_confidence_records[0]
    assert record["confidence"] == 0.8
    assert record["y_t"] is None
    assert record["status"] == "cancelled"


def test_no_parent_has_no_parent_improvement_outcome():
    controller, model, response = controller_fixture(
        [EvaluationResult(metrics={"combined_score": 0.7})]
    )
    controller.database.programs = {}
    controller.context_builder = SimpleNamespace(
        build_prompt=Mock(
            return_value={
                "system": "Task",
                "user": "Make a program",
            }
        )
    )
    response.text = "```python\nprint(1)\n```"
    result = asyncio.run(controller._run_iteration(0))
    assert result.error is None
    record = controller.database.solution_confidence_records[0]
    assert record["candidate_score"] == 0.7
    assert record["outcome_status"] == "no_parent"
    assert record["confidence"] is record["y_t"] is record["brier_score"] is None
    model.score_labels.assert_not_awaited()


def test_sequential_trace_updates_between_checkpoints_including_failure():
    controller, _, _ = controller_fixture([])
    controller.shutdown_event = multiprocessing.Event()
    controller._run_iteration = AsyncMock(return_value=SimpleNamespace(error="failed"))
    controller._process_iteration_result = Mock()
    controller._finalize_discovery = Mock()
    checkpoint = Mock()
    asyncio.run(controller._run_discovery_sequential(1, 2, checkpoint_callback=checkpoint))
    controller.database.write_evolution_trace.assert_called_once_with(controller.output_dir, 1)
    checkpoint.assert_called_once_with(2)


def test_all_diff_parse_failures_retain_unmeasured_records_for_every_attempt():
    controller, model, response = controller_fixture([])
    controller.config.diff_based_generation = True
    controller._parse_llm_response = DiscoveryController._parse_llm_response.__get__(controller)
    response.text = "```python\ndef solve(payload):\n    return [1, 2, 3, 4]\n```"
    controller.shutdown_event = multiprocessing.Event()
    controller._process_iteration_result = Mock()
    controller._finalize_discovery = Mock()
    snapshots = []
    controller.database.write_evolution_trace.side_effect = lambda *_: snapshots.append(
        json.loads(json.dumps(controller.database.solution_confidence_records))
    )

    asyncio.run(controller._run_discovery_sequential(1, 3, retry_times=3))

    assert [len(records) for records in snapshots] == [3, 6, 9]
    records = snapshots[-1]
    assert [(record["iteration"], record["attempt"]) for record in records] == [
        (iteration, attempt) for iteration in range(1, 4) for attempt in range(1, 4)
    ]
    for record in records:
        assert record["status"] == "parse_error"
        assert record["error"] == "No valid diffs found in response"
        assert record["assessment_status"] == "not_measured"
        assert record["assessment_status_verbalized"] == "not_measured"
        for field in (
            "confidence",
            "confidence_verbalized",
            "y_t",
            "brier_score",
            "brier_score_verbalized",
        ):
            assert record[field] is None
    model.score_labels.assert_not_awaited()
    model.score_verbalized.assert_not_awaited()
    controller.evaluator.evaluate_program.assert_not_awaited()


def test_concurrent_attempts_keep_each_generating_model_and_prediction():
    controller, _, _ = controller_fixture([])
    controller.config = Config()
    controller.config.solution_confidence.enabled = True
    controller.context_builder = DefaultContextBuilder(controller.config)
    controller._prompt_context = {}
    controller._build_prompt = DiscoveryController._build_prompt.__get__(controller)
    controller.database.get_statistics = Mock(
        side_effect=[
            {
                "previous_programs": [
                    Program(
                        id=f"history-{iteration}",
                        solution="previous code",
                        metrics={"combined_score": 0.1},
                        metadata={"changes": f"history snapshot for iteration {iteration}"},
                    )
                ]
            }
            for iteration in (1, 2)
        ]
    )
    controller._maybe_run_evoduet.side_effect = (
        lambda parent, history, iteration: f"web document for iteration {iteration}"
    )

    async def run():
        second_generated = asyncio.Event()
        second_assessed = asyncio.Event()

        async def first_score(system, messages, **kwargs):
            await second_assessed.wait()
            content = messages[0]["content"]
            assert "history snapshot for iteration 1" in content
            assert "history snapshot for iteration 2" not in content
            assert "candidate-1" in content
            assert "candidate-2" not in content
            assert "web document for iteration 1" in content
            assert "web document for iteration 2" not in content
            return {"status": "success", "confidence": 0.2, "model": "first"}

        async def second_score(system, messages, **kwargs):
            second_assessed.set()
            content = messages[0]["content"]
            assert "history snapshot for iteration 2" in content
            assert "history snapshot for iteration 1" not in content
            assert "candidate-2" in content
            assert "candidate-1" not in content
            assert "web document for iteration 2" in content
            assert "web document for iteration 1" not in content
            return {"status": "success", "confidence": 0.9, "model": "second"}

        async def generate(*args, **kwargs):
            iteration = kwargs["llm_context"]["iteration"]
            if iteration == 1:
                await second_generated.wait()
            else:
                second_generated.set()
            response = LLMResponse(text=f"candidate-{iteration}")
            response.generation_model = SimpleNamespace(
                score_labels=first_score if iteration == 1 else second_score,
                score_verbalized=AsyncMock(
                    return_value={"status": "success", "confidence": iteration / 4}
                ),
            )
            return response

        controller._call_llm.side_effect = generate
        controller._parse_llm_response.side_effect = lambda response, *args: (
            response,
            "change",
            None,
        )
        controller.evaluator.evaluate_program.side_effect = lambda candidate, program_id: (
            EvaluationResult(metrics={"combined_score": 0.4 if candidate == "candidate-1" else 0.8})
        )
        return await asyncio.gather(controller._run_iteration(1), controller._run_iteration(2))

    results = asyncio.run(run())
    assert all(result.error is None for result in results)
    records = {
        record["iteration"]: record for record in controller.database.solution_confidence_records
    }
    assert (records[1]["model"], records[1]["confidence"], records[1]["y_t"]) == ("first", 0.2, 0)
    assert (records[2]["model"], records[2]["confidence"], records[2]["y_t"]) == ("second", 0.9, 1)
    assert records[1]["brier_score"] == pytest.approx(0.04)
    assert records[2]["brier_score"] == pytest.approx(0.01)
    assert records[1]["confidence_verbalized"] == 0.25
    assert records[2]["confidence_verbalized"] == 0.5
    assert records[1]["brier_score_verbalized"] == pytest.approx(0.0625)
    assert records[2]["brier_score_verbalized"] == pytest.approx(0.25)
    assert "history snapshot for iteration 1" in records[1]["evolutionary_history"]
    assert "history snapshot for iteration 2" in records[2]["evolutionary_history"]
    assert controller.database.get_statistics.call_count == 2


def test_custom_controller_rejects_enabled_measurement_before_initializing_models():
    class CustomController(DiscoveryController):
        pass

    config = Config()
    config.solution_confidence.enabled = True
    with pytest.raises(ValueError, match="standard discovery controller"):
        CustomController(DiscoveryControllerInput(config, "evaluator.py", Mock()))


@pytest.mark.parametrize("mode", ["image", "agentic"])
def test_unsupported_generation_mode_rejected(mode):
    config = Config()
    config.solution_confidence.enabled = True
    if mode == "image":
        config.language = "image"
    else:
        config.agentic.enabled = True
    with pytest.raises(ValueError, match="non-agentic text/code"):
        DiscoveryController(DiscoveryControllerInput(config, "evaluator.py", Mock()))
