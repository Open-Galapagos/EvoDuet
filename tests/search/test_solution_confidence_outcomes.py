import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skydiscover import solution_confidence
from skydiscover.config import SolutionConfidenceConfig
from skydiscover.evaluation import EvaluationResult
from skydiscover.search.base_database import Program
from skydiscover.solution_confidence import abort, assess, finish, freeze_prompt, new_record


def _record(*, parent_score=0.5, metric="combined_score", **settings):
    config = SolutionConfidenceConfig(metric=metric, **settings)
    parent = Program(id="parent", solution="parent()", metrics={metric: parent_score})
    record = new_record(
        config, parent=parent, iteration=3, attempt=2, run_id="run", task="task", seed=42
    )
    record["confidence"] = 0.8
    record["assessment_status"] = "success"
    record["confidence_verbalized"] = 0.6
    record["assessment_status_verbalized"] = "success"
    return config, record


def _model():
    model = AsyncMock()
    model.score_verbalized.return_value = {"status": "unsupported", "confidence": None}
    return model


@pytest.mark.parametrize(
    ("parent_score", "score", "higher_is_better", "epsilon", "expected"),
    [
        (0.5, 0.6, True, 0.0, 1),
        (0.5, 0.5, True, 0.0, 0),
        (0.5, 0.4, True, 0.0, 0),
        (-2.0, -1.0, True, 0.0, 1),
        (0.5, 0.4, False, 0.0, 1),
        (0.5, 0.5, False, 0.0, 0),
        (0.5, 0.6, False, 0.0, 0),
        (1.0, 1.5, True, 0.5, 0),
        (1.0, 1.6, True, 0.5, 1),
        (1.0, 0.5, False, 0.5, 0),
        (1.0, 0.4, False, 0.5, 1),
        (1.0, 1.1, True, 0.1, 0),
        (1.0, 0.9, False, 0.1, 0),
    ],
)
def test_outcome_uses_frozen_parent_direction_and_strict_margin(
    parent_score, score, higher_is_better, epsilon, expected
):
    _, record = _record(
        parent_score=parent_score,
        metric="objective",
        higher_is_better=higher_is_better,
        improvement_epsilon=epsilon,
    )
    finish(record, EvaluationResult(metrics={"objective": score}))

    assert record["y_t"] == expected
    assert record["brier_score"] == pytest.approx((0.8 - expected) ** 2)
    assert record["brier_score_verbalized"] == pytest.approx((0.6 - expected) ** 2)
    assert record["outcome_status"] == "observed"
    assert record["status"] == "completed"


@pytest.mark.parametrize(
    ("metrics", "artifacts"),
    [
        ({"validity": False}, {}),
        ({"validity": 0}, {}),
        ({"validity": -1}, {}),
        ({"combined_score": float("nan"), "validity": 0}, {}),
        ({"combined_score": 100.0, "validity": 0}, {}),
        ({}, {"validity": False}),
        ({}, {"candidate_error": "Invalid candidate output"}),
        ({"execution_error": "Candidate raised ValueError"}, {}),
        ({}, {"compilation_error": "Candidate did not compile"}),
        ({}, {"status": "compilation_error"}),
        ({"validity": 0, "timeout": True}, {}),
    ],
)
def test_explicit_invalid_candidate_is_zero_even_without_score(metrics, artifacts):
    _, record = _record()
    finish(record, EvaluationResult(metrics=metrics, artifacts=artifacts))

    assert record["y_t"] == 0
    assert record["brier_score"] == pytest.approx(0.64)
    assert record["brier_score_verbalized"] == pytest.approx(0.36)
    assert record["outcome_status"] == "invalid_candidate"
    assert record["status"] == "completed"
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("metrics", "artifacts", "status"),
    [
        ({}, {}, "missing_metric"),
        ({"combined_score": float("nan")}, {}, "evaluation_error"),
        ({"combined_score": float("inf")}, {}, "evaluation_error"),
        ({"error": 0.0}, {}, "evaluation_error"),
        ({"error": 0.0, "timeout": True}, {}, "evaluation_error"),
        ({"combined_score": 0.0, "timeout": True}, {}, "evaluation_error"),
        ({"combined_score": 0.0, "error": "evaluator crashed"}, {}, "evaluation_error"),
        ({"combined_score": 1.0, "infrastructure_error": True}, {}, "evaluation_error"),
        ({"combined_score": 1.0}, {"disk_space_error": True}, "evaluation_error"),
        ({"combined_score": 1.0}, {"evaluator_error": "crashed"}, "evaluation_error"),
        ({"combined_score": 0.0}, {"status": "error"}, "evaluation_error"),
        ({"combined_score": 0.0}, {"status": "timeout"}, "evaluation_error"),
        ({"combined_score": 0.0, "evaluation_failed": True}, {}, "evaluation_error"),
    ],
)
def test_unobserved_evaluations_are_missing_not_negative_labels(metrics, artifacts, status):
    _, record = _record()
    finish(record, EvaluationResult(metrics=metrics, artifacts=artifacts))

    assert record["y_t"] is None
    assert record["brier_score"] is None
    assert record["brier_score_verbalized"] is None
    assert record["outcome_status"] == status
    assert record["status"] == "incomplete"
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("parent_score", [None, float("nan"), float("inf")])
def test_invalid_parent_prevents_false_observed_outcome(parent_score):
    _, record = _record(parent_score=parent_score)
    finish(record, EvaluationResult(metrics={"combined_score": 1.0}))

    assert record["outcome_status"] == "invalid_parent_score"
    assert record["y_t"] is None
    assert record["brier_score"] is None
    json.dumps(record, allow_nan=False)


@pytest.mark.asyncio
async def test_no_parent_skips_assessment_and_has_no_improvement_outcome():
    config = SolutionConfidenceConfig()
    record = new_record(
        config, parent=None, iteration=1, attempt=1, run_id="run", task="task", seed=42
    )
    model = _model()
    await assess(
        record,
        config,
        generation_model=model,
        candidate="candidate()",
        program_id="child",
        seen_before=False,
    )
    finish(record, EvaluationResult(metrics={"combined_score": 1.0}))

    model.score_labels.assert_not_called()
    model.score_verbalized.assert_not_called()
    assert record["assessment_status"] == "no_parent"
    assert record["assessment_status_verbalized"] == "no_parent"
    assert record["outcome_status"] == "no_parent"
    assert record["y_t"] is None
    assert record["brier_score"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status", "expected"),
    [
        ({"status": "success", "confidence": 0.8}, "success", 0.8),
        ({"status": "success", "confidence": 0.0}, "success", 0.0),
        ({"status": "success", "confidence": 1.0}, "success", 1.0),
        ({"status": "timeout", "confidence": 0.8}, "timeout", None),
        ({"status": "unavailable", "confidence": 0.8}, "unavailable", None),
        ({"confidence": 0.8}, "missing_logprobs", None),
        ({"status": "success", "confidence": -0.1}, "invalid_response", None),
        ({"status": "success", "confidence": 1.1}, "invalid_response", None),
        ({"status": "success", "confidence": True}, "invalid_response", None),
        ({"status": "success", "confidence": float("nan")}, "invalid_response", None),
        ({"status": "success", "confidence": float("inf")}, "invalid_response", None),
        ({"status": "success"}, "invalid_response", None),
        (None, "invalid_response", None),
    ],
)
async def test_assessment_only_accepts_successful_finite_probabilities(result, status, expected):
    config, record = _record()
    model = _model()
    model.score_labels.return_value = result
    await assess(
        record,
        config,
        generation_model=model,
        candidate="candidate()",
        program_id="child",
        seen_before=False,
    )

    assert record["assessment_status"] == status
    assert record["confidence"] == expected
    json.dumps(record, allow_nan=False)
    finish(record, EvaluationResult(metrics={"combined_score": 0.6}))
    assert record["y_t"] == 1
    if expected is None:
        assert record["brier_score"] is None
        assert record["status"] == "incomplete"
    else:
        assert record["brier_score"] == pytest.approx((expected - 1) ** 2)


@pytest.mark.asyncio
async def test_assessment_uses_selected_sections_and_preserves_full_context_in_trace():
    config, record = _record()
    original_prompt = {"user": "Original task", "metrics": {"score": float("nan")}}
    freeze_prompt(record, original_prompt)
    original_prompt["user"] = "Later task"
    context = {
        "messages": [
            {"role": "user", "content": "Original task"},
            {"role": "tool", "content": "Retrieved oracle contents"},
        ],
        "diagnostics": [float("inf"), float("-inf")],
    }
    record["generation_context"] = context
    record["evolutionary_history"] = "Earlier program improved from 0.3 to 0.5."
    model = _model()
    model.score_labels.return_value = {
        "status": "success",
        "confidence": 0.8,
        "logprob_true": float("nan"),
        "responses": [{"diagnostics": float("inf")}],
    }
    await assess(
        record,
        config,
        generation_model=model,
        candidate="candidate()",
        program_id="child",
        seen_before=True,
    )

    system, messages = model.score_labels.call_args.args
    user_message = messages[0]["content"]
    assert "Earlier program improved from 0.3 to 0.5." in user_message
    assert "Original task" not in user_message
    assert "Later task" not in user_message
    assert "Retrieved oracle contents" not in user_message
    assert "[Web Document]" not in user_message
    assert record["web_document"] == ""
    assert record["prompt_template"] == "solution_confidence_user.txt"
    assert record["prompt_version"] == "parent-improvement-v5"
    assert "candidate()" in user_message
    assert "parent()" in user_message
    assert (
        "[Statement]\nThe generated candidate will achieve a higher combined_score than the current parent program."
        in user_message
    )
    assert (
        "Your task is to determine whether the following statement is true or false."
        in user_message
    )
    assert user_message.endswith("Is this statement True or False?")
    assert system == ""
    assert record["assessment_prompt"]["system"] == ""
    assert messages == [{"role": "user", "content": user_message}]
    assert user_message.startswith("You are given the recent evolutionary history")
    assert "Do not execute code or call tools" in user_message
    assert "Return exactly one word: True or False." in user_message
    assert record["previously_evaluated"] is True
    assert record["generation_prompt"]["metrics"]["score"] is None
    assert record["generation_context"]["diagnostics"] == [None, None]
    assert record["logprob_true"] is None
    assert record["assessment"]["responses"][0]["diagnostics"] is None
    context["messages"][1]["content"] = "Mutated tool output"
    assert record["generation_context"]["messages"][1]["content"] == "Retrieved oracle contents"

    result = EvaluationResult(
        metrics={"combined_score": 0.6, "other": float("nan")},
        artifacts={"diagnostics": [float("inf")]},
    )
    finish(record, result)
    result.metrics["combined_score"] = 10
    assert record["evaluator_metrics"] == {"combined_score": 0.6, "other": None}
    assert record["evaluator_artifacts"]["diagnostics"] == [None]
    json.dumps(record, allow_nan=False)


@pytest.mark.asyncio
async def test_oracle_assessment_uses_exact_frozen_document_without_full_context():
    config, record = _record()
    document = (
        "  ## Retrieved source\nURL: https://example.org/method\n"
        "Use {candidate} as a literal key: {'parent_program': '{value}'}.\n\n"
    )
    prompt = {"user": f"Task-specific generation instructions\n{document}"}
    freeze_prompt(record, prompt, web_document=document)
    assert record["web_document"] == document
    assert record["prompt_template"] == "solution_confidence_user_web.txt"
    assert record["prompt_version"] == "parent-improvement-oracle-v1"
    prompt["user"] = "Later source contents"
    record["generation_context"] = {
        "messages": [{"role": "tool", "content": "An unrelated tool document"}]
    }
    record["evolutionary_history"] = "The previous candidate scored 0.4."
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}

    await assess(
        record,
        config,
        generation_model=model,
        candidate="candidate()",
        program_id="child",
        seen_before=False,
    )

    system, messages = model.score_labels.call_args.args
    user_message = messages[0]["content"]
    assert system == ""
    assert messages == [{"role": "user", "content": user_message}]
    assert f"[Web Document]\n{document}\n\n[Generated Candidate Solution]" in user_message
    assert "[Evolutionary History]\nThe previous candidate scored 0.4." in user_message
    assert "[Current Program]\nparent()" in user_message
    assert "[Generated Candidate Solution]\ncandidate()" in user_message
    assert "based on that history, parent program, and web document" in user_message
    assert (
        "Use the supplied history, programs, and web document as reference material."
        in user_message
    )
    assert "Task-specific generation instructions" not in user_message
    assert "Later source contents" not in user_message
    assert "An unrelated tool document" not in user_message
    assert user_message.endswith("Is this statement True or False?")
    assert "Return exactly one word: True or False." in user_message
    assert record["generation_prompt"]["user"].endswith(document)
    assert record["web_document"] == document
    assert record["assessment_status"] == "success"
    json.dumps(record, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("web_document", [None, ""])
async def test_empty_document_uses_default_prompt_and_resets_oracle_metadata(web_document):
    config, record = _record()
    assert record["web_document"] == ""
    assert record["prompt_template"] == "solution_confidence_user.txt"
    assert record["prompt_version"] == "parent-improvement-v5"
    freeze_prompt(record, {"user": "Oracle prompt"}, web_document="Earlier document")
    freeze_prompt(record, {"user": "Default prompt"}, web_document=web_document)
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}

    await assess(
        record,
        config,
        generation_model=model,
        candidate="candidate()",
        program_id="child",
        seen_before=False,
    )

    user_message = model.score_labels.call_args.args[1][0]["content"]
    assert "[Web Document]" not in user_message
    assert "Earlier document" not in user_message
    assert record["web_document"] == ""
    assert record["prompt_template"] == "solution_confidence_user.txt"
    assert record["prompt_version"] == "parent-improvement-v5"
    assert record["assessment_status"] == "success"


@pytest.mark.asyncio
async def test_parent_solution_is_frozen_even_if_generation_prompt_omits_it():
    config = SolutionConfidenceConfig()
    parent = Program(id="parent", solution="original_parent()", metrics={"combined_score": 0.5})
    record = new_record(
        config, parent=parent, iteration=1, attempt=1, run_id="run", task="task", seed=42
    )
    freeze_prompt(record, {"user": "Custom prompt without parent source"})
    parent.solution = "modified_parent()"
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}
    await assess(
        record,
        config,
        generation_model=model,
        candidate="candidate()",
        program_id="child",
        seen_before=False,
    )

    assert record["parent_solution"] == "original_parent()"
    user_message = model.score_labels.call_args.args[1][0]["content"]
    assert "Custom prompt without parent source" not in user_message
    assert "original_parent()" in user_message
    assert "modified_parent()" not in user_message


@pytest.mark.asyncio
async def test_template_inserts_history_parent_and_candidate_without_reformatting_code():
    config, record = _record()
    history = "Previous source: values = {'parent_program': '{candidate}'}"
    parent = "def parent():\n    return {'value': 1, 'history': '{evolutionary_history}'}"
    candidate = "def candidate():\n    return f'{2 + 3}'"
    record.update(evolutionary_history=history, parent_program=parent)
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}

    await assess(
        record,
        config,
        generation_model=model,
        candidate=candidate,
        program_id="child",
        seen_before=False,
    )

    user_message = model.score_labels.call_args.args[1][0]["content"]
    assert f"[Evolutionary History]\n{history}" in user_message
    assert f"[Current Program]\n{parent}" in user_message
    assert f"[Generated Candidate Solution]\n{candidate}" in user_message
    assert record["assessment_status"] == "success"


@pytest.mark.asyncio
async def test_assessment_failure_does_not_prevent_observed_label():
    config, record = _record()
    model = _model()
    model.score_labels.side_effect = RuntimeError("provider unavailable")
    await assess(
        record,
        config,
        generation_model=model,
        candidate="candidate()",
        program_id="child",
        seen_before=False,
    )
    finish(record, EvaluationResult(metrics={"combined_score": 0.6}))

    assert record["assessment_status"] == "error"
    assert record["confidence"] is None
    assert record["y_t"] == 1
    assert record["brier_score"] is None
    assert "provider unavailable" in record["assessment_error"]


@pytest.mark.asyncio
async def test_cancelled_assessment_propagates_without_inventing_outcome():
    config, record = _record()
    model = _model()
    model.score_labels.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await assess(
            record,
            config,
            generation_model=model,
            candidate="candidate()",
            program_id="child",
            seen_before=False,
        )
    abort(record, "cancelled", "Run interrupted")

    assert record["assessment_status"] == "cancelled"
    assert record["status"] == "cancelled"
    assert record["confidence"] is None
    assert record["y_t"] is None
    assert record["brier_score"] is None
    assert record["completed_at"] is not None


def test_abort_sanitizes_pending_records_and_does_not_overwrite_finished_results():
    _, record = _record()
    record["generation_context"] = {"diagnostics": float("nan")}
    abort(record, "generation_failed", "No candidate generated")
    assert record["generation_context"]["diagnostics"] is None
    assert record["status"] == "generation_failed"
    saved = deepcopy(record)
    abort(record, "cancelled", "Later interruption")
    assert record == saved
    json.dumps(record, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("labels", "verbalized", "status"),
    [
        (0.8, 0.6, "completed"),
        (0.8, None, "partial"),
        (None, 0.6, "partial"),
        (None, None, "incomplete"),
    ],
)
async def test_both_assessments_keep_independent_briers_for_the_same_outcome(
    labels, verbalized, status
):
    config, record = _record()
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": labels}
    model.score_verbalized.return_value = {"status": "success", "confidence": verbalized}

    await assess(
        record,
        config,
        generation_model=model,
        candidate="child()",
        program_id="child",
        seen_before=False,
    )
    finish(record, EvaluationResult(metrics={"combined_score": 0.6}))

    assert record["confidence"] == labels
    assert record["confidence_verbalized"] == verbalized
    assert record["y_t"] == 1
    assert record["brier_score"] == (None if labels is None else pytest.approx((labels - 1) ** 2))
    assert record["brier_score_verbalized"] == (
        None if verbalized is None else pytest.approx((verbalized - 1) ** 2)
    )
    assert record["status"] == status
    assert record["assessment_duration_seconds"] >= 0
    assert record["assessment_duration_seconds_verbalized"] >= 0
    assert record["assessment_verbalized"] == model.score_verbalized.return_value
    assert model.score_verbalized.call_args.kwargs == {
        "timeout": config.timeout,
        "openrouter_provider": config.openrouter_provider,
    }
    json.dumps(record, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "status"),
    [
        ({"status": "success", "confidence": True}, "invalid_response"),
        ({"status": "success", "confidence": -0.1}, "invalid_response"),
        ({"status": "success", "confidence": 1.1}, "invalid_response"),
        ({"status": "success", "confidence": float("nan")}, "invalid_response"),
        ({"status": "success", "confidence": float("inf")}, "invalid_response"),
        ({"status": "success"}, "invalid_response"),
        ({"status": "timeout", "confidence": 0.9}, "timeout"),
        ({"confidence": 0.9}, "missing_probability"),
        (None, "invalid_response"),
    ],
)
async def test_verbalized_failure_or_invalid_probability_does_not_discard_logprob_result(
    result, status
):
    config, record = _record()
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}
    model.score_verbalized.return_value = result

    await assess(
        record,
        config,
        generation_model=model,
        candidate="child()",
        program_id="child",
        seen_before=False,
    )
    finish(record, EvaluationResult(metrics={"combined_score": 0.6}))

    assert record["assessment_status_verbalized"] == status
    assert record["assessment_error_verbalized"]
    assert record["confidence_verbalized"] is None
    assert record["brier_score_verbalized"] is None
    assert record["confidence"] == 0.8
    assert record["brier_score"] == pytest.approx(0.04)
    assert record["status"] == "partial"
    json.dumps(record, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("available", ["score_labels", "score_verbalized"])
async def test_backend_can_support_only_one_assessment_method(available):
    config, record = _record()
    model = SimpleNamespace(
        **{available: AsyncMock(return_value={"status": "success", "confidence": 0.7})}
    )

    await assess(
        record,
        config,
        generation_model=model,
        candidate="child()",
        program_id="child",
        seen_before=False,
    )
    finish(record, EvaluationResult(metrics={"combined_score": 0.6}))

    supported_suffix = "" if available == "score_labels" else "_verbalized"
    missing_suffix = "_verbalized" if available == "score_labels" else ""
    assert record[f"confidence{supported_suffix}"] == 0.7
    assert record[f"assessment_status{missing_suffix}"] == "unsupported_model"
    assert record[f"confidence{missing_suffix}"] is None
    assert record["status"] == "partial"


@pytest.mark.asyncio
async def test_two_assessment_calls_run_concurrently_and_both_finish_before_return():
    config, record = _record()
    started = {method: asyncio.Event() for method in ("score_labels", "score_verbalized")}
    released = {method: asyncio.Event() for method in started}

    async def result(method, value):
        started[method].set()
        await released[method].wait()
        return {"status": "success", "confidence": value}

    model = SimpleNamespace(
        score_labels=AsyncMock(side_effect=lambda *args, **kwargs: None),
        score_verbalized=AsyncMock(side_effect=lambda *args, **kwargs: None),
    )

    async def label_call(*args, **kwargs):
        return await result("score_labels", 0.8)

    async def verbalized_call(*args, **kwargs):
        return await result("score_verbalized", 0.6)

    model.score_labels.side_effect = label_call
    model.score_verbalized.side_effect = verbalized_call
    task = asyncio.create_task(
        assess(
            record,
            config,
            generation_model=model,
            candidate="child()",
            program_id="child",
            seen_before=False,
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started.values())), timeout=1
        )
        assert not task.done()
        released["score_labels"].set()
        await asyncio.sleep(0)
        assert not task.done()
        released["score_verbalized"].set()
        await asyncio.wait_for(task, timeout=1)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert record["confidence"] == 0.8
    assert record["confidence_verbalized"] == 0.6


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_method", ["score_labels", "score_verbalized"])
@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_timeout_or_exception_is_local_to_one_method(failed_method, failure):
    config, record = _record(timeout=0.01)
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}
    model.score_verbalized.return_value = {"status": "success", "confidence": 0.6}
    stopped = asyncio.Event()

    async def fail(*args, **kwargs):
        try:
            if failure == "exception":
                raise RuntimeError("provider unavailable")
            await asyncio.Event().wait()
        finally:
            stopped.set()

    getattr(model, failed_method).side_effect = fail
    await assess(
        record,
        config,
        generation_model=model,
        candidate="child()",
        program_id="child",
        seen_before=False,
    )
    finish(record, EvaluationResult(metrics={"combined_score": 0.6}))

    suffix = "" if failed_method == "score_labels" else "_verbalized"
    assert record[f"assessment_status{suffix}"] == ("timeout" if failure == "timeout" else "error")
    assert record[f"confidence{suffix}"] is None
    assert record["status"] == "partial"
    assert stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_child", [False, True])
async def test_cancellation_cancels_and_drains_both_assessment_tasks(cancel_child):
    config, record = _record()
    started = [asyncio.Event(), asyncio.Event()]
    stopped = [asyncio.Event(), asyncio.Event()]
    trigger = asyncio.Event()

    async def run(index):
        started[index].set()
        try:
            await trigger.wait()
            if cancel_child and index == 0:
                raise asyncio.CancelledError()
            await asyncio.Event().wait()
        finally:
            stopped[index].set()

    async def labels(*args, **kwargs):
        return await run(0)

    async def verbalized(*args, **kwargs):
        return await run(1)

    model = SimpleNamespace(score_labels=labels, score_verbalized=verbalized)
    task = asyncio.create_task(
        assess(
            record,
            config,
            generation_model=model,
            candidate="child()",
            program_id="child",
            seen_before=False,
        )
    )
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), timeout=1)
    if cancel_child:
        trigger.set()
    else:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert all(event.is_set() for event in stopped)
    assert record["assessment_status"] == record["assessment_status_verbalized"] == "cancelled"
    assert record["confidence"] is record["confidence_verbalized"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("has_document", [False, True])
@pytest.mark.parametrize("broken_verbalized", [False, True])
async def test_template_failure_is_independent_between_methods(
    monkeypatch, has_document, broken_verbalized
):
    config, record = _record()
    freeze_prompt(record, {"user": "Task"}, web_document="Frozen source" if has_document else None)
    original = solution_confidence._load_prompt_template

    def load(name):
        if ("verbalized" in name) == broken_verbalized:
            raise FileNotFoundError(name)
        return original(name)

    monkeypatch.setattr(solution_confidence, "_load_prompt_template", load)
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}
    model.score_verbalized.return_value = {"status": "success", "confidence": 0.6}
    await assess(
        record,
        config,
        generation_model=model,
        candidate="child()",
        program_id="child",
        seen_before=False,
    )
    finish(record, EvaluationResult(metrics={"combined_score": 0.6}))

    suffix = "_verbalized" if broken_verbalized else ""
    assert record[f"assessment_status{suffix}"] == "error"
    assert "FileNotFoundError" in record[f"assessment_error{suffix}"]
    assert record["status"] == "partial"
    broken_method = model.score_verbalized if broken_verbalized else model.score_labels
    broken_method.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("document", ["", "Web evidence {'key': '{value}'}"])
async def test_both_templates_share_frozen_history_parent_document_and_candidate(document):
    config, record = _record()
    freeze_prompt(record, {"user": "Do not include full prompt"}, web_document=document)
    record["evolutionary_history"] = "Recent search {'score': 0.5}"
    candidate = "def child():\n    return {'value': 1}"
    model = _model()
    model.score_labels.return_value = {"status": "success", "confidence": 0.8}
    model.score_verbalized.return_value = {"status": "success", "confidence": 0.6}

    await assess(
        record,
        config,
        generation_model=model,
        candidate=candidate,
        program_id="child",
        seen_before=False,
    )

    for method in (model.score_labels, model.score_verbalized):
        system, messages = method.call_args.args
        assert system == ""
        prompt = messages[0]["content"]
        assert f"[Evolutionary History]\n{record['evolutionary_history']}" in prompt
        assert "[Current Program]\nparent()" in prompt
        assert f"[Generated Candidate Solution]\n{candidate}" in prompt
        assert "Do not include full prompt" not in prompt
        if document:
            assert f"[Web Document]\n{document}" in prompt
        else:
            assert "[Web Document]" not in prompt
    assert record["prompt_template_verbalized"] == (
        "solution_confidence_verbalized_user_web.txt"
        if document
        else "solution_confidence_verbalized_user.txt"
    )
    assert record["prompt_version_verbalized"] == (
        "parent-improvement-verbalized-web-v1" if document else "parent-improvement-verbalized-v1"
    )
    assert (
        record["assessment_prompt_verbalized"]["user"]
        == model.score_verbalized.call_args.args[1][0]["content"]
    )


@pytest.mark.asyncio
async def test_invalid_parent_skips_both_measurement_methods():
    config, record = _record(parent_score=None)
    model = _model()
    await assess(
        record,
        config,
        generation_model=model,
        candidate="child()",
        program_id="child",
        seen_before=False,
    )
    model.score_labels.assert_not_called()
    model.score_verbalized.assert_not_called()
    assert (
        record["assessment_status"]
        == record["assessment_status_verbalized"]
        == "invalid_parent_score"
    )
    assert record["confidence"] is record["confidence_verbalized"] is None


def test_abort_marks_both_pending_methods_not_measured():
    config = SolutionConfidenceConfig()
    record = new_record(
        config, parent=None, iteration=1, attempt=1, run_id="run", task="task", seed=42
    )
    abort(record, "generation_error", "No candidate generated")
    assert record["assessment_status"] == record["assessment_status_verbalized"] == "not_measured"
