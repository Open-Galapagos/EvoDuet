"""Measure parent-improvement confidence without feeding it back into search.

The generation context is frozen before a candidate is generated. Independent
label-scoring and verbalized-probability calls use the generating model before
the candidate is evaluated.
Records live outside search metrics and retain every attempt, including failures.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone
from importlib.resources import files
from typing import Any, Mapping

from skydiscover.evaluation import evaluation_failure_reason
from skydiscover.llm.label_scoring import finite_json
from skydiscover.llm.response_metadata import json_safe

PROMPT_VERSION = "parent-improvement-v5"
PROMPT_TEMPLATE = "solution_confidence_user.txt"
ORACLE_PROMPT_VERSION = "parent-improvement-oracle-v1"
ORACLE_PROMPT_TEMPLATE = "solution_confidence_user_web.txt"
VERBALIZED_PROMPT_VERSION = "parent-improvement-verbalized-v1"
VERBALIZED_PROMPT_TEMPLATE = "solution_confidence_verbalized_user.txt"
VERBALIZED_WEB_PROMPT_VERSION = "parent-improvement-verbalized-web-v1"
VERBALIZED_WEB_PROMPT_TEMPLATE = "solution_confidence_verbalized_user_web.txt"


def _load_prompt_template(name: str) -> str:
    return files("skydiscover").joinpath("prompt", name).read_text(encoding="utf-8").rstrip("\r\n")


def finite_score(metrics: Any, metric: str) -> float | None:
    value = metrics.get(metric) if isinstance(metrics, Mapping) else None
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_payload(value: Any) -> Any:
    """Copy stored metadata into strict JSON values, including nested numeric failures."""
    return finite_json(json_safe(value))


def _invalid_candidate(metrics: Any, artifacts: Any) -> bool:
    """Require explicit candidate failure evidence; generic evaluator errors are ambiguous."""
    for container in (metrics, artifacts):
        if not isinstance(container, Mapping):
            continue
        if container.get("validity") in (False, 0, -1):
            return True
        if any(
            container.get(key)
            for key in ("candidate_error", "execution_error", "compilation_error")
        ):
            return True
        status = container.get("status")
        if isinstance(status, str) and status.strip().lower() == "compilation_error":
            return True
    return False


def _comparison_threshold(record: dict[str, Any]) -> float:
    offset = record["improvement_epsilon"] * (1 if record["higher_is_better"] else -1)
    return record["parent_score"] + offset


def new_record(
    settings,
    *,
    parent,
    iteration: int,
    attempt: int,
    run_id: str | None,
    task: str | None,
    seed: Any,
) -> dict[str, Any]:
    parent_metrics = copy.deepcopy(getattr(parent, "metrics", {}) or {})
    parent_artifacts = getattr(parent, "artifacts", {}) or {}
    return _safe_payload(
        {
            "record_id": uuid.uuid4().hex,
            "event": "parent_improved",
            "prompt_version": PROMPT_VERSION,
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_version_verbalized": VERBALIZED_PROMPT_VERSION,
            "prompt_template_verbalized": VERBALIZED_PROMPT_TEMPLATE,
            "run_id": run_id,
            "task": task,
            "seed": seed,
            "iteration": iteration,
            "attempt": attempt,
            "program_id": None,
            "parent_id": getattr(parent, "id", None),
            "parent_solution": getattr(parent, "solution", None),
            "parent_program": getattr(parent, "solution", None),
            "evolutionary_history": "(No evolutionary history provided.)",
            "web_document": "",
            "metric": settings.metric,
            "higher_is_better": settings.higher_is_better,
            "improvement_epsilon": settings.improvement_epsilon,
            "parent_score": finite_score(parent_metrics, settings.metric),
            "parent_evaluation_error": evaluation_failure_reason(parent_metrics, parent_artifacts),
            "candidate_score": None,
            "confidence": None,
            "confidence_verbalized": None,
            "y_t": None,
            "brier_score": None,
            "brier_score_verbalized": None,
            "logprob_true": None,
            "logprob_false": None,
            "label_probability_mass": None,
            "status": "pending",
            "phase": "generation",
            "assessment_status": "pending",
            "assessment_verbalized": None,
            "assessment_status_verbalized": "pending",
            "assessment_error_verbalized": None,
            "assessment_prompt_verbalized": None,
            "assessment_duration_seconds_verbalized": None,
            "outcome_status": "pending",
            "created_at": _now(),
            "completed_at": None,
        }
    )


def _select_prompt_template(record: dict[str, Any]) -> str:
    has_document = bool(record.get("web_document"))
    record["prompt_template"] = ORACLE_PROMPT_TEMPLATE if has_document else PROMPT_TEMPLATE
    record["prompt_version"] = ORACLE_PROMPT_VERSION if has_document else PROMPT_VERSION
    record["prompt_template_verbalized"] = (
        VERBALIZED_WEB_PROMPT_TEMPLATE if has_document else VERBALIZED_PROMPT_TEMPLATE
    )
    record["prompt_version_verbalized"] = (
        VERBALIZED_WEB_PROMPT_VERSION if has_document else VERBALIZED_PROMPT_VERSION
    )
    return record["prompt_template"]


def freeze_prompt(
    record: dict[str, Any] | None,
    prompt: dict[str, Any],
    *,
    web_document: str | None = None,
) -> None:
    """Freeze the prompt and the exact formatted document supplied for generation."""
    if record is not None:
        record["generation_prompt"] = _safe_payload(prompt)
        record["web_document"] = web_document or ""
        _select_prompt_template(record)


async def assess(
    record: dict[str, Any],
    settings,
    *,
    generation_model,
    candidate: str,
    program_id: str,
    seen_before: bool,
) -> None:
    """Measure both predictions independently, waiting for both before evaluation."""
    record.update(_safe_payload(record))
    _select_prompt_template(record)
    record.update(
        program_id=program_id,
        candidate_solution=candidate,
        candidate_hash=hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        previously_evaluated=seen_before,
        phase="assessment",
        confidence=None,
        confidence_verbalized=None,
    )
    if record["parent_id"] is None:
        record["assessment_status"] = "no_parent"
        record["assessment_status_verbalized"] = "no_parent"
        return
    if record["parent_score"] is None or record["parent_evaluation_error"] is not None:
        record["assessment_status"] = "invalid_parent_score"
        record["assessment_status_verbalized"] = "invalid_parent_score"
        return

    tasks = [
        asyncio.create_task(
            _assess_method(record, settings, generation_model, candidate, verbalized=verbalized)
        )
        for verbalized in (False, True)
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        # gather does not cancel a sibling when a child raises CancelledError.
        # Explicitly cancel and drain every assessment before propagating.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _assess_method(record, settings, generation_model, candidate, *, verbalized: bool):
    """Keep one method's template, backend, and timeout failures local to that method."""
    suffix = "_verbalized" if verbalized else ""
    method_name = "score_verbalized" if verbalized else "score_labels"
    method_description = "Verbalized confidence" if verbalized else "Label scoring"
    status_key = f"assessment_status{suffix}"
    error_key = f"assessment_error{suffix}"
    started = time.perf_counter()
    try:
        method = getattr(generation_model, method_name, None)
        if not callable(method):
            record[status_key] = "unsupported_model"
            record[error_key] = f"The generating backend does not support {method_name}."
            return
        if record.get("generation_context") is not None:
            context_heading = (
                "Exact runtime context used during generation, including tool outputs:"
            )
            context = record["generation_context"]
        else:
            context_heading = "Original information available before generating this candidate:"
            context = record.get("generation_prompt", {})
        template_name = record[f"prompt_template{suffix}"]
        user_message = _load_prompt_template(template_name).format(
            evolutionary_history=record.get("evolutionary_history")
            or "(No evolutionary history provided.)",
            parent_program=record.get("parent_program")
            or record.get("parent_solution")
            or "(not available)",
            generation_context_heading=context_heading,
            generation_context=json.dumps(context, ensure_ascii=False, indent=2),
            parent_solution=str(record.get("parent_solution") or "(not available)"),
            web_document=record.get("web_document") or "",
            candidate=candidate,
            metric=record["metric"],
            score_direction="higher" if record["higher_is_better"] else "lower",
            comparison="greater than" if record["higher_is_better"] else "less than",
            threshold=_comparison_threshold(record),
            parent_score=record["parent_score"],
            improvement_epsilon=record["improvement_epsilon"],
        )
        record[f"assessment_prompt{suffix}"] = {"system": "", "user": user_message}
        kwargs = {
            "timeout": settings.timeout,
            "openrouter_provider": settings.openrouter_provider,
        }
        if not verbalized:
            kwargs["top_logprobs"] = settings.top_logprobs
        result = await asyncio.wait_for(
            method("", [{"role": "user", "content": user_message}], **kwargs),
            timeout=settings.timeout,
        )
        if not isinstance(result, dict):
            record[f"assessment{suffix}"] = _safe_payload(result)
            record[status_key] = "invalid_response"
            record[error_key] = f"{method_description} must return a result object."
            return
        record[f"assessment{suffix}"] = _safe_payload(result)
        record[status_key] = str(
            result.get("status", "missing_probability" if verbalized else "missing_logprobs")
        )
        if not verbalized:
            for key in (
                "model",
                "api_base",
                "logprob_true",
                "logprob_false",
                "label_probability_mass",
            ):
                if key in result:
                    record[key] = _safe_payload(result[key])
        confidence = finite_score(result, "confidence")
        if (
            result.get("status") == "success"
            and confidence is not None
            and 0.0 <= confidence <= 1.0
        ):
            record[f"confidence{suffix}"] = confidence
        else:
            if result.get("status") == "success":
                record[status_key] = "invalid_response"
            record[error_key] = str(result.get("error") or f"{method_description} is unavailable.")
    except asyncio.CancelledError:
        record[status_key] = "cancelled"
        raise
    except asyncio.TimeoutError:
        record[status_key] = "timeout"
        record[error_key] = f"{method_description} exceeded {settings.timeout} seconds."
    except Exception as exc:
        record[status_key] = "error"
        record[error_key] = f"{type(exc).__name__}: {exc}"
    finally:
        record[f"assessment_duration_seconds{suffix}"] = time.perf_counter() - started


def finish(record: dict[str, Any] | None, evaluation_result) -> None:
    """Compare the exact evaluated candidate to its frozen parent score."""
    if record is None:
        return
    metrics = evaluation_result.metrics
    artifacts = evaluation_result.artifacts or {}
    score = finite_score(metrics, record["metric"])
    error = evaluation_failure_reason(metrics, artifacts)
    record.update(
        evaluator_metrics=_safe_payload(metrics),
        evaluator_artifacts=_safe_payload(artifacts),
        candidate_score=score,
        evaluation_error=error,
        phase="completed",
        completed_at=_now(),
        y_t=None,
        brier_score=None,
        brier_score_verbalized=None,
    )
    infrastructure_error = any(
        container.get(key)
        for container in (metrics, artifacts)
        if isinstance(container, Mapping)
        for key in ("infrastructure_error", "disk_space_error", "evaluator_error")
    )
    if record["parent_id"] is None:
        record["outcome_status"] = "no_parent"
    elif record["parent_score"] is None or record["parent_evaluation_error"] is not None:
        record["outcome_status"] = "invalid_parent_score"
    elif _invalid_candidate(metrics, artifacts):
        # Explicit invalidity establishes non-improvement without needing a score.
        record["y_t"] = 0
        record["outcome_status"] = "invalid_candidate"
    elif infrastructure_error or error is not None:
        # Generic errors, timeouts and numeric sentinels alone do not distinguish
        # candidate failure from evaluator failure. Keep their outcomes unknown.
        record["outcome_status"] = "evaluation_error"
    elif score is None:
        record["outcome_status"] = "missing_metric"
    else:
        delta = score - record["parent_score"]
        if not record["higher_is_better"]:
            delta = -delta
        record["score_delta"] = delta
        # Compare against the exact threshold stated in the assessment prompt.
        # Subtracting scores first can change a boundary tie through rounding.
        threshold = _comparison_threshold(record)
        record["y_t"] = int(score > threshold if record["higher_is_better"] else score < threshold)
        record["outcome_status"] = "observed"

    available = 0
    for suffix in ("", "_verbalized"):
        confidence = record.get(f"confidence{suffix}")
        if confidence is not None and record["y_t"] is not None:
            record[f"brier_score{suffix}"] = (confidence - record["y_t"]) ** 2
            available += 1
    record["status"] = {2: "completed", 1: "partial", 0: "incomplete"}[available]
    record.update(_safe_payload(record))


def abort(record: dict[str, Any] | None, status: str, error: str) -> None:
    if record is None or record.get("phase") == "completed":
        return
    for key in ("assessment_status", "assessment_status_verbalized"):
        if record.get(key, "pending") == "pending":
            record[key] = "not_measured"
    record.update(
        status=status,
        outcome_status=status,
        error=error,
        completed_at=_now(),
        phase="completed",
    )
    record.update(_safe_payload(record))
