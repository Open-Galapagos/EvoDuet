import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Union

_FAILURE_STATUSES = {
    "error",
    "failed",
    "failure",
    "timeout",
    "timed_out",
    "compilation_error",
}


def _error_text(value: Any) -> Optional[str]:
    """Return a useful evaluator error message when *value* contains one."""
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if value not in (None, False, 0, 0.0):
        return str(value)
    return None


def evaluation_failure_reason(
    metrics: Optional[Mapping[str, Any]],
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return why an evaluation failed, or ``None`` when no failure is declared.

    Evaluators in this repository use several compatible failure conventions:
    invalid ``validity``, ``timeout``, container ``status``, and explicit error
    fields paired with sentinel scores such as ``0`` or ``-100``.  Keeping this
    interpretation in one place prevents controllers and smoke validators from
    disagreeing about whether a candidate was successfully evaluated.

    A negative score alone is not a failure because some valid objectives have
    a negative range.  It becomes a failure when accompanied by an explicit
    evaluator error marker.
    """
    metrics = metrics if isinstance(metrics, Mapping) else {}
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}

    validity = metrics.get("validity")
    if validity in (0, -1):
        return f"Evaluation reported validity={validity}"

    if metrics.get("timeout") is True:
        return _error_text(metrics.get("error_message")) or "Evaluation timed out"

    evaluation_failed = metrics.get("evaluation_failed")
    if evaluation_failed is True or (
        isinstance(evaluation_failed, (int, float))
        and not isinstance(evaluation_failed, bool)
        and evaluation_failed != 0
    ):
        return _error_text(metrics.get("error_message")) or "Evaluation reported failure"

    status = artifacts.get("status", metrics.get("status"))
    if isinstance(status, str) and status.strip().lower() in _FAILURE_STATUSES:
        return f"Evaluation status is {status.strip()!r}"

    for container in (metrics, artifacts):
        for key in ("error_message", "error"):
            message = _error_text(container.get(key))
            if message is not None:
                return message

    score_present = "combined_score" in metrics
    score = metrics.get("combined_score")
    if score_present:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            return f"Evaluation returned invalid combined_score={score!r}"
        if not math.isfinite(numeric_score):
            return f"Evaluation returned non-finite combined_score={score!r}"

    # Framework-level evaluator failures historically use {"error": 0.0}
    # without a combined_score or artifact message.  Preserve that convention
    # without treating a numeric diagnostic named "error" as fatal when the
    # evaluator also supplied a usable positive score.
    if "error" in metrics:
        try:
            numeric_score = float(score) if score_present else None
        except (TypeError, ValueError):
            numeric_score = None
        if numeric_score is None or numeric_score <= 0:
            return f"Evaluation reported error={metrics.get('error')!r}"

    return None


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts
    """

    metrics: Dict[str, float]
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)
    llm_reasoning_content: Optional[str] = None
    llm_reasoning: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, Any]:
        result = dict(self.metrics)
        if self.artifacts:
            result["artifacts"] = self.artifacts
        if self.llm_reasoning_content is not None:
            result["llm_reasoning_content"] = self.llm_reasoning_content
        if self.llm_reasoning:
            result["llm_reasoning"] = self.llm_reasoning
        return result
