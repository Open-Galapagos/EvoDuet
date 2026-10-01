"""Detailed failure feedback used only by the three SimpleTES GPU evaluators."""

from collections.abc import Mapping
from typing import Any, Optional

from skydiscover.evaluation.evaluation_result import evaluation_failure_reason

_MAX_ERROR_CHARS = 4000


def _message(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(filter(None, (_message(item) for item in value)))
    return ""


def simpletes_gpu_failure_reason(
    metrics: Optional[Mapping[str, Any]],
    artifacts: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Keep the legacy failure decision and enrich only its diagnostic text.

    GPUKernel returns actionable static/compiler/correctness errors alongside
    validity=0. Preserve those details for generation retries without changing
    metrics, scores, success classification, or other evaluators' feedback.
    """
    reason = evaluation_failure_reason(metrics, artifacts)
    if reason is None:
        return None

    metrics = metrics if isinstance(metrics, Mapping) else {}
    artifacts = artifacts if isinstance(artifacts, Mapping) else {}
    containers = (metrics, artifacts)
    detail = next(
        (
            text
            for container in containers
            for key in ("error_message", "error")
            if (text := _message(container.get(key)))
        ),
        "",
    )
    if not detail:
        # Upstream's empty-kernel rejection can carry details only in metadata.
        for container in containers:
            metadata = container.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            detail = _message(
                [metadata.get("static_errors"), metadata.get("static_bypass")]
            ) or _message(metadata.get("log_excerpt"))
            if detail:
                break

    if not detail:
        return reason
    error_name = _message(metrics.get("error_name")) or _message(artifacts.get("error_name"))
    if error_name and not detail.startswith(f"{error_name}:"):
        detail = f"{error_name}: {detail}"
    if len(detail) > _MAX_ERROR_CHARS:
        marker = "\n... (truncated) ...\n"
        head = (_MAX_ERROR_CHARS - len(marker)) // 2
        tail = _MAX_ERROR_CHARS - len(marker) - head
        detail = detail[:head] + marker + detail[-tail:]
    return detail
