"""Render a compact evolutionary history for EvoDuet prompts."""

from __future__ import annotations

import math
from typing import Any, List

_HISTORY_MAX_CHARS = 12_000


def _attr(program: Any, name: str, default=None):
    """Get a field from a ``Program`` object or a dict-shaped program."""
    if isinstance(program, dict):
        return program.get(name, default)
    return getattr(program, name, default)


def _other_metrics(metrics: dict) -> dict:
    """Per-metric breakdown, excluding headline and error fields."""
    return {
        k: number
        for k, v in (metrics or {}).items()
        if (number := _number(v)) is not None
        if k not in {"combined_score", "evoduet_score", "error"}
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _summary(value: Any, limit: int = 300) -> str:
    return " ".join(str(value).strip().replace("```", "~~~").split())[:limit]


def _render_program(program: Any, index: int) -> str:
    """Render identifiers and scores only; the parent prompt already carries code."""
    metrics = _attr(program, "metrics", {}) or {}
    iteration = _attr(program, "iteration_found")
    combined = _number(metrics.get("combined_score"))
    proxy = _number(metrics.get("evoduet_score"))

    tags = []
    if iteration is not None:
        tags.append(f"iteration {iteration}")
    if proxy is not None:
        tags.append(f"evoduet_score: {proxy:.4f} (higher is better)")
    elif combined is not None:
        tags.append(f"combined_score: {combined:.4f}")
    lines = [f"### Program {index}" + (f" ({', '.join(tags)})" if tags else "")]

    error = metrics.get("error")
    if error:
        lines.append(f"- error: {_summary(error)}")
    breakdown = _other_metrics(metrics)
    if breakdown:
        lines.append("Score breakdown:")
        for key, value in breakdown.items():
            lines.append(f"  - {key}: {value:.4f}")

    metadata = _attr(program, "metadata", {}) or {}
    changes = metadata.get("changes") if isinstance(metadata, dict) else None
    if changes:
        summary = _summary(changes)
        if summary:
            lines.append(f"Changes: {summary}")

    return "\n".join(lines)


def verbalize_history(history: Any, max_chars: int | None = _HISTORY_MAX_CHARS) -> str:
    """Render the evolutionary history into prompt text.

    An already-rendered string (the scaffold's own evolutionary history, as its
    context builder showed it in the mutation prompt) is returned verbatim: no
    truncation or re-rendering. Otherwise accepts the per-island dict
    ``{label: [Program, ...]}``, a flat ``[Program, ...]`` list, or ``None``; programs
    are numbered per group and shown with their iteration and scores, solution bodies
    omitted because the current parent is supplied separately, and the complete
    rendering is cut in the middle to ``max_chars`` (``None`` renders it whole).
    Returns ``"(none)"`` when empty.
    """
    budget = None if max_chars is None else int(max_chars)
    if budget is not None and budget <= 0:
        raise ValueError("max_chars must be positive")
    if history is None:
        return "(none)"
    if isinstance(history, str):
        return history if history else "(none)"
    groups = history if isinstance(history, dict) else {"": list(history)}

    blocks: List[str] = []
    for label, programs in groups.items():
        programs = list(programs or [])
        if not programs:
            continue
        section = [f"## {label}"] if label else []
        section.extend(_render_program(p, i) for i, p in enumerate(programs, 1))
        blocks.append("\n\n".join(section))
    rendered = "\n\n".join(blocks)
    return _bound(rendered, budget) if rendered else "(none)"


def _bound(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    marker = "\n… <history truncated> …\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    remaining = max_chars - len(marker)
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]
