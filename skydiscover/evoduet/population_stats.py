"""Deterministic, code-free statistics for the retained solution population."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def _field(program: object, name: str, default: Any = None) -> Any:
    return (
        program.get(name, default)
        if isinstance(program, Mapping)
        else getattr(program, name, default)
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _identifier(value: Any) -> str | None:
    # Do not stringify arbitrary objects: their representations can contain code.
    if isinstance(value, str):
        return value or None
    if not isinstance(value, bool) and isinstance(value, int):
        return str(value)
    return None


def _score(program: object, key: str) -> float | None:
    metrics = _field(program, "metrics")
    return _number(metrics.get(key)) if isinstance(metrics, Mapping) else None


def _quantile(scores: list[float], fraction: float) -> float:
    """Linear interpolation at (n - 1) * fraction, including small populations."""
    position = (len(scores) - 1) * fraction
    left = int(position)
    weight = position - left
    if weight == 0:
        return scores[left]
    return scores[left] * (1 - weight) + scores[left + 1] * weight


def _distribution(scores: list[float]) -> dict[str, float | None]:
    names = ("best", "mean", "population_std", "median", "q25", "q75", "worst")
    if not scores:
        return dict.fromkeys(names)
    ordered = sorted(scores)
    return dict(
        zip(
            names,
            (
                ordered[-1],
                statistics.mean(ordered),
                statistics.pstdev(ordered),
                _quantile(ordered, 0.5),
                _quantile(ordered, 0.25),
                _quantile(ordered, 0.75),
                ordered[0],
            ),
        )
    )


def _concentration(counts: Counter[str], eligible_programs: int) -> dict[str, Any]:
    total = sum(counts.values())
    most_id = min(counts, key=lambda key: (-counts[key], key)) if counts else None
    most_count = counts[most_id] if most_id is not None else 0
    return {
        "programs_with_selection": eligible_programs,
        "selection_count": total,
        "unique_ids": len(counts),
        "most_selected_id": most_id,
        "most_selected_count": most_count,
        "most_selected_program_fraction": (
            most_count / eligible_programs if eligible_programs else None
        ),
        "selection_hhi": sum((count / total) ** 2 for count in counts.values()) if total else None,
    }


def _encode(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def serialize_population_statistics(
    population: Iterable[object] | None,
    max_chars: int | None = 60_000,
    recent_k: int | None = None,
    *,
    parent: object | None = None,
) -> str:
    """Summarize retained members without reading code, metadata, prompts or artifacts.

    A finite ``evoduet_score`` anywhere selects that metric for every row;
    otherwise ``combined_score`` is used. Both are higher-is-better. Missing scores
    stay null. Quartiles use inclusive linear interpolation, standard deviation is
    the population standard deviation, and improvements use strict comparison.

    ``recent_k`` selects the latest K retained rows, with undated rows ordered first
    and IDs breaking iteration ties. Retained-record outcomes compare with preceding
    rows in that order, including rows outside the selected window. These are not
    the complete run history: failed and evicted attempts are unavailable.

    Character truncation drops oldest trace entries, then lowest-ranked top entries.
    Aggregate statistics keep their original scopes. A compact aggregate fallback is
    used for smaller budgets; its smallest form retains the metric, best, counts,
    scope and omission counts. An insufficient budget raises ValueError. ``None``
    disables the character cap. Empty populations return an empty string.
    """
    programs = list(population) if population is not None else []
    if not programs:
        return ""
    if recent_k is not None and (
        isinstance(recent_k, bool) or not isinstance(recent_k, int) or recent_k <= 0
    ):
        raise ValueError("recent_k must be a positive integer or None")
    if max_chars is not None and (
        isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0
    ):
        raise ValueError("max_chars must be a positive integer or None")

    score_key = (
        "evoduet_score"
        if any(_score(program, "evoduet_score") is not None for program in programs)
        else "combined_score"
    )
    rows = []
    for program in programs:
        contexts = _field(program, "other_context_ids")
        context_ids = (
            sorted(
                {identifier for value in contexts if (identifier := _identifier(value)) is not None}
            )
            if isinstance(contexts, (list, tuple, set, frozenset))
            else []
        )
        rows.append(
            {
                "id": _identifier(_field(program, "id")),
                "iteration": _number(_field(program, "iteration_found")),
                "parent_id": _identifier(_field(program, "parent_id")),
                "score": _score(program, score_key),
                "context_ids": context_ids,
            }
        )
    rows.sort(
        key=lambda row: (
            row["iteration"] is not None,
            row["iteration"] or 0,
            row["id"] or "",
            row["parent_id"] or "",
            row["score"] is not None,
            row["score"] or 0,
            row["context_ids"],
        )
    )
    by_id = {row["id"]: row for row in rows if row["id"] is not None}
    scores = [row["score"] for row in rows if row["score"] is not None]
    distribution = _distribution(scores)
    best = distribution["best"]
    parent_id = _identifier(_field(parent, "id")) if parent is not None else None
    parent_score = (
        by_id[parent_id]["score"]
        if parent_id in by_id
        else _score(parent, score_key) if parent is not None else None
    )
    current_parent = {
        "id": parent_id,
        "score": parent_score,
        "gap_to_retained_best": (
            _number(best - parent_score) if best is not None and parent_score is not None else None
        ),
        "score_source": (
            "retained_population"
            if parent_id in by_id
            else "provided_parent" if parent is not None else "unavailable"
        ),
    }

    trace = []
    best_before = None
    for row in rows:
        score = row["score"]
        previous = by_id.get(row["parent_id"])
        previous_score = previous["score"] if previous is not None else None
        if score is None:
            outcome = "missing_score"
        elif row["parent_id"] is None:
            outcome = "no_parent"
        elif previous_score is None:
            outcome = "missing_parent_score"
        else:
            outcome = (
                "improved"
                if score > previous_score
                else "regressed" if score < previous_score else "unchanged"
            )
        global_outcome = (
            "missing_score"
            if score is None
            else (
                "baseline"
                if best_before is None
                else "improved" if score > best_before else "not_improved"
            )
        )
        trace.append(
            {
                "id": row["id"],
                "iteration": row["iteration"],
                "parent_id": row["parent_id"],
                "score": score,
                "parent_score": previous_score,
                "delta": (
                    _number(score - previous_score)
                    if score is not None and previous_score is not None
                    else None
                ),
                "outcome": outcome,
                "retained_best_before": best_before,
                "global_outcome": global_outcome,
            }
        )
        if score is not None:
            best_before = score if best_before is None else max(best_before, score)

    selected_rows = rows[-recent_k:] if recent_k is not None else rows
    selected_trace = trace[-recent_k:] if recent_k is not None else trace
    parents = Counter(row["parent_id"] for row in selected_rows if row["parent_id"] is not None)
    contexts = Counter(identifier for row in selected_rows for identifier in row["context_ids"])
    top = [
        {"id": row["id"], "score": row["score"]}
        for row in sorted(
            (row for row in rows if row["score"] is not None),
            key=lambda row: (-row["score"], row["id"] or ""),
        )[:20]
    ]
    payload = {
        "schema": "population_statistics_v1",
        "scope": {
            "population": "All retained programs only; failed and evicted attempts are unavailable.",
            "trace_order": "Undated first, then iteration ascending, then id; ties do not establish actual execution order.",
            "global_outcome": "Strict improvement over preceding retained rows in trace_order, not complete run history.",
            "parent_score": "Same metric looked up across the entire retained population.",
            "score_statistics": "Finite selected-metric scores only; inclusive quartiles; population standard deviation.",
            "unique_scores": "Exact numeric uniqueness, not semantic or algorithmic diversity.",
            "missing_values": "null denotes unavailable or non-finite values, including arithmetic overflow.",
        },
        "score_key": score_key,
        "direction": "higher_is_better",
        "population_size": len(rows),
        "scored_population_size": len(scores),
        "missing_score_count": len(rows) - len(scores),
        "score_distribution": distribution,
        "unique_score_count": len(set(scores)),
        "current_parent": current_parent,
        "top_programs": top,
        "trace_window": {
            "requested_recent_k": recent_k,
            "selected_count": len(selected_trace),
            "first_iteration": selected_trace[0]["iteration"],
            "last_iteration": selected_trace[-1]["iteration"],
        },
        "recent_trace": selected_trace,
        "selection_concentration": {
            "scope": "Selected trace window before character truncation; context IDs counted once per program; HHI over selection slots.",
            "parent": _concentration(parents, sum(parents.values())),
            "context": _concentration(
                contexts, sum(bool(row["context_ids"]) for row in selected_rows)
            ),
        },
        "omitted": {
            "trace_by_recent_k": len(rows) - len(selected_trace),
            "trace_by_budget": 0,
            "top_beyond_20": len(scores) - len(top),
            "top_by_budget": 0,
        },
    }
    encoded = _encode(payload)
    if max_chars is None or len(encoded) <= max_chars:
        return encoded

    def limit_entries(count: int) -> str:
        top_count = min(count, len(top))
        trace_count = max(0, count - len(top))
        payload["top_programs"] = top[:top_count]
        payload["recent_trace"] = selected_trace[-trace_count:] if trace_count else []
        payload["omitted"]["trace_by_budget"] = len(selected_trace) - trace_count
        payload["omitted"]["top_by_budget"] = len(top) - top_count
        return _encode(payload)

    empty_entries = limit_entries(0)
    if len(empty_entries) <= max_chars:
        low, high, encoded = 0, len(top) + len(selected_trace), empty_entries
        while low <= high:
            count = (low + high) // 2
            candidate = limit_entries(count)
            if len(candidate) <= max_chars:
                encoded, low = candidate, count + 1
            else:
                high = count - 1
        return encoded

    compact = {
        "schema": "population_statistics_v1",
        "scope": "Retained population only; failed/evicted attempts unavailable. Finite scores; inclusive quartiles; population std.",
        "score_key": score_key,
        "direction": "higher_is_better",
        "population_size": len(rows),
        "scored_population_size": len(scores),
        "missing_score_count": len(rows) - len(scores),
        "score_distribution": distribution,
        "current_parent": current_parent,
        "omitted": dict(payload["omitted"], details=True),
    }
    encoded = _encode(compact)
    if len(encoded) > max_chars:
        del compact["current_parent"]
        compact["omitted"]["current_parent"] = True
        encoded = _encode(compact)
    if len(encoded) > max_chars:
        compact["scope"] = "Retained only; failed/evicted attempts unavailable."
        compact["score_distribution"] = {"best": best}
        compact["omitted"]["distribution_fields"] = len(distribution) - 1
        encoded = _encode(compact)
    if len(encoded) > max_chars:
        raise ValueError("max_chars is too small for population statistics")
    return encoded
