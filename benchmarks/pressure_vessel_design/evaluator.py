"""Evaluate the fixed discrete-thickness pressure-vessel benchmark independently."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _public_optimization_runtime import run_candidate


REFERENCE_COST = 6059.714335048436
FEASIBILITY_TOLERANCE = 1e-9


def task_payload():
    """Public problem data only; no reference design or target objective."""
    return {
        "thickness_step": 0.0625,
        "thickness_index_bounds": [1, 99],
        "radius_bounds": [10.0, 200.0],
        "length_bounds": [10.0, 200.0],
        "length_constraint_upper": 240.0,
        "minimum_volume": 1296000.0,
        "stress_coefficients": [0.0193, 0.00954],
        "cost_coefficients": [0.6224, 1.7781, 3.1661, 19.84],
    }


def score_design(design):
    if not isinstance(design, list) or len(design) != 4:
        raise ValueError("solve(payload) must return [shell, head, radius, length]")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in design):
        raise ValueError("Every design variable must be a finite number, excluding bool")
    shell, head, radius, length = map(float, design)
    if any(value <= 0.0 for value in design):
        raise ValueError("Design variables must be positive")
    step = 0.0625
    for thickness in (shell, head):
        index = thickness / step
        if abs(index - round(index)) > FEASIBILITY_TOLERANCE:
            raise ValueError("Thicknesses must be integer multiples of 0.0625 inches")
    volume = math.pi * radius**2 * length + 4.0 * math.pi * radius**3 / 3.0
    # Normalize only the volume residual. Other residuals are measured in inches.
    violations = [
        step - shell, shell - 99.0 * step,
        step - head, head - 99.0 * step,
        10.0 - radius, radius - 200.0,
        10.0 - length, length - 200.0,
        0.0193 * radius - shell,
        0.00954 * radius - head,
        (1296000.0 - volume) / 1296000.0,
        length - 240.0,
    ]
    max_violation = max(0.0, *violations)
    if not math.isfinite(volume) or max_violation > FEASIBILITY_TOLERANCE:
        raise ValueError(f"Infeasible design; maximum normalized violation={max_violation:.12g}")
    cost = (
        0.6224 * shell * radius * length
        + 1.7781 * head * radius**2
        + 3.1661 * shell**2 * length
        + 19.84 * shell**2 * radius
    )
    if not math.isfinite(cost) or cost <= 0.0:
        raise ValueError("The independently computed cost is not finite and positive")
    return {
        "combined_score": min(1.0, REFERENCE_COST / cost),
        "validity": 1.0,
        "cost": cost,
        "volume": volume,
        "relative_cost_gap": max(0.0, cost / REFERENCE_COST - 1.0),
        "max_constraint_violation": max_violation,
    }


def evaluate(program_path):
    try:
        design = run_candidate(program_path, task_payload(), timeout=30)
        return score_design(design)
    except (ValueError, TypeError, TimeoutError, OverflowError, ArithmeticError) as error:
        return {"combined_score": 0.0, "validity": 0.0, "error": str(error)}


evaluate_final = evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
