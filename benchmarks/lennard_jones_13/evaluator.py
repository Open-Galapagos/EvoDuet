"""Independently sum every atom-pair energy for the published LJ13 instance."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _public_optimization_runtime import run_candidate


REFERENCE_ENERGY_MAGNITUDE = 44.326801


def task_payload():
    """Public physical parameters only; no optimal structure or energy."""
    return {"n_atoms": 13, "epsilon": 1.0, "sigma": 1.0}


def score_coordinates(coordinates):
    if not isinstance(coordinates, list) or len(coordinates) != 13:
        raise ValueError("solve(payload) must return exactly 13 coordinate rows")
    points = []
    for row in coordinates:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Each coordinate row must contain exactly three numbers")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
            raise ValueError("Coordinates must be finite numbers, excluding bool")
        points.append(tuple(map(float, row)))
    pair_energies = []
    minimum_distance_squared = math.inf
    for i, point in enumerate(points):
        for other in points[:i]:
            distance_squared = math.fsum((x - y) ** 2 for x, y in zip(point, other))
            if not math.isfinite(distance_squared) or distance_squared <= 0.0:
                raise ValueError("Every pair must have a finite, strictly positive squared distance")
            minimum_distance_squared = min(minimum_distance_squared, distance_squared)
            inverse_sixth = distance_squared ** -3
            pair_energies.append(4.0 * (inverse_sixth**2 - inverse_sixth))
    energy = math.fsum(pair_energies)
    if not math.isfinite(energy):
        raise ValueError("The independently computed energy must be finite")
    return {
        "combined_score": min(1.0, max(0.0, -energy) / REFERENCE_ENERGY_MAGNITUDE),
        "validity": 1.0,
        "energy": energy,
        "energy_gap": max(0.0, energy + REFERENCE_ENERGY_MAGNITUDE),
        "minimum_pair_distance": math.sqrt(minimum_distance_squared),
    }


def evaluate(program_path):
    try:
        coordinates = run_candidate(program_path, task_payload(), timeout=30)
        return score_coordinates(coordinates)
    except (ValueError, TypeError, TimeoutError, OverflowError, ArithmeticError) as error:
        return {"combined_score": 0.0, "validity": 0.0, "error": str(error)}


evaluate_final = evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
