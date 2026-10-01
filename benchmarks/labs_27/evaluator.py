"""Independently score the returned sequence on the fixed CSPLib LABS task."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _public_optimization_runtime import run_candidate

N = 27
REFERENCE_ENERGY = 37  # Packebusch & Mertens (2016), Table 1; score normalization.


def evaluate(program_path: str) -> dict:
    started = time.monotonic()
    try:
        sequence = run_candidate(program_path, {"n": N}, timeout=30)
        if not isinstance(sequence, list) or len(sequence) != N:
            raise ValueError("solve(payload) must return a list of exactly 27 signs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value not in (-1, 1)
            for value in sequence
        ):
            raise ValueError("Every sequence entry must be the number -1 or +1")
        signs = [int(value) for value in sequence]
        energy = sum(
            sum(signs[i] * signs[i + lag] for i in range(N - lag)) ** 2
            for lag in range(1, N)
        )
        return {
            "combined_score": min(1.0, REFERENCE_ENERGY / energy),
            "validity": 1.0,
            "energy": energy,
            "merit_factor": N * N / (2 * energy),
            "evaluation_time": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "evaluation_time": time.monotonic() - started,
        }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(sys.argv[1]), allow_nan=False))
