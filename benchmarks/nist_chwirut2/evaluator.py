"""Recompute unweighted residual error from coefficients and public observations."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _public_optimization_runtime import run_candidate

DATA_PATH = Path(__file__).resolve().with_name("data.json")
CERTIFIED_RSS = 513.04802941  # NIST's published RSS; used only to normalize score.


def evaluate(program_path: str) -> dict:
    started = time.monotonic()
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        x = data["x"]
        y = data["y"]
        if len(x) != 54 or len(y) != 54:
            raise ValueError("The public Chwirut2 data must contain 54 observations")
        beta = run_candidate(program_path, {"x": list(x), "y": list(y)}, timeout=30)
        if not isinstance(beta, list) or len(beta) != 3:
            raise ValueError("solve(payload) must return [b1, b2, b3]")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in beta
        ):
            raise ValueError("Coefficients must be finite real numbers, excluding booleans")
        b1, b2, b3 = beta
        squared_residuals = []
        for predictor, observed in zip(x, y):
            prediction = math.exp(-b1 * predictor) / (b2 + b3 * predictor)
            if not math.isfinite(prediction):
                raise ValueError("The model predictions must be finite")
            squared_residuals.append((prediction - observed) ** 2)
        rss = math.fsum(squared_residuals)
        if not math.isfinite(rss):
            raise ValueError("Residual sum of squares must be finite")
        return {
            "combined_score": min(1.0, CERTIFIED_RSS / rss) if rss > 0 else 1.0,
            "validity": 1.0,
            "rss": rss,
            "rmse": math.sqrt(rss / len(x)),
            "n_observations": len(x),
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
    print(json.dumps(evaluate(sys.argv[1]), allow_nan=False))
