"""Frozen Summit one-hot, yield-only Baumgartner emulator, evaluated with NumPy.

The released five networks and scalers are preserved. Each member's inverse-
scaled yield is clipped to [0, 1] BEFORE the ensemble mean, as in Summit.
No model is fitted, and no evaluator input comes from oracle/.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _public_optimization_runtime import run_candidate

CATALYSTS = ("tBuXPhos", "tBuBrettPhos", "AlPhos")
BASES = ("DBU", "BTMG", "TMG", "TEA")
BOUNDS = {"base_equivalents": (1.0, 2.5), "temperature": (30.0, 100.0),
          "t_res": (60.0, 1800.0)}


def task_payload():
    return {"catalysts": list(CATALYSTS), "bases": list(BASES),
            "bounds": {key: list(value) for key, value in BOUNDS.items()},
            "objective": "maximize ensemble predicted yield",
            "units": {"temperature": "degree Celsius", "t_res": "seconds",
                      "base_equivalents": "equivalents"}}


@lru_cache(maxsize=1)
def _model():
    config = json.loads((HERE / "data/model.json").read_text())
    with np.load(HERE / "data/weights.npz", allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    return config["experiment_params"]["predictors"], weights


def _validate(design):
    if type(design) is not dict or set(design) != {"catalyst", "base", *BOUNDS}:
        raise ValueError("Return exactly catalyst, base, base_equivalents, temperature, t_res")
    if type(design["catalyst"]) is not str or design["catalyst"] not in CATALYSTS:
        raise ValueError("Unknown catalyst")
    if type(design["base"]) is not str or design["base"] not in BASES:
        raise ValueError("Unknown base")
    for key, (lower, upper) in BOUNDS.items():
        value = design[key]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number, excluding booleans")
        if not lower <= value <= upper:
            raise ValueError(f"{key} is outside [{lower}, {upper}]")


def predict_members(design):
    """Return raw and clipped member yields; float32 matches the released ANN."""
    _validate(design)
    parameters, weights = _model()
    numeric = np.array([design[key] for key in BOUNDS], dtype=np.float64)
    categorical = [float(design["catalyst"] == value) for value in CATALYSTS]
    categorical += [float(design["base"] == value) for value in BASES]
    predictions = []
    for index, parameter in enumerate(parameters):
        scaler = parameter["input_preprocessor"]["num"]
        scaled = (numeric - scaler["mean_"]) / scaler["scale_"]
        x = np.concatenate((scaled, categorical)).astype(np.float32)
        hidden = np.maximum(weights[f"{index}.input_layer.weight"] @ x
                            + weights[f"{index}.input_layer.bias"], 0)
        y = weights[f"{index}.output_layer.weight"] @ hidden + weights[f"{index}.output_layer.bias"]
        # StandardScaler.inverse_transform uses in-place operations, retaining float32.
        y *= parameter["output_preprocessor"]["scale_"]
        y += parameter["output_preprocessor"]["mean_"]
        predictions.append(y[0])
    raw = np.asarray(predictions, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Emulator produced non-finite output")
    return raw, np.clip(raw, 0.0, 1.0)


def score_design(design):
    raw, clipped = predict_members(design)
    mean = float(clipped.mean())
    return {"combined_score": mean, "validity": 1.0,
            "predicted_yield": mean, "ensemble_std": float(clipped.std()),
            "unclipped_mean_yield": float(raw.mean()),
            "clipped_members": int(np.count_nonzero(raw != clipped))}


def evaluate(program_path):
    try:
        return score_design(run_candidate(program_path, task_payload(), timeout=30))
    except (ValueError, TypeError, TimeoutError, ArithmeticError) as error:
        return {"combined_score": 0.0, "validity": 0.0, "error": str(error)}


evaluate_final = evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
