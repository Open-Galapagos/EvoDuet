"""Independent scoring of a submitted polynomial vector field."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent))
from _public_optimization_runtime import run_candidate

POWERS = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2],
                   [3, 0], [2, 1], [1, 2], [0, 3]], dtype=int)
VALIDATION_INITIAL_STATES = [[1.5, 0.5], [-1.0, 0.75]]


def make_payload():
    payload = json.loads((TASK_DIR / "training_data.json").read_text())
    payload["powers"] = POWERS.tolist()
    return payload


def _true_rhs(_, state):
    x, y = state
    return [-0.1*x**3 + 2.0*y**3, -2.0*x**3 - 0.1*y**3]


def _features(state):
    state = np.asarray(state)
    return state[..., 0, None]**POWERS[:, 0] * state[..., 1, None]**POWERS[:, 1]


def grade_artifact(artifact):
    if not isinstance(artifact, dict) or set(artifact) != {"coefficients"}:
        raise ValueError("Return exactly a coefficients object")
    raw = artifact["coefficients"]
    if type(raw) is not list or len(raw) != 2 or any(
        type(row) is not list or len(row) != 10 for row in raw
    ):
        raise ValueError("coefficients must be a 2 x 10 JSON array")
    for row in raw:
        for value in row:
            if type(value) not in (int, float):
                raise ValueError("coefficients must contain JSON numbers, not strings or booleans")
            if abs(value) > 1e6:
                raise ValueError("coefficient magnitude exceeds the public 1e6 limit")
            if not math.isfinite(value):
                raise ValueError("coefficients must be finite")
    coefficients = np.asarray(raw, dtype=float)

    axis = np.linspace(-2.0, 2.0, 25)
    states = np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1, 2)
    truth = np.asarray([_true_rhs(0, x) for x in states])
    prediction = _features(states) @ coefficients.T
    field_nrmse = float(np.sqrt(np.mean((prediction-truth)**2) / np.mean(truth**2)))

    evaluations = 0

    def predicted_rhs(_, state):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 50000:
            raise ValueError("Submitted dynamics exceeded 50000 RHS evaluations per trajectory")
        return _features(state) @ coefficients.T

    def runaway(_, state):
        return 100.0 - float(np.max(np.abs(state)))

    runaway.terminal = True
    runaway.direction = -1
    times = np.linspace(0.0, 10.0, 501)
    trajectory_errors = []
    for initial in VALIDATION_INITIAL_STATES:
        evaluations = 0
        reference = solve_ivp(_true_rhs, (0.0, 10.0), initial, t_eval=times,
                              method="DOP853", rtol=1e-10, atol=1e-12)
        prediction = solve_ivp(predicted_rhs, (0.0, 10.0), initial, t_eval=times,
                               events=runaway, method="DOP853", max_step=0.1,
                               rtol=1e-8, atol=1e-10)
        if (not reference.success or not prediction.success
                or prediction.y.shape != reference.y.shape
                or not np.all(np.isfinite(prediction.y))):
            raise ValueError("Submitted dynamics did not produce a bounded full trajectory")
        trajectory_errors.append(float(np.sqrt(
            np.mean((prediction.y-reference.y)**2) / np.mean(reference.y**2))))

    trajectory_nrmse = float(np.mean(trajectory_errors))
    score = 1.0 / (1.0 + 0.5*(field_nrmse + trajectory_nrmse))
    return {"validity": 1.0, "combined_score": score,
            "vector_field_nrmse": field_nrmse, "trajectory_nrmse": trajectory_nrmse,
            "active_coefficients": int(np.count_nonzero(np.abs(coefficients) > 1e-8))}


def evaluate(program_path):
    try:
        return grade_artifact(run_candidate(program_path, make_payload(), timeout=30))
    except Exception as exc:
        return {"validity": 0.0, "combined_score": 0.0,
                "error": f"{type(exc).__name__}: {exc}"[:500]}


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
