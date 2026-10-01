"""Independently integrate and grade a bounded temperature profile."""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent))
from _public_optimization_runtime import run_candidate

TIME = np.linspace(0.0, 1.0, 501)


def make_payload():
    return {"time": TIME.tolist(), "initial_state": [1.0, 0.0, 0.0],
            "temperature_bounds": [298.0, 398.0],
            "interpolation": "piecewise_linear",
            "kinetics": {"k1_prefactor": 4000.0, "k1_activation_over_R": 2500.0,
                         "k2_prefactor": 620000.0, "k2_activation_over_R": 5000.0}}


def grade_artifact(artifact):
    if not isinstance(artifact, dict) or set(artifact) != {"temperature"}:
        raise ValueError("Return exactly a temperature object")
    raw = artifact["temperature"]
    if type(raw) is not list or len(raw) != len(TIME):
        raise ValueError("temperature must be a JSON array with 501 numbers")
    for value in raw:
        if type(value) not in (int, float):
            raise ValueError("temperature must contain JSON numbers, not strings or booleans")
        if not 298.0 <= value <= 398.0:
            raise ValueError("temperature must stay within [298,398] kelvin")
        if not math.isfinite(value):
            raise ValueError("temperature must be finite")
    temperature = np.asarray(raw, dtype=float)

    state = np.array([1.0, 0.0, 0.0])
    # Integrate each interval separately so every interpolation knot is honored.
    # The candidate's objective values and state estimates are never consumed.
    for index in range(len(TIME)-1):
        start, end = TIME[index:index+2]
        first, last = temperature[index:index+2]

        def rhs(t, x):
            temp = first + (last-first)*(t-start)/(end-start)
            k1 = 4000.0*np.exp(-2500.0/temp)
            k2 = 620000.0*np.exp(-5000.0/temp)
            r1, r2 = k1*x[0]**2, k2*x[1]
            return [-r1, r1-r2, r2]

        integrated = solve_ivp(rhs, (start, end), state, method="DOP853",
                               rtol=1e-10, atol=1e-12)
        if not integrated.success or not np.all(np.isfinite(integrated.y)):
            raise ValueError("Reactor integration failed")
        state = integrated.y[:, -1]
    if np.min(state) < -1e-9 or abs(float(np.sum(state))-1.0) > 1e-8:
        raise ValueError("Reactor concentration or mass-balance validation failed")
    return {"validity": 1.0, "combined_score": float(state[1]),
            "final_B": float(state[1]), "final_A": float(state[0]),
            "final_C": float(state[2]),
            "mass_balance_error": abs(float(np.sum(state))-1.0),
            "temperature_min": float(np.min(temperature)),
            "temperature_max": float(np.max(temperature))}


def evaluate(program_path):
    try:
        return grade_artifact(run_candidate(program_path, make_payload(), timeout=45))
    except Exception as exc:
        return {"validity": 0.0, "combined_score": 0.0,
                "error": f"{type(exc).__name__}: {exc}"[:500]}


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
