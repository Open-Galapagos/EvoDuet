"""Regenerate the published, noiseless PySINDy cubic-oscillator observations."""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def generate():
    t = np.arange(0.0, 25.0, 0.01)

    def rhs(_, state):
        x, y = state
        return [-0.1 * x**3 + 2.0 * y**3, -2.0 * x**3 - 0.1 * y**3]

    solution = solve_ivp(
        rhs, (t[0], t[-1]), [2.0, 0.0], t_eval=t,
        method="LSODA", rtol=1e-12, atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return {"t": t.tolist(), "x": solution.y.T.tolist()}


if __name__ == "__main__":
    path = Path(__file__).with_name("training_data.json")
    path.write_text(json.dumps(generate(), separators=(",", ":")) + "\n")
    print(path)
