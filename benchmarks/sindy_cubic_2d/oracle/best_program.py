"""Fit a dynamical model from the supplied public time-series observations."""

import numpy as np


# EVOLVE-BLOCK-START
def solve(payload):
    t = np.asarray(payload["t"], dtype=float)
    x = np.asarray(payload["x"], dtype=float)
    powers = np.asarray(payload["powers"], dtype=int)
    dt = float(t[1] - t[0])
    # Fourth-order centered differentiation uses observed states only.
    derivatives = (x[:-4] - 8*x[1:-3] + 8*x[3:-1] - x[4:]) / (12*dt)
    state = x[2:-2]
    library = state[:, 0, None]**powers[:, 0] * state[:, 1, None]**powers[:, 1]
    coefficients = np.linalg.lstsq(library, derivatives, rcond=None)[0]
    # Sequentially remove small coefficients, then refit surviving terms.
    for _ in range(20):
        previous = np.abs(coefficients) >= 0.05
        for target in range(2):
            active = previous[:, target]
            coefficients[:, target] = 0.0
            if np.any(active):
                coefficients[active, target] = np.linalg.lstsq(
                    library[:, active], derivatives[:, target], rcond=None
                )[0]
        if np.array_equal(previous, np.abs(coefficients) >= 0.05):
            break
    return {"coefficients": coefficients.T.tolist()}
# EVOLVE-BLOCK-END
