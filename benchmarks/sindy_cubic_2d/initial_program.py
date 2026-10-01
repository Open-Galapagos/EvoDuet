"""Fit a dynamical model from the supplied public time-series observations."""

import numpy as np


# EVOLVE-BLOCK-START
def solve(payload):
    """Return a 2 x 10 coefficient matrix in payload['powers'] order."""
    t = np.asarray(payload["t"], dtype=float)
    x = np.asarray(payload["x"], dtype=float)
    derivatives = np.gradient(x, t, axis=0, edge_order=2)
    library = np.column_stack((np.ones(len(x)), x))
    fitted = np.linalg.lstsq(library[1:-1], derivatives[1:-1], rcond=None)[0]
    coefficients = np.zeros((2, len(payload["powers"])))
    coefficients[:, :3] = fitted.T
    return {"coefficients": coefficients.tolist()}
# EVOLVE-BLOCK-END
