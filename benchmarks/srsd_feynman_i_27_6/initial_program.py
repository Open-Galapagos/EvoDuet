"""Discover a scalar expression from the supplied SRSD training observations."""


# EVOLVE-BLOCK-START
def solve(payload):
    import numpy as np

    x = np.asarray(payload["x_train"], dtype=float)
    y = np.asarray(payload["y_train"], dtype=float)
    # A feasible one-variable model, fitted with the benchmark's relative loss.
    ratio = x[:, 0]/y
    coefficient = float(np.sum(ratio)/np.dot(ratio, ratio))
    return {"expression": f"({coefficient:.17g})*x0"}
# EVOLVE-BLOCK-END
