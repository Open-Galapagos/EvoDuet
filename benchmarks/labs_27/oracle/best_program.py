"""Construct a length-27 nonperiodic low-autocorrelation binary sequence."""

# EVOLVE-BLOCK-START
def solve(payload):
    """Enumerate the skew-symmetric subclass; it contains an optimum for n=27."""
    import numpy as np

    n = payload["n"]
    if n != 27:
        raise ValueError("This reference solves the published fixed n=27 task.")
    midpoint = n // 2
    free_bits = midpoint + 1
    masks = np.arange(1 << free_bits, dtype=np.uint32)
    left = 2 * ((masks[:, None] >> np.arange(free_bits)) & 1).astype(np.int64) - 1
    sequences = np.empty((len(masks), n), dtype=np.int64)
    sequences[:, :free_bits] = left
    for offset in range(1, midpoint + 1):
        sequences[:, midpoint + offset] = (-1) ** offset * sequences[:, midpoint - offset]
    energies = np.zeros(len(masks), dtype=np.int64)
    for lag in range(1, n):
        correlations = np.sum(sequences[:, :-lag] * sequences[:, lag:], axis=1)
        energies += correlations * correlations
    return sequences[int(np.argmin(energies))].tolist()
# EVOLVE-BLOCK-END
