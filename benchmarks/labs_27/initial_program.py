"""Construct a length-27 nonperiodic low-autocorrelation binary sequence."""

# EVOLVE-BLOCK-START
def solve(payload):
    """Return a JSON list containing payload['n'] values in {-1, +1}."""
    import random

    n = payload["n"]
    rng = random.Random(0)

    def energy(sequence):
        return sum(
            sum(sequence[i] * sequence[i + k] for i in range(n - k)) ** 2
            for k in range(1, n)
        )

    samples = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(32)]
    return min(samples, key=energy)
# EVOLVE-BLOCK-END
