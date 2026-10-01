"""Fit the fixed NIST Chwirut2 ultrasonic-calibration data."""

# EVOLVE-BLOCK-START
def solve(payload):
    """Return [b1, b2, b3] for exp(-b1*x)/(b2+b3*x)."""
    return [0.1, 0.01, 0.02]  # NIST's published Start 1.
# EVOLVE-BLOCK-END
