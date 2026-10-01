"""Fit the fixed NIST Chwirut2 ultrasonic-calibration data."""

# EVOLVE-BLOCK-START
def solve(payload):
    """Refine NIST's Start 1 by unweighted nonlinear least squares."""
    import numpy as np
    from scipy.optimize import least_squares

    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)

    def residual(beta):
        return np.exp(-beta[0] * x) / (beta[1] + beta[2] * x) - y

    def jacobian(beta):
        numerator = np.exp(-beta[0] * x)
        denominator = beta[1] + beta[2] * x
        prediction = numerator / denominator
        denominator_derivative = -numerator / denominator**2
        return np.column_stack(
            (-x * prediction, denominator_derivative, x * denominator_derivative)
        )

    result = least_squares(
        residual,
        [0.1, 0.01, 0.02],
        jac=jacobian,
        method="lm",
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13,
        max_nfev=10000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x.tolist()
# EVOLVE-BLOCK-END
