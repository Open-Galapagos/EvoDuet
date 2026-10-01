# EVOLVE-BLOCK-START
"""
Parallel Scaling Law for language models (Chen et al., 2025):
Effective parameter count scales as N_eff = N * (1 + 0.4 * log2(P)).
The loss follows a 4-parameter basis with power-law exponent -0.2:
Loss(N, P) = b0 + b1 * N^(-0.2) + b2 * (1 + 0.4*log2(P))^(-0.2) + b3 * N_eff^(-0.2).
"""
import numpy as np

def _design(data_points):
    X = np.atleast_2d(np.asarray(data_points, dtype=float))
    u = np.maximum(X[:, 0] * 1e-9, 1e-6) ** -0.2
    v = (1.0 + 0.4 * np.log2(np.maximum(X[:, 1], 1.0))) ** -0.2
    return np.column_stack([np.ones(len(X)), u, v, u * v])

def scaling_law_func(data_points, params):
    A = _design(data_points)
    p = np.asarray(params, dtype=float)
    if p.ndim == 1:
        return A @ p
    return A @ (p.T if p.shape[-1] == 4 else p)

def fit_scaling_law(data_points, loss_values):
    A = _design(data_points)
    y = np.asarray(loss_values, dtype=float)
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    return p.T if y.ndim > 1 else p
# EVOLVE-BLOCK-END