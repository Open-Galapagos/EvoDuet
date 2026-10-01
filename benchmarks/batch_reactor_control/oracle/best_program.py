"""Produce a bounded temperature policy for the published batch reactor."""


# EVOLVE-BLOCK-START
def solve(payload):
    import numpy as np
    from scipy.optimize import minimize

    time = np.asarray(payload["time"], dtype=float)
    lower, upper = payload["temperature_bounds"]
    kinetics = payload["kinetics"]
    knots = np.linspace(time[0], time[-1], 51)
    # Integral solution of A'=-k1*A^2 and the linear equation for B.
    # Midpoint quadrature makes repeated objective evaluations inexpensive.
    n_quad = 4000
    dt = (time[-1] - time[0]) / n_quad
    midpoints = time[0] + (np.arange(n_quad) + 0.5) * dt

    def negative_yield(control):
        temperature = lower + (upper-lower)*np.interp(midpoints, knots, control)
        k1 = kinetics["k1_prefactor"] * np.exp(-kinetics["k1_activation_over_R"]/temperature)
        k2 = kinetics["k2_prefactor"] * np.exp(-kinetics["k2_activation_over_R"]/temperature)
        cumulative_k1 = (np.cumsum(k1)-0.5*k1)*dt
        remaining_k2 = (np.cumsum(k2[::-1])[::-1]-0.5*k2)*dt
        a = 1.0/(1.0/payload["initial_state"][0] + cumulative_k1)
        final_b = np.sum(k1*a*a*np.exp(-remaining_k2))*dt
        final_b += payload["initial_state"][1]*np.exp(-np.sum(k2)*dt)
        return -float(final_b)

    result = minimize(negative_yield, np.linspace(0.85, 0.25, len(knots)),
                      method="L-BFGS-B", bounds=[(0.0, 1.0)]*len(knots),
                      options={"maxiter": 350, "ftol": 1e-13, "gtol": 1e-8,
                               "maxfun": 25000, "maxls": 30})
    temperature = lower + (upper-lower)*np.interp(time, knots, result.x)
    return {"temperature": temperature.tolist()}
# EVOLVE-BLOCK-END
