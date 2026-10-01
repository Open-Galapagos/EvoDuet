"""Return a feasible design for the published four-variable pressure vessel task.

The fixed task data are supplied to solve(payload). Return shell thickness,
head thickness, inner radius, and cylindrical length, all in inches.
"""

import math


# EVOLVE-BLOCK-START
def solve(payload):
    """Apply the active-constraint construction for the published instance."""
    volume = payload["minimum_volume"]
    max_length = payload["length_bounds"][1]
    lo, hi = payload["radius_bounds"]
    # The maximum allowed length gives the smallest radius with enough volume.
    for _ in range(100):
        radius = (lo + hi) / 2.0
        capacity = math.pi * radius**2 * max_length + 4.0 * math.pi * radius**3 / 3.0
        if capacity < volume:
            lo = radius
        else:
            hi = radius
    step = payload["thickness_step"]
    shell_stress, head_stress = payload["stress_coefficients"]
    shell = math.ceil(shell_stress * hi / step) * step
    head = math.ceil(head_stress * hi / step) * step
    radius = min(shell / shell_stress, head / head_stress)
    length = volume / (math.pi * radius**2) - 4.0 * radius / 3.0
    return [shell, head, radius, length]
# EVOLVE-BLOCK-END
