"""Return a feasible design for the published four-variable pressure vessel task.

The fixed task data are supplied to solve(payload). Return shell thickness,
head thickness, inner radius, and cylindrical length, all in inches.
"""

import math


# EVOLVE-BLOCK-START
def solve(payload):
    """A conservative feasible vessel with excess material and capacity."""
    return [1.25, 0.625, 50.0, 150.0]
# EVOLVE-BLOCK-END
