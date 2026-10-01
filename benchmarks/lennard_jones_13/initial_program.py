"""Find 13 atomic coordinates that minimize the reduced Lennard-Jones energy.

Return a JSON-compatible list of 13 [x, y, z] rows from solve(payload).
"""

import math


# EVOLVE-BLOCK-START
def solve(payload):
    """An open chain with neighboring atoms at the pair equilibrium distance."""
    n_atoms = payload["n_atoms"]
    spacing = payload["sigma"] * 2.0 ** (1.0 / 6.0)
    return [[(index - (n_atoms - 1) / 2.0) * spacing, 0.0, 0.0]
            for index in range(n_atoms)]
# EVOLVE-BLOCK-END
