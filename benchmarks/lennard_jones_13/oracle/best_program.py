"""Find 13 atomic coordinates that minimize the reduced Lennard-Jones energy.

Return a JSON-compatible list of 13 [x, y, z] rows from solve(payload).
"""

import math


# EVOLVE-BLOCK-START
def solve(payload):
    """Construct the LJ13 icosahedron and minimize its single scale parameter."""
    if payload["n_atoms"] != 13:
        raise ValueError("This fixed-instance reference is for 13 atoms")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    points = [[0.0, 0.0, 0.0]]
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            points.extend([[0.0, a, b * phi], [a, b * phi, 0.0], [b * phi, 0.0, a]])
    inverse_sixths = []
    for i, point in enumerate(points):
        for other in points[:i]:
            distance_squared = sum((x - y) ** 2 for x, y in zip(point, other))
            inverse_sixths.append(distance_squared ** -3)
    attraction = math.fsum(inverse_sixths)
    repulsion = math.fsum(value * value for value in inverse_sixths)
    # E(s) = 4 epsilon (A (sigma/s)^12 - B (sigma/s)^6).
    scale = payload["sigma"] * (2.0 * repulsion / attraction) ** (1.0 / 6.0)
    return [[scale * coordinate for coordinate in point] for point in points]
# EVOLVE-BLOCK-END
