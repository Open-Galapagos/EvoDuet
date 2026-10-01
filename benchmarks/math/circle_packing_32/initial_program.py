# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=32 circles."""

import numpy as np


def construct_packing():
    """Construct a valid starting arrangement of 32 circles in a unit square.

    Returns:
        centers: Array with shape (32, 2).
        radii: Array with shape (32,).
        sum_radii: Sum of the 32 radii.
    """
    # Six staggered rows give a simple, deterministic hexagonal seed. The last
    # row contains one fewer circle so the total is exactly 32.
    row_counts = (5, 6, 5, 6, 5, 5)
    radius = (1.0 / 12.0) - 1e-6
    vertical_spacing = np.sqrt(3.0) * radius
    first_y = 0.5 - 2.5 * vertical_spacing

    centers = []
    for row, count in enumerate(row_counts):
        # Alternate rows are offset by one radius. This makes adjacent centers
        # exactly two radii apart in the ideal hexagonal lattice.
        first_x = radius if row % 2 else 2.0 * radius
        y = first_y + row * vertical_spacing
        for column in range(count):
            centers.append((first_x + 2.0 * radius * column, y))

    centers = np.asarray(centers, dtype=float)
    radii = np.full(32, radius, dtype=float)
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


def run_packing():
    """Run the n=32 circle-packing constructor."""
    return construct_packing()


def visualize(centers, radii):
    """Visualize a returned packing."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    for index, (center, radius) in enumerate(zip(centers, radii)):
        ax.add_patch(Circle(center, radius, alpha=0.5))
        ax.text(center[0], center[1], str(index), ha="center", va="center")
    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    packing_centers, packing_radii, total = run_packing()
    print(f"Sum of radii: {total}")
