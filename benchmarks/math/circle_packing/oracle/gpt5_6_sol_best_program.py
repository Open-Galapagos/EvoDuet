# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""
import numpy as np


def construct_packing():
    """Polish explicit layered topologies, refine diverse elites, and certify radii by LP."""
    from scipy.optimize import linprog, minimize

    grid = np.array(
        [[0.1 + 0.2 * x, 0.1 + 0.2 * y] for y in range(5) for x in range(5)]
        + [[0.2, 0.2]], dtype=float)
    pairs = np.array([(i, j) for i in range(26) for j in range(i + 1, 26)])
    A = np.zeros((len(pairs), 26))
    A[np.arange(len(pairs)), pairs[:, 0]] = 1
    A[np.arange(len(pairs)), pairs[:, 1]] = 1

    def certified(p):
        border = np.min(np.c_[p, 1 - p], axis=1)
        distance = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
        if np.min(border) <= 1e-7 or np.min(distance) <= 2e-7:
            return None
        q = linprog(-np.ones(26), A_ub=A, b_ub=distance,
                    bounds=[(1e-8, b) for b in border], method="highs")
        if not q.success:
            return None
        r = q.x
        scale = min(1.0, np.min(border / r),
                    np.min(distance / (r[pairs[:, 0]] + r[pairs[:, 1]])))
        return r * scale * (1 - 1e-10)

    def constraint(z):
        p, r = z[:52].reshape(26, 2), z[52:]
        d = p[pairs[:, 0]] - p[pairs[:, 1]]
        return np.r_[p[:, 0] - r, p[:, 1] - r,
                     1 - p[:, 0] - r, 1 - p[:, 1] - r,
                     np.linalg.norm(d, axis=1)
                     - r[pairs[:, 0]] - r[pairs[:, 1]]]

    def layout(counts, stagger=False):
        """Place prescribed row populations, optionally with alternating offsets."""
        m = len(counts)
        p = np.array([[(x + .5) / k, (y + .5) / m]
                      for y, k in enumerate(counts) for x in range(k)])
        if stagger:
            p[:, 0] += np.array([
                .025 if y & 1 else -.025
                for y, k in enumerate(counts) for _ in range(k)])
        return np.clip(p, .015, .985)

    patterns = []
    for row in range(5):
        counts = [5] * 5
        counts[row] = 6
        patterns.append(counts)
    patterns += [
        [6, 4, 6, 4, 6], [4, 6, 4, 6, 6],
        [5, 4, 4, 5, 4, 4], [4, 5, 4, 4, 5, 4],
        [4, 4, 5, 4, 4, 5]]
    bases = [grid]
    for counts in patterns:
        p = layout(counts)
        q = layout(counts, True)
        bases += [p, q, q[:, ::-1].copy()]

    rng = np.random.default_rng(26)
    seeds = bases + [
        np.clip(p + rng.normal(0, s, p.shape), .015, .985)
        for p in bases for s in (.006, .016, .035)]
    candidates = [(grid, certified(grid))]

    def polish(p, rounds=2, iterations=1200, tolerance=1e-11):
        """Alternate SLSQP center updates with exact fixed-center radius LPs."""
        for _ in range(rounds):
            r = certified(p)
            if r is None:
                return
            z = minimize(lambda v: -np.sum(v[52:]), np.r_[p.ravel(), r],
                         method="SLSQP",
                         bounds=[(1e-6, 1 - 1e-6)] * 52
                         + [(1e-8, 0.25)] * 26,
                         constraints={"type": "ineq", "fun": constraint},
                         options={"maxiter": iterations, "ftol": tolerance}).x
            if not np.all(np.isfinite(z)):
                return
            p = z[:52].reshape(26, 2)
            r = certified(p)
            if r is not None:
                candidates.append((p.copy(), r))

    for p in seeds:
        polish(p)

    # Explore several strong basins instead of repeatedly polishing one fixed point.
    elites = sorted(candidates, key=lambda item: np.sum(item[1]),
                    reverse=True)[:4]
    elite_rng = np.random.default_rng(260)
    for base, _ in elites:
        for scale in (.0015, .004, .01, .025):
            for _ in range(2):
                p = np.clip(base + elite_rng.normal(0, scale, base.shape),
                            .01, .99)
                polish(p, 2, 1800, 5e-13)

    centers, radii = max(
        ((p, r) for p, r in candidates if r is not None),
        key=lambda item: np.sum(item[1]))
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
