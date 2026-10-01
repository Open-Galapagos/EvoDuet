# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=32 circles."""

import numpy as np


def construct_packing():
    """Optimize staggered hexagonal seeds, then alternate joint refinement and radius LP."""
    row_counts = (5, 6, 5, 6, 5, 5)
    seed_radius = 0.090
    dy = np.sqrt(3.0) * seed_radius
    first_y = 0.5 - 2.5 * dy

    centers = []
    for row, count in enumerate(row_counts):
        first_x = seed_radius if row % 2 else 2.0 * seed_radius
        y = first_y + row * dy
        for col in range(count):
            centers.append((first_x + 2.0 * seed_radius * col, y))

    centers = np.asarray(centers, dtype=float)
    seed_radii = np.full(32, seed_radius, dtype=float)
    initial = np.concatenate((centers.ravel(), seed_radii))

    def inequalities(z):
        """Return boundary and pairwise non-overlap slacks."""
        c = z[:64].reshape(32, 2)
        r = z[64:]
        boundary = np.concatenate((
            c[:, 0] - r,
            1.0 - c[:, 0] - r,
            c[:, 1] - r,
            1.0 - c[:, 1] - r,
        ))
        pair_slacks = []
        for i in range(32):
            d = c[i + 1:] - c[i]
            pair_slacks.extend(np.sqrt(np.sum(d * d, axis=1)) - r[i] - r[i + 1:])
        return np.concatenate((boundary, np.asarray(pair_slacks)))

    try:
        from scipy.optimize import minimize, linprog

        bounds = [(0.0, 1.0)] * 64 + [(0.0, 0.25)] * 32
        best = initial.copy()
        best_value = float(np.sum(best[64:]))

        # Search several deterministic six-row topologies.  The five-circle
        # rows can be placed against either side of the square, while
        # six-circle rows span the full available horizontal interval.
        def make_seed(counts, reverse=False):
            """Build a staggered, boundary-fitted seed for one row topology."""
            rows = tuple(reversed(counts)) if reverse else tuple(counts)
            result = []
            for row, count in enumerate(rows):
                y = first_y + row * dy
                if count == 6:
                    first_x = seed_radius
                else:
                    first_x = seed_radius if row % 2 else 2.0 * seed_radius
                for col in range(count):
                    result.append((first_x + 2.0 * seed_radius * col, y))
            return np.asarray(result, dtype=float)

        topologies = (
            (5, 6, 5, 6, 5, 5),
            (6, 5, 6, 5, 5, 5),
            (5, 5, 6, 5, 6, 5),
            (5, 6, 5, 5, 6, 5),
            (6, 5, 5, 6, 5, 5),
        )

        starts = [initial]
        for topology in topologies:
            for reverse in (False, True):
                seed = make_seed(topology, reverse=reverse)
                starts.append(np.concatenate((
                    seed.ravel(),
                    np.full(32, 0.86 * seed_radius, dtype=float),
                )))

        # Add small deterministic perturbations to the first few topology
        # seeds.  The reduced number of perturbations limits runtime while
        # still allowing asymmetric unequal-radius solutions.
        rng = np.random.default_rng(1729)
        base_starts = list(starts)
        for base_index, base in enumerate(base_starts[:6]):
            for amplitude in (0.0035, 0.0090):
                trial = base.copy()
                trial[:64] += rng.normal(0.0, amplitude, 64)
                trial[64:] *= 0.90
                rr = trial[64:].reshape(32, 1)
                tc = trial[:64].reshape(32, 2)
                tc[:] = np.maximum(tc, rr)
                tc[:] = np.minimum(tc, 1.0 - rr)
                starts.append(trial)

        for start in starts:
            result = minimize(
                lambda z: -np.sum(z[64:]),
                start,
                method="SLSQP",
                bounds=bounds,
                constraints={"type": "ineq", "fun": inequalities},
                options={
                    "maxiter": 450,
                    "ftol": 3e-11,
                    "disp": False,
                },
            )
            candidate_try = result.x
            if (np.all(np.isfinite(candidate_try)) and
                    np.min(inequalities(candidate_try)) >= -1e-7 and
                    np.sum(candidate_try[64:]) > best_value):
                best = candidate_try.copy()
                best_value = float(np.sum(best[64:]))

        candidate = best

        # For fixed centers, radius maximization is a linear program.  This
        # usually gains a small but measurable amount over joint SLSQP.
        cc = candidate[:64].reshape(32, 2)
        A = []
        b = []
        for i in range(32):
            row = np.zeros(32)
            row[i] = 1.0
            A.append(row)
            b.append(cc[i, 0])
            A.append(row.copy())
            b.append(1.0 - cc[i, 0])
            A.append(row.copy())
            b.append(cc[i, 1])
            A.append(row.copy())
            b.append(1.0 - cc[i, 1])

        for i in range(32):
            for j in range(i):
                row = np.zeros(32)
                row[i] = 1.0
                row[j] = 1.0
                A.append(row)
                b.append(np.linalg.norm(cc[i] - cc[j]))

        lp = linprog(
            -np.ones(32),
            A_ub=np.asarray(A),
            b_ub=np.asarray(b),
            bounds=[(0.0, 0.25)] * 32,
            method="highs",
        )
        if lp.success and np.all(np.isfinite(lp.x)):
            candidate[64:] = lp.x

            def optimize_fixed_centers(cc):
                """Maximize the radius sum exactly for fixed center coordinates."""
                AA = []
                bb = []

                for ii in range(32):
                    row = np.zeros(32)
                    row[ii] = 1.0
                    AA.append(row)
                    bb.append(cc[ii, 0])
                    AA.append(row.copy())
                    bb.append(1.0 - cc[ii, 0])
                    AA.append(row.copy())
                    bb.append(cc[ii, 1])
                    AA.append(row.copy())
                    bb.append(1.0 - cc[ii, 1])

                for ii in range(32):
                    for jj in range(ii):
                        row = np.zeros(32)
                        row[ii] = 1.0
                        row[jj] = 1.0
                        AA.append(row)
                        bb.append(np.linalg.norm(cc[ii] - cc[jj]))

                return linprog(
                    -np.ones(32),
                    A_ub=np.asarray(AA),
                    b_ub=np.asarray(bb),
                    bounds=[(0.0, 0.25)] * 32,
                    method="highs",
                )

            # Repeatedly optimize centers nonlinearly, then recompute the
            # globally optimal radii for those centers.  The LP must be
            # solved after every center move; otherwise SLSQP is forced to
            # optimize against obsolete radius values.
            for _ in range(6):
                refined = minimize(
                    lambda z: -np.sum(z[64:]),
                    candidate,
                    method="SLSQP",
                    bounds=bounds,
                    constraints={"type": "ineq", "fun": inequalities},
                    options={
                        "maxiter": 800,
                        "ftol": 8e-13,
                        "disp": False,
                    },
                )

                trial = refined.x
                if (
                    not np.all(np.isfinite(trial))
                    or np.min(inequalities(trial)) < -2e-7
                ):
                    break

                trial_centers = trial[:64].reshape(32, 2)
                trial_lp = optimize_fixed_centers(trial_centers)
                if (
                    not trial_lp.success
                    or not np.all(np.isfinite(trial_lp.x))
                ):
                    break

                trial[64:] = trial_lp.x
                trial_value = float(np.sum(trial[64:]))

                # Only retain monotone improvements.  This makes the
                # deterministic refinement robust to occasional SLSQP
                # steps into a different, lower-quality local basin.
                if trial_value > float(np.sum(candidate[64:])) + 1e-10:
                    candidate = trial.copy()
                else:
                    break
    except Exception:
        candidate = initial

    centers = candidate[:64].reshape(32, 2)
    radii = np.maximum(candidate[64:], 0.0)

    # Strict deterministic repair for residual roundoff.  Centers are already
    # feasible; scaling radii preserves all pairwise inequalities.
    scale = 1.0 - 2e-10
    scale = min(scale, np.min(centers[:, 0] / np.maximum(radii, 1e-30)))
    scale = min(scale, np.min((1.0 - centers[:, 0]) / np.maximum(radii, 1e-30)))
    scale = min(scale, np.min(centers[:, 1] / np.maximum(radii, 1e-30)))
    scale = min(scale, np.min((1.0 - centers[:, 1]) / np.maximum(radii, 1e-30)))

    for i in range(32):
        for j in range(i):
            distance = np.linalg.norm(centers[i] - centers[j])
            denom = radii[i] + radii[j]
            if denom > 0.0:
                scale = min(scale, distance / denom)

    radii *= max(0.0, scale)
    return centers, radii, float(np.sum(radii))


# EVOLVE-BLOCK-END


def run_packing():
    """Return the deterministic constrained-refined packing of 32 circles."""
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
