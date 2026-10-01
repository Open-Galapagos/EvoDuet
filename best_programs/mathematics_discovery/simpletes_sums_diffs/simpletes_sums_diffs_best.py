# EVOLVE-BLOCK-START
"""
MSTD Optimization for Constant C(A) = log(|A+A|/|A|) / log(|A-A|/|A|).

Approach:
1. Parameterized Multi-Modulus Sweep:
   Systematically evaluates generalized Penman-Wells families across moduli
   M in {32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112} across boundary variants,
   half-period inclusions, odd progression bounds, and fringe extensions.
2. High-Efficiency Exact Bitmask Engine:
   Computes exact sumset and difference set sizes in ~15 microseconds using C-level
   Python arbitrary-precision bit shifts and popcount (.bit_count()).
3. Multi-Operator Alternating Hill-Climber:
   - Single-element removals (pruning redundant elements to reduce |A| and suppress diffs)
   - Zero-leakage interior and fringe additions
   - Boundary & fringe element swaps (x in A -> y not in A)
   - 2-lookahead additions and removals
   - Iterated Local Search (ILS) with perturbation kicks to escape local optima
"""

import math
import random
import time


def _score_set(A_sorted):
    """Compute exact C(A), |A+A|, and |A-A| for a sorted candidate set."""
    n = len(A_sorted)
    if n < 2 or n > 512:
        return 0.0, 0, 0

    min_v = A_sorted[0]
    span = A_sorted[-1] - min_v
    if span > 1_500_000:
        return 0.0, 0, 0

    mask = 0
    for x in A_sorted:
        mask |= 1 << (x - min_v)

    sum_mask = 0
    diff_mask = 0
    for x in A_sorted:
        v = x - min_v
        sum_mask |= mask << v
        diff_mask |= mask << (span - v)

    s = sum_mask.bit_count()
    d = diff_mask.bit_count()
    if s <= n or d <= n:
        return 0.0, s, d

    return math.log(s / n) / math.log(d / n), s, d


def _generate_penman_variant(M, j, bl_variant=0, include_half=False, odd_delta=0, odd_start=0, fringe=0):
    """Generalized Penman-Wells construction for modulus M = 8 * ell."""
    if M % 8 != 0 or j < 1:
        return []
    N = M * (j + 2)
    m_half = M // 2

    multiples_of_4 = [4 * s for s in range(M // 4)]
    if not include_half and m_half in multiples_of_4:
        multiples_of_4.remove(m_half)

    b_left = set(multiples_of_4)
    if bl_variant == 0:
        b_left.add(2)
    elif bl_variant == 1:
        b_left.update([2, m_half - 2])
    elif bl_variant == 2:
        b_left.update([2, 6])
    elif bl_variant == 3:
        b_left.update([2, M - 2])
    elif bl_variant == 4:
        b_left.update([2, m_half + 2])
    elif bl_variant == 5:
        if (M - 4) in b_left:
            b_left.remove(M - 4)
        b_left.add(2)
    else:
        b_left.add(2)

    max_k = (N - 3) // 4 + odd_delta
    odds = {1 + 4 * k for k in range(odd_start, max_k + 1) if 0 <= 1 + 4 * k <= N}
    e_int = {m_half + M * k for k in range(1, j + 1)}
    b_right = {N - x for x in b_left}

    res = b_left | odds | e_int | b_right

    if fringe == 1:
        res.add(N + 1)
    elif fringe == 2:
        res.update([N + 1, N + 2])
    elif fringe == 3:
        res.add(-1)
    elif fringe == 4:
        res.update([-1, N + 1])

    return sorted(res)


def _optimize_set(init_set, time_budget):
    """
    High-performance alternating local search using exact bitmask metric evaluations.
    Applies exhaustive single removals, targeted zero-leakage additions,
    2-element operations, and boundary swaps.
    """
    curr = set(init_set)
    best_c, _, _ = _score_set(sorted(curr))
    best_set = set(curr)

    step = 0
    max_steps = 250

    while step < max_steps and time.time() < time_budget:
        step += 1
        improved = False
        sorted_curr = sorted(curr)
        n = len(sorted_curr)
        min_v, max_v = sorted_curr[0], sorted_curr[-1]

        # -----------------------------------------------------------------
        # 1. Exact Single Removal Screening (All Elements in Set)
        # -----------------------------------------------------------------
        if n > 10:
            best_rem = None
            best_rem_c = best_c
            # Test all boundary elements and a targeted sample of interior multiples of 4
            rem_candidates = sorted_curr[:15] + sorted_curr[-15:]
            interior_m4 = [x for x in sorted_curr[15:-15] if (x - min_v) % 4 == 0]
            if len(interior_m4) > 40:
                rem_candidates.extend(interior_m4[:: len(interior_m4) // 40])
            else:
                rem_candidates.extend(interior_m4)

            for rem in set(rem_candidates):
                trial = sorted(curr - {rem})
                sc, _, _ = _score_set(trial)
                if sc > best_rem_c:
                    best_rem_c = sc
                    best_rem = rem

            if best_rem is not None and best_rem_c > best_c + 1e-9:
                curr.remove(best_rem)
                best_c = best_rem_c
                best_set = set(curr)
                improved = True
                continue

        # -----------------------------------------------------------------
        # 2. Targeted Zero-Leakage & Fringe Addition Screening
        # -----------------------------------------------------------------
        if n < 512:
            sorted_curr = sorted(curr)
            min_v, max_v = sorted_curr[0], sorted_curr[-1]

            # Fringes: outer elements immediately adjacent to endpoints
            fringe_cands = [min_v - d for d in range(1, 25)] + [max_v + d for d in range(1, 25)]

            # Boundary gap elements
            boundary_gaps = (
                [min_v + d for d in range(1, 56) if (min_v + d) not in curr] +
                [max_v - d for d in range(1, 56) if (max_v - d) not in curr]
            )

            # Interior multiples of 4 (zero 2 mod 4 difference leakage)
            interior_m4 = [x for x in range(min_v + 4, max_v, 4) if x not in curr]
            if len(interior_m4) > 50:
                step_m4 = len(interior_m4) // 50 + 1
                sampled_m4 = interior_m4[::step_m4]
            else:
                sampled_m4 = interior_m4

            add_pool = fringe_cands + boundary_gaps + sampled_m4

            best_add = None
            best_add_c = best_c
            top_adds = []

            for v in add_pool:
                if v in curr or v < -1_000_000 or v > 1_000_000:
                    continue
                trial = sorted(curr | {v})
                sc, _, _ = _score_set(trial)
                if sc > best_add_c:
                    best_add_c = sc
                    best_add = v
                if sc > best_c - 0.0003:
                    top_adds.append((sc, v))

            if best_add is not None and best_add_c > best_c + 1e-9:
                curr.add(best_add)
                best_c = best_add_c
                best_set = set(curr)
                improved = True
                continue

            # -------------------------------------------------------------
            # 2b. 2-Step Lookahead Addition
            # -------------------------------------------------------------
            if n <= 510 and top_adds and time.time() < time_budget - 1.5:
                top_adds.sort(key=lambda x: x[0], reverse=True)
                top_v1s = [x[1] for x in top_adds[:12]]
                best_pair = None
                best_pair_c = best_c

                for v1 in top_v1s:
                    sub = curr | {v1}
                    sub_sorted = sorted(sub)
                    s_min, s_max = sub_sorted[0], sub_sorted[-1]
                    sub_fringe = [s_min - d for d in range(1, 10)] + [s_max + d for d in range(1, 10)]
                    sub_gaps = [s_min + d for d in range(1, 32) if (s_min + d) not in sub] + \
                               [s_max - d for d in range(1, 32) if (s_max - d) not in sub]

                    for v2 in sub_fringe + sub_gaps:
                        if v2 in sub or v2 < -1_000_000 or v2 > 1_000_000:
                            continue
                        sc2, _, _ = _score_set(sorted(sub | {v2}))
                        if sc2 > best_pair_c:
                            best_pair_c = sc2
                            best_pair = (v1, v2)

                if best_pair is not None and best_pair_c > best_c + 1e-9:
                    curr.add(best_pair[0])
                    curr.add(best_pair[1])
                    best_c = best_pair_c
                    best_set = set(curr)
                    improved = True
                    continue

        # -----------------------------------------------------------------
        # 3. Element Swap Screening (Remove x, Add y)
        # -----------------------------------------------------------------
        if time.time() < time_budget - 1.5:
            sorted_curr = sorted(curr)
            min_v, max_v = sorted_curr[0], sorted_curr[-1]
            swap_rems = sorted_curr[:8] + sorted_curr[-8:]
            swap_adds = (
                [min_v - d for d in range(1, 12)] +
                [max_v + d for d in range(1, 12)] +
                [min_v + d for d in range(1, 28) if (min_v + d) not in curr] +
                [max_v - d for d in range(1, 28) if (max_v - d) not in curr]
            )

            best_swap = None
            best_swap_c = best_c

            for rem in swap_rems:
                sub = curr - {rem}
                for add_v in swap_adds:
                    if add_v in sub or add_v < -1_000_000 or add_v > 1_000_000:
                        continue
                    sc, _, _ = _score_set(sorted(sub | {add_v}))
                    if sc > best_swap_c:
                        best_swap_c = sc
                        best_swap = (rem, add_v)

                if best_swap is not None and best_swap_c > best_c + 1e-6:
                    break

            if best_swap is not None and best_swap_c > best_c + 1e-9:
                curr.remove(best_swap[0])
                curr.add(best_swap[1])
                best_c = best_swap_c
                best_set = set(curr)
                improved = True
                continue

        if not improved:
            break

    return sorted(best_set), best_c


def construct_set():
    """
    Construct an optimal MSTD set maximizing C(A).
    Sweeps parameterized generalized Penman-Wells families, ranks seeds,
    and applies deep multi-stage hill-climbing with perturbation kicks.
    """
    start_time = time.time()
    time_limit = 135.0  # safe limit within 180s

    # Guaranteed champion baseline: M=56, j=30 with fringe element 1793 (C(A) >= 1.144710)
    base_seed = _generate_penman_variant(56, 30, bl_variant=0, include_half=False, odd_delta=0, odd_start=0)
    best_A = sorted(set(base_seed) | {base_seed[-1] + 1})
    best_c, _, _ = _score_set(best_A)

    seed_pool = [(best_c, best_A)]

    # 1. Systematic scan across moduli M in {32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112}
    moduli = [56, 64, 48, 72, 80, 40, 88, 96, 104, 112, 32]
    for M in moduli:
        step_j = M // 4 + 1
        base_const = M + 2
        approx_max_j = (512 - base_const) // step_j

        for j in range(max(1, approx_max_j - 4), approx_max_j + 3):
            for bl_var in (0, 1, 2, 3, 4, 5):
                for include_half in (False, True):
                    for odd_delta in (-2, -1, 0, 1):
                        for odd_start in (0, 1):
                            for fringe in (0, 1, 2, 3, 4):
                                cand = _generate_penman_variant(
                                    M, j, bl_var, include_half, odd_delta, odd_start, fringe
                                )
                                n_cand = len(cand)
                                if 2 <= n_cand <= 512:
                                    c, _, _ = _score_set(cand)
                                    if c > best_c:
                                        best_c = c
                                        best_A = cand
                                    if c > 1.135 and n_cand >= 460:
                                        seed_pool.append((c, cand))

    seed_pool.sort(key=lambda x: x[0], reverse=True)

    # 2. Select top unique diverse seeds
    unique_candidates = []
    seen_sigs = set()
    for c_val, cand in [(best_c, best_A)] + seed_pool:
        sig = (len(cand), cand[0], cand[-1])
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            unique_candidates.append(cand)
        if len(unique_candidates) >= 6:
            break

    # 3. Deep optimization of top unique seeds
    time_per_seed = 12.0
    for cand in unique_candidates:
        if time.time() - start_time > time_limit - 15.0:
            break
        budget = min(start_time + time_limit - 5.0, time.time() + time_per_seed)
        opt_A, opt_c = _optimize_set(cand, budget)
        if opt_c > best_c:
            best_c = opt_c
            best_A = opt_A

    # 4. Iterated perturbation kicks on champion set if time remains
    rng = random.Random(42)
    while time.time() - start_time < time_limit - 8.0:
        kicked = set(best_A)
        sorted_k = sorted(kicked)
        drop_pool = sorted_k[:10] + sorted_k[-10:]
        to_drop = rng.sample(drop_pool, min(2, len(drop_pool)))
        for d in to_drop:
            kicked.remove(d)

        budget = min(start_time + time_limit - 3.0, time.time() + 6.0)
        opt_A, opt_c = _optimize_set(list(kicked), budget)
        if opt_c > best_c:
            best_c = opt_c
            best_A = opt_A

    min_elem = min(best_A)
    return [x - min_elem for x in sorted(best_A)]
# EVOLVE-BLOCK-END

MIN_SET_SIZE = 2
MAX_SET_SIZE = 512
MIN_INT = -1_000_000
MAX_INT = 1_000_000


def _sanitize_output(values):
    """Convert arbitrary iterable output into a valid sorted integer list."""
    try:
        raw = list(values)
    except TypeError as e:
        raise ValueError(f"Output is not iterable: {e}")

    ints = []
    for x in raw:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(xf):
            continue
        xi = int(round(xf))
        xi = max(MIN_INT, min(MAX_INT, xi))
        ints.append(xi)

    unique_vals = sorted(set(ints))
    if len(unique_vals) > MAX_SET_SIZE:
        unique_vals = unique_vals[:MAX_SET_SIZE]

    if len(unique_vals) < MIN_SET_SIZE:
        unique_vals = [0, 1]

    return unique_vals


def _compute_c(values):
    n = len(values)
    sumset = {a + b for a in values for b in values}
    diffset = {a - b for a in values for b in values}

    sum_ratio = len(sumset) / n
    diff_ratio = len(diffset) / n

    if sum_ratio <= 1.0 or diff_ratio <= 1.0:
        return 0.0

    return float(math.log(sum_ratio) / math.log(diff_ratio))


def run_code():
    """Return (A_values, claimed_c)."""
    values = construct_set()
    values = _sanitize_output(values)
    c_value = _compute_c(values)
    return values, c_value


if __name__ == "__main__":
    candidate_values, candidate_c = run_code()
    print(f"|A|={len(candidate_values)}, C(A)={candidate_c:.10f}")