"""Deterministic constrained epoch optimization for the Voyager-2 grand tour."""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, least_squares

_FAMILY_DIR = Path(__file__).resolve().parent.parent
if str(_FAMILY_DIR) not in sys.path:
    sys.path.insert(0, str(_FAMILY_DIR))

from problem_config import load_problem_for_candidate
from tools_wrapper import Tools

problem = load_problem_for_candidate(__file__)
tools = Tools()

# EVOLVE-BLOCK-START

DAY = 86400.0
MIN_TOF = 5.0
INF = 1.0e30
SEED = 20260916
CACHE = {}


def _window(spec):
    """Return the legal lower and upper MJD values for a boundary specification."""
    t = spec["time"]
    if t["kind"] == "window":
        return float(t["lo"]), float(t["hi"])
    value = t.get("value", t.get("mjd"))
    return float(value), float(value)


def _state(pid, epoch, boundary=None):
    """Return the heliocentric state from ephemeris or a configured fixed boundary."""
    pid = str(pid)

    if pid == "0":
        if boundary is not None:
            spec = problem[boundary]
            if "state_r" in spec and "state_v" in spec:
                return (
                    np.asarray(spec["state_r"], float),
                    np.asarray(spec["state_v"], float),
                )

        for name in ("start", "end"):
            spec = problem[name]
            if "state_r" in spec and "state_v" in spec:
                return (
                    np.asarray(spec["state_r"], float),
                    np.asarray(spec["state_v"], float),
                )

        return np.zeros(3), np.zeros(3)

    key = (pid, round(float(epoch), 8))
    if key not in CACHE:
        r, v = tools.ephem(pid, float(epoch))
        CACHE[key] = np.asarray(r, float), np.asarray(v, float)
    return CACHE[key]


def _piecewise(value, breakpoints):
    """Evaluate a boundary launcher curve by linear interpolation and endpoint extension."""
    pts = sorted((float(x), float(y)) for x, y in breakpoints)
    x = float(value)

    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x <= x1:
            if x1 == x0:
                return float(y1)
            f = (x - x0) / (x1 - x0)
            return float(y0 + f * (y1 - y0))

    return float(pts[-1][1])


def _periapsis_cost(vinf, spec, pid):
    """Compute the configured impulsive burn at planetary periapsis."""
    mu = float(problem["planet_mu"][str(pid)])
    radius = float(problem["planet_radius"][str(pid)])
    rp = radius * (1.0 + float(spec["h_factor"]))
    period = float(spec["T_days"]) * DAY

    correction = (4.0 * np.pi**2 * mu**2 / period**2) ** (1.0 / 3.0)
    escape = np.sqrt(float(vinf) ** 2 + 2.0 * mu / rp)
    parking = np.sqrt(max(2.0 * mu / rp - correction, 0.0))
    return float(escape - parking)


def _boundary_cost(v_before, v_after, epoch, spec, is_start):
    """Evaluate the exact start or end boundary delta-v model."""
    if spec["type"] == "piecewise_linear":
        jump = np.linalg.norm(np.asarray(v_after) - np.asarray(v_before))
        return _piecewise(jump, spec["breakpoints"])

    pid = str(spec.get("planet_id", "0"))
    _, vp = _state(pid, epoch, "start" if is_start else "end")
    velocity = np.asarray(v_after if is_start else v_before, float)
    vinf = np.linalg.norm(velocity - vp)
    return _periapsis_cost(vinf, spec, pid)


def _sequence():
    """Select the longest admissible Voyager-style outer-planet flyby sequence."""
    allowed = {str(x) for x in problem.get("allowed_GA_planets", [])}
    maximum = int(problem.get("max_GA", 0))
    end_pid = str(problem["end"].get("planet_id", "8"))

    candidates = [
        ("5", "6", "7"),
        ("5", "6"),
        ("5", "7"),
        ("6", "7"),
        ("5",),
        ("6",),
        ("7",),
        (),
    ]

    for sequence in candidates:
        if len(sequence) <= maximum and all(
            p in allowed and p != end_pid for p in sequence
        ):
            return sequence

    return ()


def _historical_times(sequence):
    """Construct the canonical Voyager encounter epochs clipped to legal windows."""
    slo, shi = _window(problem["start"])
    elo, ehi = _window(problem["end"])

    centers = {
        "5": 44126.5,
        "6": 44985.6,
        "7": 46730.9,
    }

    t0 = float(np.clip(43389.2, slo, shi))
    tf = float(np.clip(48163.0, elo, ehi))

    middle = [
        float(np.clip(centers[p], t0 + MIN_TOF, tf - MIN_TOF))
        for p in sequence
    ]

    result = np.asarray([t0] + middle + [tf], float)
    if np.any(np.diff(result) <= MIN_TOF):
        result = np.linspace(t0, tf, len(sequence) + 2)

    return result


def _bounds(sequence):
    """Build broad legal encounter-epoch bounds around the historical tour."""
    slo, shi = _window(problem["start"])
    elo, ehi = _window(problem["end"])

    centers = {
        "5": 44126.5,
        "6": 44985.6,
        "7": 46730.9,
    }

    bounds = [(slo, shi)]
    for pid in sequence:
        c = centers[pid]
        bounds.append((
            max(slo + MIN_TOF, c - 3000.0),
            min(ehi - MIN_TOF, c + 3000.0),
        ))
    bounds.append((elo, ehi))
    return bounds


def _arcs(sequence, times, lowpath=True):
    """Compute planetary states and Lambert endpoint velocities for every tour leg."""
    times = np.asarray(times, float)

    start_pid = str(problem["start"].get("planet_id", "0"))
    end_pid = str(problem["end"].get("planet_id", "0"))
    pids = [start_pid] + list(sequence) + [end_pid]

    states = []
    for i, pid in enumerate(pids):
        boundary = None
        if i == 0:
            boundary = "start"
        elif i == len(pids) - 1:
            boundary = "end"
        states.append(_state(pid, times[i], boundary))

    departures = []
    arrivals = []

    for i in range(len(pids) - 1):
        tof = float(times[i + 1] - times[i])
        if tof <= MIN_TOF:
            raise ValueError("invalid transfer time")

        va, vb = tools.lambert(
            states[i][0],
            states[i + 1][0],
            tof * DAY,
            float(problem["mu_sun"]),
            prograde=True,
            lowpath=lowpath,
            M=0,
        )
        departures.append(np.asarray(va, float))
        arrivals.append(np.asarray(vb, float))

    return pids, states, departures, arrivals


def _flyby_residual(sequence, times, lowpath=True):
    """Return incoming-minus-outgoing hyperbolic excess speed at each flyby."""
    try:
        _, _, departures, arrivals = _arcs(sequence, times, lowpath)
        residual = []

        for i, pid in enumerate(sequence):
            _, vp = _state(pid, times[i + 1])
            vin = np.linalg.norm(arrivals[i] - vp)
            vout = np.linalg.norm(departures[i + 1] - vp)
            residual.append(vin - vout)

        return np.asarray(residual, float)
    except Exception:
        return np.ones(len(sequence), float) * 1.0e6


def _evaluate(sequence, times, build=False, lowpath=True):
    """Evaluate exact boundary and powered-flyby cost and optionally construct nodes."""
    times = np.asarray(times, float)

    if len(times) != len(sequence) + 2 or not np.all(np.isfinite(times)):
        return (INF, None) if build else INF

    slo, shi = _window(problem["start"])
    elo, ehi = _window(problem["end"])

    if not slo <= times[0] <= shi or not elo <= times[-1] <= ehi:
        return (INF, None) if build else INF
    if np.any(np.diff(times) <= MIN_TOF):
        return (INF, None) if build else INF

    try:
        pids, states, departures, arrivals = _arcs(sequence, times, lowpath)

        total = _boundary_cost(
            states[0][1],
            departures[0],
            times[0],
            problem["start"],
            True,
        )

        for i, pid in enumerate(sequence):
            _, vp = _state(pid, times[i + 1])
            mu = float(problem["planet_mu"][pid])
            radius = float(problem["planet_radius"][pid])
            altitude = float(
                problem.get("flyby", {})
                .get("min_altitude_km", {})
                .get(pid, 200.0)
            )

            _, dv, feasible = tools.powered_flyby(
                arrivals[i],
                departures[i + 1],
                vp,
                mu,
                radius + altitude,
            )

            if not feasible or not np.isfinite(dv):
                return (INF, None) if build else INF

            total += float(dv)

        total += _boundary_cost(
            arrivals[-1],
            states[-1][1],
            times[-1],
            problem["end"],
            False,
        )

    except Exception:
        return (INF, None) if build else INF

    if not build:
        return float(total)

    nodes = [{
        "type": "start",
        "time": float(times[0]),
        "planet_id": pids[0],
        "r": states[0][0],
        "v_before": states[0][1],
        "v_after": departures[0],
    }]

    for i, pid in enumerate(sequence):
        nodes.append({
            "type": "GA",
            "time": float(times[i + 1]),
            "planet_id": str(pid),
            "r": states[i + 1][0],
            "v_before": arrivals[i],
            "v_after": departures[i + 1],
        })

    nodes.append({
        "type": "end",
        "time": float(times[-1]),
        "planet_id": pids[-1],
        "r": states[-1][0],
        "v_before": arrivals[-1],
        "v_after": states[-1][1],
    })

    return float(total), nodes


def _boundary_objective(sequence, times, lowpath=True):
    """Return only launch and arrival cost for constrained manifold optimization."""
    try:
        _, states, departures, arrivals = _arcs(sequence, times, lowpath)
        return _boundary_cost(
            states[0][1],
            departures[0],
            times[0],
            problem["start"],
            True,
        ) + _boundary_cost(
            arrivals[-1],
            states[-1][1],
            times[-1],
            problem["end"],
            False,
        )
    except Exception:
        return INF


def _project_seed(sequence, seed, bounds, lowpath=True):
    """Project a trial epoch vector toward equal-speed flyby compatibility."""
    if not sequence:
        return np.asarray(seed, float)

    lo = np.asarray([b[0] for b in bounds], float)
    hi = np.asarray([b[1] for b in bounds], float)

    def residual(x):
        return _flyby_residual(sequence, x, lowpath)

    try:
        result = least_squares(
            residual,
            np.asarray(seed, float),
            bounds=(lo, hi),
            max_nfev=250,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        return np.asarray(result.x, float)
    except Exception:
        return np.asarray(seed, float)


def _polish(sequence, seed, bounds, lowpath=True):
    """Minimize boundary delta-v subject to exact unpowered flyby constraints."""
    if not sequence:
        return np.asarray(seed, float)

    def objective(x):
        """Evaluate the launch plus arrival boundary cost."""
        return _boundary_objective(sequence, x, lowpath)

    def constraint(x):
        """Enforce equal incoming and outgoing hyperbolic excess speeds."""
        return _flyby_residual(sequence, x, lowpath)

    try:
        result = minimize(
            objective,
            np.asarray(seed, float),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": constraint},
            options={
                "maxiter": 900,
                "ftol": 1e-12,
                "disp": False,
            },
        )
        return np.asarray(result.x, float)
    except Exception:
        return np.asarray(seed, float)


def _seeds(sequence, bounds):
    """Generate deterministic historical, corner, and distributed epoch seeds."""
    rng = np.random.default_rng(SEED)
    x0 = _historical_times(sequence)
    result = [x0]

    # Perturb each historical epoch at several deterministic scales.
    for scale in (2.0, 10.0, 35.0, 100.0, 280.0, 700.0):
        for _ in range(5):
            x = x0 + rng.normal(0.0, scale, len(x0))
            x = np.asarray([
                np.clip(x[i], bounds[i][0], bounds[i][1])
                for i in range(len(x))
            ])
            if np.all(np.diff(x) > MIN_TOF):
                result.append(x)

    # Explicit launch/arrival window combinations are useful when windows are broad.
    for a in (0.0, 0.5, 1.0):
        for b in (0.0, 0.5, 1.0):
            x = x0.copy()
            x[0] = bounds[0][0] + a * (bounds[0][1] - bounds[0][0])
            x[-1] = bounds[-1][0] + b * (bounds[-1][1] - bounds[-1][0])
            if np.all(np.diff(x) > MIN_TOF):
                result.append(x)

    # Distributed interior samples retain independently sampled boundaries.
    for _ in range(30):
        x = np.asarray([rng.uniform(lo, hi) for lo, hi in bounds])
        x[1:-1] = np.sort(x[1:-1])
        if np.all(np.diff(x) > MIN_TOF):
            result.append(x)

    return result


def _format(nodes):
    """Convert all trajectory fields to evaluator-compatible scalar and list types."""
    output = []

    for node in nodes:
        q = dict(node)
        q["type"] = str(q["type"])
        q["time"] = float(q["time"])
        q["planet_id"] = str(q["planet_id"])

        for key in ("r", "v_before", "v_after"):
            q[key] = np.asarray(q[key], float).tolist()

        output.append(q)

    return output


def run_code():
    """Search deterministic flyby manifolds and return the lowest-cost valid tour."""
    sequence = _sequence()
    bounds = _bounds(sequence)
    best_value, best_nodes = _evaluate(
        sequence,
        _historical_times(sequence),
        build=True,
        lowpath=True,
    )

    # The canonical low-path branch is expected for Voyager 2. A secondary
    # branch is sampled only as a robustness measure and can never replace a
    # better valid canonical solution.
    branches = [True, False]

    for lowpath in branches:
        for seed in _seeds(sequence, bounds):
            projected = _project_seed(sequence, seed, bounds, lowpath)
            polished = _polish(sequence, projected, bounds, lowpath)

            value, nodes = _evaluate(
                sequence,
                polished,
                build=True,
                lowpath=lowpath,
            )

            if nodes is not None and value < best_value:
                best_value = value
                best_nodes = nodes

    if best_nodes is None:
        return []

    if len(best_nodes) > int(problem.get("max_nodes", 999)):
        return []

    try:
        record.event("voyager_deterministic_manifold_multistart")
        record.set("final_nodes", len(best_nodes))
        record.set("final_objective", float(best_value))
        record.set("sequence", "E->J->S->U->N")
    except Exception:
        pass

    return _format(best_nodes)

# EVOLVE-BLOCK-END