"""Evaluator for packing exactly 32 circles in a unit square."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np


N_CIRCLES = 32
TARGET_VALUE = 2.936
EXECUTION_TIMEOUT_SECONDS = 600
TOLERANCE = 1e-6

_CHILD_PROGRAM = r"""
import importlib.util
import pickle
import sys

program_path, result_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("candidate_program", program_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load candidate: {program_path}")
program = importlib.util.module_from_spec(spec)
spec.loader.exec_module(program)
result = program.run_packing()
with open(result_path, "wb") as result_file:
    pickle.dump(result, result_file)
"""


def _run_candidate(program_path: str):
    """Execute the candidate in a child process and return its result."""
    descriptor, result_path = tempfile.mkstemp(suffix=".pickle")
    os.close(descriptor)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD_PROGRAM, os.path.abspath(program_path), result_path],
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"Candidate exited with code {completed.returncode}: {detail[-4000:]}"
            )
        if os.path.getsize(result_path) == 0:
            raise RuntimeError("Candidate did not write an evaluation result")
        with open(result_path, "rb") as result_file:
            result = pickle.load(result_file)
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError("run_packing() must return (centers, radii, sum_radii)")
        return result
    finally:
        try:
            os.unlink(result_path)
        except FileNotFoundError:
            pass


def _validate(centers: np.ndarray, radii: np.ndarray) -> tuple[bool, str]:
    """Validate shapes, finite values, boundaries, and pairwise separation."""
    if centers.shape != (N_CIRCLES, 2) or radii.shape != (N_CIRCLES,):
        return False, (
            f"invalid shapes: centers={centers.shape}, radii={radii.shape}; "
            f"expected ({N_CIRCLES}, 2) and ({N_CIRCLES},)"
        )
    if not np.isfinite(centers).all() or not np.isfinite(radii).all():
        return False, "centers and radii must contain only finite values"
    if np.any(radii < 0.0):
        return False, "radii must be non-negative"

    x = centers[:, 0]
    y = centers[:, 1]
    inside = (
        (x - radii >= -TOLERANCE)
        & (x + radii <= 1.0 + TOLERANCE)
        & (y - radii >= -TOLERANCE)
        & (y + radii <= 1.0 + TOLERANCE)
    )
    if not np.all(inside):
        return False, f"circle {int(np.flatnonzero(~inside)[0])} lies outside the unit square"

    deltas = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    required = radii[:, None] + radii[None, :]
    overlap = np.triu(distances + TOLERANCE < required, k=1)
    if np.any(overlap):
        first, second = np.argwhere(overlap)[0]
        return False, f"circles {first} and {second} overlap"
    return True, ""


def _failed_metrics(elapsed: float) -> dict[str, float]:
    return {
        "sum_radii": 0.0,
        "target_ratio": 0.0,
        "validity": 0.0,
        "eval_time": float(elapsed),
        "combined_score": 0.0,
    }


def evaluate(program_path: str) -> dict[str, float]:
    """Evaluate a candidate and normalize its valid radius sum by 2.936."""
    started = time.time()
    try:
        centers_raw, radii_raw, reported_sum_raw = _run_candidate(program_path)
        centers = np.asarray(centers_raw, dtype=float)
        radii = np.asarray(radii_raw, dtype=float)
        valid, reason = _validate(centers, radii)
        elapsed = time.time() - started
        if not valid:
            print(f"Invalid packing: {reason}")
            return _failed_metrics(elapsed)

        sum_radii = float(np.sum(radii))
        reported_sum = float(reported_sum_raw)
        if not np.isfinite(reported_sum):
            print("Warning: reported sum is not finite; using the verified sum")
        elif abs(sum_radii - reported_sum) > TOLERANCE:
            print(
                f"Warning: reported sum {reported_sum} does not match "
                f"verified sum {sum_radii}"
            )

        target_ratio = sum_radii / TARGET_VALUE
        print(
            f"Evaluation: valid=True, n={N_CIRCLES}, sum_radii={sum_radii:.9f}, "
            f"target={TARGET_VALUE}, ratio={target_ratio:.9f}, time={elapsed:.2f}s"
        )
        return {
            "sum_radii": sum_radii,
            "target_ratio": float(target_ratio),
            "validity": 1.0,
            "eval_time": float(elapsed),
            "combined_score": float(target_ratio),
        }
    except Exception as error:
        elapsed = time.time() - started
        print(f"Evaluation failed: {error}")
        traceback.print_exc()
        return _failed_metrics(elapsed)
