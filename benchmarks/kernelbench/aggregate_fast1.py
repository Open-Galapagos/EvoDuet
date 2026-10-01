#!/usr/bin/env python3
"""Aggregate final KernelBench task results into paper-style ``fast_1``.

Each SkyDiscover KernelBench run stores the frozen winner's authoritative
metrics in ``best/best_program_info.json`` with a ``test_`` prefix. This tool
groups matching runs across an explicitly supplied task set and averages their
0/1 ``test_fast_1`` contributions. The expected task list, rather than the
number of result files found, is always the denominator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


_TASK_NAME_RE = re.compile(r"^kernelbench_l(?P<level>[1-4])_p(?P<problem>[0-9]+)$")
_TASK_PAIR_RE = re.compile(r"^(?P<level>[1-4]):(?P<problem>[0-9]+)$")


def normalize_task_spec(spec: str) -> str:
    """Normalize a launcher row/name such as ``'task 1 3'`` or ``'1:3'``."""
    token = spec.strip().split()[0] if spec.strip() else ""
    if _TASK_NAME_RE.fullmatch(token):
        return token
    match = _TASK_PAIR_RE.fullmatch(token)
    if match:
        return f"kernelbench_l{match.group('level')}_p{match.group('problem')}"
    raise ValueError(f"invalid KernelBench task specification: {spec!r}")


def _task_level(task: str) -> str:
    match = _TASK_NAME_RE.fullmatch(task)
    if match is None:
        raise ValueError(f"invalid normalized KernelBench task name: {task!r}")
    return match.group("level")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def task_observation(metrics: dict[str, Any], threshold: float = 1.0) -> dict[str, Any]:
    """Extract one task's strict ``fast_1`` indicator from final metrics.

    New results carry ``test_fast_1`` directly. For older runs, raw final
    timings are preferred over the two-decimal printed speedup. A non-positive
    final combined score is a completed compile/correctness/timeout failure and
    contributes zero. If no final metric exists, the observation is incomplete.
    """
    explicit = _finite_number(metrics.get("test_fast_1"))
    correctness = _finite_number(metrics.get("test_correctness"))
    speedup = _finite_number(metrics.get("test_speedup_over_eager"))
    ref_time = _finite_number(metrics.get("test_ref_eager_time_ms"))
    kernel_time = _finite_number(metrics.get("test_kernel_time_ms"))
    combined = _finite_number(metrics.get("test_combined_score"))

    if explicit is not None:
        passed = explicit > 0.5
        source = "test_fast_1"
    elif correctness == 0.0:
        passed = False
        source = "test_correctness"
    elif ref_time is not None and kernel_time is not None and kernel_time > 0.0:
        passed = ref_time / kernel_time > threshold
        source = "test_runtime_ratio"
    elif speedup is not None:
        passed = speedup > threshold
        source = "test_speedup_over_eager"
    elif combined is not None and combined <= 0.0:
        passed = False
        source = "test_combined_score_failure"
    elif combined is not None:
        # KernelBench defines combined_score as eager speedup. This fallback is
        # only for legacy results lacking the more precise final metrics.
        passed = combined > threshold
        source = "test_combined_score"
    else:
        return {
            "available": False,
            "fast_1": 0.0,
            "status": "missing_final_metrics",
        }

    if passed:
        status = "pass"
    elif (
        correctness == 1.0
        or speedup is not None
        or (ref_time is not None and kernel_time is not None)
    ):
        status = "correct_but_not_faster"
    else:
        status = "incorrect_or_failed"

    observation: dict[str, Any] = {
        "available": True,
        "fast_1": 1.0 if passed else 0.0,
        "status": status,
        "source": source,
    }
    for key, value in (
        ("correctness", correctness),
        ("speedup_over_eager", speedup),
        ("ref_eager_time_ms", ref_time),
        ("kernel_time_ms", kernel_time),
    ):
        if value is not None:
            observation[key] = value
    return observation


def _load_observation(path: Path, threshold: float) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            raise TypeError("metrics is not an object")
        observation = task_observation(metrics, threshold)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        observation = {
            "available": False,
            "fast_1": 0.0,
            "status": "invalid_result_file",
            "error": str(exc),
        }
    observation["result_file"] = str(path)
    return observation


def aggregate_fast1(
    outputs_root: Path,
    run_namespace: str,
    task_specs: Iterable[str],
    *,
    threshold: float = 1.0,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build per-run and per-level ``fast_1`` summaries for selected tasks."""
    tasks = [normalize_task_spec(spec) for spec in task_specs]
    if not tasks:
        raise ValueError("at least one KernelBench task is required")
    if len(set(tasks)) != len(tasks):
        raise ValueError("duplicate KernelBench tasks are not allowed")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")

    namespace_path = Path(run_namespace)
    if namespace_path.is_absolute() or ".." in namespace_path.parts:
        raise ValueError("run_namespace must be a relative path without '..'")

    observations: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        task_root = outputs_root / task / namespace_path
        if not task_root.is_dir():
            continue
        for info_path in task_root.rglob("best/best_program_info.json"):
            if "checkpoints" in info_path.parts:
                continue
            run_root = info_path.parent.parent
            if run_id is not None and run_root.name != run_id:
                continue
            run_key = run_root.relative_to(task_root).as_posix()
            observations.setdefault(run_key, {})[task] = _load_observation(
                info_path, threshold
            )

    runs: dict[str, Any] = {}
    for run_key in sorted(observations):
        found = observations[run_key]
        task_results: dict[str, Any] = {}
        for task in tasks:
            task_results[task] = found.get(
                task,
                {
                    "available": False,
                    "fast_1": 0.0,
                    "status": "missing_result",
                },
            )

        passed = sum(int(result["fast_1"] > 0.5) for result in task_results.values())
        completed = sum(int(bool(result["available"])) for result in task_results.values())
        total = len(tasks)

        by_level: dict[str, Any] = {}
        for level in sorted({_task_level(task) for task in tasks}):
            level_tasks = [task for task in tasks if _task_level(task) == level]
            level_passed = sum(
                int(task_results[task]["fast_1"] > 0.5) for task in level_tasks
            )
            level_completed = sum(
                int(bool(task_results[task]["available"])) for task in level_tasks
            )
            level_total = len(level_tasks)
            by_level[level] = {
                "fast_1": level_passed / level_total,
                "fast_1_percent": 100.0 * level_passed / level_total,
                "passed": level_passed,
                "completed": level_completed,
                "total": level_total,
                "complete": level_completed == level_total,
            }

        runs[run_key] = {
            "fast_1": passed / total,
            "fast_1_percent": 100.0 * passed / total,
            "passed": passed,
            "completed": completed,
            "total": total,
            "complete": completed == total,
            "missing_tasks": [
                task for task, result in task_results.items() if not result["available"]
            ],
            "by_level": by_level,
            "tasks": task_results,
        }

    return {
        "schema_version": 1,
        "metric": "fast_1",
        "definition": (
            "fraction of expected tasks that are correct and strictly faster "
            "than PyTorch eager"
        ),
        "baseline": "pytorch_eager",
        "threshold": threshold,
        "run_namespace": run_namespace,
        "run_id_filter": run_id,
        "expected_tasks": tasks,
        "expected_task_count": len(tasks),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runs": runs,
    }


def write_summary(summary: dict[str, Any], output_path: Path) -> None:
    """Atomically write a summary so readers never observe partial JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _default_output(outputs_root: Path, run_namespace: str, run_id: str | None) -> Path:
    path = outputs_root / "kernelbench_fast1" / Path(run_namespace)
    if run_id:
        path /= run_id
    return path / "fast1_summary.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks", nargs="+", help="Expected task names, rows, or level:problem pairs"
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--run-namespace", required=True)
    parser.add_argument("--run-id", help="Only aggregate run directories with this final name")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return a non-zero status if no run group is complete",
    )
    args = parser.parse_args(argv)

    try:
        summary = aggregate_fast1(
            args.outputs_root,
            args.run_namespace,
            args.tasks,
            threshold=args.threshold,
            run_id=args.run_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output_path = args.output or _default_output(
        args.outputs_root, args.run_namespace, args.run_id
    )
    write_summary(summary, output_path)

    complete_runs = sum(int(run["complete"]) for run in summary["runs"].values())
    total_runs = len(summary["runs"])
    print(output_path)
    print(
        f"KernelBench fast_1: wrote {total_runs} run group(s), "
        f"{complete_runs} complete",
        file=sys.stderr,
    )
    if total_runs == 0:
        return 1
    if args.require_complete and complete_runs != total_runs:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
