#!/usr/bin/env python3
"""Backfill final KernelBench ``fast_1`` fields in historical outputs.

Only authoritative ``best/best_program_info.json`` files are considered.
Checkpoint snapshots and runs that have not produced final test metrics are
left untouched. Writes are opt-in, atomic, and preceded by a byte-for-byte
backup of every file that will change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

try:
    from benchmarks.kernelbench.aggregate_fast1 import task_observation
except ModuleNotFoundError:  # Direct execution from this directory.
    from aggregate_fast1 import task_observation


_TASK_NAME_RE = re.compile(r"^kernelbench_l[1-4]_p[0-9]+$")


def _result_files(outputs_root: Path) -> list[Path]:
    if not outputs_root.is_dir():
        return []

    paths: list[Path] = []
    for task_root in sorted(outputs_root.iterdir()):
        if not task_root.is_dir() or not _TASK_NAME_RE.fullmatch(task_root.name):
            continue
        for path in task_root.rglob("best/best_program_info.json"):
            relative_parts = path.relative_to(task_root).parts
            if "checkpoints" not in relative_parts:
                paths.append(path)
    return sorted(paths)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _derived_correctness(
    metrics: dict[str, Any], observation: dict[str, Any]
) -> float | None:
    """Infer correctness only where legacy final metrics make it unambiguous."""
    existing = _finite_number(metrics.get("test_correctness"))
    if existing is not None:
        return 1.0 if existing > 0.5 else 0.0

    speedup = _finite_number(metrics.get("test_speedup_over_eager"))
    ref_time = _finite_number(metrics.get("test_ref_eager_time_ms"))
    kernel_time = _finite_number(metrics.get("test_kernel_time_ms"))
    combined = _finite_number(metrics.get("test_combined_score"))

    # KernelBench emits timings/speedup only after all correctness trials pass.
    if speedup is not None or (
        ref_time is not None and kernel_time is not None and kernel_time > 0.0
    ):
        return 1.0
    if combined is not None:
        return 1.0 if combined > 0.0 else 0.0
    if observation.get("available") and observation.get("fast_1", 0.0) > 0.5:
        return 1.0
    return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    original_mode = path.stat().st_mode
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, original_mode)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _new_backup_root(outputs_root: Path) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    nonce = uuid.uuid4().hex[:8]
    return outputs_root / "kernelbench_fast1" / "backups" / f"backfill_{timestamp}_{nonce}"


def backfill_fast1(
    outputs_root: Path,
    *,
    apply: bool = False,
    backup_root: Path | None = None,
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Inspect or update legacy final result files and return a manifest."""
    outputs_root = Path(outputs_root)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    if backup_root is not None and not apply:
        raise ValueError("backup_root is only valid when apply=True")

    effective_backup_root = (
        Path(backup_root) if backup_root is not None else _new_backup_root(outputs_root)
    )
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    counts = {
        "scanned": 0,
        "changed": 0,
        "unchanged": 0,
        "unavailable": 0,
        "errors": 0,
    }

    for path in _result_files(outputs_root):
        counts["scanned"] += 1
        relative_path = path.relative_to(outputs_root)
        result: dict[str, Any] = {"result_file": str(path)}
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise TypeError("top-level JSON value is not an object")
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                raise TypeError("metrics is not an object")

            observation = task_observation(metrics, threshold)
            result["observation"] = observation
            if not observation["available"]:
                result["status"] = "unavailable"
                counts["unavailable"] += 1
                results.append(result)
                continue

            updates: dict[str, float] = {}
            if "test_fast_1" not in metrics:
                updates["test_fast_1"] = float(observation["fast_1"])
            if "test_correctness" not in metrics:
                correctness = _derived_correctness(metrics, observation)
                if correctness is not None:
                    updates["test_correctness"] = correctness

            result["updates"] = updates
            if not updates:
                result["status"] = "unchanged"
                counts["unchanged"] += 1
                results.append(result)
                continue

            counts["changed"] += 1
            if not apply:
                result["status"] = "would_update"
                results.append(result)
                continue

            backup_path = effective_backup_root / relative_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if backup_path.exists():
                raise FileExistsError(f"refusing to overwrite backup: {backup_path}")
            shutil.copy2(path, backup_path)

            metrics.update(updates)
            _write_json_atomic(path, payload)
            result["status"] = "updated"
            result["backup_file"] = str(backup_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            result["status"] = "error"
            result["error"] = str(exc)
            counts["errors"] += 1
            # A failed write is not a successfully changed result.
            if result.get("updates"):
                counts["changed"] -= 1
        results.append(result)

    return {
        "schema_version": 1,
        "metric": "fast_1",
        "threshold": threshold,
        "mode": "apply" if apply else "dry-run",
        "generated_at": generated_at,
        "outputs_root": str(outputs_root),
        "backup_root": str(effective_backup_root) if apply and counts["changed"] else None,
        **counts,
        "results": results,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--apply", action="store_true", help="write fields after backing up originals"
    )
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)

    try:
        manifest = backfill_fast1(
            args.outputs_root,
            apply=args.apply,
            backup_root=args.backup_root,
            threshold=args.threshold,
        )
    except ValueError as exc:
        parser.error(str(exc))

    manifest_path = args.manifest
    if manifest_path is None and manifest["backup_root"]:
        manifest_path = Path(manifest["backup_root"]) / "manifest.json"
    if manifest_path is not None:
        _write_manifest(manifest_path, manifest)

    print(json.dumps(manifest, indent=2))
    if manifest_path is not None:
        print(f"manifest: {manifest_path}", file=sys.stderr)
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
