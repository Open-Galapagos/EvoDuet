"""GPU-free tests for backfilling final KernelBench ``fast_1`` metrics."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.kernelbench.backfill_fast1 import backfill_fast1


def _result_path(outputs_root: Path, task: str, run: str = "run-a") -> Path:
    return (
        outputs_root
        / task
        / "example_namespace"
        / "individual"
        / "model"
        / run
        / "best"
        / "best_program_info.json"
    )


def _write_result(
    outputs_root: Path,
    task: str,
    metrics: dict,
    *,
    run: str = "run-a",
) -> tuple[Path, str]:
    path = _result_path(outputs_root, task, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(
        {
            "id": f"{task}-{run}",
            "metrics": metrics,
            "unrelated": {"must": "survive"},
        },
        indent=2,
    ) + "\n"
    path.write_text(original)
    return path, original


def _result_for(manifest: dict, path: Path) -> dict:
    matches = [
        result
        for result in manifest["results"]
        if Path(result["result_file"]) == path
    ]
    assert len(matches) == 1
    return matches[0]


def test_backfill_defaults_to_dry_run_and_scans_only_authoritative_results(tmp_path):
    outputs_root = tmp_path / "outputs"
    eligible, original = _write_result(
        outputs_root,
        "kernelbench_l1_p1",
        {
            # The rounded value is a tie, but the raw timing is strictly faster.
            "test_speedup_over_eager": 1.0,
            "test_ref_eager_time_ms": 1.0001,
            "test_kernel_time_ms": 1.0,
        },
    )

    # These resemble result files but are not authoritative KernelBench winners.
    checkpoint = (
        outputs_root
        / "kernelbench_l1_p2"
        / "namespace"
        / "run"
        / "checkpoints"
        / "checkpoint_1"
        / "best"
        / "best_program_info.json"
    )
    other_benchmark = (
        outputs_root
        / "ale_bench_l1_p1"
        / "namespace"
        / "run"
        / "best"
        / "best_program_info.json"
    )
    invalid_task = _result_path(outputs_root, "kernelbench_l5_p1")
    for ignored in (checkpoint, other_benchmark, invalid_task):
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text(json.dumps({"metrics": {"test_combined_score": 2.0}}))

    manifest = backfill_fast1(outputs_root)

    assert manifest["schema_version"] == 1
    assert manifest["metric"] == "fast_1"
    assert manifest["threshold"] == 1.0
    assert manifest["mode"] == "dry-run"
    assert manifest["scanned"] == 1
    assert manifest["changed"] == 1
    assert manifest["unchanged"] == 0
    assert manifest["unavailable"] == 0
    assert manifest["errors"] == 0

    result = _result_for(manifest, eligible)
    assert result["status"] == "would_update"
    assert result["observation"]["source"] == "test_runtime_ratio"
    assert result["updates"] == {
        "test_correctness": 1.0,
        "test_fast_1": 1.0,
    }
    assert "backup_file" not in result

    assert eligible.read_text() == original
    assert manifest["backup_root"] is None
    assert not (outputs_root / "kernelbench_fast1" / "backups").exists()
    for ignored in (checkpoint, other_benchmark, invalid_task):
        assert "test_fast_1" not in json.loads(ignored.read_text())["metrics"]


def test_apply_backfills_pass_tie_and_failure_with_exact_backups(tmp_path):
    outputs_root = tmp_path / "outputs"
    passing, passing_original = _write_result(
        outputs_root,
        "kernelbench_l1_p1",
        {
            "test_speedup_over_eager": 1.0,
            "test_ref_eager_time_ms": 1.0001,
            "test_kernel_time_ms": 1.0,
        },
    )
    tie, tie_original = _write_result(
        outputs_root,
        "kernelbench_l2_p2",
        {
            # Raw timings take precedence over this rounded passing value.
            "test_speedup_over_eager": 1.01,
            "test_ref_eager_time_ms": 1.0,
            "test_kernel_time_ms": 1.0,
        },
    )
    failed, failed_original = _write_result(
        outputs_root,
        "kernelbench_l3_p3",
        {"test_combined_score": -100.0, "test_error": "compile failed"},
    )
    backup_root = tmp_path / "backups"

    manifest = backfill_fast1(
        outputs_root,
        apply=True,
        backup_root=backup_root,
    )

    assert manifest["mode"] == "apply"
    assert manifest["scanned"] == 3
    assert manifest["changed"] == 3
    assert manifest["unchanged"] == 0
    assert manifest["unavailable"] == 0
    assert manifest["errors"] == 0

    expected = {
        passing: (passing_original, 1.0, 1.0),
        tie: (tie_original, 0.0, 1.0),
        failed: (failed_original, 0.0, 0.0),
    }
    for path, (original, fast_1, correctness) in expected.items():
        result = _result_for(manifest, path)
        assert result["status"] == "updated"
        assert result["updates"] == {
            "test_correctness": correctness,
            "test_fast_1": fast_1,
        }

        payload = json.loads(path.read_text())
        assert payload["metrics"]["test_fast_1"] == fast_1
        assert payload["metrics"]["test_correctness"] == correctness
        assert payload["unrelated"] == {"must": "survive"}

        backup_path = Path(result["backup_file"])
        assert backup_path.is_file()
        assert backup_path.is_relative_to(backup_root)
        assert backup_path.read_text() == original

    assert not list(outputs_root.rglob("*.tmp"))


def test_apply_skips_unavailable_records_errors_and_preserves_existing_values(tmp_path):
    outputs_root = tmp_path / "outputs"
    unavailable, unavailable_original = _write_result(
        outputs_root,
        "kernelbench_l1_p1",
        {"combined_score": 42.0},
    )
    existing, existing_original = _write_result(
        outputs_root,
        "kernelbench_l1_p2",
        {"test_fast_1": 0.0, "test_correctness": 1.0},
    )
    malformed = _result_path(outputs_root, "kernelbench_l1_p3")
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed_original = "{not valid json\n"
    malformed.write_text(malformed_original)

    manifest = backfill_fast1(
        outputs_root,
        apply=True,
        backup_root=tmp_path / "backups",
    )

    assert manifest["scanned"] == 3
    assert manifest["changed"] == 0
    assert manifest["unchanged"] == 1
    assert manifest["unavailable"] == 1
    assert manifest["errors"] == 1
    assert _result_for(manifest, unavailable)["status"] == "unavailable"
    assert _result_for(manifest, existing)["status"] == "unchanged"
    assert _result_for(manifest, malformed)["status"] == "error"
    assert unavailable.read_text() == unavailable_original
    assert existing.read_text() == existing_original
    assert malformed.read_text() == malformed_original


def test_apply_is_idempotent_and_does_not_create_a_second_backup(tmp_path):
    outputs_root = tmp_path / "outputs"
    result_path, original = _write_result(
        outputs_root,
        "kernelbench_l4_p4",
        {
            "test_ref_eager_time_ms": 2.0,
            "test_kernel_time_ms": 1.0,
        },
    )
    backup_root = tmp_path / "backups"

    first = backfill_fast1(outputs_root, apply=True, backup_root=backup_root)
    after_first = result_path.read_text()
    backups_after_first = sorted(path for path in backup_root.rglob("*") if path.is_file())
    second = backfill_fast1(outputs_root, apply=True, backup_root=backup_root)

    assert first["changed"] == 1
    assert _result_for(first, result_path)["status"] == "updated"
    assert second["changed"] == 0
    assert second["unchanged"] == 1
    assert _result_for(second, result_path)["status"] == "unchanged"
    assert result_path.read_text() == after_first
    assert sorted(path for path in backup_root.rglob("*") if path.is_file()) == (
        backups_after_first
    )
    assert len(backups_after_first) == 1
    assert backups_after_first[0].read_text() == original
