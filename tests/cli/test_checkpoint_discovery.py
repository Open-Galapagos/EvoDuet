"""Tests for checkpoint discovery helper in the CLI."""

import json
from pathlib import Path

from skydiscover.cli import _find_latest_checkpoint


def test_returns_highest_iteration(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "checkpoint_2").mkdir()
    (checkpoint_dir / "checkpoint_10").mkdir()
    (checkpoint_dir / "checkpoint_1").mkdir()

    latest = _find_latest_checkpoint(str(checkpoint_dir))
    assert latest == str(checkpoint_dir / "checkpoint_10")


def test_ignores_non_numeric_dirs(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "latest").mkdir()
    (checkpoint_dir / "checkpoint_old").mkdir()
    (checkpoint_dir / "checkpoint_3").mkdir()

    latest = _find_latest_checkpoint(str(checkpoint_dir))
    assert latest == str(checkpoint_dir / "checkpoint_3")


def test_returns_none_without_valid_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "latest").mkdir()
    (checkpoint_dir / "checkpoint_old").mkdir()

    assert _find_latest_checkpoint(str(checkpoint_dir)) is None


def test_prefers_completed_checkpoint_over_newer_unmarked_directory(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    completed = checkpoint_dir / "checkpoint_2"
    completed.mkdir()
    (completed / "checkpoint_complete.json").write_text(
        json.dumps({"schema_version": 1, "iteration": 2})
    )
    (checkpoint_dir / "checkpoint_10").mkdir()

    assert _find_latest_checkpoint(str(checkpoint_dir)) == str(completed)


def test_ignores_checkpoint_with_mismatched_completion_marker(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    invalid = checkpoint_dir / "checkpoint_9"
    invalid.mkdir()
    (invalid / "checkpoint_complete.json").write_text(json.dumps({"iteration": 8}))

    assert _find_latest_checkpoint(str(checkpoint_dir)) is None
