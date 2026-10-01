import json
from types import SimpleNamespace

import pytest

from skydiscover.runner import CHECKPOINT_COMPLETE_FILE, CHECKPOINT_SCHEMA_VERSION, Runner


class _Database:
    best_program_id = None

    def __init__(self, value="saved"):
        self.value = value
        self.last_iteration = 0
        self.loaded = None

    def save(self, path, iteration):
        self.last_iteration = iteration
        with open(f"{path}/database.txt", "w") as output:
            output.write(self.value)

    def load(self, path):
        self.loaded = path
        self.last_iteration = 7

    def get_best_program(self):
        return None


def _runner(tmp_path):
    runner = object.__new__(Runner)
    runner.output_dir = str(tmp_path)
    runner.file_extension = ".py"
    runner.database = _Database()
    runner.discovery_controller = None
    return runner


def test_checkpoint_is_staged_and_marked_complete(tmp_path):
    runner = _runner(tmp_path)

    runner._save_checkpoint(3)

    checkpoint = tmp_path / "checkpoints" / "checkpoint_3"
    marker = json.loads((checkpoint / CHECKPOINT_COMPLETE_FILE).read_text())
    assert marker["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert marker["iteration"] == 3
    assert (checkpoint / "database.txt").read_text() == "saved"
    assert list((tmp_path / "checkpoints").glob(".checkpoint_3.tmp-*")) == []


def test_failed_save_never_publishes_partial_checkpoint(tmp_path, monkeypatch):
    runner = _runner(tmp_path)

    def fail(_path):
        raise OSError("world state failed")

    monkeypatch.setattr(runner, "_save_evoduet_checkpoint", fail)

    with pytest.raises(OSError, match="world state failed"):
        runner._save_checkpoint(4)

    checkpoints = tmp_path / "checkpoints"
    assert not (checkpoints / "checkpoint_4").exists()
    assert list(checkpoints.glob(".checkpoint_4.tmp-*")) == []


def test_same_iteration_resave_atomically_replaces_checkpoint(tmp_path):
    runner = _runner(tmp_path)
    runner.database.value = "old"
    runner._save_checkpoint(5)

    runner.database.value = "new"
    runner._save_checkpoint(5)

    checkpoints = tmp_path / "checkpoints"
    checkpoint = checkpoints / "checkpoint_5"
    assert (checkpoint / "database.txt").read_text() == "new"
    assert (checkpoint / CHECKPOINT_COMPLETE_FILE).is_file()
    assert list(checkpoints.glob(".checkpoint_5.*-*")) == []


def test_failed_resave_preserves_previous_complete_checkpoint(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.database.value = "old"
    runner._save_checkpoint(6)

    runner.database.value = "new"

    def fail(_path):
        raise OSError("world state failed")

    monkeypatch.setattr(runner, "_save_evoduet_checkpoint", fail)
    with pytest.raises(OSError, match="world state failed"):
        runner._save_checkpoint(6)

    checkpoints = tmp_path / "checkpoints"
    checkpoint = checkpoints / "checkpoint_6"
    assert (checkpoint / "database.txt").read_text() == "old"
    assert (checkpoint / CHECKPOINT_COMPLETE_FILE).is_file()
    assert list(checkpoints.glob(".checkpoint_6.tmp-*")) == []


def test_resave_retains_old_checkpoint_without_atomic_exchange(tmp_path, monkeypatch):
    import skydiscover.runner as runner_module

    runner = _runner(tmp_path)
    runner.database.value = "old"
    runner._save_checkpoint(6)
    runner.database.value = "new"
    monkeypatch.setattr(runner_module, "_atomic_exchange", lambda *_args: False)

    runner._save_checkpoint(6)

    checkpoint = tmp_path / "checkpoints" / "checkpoint_6"
    assert (checkpoint / "database.txt").read_text() == "old"
    assert (checkpoint / CHECKPOINT_COMPLETE_FILE).is_file()
    assert list((tmp_path / "checkpoints").glob(".checkpoint_6.tmp-*")) == []


def test_load_rejects_legacy_checkpoint_without_completion_marker(tmp_path):
    runner = _runner(tmp_path)
    legacy = tmp_path / "checkpoints" / "checkpoint_7"
    legacy.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="Only current checkpoints are supported"):
        runner._load_checkpoint(str(legacy))

    assert runner.database.loaded is None


def test_load_accepts_current_complete_checkpoint(tmp_path):
    runner = _runner(tmp_path)
    runner._save_checkpoint(7)
    checkpoint = tmp_path / "checkpoints" / "checkpoint_7"
    runner._load_checkpoint(str(checkpoint))
    assert runner.database.loaded == str(checkpoint)


@pytest.mark.parametrize("version", [None, 0, 1, True, "2"])
def test_load_rejects_old_or_invalid_checkpoint_version(tmp_path, version):
    runner = _runner(tmp_path)
    runner._save_checkpoint(7)
    checkpoint = tmp_path / "checkpoints" / "checkpoint_7"
    (checkpoint / CHECKPOINT_COMPLETE_FILE).write_text(
        json.dumps({"schema_version": version, "iteration": 7})
    )
    with pytest.raises(RuntimeError, match="Only current checkpoints are supported"):
        runner._load_checkpoint(str(checkpoint))
    assert runner.database.loaded is None


def test_load_rejects_mismatched_completion_marker(tmp_path):
    runner = _runner(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "checkpoint_7"
    checkpoint.mkdir(parents=True)
    (checkpoint / CHECKPOINT_COMPLETE_FILE).write_text(
        json.dumps({"schema_version": CHECKPOINT_SCHEMA_VERSION, "iteration": 6})
    )

    with pytest.raises(RuntimeError, match="Invalid checkpoint completion marker"):
        runner._load_checkpoint(str(checkpoint))

    assert runner.database.loaded is None


def test_early_stop_uses_actual_processed_checkpoint_boundary(tmp_path):
    runner = _runner(tmp_path)
    runner.database.last_iteration = 4
    runner.discovery_controller = SimpleNamespace(
        early_stopping_triggered=False,
        shutdown_event=SimpleNamespace(is_set=lambda: True),
        last_processed_iteration=7,
    )

    assert runner._final_checkpoint_iteration(1, 100) == 7


def test_completed_run_uses_planned_checkpoint_boundary(tmp_path):
    runner = _runner(tmp_path)
    runner.discovery_controller = SimpleNamespace(
        early_stopping_triggered=False,
        shutdown_event=SimpleNamespace(is_set=lambda: False),
        last_processed_iteration=7,
    )

    assert runner._final_checkpoint_iteration(1, 100) == 100
