"""Tests for materializing evaluator-produced model checkpoints."""

import base64
import hashlib
from types import SimpleNamespace

from skydiscover.context_builder.utils import format_artifacts
from skydiscover.runner import (
    CHECKPOINT_BASE64_ARTIFACT,
    CHECKPOINT_SHA256_ARTIFACT,
    Runner,
)
from skydiscover.search.base_database import Program


class _Database:
    def __init__(self, program: Program):
        self.program = program
        self.best_program_id = program.id
        self.saved = []

    def save(self, path: str, iteration: int) -> None:
        self.saved.append((path, iteration))

    def get(self, program_id: str):
        return self.program if program_id == self.program.id else None

    def get_best_program(self):
        return self.program


def _runner(tmp_path, payload: bytes, *, checksum: str | None = None):
    artifacts = {
        CHECKPOINT_BASE64_ARTIFACT: base64.b64encode(payload).decode("ascii"),
        CHECKPOINT_SHA256_ARTIFACT: checksum or hashlib.sha256(payload).hexdigest(),
        "feedback": "training completed",
    }
    program = Program(
        id="best-program",
        solution="print('candidate')\n",
        metrics={"combined_score": 0.5},
        artifacts=artifacts,
    )
    runner = object.__new__(Runner)
    runner.output_dir = str(tmp_path)
    runner.file_extension = ".py"
    runner.config = SimpleNamespace(language="python")
    runner.database = _Database(program)
    return runner, program


def test_checkpoint_save_materializes_best_model(tmp_path):
    payload = b"model checkpoint bytes"
    runner, _ = _runner(tmp_path, payload)

    runner._save_checkpoint(5)

    checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_5"
    assert (checkpoint_dir / "checkpoint.pt").read_bytes() == payload
    assert (checkpoint_dir / "best_program.py").is_file()
    assert (checkpoint_dir / "best_program_info.json").is_file()


def test_best_save_materializes_model(tmp_path):
    payload = b"authoritative test checkpoint"
    runner, program = _runner(tmp_path, payload)

    runner._save_best_program(program)

    assert (tmp_path / "best" / "checkpoint.pt").read_bytes() == payload


def test_bad_checkpoint_checksum_is_not_materialized(tmp_path):
    runner, program = _runner(tmp_path, b"bad payload", checksum="0" * 64)

    result = runner._materialize_model_checkpoint(program, str(tmp_path / "destination"))

    assert result is None
    assert not (tmp_path / "destination" / "checkpoint.pt").exists()


def test_private_checkpoint_artifacts_are_not_added_to_llm_context(tmp_path):
    _, program = _runner(tmp_path, b"private payload")

    rendered = format_artifacts(program)

    assert "training completed" in rendered
    assert CHECKPOINT_BASE64_ARTIFACT not in rendered
    assert CHECKPOINT_SHA256_ARTIFACT not in rendered
    assert base64.b64encode(b"private payload").decode("ascii") not in rendered
