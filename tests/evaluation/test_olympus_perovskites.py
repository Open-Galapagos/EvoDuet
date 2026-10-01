"""Native data integrity, composition validation, and exact oracle replay."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "benchmarks/olympus_perovskites"


@pytest.fixture
def evaluator():
    spec = importlib.util.spec_from_file_location(
        "checked_olympus_perovskites", TASK / "evaluator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_table_and_program_scores_match_published_optimum(evaluator):
    parameters, table = evaluator._load_native_data()
    assert [len(parameter["options"]) for parameter in parameters] == [16, 3, 4]
    assert len(table) == 192
    optimum = ("hydrazinium", "Sn", "I")
    assert min(table, key=table.get) == optimum
    assert sum(value == table[optimum] for value in table.values()) == 1
    initial = evaluator.evaluate(TASK / "initial_program.py")
    best = evaluator.evaluate(TASK / "oracle/best_program.py")
    assert initial["validity"] == best["validity"] == 1.0
    assert initial["bandgap_ev"] == 5.3704
    assert initial["combined_score"] == pytest.approx(1.5249 / 5.3704)
    assert best["bandgap_ev"] == 1.5249
    assert best["combined_score"] == 1.0
    assert best["bandgap_gap_ev"] == 0.0


def test_payload_supplies_categories_without_target_table(evaluator):
    payload = evaluator.task_payload()
    assert set(payload) == {"parameters", "objective"}
    for parameter in payload["parameters"]:
        assert set(parameter) == {"name", "options"}
        assert all(type(option) is str for option in parameter["options"])
    assert payload["objective"] == {"name": "hse_gap", "direction": "minimize", "units": "eV"}


@pytest.mark.parametrize(
    "artifact",
    [
        None,
        True,
        1.5249,
        ["hydrazinium", "Sn", "I"],
        {"organic": "hydrazinium", "cation": "Sn"},
        {"organic": True, "cation": "Sn", "anion": "I"},
        {"organic": "hydrazinium", "cation": 1, "anion": "I"},
        {"organic": "hydrazinium", "cation": "Sn", "anion": None},
        {"organic": "unlisted", "cation": "Sn", "anion": "I"},
        {"organic": "hydrazinium", "cation": "I", "anion": "Sn"},
        {"organic": "hydrazinium", "cation": "Sn", "anion": "I", "bandgap_ev": 0.0},
        {"combined_score": 1.0, "bandgap_ev": 0.0},
    ],
)
def test_malformed_or_forged_candidate_outputs_get_no_credit(tmp_path, evaluator, artifact):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"def solve(payload):\n    return {artifact!r}\n")
    result = evaluator.evaluate(candidate)
    assert result["combined_score"] == 0.0
    assert result["validity"] == 0.0
    assert "error" in result


def test_checksum_detects_changed_native_data(tmp_path, evaluator, monkeypatch):
    shutil.copytree(TASK / "data", tmp_path / "data")
    path = tmp_path / "data/data.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(evaluator, "TASK_DIR", tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluator.task_payload()
