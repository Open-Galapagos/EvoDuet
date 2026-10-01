"""Scientific invariants, malformed artifacts and process isolation for the new tasks."""

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"


def load(path):
    spec = importlib.util.spec_from_file_location(path.parent.name + "_checked", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime():
    return load(BENCHMARKS / "_public_optimization_runtime.py")


def test_candidate_stdout_and_environment_are_isolated(tmp_path, monkeypatch, runtime):
    monkeypatch.setenv("PUBLIC_OPT_TEST_SECRET", "not-forwarded")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    candidate = tmp_path / "candidate.py"
    candidate.write_text("import os\ndef solve(payload):\n"
                         "    print('candidate debug output')\n"
                         "    return [payload, os.getenv('PUBLIC_OPT_TEST_SECRET'), os.getenv('PYTHONPATH')]\n")
    assert runtime.run_candidate(candidate, 42) == [42, None, None]


def test_candidate_timeout_and_nonfinite_results(tmp_path, runtime):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("import time\ndef solve(payload):\n    time.sleep(60)\n")
    with pytest.raises(TimeoutError):
        runtime.run_candidate(candidate, {}, timeout=0.2)
    candidate.write_text("def solve(payload):\n    return float('nan')\n")
    with pytest.raises(ValueError):
        runtime.run_candidate(candidate, {})


def test_pressure_discreteness_and_volume_boundary():
    evaluator = load(BENCHMARKS / "pressure_vessel_design/evaluator.py")
    radius = 0.8125 / 0.0193
    length = 1296000 / (math.pi * radius**2) - 4 * radius / 3
    result = evaluator.score_design([0.8125, 0.4375, radius, length])
    assert result["cost"] == pytest.approx(6059.714335048436, rel=1e-12)
    for bad in ([0.8, 0.4375, radius, length], [0.8125, 0.4375, radius, length-1],
                [0.8125, True, radius, length], [0.8125, 0.4375, radius, 201]):
        with pytest.raises(ValueError):
            evaluator.score_design(bad)


def test_lj_energy_matches_published_coordinates_and_rigid_motion():
    evaluator = load(BENCHMARKS / "lennard_jones_13/evaluator.py")
    text = (BENCHMARKS / "lennard_jones_13/oracle/lj_excerpt.txt").read_text()
    points = [list(map(float, line.split())) for line in text.splitlines() if line.strip()]
    original = evaluator.score_coordinates(points)["energy"]
    rotated = [[-y+3, x-2, z+1] for x, y, z in reversed(points)]
    assert original == pytest.approx(-44.326801, abs=5e-7)
    assert evaluator.score_coordinates(rotated)["energy"] == pytest.approx(original, abs=1e-11)
    points[1] = points[0]
    with pytest.raises(ValueError):
        evaluator.score_coordinates(points)


def test_labs_published_run_length_answer_and_invalid_sign(tmp_path):
    evaluator = load(BENCHMARKS / "labs_27/evaluator.py")
    signs = []
    for index, length in enumerate("34313131211211"):
        signs.extend([(-1)**index] * int(length))
    candidate = tmp_path / "candidate.py"
    candidate.write_text(f"def solve(payload):\n    return {signs!r}\n")
    assert evaluator.evaluate(candidate)["energy"] == 37
    signs[0] = 0
    candidate.write_text(f"def solve(payload):\n    return {signs!r}\n")
    assert evaluator.evaluate(candidate)["validity"] == 0


def test_chwirut_published_parameters_and_singular_model(tmp_path):
    evaluator = load(BENCHMARKS / "nist_chwirut2/evaluator.py")
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def solve(payload):\n    return [0.16657666537, 0.0051653291286, 0.012150007096]\n")
    assert evaluator.evaluate(candidate)["rss"] == pytest.approx(513.04802941, abs=1e-7)
    candidate.write_text("def solve(payload):\n    return [0, 0, 0]\n")
    assert evaluator.evaluate(candidate)["validity"] == 0
