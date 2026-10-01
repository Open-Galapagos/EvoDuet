"""Independently look up a composition in the pinned OLYMPUS HSE06 table."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _public_optimization_runtime import run_candidate

TASK_DIR = Path(__file__).resolve().parent
PARAMETER_NAMES = ("organic", "cation", "anion")


def _load_native_data():
    """Check byte hashes and the complete 16 x 3 x 4 categorical domain."""
    manifest = json.loads((TASK_DIR / "data/provenance.json").read_text())
    for name in ("config.json", "data.csv"):
        body = (TASK_DIR / "data" / name).read_bytes()
        if hashlib.sha256(body).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Pinned upstream asset checksum mismatch: {name}")
    config = json.loads((TASK_DIR / "data/config.json").read_text())
    parameters = config["parameters"]
    if tuple(parameter["name"] for parameter in parameters) != PARAMETER_NAMES:
        raise ValueError("Unexpected native parameter order")
    if [len(parameter["options"]) for parameter in parameters] != [16, 3, 4]:
        raise ValueError("Unexpected native categorical domain")
    if config["default_goal"] != "minimize":
        raise ValueError("The native objective must minimize bandgap")
    table = {}
    with (TASK_DIR / "data/data.csv").open(newline="") as source:
        for row in csv.reader(source):
            if len(row) != 4:
                raise ValueError("Malformed upstream bandgap row")
            composition = tuple(row[:3])
            bandgap = float(row[3])
            if composition in table or not math.isfinite(bandgap) or bandgap <= 0:
                raise ValueError("Duplicate composition or invalid upstream bandgap")
            table[composition] = bandgap
    expected = set(itertools.product(*(parameter["options"] for parameter in parameters)))
    if set(table) != expected or len(table) != 192:
        raise ValueError("The bandgap table must cover exactly the 192 native compositions")
    return parameters, table


def task_payload():
    """Expose the categorical domain only; targets remain in the evaluator."""
    parameters, _ = _load_native_data()
    return {
        "parameters": [
            {"name": parameter["name"], "options": parameter["options"]} for parameter in parameters
        ],
        "objective": {"name": "hse_gap", "direction": "minimize", "units": "eV"},
    }


def score_composition(artifact):
    if type(artifact) is not dict or set(artifact) != set(PARAMETER_NAMES):
        raise ValueError("Return exactly an object with organic, cation and anion keys")
    if any(type(artifact[name]) is not str for name in PARAMETER_NAMES):
        raise ValueError("All three component choices must be category-name strings")
    _, table = _load_native_data()
    composition = tuple(artifact[name] for name in PARAMETER_NAMES)
    if composition not in table:
        raise ValueError("Unknown composition; use the category names supplied in payload")
    bandgap = table[composition]
    minimum_bandgap = min(table.values())
    return {
        "combined_score": minimum_bandgap / bandgap,
        "validity": 1.0,
        "bandgap_ev": bandgap,
        "bandgap_gap_ev": bandgap - minimum_bandgap,
    }


def evaluate(program_path):
    try:
        artifact = run_candidate(program_path, task_payload(), timeout=30)
        return score_composition(artifact)
    except (ValueError, TypeError, TimeoutError, OverflowError, ArithmeticError, OSError) as error:
        return {"combined_score": 0.0, "validity": 0.0, "error": str(error)[:1000]}


evaluate_final = evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=2, allow_nan=False))
