"""Native SRSD I.27.6 validation selection and held-out final assessment."""

import json
from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR.parent))
from _srsd_feynman_runtime import evaluate_task, make_payload as _make_payload

VARIABLE_DESCRIPTIONS = ["object distance d1, metres", "refractive index n, dimensionless",
                         "image distance d2, metres"]
REFERENCE_EXPRESSION = "1/(1/x0+x1/x2)"


def make_payload():
    return _make_payload(TASK_DIR, VARIABLE_DESCRIPTIONS)


def evaluate(program_path):
    return evaluate_task(program_path, TASK_DIR, VARIABLE_DESCRIPTIONS, REFERENCE_EXPRESSION, "val")


def evaluate_final(program_path):
    return evaluate_task(program_path, TASK_DIR, VARIABLE_DESCRIPTIONS, REFERENCE_EXPRESSION, "test")


if __name__ == "__main__":
    function = evaluate_final if "--test" in sys.argv[2:] else evaluate
    print(json.dumps(function(sys.argv[1]), indent=2, allow_nan=False))
