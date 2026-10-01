"""Final-only evaluator hooks for direct SkyDiscover/API execution.

Scaling-law tasks fit on the training split and score the saved candidate on
the held-out test split through the generic final-evaluation stage.
"""

import hashlib
import math
import os
from pathlib import Path
import tempfile

from skydiscover_adapter import run_evaluator

SCALING_FINAL_PROTOCOL = "simpletes-scaling-held-out-v2"
SCALING_SEED_POLICY = "Original evaluator defaults; no additional random seed"
SCALING_FINAL_BODY = (
    "def evaluate(program_path):\n"
    "    return module.evaluate(program_path, use_test_data=True)\n"
)


def _json_metric(value):
    """Keep upstream nonfinite diagnostics in strict JSON without changing scores."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_metric(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_metric(item) for item in value]
    return value


def scaling_final_result(metrics, candidate_bytes):
    """Preserve the native reward, including the -1e6 candidate-failure penalty.

    Validity is diagnostic. It must not replace the original score or turn a
    completed candidate failure into an unevaluated run. Adapter failures have
    no R2 and remain distinguishable from an original evaluator return.
    """
    completed = "r2" in metrics
    r2 = metrics.get("r2")
    valid = not metrics.get("error") and type(r2) in (int, float) and math.isfinite(r2)
    result = {f"posthoc_test_{key}": _json_metric(value) for key, value in metrics.items()}
    result.setdefault("posthoc_test_r2", None)
    result.update(
        posthoc_test_validity=int(valid),
        posthoc_test_completed=int(completed),
        posthoc_test_protocol=SCALING_FINAL_PROTOCOL,
        posthoc_test_protocol_version=2,
        posthoc_test_candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        posthoc_test_seed_policy=SCALING_SEED_POLICY,
        combined_score=metrics.get("combined_score", -math.inf),
        validity=int(valid),
    )
    return result


def _run_wrapper(program_path, evaluator_path, body, timeout):
    source = Path(program_path).read_text()
    wrapper = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('original_final', {str(evaluator_path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n" + body
    )
    with tempfile.TemporaryDirectory(prefix="simpletes-final-") as directory:
        path = Path(directory) / "final_evaluator.py"
        path.write_text(wrapper)
        result = run_evaluator(
            path,
            source,
            timeout=timeout,
            python_executable=os.environ.get("SIMPLETES_EVAL_PYTHON") or None,
        )
    return result


def evaluate_scaling_final(program_path, evaluator_path, timeout=3000):
    """Run the released train-fit/test-score evaluator without extra score gates."""
    metrics = _run_wrapper(
        program_path, evaluator_path, SCALING_FINAL_BODY, timeout
    )
    return scaling_final_result(metrics, Path(program_path).read_bytes())
