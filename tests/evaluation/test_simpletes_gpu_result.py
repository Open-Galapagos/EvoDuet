"""SimpleTES GPU diagnostics preserve legacy decisions and stay evaluator-local."""

import copy

import pytest

from skydiscover.evaluation.evaluation_result import evaluation_failure_reason
from skydiscover.evaluation.simpletes_gpu_result import simpletes_gpu_failure_reason


@pytest.mark.parametrize(
    "metrics,artifacts,expected",
    [
        (
            {
                "combined_score": 0.0,
                "validity": 0.0,
                "error_name": "StaticCheckFailed",
                "error": ["Missing @triton.jit", "Uses torch computation op: torch.bmm"],
            },
            {},
            "StaticCheckFailed: Missing @triton.jit; Uses torch computation op: torch.bmm",
        ),
        (
            {"validity": 0, "metadata": {"static_bypass": ["custom_kernel has no return"]}},
            {},
            "custom_kernel has no return",
        ),
        (
            {"validity": 0, "metadata": {"static_errors": ["invalid Triton kernel"]}},
            {},
            "invalid Triton kernel",
        ),
        (
            {
                "validity": 0,
                "compiled": True,
                "error_name": "GpuKernelEvaluationFailed",
                "error": "test.2.error: 17 mismatched elements",
            },
            {},
            "GpuKernelEvaluationFailed: test.2.error: 17 mismatched elements",
        ),
        (
            {"validity": 0},
            {"error_message": "Triton compiler failed", "error_name": "CompilationError"},
            "CompilationError: Triton compiler failed",
        ),
        (
            {"timeout": True, "error": "GPU worker exceeded 3000 seconds"},
            {},
            "GPU worker exceeded 3000 seconds",
        ),
        (
            {"validity": 0, "metadata": {"log_excerpt": "CUDA launch failed"}},
            {},
            "CUDA launch failed",
        ),
        ({"validity": -1, "error": None}, {}, "Evaluation reported validity=-1"),
        ({"validity": 0.0, "error": 0.0}, {}, "Evaluation reported validity=0.0"),
        ({"combined_score": 0.0, "validity": 1.0}, {}, None),
        ({"combined_score": -2.5}, {}, None),
        ({"combined_score": 0.8, "error": 0.0}, {}, None),
        (
            {
                "combined_score": 1.0,
                "validity": 1,
                "metadata": {"static_warnings": ["precision downgrade"]},
            },
            {},
            None,
        ),
    ],
)
def test_gpu_details_preserve_failure_decision_and_metrics(metrics, artifacts, expected):
    before = copy.deepcopy((metrics, artifacts))
    legacy = evaluation_failure_reason(metrics, artifacts)

    actual = simpletes_gpu_failure_reason(metrics, artifacts)

    assert actual == expected
    assert (actual is None) == (legacy is None)
    assert (metrics, artifacts) == before
    assert evaluation_failure_reason(metrics, artifacts) == legacy


def test_long_compiler_error_preserves_start_and_final_exception():
    error = "Traceback (most recent call last):\n" + "compiler context\n" * 600
    error += "CompilationError: invalid tl.dot operands"
    message = simpletes_gpu_failure_reason({"validity": 0, "error": error})
    assert len(message) == 4000
    assert message.startswith("Traceback")
    assert message.endswith("CompilationError: invalid tl.dot operands")
