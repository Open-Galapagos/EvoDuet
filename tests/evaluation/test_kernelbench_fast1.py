"""GPU-free tests for KernelBench's final ``fast_1`` reporting."""

import json
import math

import pytest

from benchmarks.kernelbench.aggregate_fast1 import (
    aggregate_fast1,
    task_observation,
    write_summary,
)
from benchmarks.kernelbench.evaluator import evaluator as kernelbench_evaluator


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"speedup_over_eager": 1.0}, 1.0),
        ({"speedup_over_eager": 0.01}, 1.0),
        ({"speedup_over_eager": 2.0, "error": "compile failed"}, 0.0),
        ({"speedup_over_eager": 0.0}, 0.0),
        ({"speedup_over_eager": -1.0}, 0.0),
        ({"speedup_over_eager": True}, 0.0),
        ({"speedup_over_eager": math.nan}, 0.0),
        ({"speedup_over_eager": math.inf}, 0.0),
        ({}, 0.0),
    ],
)
def test_correctness_indicator_requires_a_successful_finite_speedup(metrics, expected):
    assert kernelbench_evaluator._correctness_indicator(metrics) == expected


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        # The displayed speedup rounds to a tie, but the raw timings are faster.
        (
            {
                "speedup_over_eager": 1.00,
                "ref_eager_time_ms": 1.0001,
                "kernel_time_ms": 1.0,
            },
            1.0,
        ),
        # fast_p uses a strict boundary: an exact tie is not a pass, even when
        # the rounded displayed speedup claims otherwise.
        (
            {
                "speedup_over_eager": 1.01,
                "ref_eager_time_ms": 1.0,
                "kernel_time_ms": 1.0,
            },
            0.0,
        ),
        (
            {
                "speedup_over_eager": 1.01,
                "ref_eager_time_ms": 0.9999,
                "kernel_time_ms": 1.0,
            },
            0.0,
        ),
        ({"speedup_over_eager": 1.01}, 1.0),
        ({"speedup_over_eager": 1.00}, 0.0),
        ({"speedup_over_eager": 10.0, "error": "incorrect"}, 0.0),
    ],
)
def test_fast_p_indicator_is_strict_and_prefers_raw_timing(metrics, expected):
    assert kernelbench_evaluator._fast_p_indicator(metrics, threshold=1.0) == expected


@pytest.mark.parametrize(
    ("evaluation", "correctness", "fast_1"),
    [
        (
            {
                "combined_score": 1.00,
                "speedup_over_eager": 1.00,
                "ref_eager_time_ms": 1.0001,
                "kernel_time_ms": 1.0,
            },
            1.0,
            1.0,
        ),
        (
            {
                "combined_score": -100.0,
                "error": "Kernel failed correctness check or did not compile",
            },
            0.0,
            0.0,
        ),
    ],
)
def test_evaluate_final_adds_task_level_indicators_without_running_gpu(
    monkeypatch, evaluation, correctness, fast_1
):
    monkeypatch.setattr(kernelbench_evaluator, "evaluate", lambda _path: dict(evaluation))

    result = kernelbench_evaluator.evaluate_final("unused-candidate.py")

    assert result["correctness"] == correctness
    assert result["fast_1"] == fast_1
    for key, value in evaluation.items():
        assert result[key] == value


def test_task_observation_prefers_raw_final_timing_over_rounded_speedup():
    rounded_tie_but_raw_pass = task_observation(
        {
            "test_speedup_over_eager": 1.00,
            "test_ref_eager_time_ms": 1.0001,
            "test_kernel_time_ms": 1.0,
        }
    )
    rounded_pass_but_raw_tie = task_observation(
        {
            "test_speedup_over_eager": 1.01,
            "test_ref_eager_time_ms": 1.0,
            "test_kernel_time_ms": 1.0,
        }
    )

    assert rounded_tie_but_raw_pass["fast_1"] == 1.0
    assert rounded_tie_but_raw_pass["source"] == "test_runtime_ratio"
    assert rounded_pass_but_raw_tie["fast_1"] == 0.0
    assert rounded_pass_but_raw_tie["status"] == "correct_but_not_faster"
    assert rounded_pass_but_raw_tie["source"] == "test_runtime_ratio"


@pytest.mark.parametrize(
    ("metrics", "available", "source"),
    [
        (
            {"test_correctness": 0.0},
            True,
            "test_correctness",
        ),
        ({"test_combined_score": -100.0}, True, "test_combined_score_failure"),
        ({}, False, None),
    ],
)
def test_task_observation_counts_failures_as_zero_and_distinguishes_missing(
    metrics, available, source
):
    observation = task_observation(metrics)

    assert observation["available"] is available
    assert observation["fast_1"] == 0.0
    if source is None:
        assert "source" not in observation
        assert observation["status"] == "missing_final_metrics"
    else:
        assert observation["source"] == source
        assert observation["status"] == "incorrect_or_failed"


def _write_result(outputs_root, task, namespace, run_key, metrics):
    info_path = (
        outputs_root
        / task
        / namespace
        / run_key
        / "best"
        / "best_program_info.json"
    )
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps({"metrics": metrics}))
    return info_path


def test_aggregate_uses_expected_denominator_and_reports_missing_and_per_level(tmp_path):
    namespace = "openevolve/example"
    run_key = "individual/model/run-a"
    tasks = [
        "kernelbench_l1_p1",
        "kernelbench_l1_p2",
        "kernelbench_l2_p3",
        "kernelbench_l3_p4",
    ]
    _write_result(
        tmp_path,
        tasks[0],
        namespace,
        run_key,
        {
            # Raw timing passes despite the rounded 1.00x value.
            "test_speedup_over_eager": 1.00,
            "test_ref_eager_time_ms": 1.0001,
            "test_kernel_time_ms": 1.0,
        },
    )
    _write_result(
        tmp_path,
        tasks[1],
        namespace,
        run_key,
        {
            # A strict tie is zero despite the rounded 1.01x value.
            "test_speedup_over_eager": 1.01,
            "test_ref_eager_time_ms": 1.0,
            "test_kernel_time_ms": 1.0,
        },
    )
    _write_result(
        tmp_path,
        tasks[2],
        namespace,
        run_key,
        {"test_combined_score": -100.0},
    )
    # tasks[3] is intentionally absent. Missing results stay in the denominator.

    summary = aggregate_fast1(tmp_path, namespace, tasks)
    run = summary["runs"][run_key]

    assert summary["expected_tasks"] == tasks
    assert summary["expected_task_count"] == 4
    assert run["passed"] == 1
    assert run["completed"] == 3
    assert run["total"] == 4
    assert run["fast_1"] == pytest.approx(0.25)
    assert run["fast_1_percent"] == pytest.approx(25.0)
    assert run["complete"] is False
    assert run["missing_tasks"] == [tasks[3]]
    assert run["tasks"][tasks[2]]["available"] is True
    assert run["tasks"][tasks[2]]["fast_1"] == 0.0
    assert run["tasks"][tasks[3]]["status"] == "missing_result"

    assert run["by_level"]["1"] == {
        "fast_1": 0.5,
        "fast_1_percent": 50.0,
        "passed": 1,
        "completed": 2,
        "total": 2,
        "complete": True,
    }
    assert run["by_level"]["2"]["fast_1"] == 0.0
    assert run["by_level"]["2"]["completed"] == 1
    assert run["by_level"]["2"]["complete"] is True
    assert run["by_level"]["3"]["fast_1"] == 0.0
    assert run["by_level"]["3"]["completed"] == 0
    assert run["by_level"]["3"]["complete"] is False


def test_aggregate_groups_only_matching_relative_run_keys(tmp_path):
    namespace = "topk/example"
    tasks = ["kernelbench_l1_p1", "kernelbench_l2_p2"]
    run_a = "individual/model/run-a"
    run_b = "individual/model/run-b"

    _write_result(tmp_path, tasks[0], namespace, run_a, {"test_fast_1": 1.0})
    _write_result(tmp_path, tasks[1], namespace, run_a, {"test_fast_1": 0.0})
    _write_result(tmp_path, tasks[0], namespace, run_b, {"test_fast_1": 0.0})
    _write_result(tmp_path, tasks[1], namespace, run_b, {"test_fast_1": 1.0})

    summary = aggregate_fast1(tmp_path, namespace, tasks)

    assert set(summary["runs"]) == {run_a, run_b}
    assert summary["runs"][run_a]["tasks"][tasks[0]]["fast_1"] == 1.0
    assert summary["runs"][run_a]["tasks"][tasks[1]]["fast_1"] == 0.0
    assert summary["runs"][run_b]["tasks"][tasks[0]]["fast_1"] == 0.0
    assert summary["runs"][run_b]["tasks"][tasks[1]]["fast_1"] == 1.0
    assert summary["runs"][run_a]["complete"] is True
    assert summary["runs"][run_b]["complete"] is True


def test_write_summary_replaces_file_with_valid_json_and_cleans_temporary_file(tmp_path):
    output_path = tmp_path / "nested" / "fast1_summary.json"
    output_path.parent.mkdir()
    output_path.write_text('{"stale": true}')
    summary = {"schema_version": 1, "metric": "fast_1", "runs": {}}

    write_summary(summary, output_path)

    assert json.loads(output_path.read_text()) == summary
    assert output_path.read_text().endswith("\n")
    assert list(output_path.parent.glob(f".{output_path.name}.*.tmp")) == []
