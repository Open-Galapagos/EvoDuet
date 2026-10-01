"""Tests for EvaluationResult dataclass and shared failure classification."""

import math

import pytest

from skydiscover.evaluation.evaluation_result import (
    EvaluationResult,
    evaluation_failure_reason,
)


class TestEvaluationResult:
    def test_from_dict(self):
        result = EvaluationResult.from_dict({"score": 0.5})
        assert result.metrics == {"score": 0.5}
        assert result.artifacts == {}

    def test_to_dict_without_artifacts(self):
        result = EvaluationResult(metrics={"score": 0.5})
        assert result.to_dict() == {"score": 0.5}

    def test_to_dict_with_artifacts(self):
        result = EvaluationResult(
            metrics={"score": 0.5},
            artifacts={"log": "ok"},
        )
        d = result.to_dict()
        assert d["score"] == 0.5
        assert d["artifacts"] == {"log": "ok"}

    def test_default_artifacts_empty(self):
        result = EvaluationResult(metrics={})
        assert result.artifacts == {}

    def test_to_dict_includes_provider_reasoning_when_present(self):
        reasoning = {"calls": [{"llm_call_id": "judge-call"}]}
        result = EvaluationResult(
            metrics={"score": 0.5},
            llm_reasoning_content="provider reasoning",
            llm_reasoning=reasoning,
        )

        assert result.to_dict()["llm_reasoning_content"] == "provider reasoning"
        assert result.to_dict()["llm_reasoning"] == reasoning


@pytest.mark.parametrize(
    ("metrics", "artifacts"),
    [
        ({"combined_score": -100.0, "error": "kernel did not compile"}, {}),
        ({"combined_score": 0.0, "error_message": "invalid candidate"}, {}),
        ({"combined_score": 1.0, "validity": 0}, {}),
        ({"combined_score": 1.0, "timeout": True}, {}),
        ({"combined_score": 1.0, "evaluation_failed": 1.0}, {}),
        ({"combined_score": 1.0}, {"status": "error"}),
        ({"combined_score": 1.0}, {"error": "container failed"}),
        ({"error": 0.0}, {}),
        ({"combined_score": math.nan}, {}),
    ],
)
def test_evaluation_failure_reason_recognizes_explicit_failures(metrics, artifacts):
    assert evaluation_failure_reason(metrics, artifacts)


@pytest.mark.parametrize(
    ("metrics", "artifacts"),
    [
        ({"combined_score": -2.5}, {}),
        ({"combined_score": 0.0}, {}),
        ({"combined_score": 0.8, "error": 0.0}, {}),
        ({"combined_score": 0.8}, {"status": "success"}),
    ],
)
def test_evaluation_failure_reason_allows_scores_without_failure_markers(metrics, artifacts):
    assert evaluation_failure_reason(metrics, artifacts) is None
