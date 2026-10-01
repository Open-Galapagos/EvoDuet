"""Tests for the evaluate_final() test-mode entrypoint."""

import asyncio
import json
import textwrap

from skydiscover.config import Config, EvaluatorConfig, load_config
from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator
from skydiscover.evaluation.evaluator import Evaluator
from skydiscover.evaluation.final_eval import _file_suffix, run_final_evaluation

WITH_FINAL = textwrap.dedent(
    """
    def evaluate(program_path):
        return {"combined_score": 0.1, "split": 0.0}

    def evaluate_stage1(program_path):
        return {"combined_score": 0.9, "split": 2.0}

    def evaluate_stage2(program_path):
        return {"combined_score": 0.9, "split": 2.0}

    def evaluate_final(program_path):
        return {"combined_score": 0.7, "split": 1.0}
    """
)

WITHOUT_FINAL = textwrap.dedent(
    """
    def evaluate(program_path):
        return {"combined_score": 0.1, "split": 0.0}
    """
)

SLOW_FINAL = textwrap.dedent(
    """
    import time

    def evaluate(program_path):
        return {"combined_score": 0.1}

    def evaluate_final(program_path):
        time.sleep(5)
        return {"combined_score": 0.7}
    """
)


def _make_evaluator(tmp_path, source, **config_kwargs):
    eval_file = tmp_path / "evaluator.py"
    eval_file.write_text(source)
    config_kwargs.setdefault("cascade_evaluation", False)
    config = EvaluatorConfig(
        evaluation_file=str(eval_file),
        max_retries=0,
        **config_kwargs,
    )
    return Evaluator(config)


def _evaluate(evaluator, mode):
    return asyncio.run(evaluator.evaluate_program("# candidate", "p1", mode=mode))


class TestFinalDispatch:
    def test_test_mode_uses_evaluate_final(self, tmp_path):
        evaluator = _make_evaluator(tmp_path, WITH_FINAL)
        try:
            assert _evaluate(evaluator, "test").metrics["split"] == 1.0
        finally:
            evaluator.close()

    def test_train_mode_never_uses_evaluate_final(self, tmp_path):
        evaluator = _make_evaluator(tmp_path, WITH_FINAL)
        try:
            assert _evaluate(evaluator, "train").metrics["split"] == 0.0
        finally:
            evaluator.close()

    def test_test_mode_without_evaluate_final_reruns_evaluate(self, tmp_path):
        evaluator = _make_evaluator(tmp_path, WITHOUT_FINAL)
        try:
            assert _evaluate(evaluator, "test").metrics["split"] == 0.0
        finally:
            evaluator.close()

    def test_evaluate_final_takes_precedence_over_cascade(self, tmp_path):
        evaluator = _make_evaluator(tmp_path, WITH_FINAL, cascade_evaluation=True)
        try:
            # split == 2.0 would mean the cascade ran instead.
            assert _evaluate(evaluator, "test").metrics["split"] == 1.0
            assert _evaluate(evaluator, "train").metrics["split"] == 2.0
        finally:
            evaluator.close()

    def test_absent_evaluate_final_is_none(self, tmp_path):
        evaluator = _make_evaluator(tmp_path, WITHOUT_FINAL)
        try:
            assert evaluator.evaluate_final_function is None
        finally:
            evaluator.close()


class TestFinalTimeout:
    def test_final_timeout_applies_instead_of_timeout(self, tmp_path):
        # The ordinary timeout is generous; only final_timeout should bite.
        evaluator = _make_evaluator(tmp_path, SLOW_FINAL, timeout=60, final_timeout=1)
        try:
            assert _evaluate(evaluator, "test").metrics["timeout"] is True
        finally:
            evaluator.close()

    def test_train_path_unaffected_by_final_timeout(self, tmp_path):
        evaluator = _make_evaluator(tmp_path, SLOW_FINAL, timeout=60, final_timeout=1)
        try:
            assert _evaluate(evaluator, "train").metrics["combined_score"] == 0.1
        finally:
            evaluator.close()


class TestConfigDefaults:
    def test_final_evaluation_on_by_default(self):
        assert EvaluatorConfig().final_evaluation is True

    def test_final_timeout_exceeds_timeout(self):
        config = EvaluatorConfig()
        assert config.final_timeout > config.timeout


class TestExternalBackendHelper:
    """run_final_evaluation() — the path external backends take."""

    def _config(self, tmp_path, source, **kwargs):
        eval_file = tmp_path / "evaluator.py"
        eval_file.write_text(source)
        kwargs.setdefault("cascade_evaluation", False)
        return Config(
            evaluator=EvaluatorConfig(
                evaluation_file=str(eval_file), max_retries=0, **kwargs
            )
        )

    def test_metrics_are_test_prefixed_and_written(self, tmp_path):
        config = self._config(tmp_path, WITH_FINAL)
        metrics = asyncio.run(
            run_final_evaluation("# candidate", config, str(tmp_path))
        )
        assert metrics["test_split"] == 1.0
        written = json.loads((tmp_path / "final_evaluation.json").read_text())
        assert written == metrics

    def test_disabled_returns_empty(self, tmp_path):
        config = self._config(tmp_path, WITH_FINAL, final_evaluation=False)
        assert asyncio.run(run_final_evaluation("# candidate", config, None)) == {}

    def test_no_solution_returns_empty(self, tmp_path):
        config = self._config(tmp_path, WITH_FINAL)
        assert asyncio.run(run_final_evaluation("", config, None)) == {}

    def test_failure_does_not_raise(self, tmp_path):
        config = self._config(tmp_path, "def evaluate(p):\n    return {}\n")
        config.evaluator.evaluation_file = str(tmp_path / "missing.py")
        assert asyncio.run(run_final_evaluation("# candidate", config, None)) == {}


class TestExternalBackendConfigFromYaml:
    """The external path has no controller, so final_eval must fill evaluator config itself."""

    def _yaml_config(self, tmp_path, language="cpp"):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(f"language: {language}\n")
        return load_config(str(cfg))

    def test_yaml_config_has_no_evaluation_file(self, tmp_path):
        # The precondition that made the first implementation dead code.
        assert self._yaml_config(tmp_path).evaluator.evaluation_file is None

    def test_evaluator_path_argument_drives_the_run(self, tmp_path):
        eval_file = tmp_path / "evaluator.py"
        eval_file.write_text(WITH_FINAL)
        config = self._yaml_config(tmp_path)
        metrics = asyncio.run(
            run_final_evaluation(
                "# candidate",
                config,
                str(tmp_path),
                evaluator_path=str(eval_file),
                program_path=str(tmp_path / "initial_program.cpp"),
            )
        )
        assert metrics["test_split"] == 1.0

    def test_file_suffix_follows_the_seed_program(self, tmp_path):
        config = self._yaml_config(tmp_path)
        assert config.evaluator.file_suffix == ".py"
        assert _file_suffix(config, str(tmp_path / "initial_program.cpp")) == ".cpp"

    def test_file_suffix_defaults_to_py(self, tmp_path):
        assert _file_suffix(self._yaml_config(tmp_path, "python"), None) == ".py"


class TestContainerFinalTimeout:
    def test_test_mode_uses_final_timeout(self):
        evaluator = object.__new__(ContainerizedEvaluator)
        evaluator.config = EvaluatorConfig(timeout=360, final_timeout=7200)
        assert evaluator._timeout_for("test") == 7200
        assert evaluator._timeout_for("train") == 360
