"""Tests for HarborEvaluator — solution path extraction, task.toml parsing, reward reading, and detection."""

import json
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation import _is_containerized, _is_harbor_task
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.harbor_evaluator import _DEFAULT_SOLUTION_PATH, HarborEvaluator


def _make_evaluator(task_dir: str) -> HarborEvaluator:
    """Create a HarborEvaluator without starting Docker."""
    inst = object.__new__(HarborEvaluator)
    inst.task_dir = task_dir
    return inst


# ------------------------------------------------------------------
# task.toml timeout parsing
# ------------------------------------------------------------------


class TestTaskTomlTimeout:
    def test_reads_verifier_timeout(self, tmp_path):
        (tmp_path / "task.toml").write_text("[verifier]\ntimeout_sec = 3600\n")
        inst = _make_evaluator(str(tmp_path))
        config = EvaluatorConfig()
        inst._apply_task_toml_timeout(config)
        assert config.timeout == 3600

    def test_no_task_toml_keeps_default(self, tmp_path):
        inst = _make_evaluator(str(tmp_path))
        config = EvaluatorConfig()
        inst._apply_task_toml_timeout(config)
        assert config.timeout == 360

    def test_missing_key_keeps_default(self, tmp_path):
        (tmp_path / "task.toml").write_text("[metadata]\nname = 'test'\n")
        inst = _make_evaluator(str(tmp_path))
        config = EvaluatorConfig()
        inst._apply_task_toml_timeout(config)
        assert config.timeout == 360

    def test_inline_timeout(self, tmp_path):
        (tmp_path / "task.toml").write_text("timeout_sec = 1200\n")
        inst = _make_evaluator(str(tmp_path))
        config = EvaluatorConfig()
        inst._apply_task_toml_timeout(config)
        assert config.timeout == 1200

    def test_malformed_toml_keeps_default(self, tmp_path):
        (tmp_path / "task.toml").write_bytes(b"\x80\x81\x82")
        inst = _make_evaluator(str(tmp_path))
        config = EvaluatorConfig()
        inst._apply_task_toml_timeout(config)
        assert config.timeout == 360


# ------------------------------------------------------------------
# task.toml manifest parsing
# ------------------------------------------------------------------


class TestMinimalTaskManifest:
    def test_parses_multiline_artifacts_and_separate_mode(self):
        manifest = HarborEvaluator._parse_minimal_task_manifest(textwrap.dedent("""\
                artifacts = [
                  "/results/output.json",
                  '/app/data/input.csv',
                ]

                [verifier]
                timeout_sec = 300.0
                environment_mode = "separate"
                """))

        assert manifest["artifacts"] == [
            "/results/output.json",
            "/app/data/input.csv",
        ]
        assert manifest["verifier"]["environment_mode"] == "separate"


# ------------------------------------------------------------------
# Solution path extraction: solve.sh (tier 1)
# ------------------------------------------------------------------


class TestExtractPathFromSolveSh:
    def _write_solve_sh(self, tmp_path, content: str):
        solution_dir = tmp_path / "solution"
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / "solve.sh").write_text(content)
        return _make_evaluator(str(tmp_path))

    def test_absolute_cat_redirect(self, tmp_path):
        inst = self._write_solve_sh(tmp_path, "cat > /app/solver.py << 'EOF'\nprint('hi')\nEOF\n")
        assert inst._extract_path_from_solve_sh() == "/app/solver.py"

    def test_bare_redirect(self, tmp_path):
        inst = self._write_solve_sh(tmp_path, "> /workspace/solution.py << 'EOF'\ncode\nEOF\n")
        assert inst._extract_path_from_solve_sh() == "/workspace/solution.py"

    def test_rust_extension(self, tmp_path):
        inst = self._write_solve_sh(tmp_path, "cat > /app/src/main.rs << 'EOF'\nfn main(){}\nEOF\n")
        assert inst._extract_path_from_solve_sh() == "/app/src/main.rs"

    def test_cpp_extension(self, tmp_path):
        inst = self._write_solve_sh(
            tmp_path, "cat > /solution/solve.cpp << 'EOF'\nint main(){}\nEOF\n"
        )
        assert inst._extract_path_from_solve_sh() == "/solution/solve.cpp"

    def test_relative_path_with_cd(self, tmp_path):
        content = textwrap.dedent("""\
            #!/bin/bash
            cd "/workspace/project"
            cat > src/interfaces/base.rs << 'EOF'
            code
            EOF
        """)
        inst = self._write_solve_sh(tmp_path, content)
        assert inst._extract_path_from_solve_sh() == "/workspace/project/src/interfaces/base.rs"

    def test_relative_path_with_variable_assignment(self, tmp_path):
        content = textwrap.dedent("""\
            #!/bin/bash
            RBENCH_DIR="/workspace/rbench_reference"
            cat > src/main.py << 'EOF'
            code
            EOF
        """)
        inst = self._write_solve_sh(tmp_path, content)
        assert inst._extract_path_from_solve_sh() == "/workspace/rbench_reference/src/main.py"

    def test_relative_path_with_dockerfile_workdir(self, tmp_path):
        (tmp_path / "solution").mkdir()
        (tmp_path / "solution" / "solve.sh").write_text("cat > solver.py << 'EOF'\ncode\nEOF\n")
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.11\nWORKDIR /opt/app\n")
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_path_from_solve_sh() == "/opt/app/solver.py"

    def test_no_solve_sh_returns_empty(self, tmp_path):
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_path_from_solve_sh() == ""

    def test_no_redirect_returns_empty(self, tmp_path):
        inst = self._write_solve_sh(tmp_path, "#!/bin/bash\necho hello\n")
        assert inst._extract_path_from_solve_sh() == ""

    def test_candidate_referenced_under_solution(self, tmp_path):
        inst = self._write_solve_sh(
            tmp_path,
            "uv run --no-project python /solution/solve.py\n",
        )
        assert inst._extract_path_from_solve_sh() == "/solution/solve.py"


# ------------------------------------------------------------------
# Solution path extraction: instruction.md (tier 2)
# ------------------------------------------------------------------


class TestExtractPathFromInstruction:
    def test_backtick_path(self, tmp_path):
        (tmp_path / "instruction.md").write_text("Write your solution in `/app/solver.py`.\n")
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_path_from_instruction() == "/app/solver.py"

    def test_quoted_path(self, tmp_path):
        (tmp_path / "instruction.md").write_text('Save your code to "/workspace/solve.py".\n')
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_path_from_instruction() == "/workspace/solve.py"

    def test_preposition_path(self, tmp_path):
        (tmp_path / "instruction.md").write_text(
            "Place your solution at /opt/solution.py and run it.\n"
        )
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_path_from_instruction() == "/opt/solution.py"

    def test_no_path_returns_empty(self, tmp_path):
        (tmp_path / "instruction.md").write_text("Solve this problem efficiently.\n")
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_path_from_instruction() == ""

    def test_no_file_returns_empty(self, tmp_path):
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_path_from_instruction() == ""


# ------------------------------------------------------------------
# Full solution path extraction (tier priority)
# ------------------------------------------------------------------


class TestExtractSolutionPath:
    def test_prefers_solve_sh_over_instruction(self, tmp_path):
        (tmp_path / "solution").mkdir()
        (tmp_path / "solution" / "solve.sh").write_text("cat > /from/solve.py << 'EOF'\nEOF\n")
        (tmp_path / "instruction.md").write_text("Write to `/from/instruction.py`.\n")
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_solution_path() == "/from/solve.py"

    def test_falls_back_to_instruction(self, tmp_path):
        (tmp_path / "instruction.md").write_text("Write to `/from/instruction.py`.\n")
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_solution_path() == "/from/instruction.py"

    def test_falls_back_to_default(self, tmp_path):
        inst = _make_evaluator(str(tmp_path))
        assert inst._extract_solution_path() == _DEFAULT_SOLUTION_PATH


# ------------------------------------------------------------------
# _read_reward
# ------------------------------------------------------------------


def _mock_docker_exec(outputs: dict):
    """Return a side_effect for subprocess.run that fakes `docker exec ... cat <path>`.

    Args:
        outputs: mapping from container path to (returncode, stdout) tuples.
    """

    def side_effect(cmd, **kwargs):
        # Detect "docker exec <cid> cat <path>" calls.
        if cmd[:2] == ["docker", "exec"] and "cat" in cmd:
            path = cmd[-1]
            if path in outputs:
                rc, stdout = outputs[path]
                return MagicMock(returncode=rc, stdout=stdout)
        return MagicMock(returncode=1, stdout="")

    return side_effect


class TestReadReward:
    def _make_inst(self):
        inst = object.__new__(HarborEvaluator)
        inst.container_id = "fake_container"
        return inst

    def test_reads_reward_txt(self):
        inst = self._make_inst()
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (1, ""),
                    "/logs/verifier/reward.txt": (0, "0.75\n"),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.75

    def test_reads_reward_json_with_reward_key(self):
        inst = self._make_inst()
        payload = json.dumps({"reward": 0.9, "time_ms": 123})
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (0, payload),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.9
        assert result.metrics["time_ms"] == 123.0

    def test_reads_reward_json_with_score_key(self):
        inst = self._make_inst()
        payload = json.dumps({"score": 0.5})
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (0, payload),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.5

    def test_json_preferred_over_txt(self):
        inst = self._make_inst()
        payload = json.dumps({"reward": 0.9})
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (0, payload),
                    "/logs/verifier/reward.txt": (0, "0.1\n"),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.9

    def test_missing_reward_key_defaults_to_zero(self):
        inst = self._make_inst()
        payload = json.dumps({"time_ms": 500})
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (0, payload),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.0

    def test_no_reward_files_returns_zero(self):
        inst = self._make_inst()
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (1, ""),
                    "/logs/verifier/reward.txt": (1, ""),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.0
        assert "error" in result.artifacts

    def test_malformed_json_falls_back_to_txt(self):
        inst = self._make_inst()
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (0, "{bad json"),
                    "/logs/verifier/reward.txt": (0, "0.42\n"),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.42

    def test_non_numeric_txt_falls_through(self):
        inst = self._make_inst()
        with patch(
            "subprocess.run",
            side_effect=_mock_docker_exec(
                {
                    "/logs/verifier/reward.json": (1, ""),
                    "/logs/verifier/reward.txt": (0, "not a number"),
                }
            ),
        ):
            result = inst._read_reward()
        assert result.metrics["combined_score"] == 0.0
        assert "error" in result.artifacts


# ------------------------------------------------------------------
# CTRF reward shaping
# ------------------------------------------------------------------


class TestCtrfRewardShaping:
    def _make_inst(self):
        inst = object.__new__(HarborEvaluator)
        inst.container_id = "fake_container"
        inst.reward_mode = "ctrf"
        inst.partial_credit_cap = 0.99
        return inst

    def test_train_mode_uses_capped_pass_ratio(self):
        inst = self._make_inst()
        ctrf = json.dumps({"results": {"summary": {"tests": 12, "passed": 9, "failed": 3}}})
        result = EvaluationResult(metrics={"combined_score": 0.0})
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=ctrf),
        ):
            inst._apply_ctrf_metrics(result, "verifier", "train")

        assert result.metrics["official_reward"] == 0.0
        assert result.metrics["verifier_pass_rate"] == 0.75
        assert result.metrics["combined_score"] == pytest.approx(0.7425)

    def test_test_mode_preserves_official_reward(self):
        inst = self._make_inst()
        ctrf = json.dumps({"results": {"summary": {"tests": 12, "passed": 9, "failed": 3}}})
        result = EvaluationResult(metrics={"combined_score": 0.0})
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=ctrf),
        ):
            inst._apply_ctrf_metrics(result, "verifier", "test")

        assert result.metrics["verifier_pass_rate"] == 0.75
        assert result.metrics["combined_score"] == 0.0

    def test_official_pass_remains_one(self):
        inst = self._make_inst()
        ctrf = json.dumps({"results": {"summary": {"tests": 12, "passed": 12, "failed": 0}}})
        result = EvaluationResult(metrics={"combined_score": 1.0})
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=ctrf),
        ):
            inst._apply_ctrf_metrics(result, "verifier", "train")

        assert result.metrics["combined_score"] == 1.0


# ------------------------------------------------------------------
# Continuous metric reward shaping
# ------------------------------------------------------------------


class TestMetricRewardShaping:
    def _make_inst(self):
        inst = object.__new__(HarborEvaluator)
        inst.container_id = "fake_container"
        inst.reward_mode = "metric"
        inst.partial_credit_cap = 0.99
        inst.metric_path = "/logs/verifier/metrics.json"
        inst.metric_upper_limits = {
            "overall_rmse": 1.75,
            "max_band_rmse": 2.1,
        }
        return inst

    def test_train_mode_uses_worst_constraint_ratio(self):
        inst = self._make_inst()
        metrics = json.dumps(
            {
                "overall_rmse": 3.5,
                "max_band_rmse": 2.1,
                "band_rmse": {"0-5m": 2.0},
                "description": "ignored",
            }
        )
        result = EvaluationResult(metrics={"combined_score": 0.0})
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=metrics),
        ):
            inst._apply_metric_metrics(result, "verifier", "train")

        assert result.metrics["official_reward"] == 0.0
        assert result.metrics["verifier_overall_rmse"] == 3.5
        assert result.metrics["verifier_band_rmse_0_5m"] == 2.0
        assert result.metrics["verifier_worst_limit_ratio"] == 2.0
        assert result.metrics["combined_score"] == pytest.approx(0.495)

    def test_test_mode_preserves_official_reward(self):
        inst = self._make_inst()
        metrics = json.dumps({"overall_rmse": 3.5, "max_band_rmse": 2.1})
        result = EvaluationResult(metrics={"combined_score": 0.0})
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=metrics),
        ):
            inst._apply_metric_metrics(result, "verifier", "test")

        assert result.metrics["verifier_worst_limit_ratio"] == 2.0
        assert result.metrics["combined_score"] == 0.0

    def test_official_pass_remains_one(self):
        inst = self._make_inst()
        metrics = json.dumps({"overall_rmse": 1.0, "max_band_rmse": 1.2})
        result = EvaluationResult(metrics={"combined_score": 1.0})
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=metrics),
        ):
            inst._apply_metric_metrics(result, "verifier", "train")

        assert result.metrics["combined_score"] == 1.0

    def test_missing_required_metric_leaves_binary_reward(self):
        inst = self._make_inst()
        metrics = json.dumps({"overall_rmse": 3.5})
        result = EvaluationResult(metrics={"combined_score": 0.0})
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout=metrics),
        ):
            inst._apply_metric_metrics(result, "verifier", "train")

        assert result.metrics["verifier_overall_rmse"] == 3.5
        assert result.metrics["combined_score"] == 0.0


# ------------------------------------------------------------------
# Harbor task detection
# ------------------------------------------------------------------


def _make_harbor_dir(tmp_path):
    """Create a minimal valid Harbor task directory."""
    (tmp_path / "instruction.md").write_text("problem")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.11\n")
    return str(tmp_path)


class TestHarborTaskDetection:
    def test_valid_harbor_task(self, tmp_path):
        assert _is_harbor_task(_make_harbor_dir(tmp_path)) is True

    def test_missing_instruction_md(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.11\n")
        assert _is_harbor_task(str(tmp_path)) is False

    def test_missing_tests_dir(self, tmp_path):
        (tmp_path / "instruction.md").write_text("problem")
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.11\n")
        assert _is_harbor_task(str(tmp_path)) is False

    def test_missing_test_sh(self, tmp_path):
        (tmp_path / "instruction.md").write_text("problem")
        (tmp_path / "tests").mkdir()
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text("FROM python:3.11\n")
        assert _is_harbor_task(str(tmp_path)) is False

    def test_missing_dockerfile(self, tmp_path):
        (tmp_path / "instruction.md").write_text("problem")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test.sh").write_text("#!/bin/bash\n")
        (tmp_path / "environment").mkdir()
        assert _is_harbor_task(str(tmp_path)) is False

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_text("hi")
        assert _is_harbor_task(str(f)) is False


class TestContainerizedDetection:
    def test_valid_containerized(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
        (tmp_path / "evaluate.sh").write_text("#!/bin/bash\n")
        assert _is_containerized(str(tmp_path)) is True

    def test_missing_evaluate_sh(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
        assert _is_containerized(str(tmp_path)) is False

    def test_missing_dockerfile(self, tmp_path):
        (tmp_path / "evaluate.sh").write_text("#!/bin/bash\n")
        assert _is_containerized(str(tmp_path)) is False


class TestDetectionPriority:
    """A dir that matches both Harbor and containerized should be detected as Harbor."""

    def test_harbor_wins_over_containerized(self, tmp_path):
        # Set up Harbor structure.
        _make_harbor_dir(tmp_path)
        # Also add containerized markers at root.
        (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
        (tmp_path / "evaluate.sh").write_text("#!/bin/bash\n")

        assert _is_harbor_task(str(tmp_path)) is True
        assert _is_containerized(str(tmp_path)) is True
        # create_evaluator checks harbor first — verify the detection functions
        # agree that both match, confirming the ordering in create_evaluator matters.
