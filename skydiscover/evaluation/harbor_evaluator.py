"""Harbor evaluator for Harbor/Frontier-Bench task directories.

Harbor tasks use a different container protocol from the standard
ContainerizedEvaluator:

  - Solution is injected at a task-specific path (e.g. ``/app/solver.py``)
    extracted from ``solution/solve.sh`` or ``instruction.md``.
  - Tasks may execute the candidate in an environment container and verify
    only declared artifacts in a separate, trusted verifier container.
  - Evaluation runs ``tests/test.sh`` instead of ``evaluate.sh``.
  - The reward is read from ``/logs/verifier/reward.txt`` (float) or
    ``/logs/verifier/reward.json`` (dict) instead of JSON on stdout.

A Harbor task directory has this structure::

    task_dir/
    ├── task.toml              # metadata, timeouts
    ├── instruction.md         # problem description (shown to LLM)
    ├── environment/
    │   └── Dockerfile
    ├── tests/
    │   ├── test.sh            # verification entrypoint
    │   └── ...                # supporting test files
    └── solution/              # reference solution (optional, never shown to LLM)
        └── solve.sh

Each evaluation gets fresh containers.  This both matches Harbor trial
isolation and prevents one generated candidate from poisoning a later one.
"""

import json
import logging
import math
import os
import re
import shlex
import subprocess
import tempfile
import threading

from skydiscover.evaluation.container_evaluator import ContainerizedEvaluator
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.utils.async_utils import TaskPool

logger = logging.getLogger(__name__)

# Most common solution path across Harbor benchmarks — used as fallback.
_DEFAULT_SOLUTION_PATH = "/app/solution.py"


class HarborEvaluator(ContainerizedEvaluator):
    """Evaluates programs using the Harbor container protocol.

    ``environment_mode = "separate"`` is implemented with two images:

    1. the untrusted candidate runs in ``environment/Dockerfile``;
    2. only paths declared by ``artifacts`` in ``task.toml`` are copied to a
       fresh container built from ``tests/Dockerfile``;
    3. ``tests/test.sh`` and reward files live exclusively in that verifier.

    For single-file discovery tasks, ``EvaluatorConfig.harbor_run_command``
    can execute the injected candidate before artifact transfer.  Code-only
    tasks can leave it unset and let the verifier compile/import the artifact.
    """

    def __init__(self, benchmark_dir, config, max_concurrent=4, env_vars=None):
        self.task_dir = os.path.abspath(benchmark_dir)
        self.benchmark_dir = self.task_dir
        self.manifest = self._load_task_manifest()

        self._apply_task_toml_timeout(config)
        self.verifier_timeout = int(config.timeout)
        self.config = config
        self.program_suffix = config.file_suffix
        self.task_pool = TaskPool(max_concurrency=max_concurrent)
        self.llm_judge = None
        self.env_vars = dict(env_vars or {})

        configured_solution_path = getattr(config, "harbor_solution_path", None)
        self.solution_path = configured_solution_path or self._extract_solution_path()
        self.run_command = getattr(config, "harbor_run_command", None)
        self.reward_mode = getattr(config, "harbor_reward_mode", "official")
        self.partial_credit_cap = float(getattr(config, "harbor_partial_credit_cap", 0.99))
        self.metric_path = getattr(config, "harbor_metric_path", None)
        raw_metric_limits = getattr(config, "harbor_metric_upper_limits", {}) or {}
        if not isinstance(raw_metric_limits, dict):
            raise ValueError("evaluator.harbor_metric_upper_limits must be a mapping")
        self.metric_upper_limits = {
            str(key): float(value) for key, value in raw_metric_limits.items()
        }
        self.expose_verifier_output = bool(getattr(config, "harbor_expose_verifier_output", True))
        self.network_mode = str(getattr(config, "harbor_network_mode", "default"))
        self.candidate_timeout = int(
            getattr(config, "harbor_candidate_timeout", None) or self.verifier_timeout
        )
        # ContainerizedEvaluator's async wrapper uses config.timeout around the
        # whole synchronous trial.  Budget both phases plus Docker transfer
        # overhead while retaining the manifest's verifier-only timeout above.
        config.timeout = (
            self.verifier_timeout + (self.candidate_timeout if self.run_command else 0) + 30
        )
        if self.reward_mode not in {"official", "ctrf", "metric"}:
            raise ValueError("evaluator.harbor_reward_mode must be 'official', 'ctrf', or 'metric'")
        if not 0.0 < self.partial_credit_cap < 1.0:
            raise ValueError("evaluator.harbor_partial_credit_cap must be between 0 and 1")
        if self.reward_mode == "metric" and not self.metric_path:
            raise ValueError(
                "evaluator.harbor_metric_path is required when " "harbor_reward_mode='metric'"
            )
        if self.reward_mode == "metric" and not self.metric_upper_limits:
            raise ValueError(
                "evaluator.harbor_metric_upper_limits is required when "
                "harbor_reward_mode='metric'"
            )
        if any(
            not math.isfinite(limit) or limit <= 0.0 for limit in self.metric_upper_limits.values()
        ):
            raise ValueError(
                "all evaluator.harbor_metric_upper_limits values must be "
                "finite and greater than zero"
            )
        if self.network_mode not in {"default", "none"}:
            raise ValueError("evaluator.harbor_network_mode must be 'default' or 'none'")

        verifier = self.manifest.get("verifier", {})
        self.environment_mode = str(verifier.get("environment_mode", "same"))
        self.artifact_paths = self._manifest_artifacts()
        self.separate_verifier = self.environment_mode == "separate"

        if self.separate_verifier and not os.path.exists(
            os.path.join(self.task_dir, "tests", "Dockerfile")
        ):
            raise RuntimeError(
                "task.toml requests verifier.environment_mode='separate', "
                "but tests/Dockerfile is missing"
            )

        self.image_tag = self._build_image()
        self.verifier_image_tag = self._build_verifier_image() if self.separate_verifier else None

        # Harbor trials use per-evaluation containers rather than the persistent
        # container used by ContainerizedEvaluator.
        self.container_id = None
        self._active_containers: set[str] = set()
        self._active_containers_lock = threading.Lock()
        logger.info(
            "HarborEvaluator ready: task=%s mode=%s artifacts=%d",
            os.path.basename(self.task_dir),
            self.environment_mode,
            len(self.artifact_paths),
        )

    # ------------------------------------------------------------------
    # Override: image building
    # ------------------------------------------------------------------

    def _build_image(self) -> str:
        """Build from environment/Dockerfile."""
        name = os.path.basename(os.path.normpath(self.task_dir))
        tag = f"skydiscover-harbor-{name}:latest"
        dockerfile_dir = os.path.join(self.task_dir, "environment")

        logger.info(f"Building Harbor image: {tag} (from {dockerfile_dir})")
        result = subprocess.run(
            ["docker", "build", "-t", tag, dockerfile_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed for {dockerfile_dir}:\n{result.stderr}")
        return tag

    def _build_verifier_image(self) -> str:
        """Build the trusted verifier image from tests/Dockerfile."""
        name = os.path.basename(os.path.normpath(self.task_dir))
        tag = f"skydiscover-harbor-{name}-verifier:latest"
        tests_dir = os.path.join(self.task_dir, "tests")

        logger.info("Building Harbor verifier image: %s (from %s)", tag, tests_dir)
        result = subprocess.run(
            ["docker", "build", "-t", tag, tests_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker verifier build failed for {tests_dir}:\n{result.stderr}")
        return tag

    # ------------------------------------------------------------------
    # Override: container interaction
    # ------------------------------------------------------------------

    def _run_container(self, program_solution: str, mode: str) -> EvaluationResult:
        """Run one isolated Harbor trial."""
        environment_id = None
        verifier_id = None
        try:
            environment_id = self._start_trial_container(self.image_tag)
            verifier_id = (
                self._start_trial_container(self.verifier_image_tag)
                if self.separate_verifier
                else environment_id
            )

            self._prepare_trial_container(environment_id, upload_tests=not self.separate_verifier)
            if verifier_id != environment_id:
                self._prepare_trial_container(verifier_id, upload_tests=False)

            injection_error = self._inject_solution(environment_id, program_solution)
            if injection_error:
                return EvaluationResult(
                    metrics={"combined_score": 0.0},
                    artifacts={"error": injection_error},
                )

            run_artifacts = {}
            if self.run_command:
                candidate_result = self._run_candidate(environment_id, mode)
                if candidate_result.stdout.strip():
                    run_artifacts["candidate_stdout"] = candidate_result.stdout
                if candidate_result.stderr.strip():
                    run_artifacts["candidate_stderr"] = candidate_result.stderr
                if candidate_result.returncode != 0:
                    run_artifacts["candidate_exit_code"] = str(candidate_result.returncode)

            if self.separate_verifier:
                transfer_error = self._transfer_artifacts(environment_id, verifier_id)
                if transfer_error:
                    run_artifacts["error"] = transfer_error
                    return EvaluationResult(
                        metrics={"combined_score": 0.0},
                        artifacts=run_artifacts,
                    )

            proc = subprocess.run(
                [
                    "docker",
                    "exec",
                    verifier_id,
                    "bash",
                    "-c",
                    "chmod +x /tests/test.sh && /tests/test.sh",
                ],
                capture_output=True,
                text=True,
                timeout=self.verifier_timeout,
            )

            # Read reward regardless of exit code — test.sh may exit non-zero
            # but still write a reward (e.g. partial credit).
            result = self._read_reward(
                proc.stdout if self.expose_verifier_output else "",
                proc.stderr if self.expose_verifier_output else "",
                container_id=verifier_id,
            )
            self._apply_ctrf_metrics(result, verifier_id, mode)
            self._apply_metric_metrics(result, verifier_id, mode)
            result.artifacts.update(
                {key: value for key, value in run_artifacts.items() if key not in result.artifacts}
            )

            if proc.returncode != 0:
                result.artifacts.setdefault("test_exit_code", str(proc.returncode))
            if self.expose_verifier_output:
                if proc.stderr.strip():
                    result.artifacts.setdefault("stderr", proc.stderr)
                if proc.stdout.strip():
                    result.artifacts.setdefault("stdout", proc.stdout)

            return result
        except subprocess.TimeoutExpired:
            logger.error(
                "Harbor trial timed out (candidate=%ss, verifier=%ss)",
                self.candidate_timeout,
                self.verifier_timeout,
            )
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": "Harbor candidate or verifier timed out"},
            )
        finally:
            self._remove_trial_container(verifier_id)
            if environment_id != verifier_id:
                self._remove_trial_container(environment_id)

    # ------------------------------------------------------------------
    # Harbor-specific helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Remove any trial containers still alive after interruption."""
        with self._active_containers_lock:
            container_ids = list(self._active_containers)
        for container_id in container_ids:
            self._remove_trial_container(container_id)

    def _load_task_manifest(self) -> dict:
        path = os.path.join(self.task_dir, "task.toml")
        if not os.path.exists(path):
            return {}
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                # Python 3.10 is supported by SkyDiscover.  The adapter only
                # needs two simple fields, so avoid a mandatory TOML backport.
                with open(path) as handle:
                    text = handle.read()
                return self._parse_minimal_task_manifest(text)
            else:
                with open(path, "rb") as handle:
                    return tomllib.load(handle)
        except Exception as exc:
            logger.warning("Failed to parse Harbor task.toml: %s", exc)
            return {}

    @staticmethod
    def _parse_minimal_task_manifest(text: str) -> dict:
        """Parse fields needed by this adapter when stdlib tomllib is absent."""
        artifacts = []
        artifacts_match = re.search(
            r"(?ms)^\s*artifacts\s*=\s*\[(.*?)\]",
            text,
        )
        if artifacts_match:
            artifacts = [
                double or single
                for double, single in re.findall(
                    r'"([^"]*)"|\'([^\']*)\'',
                    artifacts_match.group(1),
                )
            ]

        environment_mode = "same"
        verifier_match = re.search(
            r"(?ms)^\s*\[verifier\]\s*(.*?)(?=^\s*\[|\Z)",
            text,
        )
        if verifier_match:
            mode_match = re.search(
                r'(?m)^\s*environment_mode\s*=\s*["\']([^"\']+)["\']',
                verifier_match.group(1),
            )
            if mode_match:
                environment_mode = mode_match.group(1)

        return {
            "artifacts": artifacts,
            "verifier": {"environment_mode": environment_mode},
        }

    def _manifest_artifacts(self) -> list[str]:
        raw = self.manifest.get("artifacts", [])
        if not isinstance(raw, list):
            raise ValueError("Harbor task.toml 'artifacts' must be a list")

        artifacts = []
        for value in raw:
            path = str(value)
            normalized = os.path.normpath(path)
            if not path.startswith("/") or normalized == "/" or ".." in path.split("/"):
                raise ValueError(f"Unsafe Harbor artifact path: {path!r}")
            artifacts.append(path)
        return artifacts

    def _apply_task_toml_timeout(self, config) -> None:
        """Read verifier.timeout_sec from task.toml and apply it to config."""
        toml_path = os.path.join(self.task_dir, "task.toml")
        if not os.path.exists(toml_path):
            return
        try:
            with open(toml_path) as f:
                text = f.read()
            match = re.search(r"timeout_sec\s*=\s*(\d+)", text)
            if match:
                config.timeout = int(match.group(1))
                logger.debug(f"Harbor task.toml: set evaluator timeout to {config.timeout}s")
        except Exception as e:
            logger.warning(f"Failed to read task.toml: {e}")

    def _start_trial_container(self, image_tag: str) -> str:
        cmd = ["docker", "run", "-d", "--rm"]
        if self.network_mode == "none":
            cmd.extend(["--network", "none"])
        for key, value in self.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(["--entrypoint", "sleep", image_tag, "infinity"])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        container_id = result.stdout.strip()
        with self._active_containers_lock:
            self._active_containers.add(container_id)
        return container_id

    def _remove_trial_container(self, container_id: str | None) -> None:
        if not container_id:
            return
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            text=True,
        )
        with self._active_containers_lock:
            self._active_containers.discard(container_id)

    def _prepare_trial_container(self, container_id: str, *, upload_tests: bool) -> None:
        self._exec(
            "mkdir -p /logs/verifier /logs/agent /logs/artifacts",
            container_id=container_id,
            check=True,
        )
        if not upload_tests:
            return

        tests_dir = os.path.join(self.task_dir, "tests")
        if os.path.isdir(tests_dir):
            self._exec("rm -rf /tests", container_id=container_id, check=True)
            subprocess.run(
                ["docker", "cp", tests_dir, f"{container_id}:/tests"],
                capture_output=True,
                check=True,
            )
        else:
            raise RuntimeError(f"No tests/ directory found in {self.task_dir}")

    def _inject_solution(self, container_id: str, program_solution: str) -> str | None:
        parent_dir = os.path.dirname(self.solution_path)
        if parent_dir:
            proc = self._exec(
                f"mkdir -p {shlex.quote(parent_dir)}",
                container_id=container_id,
            )
            if proc.returncode != 0:
                return f"failed to create solution directory: {proc.stderr}"

        inject = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_id,
                "/bin/sh",
                "-c",
                f"cat > {shlex.quote(self.solution_path)}",
            ],
            input=program_solution.encode(),
            capture_output=True,
        )
        if inject.returncode != 0:
            return f"injection failed: {inject.stderr.decode()}"
        return None

    def _run_candidate(self, container_id: str, mode: str) -> subprocess.CompletedProcess:
        command = self.run_command.format(solution_path=shlex.quote(self.solution_path))
        return subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"SKYDISCOVER_EVALUATION_MODE={mode}",
                "-e",
                f"SKYDISCOVER_SOLUTION_PATH={self.solution_path}",
                container_id,
                "bash",
                "-lc",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=self.candidate_timeout,
        )

    def _transfer_artifacts(self, environment_id: str, verifier_id: str) -> str | None:
        if not self.artifact_paths:
            return (
                "separate Harbor verifier requires at least one declared " "artifact in task.toml"
            )

        with tempfile.TemporaryDirectory(prefix="skydiscover-harbor-artifacts-") as staging_dir:
            for index, artifact_path in enumerate(self.artifact_paths):
                is_dir = (
                    self._exec(
                        f"test -d {shlex.quote(artifact_path)}",
                        container_id=environment_id,
                    ).returncode
                    == 0
                )
                exists = (
                    self._exec(
                        f"test -e {shlex.quote(artifact_path)}",
                        container_id=environment_id,
                    ).returncode
                    == 0
                )
                if not exists:
                    return f"declared artifact was not produced: {artifact_path}"

                local_path = os.path.join(staging_dir, f"artifact-{index}")
                if is_dir:
                    os.makedirs(local_path, exist_ok=True)
                    copy_out_source = f"{environment_id}:{artifact_path.rstrip('/')}/."
                else:
                    copy_out_source = f"{environment_id}:{artifact_path}"

                copy_out = subprocess.run(
                    ["docker", "cp", copy_out_source, local_path],
                    capture_output=True,
                    text=True,
                )
                if copy_out.returncode != 0:
                    return (
                        f"failed to copy artifact out of environment "
                        f"{artifact_path}: {copy_out.stderr}"
                    )

                if is_dir:
                    mkdir = self._exec(
                        f"mkdir -p {shlex.quote(artifact_path)}",
                        container_id=verifier_id,
                    )
                    copy_in_source = f"{local_path}/."
                    copy_in_target = f"{verifier_id}:{artifact_path.rstrip('/')}/"
                else:
                    parent = os.path.dirname(artifact_path)
                    mkdir = self._exec(
                        f"mkdir -p {shlex.quote(parent)}",
                        container_id=verifier_id,
                    )
                    copy_in_source = local_path
                    copy_in_target = f"{verifier_id}:{artifact_path}"
                if mkdir.returncode != 0:
                    return (
                        f"failed to create verifier artifact path "
                        f"{artifact_path}: {mkdir.stderr}"
                    )

                copy_in = subprocess.run(
                    ["docker", "cp", copy_in_source, copy_in_target],
                    capture_output=True,
                    text=True,
                )
                if copy_in.returncode != 0:
                    return (
                        f"failed to copy artifact into verifier "
                        f"{artifact_path}: {copy_in.stderr}"
                    )
        return None

    def _read_reward(
        self,
        test_stdout: str = "",
        test_stderr: str = "",
        *,
        container_id: str | None = None,
    ) -> EvaluationResult:
        """Read the reward from /logs/verifier/reward.txt or reward.json."""
        container_id = container_id or self.container_id
        for path, is_json in [
            ("/logs/verifier/reward.json", True),
            ("/logs/verifier/reward.txt", False),
        ]:
            proc = subprocess.run(
                ["docker", "exec", container_id, "cat", path],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                continue

            try:
                if is_json:
                    data = json.loads(proc.stdout.strip())
                    raw = data.get("reward", data.get("score"))
                    if raw is None:
                        logger.warning(
                            "No 'reward' or 'score' key in %s; defaulting to 0",
                            path,
                        )
                        raw = 0
                    reward = float(raw)
                    metrics = {"combined_score": reward}
                    for k, v in data.items():
                        if isinstance(v, (int, float)) and k not in (
                            "reward",
                            "score",
                        ):
                            metrics[k] = float(v)
                    return EvaluationResult(metrics=metrics)
                else:
                    reward = float(proc.stdout.strip())
                    return EvaluationResult(metrics={"combined_score": reward})
            except (ValueError, json.JSONDecodeError, StopIteration) as e:
                logger.warning(f"Failed to parse reward from {path}: {e}")
                continue

        logger.error("No reward file found in /logs/verifier/")
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "error": "no reward file written by test.sh",
                "test_stdout": test_stdout,
                "test_stderr": test_stderr,
            },
        )

    def _apply_ctrf_metrics(self, result: EvaluationResult, container_id: str, mode: str) -> None:
        """Attach per-test CTRF metrics and optionally shape train rewards.

        The official verifier reward remains authoritative in ``test`` mode.
        During discovery, ``harbor_reward_mode='ctrf'`` maps the individual
        pytest pass ratio into ``[0, harbor_partial_credit_cap]`` until the
        official verifier passes.  This creates a useful search gradient while
        making a true pass strictly better than every partial result.
        """
        official_reward = float(result.metrics.get("combined_score", 0.0))
        result.metrics["official_reward"] = official_reward

        proc = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "cat",
                "/logs/verifier/ctrf.json",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return

        try:
            data = json.loads(proc.stdout)
            results = data.get("results", data)
            summary = results.get("summary", {})
            passed = int(summary.get("passed", 0))
            total = int(
                summary.get(
                    "tests",
                    summary.get("total", 0),
                )
            )

            if total <= 0:
                tests = results.get("tests", [])
                total = len(tests)
                passed = sum(
                    str(test.get("status", "")).lower() in {"passed", "pass", "success"}
                    for test in tests
                )
            if total <= 0:
                return

            pass_rate = passed / total
            result.metrics["verifier_tests_passed"] = float(passed)
            result.metrics["verifier_tests_total"] = float(total)
            result.metrics["verifier_pass_rate"] = pass_rate

            if mode == "train" and self.reward_mode == "ctrf" and official_reward < 1.0:
                result.metrics["combined_score"] = min(
                    self.partial_credit_cap,
                    pass_rate * self.partial_credit_cap,
                )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse CTRF report: %s", exc)

    def _apply_metric_metrics(
        self,
        result: EvaluationResult,
        container_id: str,
        mode: str,
    ) -> None:
        """Attach numeric verifier metrics and optionally shape train rewards.

        ``harbor_reward_mode='metric'`` uses lower-is-better constraints from
        ``harbor_metric_upper_limits``. A failed candidate receives the capped
        fraction of the most poorly satisfied constraint::

            cap * min(1, min(limit / measured_value))

        A complete official pass still receives ``1.0`` and test-mode scoring
        always preserves the official verifier reward.
        """
        official_reward = float(
            result.metrics.get(
                "official_reward",
                result.metrics.get("combined_score", 0.0),
            )
        )
        result.metrics["official_reward"] = official_reward
        if not self.metric_path:
            return

        proc = subprocess.run(
            ["docker", "exec", container_id, "cat", self.metric_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return

        try:
            data = json.loads(proc.stdout)
            if not isinstance(data, dict):
                raise TypeError("metric report must contain a JSON object")

            for path, value in self._flatten_numeric_metrics(data):
                safe_path = re.sub(r"[^A-Za-z0-9_]+", "_", path).strip("_")
                if safe_path:
                    result.metrics[f"verifier_{safe_path}"] = value

            ratios = []
            for path, limit in self.metric_upper_limits.items():
                raw_value = self._json_path_value(data, path)
                if isinstance(raw_value, bool):
                    raise TypeError(f"metric {path!r} is boolean, not numeric")
                measured = float(raw_value)
                if not math.isfinite(measured) or measured < 0.0:
                    raise ValueError(f"metric {path!r} must be finite and non-negative")
                ratios.append(measured / limit)

            if ratios:
                worst_ratio = max(ratios)
                result.metrics["verifier_worst_limit_ratio"] = worst_ratio
                if mode == "train" and self.reward_mode == "metric" and official_reward < 1.0:
                    constraint_fraction = min(
                        1.0,
                        1.0 / max(worst_ratio, 1e-12),
                    )
                    result.metrics["combined_score"] = self.partial_credit_cap * constraint_fraction
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse Harbor metric report: %s", exc)

    @staticmethod
    def _json_path_value(data: dict, path: str):
        value = data
        for component in path.split("."):
            if not isinstance(value, dict) or component not in value:
                raise KeyError(path)
            value = value[component]
        return value

    @classmethod
    def _flatten_numeric_metrics(cls, data: dict, prefix: str = ""):
        for key, value in data.items():
            path = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield from cls._flatten_numeric_metrics(value, path)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    yield path, numeric

    def _extract_solution_path(self) -> str:
        """Extract the expected solution file path for this Harbor task.

        Uses a three-tier strategy (most reliable first):

        1. **Parse ``solution/solve.sh``** — the authoritative reference solution
           script almost always contains a ``cat > /path/to/file`` redirect that
           reveals the exact injection path.
        2. **Parse ``instruction.md``** — look for explicit absolute paths in
           backticks or after prepositions like "in", "at", "to".
        3. **Default to ``/app/solution.py``** — the most common path across
           Harbor benchmarks (evoeval, livecodebench, usaco, etc.).
        """
        # Tier 1: parse solution/solve.sh (most reliable).
        path = self._extract_path_from_solve_sh()
        if path:
            logger.debug(f"Extracted solution path from solve.sh: {path}")
            return path

        # Tier 2: parse instruction.md.
        path = self._extract_path_from_instruction()
        if path:
            logger.debug(f"Extracted solution path from instruction.md: {path}")
            return path

        # Tier 3: default.
        logger.warning(f"Could not extract solution path, using default: {_DEFAULT_SOLUTION_PATH}")
        return _DEFAULT_SOLUTION_PATH

    def _extract_path_from_solve_sh(self) -> str:
        """Extract the solution target path from ``solution/solve.sh``.

        Looks for shell redirect patterns like ``cat > /app/solver.py``
        or ``> /workspace/solution.py``.  If the path is relative, resolves
        it against the last ``cd`` target found before the redirect.
        """
        solve_sh = os.path.join(self.task_dir, "solution", "solve.sh")
        if not os.path.exists(solve_sh):
            return ""

        try:
            with open(solve_sh) as f:
                text = f.read()
        except Exception:
            return ""

        _CODE_EXTS = r"\.(?:py|sh|js|ts|cpp|c|rs|go|java|rb)"

        # First try: absolute path redirects.
        for pattern in [
            rf"cat\s+>\s*(/\S+{_CODE_EXTS})",
            rf">\s*(/\S+{_CODE_EXTS})",
        ]:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        # A reference runner often executes a candidate already placed under
        # /solution instead of creating it with a redirect.
        referenced_solution = re.search(rf"(/solution/[A-Za-z0-9_./-]+{_CODE_EXTS})", text)
        if referenced_solution:
            return referenced_solution.group(1)

        # Second try: relative path redirect (e.g. crustbench writes to
        # src/interfaces/base122.rs after cd-ing into a project directory).
        redirect_pattern = rf"cat\s+>\s*(\S+{_CODE_EXTS})"
        redirect_match = re.search(redirect_pattern, text)
        if redirect_match:
            rel_path = redirect_match.group(1)

            # Resolve the base directory.  Strategy:
            # 1. Look for concrete absolute paths in cd commands.
            # 2. Look for absolute paths assigned to shell variables (the
            #    variable may be used with cd later — e.g. RBENCH_DIR).
            # 3. Fall back to the Dockerfile WORKDIR.
            candidates = re.findall(r'cd\s+"?(/[^"$\s]+)"?\s*$', text, re.MULTILINE)
            if not candidates:
                # Variable assignments like RBENCH_DIR="/workspace/rbench_reference"
                candidates = re.findall(r'[A-Z_]+=\s*"?(/[^"$\s]+)"?\s*$', text, re.MULTILINE)

            if candidates:
                base = candidates[0].rstrip('"')
            else:
                # Dockerfile WORKDIR fallback.
                base = "/workspace"
                dockerfile = os.path.join(self.task_dir, "environment", "Dockerfile")
                if os.path.exists(dockerfile):
                    try:
                        with open(dockerfile) as f:
                            for line in f:
                                m = re.match(r"WORKDIR\s+(/\S+)", line)
                                if m:
                                    base = m.group(1)
                    except Exception:
                        pass

            return os.path.join(base, rel_path)

        return ""

    def _extract_path_from_instruction(self) -> str:
        """Extract the solution file path from ``instruction.md``."""
        instruction_path = os.path.join(self.task_dir, "instruction.md")
        if not os.path.exists(instruction_path):
            return ""

        try:
            with open(instruction_path) as f:
                text = f.read()
        except Exception:
            return ""

        patterns = [
            r'[`"\'](/\S+\.(?:py|sh|js|ts|cpp|c|rs|go|java))[`"\']',
            r"(?:in|at|to|into)\s+(/\S+\.(?:py|sh|js|ts|cpp|c|rs|go|java))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        return ""

    def _exec(
        self,
        cmd: str,
        *,
        container_id: str | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a shell command inside a trial container."""
        container_id = container_id or self.container_id
        return subprocess.run(
            ["docker", "exec", container_id, "/bin/sh", "-c", cmd],
            capture_output=True,
            text=True,
            check=check,
        )
