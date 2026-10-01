"""Native Frontier-CS evaluator for the locally bundled algorithmic problems.

The candidate and the official checker are compiled and executed directly on
the host.  This evaluator intentionally does not contact the judge HTTP service.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
ALGORITHMIC_ROOT = ROOT / "Frontier-CS" / "algorithmic"
PROBLEMS_ROOT = ALGORITHMIC_ROOT / "problems"
TESTLIB_INCLUDE = ALGORITHMIC_ROOT / "judge" / "include"

RATIO_RE = re.compile(r"Ratio:\s*([-+0-9.eE]+)")
UNBOUNDED_RE = re.compile(r"RatioUnbounded:\s*([-+0-9.eE]+)")


def _cache_root() -> Path:
    configured = os.environ.get("FRONTIERCS_NATIVE_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "skydiscover" / "frontiercs-native"


def _compiler() -> str:
    requested = os.environ.get("CXX", "g++")
    resolved = shutil.which(requested)
    if resolved is None:
        raise RuntimeError(f"native Frontier-CS evaluation requires a C++ compiler: {requested}")
    return resolved


def _compile(source: Path, output: Path, *, checker: bool) -> None:
    command = [
        _compiler(),
        "-std=gnu++17",
        "-O2",
        "-pipe",
        "-pthread",
    ]
    if checker:
        command.extend([f"-I{TESTLIB_INCLUDE}"])
    command.extend([str(source), "-o", str(output)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        kind = "checker" if checker else "candidate"
        raise RuntimeError(f"{kind} compilation failed:\n{result.stderr[-8000:]}")


def _checker_binary(problem_id: str, checker_source: Path) -> Path:
    digest = hashlib.sha256()
    digest.update(checker_source.read_bytes())
    digest.update((TESTLIB_INCLUDE / "testlib.h").read_bytes())
    digest.update(_compiler().encode())
    cache_dir = _cache_root() / "checkers" / problem_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / f"checker-{digest.hexdigest()[:20]}"
    lock_path = cache_dir / ".build.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not binary.is_file():
            temporary = cache_dir / f".{binary.name}.{os.getpid()}.tmp"
            try:
                _compile(checker_source, temporary, checker=True)
                temporary.chmod(0o755)
                os.replace(temporary, binary)
            finally:
                temporary.unlink(missing_ok=True)
    return binary


def _duration_seconds(value: object) -> float:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?\s*", str(value))
    if match is None:
        raise ValueError(f"unsupported Frontier-CS time limit: {value!r}")
    amount = float(match.group(1))
    return amount / 1000.0 if match.group(2) == "ms" else amount


def _memory_bytes(value: object) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?)(?:i?b)?\s*", str(value), re.I)
    if match is None:
        raise ValueError(f"unsupported Frontier-CS memory limit: {value!r}")
    powers = {"": 0, "k": 1, "m": 2, "g": 3, "t": 4}
    return int(float(match.group(1)) * (1024 ** powers[match.group(2).lower()]))


def _run_case(
    candidate: Path,
    checker: Path,
    input_path: Path,
    answer_path: Path,
    time_limit: float,
    memory_limit: int,
) -> dict:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="frontiercs-native-case-") as temporary_dir:
        output_path = Path(temporary_dir) / "candidate.out"
        stderr_path = Path(temporary_dir) / "candidate.err"
        cpu_limit = max(1, math.ceil(time_limit))
        wall_limit = max(1.0, time_limit * 2.0)
        command = [
            "timeout",
            "--kill-after=1",
            str(wall_limit),
            "prlimit",
            f"--as={memory_limit}",
            f"--cpu={cpu_limit}",
            "--",
            str(candidate),
        ]
        with input_path.open("rb") as stdin_file, output_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            run = subprocess.run(command, stdin=stdin_file, stdout=stdout_file, stderr=stderr_file)
        elapsed = time.perf_counter() - started
        candidate_stderr = stderr_path.read_text(errors="replace")[-4000:]
        if run.returncode != 0:
            status = "timeout" if run.returncode in {124, 137} else "runtime_error"
            return {
                "ratio": 0.0,
                "ratio_unbounded": 0.0,
                "status": status,
                "message": candidate_stderr,
                "duration_seconds": elapsed,
            }

        checked = subprocess.run(
            [str(checker), str(input_path), str(output_path), str(answer_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        message = (checked.stdout + "\n" + checked.stderr).strip()
        scoring_verdict = checked.returncode in {0, 7}
        ratio_match = RATIO_RE.search(message) if scoring_verdict else None
        unbounded_match = UNBOUNDED_RE.search(message) if scoring_verdict else None
        ratio = float(ratio_match.group(1)) if ratio_match else (1.0 if checked.returncode == 0 else 0.0)
        ratio_unbounded = float(unbounded_match.group(1)) if unbounded_match else ratio
        if not math.isfinite(ratio):
            ratio = 0.0
        if not math.isfinite(ratio_unbounded):
            ratio_unbounded = ratio
        return {
            "ratio": min(1.0, max(0.0, ratio)),
            "ratio_unbounded": ratio_unbounded,
            "status": "scored" if scoring_verdict else "wrong_answer",
            "message": message[-4000:],
            "duration_seconds": elapsed,
        }


def _evaluate_native(program_path: str, problem_id: str) -> dict:
    problem_dir = PROBLEMS_ROOT / problem_id
    config_path = problem_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Frontier-CS problem is not bundled locally: {problem_id}")

    config = yaml.safe_load(config_path.read_text())
    if config.get("type", "default") != "default":
        raise RuntimeError(f"native evaluator currently supports default problems only: {problem_id}")
    checker_name = config.get("checker")
    if not checker_name:
        raise RuntimeError(f"problem {problem_id} has no checker")

    checker = _checker_binary(problem_id, problem_dir / checker_name)
    time_limit = _duration_seconds(config.get("time", "2s"))
    memory_limit = _memory_bytes(config.get("memory", "256m"))
    test_inputs = sorted((problem_dir / "testdata").glob("*.in"), key=lambda item: int(item.stem))
    if not test_inputs:
        raise RuntimeError(f"problem {problem_id} has no local test data")

    with tempfile.TemporaryDirectory(prefix=f"frontiercs-native-{problem_id}-") as temporary_dir:
        temporary = Path(temporary_dir)
        source = temporary / "main.cpp"
        candidate = temporary / "main"
        code = Path(program_path).read_text()
        code = code.replace("// EVOLVE-BLOCK-START", "").replace("// EVOLVE-BLOCK-END", "").strip()
        source.write_text(code)
        _compile(source, candidate, checker=False)

        workers = max(1, int(os.environ.get("FRONTIERCS_NATIVE_WORKERS", min(8, len(test_inputs)))))
        args = []
        for input_path in test_inputs:
            answer_path = input_path.with_suffix(".ans")
            if not answer_path.is_file():
                answer_path = input_path.with_suffix(".out")
            args.append((candidate, checker, input_path, answer_path, time_limit, memory_limit))
        with ThreadPoolExecutor(max_workers=min(workers, len(args))) as executor:
            cases = list(executor.map(lambda values: _run_case(*values), args))

    score = 100.0 * sum(case["ratio"] for case in cases) / len(cases)
    score_unbounded = 100.0 * sum(case["ratio_unbounded"] for case in cases) / len(cases)
    passed = all(case["ratio"] == 1.0 for case in cases)
    return {
        "combined_score": score,
        "score_unbounded": score_unbounded,
        "runs_successfully": 1.0,
        "status": "success",
        "message": "Native local evaluation completed",
        "problem_id": problem_id,
        "program_path": program_path,
        "duration_seconds": sum(case["duration_seconds"] for case in cases),
        "metadata": {
            "status": "done",
            "passed": passed,
            "result": "Correct Answer" if passed else "Wrong Answer",
            "score": score,
            "scoreUnbounded": score_unbounded,
            "case_statuses": [case["status"] for case in cases],
        },
    }


def evaluate(program_path: str, problem_id: str | None = None, **kwargs) -> dict:
    problem_id = str(problem_id or os.environ.get("FRONTIER_CS_PROBLEM") or kwargs.get("frontier_cs_problem") or "263")
    logger.info("Native Frontier-CS evaluation: program=%s problem=%s", program_path, problem_id)
    try:
        return _evaluate_native(program_path, problem_id)
    except Exception as error:
        logger.error("Native Frontier-CS evaluation failed: %s", error)
        logger.debug(traceback.format_exc())
        return {
            "combined_score": 0.0,
            "score_unbounded": 0.0,
            "runs_successfully": 0.0,
            "status": "error",
            "message": str(error),
            "problem_id": problem_id,
            "program_path": program_path,
            "error": str(error),
        }
