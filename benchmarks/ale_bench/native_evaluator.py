"""Native ALE-Bench Lite evaluator.

The official Rust generator/tester and the candidate C++ program are built and
run directly on the host.  Built tools and generated cases are cached between
SkyDiscover iterations.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

SCORE_RE = re.compile(r"Score = (-?[0-9]+)")
ALLOW_NON_AC_PUBLIC = {"ahc016", "ahc025", "ahc027"}


def _native_cache() -> Path:
    configured = os.environ.get("ALEBENCH_NATIVE_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "skydiscover" / "alebench-native"


def _dataset_cache() -> Path:
    configured = os.environ.get("ALE_BENCH_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "ale-bench"


def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.open("a+")
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    return lock_file


def _problem_root(problem_id: str) -> Path:
    cache = _native_cache()
    root = cache / "problems" / problem_id
    marker = root / ".extracted"
    lock_file = _locked(cache / "locks" / f"{problem_id}.extract.lock")
    try:
        if marker.is_file():
            return root
        archive = Path(
            hf_hub_download(
                repo_id="SakanaAI/ALE-Bench",
                filename=f"{problem_id}.zip",
                repo_type="dataset",
                cache_dir=_dataset_cache(),
            )
        )
        staging_parent = cache / "staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{problem_id}-", dir=staging_parent) as temporary_dir:
            temporary = Path(temporary_dir)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(temporary)
            data_path = next(temporary.rglob("data.json"), None)
            if data_path is None:
                raise RuntimeError(f"ALE-Bench archive has no data.json: {archive}")
            extracted_root = data_path.parent
            root.parent.mkdir(parents=True, exist_ok=True)
            stale = root.with_name(f".{root.name}.stale.{os.getpid()}")
            if root.exists():
                os.replace(root, stale)
            os.replace(extracted_root, root)
            shutil.rmtree(stale, ignore_errors=True)
        marker.touch()
        return root
    finally:
        lock_file.close()


def _cargo() -> str:
    requested = os.environ.get("CARGO_BIN", "cargo")
    resolved = shutil.which(requested)
    if resolved is not None:
        return resolved
    if requested != "cargo":
        raise RuntimeError(
            f"configured CARGO_BIN was not found: {requested}"
        )
    if os.environ.get("ALEBENCH_AUTO_INSTALL_RUST", "1") != "1":
        raise RuntimeError(
            "native ALE-Bench evaluation requires cargo/rustc; install Rust or enable ALEBENCH_AUTO_INSTALL_RUST"
        )

    machine = platform.machine().lower()
    architecture = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}.get(machine)
    if architecture is None or platform.system() != "Linux":
        raise RuntimeError(f"automatic Rust setup is unsupported on {platform.system()} {machine}")

    toolchain_root = _native_cache() / "toolchain"
    cargo_home = toolchain_root / "cargo"
    rustup_home = toolchain_root / "rustup"
    cargo = cargo_home / "bin" / "cargo"
    lock_file = _locked(_native_cache() / "locks" / "rust-toolchain.lock")
    try:
        if not cargo.is_file():
            toolchain_root.mkdir(parents=True, exist_ok=True)
            target = f"{architecture}-unknown-linux-gnu"
            base_url = f"https://static.rust-lang.org/rustup/dist/{target}/rustup-init"
            installer = toolchain_root / "rustup-init"
            with urlopen(base_url, timeout=120) as response:
                installer.write_bytes(response.read())
            with urlopen(f"{base_url}.sha256", timeout=30) as response:
                expected_hash = response.read().decode().split()[0]
            actual_hash = hashlib.sha256(installer.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                installer.unlink(missing_ok=True)
                raise RuntimeError("downloaded rustup-init checksum mismatch")
            installer.chmod(0o755)
            environment = os.environ.copy()
            environment.update({"CARGO_HOME": str(cargo_home), "RUSTUP_HOME": str(rustup_home)})
            result = subprocess.run(
                [
                    str(installer),
                    "-y",
                    "--profile",
                    "minimal",
                    "--default-toolchain",
                    "stable",
                    "--no-modify-path",
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("ALEBENCH_RUSTUP_TIMEOUT", "900")),
            )
            installer.unlink(missing_ok=True)
            if result.returncode != 0:
                raise RuntimeError(f"automatic Rust toolchain setup failed:\n{result.stderr[-12000:]}")
        if not cargo.is_file():
            raise RuntimeError("automatic Rust toolchain setup did not produce cargo")
        os.environ["CARGO_HOME"] = str(cargo_home)
        os.environ["RUSTUP_HOME"] = str(rustup_home)
        return str(cargo)
    finally:
        lock_file.close()


def _compiler() -> str:
    requested = os.environ.get("CXX", "g++")
    resolved = shutil.which(requested)
    if resolved is None:
        raise RuntimeError(f"native ALE-Bench evaluation requires a C++ compiler: {requested}")
    return resolved


def _build_tools(problem_id: str, root: Path) -> tuple[Path, Path]:
    tools = root / "tools"
    generator = tools / "target" / "release" / "gen"
    tester = tools / "target" / "release" / "tester"
    lock_file = _locked(_native_cache() / "locks" / f"{problem_id}.build.lock")
    try:
        if not generator.is_file() or not tester.is_file():
            cargo = _cargo()
            environment = os.environ.copy()
            environment.setdefault("RUSTFLAGS", "-Awarnings")
            result = subprocess.run(
                [cargo, "build", "--release", "--locked"],
                cwd=tools,
                env=environment,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("ALEBENCH_CARGO_TIMEOUT", "900")),
            )
            if result.returncode != 0:
                raise RuntimeError(f"failed to build ALE-Bench Rust tools:\n{result.stderr[-12000:]}")
        if not generator.is_file() or not tester.is_file():
            raise RuntimeError(f"ALE-Bench native tools were not produced for {problem_id}")
        return generator, tester
    finally:
        lock_file.close()


def _metadata(root: Path) -> dict:
    return json.loads((root / "data.json").read_text())


def _generated_inputs(problem_id: str, generator: Path, seeds: list[int]) -> list[Path]:
    seed_text = "\n".join(str(seed) for seed in seeds) + "\n"
    digest = hashlib.sha256(seed_text.encode()).hexdigest()[:20]
    cases_root = _native_cache() / "cases" / problem_id / digest
    marker = cases_root / ".complete"
    lock_file = _locked(_native_cache() / "locks" / f"{problem_id}.{digest}.cases.lock")
    try:
        if not marker.is_file():
            cases_root.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"{problem_id}-cases-", dir=cases_root.parent) as temporary_dir:
                temporary = Path(temporary_dir)
                (temporary / "in").mkdir()
                seeds_file = temporary / "seeds.txt"
                seeds_file.write_text(seed_text)
                result = subprocess.run(
                    [str(generator), str(seeds_file)],
                    cwd=temporary,
                    capture_output=True,
                    text=True,
                    timeout=int(os.environ.get("ALEBENCH_GENERATOR_TIMEOUT", "300")),
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ALE-Bench input generation failed:\n{result.stderr[-8000:]}")
                generated = sorted((temporary / "in").glob("*.txt"))
                if len(generated) != len(seeds):
                    raise RuntimeError(
                        f"ALE-Bench generated {len(generated)} inputs for {len(seeds)} seeds"
                    )
                staging = temporary / "ready"
                os.replace(temporary / "in", staging)
                stale = cases_root.with_name(f".{cases_root.name}.stale.{os.getpid()}")
                if cases_root.exists():
                    os.replace(cases_root, stale)
                os.replace(staging, cases_root)
                shutil.rmtree(stale, ignore_errors=True)
            marker.touch()
        inputs = sorted(cases_root.glob("*.txt"))
        if len(inputs) != len(seeds):
            raise RuntimeError(f"cached ALE-Bench input count mismatch for {problem_id}")
        return inputs
    finally:
        lock_file.close()


def _compile_candidate(program_path: str, output: Path) -> None:
    source = output.with_suffix(".cpp")
    code = Path(program_path).read_text()
    for marker in (
        "// EVOLVE-BLOCK-START",
        "// EVOLVE-BLOCK-END",
        "# EVOLVE-BLOCK-START",
        "# EVOLVE-BLOCK-END",
    ):
        code = code.replace(marker, "")
    code = code.strip()
    source.write_text(code)
    command = [
        _compiler(),
        "-std=gnu++20",
        "-O2",
        "-pipe",
        "-pthread",
        "-march=native",
        "-DONLINE_JUDGE",
        str(source),
        "-o",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"candidate compilation failed:\n{result.stderr[-12000:]}")


def _score_from_tester(stderr: str) -> int | None:
    matches = SCORE_RE.findall(stderr)
    return int(matches[-1]) if matches else None


def _run_case(
    candidate: Path,
    tester: Path,
    input_path: Path,
    problem_type: str,
    time_limit: float,
    memory_limit: int,
) -> dict:
    started = time.perf_counter()
    cpu_limit = max(1, math.ceil(time_limit + 0.1))
    wall_limit = max(1.0, time_limit + 0.5)
    with tempfile.TemporaryDirectory(prefix="alebench-native-case-") as temporary_dir:
        temporary = Path(temporary_dir)
        output_path = temporary / "output.txt"
        candidate_stderr_path = temporary / "candidate.err"
        if problem_type == "batch":
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
            with input_path.open("rb") as stdin_file, output_path.open("wb") as stdout_file, candidate_stderr_path.open("wb") as stderr_file:
                run = subprocess.run(command, stdin=stdin_file, stdout=stdout_file, stderr=stderr_file)
            elapsed = time.perf_counter() - started
            candidate_stderr = candidate_stderr_path.read_text(errors="replace")[-4000:]
            if run.returncode != 0:
                return {
                    "judge_result": "TIME_LIMIT_EXCEEDED" if run.returncode in {124, 137} else "RUNTIME_ERROR",
                    "score": 0,
                    "execution_time": elapsed,
                    "stderr": candidate_stderr,
                    "message": "candidate execution failed",
                }
            judged = subprocess.run(
                [str(tester), str(input_path), str(output_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        elif problem_type == "reactive":
            command = [
                "timeout",
                "--kill-after=1",
                str(wall_limit),
                "prlimit",
                f"--as={memory_limit}",
                f"--cpu={cpu_limit}",
                "--",
                str(tester),
                str(candidate),
            ]
            with input_path.open("rb") as stdin_file, output_path.open("wb") as stdout_file:
                judged = subprocess.run(command, stdin=stdin_file, stdout=stdout_file, capture_output=False, stderr=subprocess.PIPE)
            judged.stdout = ""
            judged.stderr = judged.stderr.decode(errors="replace") if isinstance(judged.stderr, bytes) else judged.stderr
        else:
            raise RuntimeError(f"unsupported ALE-Bench problem type: {problem_type}")

        elapsed = time.perf_counter() - started
        judge_stderr = (judged.stderr or "")[-8000:]
        score = _score_from_tester(judge_stderr)
        if judged.returncode != 0 or score is None:
            return {
                "judge_result": "TIME_LIMIT_EXCEEDED" if judged.returncode in {124, 137} else "WRONG_ANSWER",
                "score": 0,
                "execution_time": elapsed,
                "stderr": judge_stderr,
                "message": "official tester rejected the output",
            }
        return {
            "judge_result": "ACCEPTED",
            "score": score,
            "execution_time": elapsed,
            "stderr": judge_stderr,
            "message": "",
        }


def _overall_judge(cases: list[dict]) -> str:
    priority = [
        "INTERNAL_ERROR",
        "WRONG_ANSWER",
        "RUNTIME_ERROR",
        "TIME_LIMIT_EXCEEDED",
        "MEMORY_LIMIT_EXCEEDED",
        "COMPILATION_ERROR",
    ]
    present = {case["judge_result"] for case in cases}
    return next((status for status in priority if status in present), "ACCEPTED")


def _evaluate_seeds(program_path: str, problem_id: str, seeds: list[int], *, allow_non_ac: bool) -> dict:
    root = _problem_root(problem_id)
    data = _metadata(root)
    metadata = data["metadata"]
    constraints = data["constraints"]
    generator, tester = _build_tools(problem_id, root)
    inputs = _generated_inputs(problem_id, generator, seeds)

    with tempfile.TemporaryDirectory(prefix=f"alebench-native-{problem_id}-") as temporary_dir:
        candidate = Path(temporary_dir) / "candidate"
        _compile_candidate(program_path, candidate)
        workers = max(1, int(os.environ.get("ALEBENCH_NATIVE_WORKERS", "13")))
        arguments = [
            (
                candidate,
                tester,
                input_path,
                metadata["problem_type"],
                float(constraints["time_limit"]),
                int(constraints["memory_limit"]),
            )
            for input_path in inputs
        ]
        with ThreadPoolExecutor(max_workers=min(workers, len(arguments))) as executor:
            cases = list(executor.map(lambda values: _run_case(*values), arguments))

    overall_judge = _overall_judge(cases)
    accepted_score = sum(case["score"] for case in cases if case["judge_result"] == "ACCEPTED")
    overall_score = accepted_score if allow_non_ac or overall_judge == "ACCEPTED" else 0
    optim_factor = 1 if metadata["score_type"] == "maximize" else -1
    combined_score = overall_score * optim_factor / len(cases)
    if overall_judge != "ACCEPTED" and optim_factor < 0:
        combined_score = -sys.float_info.max
    failed = next((case for case in cases if case["judge_result"] != "ACCEPTED"), cases[0])
    return {
        "judge_result": overall_judge,
        "overall_score": overall_score,
        "max_execution_time_sec": max(case["execution_time"] for case in cases),
        "max_memory_usage_mib": 0,
        "standard_error": failed["stderr"],
        "message": failed["message"],
        "combined_score": combined_score,
        "runs_successfully": 1.0,
        "num_cases": len(cases),
        "num_passed_cases": sum(case["judge_result"] == "ACCEPTED" for case in cases),
        "num_failed_cases": sum(case["judge_result"] != "ACCEPTED" for case in cases),
    }


def evaluate(program_path: str, **kwargs) -> dict:
    problem_id = str(kwargs.get("problem_id") or os.environ.get("ALEBENCH_PROBLEM_ID") or "ahc011")
    num_public_cases = max(1, int(os.environ.get("ALEBENCH_PUBLIC_CASES", "50")))
    logger.info("Native ALE-Bench evaluation: program=%s problem=%s", program_path, problem_id)
    try:
        return _evaluate_seeds(
            program_path,
            problem_id,
            list(range(num_public_cases)),
            allow_non_ac=problem_id in ALLOW_NON_AC_PUBLIC,
        )
    except Exception as error:
        logger.error("Native ALE-Bench evaluation failed: %s", error)
        logger.debug(traceback.format_exc())
        return {
            "overall_score": 0.0,
            "combined_score": 0.0,
            "runs_successfully": 0.0,
            "error": str(error),
        }


def evaluate_final(program_path: str, **kwargs) -> dict:
    problem_id = str(kwargs.get("problem_id") or os.environ.get("ALEBENCH_PROBLEM_ID") or "ahc011")
    try:
        root = _problem_root(problem_id)
        data = _metadata(root)
        seed_key = os.environ.get("ALEBENCH_FINAL_SEED_SET", "private_lite")
        if seed_key not in {"private", "private_lite"}:
            raise ValueError("ALEBENCH_FINAL_SEED_SET must be private or private_lite")
        result = _evaluate_seeds(
            program_path,
            problem_id,
            list(data["seeds"][seed_key]),
            allow_non_ac=True,
        )
        return {
            "combined_score": result["combined_score"],
            "private_score": result["overall_score"],
            "private_score_mean": abs(result["combined_score"]),
            "num_private_cases": float(result["num_cases"]),
            "num_private_passed_cases": float(result["num_passed_cases"]),
            "num_private_failed_cases": float(result["num_failed_cases"]),
            "validity": 1.0 if result["judge_result"] == "ACCEPTED" else 0.0,
            "private_eval_runs": 1.0,
        }
    except Exception as error:
        logger.error("Native ALE-Bench final evaluation failed: %s", error)
        return {
            "combined_score": 0.0,
            "private_score": 0.0,
            "validity": 0.0,
            "error": str(error),
        }
