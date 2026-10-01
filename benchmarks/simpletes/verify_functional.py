#!/usr/bin/env python3
"""Cross-check a ported SimpleTES task against the upstream checkout, functionally.

Two legs, same program:

* **upstream** — the evaluator from the SimpleTES checkout, run exactly the way
  SimpleTES's ``EvaluatorWorker`` runs it (fresh subprocess, cwd = TMPDIR = a
  throw-away dir, ``PYTHONPATH`` = SimpleTES repo root so ``sitecustomize.py``
  runs, ``SIMPLETES_CAPTURE_CONSTRUCTION_PATH`` inside that dir, evaluator loaded
  from its path as ``user_evaluator``, JSON on stdout).
* **port** — the copy under ``benchmarks/simpletes/datasets/`` run through
  SkyDiscover's own evaluator (``create_evaluator``: the plain ``Evaluator`` on the
  task's ``skydiscover_evaluator.py`` stub, which hands the untouched
  ``evaluator.py`` to ``skydiscover_adapter.py``; the Docker
  ``ContainerizedEvaluator`` for the two container families), i.e. what
  ``skydiscover-run`` does.

Every numeric metric of the two legs is compared key by key, including numbers
nested in lists/dicts (``nmse_per_dim``, astrodynamics ``per_instance``).  A leg
that reports an ``error`` (top level or per instance) never counts as a match,
even when both legs fail identically.  Deterministic tasks must agree exactly.
Timing-dependent metrics can use a tolerance band (``--rtol``); validity and
correctness metrics can still be compared exactly (``--exact validity,ok_cases,...``).

``--port-sitecustomize`` additionally exports ``PYTHONPATH=benchmarks/simpletes`` for
the port leg; it is redundant with the adapter (which sets that ``PYTHONPATH`` for the
evaluator subprocess itself, so the ``GLOBAL_BEST_CONSTRUCTION`` builtin is always
defined) and only kept for experiments.  ``--shared-construction FILE`` hands both
legs a released ``*_best_construction.json`` (SimpleTES's warm start).

Usage:
    uv run python benchmarks/simpletes/verify_functional.py --task circle_packing/simpletes_circle_packing_26
    uv run python benchmarks/simpletes/verify_functional.py --all --json out.json
    uv run python benchmarks/simpletes/verify_functional.py --task astrodynamics/cassini \\
        --program ../references/SimpleTES/best_results/astrodynamics/cassini/cassini_best.py

``--upstream-python`` picks the interpreter for the upstream leg (default: this
one); point it at a family venv (``uv sync --project <family>``) to reproduce
upstream's pinned environment.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SUITE_DIR / "datasets"
REPO_ROOT = SUITE_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from skydiscover.config import EvaluatorConfig  # noqa: E402
from skydiscover.evaluation import create_evaluator  # noqa: E402
from skydiscover.evaluation.evaluation_result import evaluation_failure_reason  # noqa: E402

MANIFEST = json.loads((SUITE_DIR / "UPSTREAM_MANIFEST.json").read_text())

# simpletes/construction.py env names (SimpleTES @ manifest commit).
CAPTURE_CONSTRUCTION_ENV = "SIMPLETES_CAPTURE_CONSTRUCTION_PATH"
SHARED_CONSTRUCTION_ENV = "SIMPLETES_SHARED_CONSTRUCTION_PATH"
MAX_SNAPSHOT_BYTES_ENV = "SIMPLETES_SHARED_CONSTRUCTION_MAX_BYTES"
DEFAULT_MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024

# Verbatim copy of simpletes/evaluator.py::_EVAL_RUNNER (SimpleTES @ manifest commit), whitespace included.
EVAL_RUNNER = textwrap.dedent(r'''
import importlib.util
import json
import sys
import traceback

def main():
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Invalid arguments", "combined_score": float("-inf")}))
        sys.exit(1)
    
    evaluator_path = sys.argv[1]
    target_file = sys.argv[2]
    
    try:
        # Load evaluator module
        spec = importlib.util.spec_from_file_location("user_evaluator", evaluator_path)
        if spec is None or spec.loader is None:
            print(json.dumps({"error": f"Could not load evaluator: {evaluator_path}", "combined_score": float("-inf")}))
            sys.exit(1)
        
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        
        # Run evaluation
        result = mod.evaluate(target_file)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_msg, "combined_score": float("-inf")}))
        sys.exit(1)

if __name__ == "__main__":
    main()
''').strip()


def task_info(key: str) -> dict:
    if key not in MANIFEST["tasks"]:
        sys.exit(f"unknown task {key!r}; known: {', '.join(sorted(MANIFEST['tasks']))}")
    info = dict(MANIFEST["tasks"][key])
    family, dest_task = key.split("/")
    info["family"] = family
    info["dest_task"] = dest_task
    info["task"] = info["upstream_task"].rsplit("/", 1)[1]
    info["suffix"] = ".rs" if info["language"] == "rust" else ".py"
    info["task_dir"] = DATASETS_DIR / family / dest_task
    return info


def parse_json_from_output(stdout_text: str) -> dict:
    """SimpleTES's _parse_json_from_output: last line that is a JSON object, error dict otherwise."""
    if not stdout_text or not stdout_text.strip():
        return {"error": "Empty evaluator output", "combined_score": float("-inf")}
    for line in reversed(stdout_text.split("\n")):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(stdout_text)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON output: {e}", "combined_score": float("-inf")}


@contextmanager
def scoped_environ(updates: dict[str, str | None]):
    """Temporarily set (value) / unset (None) environment variables."""
    old = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_upstream(info: dict, upstream: Path, program: Path, python: str, timeout: float,
                 shared_construction: str | None, *, preserve_source_layout: bool = False) -> dict:
    evaluator = upstream / info["upstream_task"] / "evaluator.py"
    eval_dir = Path(tempfile.mkdtemp(prefix="eval_upstream_"))
    try:
        target = eval_dir / f"program{info['suffix']}"
        if preserve_source_layout:
            # Released programs may locate dataset helpers relative to __file__.
            # Preserve their repository layout without editing the source or
            # writing candidate files into the original checkout.
            relative = program.resolve().relative_to(upstream.resolve())
            if relative.parts[0] != "best_results":
                raise ValueError("Source-layout preservation requires a released best_results program")
            mirror = eval_dir / "source"
            target = mirror / relative
            target.parent.mkdir(parents=True)
            (mirror / "datasets").symlink_to(upstream / "datasets", target_is_directory=True)
        shutil.copyfile(program, target)
        # EvaluatorWorker._subprocess_env + evaluate(): PYTHONPATH=<repo root>,
        # capture path in the eval dir, TMPDIR/TMP/TEMP = eval dir, cwd = eval dir.
        env = os.environ.copy()
        env[CAPTURE_CONSTRUCTION_ENV] = str(eval_dir / "captured_construction.json")
        env[MAX_SNAPSHOT_BYTES_ENV] = env.get(MAX_SNAPSHOT_BYTES_ENV) or str(DEFAULT_MAX_SNAPSHOT_BYTES)
        if shared_construction:
            env[SHARED_CONSTRUCTION_ENV] = shared_construction
        else:
            env.pop(SHARED_CONSTRUCTION_ENV, None)
        env["PYTHONPATH"] = str(upstream) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(eval_dir)
        if info["family"] == "ahc" and "AHC_CACHE_DIR" not in os.environ:
            # Upstream tracks the official testers as non-executable (its README's
            # setup step is `chmod +x cache/tester_binaries/*`); point the upstream
            # evaluator at the port's byte-identical, executable copy through the env
            # override the evaluator provides, instead of chmod-ing the checkout.
            env["AHC_CACHE_DIR"] = str(DATASETS_DIR / "ahc" / "cache")
        t0 = time.time()
        # EvaluatorWorker.evaluate(): own session so the whole process group can be
        # killed on timeout; error dicts follow the worker's conventions.
        proc = subprocess.Popen(
            [python, "-c", EVAL_RUNNER, str(evaluator), str(target)],
            cwd=eval_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            return {"metrics": {"error": "timeout", "combined_score": float("-inf")}, "returncode": None,
                    "stderr_tail": "", "wall_seconds": time.time() - t0}
        elapsed = time.time() - t0
        if proc.returncode != 0:
            metrics = {"error": (stderr.strip() or stdout.strip() or f"Process exited with code {proc.returncode}")[-4000:],
                       "combined_score": float("-inf")}
        else:
            metrics = parse_json_from_output(stdout)
        return {"metrics": metrics, "returncode": proc.returncode, "stderr_tail": stderr[-2000:],
                "wall_seconds": elapsed, "captured_construction": (eval_dir / "captured_construction.json").exists()}
    finally:
        shutil.rmtree(eval_dir, ignore_errors=True)


def run_port(info: dict, program: Path, timeout: int, sitecustomize: bool, shared_construction: str | None) -> dict:
    if info["evaluator_mode"] == "container":
        evaluation_file = DATASETS_DIR / info["family"]
    else:
        # The stub skydiscover-run is pointed at: runs the untouched evaluator.py
        # through skydiscover_adapter.py (SimpleTES's EvaluatorWorker semantics).
        evaluation_file = info["task_dir"] / "skydiscover_evaluator.py"
    config = EvaluatorConfig(
        evaluation_file=str(evaluation_file),
        file_suffix=info["suffix"],
        timeout=timeout,
        final_timeout=timeout,
        max_retries=0,
        cascade_evaluation=False,
    )
    env: dict[str, str | None] = {}
    if sitecustomize:
        env["PYTHONPATH"] = str(SUITE_DIR) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    env[SHARED_CONSTRUCTION_ENV] = shared_construction  # None -> unset, as SkyDiscover leaves it
    t0 = time.time()
    with scoped_environ(env):
        evaluator = create_evaluator(config, max_concurrent=1)
        try:
            result = asyncio.run(evaluator.evaluate_program(program.read_text(encoding="utf-8"), program_id=info["dest_task"]))
        finally:
            close = getattr(evaluator, "close", None)
            if close:
                close()
    return {
        "metrics": dict(result.metrics),
        "artifacts": {k: (v if isinstance(v, str) else str(v))[-2000:] for k, v in result.artifacts.items()},
        "wall_seconds": time.time() - t0,
    }


def _numeric_string(value: str):
    """Parse numeric strings while preserving fractions such as '36/36'."""
    try:
        return float(value)
    except ValueError:
        return None


def flatten_numeric(value, prefix: str = "", out: dict | None = None) -> dict:
    """Numeric leaves of a nested dict/list (numeric strings included), keyed by dotted path."""
    if out is None:
        out = {}
    if isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        out[prefix] = float(value)
    elif isinstance(value, str) and _numeric_string(value) is not None:
        out[prefix] = _numeric_string(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            flatten_numeric(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            flatten_numeric(v, f"{prefix}[{i}]", out)
    return out


def collect_errors(value, prefix: str = "", out: list | None = None) -> list:
    """Non-empty string ``error`` fields anywhere in a nested result."""
    if out is None:
        out = []
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if k == "error" and isinstance(v, str) and v.strip():
                out.append(f"{path}: {v.strip().splitlines()[-1][:200]}")
            else:
                collect_errors(v, path, out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            collect_errors(v, f"{prefix}[{i}]", out)
    return out


def leg_errors(leg: dict) -> list:
    metrics = leg.get("metrics", {})
    artifacts = leg.get("artifacts", {})
    errors = collect_errors(metrics)
    errors += [f"artifacts.{e}" for e in collect_errors(artifacts)]
    score = metrics.get("combined_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isinf(score) and score < 0:
        errors.append("combined_score is -inf (runner-level failure)")
    # SkyDiscover-level failures ({"error": 0.0}, {"timeout": True}, docker exec
    # failures, missing combined_score) are what SkyDiscover itself would treat as
    # a failed evaluation; never let them count as agreement.
    reason = evaluation_failure_reason(metrics, artifacts)
    if reason:
        errors.append(reason)
    elif not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
        errors.append(f"combined_score missing or non-finite: {score!r}")
    return errors


def flatten_discrete(value, prefix: str = "", out: dict | None = None) -> dict:
    """Bool and string leaves (status flags and "0"/"1" strings), keyed by dotted path."""
    if out is None:
        out = {}
    if isinstance(value, bool) or (isinstance(value, str) and _numeric_string(value) is None):
        out[prefix] = value
    elif isinstance(value, dict):
        for k, v in value.items():
            flatten_discrete(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            flatten_discrete(v, f"{prefix}[{i}]", out)
    return out


def compare(up: dict, port: dict, rtol: float, ignore: set[str], exact: set[str], atol: float = 0.0) -> list[str]:
    problems = []
    a, b = flatten_numeric(up["metrics"]), flatten_numeric(port["metrics"])
    if "combined_score" not in a:
        problems.append("upstream leg has no combined_score")
    if "combined_score" not in b:
        problems.append("port leg has no combined_score")
    for side, leg in (("upstream", up), ("port", port)):
        for err in leg_errors(leg):
            problems.append(f"{side} leg errored: {err}")

    def ignored(key: str) -> bool:
        base = key.split(".")[-1].split("[")[0]
        if key in ignore or base in ignore or base == "error" or base.endswith("_traceback"):
            return True
        # Any segment of the path that is itself an ignored key (e.g. the
        # "<circuit>_regression_snapshot" sub-tree, emitted by an unseeded 25 % gate).
        return any(seg.split("[")[0] in ignore or seg.endswith("_regression_snapshot") for seg in key.split("."))

    def must_be_exact(key: str) -> bool:
        base = key.split(".")[-1].split("[")[0]
        return key in exact or base in exact

    # Bool / string leaves must agree exactly regardless of the numeric band.
    da, db = flatten_discrete(up["metrics"]), flatten_discrete(port["metrics"])
    for key in sorted(set(da) | set(db)):
        if ignored(key) or key in a or key in b:  # bools also appear in the numeric view; handled here
            if key in da and key in db and da[key] != db[key] and not ignored(key):
                problems.append(f"{key}: upstream={da[key]!r} port={db[key]!r}")
            continue
        if key not in da:
            problems.append(f"{key}: only in port ({db[key]!r})")
        elif key not in db:
            problems.append(f"{key}: only in upstream ({da[key]!r})")
        elif da[key] != db[key]:
            problems.append(f"{key}: upstream={da[key]!r} port={db[key]!r}")

    for key in sorted(set(a) | set(b)):
        if ignored(key) or key in da or key in db:
            continue
        if key not in a:
            problems.append(f"{key}: only in port ({b[key]})")
            continue
        if key not in b:
            # SkyDiscover keeps non-numeric values as artifacts; a numeric key missing in the port is real.
            problems.append(f"{key}: only in upstream ({a[key]})")
            continue
        x, y = a[key], b[key]
        if math.isnan(x) and math.isnan(y):
            continue
        if x == y:
            continue
        tol = 0.0 if must_be_exact(key) else rtol
        abs_tol = 0.0 if must_be_exact(key) else atol
        if math.isfinite(x) and math.isfinite(y) and abs(x - y) <= max(abs_tol, tol * max(abs(x), abs(y), 1e-12)):
            continue
        problems.append(f"{key}: upstream={x!r} port={y!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", action="append", default=[], help="family/dest_task (repeatable)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--program", help="program to score (default: the task's initial_program)")
    parser.add_argument("--upstream", default=str(REPO_ROOT.parent / "references" / "SimpleTES"))
    parser.add_argument("--upstream-python", default=sys.executable)
    parser.add_argument("--skip-upstream", action="store_true")
    parser.add_argument("--skip-port", action="store_true")
    parser.add_argument("--port-sitecustomize", action="store_true",
                        help="run the port leg with PYTHONPATH=benchmarks/simpletes (upstream's sitecustomize semantics)")
    parser.add_argument("--shared-construction", help="SIMPLETES_SHARED_CONSTRUCTION_PATH for both legs (warm start)")
    parser.add_argument("--rtol", type=float, default=0.0, help="relative tolerance for numeric metrics (0 = exact)")
    parser.add_argument("--atol", type=float, default=0.0,
                        help="absolute tolerance for numeric metrics (e.g. 1e-9 for floating-point noise)")
    parser.add_argument("--exact", default="validity,ok_cases,failed_cases,total_cases,poisson_pass,n_points,"
                        "num_instances,failed_instances,mission_success_rate,set_size,sumset_size,diffset_size",
                        help="comma-separated metric keys that must agree exactly even with --rtol")
    parser.add_argument("--ignore", default="eval_time,elapsed_seconds,wall_seconds,slot_wait_time,build_time,"
                        "router_time,total_runtime_ms,slot_id,eval_time_limit,sol_time_ms,"
                        "strategy_diagnostics,compilation_time,component_total_times,component_time_total_sum,"
                        "results_path",
                        help="comma-separated metric keys excluded from the comparison (wall-clock / timestamped "
                             "diagnostics); keys ending in _traceback or _regression_snapshot are always excluded")
    parser.add_argument("--timeout", type=int, help="override the evaluator timeout (seconds)")
    parser.add_argument("--json", help="write the full report here")
    args = parser.parse_args()

    keys = sorted(MANIFEST["tasks"]) if args.all else args.task
    if not keys:
        parser.error("--task or --all is required")
    if args.program and len(keys) != 1:
        parser.error("--program applies to exactly one --task")
    upstream = Path(args.upstream).resolve()
    ignore = {k for k in args.ignore.split(",") if k}
    exact = {k for k in args.exact.split(",") if k}
    shared = str(Path(args.shared_construction).resolve()) if args.shared_construction else None

    report = {"upstream": str(upstream), "port_sitecustomize": args.port_sitecustomize,
              "shared_construction": shared, "tasks": {}}
    failed = 0
    for key in keys:
        info = task_info(key)
        program = Path(args.program).resolve() if args.program else info["task_dir"] / f"initial_program{info['suffix']}"
        # Upstream leg: SimpleTES's --eval-timeout for the task.  Port leg: what
        # config.yaml gives SkyDiscover (the same budget plus the adapter margin).
        eval_timeout = args.timeout or info["eval_timeout"]
        timeout = args.timeout or info["config_timeout"]
        entry = {"program": str(program), "eval_timeout": eval_timeout, "config_timeout": timeout}
        print(f"== {key}  ({info['evaluator_mode']}, eval timeout {eval_timeout}s)  program={program.name}", flush=True)
        if not args.skip_upstream:
            entry["upstream"] = run_upstream(info, upstream, program, args.upstream_python, eval_timeout + 60, shared)
            print(f"   upstream: combined_score={entry['upstream']['metrics'].get('combined_score')!r}  "
                  f"({entry['upstream']['wall_seconds']:.1f}s, rc={entry['upstream']['returncode']})", flush=True)
            for err in leg_errors(entry["upstream"]):
                print(f"   upstream ERROR: {err}", flush=True)
        if not args.skip_port:
            entry["port"] = run_port(info, program, timeout, args.port_sitecustomize, shared)
            print(f"   port:     combined_score={entry['port']['metrics'].get('combined_score')!r}  "
                  f"({entry['port']['wall_seconds']:.1f}s)", flush=True)
            for err in leg_errors(entry["port"]):
                print(f"   port ERROR: {err}", flush=True)
        if "upstream" in entry and "port" in entry:
            problems = compare(entry["upstream"], entry["port"], args.rtol, ignore, exact, args.atol)
            entry["problems"] = problems
            if problems:
                failed += 1
                print("   MISMATCH:\n     " + "\n     ".join(problems), flush=True)
            else:
                n = len([k for k in flatten_numeric(entry["port"]["metrics"]) if k.split(".")[-1].split("[")[0] not in ignore])
                print(f"   MATCH ({n} numeric metrics, rtol={args.rtol})", flush=True)
        elif "port" in entry and leg_errors(entry["port"]):
            failed += 1
            entry["problems"] = [f"port leg errored: {e}" for e in leg_errors(entry["port"])]
        report["tasks"][key] = entry

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\n{len(keys) - failed}/{len(keys)} tasks agree")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
