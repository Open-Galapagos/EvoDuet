"""Run a candidate's solve(payload) in a disposable, resource-limited process.

Only the candidate source and public JSON input are copied to the working
directory. Objective evaluation stays in the parent. This gives reproducible
process isolation, not an OS security sandbox against hostile Python code.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


_MAX_BYTES = 8 * 1024 * 1024
_CHILD = r'''
import contextlib
import importlib.util
import json
import random
import resource
import sys

cpu_seconds = int(sys.argv[1])
resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
random.seed(0)
payload = json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
    candidate = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = candidate
    spec.loader.exec_module(candidate)
    output = candidate.solve(payload)
json.dump(output, sys.stdout, allow_nan=False)
'''


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def run_candidate(program_path, payload, *, timeout=30):
    """Return the candidate's JSON artifact or raise a bounded execution error.

Candidate code must implement solve(payload) and return plain JSON values.
Imports/prints are isolated from the result. API credentials and repository
PYTHONPATH are not inherited. Linux resource limits bound CPU, memory and logs.
"""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    path = Path(program_path).resolve()
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("Candidate source exceeds 8 MiB")
    source = path.read_text(encoding="utf-8")
    encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        raise ValueError("Public input exceeds 8 MiB")
    with tempfile.TemporaryDirectory(prefix="skydiscover-public-opt-") as temporary:
        folder = Path(temporary)
        (folder / "candidate.py").write_text(source, encoding="utf-8")
        environment = {
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "TZ": "UTC",
            "TMPDIR": temporary,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        # Files bound captured logs without growing the parent's memory.
        with (folder / "input.json").open("wb+") as stdin, \
                (folder / "stdout.txt").open("wb+") as stdout, \
                (folder / "stderr.txt").open("wb+") as stderr:
            stdin.write(encoded)
            stdin.seek(0)
            process = subprocess.Popen(
                [sys.executable, "-I", "-c", _CHILD, str(max(1, math.ceil(timeout)))],
                cwd=folder, env=environment, stdin=stdin, stdout=stdout,
                stderr=stderr, start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Candidate exceeded {timeout:g} seconds") from exc
            finally:
                # Also reap descendants if solve returned after spawning workers.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if process.returncode:
                stderr.seek(0, os.SEEK_END)
                stderr.seek(max(0, stderr.tell() - 1500))
                detail = stderr.read().decode("utf-8", errors="replace").strip()
                raise ValueError(f"Candidate exited {process.returncode}: {detail}")
            stdout.seek(0)
            result = stdout.read(_MAX_BYTES + 1)
            if len(result) > _MAX_BYTES:
                raise ValueError("Candidate output exceeds 8 MiB")
            try:
                return json.loads(result, parse_constant=_reject_constant)
            except (ValueError, UnicodeError) as exc:
                raise ValueError("Candidate must return a finite JSON artifact") from exc
