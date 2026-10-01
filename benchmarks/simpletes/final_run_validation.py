"""Validate the fixed search winner in a trace-only exported run."""

import hashlib
import json
from pathlib import Path


def validate_trace_winner(output, info, code, expected_iteration):
    if expected_iteration is None:
        raise ValueError("Trace-only final evaluation requires an explicit expected iteration")
    path = Path(output) / "evolution_trace.json"
    raw = path.read_bytes()
    trace = json.loads(raw)
    programs = trace.get("programs", [])
    if isinstance(programs, dict):
        programs = programs.values()
    winners = [p for p in programs if p.get("id") == info.get("id")]
    if (trace.get("last_iteration") != expected_iteration
            or trace.get("best_program_id") != info.get("id") or len(winners) != 1
            or not isinstance(winners[0].get("solution"), str)
            or winners[0]["solution"].encode() != code):
        raise RuntimeError(f"Best program does not match the completed exported trace: {output}")
    return {"kind": "exported_trace", "iteration": expected_iteration, "best_id": info["id"],
            "trace_sha256": hashlib.sha256(raw).hexdigest(), "candidate_sha256": hashlib.sha256(code).hexdigest()}
