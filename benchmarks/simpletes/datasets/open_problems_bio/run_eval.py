#!/usr/bin/env python3
"""SkyDiscover container glue: evaluate one candidate with the upstream SimpleTES evaluator.

Inside the image ``$SIMPLETES_ROOT`` (default ``/opt/simpletes``) mirrors the SimpleTES
repo root — ``datasets/<family>/…``, ``simpletes/``, ``sitecustomize.py`` — and carries a
copy of ``benchmarks/simpletes/skydiscover_adapter.py``, the re-implementation of
SimpleTES's ``EvaluatorWorker`` that the plain tasks use on the host.  This script is the
container-side counterpart of the plain-task stub: it hands the candidate to the adapter
(fresh subprocess per evaluation, cwd = ``TMPDIR`` = a throw-away dir holding the
candidate as ``program.<ext>``, ``PYTHONPATH`` = ``$SIMPLETES_ROOT`` so ``sitecustomize.py``
and ``simpletes.construction`` behave as upstream, ``SIMPLETES_CAPTURE_CONSTRUCTION_PATH``
set, hard kill at ``$SIMPLETES_EVAL_TIMEOUT`` with SimpleTES's timeout/error conventions)
and prints the result dict in SkyDiscover's container JSON protocol through ``wrapper.py``
(SkyDiscover's own, copied verbatim).

Usage: run_eval.py <evaluator.py> <candidate-file>
Environment: SIMPLETES_ROOT, SIMPLETES_EVAL_TIMEOUT (both exported by evaluate.sh).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIMPLETES_ROOT = os.environ.get("SIMPLETES_ROOT") or str(HERE.parents[1])
for entry in (SIMPLETES_ROOT, str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from skydiscover_adapter import run_evaluator  # noqa: E402  (copy at $SIMPLETES_ROOT)
from wrapper import run  # noqa: E402  (SkyDiscover's container-protocol wrapper)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: run_eval.py <evaluator.py> <candidate-file>")
    evaluator_path = os.path.abspath(sys.argv[1])
    target = Path(sys.argv[2]).resolve()
    timeout = float(os.environ["SIMPLETES_EVAL_TIMEOUT"])
    code = target.read_text(encoding="utf-8")
    result = run_evaluator(
        evaluator_path,
        code,
        timeout=timeout,
        python_executable=sys.executable,
        target_filename=f"program{target.suffix or '.py'}",
    )
    if Path(evaluator_path).name == "final_evaluator.py":
        # Preserve final-evaluation failure as unavailable, even if an older
        # manual held-out score is present in the saved search metrics.
        for name in ("pbmc", "tabula"):
            result.setdefault(f"posthoc_{name}_reported_score", None)
            result.setdefault(f"posthoc_{name}_execution_valid", 0)
            result.setdefault(f"posthoc_{name}_protocol_valid", 0)
            result.setdefault(f"posthoc_{name}_completed", 0)
        result.setdefault("validity", 0)
    sys.argv = [evaluator_path, str(target)]
    run(lambda _program_path: result)


if __name__ == "__main__":
    main()
