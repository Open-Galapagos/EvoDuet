#!/usr/bin/env bash
# SkyDiscover container entrypoint for SimpleTES open_problems_bio/denoising.
# $1 = candidate .py file, $2 = mode ("train"/"test"). Test mode uses the
# PBMC/Tabula posthoc protocol; train mode keeps the original Pancreas evaluator.
# The interpreter is the venv upstream's setup.sh
# builds (pinned scanpy stack + patched OpenProblems v1.0.0); run_eval.py applies
# SimpleTES's EvaluatorWorker semantics through the adapter copy at
# $SIMPLETES_ROOT (fresh subprocess with PYTHONPATH=$SIMPLETES_ROOT, so
# `simpletes.construction` resolves for the evaluator and every Python child — the
# candidate included — runs sitecustomize.py, which installs the
# GLOBAL_BEST_CONSTRUCTION builtin; hard kill at SIMPLETES_EVAL_TIMEOUT, whose
# default is the manifest's eval_timeout for this task: TIMEOUT_SECONDS=400 + margin).
set -euo pipefail
CANDIDATE="${1:?candidate path is required}"
export SIMPLETES_ROOT="${SIMPLETES_ROOT:-/opt/simpletes}"
FAMILY_DIR="${FAMILY_DIR:-$SIMPLETES_ROOT/datasets/open_problems_bio}"
if [[ "${2:-train}" == "test" ]]; then
  export SIMPLETES_EVAL_TIMEOUT="${SIMPLETES_FINAL_EVAL_TIMEOUT:-17940}"
  exec "$FAMILY_DIR/.venv/bin/python" "$FAMILY_DIR/run_eval.py" "$FAMILY_DIR/final_evaluator.py" "$CANDIDATE"
fi
export SIMPLETES_EVAL_TIMEOUT="${SIMPLETES_EVAL_TIMEOUT:-480}"
exec "$FAMILY_DIR/.venv/bin/python" "$FAMILY_DIR/run_eval.py" "$FAMILY_DIR/denoising/evaluator.py" "$CANDIDATE"
