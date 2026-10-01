#!/usr/bin/env bash
# SkyDiscover container entrypoint for SimpleTES qubit_routing/swap_reduction.
# $1 = candidate .rs file, $2 = mode ("train"/"test"); upstream has no held-out
# split, so mode is ignored and every evaluation is the full benchmark suite.
# Installed at /benchmark/evaluate.sh (the path SkyDiscover calls) via a symlink
# to the family dir, which the Dockerfile places at $FAMILY_DIR
# (= <repo-root mirror>/datasets/qubit_routing, so evaluator.py's
# REPO_ROOT = parents[2] resolves).
#
# run_eval.py applies SimpleTES's EvaluatorWorker semantics through the adapter
# copy at $SIMPLETES_ROOT (fresh subprocess, PYTHONPATH=$SIMPLETES_ROOT, hard kill).
# SIMPLETES_EVAL_TIMEOUT defaults to the manifest's eval_timeout for this task
# (SimpleTES's default --eval-timeout), and QUBIT_ROUTING_PARENT_EVAL_TIMEOUT_SECONDS
# is what SimpleTES's engine exports for this family (slot_workspace.py): the
# evaluator derives its build+routing budget from it (parent - 60 s) instead of
# falling back to DEFAULT_TOTAL_BUDGET_SECONDS=1200.  QUBIT_ROUTING_SLOT_COUNT is
# left at the evaluator default (4), which matches SkyDiscover's default evaluation
# concurrency; raise it together with max_parallel_iterations.
set -euo pipefail
CANDIDATE="${1:?candidate path is required}"
export SIMPLETES_ROOT="${SIMPLETES_ROOT:-/opt/simpletes}"
FAMILY_DIR="${FAMILY_DIR:-$SIMPLETES_ROOT/datasets/qubit_routing}"
export SIMPLETES_EVAL_TIMEOUT="${SIMPLETES_EVAL_TIMEOUT:-3000}"
export QUBIT_ROUTING_PARENT_EVAL_TIMEOUT_SECONDS="${QUBIT_ROUTING_PARENT_EVAL_TIMEOUT_SECONDS:-$SIMPLETES_EVAL_TIMEOUT}"
exec python3 "$FAMILY_DIR/run_eval.py" "$FAMILY_DIR/swap_reduction/evaluator.py" "$CANDIDATE"
