#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
MODE="${2:-train}"

# Evolution scores the five training realizations (seed offset 42).  The final
# (test-mode) pass scores a disjoint draw: 20 signals from seed offset 1042.
if [ "$MODE" = "test" ]; then
  export SIGPROC_EVAL_SEED_OFFSET=1042
  export SIGPROC_EVAL_NUM_SIGNALS=20
fi

python /benchmark/evaluator.py "$PROGRAM"
