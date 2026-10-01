#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
MODE="${2:-train}"

# Evolution scores the fixed validation draw.  The final (test-mode) pass scores
# a fresh, larger draw from a different seed.
if [ "$MODE" = "test" ]; then
  export QNN_EVAL_FINAL_TEST=1
fi

python /benchmark/evaluator.py "$PROGRAM"
