#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
MODE="${2:-train}"

# Evolution scores the train demonstration pairs only.  The final (test-mode)
# pass scores the task's held-out test grids against the solutions file.
if [ "$MODE" = "test" ]; then
  export ARC_EVAL_INCLUDE_TEST=1
  export ARC_EVAL_SCORE_ON_TEST_ONLY=1
fi

python /benchmark/evaluator.py "$PROGRAM"
