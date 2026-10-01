#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
MODE="${2:-train}"

# Evolution scores the fixed seed-42 draw of placement instances.  The final
# (test-mode) pass scores a disjoint draw from a different seed.
if [ "$MODE" = "test" ]; then
  export PRISM_EVAL_USE_FINAL_SET=1
fi

python /benchmark/evaluator.py "$PROGRAM"
