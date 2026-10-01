#!/usr/bin/env bash
set -euo pipefail

PROGRAM="$1"
MODE="${2:-train}"

python /benchmark/evaluator.py "$PROGRAM" "$MODE"
