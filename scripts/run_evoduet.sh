#!/usr/bin/env bash
set -euo pipefail

# OpenEvolve with EvoDuet. Edit TASKS to choose the SimpleTES tasks to run.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$REPO_ROOT/benchmarks/simpletes/datasets"
TASKS=(
  "ahc/simpletes_ahc039"
  "ahc/simpletes_ahc058"
  "astrodynamics/cassini"
  "astrodynamics/galileo"
  "astrodynamics/mariner_10"
  "astrodynamics/rosetta"
  "astrodynamics/voyager_2"
  "autocorrelation/simpletes_autocorrelation_first"
  "autocorrelation/simpletes_autocorrelation_second"
  "autocorrelation/simpletes_autocorrelation_third"
  "circle_packing/simpletes_circle_packing_26"
  "circle_packing/simpletes_circle_packing_32"
  "erdos/simpletes_erdos_min_overlap"
  "hadamard_maximal_det/hadamard_maximal_det_29"
  "open_problems_bio/denoising"
  "qubit_routing/swap_reduction"
  "scaling_law/domain_mixture_scaling_law"
  "scaling_law/easy_question_scaling_law"
  "scaling_law/lr_bsz_scaling_law"
  "scaling_law/parallel_scaling_law"
  "sums_diffs/simpletes_sums_diffs"
)

MODEL="${MODEL:-}" # Empty uses the model in each task's config.yaml.
API_BASE="${API_BASE:-}"
ITERATIONS="${ITERATIONS:-100}"
SEED="${SEED:-42}"
INNER_ROUNDS="${INNER_ROUNDS:-3}"
QUERY_COUNT="${QUERY_COUNT:-1}"
DOCUMENTS_PER_QUERY="${DOCUMENTS_PER_QUERY:-5}"
SEARCH_TOP_K="${SEARCH_TOP_K:-3}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/simpletes/openevolve/evoduet}"

# Optional task paths before flags replace the list above.
if (($# > 0)) && [[ "$1" != -* ]]; then
  TASKS=()
  while (($# > 0)) && [[ "$1" != -* ]]; do
    TASKS+=("$1")
    shift
  done
fi

RUN_ARGS=(
  --iterations "$ITERATIONS" --seed "$SEED" --num-generations 1
  --inner-rounds "$INNER_ROUNDS" --query-count "$QUERY_COUNT"
  --documents-per-query "$DOCUMENTS_PER_QUERY" --search-top-k "$SEARCH_TOP_K"
)
if [[ -n "$MODEL" ]]; then RUN_ARGS+=(--model "$MODEL"); fi
if [[ -n "$API_BASE" ]]; then RUN_ARGS+=(--api-base "$API_BASE"); fi

cd -- "$REPO_ROOT"
for TASK in "${TASKS[@]}"; do
  TASK_DIR="$TASK"
  if [[ ! -d "$TASK_DIR" ]]; then TASK_DIR="$DATASET_ROOT/$TASK"; fi
  TASK_DIR="$(cd -- "$TASK_DIR" && pwd)"
  TASK_NAME="${TASK_DIR#"$DATASET_ROOT/"}"
  if [[ "$TASK_NAME" == "$TASK_DIR" ]]; then TASK_NAME="${TASK_DIR##*/}"; fi
  EVALUATOR_ARGS=()
  case "$TASK_DIR" in
    "$DATASET_ROOT"/open_problems_bio/*|"$DATASET_ROOT"/qubit_routing/*)
      EVALUATOR_ARGS=(--evaluator "$(dirname -- "$TASK_DIR")")
      ;;
  esac

  printf '\n[OpenEvolve + EvoDuet] %s\n' "$TASK_NAME" >&2
  uv run --project "$REPO_ROOT" --frozen --no-sync python -m skydiscover.evoduet_cli \
    "$TASK_DIR" --search openevolve_native \
    --output "$OUTPUT_ROOT/$RUN_NAME/$TASK_NAME" \
    "${RUN_ARGS[@]}" "${EVALUATOR_ARGS[@]}" "$@"
done
