# Running SimpleTES with OpenEvolve

After `uv sync --frozen --extra dev`, prepare the tasks following the
[SimpleTES setup guide](../benchmarks/simpletes/README.md#setup). These four scripts
use the `openevolve_native` search scaffold:

| Script | Retrieval | Solution candidates per iteration |
| --- | --- | --- |
| `run_baseline.sh` | Disabled | 1 |
| `run_evoduet.sh` | EvoDuet | 1 |
| `run_evoduet_parallel.sh` | EvoDuet | N after retrieve/look-up; 1 after no-op |
| `run_baseline_parallel.sh` | Disabled | N every iteration |

Each script contains its own `TASKS=(...)` array and `for` loop. All 21 SimpleTES
tasks are listed by default; remove or comment out entries to choose a subset:

```bash
TASKS=(
  "astrodynamics/cassini"
  "circle_packing/simpletes_circle_packing_26"
  "scaling_law/easy_question_scaling_law"
)
```

Tasks run sequentially. Parallel scripts generate multiple solution candidates
within each iteration; `NUM_GENERATIONS` defaults to 8. The parallel baseline
evaluates one candidate at a time, matching the original baseline experiments.
EvoDuet allows N evaluations concurrently. Denoising and Swap Reduction
automatically use their family directories as Docker evaluators.

```bash
MODEL=YOUR_MODEL bash scripts/run_baseline.sh
MODEL=YOUR_MODEL bash scripts/run_evoduet.sh
MODEL=YOUR_MODEL NUM_GENERATIONS=8 bash scripts/run_evoduet_parallel.sh
MODEL=YOUR_MODEL NUM_GENERATIONS=8 bash scripts/run_baseline_parallel.sh
```

The settings at the top of each script also accept environment overrides:

| Variable | Default |
| --- | --- |
| `MODEL`, `API_BASE` | Each task's configuration |
| `ITERATIONS`, `SEED` | 100, 42 |
| `NUM_GENERATIONS` (parallel scripts) | 8 |
| `INNER_ROUNDS`, `QUERY_COUNT` (EvoDuet scripts) | 3, 1 |
| `DOCUMENTS_PER_QUERY`, `SEARCH_TOP_K` (EvoDuet scripts) | 5, 3 |
| `RUN_NAME` | UTC timestamp |
| `OUTPUT_ROOT` | `outputs/simpletes/openevolve/<script mode>` |

Each task writes to `OUTPUT_ROOT/RUN_NAME/<family>/<task>/`.
Use a separate `OUTPUT_ROOT` for each experiment collection. For example:

```bash
MODEL=YOUR_MODEL ITERATIONS=200 SEED=7 RUN_NAME=experiment-01 \
  bash scripts/run_evoduet.sh --dry-run
```

`--dry-run` validates every selected task without API calls, evaluation, or output
creation. Optional task paths before the flags replace the script's `TASKS` list:

```bash
bash scripts/run_evoduet.sh astrodynamics/cassini scaling_law/easy_question_scaling_law \
  --model YOUR_MODEL --dry-run
```

Paths can be relative to the SimpleTES datasets directory, relative to the
repository root, or absolute. Remaining flags are forwarded to every task;
`--model`, `--iterations`, and `--num-generations` override the script settings.
For a single task, `evoduet-run` and `python -m skydiscover.evoduet_cli` are also
available. Run `evoduet-run --help` for the full interface.

The CLI applies the common settings from the original OpenEvolve experiments:

| Setting | Default |
| --- | --- |
| Temperature / top-p | 0.7 / 0.95 |
| Maximum output tokens | 32768 |
| Reasoning effort | `medium` |
| LLM timeout / retries | 1800 seconds / 3 |
| Checkpoint interval | Every iteration |
| Parallel evaluations | Baseline: 1; EvoDuet: number of candidates |

Model tools, solution confidence, cascade evaluation, and evaluator context
injection are disabled. The task's prompt and evaluator timeout are preserved.
Explicit dotted flags override these defaults, including
`--max_parallel_evaluations 2` and `--llm.max_tokens 16384`.
Model and API endpoint selection still comes from the task config or
`--model` / `--api-base`; set them to the model and endpoint used in your experiment.
The scripts run one seed/edit setting per task (seed 42 and diff edits by default).

The task launcher uses `skydiscover.evoduet.EvoDuet`, which preserves the observed
search behavior of the original `scripts/default_ours_v7/` presets, with Tavily
as the web-search backend. The baseline disables retrieval. Observed search
uses these six templates:

| Template | Role |
| --- | --- |
| `retrieval_gating.txt` | Decide whether to retrieve, look up, or skip |
| `population_analysis.txt` | Analyze the population after a retrieve decision |
| `knowledge_analysis.txt` | Supply knowledge state for always, random, and stagnation gates |
| `query_generation.txt` | Construct the first query |
| `query_refinement.txt` | Refine queries using observed documents |
| `evidence_scoring.txt` | Predict document utility and update knowledge state |

The runtime contains no separate reranker, document summarizer, prompt-policy
optimizer, Claude CLI backend, or RAG seeding workflow. Evidence scoring selects
documents within the observed-search loop. The config loader accepts inactive
settings in older saved configs without exposing them as runtime options. Stored
record fields are retained for checkpoint restoration. CI runs the full EvoDuet test directory,
including Tavily retrieval, checkpoint restoration, and concurrent model routing.

EvoDuet defaults to three inner rounds, one query per round, five retrieved
documents per query, and three selected documents. Adjust these with
`--inner-rounds`, `--query-count`, `--documents-per-query`, and `--search-top-k`.
The parent program stays fixed during these rounds; only the selected evidence
is credited with the next evaluated solution's result.

`--num-generations N` controls solution candidates. EvoDuet uses N after a
retrieve/look-up decision and one after no-op. The baseline uses N every iteration.
These scripts select OpenEvolve. The single-task CLI also accepts `--search` to
select another scaffold; controllers validate support for concurrent candidates.

Select one task and use `--checkpoint PATH` with the same settings to resume a
checkpoint from this release. EvoDuet saves and restores search memory from
`evoduet.json` inside the checkpoint directory. The CLI requires this file and a
complete checkpoint using the current format; missing state and older checkpoint
formats are rejected. Start a new run for older outputs. Baselines do not require
an EvoDuet state file, except EvoX, which stores its controller state there.

`--iterations` is the total target, including completed iterations. For example,
this resumes iteration 40 and runs the remaining 60 iterations:

```bash
uv run --no-sync evoduet-run benchmarks/simpletes/datasets/astrodynamics/cassini \
  --model YOUR_MODEL --iterations 100 \
  --checkpoint outputs/cassini/checkpoints/checkpoint_40
```

When `--output` is omitted from the single-task CLI, a checkpoint under
`RUN/checkpoints/` resumes into `RUN`. The shell scripts supply an output path;
pass `--output` explicitly to resume into the same run directory through a script.
Older research checkpoints with a different namespace are not migrated by this
launcher. The saved best still receives the task's final evaluation; initial
held-out evaluations and NDG postprocessing are not included.

Advanced settings accept dotted flags, such as `--llm.temperature 0.7` or
`--evoduet.retrieval_gating_backend_type always`. `--config`, `--initial-program`,
and `--evaluator` override task file selection when needed.
