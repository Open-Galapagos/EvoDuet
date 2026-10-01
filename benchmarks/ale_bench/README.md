# ALE-Bench: AtCoder Heuristic Contest Benchmark

10 problems from AtCoder Heuristic Contests (AHC), evaluated via the `ale_bench` package. Programs are written in C++ and scored on 50 public test cases during evolution. A separate private evaluator runs the full hidden test set for final ranking.

One further AHC problem, [`ahc058/`](ahc058/), is self-contained and does **not** use the `ale_bench` package — see [AHC058](#ahc058) below.

## Problems

| Problem | Description |
|---------|-------------|
| `ahc008` | Pet partitioning — place walls to create pet-free areas on a 30×30 grid over 300 turns |
| `ahc011` | AtCoder Heuristic Contest 11 |
| `ahc015` | AtCoder Heuristic Contest 15 |
| `ahc016` | AtCoder Heuristic Contest 16 |
| `ahc024` | AtCoder Heuristic Contest 24 |
| `ahc025` | Balance weighing — use a balance scale to divide N items into D equal-weight sets using Q queries |
| `ahc026` | AtCoder Heuristic Contest 26 |
| `ahc027` | AtCoder Heuristic Contest 27 |
| `ahc039` | AtCoder Heuristic Contest 39 |
| `ahc046` | AtCoder Heuristic Contest 46 |

## Quick Start

Run evolution on a single problem:

```bash
uv run skydiscover-run \
  benchmarks/ale_bench/ale-bench-lite-problems/ahc025/initial_program.cpp \
  benchmarks/ale_bench/ale-bench-lite-problems/ahc025/evaluator.py \
  -c benchmarks/ale_bench/ale-bench-lite-problems/ahc025/config.yaml \
  --search evox \
  -i 100
```

## Scoring

During evolution, each iteration runs 50 public test cases:

```
combined_score = overall_absolute_score * optim_factor / num_public_cases
```

`optim_factor` is `+1` for maximize problems and `-1` for minimize problems (so `combined_score` is always higher-is-better).

## Private Evaluation

Each problem's `evaluate_final()` scores the best program on the full private test set, and SkyDiscover calls it automatically when the search ends. The results land in `best/best_program_info.json` under a `test_` prefix:

```
test_private_score          Contest-unit absolute score on the private set
test_private_rank           Estimated AtCoder rank
test_private_performance    Estimated performance score
test_num_private_passed_cases / test_num_private_failed_cases
test_combined_score         private_score, signed so higher is better
```

The private set is much larger than the 50 public cases evolution uses — 150 cases for ahc024/026/039/046, but 2000 for ahc008/016/027, 3000 for ahc011 and 5000 for ahc025 — so this stage takes minutes to hours. It defaults to one run; `AHC_PRIVATE_EVAL_RUNS=3` averages three. To skip it entirely:

```yaml
evaluator:
  final_evaluation: false
```

The same evaluation is still available by hand, which additionally averages 3 runs:

```bash
python benchmarks/ale_bench/private_eval.py \
  --program-path path/to/best_program.cpp \
  --problem-id ahc025
```

## Directory Structure

```
ale_bench/
├── ale-bench-lite-problems/
│   └── ahcXXX/
│       ├── initial_program.cpp   # Starting C++ solution
│       ├── evaluator.py          # Runs 50 public cases via ale_bench
│       └── config.yaml           # Search config (cpp, diff-based, 100 iterations)
├── ale_agent_best/
│   └── ahcXXX.cpp               # Best known solutions (reference)
├── ahc058/                      # Self-contained SimpleTES task (own tester + inputs)
└── private_eval.py              # Full private set evaluation + ranking
```

## AHC058

[`ahc058/`](ahc058/) is AtCoder Heuristic Contest 58, taken from the SimpleTES distribution because
the `ale_bench` package has no ahc058. It bundles its own official tester and 150 public inputs, and
judges candidates in the pinned `yimjk/ale-bench:cpp20-202301` image through the host Docker daemon.
The evolved file is a Python wrapper holding the C++ solution in a `CPP_CODE` string, so its config
uses `language: python`.

```bash
uv run skydiscover-run \
  benchmarks/ale_bench/ahc058/initial_program.py \
  benchmarks/ale_bench/ahc058/evaluator.py \
  -c benchmarks/ale_bench/ahc058/config.yaml \
  --search evox \
  -i 100
```

It has no held-out split: SimpleTES ships only the 150 public inputs, so its `total_score` is
measured on exactly the inputs the search optimized against. See [`ahc058/README.md`](ahc058/README.md).

## Requirements

Requires the `ale_bench` and `ale_bench_eval` packages, vendored here as the
`ALE-Bench/` git submodule. They are not in the default `uv sync` — ask for the
extra:

```bash
git submodule update --init benchmarks/ale_bench/ALE-Bench
uv sync --extra ale-bench
```

The extra pulls `ale_bench[eval]`; the `[eval]` part is not optional, because
`ale_bench_eval` imports it at package import. Installing the package ad hoc
(`uv pip install ...`) also works, but the next `uv sync` prunes it again — that
is what the extra exists to prevent.

## Config Defaults

All lite problems share the same base config (ahc058 differs — it uses `language: python`):

```yaml
language: cpp
diff_based_evolution: true
max_iterations: 100
max_solution_length: 60000
evaluator:
  timeout: 10000
```
