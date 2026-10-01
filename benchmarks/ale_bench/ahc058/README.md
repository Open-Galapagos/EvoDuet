# AHC058: Apple Incremental Game

AtCoder Heuristic Contest 58, taken from the [SimpleTES](https://github.com/wq-will/SimpleTES)
distribution (`datasets/ahc/ahc058`). The task bundles the complete source prompt, the official
Linux tester, and the exact 150 public input files, so it is self-contained.

Unlike the problems in [`../ale-bench-lite-problems/`](../ale-bench-lite-problems/), ahc058 does
**not** go through the `ale_bench` package — that package has no ahc058. Everything it needs lives
in this directory.

## Requirements

- Docker on the host, reachable by the evaluator.
- The pinned judge image `yimjk/ale-bench:cpp20-202301` (pull it once: `docker pull yimjk/ale-bench:cpp20-202301`).

## Run

The evolved file is a Python wrapper whose `CPP_CODE` string holds the C++ solution:

```bash
uv run skydiscover-run \
  benchmarks/ale_bench/ahc058/initial_program.py \
  benchmarks/ale_bench/ahc058/evaluator.py \
  -c benchmarks/ale_bench/ahc058/config.yaml \
  --search evox \
  -i 100
```

## Evaluation

For each of three runs, the evaluator compiles `CPP_CODE` in the pinned image and judges all 150
cases with the bundled official tester. A rejected or timed-out case makes that run's combined
score zero. Otherwise:

```text
run_score      = (sum(case_scores) / 150) / 3000000
combined_score = mean(run_score across 3 runs)
```

`total_score` — the same sum in AtCoder units, before the `/150/3000000` normalization — is the
number to quote when comparing against the contest.

**No held-out split.** SimpleTES ships only `public_inputs_150` and no private set, so
`total_score` comes from exactly the 150 inputs the search optimized against. The ALE-Bench
problems next door are different: their private evaluation scores the frozen winner on a set the
search never saw. Both are contest-unit sums, but only one is evidence of generalization.

Because there is nothing to hold out, this task defines no `evaluate_final()`. The post-search
test-mode pass therefore just re-runs the same 150-case evaluation (~2 minutes); set
`evaluator.final_evaluation: false` to skip it.

The full evaluation is deliberately long: 150 cases, a two-second candidate limit per case, eight
case workers, and three complete runs (~2 minutes per run on an idle host).

## Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `AHC_DOCKER_IMAGE` | `yimjk/ale-bench:cpp20-202301` | Judge image |
| `AHC_CASE_WORKERS` | `8` | Parallel case workers inside the judge |
| `AHC_DOCKER_TIMEOUT` | `180` | Per-run wall limit, seconds |
| `AHC_PIN_CORES` | `1` | Pin the judge to a fixed cpuset |
| `SIMPLETES_WORKER_ID` | `0` | Which cpuset block to pin to |
| `AHC_CACHE_DIR` | `./cache` | Bundled inputs and tester |

With `AHC_PIN_CORES=1` and the default worker id, the judge is pinned to cores 0–7. If you run
several evaluations on one host without giving each a distinct `SIMPLETES_WORKER_ID`, they will
contend for the same cores — set `AHC_PIN_CORES=0` in that case.

## Files

```
ahc058/
├── initial_program.py     # Python wrapper; CPP_CODE holds the evolved C++ solution
├── evaluator.py           # Starts the judge container, aggregates 3 runs
├── docker_runner.py       # Runs inside the judge: compile, run 150 cases, score, emit JSON
├── config.yaml            # Search config; system message is the upstream prompt verbatim
├── UPSTREAM_PROMPT.txt    # The SimpleTES problem prompt
└── cache/
    ├── public_inputs_150/ # The 150 official public inputs
    └── tester_binaries/   # The official Linux tester
```

## Source

- Contest: <https://atcoder.jp/contests/ahc058>
- Upstream: SimpleTES `datasets/ahc/ahc058`, revision `b7e0367b5ab19554fb40859e8069e091bb8c4ca2`
- License: AGPL-3.0-or-later; the official AtCoder tester and input terms also apply
