# SimpleTES tasks used in EvoDuet

This directory contains the **21 SimpleTES tasks evaluated in the EvoDuet
paper**, adapted from [SimpleTES](https://github.com/wq-will/SimpleTES) at commit
`47d3413da1d85dc24341219d47452d2601e56a57`.

## Tasks

| Task | Upstream | Language | Mode | eval_timeout (s) | SkyDiscover equivalent |
| --- | --- | --- | --- | ---: | --- |
| [`ahc/simpletes_ahc039`](datasets/ahc/simpletes_ahc039/) | `ahc/ahc039` | C++ in a Python `CPP_CODE` wrapper | plain (host Docker judge) | 180 | `ale_bench/ale-bench-lite-problems/ahc039` (ALE-Bench, private set) |
| [`ahc/simpletes_ahc058`](datasets/ahc/simpletes_ahc058/) | `ahc/ahc058` | C++ in a Python `CPP_CODE` wrapper | plain (host Docker judge) | 180 | `ale_bench/ahc058` (earlier port of the same SimpleTES task) |
| [`astrodynamics/mariner_10`](datasets/astrodynamics/mariner_10/) | `astrodynamics/mariner_10` | Python | plain | 3000 | — |
| [`astrodynamics/voyager_2`](datasets/astrodynamics/voyager_2/) | `astrodynamics/voyager_2` | Python | plain | 3000 | — |
| [`astrodynamics/galileo`](datasets/astrodynamics/galileo/) | `astrodynamics/galileo` | Python | plain | 3000 | — |
| [`astrodynamics/cassini`](datasets/astrodynamics/cassini/) | `astrodynamics/cassini` | Python | plain | 3000 | — |
| [`astrodynamics/rosetta`](datasets/astrodynamics/rosetta/) | `astrodynamics/rosetta` | Python | plain | 3000 | — |
| [`autocorrelation/simpletes_autocorrelation_first`](datasets/autocorrelation/simpletes_autocorrelation_first/) | `autocorrelation/autocorrelation_first` | Python | plain | 1200 | `math/first_autocorr_ineq` |
| [`autocorrelation/simpletes_autocorrelation_second`](datasets/autocorrelation/simpletes_autocorrelation_second/) | `autocorrelation/autocorrelation_second` | Python | plain | 1200 | `math/second_autocorr_ineq` |
| [`autocorrelation/simpletes_autocorrelation_third`](datasets/autocorrelation/simpletes_autocorrelation_third/) | `autocorrelation/autocorrelation_third` | Python | plain | 120 | `math/third_autocorr_ineq` |
| [`circle_packing/simpletes_circle_packing_26`](datasets/circle_packing/simpletes_circle_packing_26/) | `circle_packing/circle_packing_26` | Python | plain | 600 | `math/circle_packing` |
| [`circle_packing/simpletes_circle_packing_32`](datasets/circle_packing/simpletes_circle_packing_32/) | `circle_packing/circle_packing_32` | Python | plain | 600 | `math/circle_packing_32` |
| [`erdos/simpletes_erdos_min_overlap`](datasets/erdos/simpletes_erdos_min_overlap/) | `erdos/erdos_min_overlap` | Python | plain | 1200 | `math/erdos_min_overlap` |
| [`hadamard_maximal_det/hadamard_maximal_det_29`](datasets/hadamard_maximal_det/hadamard_maximal_det_29/) | `hadamard_maximal_det/hadamard_maximal_det_29` | Python | plain | 420 | — |
| [`open_problems_bio/denoising`](datasets/open_problems_bio/denoising/) | `open_problems_bio/denoising` | Python | container | 480 | — |
| [`qubit_routing/swap_reduction`](datasets/qubit_routing/swap_reduction/) | `qubit_routing/swap_reduction` | Rust | container | 3000 | — |
| [`scaling_law/domain_mixture_scaling_law`](datasets/scaling_law/domain_mixture_scaling_law/) | `scaling_law/domain_mixture_scaling_law` | Python | plain | 3000 | — |
| [`scaling_law/easy_question_scaling_law`](datasets/scaling_law/easy_question_scaling_law/) | `scaling_law/easy_question_scaling_law` | Python | plain | 3000 | — |
| [`scaling_law/lr_bsz_scaling_law`](datasets/scaling_law/lr_bsz_scaling_law/) | `scaling_law/lr_bsz_scaling_law` | Python | plain | 3000 | — |
| [`scaling_law/parallel_scaling_law`](datasets/scaling_law/parallel_scaling_law/) | `scaling_law/parallel_scaling_law` | Python | plain | 3000 | — |
| [`sums_diffs/simpletes_sums_diffs`](datasets/sums_diffs/simpletes_sums_diffs/) | `sums_diffs/sums_diffs` | Python | plain | 240 | `math/sums_diffs_finite_sets` |

The paper's **U Shape** task is `scaling_law/easy_question_scaling_law`.
Family READMEs describe the upstream problems and scoring. Their SimpleTES
launch commands can be replaced by the EvoDuet commands below.

## Layout and provenance

`datasets/<family>/<task>/` contains the seed, evaluator, and `config.yaml`.
Shared family modules, data, and vendored libraries remain in their upstream
layout so evaluators can resolve their dependencies. `simpletes/` and
`sitecustomize.py` supply the construction helpers imported by those evaluators.

[`UPSTREAM_MANIFEST.json`](UPSTREAM_MANIFEST.json) records the selected tasks,
source paths, and SHA-256 hashes of unchanged upstream files. `renamed` entries
retain upstream bytes under a new path, including `init_program.*` renamed to
`initial_program.*`. `generated` and `glue` entries identify adapter files and
release documentation. The AHC tester executables have the executable bits
required by upstream setup. Bundled libraries retain their own license files.

Plain tasks use `skydiscover_evaluator.py` and the shared
[`skydiscover_adapter.py`](skydiscover_adapter.py), which runs each candidate in a
fresh interpreter with the SimpleTES root on `PYTHONPATH`. The **Denoising** and
**Swap Reduction** tasks use their family directories as Docker build contexts.
AHC tasks use a host adapter that invokes the official Docker judge.

## Setup

Start with the [repository installation](../../README.md#installation).
For the host-evaluated math tasks, install:

```bash
uv pip install numpy scipy scikit-learn psutil cvxpy
```

For astrodynamics and scaling laws, install their additional dependencies and
prepare data before running a search:

```bash
uv pip install -r benchmarks/simpletes/datasets/astrodynamics/requirements.txt
uv pip install datasets
uv run --no-sync python benchmarks/simpletes/datasets/astrodynamics/download_ephemerides.py
(cd benchmarks/simpletes/datasets/scaling_law && uv run --no-sync python prepare_dataset.py)
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

Astrodynamics downloads the NAIF ephemerides; scaling-law tasks use the cached
`pkuHaowei/sldbench` dataset. Some upstream requirements include optional
commercial solvers; follow the family README if a candidate needs those solvers.
Family `pyproject.toml` / `uv.lock` files retain upstream environment pins. To
use a separate evaluator interpreter, set `SIMPLETES_EVAL_PYTHON` to its absolute
path.

Docker is required for AHC, Denoising, and Swap Reduction. The AHC evaluator uses
`yimjk/ale-bench:cpp20-202301` and the bundled official inputs and tester binaries.
The two container families build on first run; they can also be built in advance:

```bash
docker build -t skydiscover-qubit_routing:latest benchmarks/simpletes/datasets/qubit_routing
docker build -t skydiscover-open_problems_bio:latest benchmarks/simpletes/datasets/open_problems_bio
```

The Swap Reduction image supplies Rust and Qiskit. The Denoising image prepares
its pinned Scanpy/OpenProblems environment and downloads the Pancreas data.
[`prepare_pancreas.py`](datasets/open_problems_bio/prepare_pancreas.py) reconstructs
the seed-42 split with the upstream OpenProblems pipeline; the build checks the
result against the evaluator's baseline constants.

## Run

Run from the repository root after task setup:

```bash
bash scripts/run_evoduet.sh \
  benchmarks/simpletes/datasets/circle_packing/simpletes_circle_packing_26 \
  --model YOUR_MODEL --iterations 100 --seed 42

bash scripts/run_evoduet.sh \
  benchmarks/simpletes/datasets/qubit_routing/swap_reduction \
  --evaluator benchmarks/simpletes/datasets/qubit_routing \
  --model YOUR_MODEL --iterations 100 --seed 42

bash scripts/run_evoduet.sh \
  benchmarks/simpletes/datasets/open_problems_bio/denoising \
  --evaluator benchmarks/simpletes/datasets/open_problems_bio \
  --model YOUR_MODEL --iterations 100 --seed 42
```

Add `--dry-run` to check task selection without model calls or evaluation.
The [four OpenEvolve scripts](../../scripts/README.md) provide EvoDuet and baseline
runs with single or parallel solution candidates. Each script has an editable
`TASKS=(...)` list containing all 21 tasks and runs that list when no task paths
are passed. Denoising and Swap Reduction evaluator directories are selected
automatically by these scripts.

## Evaluation details

- **Final evaluation:** the generic final stage invokes `evaluate_final()` when
  provided. Scaling-law tasks fit on training data and score held-out data;
  Denoising uses its container test mode. Native final metrics carry a `test_`
  prefix. [`report_metrics.py`](report_metrics.py) defines the paper's reporting
  metric for each of the 21 tasks, separately from its search reward.
- **Construction state:** the adapter defines `GLOBAL_BEST_CONSTRUCTION` and
  forwards `SIMPLETES_SHARED_CONSTRUCTION_PATH` when set. SkyDiscover does not
  automatically update this upstream warm-start feature during search.
- **Randomness:** some seed and reference programs are stochastic. Their scores
  can vary between evaluations even when the evaluator settings match.
- **Memory and concurrency:** math evaluators derive memory limits from
  `EVALUATOR_CONCURRENT_PROCESSES`. Adjust this for the host (for example, 16).
  AHC uses `SIMPLETES_WORKER_ID` for CPU pinning; set `AHC_PIN_CORES=0` when
  independent searches would otherwise share the same pinned cores. Swap
  Reduction exposes `QUBIT_ROUTING_SLOT_COUNT` for its evaluation slots.
- **Reference programs:** some astrodynamics references assume a deeply nested
  temporary path; use an existing directory such as `/tmp/simpletes/eval` as
  `TMPDIR` when evaluating those programs.

## Compare with upstream

With an upstream checkout and the task's runtime installed, use:

```bash
uv run --no-sync python benchmarks/simpletes/verify_functional.py \
  --upstream /path/to/SimpleTES \
  --task circle_packing/simpletes_circle_packing_26
```

The verifier compares both evaluators on the same candidate. Use `--help` for
interpreter, tolerance, and shared-construction options.
