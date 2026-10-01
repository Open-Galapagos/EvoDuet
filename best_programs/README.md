# Best programs found by EvoDuet

The eleven programs behind the paper's best-programs table, organized by
domain. Each task directory holds the evolved program (`<task>_best.py`, or
`<task>_best.rs` for Swap Reduction) and the run's saved evaluation record
(`<task>_best_info.json`). Every file is a byte-for-byte copy of the selected
run's output. The seven construction tasks also include the evaluated
construction (`<task>_best_construction.json`).

SimpleTES is the released SimpleTES program or construction scored with the
same evaluator. EvoDuet improves it on eight tasks and matches it within
$10^{-9}$ on Hadamard and both circle-packing tasks.

## Quantum Compilation

| Task | What it is | SimpleTES | EvoDuet | Model | Search |
| --- | --- | ---: | ---: | --- | --- |
| [`quantum_compilation/swap_reduction/`](quantum_compilation/swap_reduction/) ↓ | Routing policy for two-qubit gates on a superconducting chip, minimizing added SWAPs (Rust) | 15,186 | **14,835** | GPT-5.6-Luna | Parallel EvoDuet (8 candidates) |

## Astrodynamics

| Task | What it is | SimpleTES | EvoDuet | Model | Search |
| --- | --- | ---: | ---: | --- | --- |
| [`astrodynamics/rosetta/`](astrodynamics/rosetta/) ↓ | Gravity-assist trajectory design for Rosetta | 1.552968 | **1.396424** | GPT-5.6-Sol | EvoDuet |
| [`astrodynamics/voyager_2/`](astrodynamics/voyager_2/) ↓ | Gravity-assist trajectory design for Voyager 2 | 3.430214 | **3.430206** | GPT-5.6-Luna | Parallel EvoDuet (8 candidates) |

## Scientific Algorithms

| Task | What it is | SimpleTES | EvoDuet | Model | Search |
| --- | --- | ---: | ---: | --- | --- |
| [`scientific_algorithms/denoising/`](scientific_algorithms/denoising/) ↑ | Single-cell RNA-seq denoising policy, evaluated on the OpenProblems PBMC and Tabula datasets | 0.722690 | **0.722906** | GPT-5.6-Luna | Parallel EvoDuet (8 candidates) |

## AI Foundations

| Task | What it is | SimpleTES | EvoDuet | Model | Search |
| --- | --- | ---: | ---: | --- | --- |
| [`ai_foundations/domain_mixture_scaling_law/`](ai_foundations/domain_mixture_scaling_law/) ↑ | Symbolic scaling law for the domain-mixture split | 0.996922 | **0.997062** | GPT-5.6-Luna | Parallel EvoDuet (8 candidates) |
| [`ai_foundations/parallel_scaling_law/`](ai_foundations/parallel_scaling_law/) ↑ | Symbolic scaling law for the parallel split | 0.999970 | **0.999975** | Gemini-3.8-Flash | Parallel EvoDuet (8 candidates) |

## Mathematics Discovery

| Task | What it is | SimpleTES | EvoDuet | Model | Search |
| --- | --- | ---: | ---: | --- | --- |
| [`mathematics_discovery/simpletes_erdos_min_overlap/`](mathematics_discovery/simpletes_erdos_min_overlap/) ↓ | Erdős minimum overlap problem | 0.380868 | **0.380859** | GPT-5.6-Luna | Parallel EvoDuet (8 candidates) |
| [`mathematics_discovery/hadamard_maximal_det_29/`](mathematics_discovery/hadamard_maximal_det_29/) ↑ | ±1 matrix of order 29 maximizing the absolute determinant | 0.935673 | 0.935673 | Gemini-3.8-Flash | EvoDuet |
| [`mathematics_discovery/simpletes_sums_diffs/`](mathematics_discovery/simpletes_sums_diffs/) ↑ | Finite set A maximizing log(\|A+A\|/\|A\|) / log(\|A-A\|/\|A\|) | 1.144887 | **1.144999** | Gemini-3.8-Flash | Parallel EvoDuet (8 candidates) |
| [`mathematics_discovery/simpletes_circle_packing_26/`](mathematics_discovery/simpletes_circle_packing_26/) ↑ | 26 non-overlapping circles in a unit square, maximizing the sum of radii | 2.635983 | 2.635983 | GPT-5.6-Luna | EvoDuet |
| [`mathematics_discovery/simpletes_circle_packing_32/`](mathematics_discovery/simpletes_circle_packing_32/) ↑ | Same task at N = 32 | 2.939573 | 2.939573 | GPT-5.6-Luna | Parallel EvoDuet (16 candidates) |

↑ higher is better, ↓ lower is better. Bold marks an improvement over SimpleTES.
All runs use 100 iterations and seed 42, except Domain Mixture Scaling (seed 43).

## Reproduction

Each program was re-evaluated once with its original evaluator: the
construction evaluator for the seven construction tasks, the held-out final
evaluation for the two scaling laws, and the benchmark Docker images for Swap
Reduction and Denoising (held-out PBMC and Tabula). Scored with
[`report_metrics.task_score`](../benchmarks/simpletes/report_metrics.py), all
eleven reproduce the table values exactly, with a difference of 0 at full
precision. [`reproduction.json`](reproduction.json) records each score,
runtime, and evaluator or image hash. The evaluator files are byte-identical to
those under [`benchmarks/simpletes/`](../benchmarks/simpletes/).

## Running the programs

Each program is a candidate for its original benchmark harness under
[`benchmarks/simpletes/datasets/`](../benchmarks/simpletes/datasets/), and needs
that task's environment, dependencies, and data. Rosetta and Voyager 2 import
the benchmark's trajectory tools, Swap Reduction uses the benchmark's Rust
interfaces, and the Erdős program downloads public witness files at runtime.

## File conventions

| File | What it is |
| --- | --- |
| `<task>_best.py` | Evolved Python program |
| `<task>_best.rs` | Evolved Rust program (Swap Reduction) |
| `<task>_best_info.json` | Saved evaluation record: program ID, iteration, metrics, and `test_` final-evaluation metrics |
| `<task>_best_construction.json` | Construction captured during re-evaluation, stored as tagged JSON (numpy arrays round-trip via `simpletes.construction.decode_construction`); astrodynamics files list the trajectory events |
| `reproduction.json` | Re-evaluation results for all eleven programs |
