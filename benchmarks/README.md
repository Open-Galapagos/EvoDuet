# EvoDuet benchmarks

This release includes the **31 tasks evaluated in the EvoDuet paper** and the
benchmarks distributed with the original SkyDiscover checkout. Each task keeps
its evaluator, seed program, configuration, and required data or setup files.

## Paper tasks

### Simple Scientific Optimization (10 tasks)

| Paper task | Directory |
| --- | --- |
| Pressure Vessel | [pressure_vessel_design](pressure_vessel_design/) |
| SINDy | [sindy_cubic_2d](sindy_cubic_2d/) |
| Batch Reactor | [batch_reactor_control](batch_reactor_control/) |
| LABS-27 | [labs_27](labs_27/) |
| LJ-13 | [lennard_jones_13](lennard_jones_13/) |
| Chwirut2 | [nist_chwirut2](nist_chwirut2/) |
| Perovskites | [olympus_perovskites](olympus_perovskites/) |
| SRSD | [srsd_feynman_i_27_6](srsd_feynman_i_27_6/) |
| GuacaMol | [guacamol_c11h24](guacamol_c11h24/) |
| Summit | [summit_baumgartner](summit_baumgartner/) |

The shared evaluator helpers are `_public_optimization_runtime.py` and
`_srsd_feynman_runtime.py`. Install the selected task's `requirements.txt` and
follow its README for data preparation.

### SimpleTES (21 tasks)

| Domain | Tasks | Count |
| --- | --- | ---: |
| Quantum compilation | Swap Reduction | 1 |
| Astrodynamics | Cassini, Galileo, Mariner 10, Rosetta, Voyager 2 | 5 |
| Scientific algorithms | Denoising | 1 |
| AI foundations | Domain Mixture, U Shape, LR & BSZ, Parallel | 4 |
| Algorithm engineering | AHC039, AHC058 | 2 |
| Mathematics | Erdős, AC1–AC3, CP (n=26, 32), Hadamard, Sums/Diffs | 8 |

See the [SimpleTES guide](simpletes/README.md) for task paths, dependencies,
Docker setup, and final-evaluation behavior. U Shape uses the upstream
`easy_question_scaling_law` directory.

## Original SkyDiscover benchmarks

These ten families are retained from the
[SkyDiscover checkout at `dc10ece`](https://github.com/skydiscover-ai/skydiscover/tree/dc10ece02c55f49fd2f20cc1b5355fab4d8ba9e9/benchmarks):

| Family | Directory |
| --- | --- |
| ADRS | [ADRS](ADRS/) |
| ALE-Bench | [ale_bench](ale_bench/) |
| ARC | [arc_benchmark](arc_benchmark/) |
| Frontier-CS | [frontier-cs-eval](frontier-cs-eval/) |
| GPU MODE | [gpu_mode](gpu_mode/) |
| Image generation | [image_gen](image_gen/) |
| KernelBench | [kernelbench](kernelbench/) |
| Mathematics | [math](math/) |
| Prompt optimization | [prompt_optimization](prompt_optimization/) |
| Quantum circuit topology | [qnn_circuit_topology](qnn_circuit_topology/) |

Their family READMEs describe task-specific setup. The ALE-Bench submodule and
support files needed by the paper tasks are also retained.

## Run a task

After the [repository installation](../README.md#installation), run from the
repository root:

```bash
uv pip install -r benchmarks/pressure_vessel_design/requirements.txt
bash scripts/run_evoduet.sh benchmarks/pressure_vessel_design --dry-run
bash scripts/run_evoduet.sh benchmarks/pressure_vessel_design \
  --model YOUR_MODEL --iterations 100 --seed 42 --output outputs/pressure-vessel
```

Use `run_evoduet_parallel.sh` for multiple solution candidates with retrieval,
or `run_baseline.sh` / `run_baseline_parallel.sh` for program evolution without
retrieval. All four launchers accept the same task directory. Container tasks
additionally use `--evaluator FAMILY_DIRECTORY`, as
shown in the [SimpleTES guide](simpletes/README.md#run). See the
[launcher guide](../scripts/README.md) for all options.
