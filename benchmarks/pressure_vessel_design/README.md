# Pressure vessel design

A fixed, published mechanical-engineering optimization benchmark: minimize the
fabrication cost of a cylindrical pressure vessel with hemispherical ends.
This adapter preserves the **discrete-thickness, length-at-most-200** instance.

Source: Xin-She Yang, Christian Huyck, Mehmet Karamanoglu, and Nawaz Khan,
“True Global Optimality of the Pressure Vessel Design Problem: A Benchmark for
Bio-Inspired Optimisation Algorithms,” *International Journal of Bio-Inspired
Computation* 5(6), 329–335 (2013), [public paper](https://arxiv.org/pdf/1403.7793).
The problem is defined in Section 2 and the reference design and derivation are
in Section 3. The paper surveys earlier uses of this benchmark.

## Contract and original objective

`solve(payload)` returns `[d1, d2, r, L]`: shell thickness, head thickness,
inner radius, and cylindrical length in inches. The evaluator supplies a new
dictionary of the public constants for this single instance. It supplies no
optimal design or target objective.

Minimize

```text
cost = 0.6224*d1*r*L + 1.7781*d2*r^2 + 3.1661*d1^2*L + 19.84*d1^2*r
```

subject to

```text
d1, d2 in {0.0625, 2*0.0625, ..., 99*0.0625}
10 <= r <= 200; 10 <= L <= 200
d1 >= 0.0193*r; d2 >= 0.00954*r
pi*r^2*L + 4*pi*r^3/3 >= 1296000
L <= 240
```

The last constraint is redundant under the original `L <= 200` bound but is
retained. A continuous-thickness variant or a variant allowing lengths up to
240 has a different optimum and is not this task.

The evaluator recomputes cost, volume, every constraint, and integrality from
the returned four numbers. Non-finite values, booleans, malformed outputs,
infeasible designs, candidate exceptions, and execution timeouts score zero.
The feasibility tolerance is `1e-9`: the volume residual is divided by 1296000,
other continuous residuals are in inches, and thickness integrality is checked
in units of the thickness index. No design is rounded or repaired for scoring.

For a valid design, `combined_score = min(1, 6059.714335048436 / cost)`;
`validity = 1`. This bounded higher-is-better normalization is an adapter
addition; `cost` remains the original objective. The normalization constant is
the published optimum. A candidate gets no credit for merely reporting a cost.

## Programs and measured validation

These are **locally authored adapter programs**, not an original upstream
initial/best program pair. `initial_program.py` returns a conservative feasible
vessel with excess material and capacity. `oracle/best_program.py` implements
the published active-constraint construction: use the maximum length to derive
the minimum feasible radius, round the required thicknesses up to the allowed
grid, then compute radius and length at the limiting constraints. This
construction is used only for this fixed published instance.

Fresh candidate subprocesses produced the following results on 2026-09-09:

| Program | Original cost | Volume | Validity | Combined score |
|---|---:|---:|---:|---:|
| Initial | 10905.3359375 | 1701696.0206944712 | 1 | 0.5556650771491591 |
| Reference | 6059.714335048436 | 1296000 | 1 | 1.0 |

The reference design is
`[0.8125, 0.4375, 42.09844559585492, 176.6365958424394]` up to floating-point
roundoff. Evaluation never imports the reference program or reads `oracle/`.
There are no held-out cases: this is one fixed design task, as in the paper.
No LLM evolution or no-document/document comparison was run.

[validation.json](validation.json) records these measurements and eight rejected
invalid candidate outputs, covering claimed costs instead of designs, malformed
shapes, booleans, NaN, off-grid thicknesses, inadequate capacity, stress failure,
and the length bound. Configuration loading and identical scaffolding outside
the two programs' EVOLVE blocks were also checked.

## Oracle evidence

The public paper is **direct** evidence for this fixed task: it provides both
the design and the cost, and explains the reduction using discrete thicknesses
and active constraints. Section 3 is the useful explanatory material, rather
than just a scalar target. Oracle snapshots and search metadata, when selected,
are separate from the evaluator and candidate payload.

## Run

From the repository root, using the repository virtual environment:

```bash
.venv/bin/python benchmarks/pressure_vessel_design/evaluator.py benchmarks/pressure_vessel_design/initial_program.py
.venv/bin/python benchmarks/pressure_vessel_design/evaluator.py benchmarks/pressure_vessel_design/oracle/best_program.py
.venv/bin/skydiscover-run benchmarks/pressure_vessel_design/initial_program.py benchmarks/pressure_vessel_design/evaluator.py -c benchmarks/pressure_vessel_design/config.yaml
```

The evaluator and both supplied programs require only the Python standard
library. The shared runner is `benchmarks/_public_optimization_runtime.py`.
Candidate execution is limited to 30 seconds and 2 GiB, with numerical-library
thread counts set to one. It runs in a temporary working directory with only
the candidate source and public input copied there; this is process isolation,
not an OS security sandbox. The default config performs 50 evolution iterations
and needs separately configured model credentials.
