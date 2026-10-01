# SINDy cubic 2D oscillator

Domain: scientific model discovery / sparse system identification.

This ports the **Cubic 2D ODE** experiment from the [PySINDy original-paper
examples](https://pysindy.readthedocs.io/en/latest/examples/sindy-original-example/30f4dcd/original_paper.html#cubic-2d-ode),
which reproduces Brunton, Proctor, and Kutz, *Discovering governing equations
from data by sparse identification of nonlinear dynamical systems*, PNAS
113(15), 3932–3937 (2016), [doi:10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113).
The published fixed system is

\[
\dot x=-0.1x^3+2y^3,\qquad \dot y=-2x^3-0.1y^3.
\]

The observations start at `[2,0]`, at `dt=0.01` over `[0,25)`. The published
example fits a polynomial library with sequentially thresholded least squares.
Its task is recovery from measured states, without measured derivatives.
`training_data.json` is regenerated with the same system, times, initial state,
LSODA solver and `rtol=atol=1e-12`. It contains no added noise. Run
`.venv/bin/python benchmarks/sindy_cubic_2d/generate_data.py` to regenerate it.

## Candidate contract and grading

Implement `solve(payload)`. The evaluator passes `t`, `x`, and `powers` as JSON.
Return `{"coefficients": [[...], [...]]}` with shape **2 × 10**, rows for
`xdot` and `ydot` and columns ordered:

`1, x, y, x², xy, y², x³, x²y, xy², y³`.

The independent evaluator computes vector-field NRMSE on the public 25 × 25
Cartesian grid over `[-2,2]²`. It also integrates the submitted field at
initial states `[1.5,0.5]` and `[-1,0.75]`, sampling 501 times over `[0,10]`.
Each NRMSE is RMSE divided by the RMS of the reference values, aggregating
both coordinates; trajectory NRMSE averages the two trajectories.

`combined_score = 1 / (1 + (vector_field_nrmse + trajectory_nrmse) / 2)`.

Finite coefficient magnitudes must not exceed `1e6`. Wrong output shapes,
nonfinite values, solver failures, trajectories leaving `[-100,100]²`, or
trajectories requiring over 50000 RHS calls receive validity and score zero.
The evaluator reports the number of coefficients exceeding `1e-8` in magnitude;
that diagnostic does not add a sparsity penalty to the objective.

**Port changes:** the original is a reproducible paper experiment, not a
packaged scalar-score benchmark. This port introduces the above scalar metric,
validation grid and initial states, JSON interface, and degree-three output
representation. The upstream example searches through degree five; degree
three still contains its unchanged true system. No new target coefficients,
task instances, or train/test claim are introduced.

The initial program performs a linear fit. `oracle/best_program.py` is a locally
authored reference using fourth-order finite differences, a cubic library,
thresholding and least-squares refitting. It fits the input observations; it
does not hardcode the target coefficients. It is not an upstream program or
a certified globally best program.

## Local validation and execution

Measured with Python 3.12, NumPy 2.4.2 and SciPy 1.17.1:

| Program | Score | Field NRMSE | Trajectory NRMSE | Active coefficients |
|---|---:|---:|---:|---:|
| Initial linear fit | 0.5172071093 | 0.6695886547 | 1.1973342162 | 6 |
| Authored sparse reference | 0.9999463618 | 0.0000045652 | 0.0001027169 | 4 |

From the repository root with its virtual environment:

```bash
.venv/bin/python benchmarks/sindy_cubic_2d/evaluator.py benchmarks/sindy_cubic_2d/initial_program.py
.venv/bin/python benchmarks/sindy_cubic_2d/evaluator.py benchmarks/sindy_cubic_2d/oracle/best_program.py
.venv/bin/python -m unittest discover -s benchmarks/sindy_cubic_2d/tests
```

Candidates run through the shared subprocess JSON runner; candidate-reported
scores are never trusted. This is process separation, not a security boundary
against intentionally hostile programs. NumPy and SciPy are the only task
dependencies; PySINDy is not required.
