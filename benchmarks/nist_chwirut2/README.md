# NIST Chwirut2: nonlinear ultrasonic-calibration fitting

Fit three real coefficients to the 54 observed pairs in the public NIST StRD
Chwirut2 dataset by minimizing unweighted residual sum of squares:

```text
f(x; b1,b2,b3) = exp(-b1*x) / (b2+b3*x)
RSS = sum((f(x_i)-y_i)**2 for all 54 observations)
```

The response is ultrasonic response, and the predictor is metal distance.
[NIST's dataset page](https://www.itl.nist.gov/div898/strd/nls/data/chwirut2.shtml)
labels it **Lower Level of Difficulty** and attributes the data to
**D. Chwirut, NIST (1979), Ultrasonic Reference Block Study**. The original ASCII
file instead prints `197?`; the 1979 attribution here follows the dataset page.

This is a SkyDiscover adaptation of the original regression-fit problem, with
locally authored Python wrappers and programs. The model, observations and
unweighted objective are preserved. All 54 points are supplied and scored;
there is no train/test split or hidden prediction task.

## Contract and data

`solve(payload)` receives `{"x": [...54 values...], "y": [...54 values...]}`
and returns the plain JSON list `[b1,b2,b3]`. No parameter bounds are added to
the original problem. Coefficients and model predictions must be finite and
model denominators nonzero at the observed points. Wrong shapes, boolean
coefficients, invalid numeric values and execution failures receive zero
validity and score.

`data.json` contains only observed `x` and `y` values and provenance metadata.
It was parsed from the actual fetched
[NIST ASCII file](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Chwirut2.dat)
on 2026-09-09. The original data are ordered `y x`; the arrays name each column
explicitly and preserve row order. The SHA-256 of the fetched source bytes is
recorded in that JSON. Certified parameters and objective values are excluded
from the candidate payload and public data file.

## Evaluator and reference

The evaluator runs the candidate for at most 30 seconds and recomputes every
prediction and residual independently. It reports `rss`, `rmse`, `validity`
and `combined_score = min(1, 513.04802941/RSS)`. The constant is NIST's certified
RSS, used for score normalization. It is not a proof of global optimality over
all real parameter triples. The evaluator never reads `oracle/` and accepts
any finite coefficients based on their actual residuals.

The initial program returns NIST's published Start 1 `[0.1,0.01,0.02]`. The
reference program refines the same start using SciPy's Levenberg–Marquardt
least-squares solver and an analytic Jacobian. It uses only the supplied
observations and does not load certified coefficients. NIST's original file
directly lists certified coefficients and residuals, so it supplies **direct**
oracle evidence for this fixed fit.

Local evaluation on 2026-09-09 with Python 3.12, NumPy 2.4.2, SciPy 1.17.1:

| Program | RSS ↓ | RMSE ↓ | Score ↑ | Validity |
|---|---:|---:|---:|---:|
| `initial_program.py` | 14794.7901547973 | 16.5522685895 | 0.0346776145 | 1 |
| `oracle/best_program.py` | 513.0480294069 | 3.0823512833 | 1 | 1 |

Floating-point results may vary in final digits across solver versions.
No LLM or no-document control experiment was performed; these measurements
verify the task and reference, not the causal effect of supplying a document.

## Run from the repository root

```bash
uv pip install --python .venv/bin/python -r benchmarks/nist_chwirut2/requirements.txt
.venv/bin/python benchmarks/nist_chwirut2/evaluator.py benchmarks/nist_chwirut2/initial_program.py
.venv/bin/python benchmarks/nist_chwirut2/evaluator.py benchmarks/nist_chwirut2/oracle/best_program.py

uv run --no-sync skydiscover-run benchmarks/nist_chwirut2/initial_program.py \
  benchmarks/nist_chwirut2/evaluator.py --config benchmarks/nist_chwirut2/config.yaml \
  --search adaevolve --output outputs/nist_chwirut2
```

The last command requires model credentials and makes model calls. The default
configuration has 50 iterations and disables separate final evaluation for this
fixed public fit. This task imports `benchmarks/_public_optimization_runtime.py`;
retain that shared runner when copying the task.
