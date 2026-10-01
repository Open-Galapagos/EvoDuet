# LABS-27: low-autocorrelation binary sequence optimization

Construct 27 signs in `{-1,+1}` that minimize the sum of squared nonperiodic
autocorrelations. This is the fixed `n=27` instance of the public
[CSPLib problem 005](https://www.csplib.org/Problems/prob005/), proposed for that
library by Toby Walsh. Applications include communication and electrical
engineering and the ground states of the Bernasconi spin model.

This directory adapts the published optimization problem to SkyDiscover's
program-generation interface. Its wrapper, initial program and reference
program are locally authored; they are not claimed to be original CSPLib code.
The instance and objective are unchanged. There is no private or held-out set.

## Contract and score

`solve({"n": 27})` returns a plain JSON list of exactly 27 numeric signs.
The evaluator calculates

```text
C[k] = sum(s[i] * s[i+k] for i in range(27-k)), k=1,...,26
energy = sum(C[k]**2 for k=1,...,26)
combined_score = min(1, 37/energy)
```

It also reports `energy`, `merit_factor = 27**2/(2*energy)` and `validity`.
Malformed shapes, booleans, nonfinite numbers and entries other than ±1 receive
zero validity and score. Candidate execution has a 30-second limit; objective
calculation occurs independently in the evaluator process. The evaluator does
not read `oracle/`.

## Reference and oracle evidence

Tom Packebusch and Stephan Mertens, **Low Autocorrelation Binary Sequences**,
*Journal of Physics A: Mathematical and Theoretical* **49** (2016), 165001,
[DOI: 10.1088/1751-8113/49/16/165001](https://doi.org/10.1088/1751-8113/49/16/165001).
The [open paper](https://arxiv.org/html/1512.02475) gives the optimal energy
`37` and a corresponding run-length encoding in Table 1. That table is direct
oracle evidence. The first preprint dates to 2015; the journal publication is
2016.

The reference program applies the paper's skew-symmetry relation and enumerates
all `2**14` free assignments. It computes the objective to choose its output and
does not read a stored answer. This restricted search attains the unrestricted
optimum for `n=27`, as documented in the paper. The relation does **not** preserve
the unrestricted optimum for every odd length.

The seed selects the best of 32 sequences from Python's RNG with seed 0.
Local evaluation on 2026-09-09 with Python 3.12 and NumPy 2.4.2:

| Program | Energy ↓ | Merit factor ↑ | Score ↑ | Validity |
|---|---:|---:|---:|---:|
| `initial_program.py` | 177 | 2.0593220339 | 0.2090395480 | 1 |
| `oracle/best_program.py` | 37 | 9.8513513514 | 1 | 1 |

These are program evaluations, not an LLM experiment. Whether the oracle helps
a model relative to a no-document run remains unmeasured.

## Run from the repository root

```bash
uv pip install --python .venv/bin/python -r benchmarks/labs_27/requirements.txt
.venv/bin/python benchmarks/labs_27/evaluator.py benchmarks/labs_27/initial_program.py
.venv/bin/python benchmarks/labs_27/evaluator.py benchmarks/labs_27/oracle/best_program.py

uv run --no-sync skydiscover-run benchmarks/labs_27/initial_program.py \
  benchmarks/labs_27/evaluator.py --config benchmarks/labs_27/config.yaml \
  --search adaevolve --output outputs/labs_27
```

The last command requires model credentials and makes model calls. The default
configuration has 50 iterations and disables separate final evaluation because
every evaluation scores the same public instance. This task imports the shared
runner `benchmarks/_public_optimization_runtime.py`; keep it when copying the task.
