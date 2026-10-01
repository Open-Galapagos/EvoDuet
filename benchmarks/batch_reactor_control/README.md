# Batch reactor temperature control

Domain: chemical engineering / dynamic optimization.

This ports the public [APMonitor Batch Reactor Optimization](https://www.apmonitor.com/wiki/index.php/Apps/BatchReactor)
example. A desired intermediate `B` is formed from reactant `A`, then lost to
product `C`. The unchanged fixed problem maximizes `B(1)` subject to

\[
\dot A=-k_1 A^2,\quad \dot B=k_1 A^2-k_2B,\quad \dot C=k_2B,
\]
\[
k_1=4000e^{-2500/T},\quad k_2=620000e^{-5000/T},\quad
[A,B,C](0)=[1,0,0],\quad 298\le T(t)\le398.
\]

Time runs from `0` to `1`; temperature is in kelvin. The source provides a
GEKKO formulation with 501 time points and reports the terminal value
**0.6108**. Software reference: Beal et al., *GEKKO Optimization Suite*,
Processes 6(8), 106 (2018), [doi:10.3390/pr6080106](https://doi.org/10.3390/pr6080106).

## Candidate contract and grading

Implement `solve(payload)`, returning exactly `{"temperature": [501 numbers]}`.
`payload.time` is `linspace(0,1,501)`; it also supplies `initial_state`,
`temperature_bounds`, `interpolation`, and the four Arrhenius constants in
`kinetics`. Every temperature must be finite and within `[298,398]`.

The evaluator linearly interpolates consecutive temperatures. It independently
integrates the three differential equations with adaptive DOP853 on every
grid interval, at `rtol=1e-10`, `atol=1e-12`. It checks mass conservation and
nonnegative final concentrations. `combined_score` is the independently
computed **B(1)**, without rounding or clipping to the reported 0.6108.
Malformed, nonfinite, out-of-bounds or failed results receive zero validity
and zero score. Candidate-reported concentrations and objectives are not inputs.

**Port changes:** the continuous mathematical objective, kinetics, initial
conditions, horizon and temperature bounds are retained. The JSON control
artifact and explicit linear interpolation are introduced. The source's GEKKO
code sets `T.DCOST=1e-6`; this port grades only the stated mathematical objective
`B(1)` and adds no control-change penalty or slew constraint. The independent
integration uses SciPy instead of GEKKO collocation. No new reactor instances
or train/test split are claimed.

The initial program holds `T=350 K`. The authored reference in
`oracle/best_program.py` optimizes 51 temperature knots by bounded L-BFGS-B
and interpolates them onto the required 501-point grid. Its inexpensive
objective uses 4000-point midpoint quadrature of the integral solution for
`A` and `B`. This is independent of the evaluator's adaptive ODE integration.
The reference is a locally computed feasible solution, not a copied upstream
program or a certificate of global optimality.

## Local validation and execution

Measured with Python 3.12, NumPy 2.4.2 and SciPy 1.17.1:

| Program | Independently computed B(1) | Temperature range, K |
|---|---:|---:|
| Initial constant 350 K | 0.577061946781 | 350–350 |
| Authored optimized profile | 0.610784102821 | 326.290824–398 |
| Source's reported optimum | 0.6108, rounded | Source report |

The reference's final mass-balance error was `1.55e-15`. Both candidates
completed local evaluation in under one second on the available machine;
this is an observation, not a runtime guarantee.

```bash
.venv/bin/python benchmarks/batch_reactor_control/evaluator.py benchmarks/batch_reactor_control/initial_program.py
.venv/bin/python benchmarks/batch_reactor_control/evaluator.py benchmarks/batch_reactor_control/oracle/best_program.py
.venv/bin/python -m unittest discover -s benchmarks/batch_reactor_control/tests
```

The shared runner executes the candidate in a separate process and transfers
only JSON artifacts to grading. This is not a hardened sandbox for hostile
programs. Task dependencies are only NumPy and SciPy; neither GEKKO nor a
remote optimization service is needed.
