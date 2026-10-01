# Lennard-Jones 13-atom cluster

A fixed, published chemical-physics optimization task: arrange 13 atoms in
three-dimensional space to minimize their total Lennard-Jones pair energy.
The original search has 39 Cartesian coordinates; translation and rotation
leave the objective unchanged.

Source: David J. Wales and Jonathan P. K. Doye, “Global Optimization by
Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters
Containing up to 110 Atoms,” *Journal of Physical Chemistry A* 101, 5111–5116
(1997), [public paper](https://www-wales.ch.cam.ac.uk/pdf/JPCA.101.5111.1997.pdf).
The [Cambridge table](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html)
publishes the energy and structure files for each cluster size, including
[the 13-atom reference coordinates](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/13).

## Contract and original objective

`solve(payload)` receives `{"n_atoms": 13, "epsilon": 1.0, "sigma": 1.0}` and
returns a list of 13 coordinate lists, each containing three finite numbers.
It receives no optimal structure or target energy. The original potential is

```text
E = 4*epsilon*sum((sigma/r_ij)^12 - (sigma/r_ij)^6, over every i < j)
```

where `r_ij` is the Euclidean separation. All 78 pairs are evaluated without
a cutoff. Reduced units set `epsilon = sigma = 1`. There is no simulation
box, periodic boundary, external potential, or penalty on translations and
rotations. Coincident atoms are invalid. The evaluator rejects malformed
shapes, non-finite values, booleans, and numerical overflow rather than
assigning a finite score to an undefined energy.

For valid coordinates,
`combined_score = min(1, max(0, -energy) / 44.326801)` and `validity = 1`.
The bounded higher-is-better normalization is an adapter addition; `energy`
is the original physical objective. `44.326801` is the magnitude of the
published energy rounded to six decimal places; clipping absorbs the small
extra precision obtained when recomputing the same geometry. Finite positive
energies are valid but score zero. The evaluator computes energy itself and
never accepts a candidate-reported energy.

## Programs and measured validation

The supplied programs are **locally authored adapters**, not an original
upstream initial/best program pair. `initial_program.py` places atoms along
a straight line, with adjacent atoms at the pair equilibrium distance.
`oracle/best_program.py` constructs an atom at the center of a regular
icosahedron and 12 at its vertices. It then analytically minimizes the single
scale parameter of that geometry. If `A` and `B` sum inverse twelfth and sixth
powers of the unscaled pair distances, the scale is
`sigma * (2*A/B)^(1/6)`. It does not load the published coordinate file.

Fresh candidate subprocesses produced these results on 2026-09-09:

| Program | Original energy | Minimum pair distance | Validity | Combined score |
|---|---:|---:|---:|---:|
| Initial | -12.374362631890557 | 1.122462048309373 | 1 | 0.2791620950018603 |
| Reference | -44.32680141953402 | 1.0818382885513658 | 1 | 1.0 |

The evaluator never imports the reference program or reads `oracle/`.
There are no held-out atom counts or generated tasks. The small LJ13 instance
was already studied in the cited literature. No LLM evolution or
no-document/document comparison was run.

[validation.json](validation.json) records these measurements and seven rejected
invalid candidate outputs, covering claimed energies instead of coordinates,
malformed shapes, booleans, infinity, coincident atoms, and numerical overflow.
A combined translation, rotation, and atom permutation preserved the reference
energy. Configuration loading and identical scaffolding outside the two
programs' EVOLVE blocks were also checked.

## Oracle evidence

The author's [short landscape explanation](https://www-wales.ch.cam.ac.uk/~jon/forest/LJ.html)
identifies the complete icosahedron as the LJ13 minimum and describes its
favorable funnel. This is conservatively **inferential** for producing actual
optimal coordinates: the reader must construct and scale the geometry.
Section II of the basin-hopping paper provides an alternative **inferential**
oracle explaining perturbation, local minimization, and acceptance on the
transformed landscape. The [13-atom coordinate file](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/13)
is **direct** fixed-instance answer evidence. The scalar energy table by itself
does not explain how to generate the required coordinate artifact.
Oracle snapshots and search metadata, when selected, are separate from scoring.

## Run

From the repository root, using the repository virtual environment:

```bash
.venv/bin/python benchmarks/lennard_jones_13/evaluator.py benchmarks/lennard_jones_13/initial_program.py
.venv/bin/python benchmarks/lennard_jones_13/evaluator.py benchmarks/lennard_jones_13/oracle/best_program.py
.venv/bin/skydiscover-run benchmarks/lennard_jones_13/initial_program.py benchmarks/lennard_jones_13/evaluator.py -c benchmarks/lennard_jones_13/config.yaml
```

The evaluator and both supplied programs require only the Python standard
library. The shared runner is `benchmarks/_public_optimization_runtime.py`.
Candidate execution is limited to 30 seconds and 2 GiB, with numerical-library
thread counts set to one. It runs in a temporary working directory with only
the candidate source and public input copied there; this is process isolation,
not an OS security sandbox. The default config performs 50 evolution iterations
and needs separately configured model credentials.
