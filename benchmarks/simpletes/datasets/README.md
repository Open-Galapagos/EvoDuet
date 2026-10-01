# SimpleTES datasets in EvoDuet

This release contains the following 21 tasks from SimpleTES. See the
[task guide](../README.md) for individual task paths, installation, and launch
commands, and the [source manifest](../UPSTREAM_MANIFEST.json) for provenance.

| Domain | Family | Tasks |
| --- | --- | ---: |
| Quantum compilation | [qubit_routing](qubit_routing/) | 1 |
| Astrodynamics | [astrodynamics](astrodynamics/) | 5 |
| Scientific algorithms | [open_problems_bio](open_problems_bio/) | 1 |
| AI foundations | [scaling_law](scaling_law/) | 4 |
| Algorithm engineering | [ahc](ahc/) | 2 |
| Mathematics | [autocorrelation](autocorrelation/) | 3 |
| Mathematics | [circle_packing](circle_packing/) | 2 |
| Mathematics | [erdos](erdos/) | 1 |
| Mathematics | [hadamard_maximal_det](hadamard_maximal_det/) | 1 |
| Mathematics | [sums_diffs](sums_diffs/) | 1 |

Each task supplies `initial_program.*`, `config.yaml`, and either a
`skydiscover_evaluator.py` adapter or a family-level Docker evaluator. Shared
libraries and data stay beside their task families in the upstream layout.
