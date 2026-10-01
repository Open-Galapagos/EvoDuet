# Summit Baumgartner aniline cross-coupling

An adapter of the published **Summit** reaction-optimization benchmark,
`BaumgartnerCrossCouplingEmulator`. Domain: reaction chemistry / process
optimization. Difficulty: small mixed categorical/continuous task; actual LLM
difficulty and document benefit are unmeasured. Source year: 2021.

- [Summit paper, Felton et al., Chemistry–Methods (2021)](https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/cmtd.202000051)
- [Official implementation](https://github.com/sustainable-processes/summit/blob/1de682d05e97adcfb96cd8376e876cef2d6160d3/summit/benchmarks/experimental_emulator.py)
- [Original experiment, Baumgartner et al. (2019)](https://doi.org/10.1021/acs.oprd.9b00236)

## Registration review and preserved variant

The original library provides the selected variant directly:
`get_pretrained_baumgartner_cc_emulator(include_cost=False, use_descriptors=False)`.
This port retains its five pretrained networks, feature order, one-hot category
order, per-member input/output standardization, and clipping each predicted yield
to [0, 1] **before** averaging. The objective is that averaged predicted yield.
The paper also studies yield/cost Pareto fronts; this is the library's supported
**yield-only** variant, not that multi-objective experiment.

The old Summit training dependency stack is not required. The small frozen
networks are evaluated with NumPy float32. The exact original tensors, source,
JSON scaler parameters, MIT license, pinned commit and SHA-256 hashes are bundled
in `data/` and [UPSTREAM_MANIFEST.json](UPSTREAM_MANIFEST.json). There is no fitting,
download, network request or GPU use at evaluation time. Rebuilding the lossless
NumPy tensor export uses `rebuild_weights.py` and optionally PyTorch.

## Candidate contract

`solve(payload)` returns a dictionary with exactly these five fields:

| Field | Allowed value |
|---|---|
| `catalyst` | `tBuXPhos`, `tBuBrettPhos`, `AlPhos` |
| `base` | `DBU`, `BTMG`, `TMG`, `TEA` |
| `base_equivalents` | Real number in [1.0, 2.5] |
| `temperature` | Real number in [30.0, 100.0], degrees Celsius |
| `t_res` | Real number in [60.0, 1800.0], seconds |

The payload supplies these options and bounds, not model weights, observations,
or good conditions. Each call submits one design. SkyDiscover's evolution budget
controls the number of proposals; the adapter does not reproduce Summit's whole
optimizer leaderboard or provide a callable simulator inside the candidate.
`combined_score = predicted_yield`; higher is better. Additional metrics report
ensemble standard deviation, unclipped mean and the number of clipped members.
These are virtual predictions, not experimentally measured yields.

Malformed outputs, boolean/nonfinite numbers, unknown categories, out-of-bounds
conditions, candidate exceptions and timeouts score zero with `validity=0`.
Conditions are neither rounded nor repaired. There are no hidden task instances.
Evaluation never reads `oracle/`. The candidate runs for at most 30 seconds and
2 GiB in a disposable subprocess. This is process isolation, not an OS security
sandbox; candidates must not read evaluator/model/oracle files.

## Programs and measured results

Both programs are locally authored adapters, not an upstream initial/best pair.
Their fixed scaffolds are identical outside the EVOLVE block. The initial program
uses tBuXPhos/DBU and the three continuous midpoints. The strong reference uses
the AlPhos/BTMG row of the paper's Table 1: 2.25 equivalents, 99.89 degrees Celsius,
1763.66 seconds. It is not a certified global optimizer of the frozen emulator.

Fresh evaluation on 2026-09-09:

| Program | combined_score / predicted yield | Validity |
|---|---:|---:|
| Initial | 0.6551045775413513 | 1 |
| Reference | 0.9770283699035645 | 1 |

The paper reports 1.09 for its historical prediction. The current frozen
ensemble produces an unclipped mean of about 1.02950 at those conditions, and
two of its five members are clipped. We report the actual current score, not
the historical value or an artificial normalization to make the reference 1.

`tests/evaluation/test_summit_baumgartner.py` passed 16 tests, including native
inference parity on 241 designs spanning all 12 catalyst/base pairs. That test
executes the pinned upstream ANN class with the released PyTorch tensors and
scikit-learn preprocessors; NumPy predictions agree within 2e-6. It checks the
important memberwise clipping behavior, data hashes and invalid artifacts.
The legacy Summit package's training/optimization API is not installed or tested.
SkyDiscover's actual evaluator API and oracle loader are checked by
`scripts/verify_ai4s_optimization.py`.

## Oracle evidence

[oracle/web_search.json](oracle/web_search.json) contains a manually curated,
actually fetched excerpt of Table 1. It is not presented as a Tavily search
response. The exact retained text, URL, retrieval method and hash are recorded
in [oracle/provenance.json](oracle/provenance.json). Only the short table header
and factual AlPhos row are retained, not the full article.

The document explicitly supplies the reference conditions. The task-level label
remains **inferential** because their global optimality for this pinned current
emulator is not established. The replay summary describes only what the table
states; current-model measurements and this interpretation belong in this README.
No LLM evolution, unaided-recall probe or oracle-versus-control experiment was run.
This is a registered published benchmark adapter, not a validated positive control.

## Run

From the repository root, with NumPy installed (see `requirements.txt`):

```bash
.venv/bin/python benchmarks/summit_baumgartner/evaluator.py benchmarks/summit_baumgartner/initial_program.py
.venv/bin/python benchmarks/summit_baumgartner/evaluator.py benchmarks/summit_baumgartner/oracle/best_program.py
.venv/bin/skydiscover-run benchmarks/summit_baumgartner/initial_program.py benchmarks/summit_baumgartner/evaluator.py -c benchmarks/summit_baumgartner/config.yaml
```
