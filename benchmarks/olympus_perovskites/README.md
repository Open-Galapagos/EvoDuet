# OLYMPUS perovskites: minimum HSE06 bandgap

Select a composition from the existing OLYMPUS `perovskites` benchmark to
minimize its calculated HSE06 bandgap. The native domain contains 16 organic
components, 3 metal cations (`Ge`, `Sn`, `Pb`), and 4 halide anions
(`F`, `Cl`, `Br`, `I`): all 192 compositions have a published value.
This is a materials discovery optimization task over a fixed candidate set.

## Published benchmark and retained data

- [OLYMPUS repository](https://github.com/the-matter-lab/olympus), including
  [the native perovskites task](https://github.com/the-matter-lab/olympus/tree/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_perovskites).
- Hase et al., *Olympus: a benchmarking framework for noisy optimization and
  experiment planning*, Machine Learning: Science and Technology 2, 035021
  (2021), [DOI: 10.1088/2632-2153/abedc8](https://doi.org/10.1088/2632-2153/abedc8).
- The repository also cites Hickman et al., *Olympus, enhanced: benchmarking
  mixed-parameter and multi-objective optimization in chemistry and materials
  science* (2023), DOI: 10.26434/chemrxiv-2023-74w8d.
- Hase et al., *Gryffin: An algorithm for Bayesian optimization of categorical
  variables informed by expert knowledge*, Applied Physics Reviews 8, 031406
  (2021), [paper](https://arxiv.org/abs/2003.12127).

`data/data.csv` and `data/config.json` are unchanged upstream response bytes,
pinned to OLYMPUS revision `440b6b58ebfcaa2391cff7e94b570fb4fda98d68`.
`data/provenance.json` records their exact URLs, fetch times, and SHA-256 hashes.
The upstream MIT license is retained in `data/OLYMPUS_LICENSE`.

## Program contract and evaluation

Implement `solve(payload)` and return exactly an object with three strings:
`organic`, `cation`, and `anion`. Each must be one of the options under
`payload.parameters`. The payload contains the allowed categories and objective
metadata only. It contains no measured bandgaps, objective table, optimum, or
data-file paths. Category names remain in the native spelling and order.

The evaluator independently checks the three choices and retrieves their
bandgap from the pinned table. This preserves the native
`Dataset(kind="perovskites").run(..., noiseless=True)` objective: the stored
HSE06 value, with no additional measurement noise. No OLYMPUS installation,
surrogate training, or new DFT calculation is needed. Candidate-reported scores
and bandgaps are not accepted; an extra field invalidates the whole artifact.
Unknown categories, wrong schemas/types, exceptions, and timeouts receive
`validity=0` and `combined_score=0`.

For valid outputs:

```text
bandgap_ev = native table value for the returned composition
combined_score = min(native table bandgaps) / bandgap_ev
bandgap_gap_ev = bandgap_ev - min(native table bandgaps)
```

The table minimum is `1.5249 eV`; the bounded score increases as bandgap
decreases, reaching 1 at that minimum. The raw physical objective is always
reported. The evaluator checks asset hashes, positive finite bandgaps, unique
compositions, and complete coverage of the native 16 x 3 x 4 domain.

**Adapter changes:** this port introduces the program interface, bounded scalar
score, and authored initial/reference programs. It evaluates one returned
composition per program, rather than reproducing an entire OLYMPUS planner
campaign or its number-of-experiments metric. Optional physicochemical
descriptors are not supplied. The original candidate set and noiseless
bandgaps are unchanged. There are no hidden instances or claimed train/test
generalization. Unrestricted access to all 192 objective values would make
exhaustive search trivial; those values stay in the evaluator's task data.

Candidates run in the shared resource-limited subprocess with a 30-second
timeout, one numerical-library thread, and 2 GiB memory. The runtime copies only
candidate source and public JSON input to a temporary directory, and does not
forward API credentials or the repository PYTHONPATH. This is process isolation,
not an OS security sandbox against hostile Python code. It does not prevent a
hostile process from accessing known absolute filesystem paths. The task
contract prohibits reading evaluator data, reference, and oracle files.

## Oracle document

The [official Gryffin perovskites tutorial](https://gryffin.readthedocs.io/en/latest/tutorials/perovskites.html)
explicitly identifies `hydrazinium`, `I`, and `Sn` as the minimum-bandgap choices,
with value `1.5249 eV`. Its
[original notebook](https://raw.githubusercontent.com/aspuru-guzik-group/gryffin/fb149d18f9c81b96179cc682c46e05db60bc7229/docs/source/tutorials/perovskites.ipynb)
was fetched at pinned revision `fb149d18f9c81b96179cc682c46e05db60bc7229`
and retained unchanged as `oracle/perovskites.ipynb`, with its Apache 2.0 license.
`oracle/gryffin_optimum_excerpt.txt` preserves the complete source of notebook
cell 5. `oracle/provenance.json` records the fetch and exact extraction.

This is **direct evidence for the fixed noiseless dataset**. The curator summary
in `oracle/web_search.json` preserves the exact answer because `web_document`
replay injects `content`, rather than falling back to `raw_content`.
The excerpt is marked truncated relative to the complete notebook. No fetched
text was reconstructed from search snippets or model memory. Evidence scores
are curator labels, not measured usefulness.

## Local validation and execution

Measured through the real candidate subprocess with Python 3.12.3:

| Program | Composition (organic / cation / anion) | Bandgap, eV, lower is better | Combined score | Validity |
|---|---|---:|---:|---:|
| Initial: first allowed options | ethylammonium / Ge / F | 5.3704 | 0.2839453299568002 | 1 |
| Reference: documented optimum | hydrazinium / Sn / I | 1.5249 | 1.0 | 1 |

The minimum is unique in the 192-row native table. `validation.json` records
local metrics and file hashes. Focused tests cover both real programs, complete
domain coverage, checksum tampering, missing/extra keys, wrong types, booleans,
unknown/swapped categories, forged scalar scores, and actual OracleRetrieval
replay with network connections blocked.

The seed/reference gap is verified. The causal benefit of providing the oracle
document to an LLM has not been measured: no model probes or no-document control
arm were run. This remains an unproven positive-control candidate.
