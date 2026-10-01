# GuacaMol v2: C11H24 isomer generation

A fixed molecular-generation optimization task from **GuacaMol**, an existing
benchmark for AI methods in de novo molecular design. Generate diverse structures
with molecular formula C11H24, the formula of undecane. The original benchmark
asks for the 159 constitutional isomers, ignoring stereochemistry. This is a
chemical-space coverage task; its score is not a prediction of drug efficacy,
synthesizability, or chemical stability.

Source: Nathan Brown, Marco Fiscato, Marwin H. S. Segler, and Alain C. Vaucher,
“GuacaMol: Benchmarking Models for De Novo Molecular Design,” *Journal of Chemical
Information and Modeling* **59**(3), 1096–1108 (2019),
[public manuscript](https://arxiv.org/abs/1811.09621),
[DOI](https://doi.org/10.1021/acs.jcim.8b00839).
The task is `isomers_c11h24()` in the
[official definitions](https://github.com/BenevolentAI/guacamol/blob/master/guacamol/standard_benchmarks.py)
and appears in the
[v2 suite](https://github.com/BenevolentAI/guacamol/blob/master/guacamol/benchmark_suites.py).
The v2 geometric mean is retained; the v1 arithmetic-mean variant is different.

## Contract and original objective

`solve(payload)` returns a JSON-compatible **list of SMILES strings**. The payload
contains only the fixed public formula, atom-count targets, score parameters,
requested count, and adapter resource limits. It contains no reference structures.
The evaluator runs the submitted program in a fresh process and independently
recomputes the score from the returned structures.

The original evaluation procedure is:

1. Parse SMILES with RDKit and remove strings that cannot be parsed.
2. Canonicalize with `isomericSmiles=False` and remove duplicate canonical strings.
   This ignores stereochemistry and isotope labels. Score the canonical molecules,
   rather than their original annotated representations.
3. For each distinct molecule, count carbon atoms `C`, hydrogen atoms `H`, and all
   atoms `N`. Hydrogens include implicit hydrogens, using `Chem.AddHs`.
4. Compute the three Gaussian contributions and their geometric mean:

   ```text
   gC = exp(-0.5 * (C - 11)^2)
   gH = exp(-0.5 * (H - 24)^2)
   gN = exp(-0.5 * ((N - 35)/2)^2)
   molecule_score = (gC * gH * gN)^(1/3)
   ```

5. Sort scores in descending order and return `sum(top 159 scores)/159`.
   Missing structures therefore contribute zero. Additional structures beyond
   159 are accepted within the adapter limit, but only the best 159 count.

`combined_score` and `top_159` are both this **unmodified native scalar objective**,
already in `[0, 1]`. This is not a single-best-molecule score, exact-formula-only
hit count, or PMO sample-efficiency AUC. Formula mismatches receive partial credit.
The upstream objective does not impose extra connectivity, neutral-charge,
drug-likeness, stability, or synthesis constraints; neither does this adapter.

`validity` is an adapter diagnostic: 1 if at least one canonical molecule remains,
otherwise 0. Invalid SMILES within a list are filtered without discarding its
valid entries. Non-list outputs or non-string entries invalidate the whole
artifact. Empty lists and all-unparseable lists score zero. One upstream edge
case is deliberately preserved: RDKit parses the literal empty string as a
zero-atom molecule, and upstream canonicalization retains it; it receives the
corresponding extremely small Gaussian score rather than being silently assigned
a different objective. This diagnostic is not GuacaMol's separate
distribution-learning validity benchmark.

The adapter accepts at most 10,000 submitted strings of at most 4,096 characters
each, in addition to the shared runner's 8 MiB output limit. These are execution
limits, not new chemical constraints. The normal native return of 159 small
molecules is far below them. It omits upstream's informational pairwise-similarity
histogram and call-count bookkeeping, which do not enter this task's score.

## Programs and validation

Both supplied programs are **locally authored adapters**, not an upstream
initial/best pair. `initial_program.py` enumerates the straight chain and singly
methyl-branched chains, missing structures with richer branching.
`oracle/best_program.py` enumerates all nonisomorphic trees on 11 vertices using
NetworkX and keeps only those whose maximum degree is at most four. It converts
each vertex to carbon and each edge to a single bond, then produces canonical
SMILES with RDKit. It never reads source snapshots or oracle documents.

This reference is a specialization for neutral acyclic saturated hydrocarbons:
their carbon skeletons are trees; with 11 carbons and 10 single carbon-carbon
bonds, valence completion gives `4*11 - 2*10 = 24` hydrogens. Of the 235 tree
topologies on 11 vertices, exactly 159 satisfy the degree constraint. These
independently score 1, the maximum possible native score. The implementation is
not a port of MAYGEN's more general orderly graph-generation algorithm.

Fresh candidate subprocesses produced:

| Program | Submitted | Distinct structures | Exact formula | Native score |
|---|---:|---:|---:|---:|
| Initial | 9 | 5 | 5 | 0.031446540880503145 |
| Reference | 159 | 159 | 159 | 1.0 |

[validation.json](validation.json) records the environment, measurements, and
model-free checks. The task-specific regression tests cover duplicate and
stereochemical equivalence, isotope handling, formula-mismatch partial credit,
invalid SMILES, malformed artifacts, empty outputs, extra-element and disconnected
structures, top-159 aggregation, source hashes, configuration loading, and native
oracle replay. They also execute the fetched upstream scoring definitions against
the adapter, avoiding unrelated obsolete dependencies in the original package.
The source surrounding the two programs' EVOLVE blocks is identical.

There are no held-out molecules or hidden formulas. This preserves the original
fixed objective. No LLM evolution or no-document/document comparison was run;
the score gap alone does not establish that a model needs or benefits from the
documents.

## Oracle evidence and provenance

The bundled documents are **inferential** evidence, not a table of the 159 answers:

- [MAYGEN author documentation](https://github.com/MehmetAzizYirik/MAYGEN) describes
  generating all non-isomorphic constitutional isomers from a molecular formula
  and provides source, executables, formula input, and SMILES-output options.
  Its associated paper is Mehmet Aziz Yirik, Maria Sorokina, and Christoph
  Steinbeck, “MAYGEN: an open-source chemical structure generator for constitutional
  isomers based on the orderly generation principle,” *Journal of Cheminformatics*
  **13**, 48 (2021), [full paper](https://doi.org/10.1186/s13321-021-00529-9).
  The bundled raw document is the fetched author README, not reconstructed paper
  text. The README's large-formula timings are not timings for this task.
- [NetworkX 3.3's generator source and documentation](https://github.com/networkx/networkx/blob/networkx-3.3/networkx/generators/nonisomorphic_trees.py)
  supplies `nonisomorphic_trees(order)` and the implementation of the
  Wright–Richmond–Odlyzko–McKay free-tree enumeration algorithm.

Applying these methods to the C11H24 formula and choosing the carbon-degree
constraint requires task-specific reasoning. The replay summaries contain only
what the fetched documents explicitly state; they do not add this adaptation.
Neither document is classified as a direct answer for the fixed benchmark.

[oracle/web_search.json](oracle/web_search.json) uses the native replay schema.
Its `raw_content` fields preserve complete fetched UTF-8 source documents,
including original line endings; `raw_content_truncated` is false. `content`
contains faithful summaries, which are the text replayed by `web_document`.
[oracle/provenance.json](oracle/provenance.json) records retrieval timestamps,
URLs, hashes, licenses, and the evidence limits. Search-query strings and relevance
scores are curator labels; this collection used direct HTTP fetches, not a search
provider or model-generated oracle pass.

Upstream GuacaMol source snapshots are retained in [source/upstream](source/upstream)
under its MIT license, alongside author documentation under MIT and NetworkX
source under BSD-3-Clause in [oracle/sources](oracle/sources). All response bytes
are recorded in [source/provenance.json](source/provenance.json). Mutable upstream
URLs are recorded with content hashes so the fetched versions remain reviewable.

## Run

From the repository root:

```bash
uv pip install --python .venv/bin/python -r benchmarks/guacamol_c11h24/requirements.txt
.venv/bin/python benchmarks/guacamol_c11h24/evaluator.py benchmarks/guacamol_c11h24/initial_program.py
.venv/bin/python benchmarks/guacamol_c11h24/evaluator.py benchmarks/guacamol_c11h24/oracle/best_program.py
.venv/bin/python -m pytest tests/evaluation/test_guacamol_c11h24.py -q
.venv/bin/skydiscover-run benchmarks/guacamol_c11h24/initial_program.py benchmarks/guacamol_c11h24/evaluator.py -c benchmarks/guacamol_c11h24/config.yaml
```

The evaluator requires RDKit; the reference additionally uses NetworkX. Exact
versions used for validation are pinned in [requirements.txt](requirements.txt).
The shared [runner](../_public_optimization_runtime.py) permits 30 seconds, one
numerical-library thread, and 2 GiB of memory in a temporary working directory.
Only candidate source and public payload are copied there. This is process
isolation, not an operating-system security sandbox.
