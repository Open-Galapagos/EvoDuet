# SRSD-Feynman Easy I.27.6: geometrical optics

This ports the existing **SRSD-Feynman Easy** benchmark task `feynman-i.27.6`; it does not generate a new equation or replace the public observations.

- Domain: Physics / geometrical optics / symbolic regression.
- Source mode: published benchmark.
- Year: 2022 dataset/preprint; 2024 DMLR publication.
- Difficulty: **Easy** in the source benchmark. Three inputs, one short equation, and subsecond local evaluation make this a small implementation target. This is not a measured difficulty comparison with an LLM.
- Oracle evidence: **direct / Type A**, because the fetched textbook equations state the target physical relationship.

## Task and provenance

Recover focal distance from object distance `x0=d1`, refractive index `x1=n`, and image distance `x2=d2`. The published equation is `f=1/(1/d1+n/d2)`. Discovering reciprocal combinations rather than fitting an arbitrary interpolator is the scientific target.

The official sampling ranges are positive distances from 10^-3 to 10^-1 metres and positive refractive index from 10^-1 to 10^1. The task preserves those benchmark ranges, including refractive-index values that are not a typical ordinary-glass experiment.

The benchmark is Matsubara, Chiba, Igarashi and Ushiku, [*Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery*](https://data.mlr.press/assets/pdf/v01-3.pdf), DMLR 2024. The [official code](https://github.com/omron-sinicx/srsd-benchmark) and [author-published Easy dataset](https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy) provide the task, splits and native evaluation method.

[Data provenance](data_provenance.json) records the downloaded files' URLs, SHA-256 hashes and retrieval metadata. Numeric data are pinned to dataset revision `78a2026e61eaa000fb7ed5e30f0639329e912f9d`: **8,000 training, 1,000 validation and 1,000 test rows**, unchanged byte for byte, with filenames shortened to `train.txt`, `val.txt`, and `test.txt`. The final column is the target. No sampling or split regeneration is performed.

The formula is transcribed from the official `FeynmanICh27Eq6` class at code revision `7d00b45d56250717ac08a256dc9f79b836d61027`, retained in [source/equation_source.py](source/equation_source.py). The evaluator does not deserialize upstream pickle files. It verifies data checksums when loading a split.

The data are **CC BY 4.0**; attribution and the original card are retained in [source/DATASET_CARD.md](source/DATASET_CARD.md). The official metric and equation source snapshots are **MIT** licensed; retain [source/UPSTREAM_LICENSE.txt](source/UPSTREAM_LICENSE.txt) when redistributing them. Our seed, reference and adapter implementation are locally authored.

## Program contract and scoring

Implement `solve(payload)` and return exactly:

```python
{"expression": "<one arithmetic expression using x0, x1, x2>"}
```

The candidate receives only training observations in `x_train` and `y_train`, plus variable names and descriptions. The restricted expression language accepts `+`, `-`, `*`, `/`, numeric powers with exponent in [-8,8], `sin`, `cos`, `exp`, `log`, `sqrt`, and `pi`. Limits are 2,000 characters, 80 AST nodes, depth 16, finite numeric literals of magnitude at most 1e100, and a 200-node canonical tree. Python attributes, imports and indexing are not expression syntax.

Pure constant models are excluded, following the upstream model selector. Malformed output, singular/nonfinite predictions and any complex-valued predictions, including complex arrays with zero imaginary components, are invalid. Validation and final results always expose finite `combined_score` and `validity`; failures score zero.

[Source model selection](source/model_selector.py) minimizes validation **mean squared relative error**:

```text
MSRE = mean(((prediction - target) / target)**2)
combined_score = 1 / (1 + MSRE)
```

The scalar `combined_score` is this adapter's monotone maximization transform, not a new SRSD metric. There is no absolute-error epsilon floor. All shipped targets are nonzero; a zero target would make this native relative metric undefined. Zero predictions would have MSRE=1, regardless of the target's physical scale; pure constant candidates themselves are rejected.

`evaluate()` scores only the official validation split and does not return ground-truth structural feedback. With `final_evaluation: true`, `evaluate_final()` reports test MSRE and **normalized equation-tree edit distance (NED)**. The trusted evaluator canonicalizes equations using the upstream SymPy sequence, turns numbers into `Const` nodes, and computes exact ordered tree edit distance divided by the ground-truth tree size, capped at 1. NED is a separate final diagnostic, not a term in the search score.

The local Zhang–Shasha implementation fixes insertion, deletion and unequal-label replacement costs to one. This matches `zss.simple_distance`'s fallback when optional string-edit packages are absent; upstream does not pin these optional packages, so environments that install them can change label costs. All 256 randomly generated tree pairs matched an independently fetched `zss==1.2.0` reference under explicit unit costs. Thirty-two cases are retained with this task as offline regression fixtures. Canonicalization follows upstream, so SymPy version can affect structural representation; the measured version is recorded in [validation.json](validation.json). NED ignores the magnitude of an existing numeric constant but can detect the insertion of a new constant node. It therefore cannot replace numerical prediction error.

The candidate and expression grader run in separate temporary, resource-limited processes: 30 seconds and 2 GiB each, with an outer 90-second evaluator limit. Candidate text is parsed through a whitelist, not passed to Python `eval`. The grader never reads `oracle/`. Public test files and reference equations are present in the repository for reproducibility but are withheld from the candidate payload. These process limits and import isolation are **not an OS security sandbox** against hostile Python accessing arbitrary absolute paths.

Final evaluation reruns the selected program on the same training data; it does not persist its earlier fitted expression. Both shipped programs are deterministic. A randomized candidate should manage its own seed to make the selected model reproducible.

## Initial program, reference and oracle

The locally authored initial program fits `a*x0` using training relative squared error. It cannot express the coupled dependence on refractive index and image distance.

The locally authored reference uses the source's reciprocal-distance form, fits the coefficients of `1/x0` and `x1/x2` to `1/y` by training-only least squares, and snaps coefficients within 1e-10 of integers. Both coefficients are recovered as 1. This is executable reference code derived from the initial scaffold, not a claimed upstream leaderboard submission.

The oracle is [the original Feynman Lectures chapter](https://www.feynmanlectures.caltech.edu/I_27.html#Eq:I:27:6). The fetched equation (27.6), `1/s+n/s'=1/f`, directly states the target physical relation. The documented variable correspondence is `s=d1`, `s'=d2`, and the returned expression solves it for `f`. The oracle summary only describes the source equation; this mapping belongs to the task adapter. The [retained excerpt](oracle/source_excerpt.txt) contains only compact, fetched mathematical evidence; it is not a copied chapter.

[oracle/web_search.json](oracle/web_search.json) is the replay format and [oracle/evidence.json](oracle/evidence.json) preserves the user's direct/inferential/insufficient schema. The exact fetched excerpt is retained as `raw_content`, with `raw_content_truncated: true`. [Oracle provenance](oracle/provenance.json) records that the session web tool successfully retrieved the page while direct HTTP requests returned 403; it does not invent raw HTTP response hashes. The textbook's copyright is separate from the CC-licensed benchmark data.

This verifies that the locally authored reference reaches the public benchmark target. **No LLM search, no model API call and no with/without-document experiment were run.** Relevance and helpfulness numbers in oracle JSON are curator labels, not measured model benefit. The equation is public and may already be familiar to a model.

## Measured local validation

Measured with the versions, source hashes and full-precision outputs in [validation.json](validation.json):

| Program | Validation MSRE | Validation combined score | Test MSRE | Test NED |
|---|---:|---:|---:|---:|
| Initial | 0.9051951435222695 | 0.5248806157206705 | 0.9175254705904884 | 0.8 |
| Reference | 0 | 1 | 0 | 0 |

Both programs are valid. Measured end-to-end evaluations took about 0.3–0.4 seconds on this workspace's CPU. The reference's test score is 1.0. The 9 task regression tests cover official split/hash integrity, train-only payloads, restricted syntax, malformed/forged output, constant and singular models, complex projection rejection, equivalent formulas, scale-sensitive relative loss, independent scoring and matching EVOLVE scaffolds.

Run from the repository root:

```bash
.venv/bin/python benchmarks/srsd_feynman_i_27_6/evaluator.py benchmarks/srsd_feynman_i_27_6/initial_program.py
.venv/bin/python benchmarks/srsd_feynman_i_27_6/evaluator.py benchmarks/srsd_feynman_i_27_6/oracle/best_program.py --test
.venv/bin/python -m unittest discover -s benchmarks/srsd_feynman_i_27_6 -p 'test_srsd_*.py'
uv run --no-sync skydiscover-run benchmarks/srsd_feynman_i_27_6/initial_program.py benchmarks/srsd_feynman_i_27_6/evaluator.py --config benchmarks/srsd_feynman_i_27_6/config.yaml
```
