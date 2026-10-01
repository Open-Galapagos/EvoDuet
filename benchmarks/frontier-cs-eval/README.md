# Frontier-CS Benchmark

Evolves C++ solutions for [Frontier-CS](https://github.com/facebookresearch/Frontier-CS) algorithmic optimization problems using SkyDiscover.

## Setup

```bash
# 1. Clone Frontier-CS
cd benchmarks/frontier-cs-eval
git clone https://github.com/FrontierCS/Frontier-CS.git

# 1b. Apply local statement fixes (Frontier-CS/ is git-ignored, so redo this after every fresh clone)
git -C Frontier-CS apply ../patches/263-statement-input-format.patch

# 2. Start the judge server (requires Docker)
cd Frontier-CS/algorithmic
docker compose up -d

# 3. Install dependencies (from project root)
cd ../../..
uv sync --extra frontier-cs

# 4. Set your API key
export OPENAI_API_KEY=...
```

## Run

Supported algorithms: `adaevolve`, `evox`, `openevolve`, `gepa`, `shinkaevolve`


Single problem:
```bash
cd benchmarks/frontier-cs-eval
FRONTIER_CS_PROBLEM=0 uv run skydiscover-run initial_program.cpp evaluator.py \
  -c config.yaml -s [search_algorithm] -i 50
```

All problems in parallel:
```bash
uv run python run_all_frontiercs.py --search [search_algorithm] --iterations 50 --workers 6
```

## Evaluate best programs (post-discovery)

```bash
uv run python run_best_programs_frontiercs.py
```

## Analyze results

```bash
uv run python combine_results.py   # merge training/testing scores into CSV
uv run python analyze_results.py   # generate plots and statistics
```

## Files

| File | Description |
|------|-------------|
| `initial_program.cpp` | Seed C++ program |
| `evaluator.py` | Evaluates C++ solutions via Frontier-CS docker judge |
| `config.yaml` | Config with system prompt template |
| `run_all_frontiercs.py` | Parallelizes evolution across all problems |
| `run_best_programs_frontiercs.py` | Re-evaluates best programs after evolution |
| `combine_results.py` | Combines training/testing scores into CSV |
| `analyze_results.py` | Generates score analysis plots and statistics |
| `patches/` | Local fixes to upstream Frontier-CS data (see [Known data issues](#known-data-issues-and-local-patches)) |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | API key |
| `FRONTIER_CS_PROBLEM` | `0` | Problem ID to evolve |
| `JUDGE_URLS` | `http://localhost:8081` | Comma-separated judge server URLs |

## Known data issues and local patches

### Problem 263: statement/testdata mismatch (`L_ref`)

**Symptom.** Runs on problem 263 evolve programs that fail with `runtime_error` on all 10 test cases, including the tiny ones (N=10). In `outputs/frontiercs_263/openevolve_native_openrouter_tavily_default/.../qwen3.5-27b-1.0/...` (2026-09-02, 100 iterations), 106 of 110 evolved programs read an `L_ref` value from stdin; 94 of them segfault on every case and the remaining 12 "score" only because they clamp out-of-range indices after reading shifted input. The reported best score (0.58 on the 0–100 scale) is an artifact, not a solution.

**Root cause.** The upstream statement (`Frontier-CS/algorithmic/problems/263/statement.txt`, added in upstream commit c20fd862, still unchanged as of origin/main 15ce5cc2) says "`L_ref` is provided in the input" and its sample input has a second line `120.0`. The bundled `testdata/*.in` files contain no such line and `chk.cc` never reads one: running the checker on the statement's own sample input fails with `Expected integer, but "120.0" found`. Any program that follows the statement consumes the first constraint's row index as `L_ref` and misparses everything after it. The statement's scoring formula (`(L_base - L_sub)/(L_base - L_ref)` scaled by 10^6) and output format (`a_0 ... a_N`) also disagree with the checker, which computes `clamp((L_base - L_sub)/L_base, 0, 1)` and reads exactly N values.

**Fix (local).** `patches/263-statement-input-format.patch` edits only the statement: it removes the `120.0` sample line and the `L_ref` note, states explicitly that the input contains nothing beyond the `N M D` line and the M constraint lines, rewrites the scoring section to match `chk.cc`, changes `a_0 ... a_N` to `a_1 ... a_N`, and adds the sample's actual `L_base` (12.2738) and score (0.538). Test data, checker, and evaluators are untouched. Because `Frontier-CS/` is git-ignored in this repository, the patch has to be re-applied after every fresh clone (Setup step 1b). The docker judge reads `statement.txt` from the mounted `problems/` directory on each request, so `/problem/263/statement` returns the patched text as well.

**Impact on existing results.** Every `outputs/frontiercs_263/*` run produced before 2026-09-02 used the broken statement and is not comparable with runs made after the patch. Under the old statement the score mostly measured whether a model distrusted the statement and parsed input defensively: gpt-5.6 solutions detect an "optional one-token L_ref line" and reach 95, while Qwen3.5 solutions that read `L_ref` before the constraints crash or stay at or below 42. Re-run problem 263 for every condition before reporting it.

**Upstream status.** Not fixed upstream and no issue filed as of 2026-09-02.
