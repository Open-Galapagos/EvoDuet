# SkyDiscover image for SimpleTES open_problems_bio/denoising.
#
# The image mirrors the SimpleTES repository root at /opt/simpletes exactly the
# way benchmarks/simpletes/ mirrors it on the host:
#
#   /opt/simpletes/sitecustomize.py                     (upstream, verbatim)
#   /opt/simpletes/simpletes/{construction,utils}       (upstream, verbatim)
#   /opt/simpletes/skydiscover_adapter.py               (SkyDiscover's EvaluatorWorker re-implementation)
#   /opt/simpletes/datasets/open_problems_bio/...       (this build context)
#
# so that PYTHONPATH=/opt/simpletes (set by the adapter for the evaluator
# subprocess, as SimpleTES's EvaluatorWorker sets PYTHONPATH=<repo root>) gives
# the evaluator and the candidate subprocess the same import environment upstream
# has, including the GLOBAL_BEST_CONSTRUCTION builtin that sitecustomize.py installs.
#
# Upstream's setup.sh runs unmodified: it builds <family>/.venv with the pinned
# scanpy stack from requirements.txt, installs molecular-cross-validation from
# GitHub (unpinned upstream; the split validated below was produced with commit
# 04d9df0e59a247200153b3ce38952a086169769f), patches the vendored OpenProblems
# v1.0.0 checkout (openproblems_api_fix.sh; the checkout is already patched, so the
# script's `git checkout -- .` fails and setup.sh tolerates that) and installs it.
# prepare_pancreas.py then materialises the pancreas train/test split the
# evaluator reads and, with --strict, fails the build unless the split reproduces
# the evaluator's BASELINES constants (see that file).  Layers are ordered so that
# editing the SkyDiscover glue or the generated config.yaml never re-runs the venv
# build or the figshare download.
FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv==0.9.2

ENV SIMPLETES_ROOT=/opt/simpletes \
    FAMILY_DIR=/opt/simpletes/datasets/open_problems_bio
WORKDIR $FAMILY_DIR

# 1. upstream environment (setup.sh needs exactly these inputs, beside itself)
COPY setup.sh requirements.txt openproblems_api_fix.sh $FAMILY_DIR/
COPY openproblems $FAMILY_DIR/openproblems
RUN bash $FAMILY_DIR/setup.sh && rm -rf /tmp/uv-cache-openproblems-bio

# 2. upstream repo-root files + the adapter (the copies the port keeps in this build context)
COPY simpletes $SIMPLETES_ROOT/simpletes
COPY sitecustomize.py skydiscover_adapter.py $SIMPLETES_ROOT/

# 3. the pancreas split (downloads from figshare) — validated against the evaluator's constants
COPY denoising/evaluator.py $FAMILY_DIR/denoising/evaluator.py
COPY prepare_pancreas.py $FAMILY_DIR/
RUN $FAMILY_DIR/.venv/bin/python $FAMILY_DIR/prepare_pancreas.py --strict

# 4. the rest of the task files + SkyDiscover container glue; SkyDiscover always invokes /benchmark/evaluate.sh
COPY denoising $FAMILY_DIR/denoising
COPY README.md data_manifest.json evaluate.sh run_eval.py wrapper.py $FAMILY_DIR/
COPY posthoc_final_eval.py final_evaluator.py $FAMILY_DIR/
RUN chmod 0755 $FAMILY_DIR/evaluate.sh \
    && mkdir -p /benchmark && ln -s $FAMILY_DIR/evaluate.sh /benchmark/evaluate.sh

# Environment knobs the upstream evaluator reads (it is host-dependent by design):
# the candidate runs under RLIMIT_AS = host_RAM * 0.9 / EVALUATOR_CONCURRENT_PROCESSES
# (upstream default 256, sized for SimpleTES's own process pool); SkyDiscover runs at
# most a handful of evaluations per container, and the released best program needs
# more than the ~2.7 GB that 256 leaves on a 754 GB host.  NUMBA_NUM_THREADS bounds
# numba's thread pool (default = every host CPU) to the 8 CPUs the task prompt
# promises.  Neither changes how a candidate is scored; see benchmarks/simpletes/README.md.
ENV EVALUATOR_CONCURRENT_PROCESSES=4 NUMBA_NUM_THREADS=8

ENTRYPOINT ["/benchmark/evaluate.sh"]
