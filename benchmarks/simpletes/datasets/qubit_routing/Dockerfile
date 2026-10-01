# SkyDiscover image for SimpleTES qubit_routing/swap_reduction.
#
# The image mirrors the SimpleTES repository root at /opt/simpletes the way
# benchmarks/simpletes/ mirrors it on the host:
#
#   /opt/simpletes/sitecustomize.py                      (upstream, verbatim)
#   /opt/simpletes/simpletes/{construction,utils}        (upstream, verbatim)
#   /opt/simpletes/skydiscover_adapter.py                (SkyDiscover's EvaluatorWorker re-implementation)
#   /opt/simpletes/datasets/qubit_routing/...            (this build context)
#
# evaluator.py computes REPO_ROOT = ROOT.parents[2] at import time (ROOT =
# <family>/swap_reduction), so the datasets/ level is what makes it resolve.
# The evaluator compiles the vendored Rust router (router_core links Qiskit's
# Rust crates through pyo3 with auto-initialize) with cargo and embeds the Python
# it finds via PYO3_PYTHON / sys.executable, which must carry qiskit==1.2.0 and
# qiskit-qasm3-import (requirements.txt).  python:3.12-bookworm ships a shared
# libpython with headers, so one interpreter serves both the evaluator and the
# embedded runtime.
FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    # Pin the interpreter pyo3 builds against to the one evaluate.sh runs
    # (evaluator.py prefers PYO3_PYTHON, then sys.executable).
    PYO3_PYTHON=/usr/local/bin/python3

# rust-version = "1.85" in rust/Cargo.toml; pin the toolchain to that release.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
       | sh -s -- -y --profile minimal --default-toolchain 1.85.1 --no-modify-path

ENV SIMPLETES_ROOT=/opt/simpletes \
    FAMILY_DIR=/opt/simpletes/datasets/qubit_routing
WORKDIR $FAMILY_DIR
COPY requirements.txt $FAMILY_DIR/requirements.txt
RUN pip install -r $FAMILY_DIR/requirements.txt

# Repo-root files (the copies the port keeps in this build context).
COPY simpletes $SIMPLETES_ROOT/simpletes
COPY sitecustomize.py skydiscover_adapter.py $SIMPLETES_ROOT/

COPY . $FAMILY_DIR
# SkyDiscover's ContainerizedEvaluator always invokes /benchmark/evaluate.sh.
RUN chmod 0755 $FAMILY_DIR/evaluate.sh \
    && mkdir -p /benchmark && ln -s $FAMILY_DIR/evaluate.sh /benchmark/evaluate.sh

# Upstream derives its result dir from REPO_ROOT/checkpoints (only used with
# --save-results); keep it under /tmp inside the image.  Slot workspaces stay
# at the upstream default (/tmp/simpletes-qubit-routing-slots).
ENV QUBIT_ROUTING_RESULTS_DIR=/tmp/qubit_routing_results

# Warm the canonical cargo target dir (evaluator.py TARGET_DIR = rust/.cargo_target);
# prepare_slot_workspace() copies it into each slot so the first evaluation only
# rebuilds router_core/router_cli instead of the whole Qiskit dependency tree.
RUN cd $FAMILY_DIR/swap_reduction/rust \
    && cargo build --release --locked -p router_cli --target-dir .cargo_target \
    && du -sh .cargo_target

ENTRYPOINT ["/benchmark/evaluate.sh"]
