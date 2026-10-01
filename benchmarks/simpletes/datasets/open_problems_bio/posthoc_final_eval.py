#!/usr/bin/env python3
"""Paper-style post-hoc evaluation for SimpleTES denoising outputs.

Run this script with the Python environment from the open_problems_bio image.
It evaluates every canonical ``best/best_program.py`` below an outputs root on
the held-out PBMC and Tabula Muris Senis Lung datasets.  Dataset splits and
candidate calls follow the public SimpleTES evaluator's seed-42 convention.

The expensive evaluation first writes an aggregate JSON file.  ``--apply-only``
then appends the verified results to each adjacent ``best_program_info.json``;
the original JSON is backed up once before that mutation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any


SCHEMA_VERSION = 1
PROTOCOL_NAME = "simpletes-denoising-paper-posthoc-v2"
DEFAULT_SPLIT_SEED = 42
DEFAULT_RANDOM_STATE = 42
DEFAULT_CANDIDATE_TIMEOUT = 7200
REFERENCE_SEARCH_TIMEOUT = 400
BASELINE_TOLERANCE = 5e-7
POISSON_THRESHOLD = 0.97
POISSON_SANITY_TOLERANCE = 0.0
MIN_FREE_CACHE_BYTES = 16 * 1024**3

PUBLISHED_BASELINES = {
    "pbmc": {
        "baseline_mse": 0.270945,
        "baseline_poisson": 0.300447,
        "perfect_mse": 0.0,
        "perfect_poisson": 0.043569,
    },
    "tabula": {
        "baseline_mse": 0.261763,
        "baseline_poisson": 0.206542,
        "perfect_mse": 0.0,
        "perfect_poisson": 0.026961,
    },
}

RAW_SOURCE_ARTIFACTS = {
    "pbmc": {
        "cache_file": (
            "openproblems_02ad74d5ee9e6e5310c88a3a95e45a524f2a4a7a26a9c73b914584d61d3cb7ec.h5ad"
        ),
        "bytes": 40065933,
        "sha256": "c7c9ce59681be0906087d7ec02a9ebd9b5ec2225a28f7146303396458c6b78f8",
    },
    "tabula": {
        "cache_file": (
            "openproblems_b8b3b8e7299dbca6b07e921684c551520db6f49f80828a5776aa29aaf5825de2.h5ad"
        ),
        "bytes": 611348452,
        "sha256": "f5523cc4c5aeff9cd92af9e40d594ef0904cd2d265de0a3121acf146e26ccf0a",
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "python",
        "numpy",
        "scipy",
        "scanpy",
        "anndata",
        "scprep",
        "graphtools",
        "magic-impute",
        "molecular-cross-validation",
        "openproblems",
        "scikit-learn",
    ):
        if name == "python":
            versions[name] = sys.version.split()[0]
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def package_provenance() -> dict[str, Any]:
    """Record immutable VCS provenance when installed metadata provides it."""
    provenance: dict[str, Any] = {}
    for name in ("molecular-cross-validation", "openproblems"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in distribution.files or ():
            if item.name != "direct_url.json":
                continue
            path = Path(distribution.locate_file(item))
            provenance[name] = json.loads(path.read_text(encoding="utf-8"))
            break
    return provenance


def container_memory_limit_bytes() -> int | None:
    """Return the cgroup v2 memory cap when this process has one."""
    path = Path("/sys/fs/cgroup/memory.max")
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def verify_raw_source_artifact(name: str) -> dict[str, Any] | None:
    """Verify the exact cached source artifact, if it is currently available."""
    cache_dir = os.environ.get("OPENPROBLEMS_CACHE_DIR")
    if not cache_dir:
        return None
    expected = RAW_SOURCE_ARTIFACTS[name]
    path = Path(cache_dir) / expected["cache_file"]
    if not path.is_file():
        return None
    actual = {
        "cache_file": expected["cache_file"],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "verified": True,
    }
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise RuntimeError(
            f"raw {name} source artifact differs from the pinned input: {actual}"
        )
    return actual


def install_mcv_import_shim() -> None:
    """Expose molecular_cross_validation.util without importing torch."""
    if "molecular_cross_validation" in sys.modules:
        return
    import types

    spec = importlib.util.find_spec("molecular_cross_validation")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("molecular-cross-validation is not installed")
    package = types.ModuleType("molecular_cross_validation")
    package.__path__ = list(spec.submodule_search_locations)
    sys.modules["molecular_cross_validation"] = package
    import molecular_cross_validation.util  # noqa: F401


def install_mcv_metric_shim() -> None:
    """Install the exact torch-free loss helper used by the public evaluator."""
    import types
    import numpy as np

    package = types.ModuleType("molecular_cross_validation")
    sweep = types.ModuleType("molecular_cross_validation.mcv_sweep")

    def poisson_nll_loss(y_pred, y_true):
        return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()

    sweep.poisson_nll_loss = poisson_nll_loss
    package.mcv_sweep = sweep
    sys.modules.setdefault("molecular_cross_validation", package)
    sys.modules.setdefault("molecular_cross_validation.mcv_sweep", sweep)


def dataset_paths(cache_root: Path, name: str, seed: int) -> dict[str, Path]:
    directory = cache_root / "prepared" / f"{name}_seed{seed}"
    return {
        "directory": directory,
        "train": directory / "train.npy",
        "test": directory / "test.npy",
        "metadata": directory / "metadata.json",
    }


def _load_raw_dataset(name: str):
    if name == "pbmc":
        from openproblems.data.tenx import load_tenx_1k_pbmc

        return load_tenx_1k_pbmc(test=False)
    if name == "tabula":
        from openproblems.data.tabula_muris_senis import load_tabula_muris_senis

        return load_tabula_muris_senis(
            organ_list=["lung"],
            method_list=["droplet"],
            test=False,
        )
    raise ValueError(f"unknown dataset: {name}")


def prepare_dataset(cache_root: Path, name: str, seed: int) -> dict[str, Any]:
    paths = dataset_paths(cache_root, name, seed)
    raw_source = verify_raw_source_artifact(name)
    if all(paths[key].is_file() for key in ("train", "test", "metadata")):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        if (
            metadata.get("split_seed") == seed
            and metadata.get("train_sha256") == sha256_file(paths["train"])
            and metadata.get("test_sha256") == sha256_file(paths["test"])
        ):
            if raw_source is not None and metadata.get("raw_source") != raw_source:
                metadata["raw_source"] = raw_source
                atomic_write_json(paths["metadata"], metadata)
            print(f"[data] reusing {name} seed-{seed} split: {paths['directory']}", flush=True)
            return metadata
        raise RuntimeError(f"cached {name} split failed its checksum validation")

    paths["directory"].mkdir(parents=True, exist_ok=True)
    print(f"[data] loading {name} raw dataset", flush=True)

    import openproblems.data

    openproblems.data.no_cleanup()
    install_mcv_import_shim()
    from openproblems.tasks.denoising.datasets.utils import split_data

    adata = split_data(_load_raw_dataset(name), train_frac=0.9, seed=seed)
    raw_source = verify_raw_source_artifact(name)
    if raw_source is None:
        raise RuntimeError(f"raw {name} source artifact was not cached after loading")
    shape = [int(adata.shape[0]), int(adata.shape[1])]
    raw_from_cache = bool(adata.uns.get("_from_cache", False))

    for key in ("train", "test"):
        destination = paths[key]
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        dense = adata.obsm[key].toarray().astype("float64", copy=False)
        with temporary.open("wb") as handle:
            import numpy as np

            np.save(handle, dense, allow_pickle=False)
        os.replace(temporary, destination)
        del dense
        gc.collect()

    del adata
    gc.collect()

    import numpy as np

    train = np.load(paths["train"], mmap_mode="r", allow_pickle=False)
    test = np.load(paths["test"], mmap_mode="r", allow_pickle=False)
    metadata = {
        "dataset": name,
        "split_seed": seed,
        "train_fraction": 0.9,
        "shape": shape,
        "dtype": str(train.dtype),
        "train_sum": float(train.sum()),
        "test_sum": float(test.sum()),
        "train_sha256": sha256_file(paths["train"]),
        "test_sha256": sha256_file(paths["test"]),
        "train_bytes": paths["train"].stat().st_size,
        "test_bytes": paths["test"].stat().st_size,
        "raw_loader_cache_hit": raw_from_cache,
        "raw_source": raw_source,
        "prepared_at": utc_now(),
    }
    atomic_write_json(paths["metadata"], metadata)
    print(f"[data] prepared {name}: shape={tuple(shape)}", flush=True)
    return metadata


def compute_mse(test_path: Path, denoised_path: Path) -> float:
    """Mirror OpenProblems v1.0.0 log-normalized MSE exactly."""
    import anndata
    import numpy as np
    import scanpy as sc
    import sklearn.metrics

    test_array = np.array(np.load(test_path, mmap_mode="r", allow_pickle=False), copy=True)
    denoised_array = np.array(
        np.load(denoised_path, mmap_mode="r", allow_pickle=False), copy=True
    )
    test_data = anndata.AnnData(X=test_array)
    denoised_data = anndata.AnnData(X=denoised_array)
    sc.pp.normalize_total(test_data, target_sum=10000)
    sc.pp.log1p(test_data)
    sc.pp.normalize_total(denoised_data, target_sum=10000)
    sc.pp.log1p(denoised_data)
    value = float(sklearn.metrics.mean_squared_error(test_data.X, denoised_data.X))
    del test_data, denoised_data, test_array, denoised_array
    gc.collect()
    return value


def compute_poisson(
    test_path: Path,
    denoised_path: Path,
    train_sum: float,
    test_sum: float,
) -> float:
    """Mirror OpenProblems' historical argument order and loss implementation."""
    import numpy as np

    test_array = np.array(np.load(test_path, mmap_mode="r", allow_pickle=False), copy=True)
    denoised_array = np.array(
        np.load(denoised_path, mmap_mode="r", allow_pickle=False), copy=True
    )
    # Prepared train/test arrays are float64, so their sums in the original
    # _compute_metrics are NumPy float64 scalars. Preserve both that promotion
    # and its multiply-then-divide order. In-place multiplication rejects integer
    # outputs and silently rounds float32 outputs before computing the loss.
    denoised_array = denoised_array * np.float64(test_sum) / np.float64(train_sum)
    value = float(
        (test_array - denoised_array * np.log(test_array + 1e-6)).mean()
    )
    del test_array, denoised_array
    gc.collect()
    return value


def compute_raw_metrics(
    train_path: Path,
    test_path: Path,
    denoised_path: Path,
    metadata: dict[str, Any],
) -> dict[str, float]:
    del train_path  # train_sum is already recorded without retaining another matrix.
    return {
        "mse_raw": compute_mse(test_path, denoised_path),
        "poisson_raw": compute_poisson(
            test_path,
            denoised_path,
            float(metadata["train_sum"]),
            float(metadata["test_sum"]),
        ),
    }


def compute_live_baselines(
    paths: dict[str, Path], metadata: dict[str, Any], name: str
) -> dict[str, Any]:
    print(f"[metric] computing {name} live baselines", flush=True)
    baseline = compute_raw_metrics(paths["train"], paths["test"], paths["train"], metadata)
    perfect_poisson = compute_poisson(
        paths["test"],
        paths["test"],
        float(metadata["train_sum"]),
        float(metadata["test_sum"]),
    )
    live = {
        "baseline_mse": baseline["mse_raw"],
        "baseline_poisson": baseline["poisson_raw"],
        "perfect_mse": 0.0,
        "perfect_poisson": perfect_poisson,
    }
    published = PUBLISHED_BASELINES[name]
    differences = {key: live[key] - published[key] for key in published}
    matches = all(abs(value) <= BASELINE_TOLERANCE for value in differences.values())
    print(
        f"[metric] {name} baseline match={matches} "
        f"mse={live['baseline_mse']:.9f} poisson={live['baseline_poisson']:.9f} "
        f"perfect_poisson={live['perfect_poisson']:.9f}",
        flush=True,
    )
    if not matches:
        raise RuntimeError(
            f"{name} live baselines do not reproduce the published SimpleTES constants: "
            f"differences={differences}"
        )
    return {
        "live": live,
        "published": published,
        "difference_live_minus_published": differences,
        "matches_published_within": BASELINE_TOLERANCE,
    }


def normalize_metrics(raw: dict[str, float], baseline_info: dict[str, Any]) -> dict[str, Any]:
    published = baseline_info["published"]
    live = baseline_info["live"]

    mse_norm_unclipped = (published["baseline_mse"] - raw["mse_raw"]) / (
        published["baseline_mse"] - published["perfect_mse"]
    )
    mse_norm = min(1.0, max(0.0, mse_norm_unclipped))
    poisson_norm = (published["baseline_poisson"] - raw["poisson_raw"]) / (
        published["baseline_poisson"] - published["perfect_poisson"]
    )
    mean_score = (mse_norm + poisson_norm) / 2.0
    poisson_constraint_pass = bool(poisson_norm >= POISSON_THRESHOLD)
    # Candidate validation uses the original exact threshold. The tolerance for
    # verifying downloaded baseline data is not a tolerance for candidate scores.
    poisson_sanity_pass = bool(
        raw["poisson_raw"]
        >= published["perfect_poisson"]
    )
    protocol_valid = bool(poisson_constraint_pass and poisson_sanity_pass)

    live_mse_norm = (live["baseline_mse"] - raw["mse_raw"]) / (
        live["baseline_mse"] - live["perfect_mse"]
    )
    live_poisson_norm = (live["baseline_poisson"] - raw["poisson_raw"]) / (
        live["baseline_poisson"] - live["perfect_poisson"]
    )

    return {
        **raw,
        "mse_norm_unclipped": float(mse_norm_unclipped),
        "mse_norm": float(mse_norm),
        "poisson_norm": float(poisson_norm),
        "mean_score": float(mean_score),
        "poisson_pass": poisson_constraint_pass,
        "poisson_constraint_pass": poisson_constraint_pass,
        "poisson_sanity_pass": poisson_sanity_pass,
        "protocol_valid": protocol_valid,
        "reported_score": float(mean_score if protocol_valid else 0.0),
        "live_baseline_mse_norm": float(live_mse_norm),
        "live_baseline_poisson_norm": float(live_poisson_norm),
        "live_baseline_mean_score": float((live_mse_norm + live_poisson_norm) / 2.0),
    }


def candidate_worker(candidate: Path, train_path: Path, output_path: Path, seed: int) -> int:
    """Worker process: it can see X_train but never X_test."""
    import numpy as np

    install_mcv_metric_shim()
    sys.path.insert(0, str(candidate.parent))
    spec = importlib.util.spec_from_file_location("posthoc_candidate", candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load candidate: {candidate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "magic_denoise", None)
    if not callable(function):
        raise RuntimeError("candidate does not define callable magic_denoise")

    x_train = np.load(train_path, allow_pickle=False)
    denoised = np.asarray(function(x_train, random_state=seed))
    if denoised.shape != x_train.shape:
        raise ValueError(f"shape mismatch: output={denoised.shape}, input={x_train.shape}")
    if not np.isfinite(denoised).all():
        raise ValueError("denoised output contains non-finite values")
    if np.any(denoised < 0):
        raise ValueError("denoised output contains negative values")
    if denoised.size and float(denoised.max()) > float(x_train.sum()):
        raise ValueError("denoised output maximum exceeds total train count")
    with output_path.open("wb") as handle:
        np.save(handle, denoised, allow_pickle=False)
    return 0


def process_peak_rss_bytes(pid: int) -> int:
    try:
        import psutil

        process = psutil.Process(pid)
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return int(total)
    except Exception:
        return 0


def run_candidate(
    script_path: Path,
    candidate: Path,
    paths: dict[str, Path],
    metadata: dict[str, Any],
    baseline_info: dict[str, Any],
    cache_root: Path,
    seed: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    work_parent = cache_root / "work"
    work_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="candidate-", dir=work_parent))
    output_path = work_dir / "denoised.npy"
    started = time.monotonic()
    peak_rss = 0
    command = [
        sys.executable,
        str(script_path),
        "--worker",
        "--candidate",
        str(candidate),
        "--train",
        str(paths["train"]),
        "--denoised-output",
        str(output_path),
        "--random-state",
        str(seed),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        entry
        for entry in ("/opt/simpletes", environment.get("PYTHONPATH", ""))
        if entry
    )
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        environment.setdefault(variable, "8")

    print(f"[run] {candidate} timeout={timeout_seconds}s", flush=True)
    worker_log_path = work_dir / "worker.log"
    timed_out = False
    with worker_log_path.open("w+", encoding="utf-8") as worker_log:
        process = subprocess.Popen(
            command,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        while process.poll() is None:
            peak_rss = max(peak_rss, process_peak_rss_bytes(process.pid))
            if time.monotonic() - started > timeout_seconds:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.25)
        process.wait()
        worker_log.flush()
        worker_log.seek(0)
        worker_output = worker_log.read()[-8000:]
    candidate_seconds = time.monotonic() - started

    try:
        if timed_out:
            return {
                "status": "timeout",
                "execution_valid": False,
                "valid": False,
                "protocol_valid": False,
                "candidate_seconds": candidate_seconds,
                "within_reference_search_budget": False,
                "peak_rss_bytes": peak_rss,
                "timeout_seconds": timeout_seconds,
                "reported_score": 0.0,
                "error": f"candidate exceeded {timeout_seconds} seconds",
            }
        if process.returncode != 0 or not output_path.is_file():
            return {
                "status": "error",
                "execution_valid": False,
                "valid": False,
                "protocol_valid": False,
                "candidate_seconds": candidate_seconds,
                "within_reference_search_budget": bool(
                    candidate_seconds <= REFERENCE_SEARCH_TIMEOUT
                ),
                "peak_rss_bytes": peak_rss,
                "timeout_seconds": timeout_seconds,
                "reported_score": 0.0,
                "error": worker_output or f"worker exited {process.returncode}",
            }

        metric_started = time.monotonic()
        raw = compute_raw_metrics(paths["train"], paths["test"], output_path, metadata)
        metric_seconds = time.monotonic() - metric_started
        normalized = normalize_metrics(raw, baseline_info)
        result = {
            "status": "success",
            "execution_valid": True,
            "valid": bool(normalized["protocol_valid"]),
            "candidate_seconds": candidate_seconds,
            "within_reference_search_budget": bool(
                candidate_seconds <= REFERENCE_SEARCH_TIMEOUT
            ),
            "metric_seconds": metric_seconds,
            "peak_rss_bytes": peak_rss,
            "timeout_seconds": timeout_seconds,
            **normalized,
        }
        print(
            f"[done] score={result['reported_score']:.9f} "
            f"mse={result['mse_raw']:.9f} poisson={result['poisson_raw']:.9f} "
            f"poisson_pass={result['poisson_pass']}",
            flush=True,
        )
        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


def discover_candidates(outputs_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidate_paths = list(outputs_root.rglob("best/best_program.py"))
    # Preserve all semantics while evaluating sparse implementations before
    # candidates that materialize dense matrix powers on the 24k-cell dataset.
    candidate_paths.sort(
        key=lambda path: (
            "np.linalg.matrix_power" in path.read_text(encoding="utf-8"),
            str(path),
        )
    )
    for candidate in candidate_paths:
        info_path = candidate.with_name("best_program_info.json")
        if not info_path.is_file():
            raise FileNotFoundError(f"missing best_program_info.json beside {candidate}")
        run_dir = candidate.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "run_key": str(run_dir.relative_to(outputs_root)),
                "candidate": candidate,
                "info": info_path,
                "best_id": info.get("id"),
                "candidate_sha256": sha256_file(candidate),
            }
        )
    if not candidates:
        raise RuntimeError(f"no canonical best programs found under {outputs_root}")
    return candidates


def new_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "name": PROTOCOL_NAME,
            "paper_reporting": "per-dataset mean of normalized MSE and Poisson",
            "headline_dataset": "tabula",
            "search_dataset": "pancreas",
            "held_out_datasets": ["pbmc", "tabula"],
            "split_seed": args.split_seed,
            "train_fraction": 0.9,
            "candidate_random_state": args.random_state,
            "reconstruction_status": (
                "Reconstructed from the SimpleTES constants and search evaluator; "
                "the authors did not publish their PBMC/Tabula final runner."
            ),
            "split_seed_evidence": (
                "Seed 42 reproduces all six published PBMC/Tabula no-denoising and "
                "perfect-denoising constants within 5e-7 in this fixed image."
            ),
            "known_protocol_ambiguity": (
                "The comparison TTT-Discover final notebook at commit "
                "6c40e82dab9d5de7416ac873ad5cd3106084aaed uses its wrapper defaults "
                "(split seed 0 and no random_state keyword), whereas SimpleTES uses "
                "seed/random_state 42 during search."
            ),
            "simpletes_source_revision": (
                "47d3413da1d85dc24341219d47452d2601e56a57"
            ),
            "comparison_notebook_revision": (
                "6c40e82dab9d5de7416ac873ad5cd3106084aaed"
            ),
            "candidate_timeout_seconds_per_dataset": args.candidate_timeout,
            "candidate_timeout_note": (
                "Post-hoc safety cap; the reference search evaluator uses 400 seconds, "
                "but in this container/host audit the released paper program itself "
                "exceeded that on held-out Tabula."
            ),
            "reference_search_timeout_seconds": REFERENCE_SEARCH_TIMEOUT,
            "container_memory_limit_bytes": container_memory_limit_bytes(),
            "paper_final_memory_limit": None,
            "poisson_norm_constraint": {
                "minimum": POISSON_THRESHOLD,
            },
            "poisson_sanity_check": {
                "minimum_raw": "published perfect_poisson (exact upstream threshold)",
                "tolerance": POISSON_SANITY_TOLERANCE,
            },
            "normalization_baselines": "published SimpleTES evaluator constants",
            "baseline_integrity_tolerance": BASELINE_TOLERANCE,
            "raw_source_artifacts": RAW_SOURCE_ARTIFACTS,
            "versions": package_versions(),
            "package_provenance": package_provenance(),
            "container_image_id": os.environ.get("FINAL_EVAL_IMAGE_ID", "unknown"),
        },
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_at": None,
        "datasets": {},
        "runs": {},
    }


def load_aggregate(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not path.exists():
        return new_aggregate(args)
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    protocol = aggregate.get("protocol", {})
    expected = {
        "name": PROTOCOL_NAME,
        "split_seed": args.split_seed,
        "candidate_random_state": args.random_state,
        "candidate_timeout_seconds_per_dataset": args.candidate_timeout,
    }
    actual = {key: protocol.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"existing result uses a different protocol: {actual} != {expected}")
    # Backfill metadata without relabelling an existing run with package or
    # container provenance from a later resume environment.
    template = new_aggregate(args)["protocol"]
    for key, value in template.items():
        protocol.setdefault(key, value)
    for name, version in template["versions"].items():
        protocol["versions"].setdefault(name, version)
    # This sentence is descriptive rather than execution provenance and may be
    # clarified across compatible script revisions.
    protocol["candidate_timeout_note"] = template["candidate_timeout_note"]
    return aggregate


def update_paper_reporting(run: dict[str, Any]) -> None:
    datasets = run.get("datasets", {})
    if not all(name in datasets for name in ("pbmc", "tabula")):
        return
    pbmc = datasets["pbmc"]
    tabula = datasets["tabula"]
    run["paper_reporting"] = {
        "pbmc_score": pbmc.get("reported_score"),
        "tabula_score": tabula.get("reported_score"),
        "headline_dataset": "tabula",
        "headline_score": tabula.get("reported_score"),
        "all_datasets_execution_valid": bool(
            pbmc.get("execution_valid") and tabula.get("execution_valid")
        ),
        "all_datasets_protocol_valid": bool(
            pbmc.get("protocol_valid") and tabula.get("protocol_valid")
        ),
        # Compatibility alias: validity means the paper scoring protocol, not
        # merely that both candidate calls returned successfully.
        "all_datasets_valid": bool(
            pbmc.get("protocol_valid") and tabula.get("protocol_valid")
        ),
    }


def evaluate(args: argparse.Namespace) -> int:
    outputs_root = args.outputs_root.resolve()
    cache_root = args.cache_root.resolve()
    result_json = args.result_json.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    raw_cache = Path(
        os.environ.setdefault(
            "OPENPROBLEMS_CACHE_DIR", str(cache_root / "openproblems")
        )
    ).resolve()
    raw_cache.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_root).free
    if free_bytes < MIN_FREE_CACHE_BYTES:
        raise RuntimeError(
            f"final-eval cache needs at least {MIN_FREE_CACHE_BYTES} free bytes; "
            f"only {free_bytes} are available at {cache_root}"
        )
    print(
        f"[cache] prepared={cache_root} raw={raw_cache} free={free_bytes} bytes",
        flush=True,
    )
    candidates = discover_candidates(outputs_root)
    print(f"[plan] {len(candidates)} canonical best program(s)", flush=True)

    aggregate = load_aggregate(result_json, args)
    for entry in candidates:
        run = aggregate["runs"].setdefault(
            entry["run_key"],
            {
                "best_id": entry["best_id"],
                "candidate_sha256": entry["candidate_sha256"],
                "best_program": str(entry["candidate"].relative_to(outputs_root)),
                "best_program_info": str(entry["info"].relative_to(outputs_root)),
                "datasets": {},
            },
        )
        if run.get("candidate_sha256") != entry["candidate_sha256"]:
            raise RuntimeError(f"candidate changed since partial result: {entry['run_key']}")

    atomic_write_json(result_json, aggregate)

    for name in args.datasets:
        metadata = prepare_dataset(cache_root, name, args.split_seed)
        paths = dataset_paths(cache_root, name, args.split_seed)
        cached_dataset = aggregate["datasets"].get(name)
        if (
            cached_dataset
            and cached_dataset.get("data", {}).get("train_sha256") == metadata["train_sha256"]
            and cached_dataset.get("data", {}).get("test_sha256") == metadata["test_sha256"]
        ):
            baseline_info = cached_dataset["baselines"]
            cached_dataset["data"] = metadata
            print(f"[metric] reusing verified {name} baselines", flush=True)
        else:
            baseline_info = compute_live_baselines(paths, metadata, name)
            aggregate["datasets"][name] = {
                "data": metadata,
                "baselines": baseline_info,
            }
            aggregate["updated_at"] = utc_now()
            atomic_write_json(result_json, aggregate)

        for entry in candidates:
            run = aggregate["runs"][entry["run_key"]]
            previous = run["datasets"].get(name)
            if previous and previous.get("status") in {"success", "timeout", "error"} and not args.force:
                print(f"[skip] {name} {entry['run_key']}: {previous['status']}", flush=True)
                continue
            print(f"[case] dataset={name} run={entry['run_key']}", flush=True)
            try:
                result = run_candidate(
                    Path(__file__).resolve(),
                    entry["candidate"],
                    paths,
                    metadata,
                    baseline_info,
                    cache_root,
                    args.random_state,
                    args.candidate_timeout,
                )
            except Exception as error:
                result = {
                    "status": "evaluator_error",
                    "execution_valid": False,
                    "valid": False,
                    "protocol_valid": False,
                    "reported_score": 0.0,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-8000:],
                }
            result["evaluated_at"] = utc_now()
            run["datasets"][name] = result
            update_paper_reporting(run)
            aggregate["updated_at"] = utc_now()
            atomic_write_json(result_json, aggregate)

        gc.collect()

    if all(
        all(name in run.get("datasets", {}) for name in ("pbmc", "tabula"))
        for run in aggregate["runs"].values()
    ):
        aggregate["completed_at"] = utc_now()
    aggregate["updated_at"] = utc_now()
    atomic_write_json(result_json, aggregate)
    print(f"[result] {result_json}", flush=True)
    return 0


def compact_dataset_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "execution_valid",
        "valid",
        "evaluated_at",
        "candidate_seconds",
        "within_reference_search_budget",
        "metric_seconds",
        "peak_rss_bytes",
        "timeout_seconds",
        "mse_raw",
        "poisson_raw",
        "mse_norm_unclipped",
        "mse_norm",
        "poisson_norm",
        "mean_score",
        "poisson_pass",
        "poisson_constraint_pass",
        "poisson_sanity_pass",
        "protocol_valid",
        "reported_score",
        "live_baseline_mse_norm",
        "live_baseline_poisson_norm",
        "live_baseline_mean_score",
        "error",
    )
    return {key: result[key] for key in keys if key in result}


def apply_results(args: argparse.Namespace) -> int:
    outputs_root = args.outputs_root.resolve()
    result_json = args.result_json.resolve()
    aggregate = json.loads(result_json.read_text(encoding="utf-8"))
    if aggregate.get("protocol", {}).get("name") != PROTOCOL_NAME:
        raise RuntimeError("aggregate result protocol is not recognized")
    if not aggregate.get("completed_at"):
        raise RuntimeError("aggregate result is not complete")

    discovered = {
        entry["run_key"]: entry for entry in discover_candidates(outputs_root)
    }
    recorded = aggregate.get("runs", {})
    if set(recorded) != set(discovered):
        raise RuntimeError(
            "aggregate/canonical run set mismatch: "
            f"recorded={sorted(recorded)}, discovered={sorted(discovered)}"
        )

    updates: list[tuple[Path, Path, dict[str, Any]]] = []
    for run_key, run_result in recorded.items():
        candidate = outputs_root / run_result["best_program"]
        info_path = outputs_root / run_result["best_program_info"]
        if sha256_file(candidate) != run_result["candidate_sha256"]:
            raise RuntimeError(f"candidate checksum changed: {candidate}")
        if not all(name in run_result.get("datasets", {}) for name in ("pbmc", "tabula")):
            raise RuntimeError(f"run lacks both held-out results: {run_key}")
        if not all(
            run_result["datasets"][name].get("status")
            in {"success", "timeout", "error", "evaluator_error"}
            for name in ("pbmc", "tabula")
        ):
            raise RuntimeError(f"run has a non-terminal held-out result: {run_key}")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("id") != run_result.get("best_id"):
            raise RuntimeError(f"best program id mismatch: {info_path}")

        backup = info_path.with_name("best_program_info.pre_posthoc_final_eval.json")

        pbmc = compact_dataset_result(run_result["datasets"]["pbmc"])
        tabula = compact_dataset_result(run_result["datasets"]["tabula"])
        reporting = run_result.get("paper_reporting", {})
        block = {
            "schema_version": SCHEMA_VERSION,
            "protocol": aggregate["protocol"],
            "aggregate_result": str(result_json),
            "candidate_sha256": run_result["candidate_sha256"],
            "datasets": {"pbmc": pbmc, "tabula": tabula},
            "paper_reporting": reporting,
            "appended_at": utc_now(),
            "note": (
                "Reproducible held-out post-hoc evaluation. Existing test_* metrics "
                "are preserved legacy Pancreas re-evaluation values."
            ),
        }
        info["posthoc_final_evaluation"] = block
        metrics = info.setdefault("metrics", {})
        for name, result in (("pbmc", pbmc), ("tabula", tabula)):
            metrics[f"posthoc_{name}_completed"] = int(
                result.get("status") in ("success", "timeout", "error")
            )
            metrics[f"posthoc_{name}_protocol"] = PROTOCOL_NAME
            metrics[f"posthoc_{name}_protocol_version"] = 2
            metrics[f"posthoc_{name}_candidate_sha256"] = run_result["candidate_sha256"]
            for key in (
                "mse_raw",
                "poisson_raw",
                "mse_norm",
                "poisson_norm",
                "poisson_pass",
                "poisson_constraint_pass",
                "poisson_sanity_pass",
                "protocol_valid",
                "execution_valid",
                "mean_score",
                "reported_score",
            ):
                if key in result:
                    value = result[key]
                    metrics[f"posthoc_{name}_{key}"] = float(value) if isinstance(value, bool) else value
        metrics["posthoc_final_score"] = reporting.get("headline_score")
        metrics["posthoc_final_valid"] = float(bool(reporting.get("all_datasets_valid")))
        updates.append((info_path, backup, info))

    # Mutate only after every candidate, ID, checksum, and result has passed the
    # preflight above, so a stale aggregate cannot produce a partial publication.
    for info_path, backup, info in updates:
        if not backup.exists():
            shutil.copy2(info_path, backup)
        atomic_write_json(info_path, info)
        print(f"[apply] {info_path}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("pbmc", "tabula"),
        default=("pbmc", "tabula"),
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--candidate-timeout", type=int, default=DEFAULT_CANDIDATE_TIMEOUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--apply-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--train", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--denoised-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        for field in ("candidate", "train", "denoised_output"):
            if getattr(args, field) is None:
                parser.error(f"--worker requires --{field.replace('_', '-')}")
        return args
    for field in ("outputs_root", "result_json"):
        if getattr(args, field) is None:
            parser.error(f"--{field.replace('_', '-')} is required")
    if not args.apply_only and args.cache_root is None:
        parser.error("--cache-root is required unless --apply-only is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return candidate_worker(args.candidate, args.train, args.denoised_output, args.random_state)
    if args.apply_only:
        return apply_results(args)
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
