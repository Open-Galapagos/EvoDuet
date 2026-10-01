"""PBMC and Tabula final evaluation for the container's test mode.

Uses the same preparation, normalization, Poisson gate and candidate runner as
the audited posthoc workflow. Pancreas remains the evolution-loop evaluator.
"""

import hashlib
import importlib.util
import os
from pathlib import Path

DATASETS = ("pbmc", "tabula")


def load_posthoc():
    path = Path(__file__).with_name("posthoc_final_eval.py")
    spec = importlib.util.spec_from_file_location("denoising_posthoc_final", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(program_path):
    result = {"validity": 0, "combined_score": 0.0}
    for name in DATASETS:
        result.update(
            {
                f"posthoc_{name}_reported_score": None,
                f"posthoc_{name}_execution_valid": 0,
                f"posthoc_{name}_protocol_valid": 0,
                f"posthoc_{name}_completed": 0,
            }
        )
    try:
        posthoc = load_posthoc()
        cache = Path(
            os.environ.get("SIMPLETES_DENOISING_FINAL_CACHE")
            or Path(os.environ.get("OPENPROBLEMS_CACHE_DIR", "/tmp")) / "simpletes_final"
        ).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        raw_cache = Path(
            os.environ.setdefault("OPENPROBLEMS_CACHE_DIR", str(cache / "openproblems"))
        ).resolve()
        raw_cache.mkdir(parents=True, exist_ok=True)
        candidate = Path(program_path).resolve()
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    except Exception as exc:
        for name in DATASETS:
            result[f"posthoc_{name}_error"] = f"{type(exc).__name__}: {exc}"
        return result

    details = {"split_seed": 42, "random_state": 42, "datasets": {}}
    result["denoising_final_evaluation"] = details
    for name in DATASETS:
        try:
            metadata = posthoc.prepare_dataset(cache, name, 42)
            paths = posthoc.dataset_paths(cache, name, 42)
            baselines = posthoc.compute_live_baselines(paths, metadata, name)
            evaluated = posthoc.run_candidate(
                Path(posthoc.__file__),
                candidate,
                paths,
                metadata,
                baselines,
                cache,
                42,
                int(os.environ.get("SIMPLETES_DENOISING_FINAL_CANDIDATE_TIMEOUT", "7200")),
            )
            details["datasets"][name] = {
                "data": metadata,
                "baselines": baselines,
                "evaluation": evaluated,
            }
            for key in (
                "mse_raw",
                "poisson_raw",
                "mse_norm",
                "poisson_norm",
                "mean_score",
                "reported_score",
                "execution_valid",
                "protocol_valid",
                "poisson_pass",
                "poisson_constraint_pass",
                "poisson_sanity_pass",
            ):
                if key in evaluated:
                    value = evaluated[key]
                    result[f"posthoc_{name}_{key}"] = (
                        int(value) if isinstance(value, bool) else value
                    )
            result[f"posthoc_{name}_completed"] = int(
                evaluated.get("status") in ("success", "timeout", "error")
            )
            result[f"posthoc_{name}_candidate_sha256"] = digest
            result[f"posthoc_{name}_protocol"] = posthoc.PROTOCOL_NAME
            # Keep the upstream zero penalty for candidate errors/timeouts.
            # Infrastructure exceptions above remain unavailable for this dataset.
            result[f"posthoc_{name}_protocol_version"] = 2
        except Exception as exc:
            result[f"posthoc_{name}_error"] = f"{type(exc).__name__}: {exc}"

    result["validity"] = int(all(result[f"posthoc_{name}_execution_valid"] for name in DATASETS))
    if all(result[f"posthoc_{name}_reported_score"] is not None for name in DATASETS):
        # Completed candidate failures contribute their native zero penalty.
        result["posthoc_heldout_mean_score"] = sum(
            result[f"posthoc_{name}_reported_score"] for name in DATASETS
        ) / len(DATASETS)
        result["combined_score"] = result["posthoc_heldout_mean_score"]
    return result
