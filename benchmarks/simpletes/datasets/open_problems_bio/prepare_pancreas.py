#!/usr/bin/env python3
"""Rebuild the pancreas train/test split that denoising/evaluator.py reads.

SimpleTES ships neither ``denoising/denoising_datasets/pancreas/pancreas_{train,test}_seed42.npy``
nor the script that produced them (its README says ``setup.sh`` generates them; it
does not).  This regenerates them with the OpenProblems v1.0.0 denoising pipeline the
evaluator's baseline constants come from:

    load_pancreas(keep_techs=["inDrop1"])          # openproblems.data.pancreas
    split_data(adata, train_frac=0.9, seed=SEED)   # molecular cross-validation
    obsm["train"], obsm["test"]  ->  dense float64 .npy

and then checks the result against the evaluator itself: with the evaluator's own
``_compute_metrics``, "no denoising" (Y = X_train) must reproduce
``BASELINES["pancreas"]["baseline_mse"]`` / ``["baseline_poisson"]`` and "perfect
denoising" (Y = X_test) must reproduce ``["perfect_poisson"]``.  Those constants
are the only trace of the original files, so matching them is the acceptance test.

Note that the vendored ``split_data`` ignores ``train_frac`` (it hard-codes 0.9), so
the only real degrees of freedom are the seed and the batches kept.

Run inside the family venv (setup.sh):  .venv/bin/python prepare_pancreas.py [--seed 42]
The Docker build runs it without ``--strict``: the files are always written and a
mismatch is only reported, so the image still builds; ``--check-only --strict`` is the
hard acceptance test.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

FAMILY_ROOT = Path(__file__).resolve().parent
DATA_DIR = FAMILY_ROOT / "denoising" / "denoising_datasets" / "pancreas"
TRAIN = DATA_DIR / "pancreas_train_seed42.npy"
TEST = DATA_DIR / "pancreas_test_seed42.npy"
TOL = 5e-7  # the evaluator stores its constants rounded to 6 decimals


def load_evaluator():
    spec = importlib.util.spec_from_file_location("denoising_evaluator", FAMILY_ROOT / "denoising" / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    # The evaluator imports simpletes.construction: the vendored package sits at the
    # repo-root mirror (<root>/simpletes) in the image and on the host; the build
    # context additionally carries a copy beside this file.
    for candidate in (FAMILY_ROOT.parents[1], FAMILY_ROOT):
        if (candidate / "simpletes" / "construction.py").exists():
            sys.path.insert(0, str(candidate))
            break
    spec.loader.exec_module(module)
    return module


def import_mcv_util_without_torch():
    """Make ``import molecular_cross_validation.util`` work without torch.

    The evaluation venv deliberately has no torch (the evaluator shims
    ``molecular_cross_validation.mcv_sweep`` itself), but the package's
    ``__init__`` imports ``.models`` -> torch.  ``split_data`` only needs
    ``molecular_cross_validation.util.split_molecules`` (numpy/scipy/numba), so
    register a bare parent package pointing at the installed directory and let
    the real ``util`` submodule import normally.
    """
    import types

    if "molecular_cross_validation" in sys.modules:
        return
    spec = importlib.util.find_spec("molecular_cross_validation")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("molecular-cross-validation is not installed (setup.sh step 2)")
    pkg = types.ModuleType("molecular_cross_validation")
    pkg.__path__ = list(spec.submodule_search_locations)
    sys.modules["molecular_cross_validation"] = pkg
    import molecular_cross_validation.util  # noqa: F401  (real file, torch-free)


def build(seed: int, techs: list[str]) -> None:
    import_mcv_util_without_torch()
    from openproblems.data.pancreas import load_pancreas
    from openproblems.tasks.denoising.datasets.utils import split_data

    adata = load_pancreas(test=False, keep_techs=techs)
    adata = split_data(adata, train_frac=0.9, seed=seed)
    X_train = adata.obsm["train"].toarray().astype(np.float64)
    X_test = adata.obsm["test"].toarray().astype(np.float64)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN, X_train)
    np.save(TEST, X_test)
    print(f"wrote {TRAIN.name} {X_train.shape} and {TEST.name} {X_test.shape} (seed={seed}, techs={techs})")


def check() -> int:
    evaluator = load_evaluator()
    expected = evaluator.BASELINES["pancreas"]
    X_train = np.load(TRAIN)
    X_test = np.load(TEST)
    base_mse, base_poisson = evaluator._compute_metrics(X_train.copy(), X_train, X_test)
    perf_mse, perf_poisson = evaluator._compute_metrics(X_test.copy(), X_train, X_test)
    rows = [
        ("baseline_mse", base_mse, expected["baseline_mse"]),
        ("baseline_poisson", base_poisson, expected["baseline_poisson"]),
        ("perfect_mse", perf_mse, expected["perfect_mse"]),
        ("perfect_poisson", perf_poisson, expected["perfect_poisson"]),
    ]
    bad = 0
    for name, got, want in rows:
        ok = abs(got - want) <= TOL
        bad += not ok
        print(f"  {name:17s} got={got:.6f} expected={want:.6f} {'OK' if ok else 'MISMATCH'}")
    print("pancreas split reproduces the evaluator's baseline constants" if not bad else
          f"{bad} constant(s) do not match: the split is not the one the evaluator was calibrated on")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--techs", default="inDrop1", help="comma-separated pancreas batches to keep")
    parser.add_argument("--check-only", action="store_true", help="only validate existing .npy files")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--strict", action="store_true", help="exit 1 when a constant does not reproduce")
    args = parser.parse_args()
    if not args.check_only:
        build(args.seed, args.techs.split(","))
    if args.no_check:
        return 0
    bad = check()
    if bad and not args.strict:
        print("(non-strict: leaving the files in place; rerun with --check-only --strict for a hard check)")
        return 0
    return bad


if __name__ == "__main__":
    sys.exit(main())
