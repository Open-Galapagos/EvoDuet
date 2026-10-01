"""Task-specific reporting scores; search rewards keep their original semantics.

Shared by the report and reference exporter. A held-out score may be written by
the posthoc runner directly or by SkyDiscover with a ``test_`` prefix. Neither
train metrics nor a failed new final evaluation may substitute for that score.
"""

import math


def _spec(task, metric, direction, unit, stage, aggregation):
    return dict(
        task=task,
        metric=metric,
        direction=direction,
        unit=unit,
        stage=stage,
        aggregation=aggregation,
    )


TASK_METRICS = {
    "Erdos": _spec(
        "erdos/simpletes_erdos_min_overlap",
        "c5",
        "min",
        "overlap",
        "construction",
        "validated construction",
    ),
    "AC1": _spec(
        "autocorrelation/simpletes_autocorrelation_first",
        "c1",
        "min",
        "bound",
        "construction",
        "validated construction",
    ),
    "AC2": _spec(
        "autocorrelation/simpletes_autocorrelation_second",
        "c2",
        "max",
        "bound",
        "construction",
        "validated construction",
    ),
    "AC3": _spec(
        "autocorrelation/simpletes_autocorrelation_third",
        "c3",
        "min",
        "bound",
        "construction",
        "validated construction",
    ),
    "CP (n=26)": _spec(
        "circle_packing/simpletes_circle_packing_26",
        "sum_radii",
        "max",
        "sum of radii",
        "construction",
        "sum over 26 circles",
    ),
    "CP (n=32)": _spec(
        "circle_packing/simpletes_circle_packing_32",
        "sum_radii",
        "max",
        "sum of radii",
        "construction",
        "sum over 32 circles",
    ),
    "Hadamard": _spec(
        "hadamard_maximal_det/hadamard_maximal_det_29",
        "determinant_ratio",
        "max",
        "normalized determinant",
        "construction",
        "order-29 normalized determinant",
    ),
    "Sums/Diffs": _spec(
        "sums_diffs/simpletes_sums_diffs",
        "c_value",
        "max",
        "log ratio",
        "construction",
        "log(|A+A|/|A|) / log(|A-A|/|A|)",
    ),
    **{
        label: _spec(
            f"astrodynamics/{name}",
            "mean_total_dv",
            "min",
            "delta-v",
            "construction",
            "original evaluator mean total delta-v",
        )
        for label, name in (
            ("Cassini", "cassini"),
            ("Galileo", "galileo"),
            ("Mariner 10", "mariner_10"),
            ("Rosetta", "rosetta"),
            ("Voyager 2", "voyager_2"),
        )
    },
    **{
        label: _spec(
            f"scaling_law/{name}_scaling_law",
            "posthoc_test_r2",
            "max",
            "R2",
            "final",
            "train fitting; concatenate held-out predictions; R2 clipped to [-1,1] before run aggregation",
        )
        for label, name in (
            ("Domain Mixture Scaling", "domain_mixture"),
            ("Easy Question Scaling", "easy_question"),
            ("LR/BSZ Scaling", "lr_bsz"),
            ("Parallel Scaling", "parallel"),
        )
    },
    "Denoising": _spec(
        "open_problems_bio/denoising",
        "posthoc_heldout_mean_score",
        "max",
        "normalized score",
        "final",
        "Arithmetic mean of PBMC and Tabula reported scores; each dataset averages normalized MSE/Poisson; completed candidate/protocol failures score 0; missing evaluation is unavailable",
    ),
    "Swap Reduction": _spec(
        "qubit_routing/swap_reduction",
        "q20_total_swaps",
        "min",
        "added SWAPs",
        "final",
        "saved search winner's final Q20 sum over 24 circuits; failed cases use baseline-equivalent SWAPs as in upstream CNOT scoring",
    ),
    "AHC039": _spec(
        "ahc/simpletes_ahc039",
        "total_score",
        "max",
        "total points",
        "final",
        "saved search winner's final combined_score * 150 * 1500, rounded to total points; failed repeats score 0 before the original 3-repeat aggregation",
    ),
    "AHC058": _spec(
        "ahc/simpletes_ahc058",
        "total_score",
        "max",
        "total points",
        "final",
        "saved search winner's final combined_score * 150 * 3000000, rounded to total points; failed repeats score 0 before the original 3-repeat aggregation",
    ),
}

DENOISING_REPORT_METRIC = TASK_METRICS["Denoising"]["metric"]
SWAP_SUITE_SIGNATURE = {
    "total_cases": 72,
    "total_weight": 24,
    "total_original_cnot_added": 333756,
    "weighted_total_original_cnot_added": 119874,
}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def q20_total_swaps(metrics):
    """Recover Q20 including upstream baseline penalties for failed circuits."""
    expected = SWAP_SUITE_SIGNATURE
    if any(
        not finite_number(metrics.get(key))
        or not math.isclose(metrics[key], value, rel_tol=0, abs_tol=1e-6)
        for key, value in expected.items()
    ):
        return None
    ok, failed = metrics.get("ok_cases"), metrics.get("failed_cases")
    if not all(finite_number(v) and v >= 0 and v == int(v) for v in (ok, failed)):
        return None
    if ok + failed != expected["total_cases"]:
        return None
    swaps, cnot, weighted = (
        metrics.get(key)
        for key in (
            "total_swaps",
            "total_added_cnot_by_candidate",
            "weighted_total_added_cnot_by_candidate",
        )
    )
    if not all(finite_number(value) and value >= 0 for value in (swaps, cnot, weighted)):
        return None
    if swaps != int(swaps) or cnot < 3 * swaps - 1e-6:
        return None
    if failed == 0 and not math.isclose(cnot, 3 * swaps, rel_tol=0, abs_tol=1e-6):
        return None
    value = (0.4 * cnot - weighted) / 0.6
    integer = round(value)
    if not math.isclose(value, integer, rel_tol=0, abs_tol=1e-6) or not 0 <= integer <= cnot / 3 + 1e-6:
        return None
    return integer


FINAL_METRIC_FAMILIES = {
    "Denoising": "posthoc_tabula_",
    "AHC039": "",
    "AHC058": "",
    "Swap Reduction": "",
}


def final_metric_family(task):
    """Key prefix of the task's final (posthoc) evaluation metrics."""
    return FINAL_METRIC_FAMILIES.get(task, "posthoc_test_")


DENOISING_FINAL_PROTOCOL = "simpletes-denoising-paper-posthoc-v2"
SCALING_FINAL_PROTOCOL = "simpletes-scaling-held-out-v2"


def _current_protocol(metrics, family, protocol):
    """Strings may move to container artifacts; numeric versions remain metrics."""
    name, version = metrics.get(family + "protocol"), metrics.get(family + "protocol_version")
    return (name in (None, protocol) and version in (None, 2)
            and (name == protocol or version == 2))


def final_metric_namespace(metrics, task):
    """A recognized automatic final result supersedes older manual metrics."""
    family = final_metric_family(task)
    # Corrected posthoc backfills supersede a native result from the old protocol.
    # Numeric versions survive container serialization, which moves strings to artifacts.
    if task == "Denoising" or TASK_METRICS[task]["metric"] == "posthoc_test_r2":
        version_key = family + "protocol_version"
        manual_version = metrics.get(version_key, 1)
        native_version = metrics.get("test_" + version_key, 1)
        if (finite_number(manual_version) and finite_number(native_version)
                and manual_version > native_version):
            return metrics, False
    if TASK_METRICS[task]["stage"] == "final" and any(
        key.startswith("test_" + family) for key in metrics
    ):
        return {
            key.removeprefix("test_"): value
            for key, value in metrics.items()
            if key.startswith("test_")
        }, True
    return metrics, False


def final_evaluation_recorded(metrics, task):
    """True when a final evaluation of the saved best is recorded, whether valid or not."""
    if TASK_METRICS[task]["stage"] != "final":
        return False
    namespace, automatic = final_metric_namespace(metrics, task)
    if task in ("AHC039", "AHC058", "Swap Reduction"):
        return automatic
    if task == "Denoising":
        return all(_current_protocol(namespace, f"posthoc_{dataset}_", DENOISING_FINAL_PROTOCOL)
                   and namespace.get(f"posthoc_{dataset}_completed") == 1
                   for dataset in ("pbmc", "tabula"))
    if TASK_METRICS[task]["metric"] == "posthoc_test_r2":
        return (_current_protocol(namespace, "posthoc_test_", SCALING_FINAL_PROTOCOL)
                and namespace.get("posthoc_test_completed") == 1)
    return any(key.startswith(final_metric_family(task)) for key in namespace)


def raw_task_score(metrics, task):
    """Return the task score, preserving upstream penalties and raw finite R2."""
    spec = TASK_METRICS[task]
    metrics, automatic = final_metric_namespace(metrics, task)
    if task == "Swap Reduction":
        return q20_total_swaps(metrics)
    if task == "Denoising":
        # Recompute from both dataset scores so an absent dataset cannot be
        # replaced by an old cached aggregate or the other dataset's score.
        values = [_denoising_dataset_score(metrics, dataset) for dataset in ("pbmc", "tabula")]
        return sum(values) / 2 if all(value is not None for value in values) else None
    elif spec["metric"] == "posthoc_test_r2":
        value = metrics.get(spec["metric"])
        # Native v1 discarded R2 on failure but retained the original -1e6 reward.
        if value is None and metrics.get("posthoc_test_combined_score") == -1e6:
            value = -1e6
        return value if finite_number(value) else None
    elif task in ("AHC039", "AHC058"):
        combined = metrics.get("combined_score")
        if finite_number(combined) and (
            metrics.get("num_cases") == 150 or combined == 0
        ):
            # total_score includes AC subsets of failed repeats; combined_score
            # already applies the upstream all-cases-pass rule within each repeat.
            norm = 1500 if task == "AHC039" else 3000000
            return round(combined * 150 * norm)
        if any(
            metrics.get(key, expected) != expected
            for key, expected in (("num_cases", 150), ("num_accepted", 150))
        ):
            return None
        # Legacy records with only total_score are usable only when fully accepted.
        if metrics.get("validity", 1) != 1:
            return None
    elif task != "Denoising":
        validity = metrics.get("validity", 1)
        if (not isinstance(validity, (int, float))
                or not math.isfinite(validity) or validity <= 0):
            return None
    value = metrics.get(spec["metric"])
    if not finite_number(value):
        return None
    return value


def task_score(metrics, task):
    value = raw_task_score(metrics, task)
    if TASK_METRICS[task]["metric"] == "posthoc_test_r2":
        if value is not None:
            return max(-1.0, min(1.0, value))
        namespace, _ = final_metric_namespace(metrics, task)
        # Keep strict-JSON diagnostics while applying the paper's clipping rule.
        r2 = namespace.get("posthoc_test_r2")
        if r2 in ("-Infinity", -math.inf):
            return -1.0
        if r2 in ("Infinity", math.inf):
            return 1.0
    return value


def denoising_tabula_score(metrics):
    """Tabula's individual score, for per-dataset comparisons only."""
    namespace, _ = final_metric_namespace(metrics, "Denoising")
    return _denoising_dataset_score(namespace, "tabula")


def _denoising_dataset_score(metrics, dataset):
    value = metrics.get(f"posthoc_{dataset}_reported_score")
    if not finite_number(value) or metrics.get(f"posthoc_{dataset}_completed") == 0:
        return None
    if metrics.get(f"posthoc_{dataset}_execution_valid", 1) != 1 and value != 0:
        return None
    return value


def denoising_mean_score(metrics):
    return raw_task_score(metrics, "Denoising")
