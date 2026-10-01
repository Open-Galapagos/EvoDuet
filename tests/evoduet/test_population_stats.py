"""Behavioral tests for deterministic retained-population statistics."""

import json
import math
from types import SimpleNamespace

import pytest

from skydiscover.evoduet.population_stats import serialize_population_statistics


def program(identifier, score=None, iteration=None, parent_id=None, contexts=(), **metrics):
    return SimpleNamespace(
        id=identifier,
        iteration_found=iteration,
        parent_id=parent_id,
        other_context_ids=contexts,
        metrics={"combined_score": score, **metrics},
    )


def read(population, **kwargs):
    return json.loads(serialize_population_statistics(population, **kwargs))


def test_distribution_uses_whole_population_and_trace_keeps_chronology():
    population = [
        program("d", 4, 3, "b"),
        program("b", 2, 1, "a"),
        program("a", 1),
        program("c", 3, 2, "b"),
    ]
    result = read(population, recent_k=2, parent=population[1])
    assert result["population_size"] == 4
    assert result["score_distribution"] == {
        "best": 4,
        "mean": 2.5,
        "population_std": math.sqrt(1.25),
        "median": 2.5,
        "q25": 1.75,
        "q75": 3.25,
        "worst": 1,
    }
    assert [row["id"] for row in result["recent_trace"]] == ["c", "d"]
    assert result["recent_trace"][0]["parent_score"] == 2
    assert result["recent_trace"][0]["retained_best_before"] == 2
    assert result["current_parent"]["gap_to_retained_best"] == 2
    assert result["omitted"]["trace_by_recent_k"] == 2
    assert result["trace_window"]["selected_count"] == 2
    assert serialize_population_statistics(population) == serialize_population_statistics(
        reversed(population)
    )


def test_proxy_is_selected_globally_without_mixing_scales_or_parent_views():
    original_parent = program("p", 900, 0)
    population = [
        program("p", 900, 0, evoduet_score=-0.4),
        program("child", 500, 1, "p", evoduet_score=-0.3),
        program("missing_proxy", 99999, 2),
    ]
    result = read(population, parent=original_parent)
    assert result["score_key"] == "evoduet_score"
    assert result["scored_population_size"] == 2
    assert result["score_distribution"]["best"] == -0.3
    assert result["recent_trace"][-1]["score"] is None
    assert result["current_parent"]["score"] == -0.4
    assert result["current_parent"]["score_source"] == "retained_population"
    assert read(population, parent=program("outside", 1))["current_parent"]["score"] is None
    assert (
        read(population, parent=program("outside", 1, evoduet_score=-0.9))["current_parent"][
            "score"
        ]
        == -0.9
    )


def test_missing_retained_parent_score_does_not_fall_back_to_other_view():
    result = read(
        [program("p", 10), program("proxy", 20, evoduet_score=0.5)],
        parent=program("p", 10, evoduet_score=0.7),
    )
    assert result["current_parent"]["score"] is None


@pytest.mark.parametrize(
    "invalid", [True, False, float("nan"), float("inf"), -float("inf"), "3", None]
)
def test_invalid_proxy_does_not_select_proxy_key(invalid):
    result = read([program("p", 3, evoduet_score=invalid)])
    assert result["score_key"] == "combined_score"
    assert result["score_distribution"]["best"] == 3


def test_all_missing_scores_and_unavailable_lineage_are_explicit():
    result = read([program("a", True), program("b", float("nan"), 2, "evicted")])
    assert result["scored_population_size"] == 0
    assert result["missing_score_count"] == 2
    assert all(value is None for value in result["score_distribution"].values())
    assert result["top_programs"] == []
    assert result["recent_trace"][1]["parent_score"] is None
    assert result["recent_trace"][1]["delta"] is None
    assert result["recent_trace"][1]["outcome"] == "missing_score"
    assert result["selection_concentration"]["parent"]["most_selected_id"] == "evicted"
    assert "evicted" in result["scope"]["population"]


def test_strict_parent_improvement_is_distinct_from_retained_record():
    result = read(
        [
            program("best", 5, 0),
            program("parent", 1, 1, "best"),
            program("small_improvement", 1.000000000001, 2, "parent"),
            program("tie", 1, 3, "parent"),
            program("orphan", 6, 4, "evicted"),
        ],
        recent_k=3,
    )
    small, tie, orphan = result["recent_trace"]
    assert small["outcome"] == "improved"
    assert small["delta"] > 0
    assert small["global_outcome"] == "not_improved"
    assert small["retained_best_before"] == 5
    assert tie["outcome"] == "unchanged"
    assert orphan["outcome"] == "missing_parent_score"
    assert orphan["global_outcome"] == "improved"


def test_selection_concentration_uses_selected_window_and_unique_contexts():
    result = read(
        [
            program("a", 1, 0),
            program("b", 2, 1, "a", ["x", "x", "y"]),
            program("c", 3, 2, "a", ["x"]),
            program("d", 4, 3, "b", ["y"]),
        ],
        recent_k=3,
    )
    concentration = result["selection_concentration"]
    assert concentration["parent"]["most_selected_program_fraction"] == 2 / 3
    assert concentration["parent"]["selection_hhi"] == pytest.approx(5 / 9)
    assert concentration["context"]["selection_count"] == 4
    assert concentration["context"]["most_selected_id"] == "x"
    assert concentration["context"]["most_selected_program_fraction"] == 2 / 3
    assert concentration["context"]["selection_hhi"] == 0.5


def test_forbidden_fields_are_never_accessed_and_mapping_rows_work():
    class GuardedProgram:
        id = "guarded"
        iteration_found = 0
        parent_id = None
        other_context_ids = []
        metrics = {"combined_score": 1}

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected field access: {name}")

    result = read(
        [
            GuardedProgram(),
            {
                "id": "dict",
                "iteration_found": 1,
                "parent_id": "guarded",
                "metrics": {"combined_score": 2},
            },
        ]
    )
    assert result["recent_trace"][1]["outcome"] == "improved"


def test_budget_drops_oldest_trace_then_lowest_top_and_preserves_aggregates():
    population = [program(str(i), i, i, str(i - 1) if i else None) for i in range(100)]
    full = read(population)
    encoded = serialize_population_statistics(population, max_chars=5_000)
    limited = json.loads(encoded)
    assert len(encoded) <= 5_000
    assert len(full["top_programs"]) == 20
    assert limited["score_distribution"] == full["score_distribution"]
    assert limited["selection_concentration"] == full["selection_concentration"]
    assert limited["recent_trace"][-1]["id"] == "99"
    assert limited["omitted"]["trace_by_budget"] + len(limited["recent_trace"]) == 100
    assert limited["omitted"]["top_beyond_20"] == 80


def test_compact_800_character_budget_preserves_full_distribution():
    population = [program(f"id-{i}", i, i) for i in range(40)]
    encoded = serialize_population_statistics(
        population, max_chars=800, recent_k=10, parent=population[0]
    )
    result = json.loads(encoded)
    assert len(encoded) <= 800
    assert encoded == serialize_population_statistics(
        population, max_chars=800, recent_k=10, parent=population[0]
    )
    assert result["population_size"] == 40
    assert result["score_distribution"] == read(population)["score_distribution"]
    assert result["omitted"]["trace_by_recent_k"] == 30
    assert result["omitted"]["trace_by_budget"] == 10
    assert result["omitted"]["top_by_budget"] == 20
    assert result["omitted"]["details"] is True


def test_huge_identifiers_cannot_break_character_cap():
    member = program("x" * 10_000, 1, 1)
    encoded = serialize_population_statistics([member], max_chars=800, parent=member)
    result = json.loads(encoded)
    assert len(encoded) <= 800
    assert result["omitted"]["current_parent"] is True


@pytest.mark.parametrize("score_key", ["combined_score", "evoduet_score"])
def test_supported_512_character_budget_keeps_metric_best_and_counts(score_key):
    population = [program("parent", **{score_key: 0.12345678901234567})]
    encoded = serialize_population_statistics(population, max_chars=512, parent=population[0])
    result = json.loads(encoded)
    assert len(encoded) <= 512
    assert result["population_size"] == result["scored_population_size"] == 1
    assert result["missing_score_count"] == 0
    assert result["score_key"] == score_key
    assert result["direction"] == "higher_is_better"
    assert result["score_distribution"]["best"] == 0.12345678901234567
    assert result["omitted"]["trace_by_budget"] == 1
    assert result["omitted"]["top_by_budget"] == 1
    assert result["omitted"]["distribution_fields"] == 6


def test_finite_extreme_scores_do_not_emit_nonfinite_json():
    result = read([program("a", -1e308, 0), program("b", 1e308, 1, "a")])
    assert result["score_distribution"]["mean"] == 0
    assert result["score_distribution"]["population_std"] == 1e308
    assert result["recent_trace"][1]["delta"] is None


def test_empty_and_invalid_budgets():
    assert serialize_population_statistics(None) == ""
    assert serialize_population_statistics([]) == ""
    for budget in [0, 1, 100, -5, True]:
        with pytest.raises(ValueError):
            serialize_population_statistics([program("p", 1)], max_chars=budget)
    with pytest.raises(ValueError):
        serialize_population_statistics([program("p", 1)], recent_k=0)
