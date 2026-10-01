"""EvoX stagnation parity, retry idempotence, and strict checkpoint restoration."""

import json
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from skydiscover.evoduet.retrieval_gating.base_retrieval_gating import GateDecision
from skydiscover.evoduet.stagnation_gating import EvoXStagnationRetrievalGating
from skydiscover.search.evox.controller import CoEvolutionController


def make_gate(window=None, budget=100):
    return EvoXStagnationRetrievalGating(
        SimpleNamespace(max_iterations=budget, search=SimpleNamespace(switch_interval=window))
    )


@pytest.mark.parametrize(("budget", "expected"), [(1, 1), (9, 1), (19, 1), (100, 10), (250, 25)])
def test_default_window_matches_evox_budget_rule(budget, expected):
    assert make_gate(budget=budget).switch_interval == expected


def test_explicit_window_and_no_analysis_or_model_requirement():
    gate = make_gate(window=7)
    assert gate.switch_interval == 7
    assert not hasattr(gate, "llm")


@pytest.mark.parametrize("window", [0, -1, True, 2.0, "2", math.inf, math.nan])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError):
        make_gate(window=window)


@pytest.mark.parametrize("budget", [0, -1, True, None, "100", math.inf, -math.inf, math.nan])
def test_invalid_budget_rejected_even_with_window_override(budget):
    with pytest.raises(ValueError):
        make_gate(window=3, budget=budget)


@pytest.mark.asyncio
@pytest.mark.parametrize("window", [1, 3, 10])
async def test_decisions_and_counters_match_actual_evox_on_finite_observations(window):
    gate = make_gate(window=window)
    evox = object.__new__(CoEvolutionController)
    evox._meta_llm_available = True
    evox._last_tracked_best_score = None
    evox._stagnant_count = 0
    evox._switch_interval = window
    scores = [0.0, 0.005, 0.010, 0.015, 0.01, 0.03, 0.03, 0.04] + [0.04] * 25
    for iteration, score in enumerate(scores):
        evox._get_best_score = lambda: score
        expected = evox._should_evolve_search()
        actual = await gate.decide(iteration=iteration, best_score=score)
        assert (actual is GateDecision.RETRIEVE) == expected
        assert gate.diagnostics["stagnant_count"] == evox._stagnant_count
        assert gate.state_dict()["last_best_score"] == evox._last_tracked_best_score


@pytest.mark.asyncio
async def test_first_finite_observation_is_baseline_and_small_gains_trigger():
    gate = make_gate(window=3)
    decisions = [
        await gate.decide(iteration=i, best_score=score)
        for i, score in enumerate([0.0, 0.005, 0.010, 0.015])
    ]
    assert decisions == [GateDecision.NO_OP] * 3 + [GateDecision.RETRIEVE]
    assert gate.diagnostics["stagnant_count_before"] == 2
    assert gate.diagnostics["stagnant_count_at_decision"] == 3
    assert gate.diagnostics["stagnant_count"] == 0
    assert gate.diagnostics["previous_best_score"] == 0.010
    assert gate.state_dict()["last_best_score"] == 0.015


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gain", "expected"),
    [
        (math.nextafter(0.01, -math.inf), GateDecision.RETRIEVE),
        (0.01, GateDecision.RETRIEVE),
        (math.nextafter(0.01, math.inf), GateDecision.NO_OP),
    ],
)
async def test_threshold_uses_strict_float_comparison(gain, expected):
    gate = make_gate(window=1)
    assert await gate.decide(iteration=0, best_score=0.0) is GateDecision.NO_OP
    assert gate.diagnostics["status"] == "baseline"
    assert await gate.decide(iteration=1, best_score=gain) is expected


@pytest.mark.asyncio
async def test_large_improvement_resets_and_retrieval_restarts_window():
    gate = make_gate(window=2)
    expected = [GateDecision.NO_OP] * 4 + [
        GateDecision.RETRIEVE,
        GateDecision.NO_OP,
        GateDecision.RETRIEVE,
    ]
    actual = [
        await gate.decide(iteration=i, best_score=score)
        for i, score in enumerate([1.0, 1.0, 1.2, 1.2, 1.2, 1.2, 1.2])
    ]
    assert actual == expected


@pytest.mark.asyncio
async def test_missing_and_nonfinite_scores_preserve_previous_finite_baseline_and_counter():
    gate = make_gate(window=2)
    await gate.decide(iteration=0, best_score=None)
    assert gate.state_dict()["last_best_score"] is None
    assert gate.diagnostics["status"] == "missing_score"
    await gate.decide(iteration=1, best_score=1.0)
    assert gate.diagnostics["status"] == "baseline"
    await gate.decide(iteration=2, best_score=1.0)
    for i, score in enumerate([None, math.nan, math.inf, -math.inf], start=3):
        assert await gate.decide(iteration=i, best_score=score) is GateDecision.NO_OP
        assert gate.state_dict()["last_best_score"] == 1.0
        assert gate.diagnostics["stagnant_count"] == 1
        assert gate.diagnostics["current_best_score"] is None
        json.dumps(gate.state_dict(), allow_nan=False)
    assert await gate.decide(iteration=7, best_score=1.0) is GateDecision.RETRIEVE


@pytest.mark.asyncio
async def test_retries_reuse_result_without_double_counting_or_changing_audit():
    gate = make_gate(window=1)
    await gate.decide(iteration=1, best_score=1.0)
    first = await gate.decide_with_details(iteration=2, best_score=1.0)
    saved = gate.state_dict()
    assert first.decision is GateDecision.RETRIEVE
    assert await gate.decide_with_details(iteration=2, best_score=99.0) is first
    assert gate.state_dict() == saved
    with pytest.raises(ValueError, match="precedes"):
        await gate.decide(iteration=1, best_score=1.0)
    assert gate.state_dict() == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", [None, -1, True, 1.0, "1"])
async def test_invalid_iteration_does_not_mutate_state(iteration):
    gate = make_gate()
    saved = gate.state_dict()
    with pytest.raises(ValueError):
        await gate.decide(iteration=iteration, best_score=1.0)
    assert gate.state_dict() == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [True, "1.0", {}, [], complex(1.0)])
async def test_invalid_score_type_does_not_mutate_state(score):
    gate = make_gate()
    saved = gate.state_dict()
    with pytest.raises(ValueError):
        await gate.decide(iteration=0, best_score=score)
    assert gate.state_dict() == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_after", range(7))
async def test_json_checkpoint_resume_preserves_future_and_cached_decisions(resume_after):
    gate = make_gate(window=2)
    scores = [None, 0.0, 0.0, 0.0, None, 0.005, 0.005]
    for iteration in range(resume_after + 1):
        result = await gate.decide_with_details(iteration=iteration, best_score=scores[iteration])
    state = json.loads(json.dumps(gate.state_dict(), allow_nan=False))
    restored = make_gate(window=99)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.switch_interval == 2
    assert await restored.decide_with_details(iteration=resume_after, best_score=999.0) == result
    assert restored.state_dict() == state
    for iteration in range(resume_after + 1, len(scores)):
        assert await restored.decide(
            iteration=iteration, best_score=scores[iteration]
        ) == await gate.decide(iteration=iteration, best_score=scores[iteration])
        assert restored.state_dict() == gate.state_dict()


def test_empty_checkpoint_roundtrip_and_independent_snapshots():
    gate = make_gate(window=2)
    restored = make_gate(window=5)
    restored.load_state_dict(json.loads(json.dumps(gate.state_dict())))
    assert restored.state_dict() == gate.state_dict()
    assert restored.diagnostics == {}


@pytest.mark.asyncio
async def test_diagnostics_and_state_do_not_expose_mutable_internal_records():
    gate = make_gate(window=2)
    await gate.decide(iteration=1, best_score=1.0)
    pristine = gate.state_dict()
    diagnostic = gate.diagnostics
    diagnostic["stagnant_count"] = 900
    state = gate.state_dict()
    state["diagnostics"]["stagnant_count"] = 800
    state["last_result"]["decision"] = "look-up"
    assert gate.state_dict() == pristine
    restored = make_gate()
    restored.load_state_dict(pristine)
    pristine["diagnostics"]["status"] = "changed"
    assert restored.state_dict() == gate.state_dict()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("version",), 2),
        (("version",), True),
        (("switch_interval",), 0),
        (("improvement_threshold",), 0.02),
        (("last_best_score",), math.nan),
        (("last_best_score",), 5.0),
        (("stagnant_count",), -1),
        (("stagnant_count",), 2),
        (("stagnant_count",), True),
        (("last_iteration",), None),
        (("last_iteration",), 0),
        (("last_result", "decision"), "look-up"),
        (("last_result", "reasoning"), "changed"),
        (("diagnostics",), []),
        (("diagnostics", "current_best_score"), math.inf),
        (("diagnostics", "previous_best_score"), None),
        (("diagnostics", "improvement"), 1.0),
        (("diagnostics", "improvement_overflow"), 0),
        (("diagnostics", "stagnant_count_before"), 1),
        (("diagnostics", "stagnant_count_at_decision"), 0),
        (("diagnostics", "status"), "baseline"),
    ],
)
async def test_corrupt_checkpoint_rejected_atomically(path, value):
    gate = make_gate(window=2)
    await gate.decide(iteration=1, best_score=1.0)
    await gate.decide(iteration=2, best_score=1.0)
    pristine = gate.state_dict()
    corrupted = deepcopy(pristine)
    target = corrupted
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        gate.load_state_dict(corrupted)
    assert gate.state_dict() == pristine


@pytest.mark.parametrize("state", [None, [], {}, {"version": 1}])
def test_invalid_state_shape_rejected(state):
    gate = make_gate()
    pristine = gate.state_dict()
    with pytest.raises(ValueError):
        gate.load_state_dict(state)
    assert gate.state_dict() == pristine


@pytest.mark.asyncio
async def test_extreme_finite_scores_follow_evox_comparison_and_remain_strict_json():
    gate = make_gate(window=1)
    await gate.decide(iteration=0, best_score=-1e308)
    assert await gate.decide(iteration=1, best_score=1e308) is GateDecision.NO_OP
    assert gate.diagnostics["improvement_overflow"] is True
    restored = make_gate()
    restored.load_state_dict(json.loads(json.dumps(gate.state_dict(), allow_nan=False)))
    assert await restored.decide(iteration=2, best_score=-1e308) is GateDecision.RETRIEVE
    json.dumps(restored.state_dict(), allow_nan=False)
