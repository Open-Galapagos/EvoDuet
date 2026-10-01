"""Stateful retrieval trigger matching EvoX's score-stagnation rule.

The caller supplies the global population best score in the scaffold's
higher-is-better scale. Parent scores and retrieved documents do not determine
this gate. It makes no LLM calls and emits only ``no-op`` or ``retrieve``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from typing import Any

from skydiscover.evoduet.retrieval_gating.base_retrieval_gating import (
    BaseRetrievalGating,
    GateDecision,
    GateResult,
)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _score(value: Any, *, allow_missing: bool) -> float | None:
    if value is None and allow_missing:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("best_score must be a real number or None")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        if allow_missing:
            return None
        raise ValueError("saved scores must be finite")
    return number


class EvoXStagnationRetrievalGating(BaseRetrievalGating):
    """Retrieve after consecutive improvements of at most 0.01.

    The first finite observation establishes the baseline. Each later finite
    observation compares against the immediately preceding finite score;
    small improvements therefore accumulate *stagnation*, not improvement.
    Repeated calls for one iteration reuse its decision, including after resume.
    """

    IMPROVEMENT_THRESHOLD = 0.01
    STATE_VERSION = 1

    def __init__(self, config):
        super().__init__(config)
        budget = _score(getattr(config, "max_iterations", None), allow_missing=False)
        if budget <= 0:
            raise ValueError("config.max_iterations must be finite and positive")
        configured_window = getattr(getattr(config, "search", None), "switch_interval", None)
        self.switch_interval = (
            max(1, int(budget * 0.10))
            if configured_window is None
            else _integer(configured_window, "config.search.switch_interval", 1)
        )
        self._last_best_score: float | None = None
        self._stagnant_count = 0
        self._last_iteration: int | None = None
        self._last_result: GateResult | None = None
        self._diagnostics: dict[str, Any] = {}

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return an independent snapshot of the latest observation and decision."""
        return deepcopy(self._diagnostics)

    @classmethod
    def _evaluate(cls, iteration, current, previous, count_before, window):
        count_at_decision = count_before
        improvement = None
        status = "observed"
        decision = GateDecision.NO_OP
        if current is None:
            status = "missing_score"
            reasoning = "No finite population best score; preserve the stagnation state."
        elif previous is None:
            status = "baseline"
            count_at_decision = 0
            reasoning = "First finite population best score establishes the baseline."
        else:
            improvement = current - previous
            if improvement > cls.IMPROVEMENT_THRESHOLD:
                count_at_decision = 0
                reasoning = "Best-score improvement exceeds 0.01; reset the stagnation counter."
            else:
                count_at_decision += 1
                reasoning = f"Stagnation count {count_at_decision}/{window}."
                if count_at_decision >= window:
                    decision = GateDecision.RETRIEVE
                    reasoning += " Retrieve and reset the stagnation counter."

        count_after = 0 if decision is GateDecision.RETRIEVE else count_at_decision
        # Extreme finite operands can overflow on subtraction. Preserve EvoX's
        # comparison above while keeping the audit record strict JSON-compatible.
        overflow = improvement is not None and not math.isfinite(improvement)
        diagnostics = {
            "iteration": iteration,
            "current_best_score": current,
            "previous_best_score": previous,
            "improvement": None if overflow else improvement,
            "improvement_overflow": overflow,
            "switch_interval": window,
            "improvement_threshold": cls.IMPROVEMENT_THRESHOLD,
            "stagnant_count_before": count_before,
            "stagnant_count_at_decision": count_at_decision,
            "stagnant_count": count_after,
            "status": status,
            "decision": decision.value,
        }
        return (
            GateResult(decision, reasoning=reasoning),
            diagnostics,
            previous if current is None else current,
            count_after,
        )

    async def decide_with_details(
        self,
        *,
        parent=None,
        history=None,
        search_store=None,
        iteration: int,
        task_context: str = "",
        best_score: float | None = None,
    ) -> GateResult:
        iteration = _integer(iteration, "iteration")
        if self._last_iteration is not None:
            if iteration < self._last_iteration:
                raise ValueError(
                    "iteration precedes the last gate observation; restore its checkpoint"
                )
            if iteration == self._last_iteration:
                return self._last_result
        current = _score(best_score, allow_missing=True)
        result, diagnostics, score, count = self._evaluate(
            iteration, current, self._last_best_score, self._stagnant_count, self.switch_interval
        )
        self._last_best_score = score
        self._stagnant_count = count
        self._last_iteration = iteration
        self._last_result = result
        self._diagnostics = diagnostics
        return result

    async def decide(self, **kwargs) -> GateDecision:
        return (await self.decide_with_details(**kwargs)).decision

    def state_dict(self) -> dict[str, Any]:
        """Persist the counter, preceding score, and last iteration's cached result."""
        return {
            "version": self.STATE_VERSION,
            "switch_interval": self.switch_interval,
            "improvement_threshold": self.IMPROVEMENT_THRESHOLD,
            "last_best_score": self._last_best_score,
            "stagnant_count": self._stagnant_count,
            "last_iteration": self._last_iteration,
            "last_result": (
                {
                    "decision": self._last_result.decision.value,
                    "reasoning": self._last_result.reasoning,
                }
                if self._last_result is not None
                else None
            ),
            "diagnostics": self.diagnostics,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore atomically, rejecting malformed or internally inconsistent state.

        The saved window is authoritative, so resuming with a larger overall
        iteration budget does not silently change the trigger interval.
        """
        if not isinstance(state, Mapping) or set(state) != set(self.state_dict()):
            raise ValueError("invalid stagnation gate state fields")
        if _integer(state["version"], "version", 1) != self.STATE_VERSION:
            raise ValueError("unsupported stagnation gate state version")
        threshold = _score(state["improvement_threshold"], allow_missing=False)
        if threshold != self.IMPROVEMENT_THRESHOLD:
            raise ValueError("saved improvement threshold differs from EvoX's fixed 0.01")
        window = _integer(state["switch_interval"], "switch_interval", 1)
        count = _integer(state["stagnant_count"], "stagnant_count")
        if count >= window:
            raise ValueError("saved stagnation counter must be below the trigger window")
        score = state["last_best_score"]
        if score is not None:
            score = _score(score, allow_missing=False)
        iteration = state["last_iteration"]
        diagnostics = deepcopy(state["diagnostics"])
        result = None
        if iteration is None:
            if (
                score is not None
                or count != 0
                or state["last_result"] is not None
                or diagnostics != {}
            ):
                raise ValueError("unobserved gate state must be empty")
        else:
            iteration = _integer(iteration, "last_iteration")
            if not isinstance(diagnostics, dict):
                raise ValueError("saved diagnostics must be an object")
            required = {
                "iteration",
                "current_best_score",
                "previous_best_score",
                "improvement",
                "improvement_overflow",
                "switch_interval",
                "improvement_threshold",
                "stagnant_count_before",
                "stagnant_count_at_decision",
                "stagnant_count",
                "status",
                "decision",
            }
            if set(diagnostics) != required:
                raise ValueError("invalid saved diagnostic fields")
            for key in (
                "iteration",
                "stagnant_count_before",
                "stagnant_count_at_decision",
                "stagnant_count",
            ):
                _integer(diagnostics[key], key)
            _integer(diagnostics["switch_interval"], "diagnostic switch_interval", 1)
            for key in ("current_best_score", "previous_best_score", "improvement"):
                if diagnostics[key] is not None:
                    _score(diagnostics[key], allow_missing=False)
            _score(diagnostics["improvement_threshold"], allow_missing=False)
            if not isinstance(diagnostics["improvement_overflow"], bool):
                raise ValueError("improvement_overflow must be a boolean")
            previous = diagnostics["previous_best_score"]
            count_before = diagnostics["stagnant_count_before"]
            if count_before >= window or (previous is None and count_before != 0):
                raise ValueError("invalid saved preceding stagnation state")
            result, expected, expected_score, expected_count = self._evaluate(
                iteration, diagnostics["current_best_score"], previous, count_before, window
            )
            if (
                diagnostics != expected
                or score != expected_score
                or count != expected_count
                or state["last_result"]
                != {
                    "decision": result.decision.value,
                    "reasoning": result.reasoning,
                }
            ):
                raise ValueError("saved stagnation state does not match its cached decision")

        self.switch_interval = window
        self._last_best_score = score
        self._stagnant_count = count
        self._last_iteration = iteration
        self._last_result = result
        self._diagnostics = diagnostics
