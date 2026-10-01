"""WKL calls routed to the current iteration's solution model."""

from __future__ import annotations

import copy
from contextvars import ContextVar

from skydiscover.llm.llm_pool import LLMPool

active_solution_model = ContextVar("evoduet_solution_model", default=None)


def without_tools(model):
    """Preserve the solution model's transport and generation settings."""
    model = copy.deepcopy(model)
    model.tools = []
    model.tool_choice = None
    model.max_tool_rounds = 0
    return model


class SolutionModelPool:
    """Share one text-only client per solution model across all WKL stages.

    The controller selects the solution model before retrieval. A ContextVar
    keeps concurrent iterations isolated; standalone WKL calls use the first
    configured solution model.
    """

    def __init__(self, models, evoduet=None):
        if not models:
            raise ValueError("EvoDuet requires llm.models")
        self.models_cfg = list(models)
        self.evoduet = evoduet
        self.pools = []
        for model in self.models_cfg:
            model = without_tools(model)
            # Selection already happened in the solution pool.
            model.weight = 1.0
            self.pools.append(LLMPool([model]))
        self.models = [model for pool in self.pools for model in pool.models]

    async def generate(self, system_message, messages, **kwargs):
        selected = active_solution_model.get()
        index = 0
        if selected is not None:
            index = next(
                (
                    number
                    for number, model in enumerate(self.models_cfg)
                    if model is selected or model == selected
                ),
                None,
            )
            if index is None:
                raise ValueError("WKL solution_model must belong to llm.models")
        phase = (kwargs.get("llm_context") or {}).get("phase") or ""
        if phase == "wk-query" or phase.startswith("wk-query-sample-"):
            kwargs.setdefault(
                "max_tokens",
                getattr(self.evoduet, "query_construction_max_tokens", 32_768),
            )
        # Population analysis uses EvoX's system instructions. Other stages
        # retain the user-only contract even if a system role is supplied.
        system = system_message if phase == "wk-population-analysis" else ""
        return await self.pools[index].generate(system, messages, **kwargs)

    async def _generate(self, system, prompt, *, label, iteration=None):
        response = await self.generate(
            system,
            [{"role": "user", "content": prompt}],
            llm_context={"iteration": iteration, "phase": label},
        )
        return getattr(response, "text", "") or ""
