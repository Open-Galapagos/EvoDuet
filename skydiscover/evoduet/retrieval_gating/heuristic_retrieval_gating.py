"""Heuristic retrieval gating — random on/off, no LLM.

Selected when ``config.evoduet.retrieval_gating_backend_type == "heuristic"``.
With probability ``config.evoduet.gating_retrieve_probability`` it
returns ``retrieve`` (turn information-seeking on); otherwise ``no-op``.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Optional

from skydiscover.evoduet.retrieval_gating.base_retrieval_gating import (
    BaseRetrievalGating,
    GateDecision,
)

if TYPE_CHECKING:
    from skydiscover.search.base_database import Program


class HeuristicRetrievalGating(BaseRetrievalGating):
    """Turn retrieval on with a fixed probability; otherwise no-op."""

    def __init__(self, config):
        super().__init__(config)
        wk = config.evoduet
        self.retrieve_probability = float(wk.gating_retrieve_probability)
        self.seed = int(getattr(wk, "random_seed", 0))

    async def decide(
        self,
        *,
        parent: "Program",
        history: Any,
        search_store: Any,
        iteration: Optional[int] = None,
        task_context: str = "",
    ) -> GateDecision:
        draw = random.Random(f"{self.seed}:{iteration}").random()
        if draw < self.retrieve_probability:
            return GateDecision.RETRIEVE
        return GateDecision.NO_OP
