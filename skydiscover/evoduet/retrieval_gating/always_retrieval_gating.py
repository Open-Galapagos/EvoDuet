"""Always-retrieve gating — fire a fresh search every iteration, no LLM.

Selected when ``config.evoduet.retrieval_gating_backend_type == "always"``.
Unconditionally returns ``retrieve``, so the
EvoDuet runs a search on every iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from skydiscover.evoduet.retrieval_gating.base_retrieval_gating import (
    BaseRetrievalGating,
    GateDecision,
)

if TYPE_CHECKING:
    from skydiscover.search.base_database import Program


class AlwaysRetrievalGating(BaseRetrievalGating):
    """Retrieve every iteration."""

    async def decide(
        self,
        *,
        parent: "Program",
        history: Any,
        search_store: Any,
        iteration: Optional[int] = None,
        task_context: str = "",
    ) -> GateDecision:
        return GateDecision.RETRIEVE
