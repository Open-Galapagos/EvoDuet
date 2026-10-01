"""Base retrieval gating (G) — decides per-iteration information-seeking behavior.

    g_t ~ G(H_t, x_t, D_search; psi),   g_t in {no-op, look-up, retrieve}

``BaseRetrievalGating`` holds the shared ``GateDecision`` enum, prompt rendering
(from ``prompts/retrieval_gating.txt``), and JSON-decision parsing. Concrete
gates (``llm`` / ``always`` / ``heuristic`` / ``stagnation``) implement ``decide``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from skydiscover.search.base_database import Program

logger = logging.getLogger("skydiscover.evoduet")

# Prompt templates (.txt) live at evoduet/prompts (one level up).
_PROMPTS_DIR = str(Path(__file__).parent.parent / "prompts")

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class GateDecision(str, Enum):
    """Outcome of retrieval gating for a single iteration (``g_t``)."""

    NO_OP = "no-op"  # stay in the main loop; no information-seeking this iteration
    LOOK_UP = "look-up"  # reuse selected documents from the existing search database
    RETRIEVE = "retrieve"  # run a fresh search (query construction -> web search)


@dataclass(frozen=True)
class GateResult:
    """One gate response, including cached evidence IDs and its knowledge analysis."""

    decision: GateDecision
    search_document_ids: tuple[str, ...] = ()
    knowledge_state_analysis: str = ""
    reasoning: str = ""


def _extract_json_object(text: str):
    """Pull a JSON object out of the LLM's text (handles ``` fences / prose)."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = (fence.group(1) if fence else text).strip()
    candidates = [candidate]
    m = _JSON_OBJECT_RE.search(candidate)
    if m:
        candidates.append(m.group(0))
    for s in candidates:
        try:
            data = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _decision_from_object(obj: Any, text: str) -> GateDecision:
    if not obj or "decision" not in obj:
        logger.warning("gating: no parseable decision; defaulting to no-op. Raw: %s", text[:200])
        return GateDecision.NO_OP
    value = str(obj["decision"]).strip().strip('"').lower().replace("_", "-").replace(" ", "-")
    try:
        return GateDecision(value)
    except ValueError:
        logger.warning("gating: unknown decision %r; defaulting to no-op", value)
        return GateDecision.NO_OP


def _parse_gate_result(text: str) -> GateResult:
    """Validate lookup IDs without changing their spelling or trusting other fields."""
    obj = _extract_json_object(text)
    decision = _decision_from_object(obj, text)
    if obj is None:
        return GateResult(decision)
    knowledge = obj.get("knowledge_state_analysis")
    reasoning = obj.get("reasoning")
    document_ids: tuple[str, ...] = ()
    if decision is GateDecision.LOOK_UP:
        ids = obj.get("search_document_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids)
        ):
            logger.warning("gating: look-up requires a non-empty list of document IDs; using no-op")
            decision = GateDecision.NO_OP
        else:
            document_ids = tuple(dict.fromkeys(ids))
    return GateResult(
        decision=decision,
        search_document_ids=document_ids,
        knowledge_state_analysis=knowledge if isinstance(knowledge, str) else "",
        reasoning=reasoning if isinstance(reasoning, str) else "",
    )


class BaseRetrievalGating:
    """Common interface and prompt inputs for retrieval gates."""

    def __init__(self, config):
        self.config = config

    async def decide(
        self,
        *,
        parent: "Program",
        history: Any,
        search_store: Any,
        iteration: Optional[int] = None,
        task_context: str = "",
    ) -> GateDecision:
        """Return the gate decision ``g_t`` for this iteration."""
        raise NotImplementedError

    async def decide_with_details(
        self,
        *,
        parent: "Program",
        history: Any,
        search_store: Any,
        iteration: Optional[int] = None,
        task_context: str = "",
    ) -> GateResult:
        """Keep legacy/custom gates working when callers request structured results."""
        return GateResult(
            await self.decide(
                parent=parent,
                history=history,
                search_store=search_store,
                iteration=iteration,
                task_context=task_context,
            )
        )

    @staticmethod
    def _read_template(name: str) -> str:
        filename = name if name.endswith(".txt") else f"{name}.txt"
        return Path(_PROMPTS_DIR, filename).read_text()

    @staticmethod
    def _render_history(history: Any) -> str:
        """Render the evolutionary history into prompt text."""
        from skydiscover.evoduet.history import verbalize_history

        return verbalize_history(history)

    @staticmethod
    def _render_search_database(search_store: Any) -> str:
        """Render the policy-selected D_search into prompt text.

        Accepts either the already-verbalized text (as the layer passes it) or a
        SearchStore (selected records rendered as search experiences)."""
        if isinstance(search_store, str):
            return search_store or "(empty — no prior searches)"
        if hasattr(search_store, "search_database"):
            return str(search_store.search_database or "(empty — no prior searches)")
        if hasattr(search_store, "select_records"):
            from skydiscover.evoduet.search_context import render_search_experience

            return render_search_experience(search_store)
        return "(empty — no prior searches)"
