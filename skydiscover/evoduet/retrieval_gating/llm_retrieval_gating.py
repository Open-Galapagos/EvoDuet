"""Analyze knowledge and choose retrieve, look-up, or no-op in one call."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

from skydiscover.evoduet.retrieval_gating.base_retrieval_gating import (
    BaseRetrievalGating,
    GateDecision,
    GateResult,
    _parse_gate_result,
)

if TYPE_CHECKING:
    from skydiscover.search.base_database import Program

logger = logging.getLogger("skydiscover.evoduet")

# Require the decision after the completed analysis block. JSON examples and
# gate-looking text inside the analysis must never become the actual decision.
_REPORT_AND_GATE = re.compile(
    r"\A\s*<knowledge_state_analysis>(?P<analysis>.*?)"
    r"</knowledge_state_analysis>\s*<gate[_ ]decision>(?P<gate>.*?)"
    r"</gate_decision>\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_REPORT = re.compile(
    r"\A\s*<knowledge_state_analysis>(.*?)</knowledge_state_analysis>",
    re.DOTALL | re.IGNORECASE,
)
_JSON_FENCE = re.compile(r"\A```(?:json)?\s*(.*?)\s*```\Z", re.DOTALL | re.IGNORECASE)


def _whole_json_object(text):
    """Read only a whole JSON object, optionally enclosed in one code fence."""
    candidate = text.strip()
    fence = _JSON_FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group(1)
    try:
        value = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _validated_result(obj):
    # The shared validator also searches for Markdown fences. Escape backticks
    # inside JSON strings so a quoted example cannot be mistaken for its wrapper.
    return _parse_gate_result(json.dumps(obj).replace("`", r"\u0060"))


def parse_gate_result(text: str) -> GateResult:
    """Read the integrated knowledge and decision from one JSON object.

    Accept plain or Markdown-fenced JSON; tagged reports remain supported for
    earlier runs. Truncated or malformed responses fail closed to no-op, while
    retaining a complete tagged report when available. ID validation is unchanged.
    """
    obj = _whole_json_object(text)
    if obj is not None:
        return _validated_result(obj)

    blocks = _REPORT_AND_GATE.fullmatch(text)
    if blocks:
        knowledge = blocks.group("analysis").strip()
        obj = _whole_json_object(blocks.group("gate"))
        if obj is not None:
            # The outer report is authoritative; ignore any JSON knowledge field.
            return replace(_validated_result(obj), knowledge_state_analysis=knowledge)
    else:
        report = _REPORT.match(text)
        knowledge = report.group(1).strip() if report else ""

    logger.warning("retrieval gating: malformed report/decision output; defaulting to no-op")
    return GateResult(GateDecision.NO_OP, knowledge_state_analysis=knowledge)


class LLMRetrievalGating(BaseRetrievalGating):
    """Use the current solution model for the integrated retrieval decision."""

    def __init__(self, config, *, llm):
        super().__init__(config)
        self.llm = llm

    async def _generate(self, system: str, prompt: str, *, label: str, iteration=None) -> str:
        response = await self.llm.generate(
            system,
            [{"role": "user", "content": prompt}],
            llm_context={"iteration": iteration, "phase": label},
        )
        return getattr(response, "text", "") or ""

    async def decide(
        self,
        *,
        parent: "Program",
        history: Any,
        search_store: Any,
        iteration: Optional[int] = None,
        task_context: str = "",
    ) -> GateDecision:
        return (
            await self.decide_with_details(
                parent=parent,
                history=history,
                search_store=search_store,
                iteration=iteration,
                task_context=task_context,
            )
        ).decision

    def _build_prompt(self, parent, history, search_store, task_context=""):
        template_name = self.config.evoduet.retrieval_gating_prompt_template_name
        if not template_name:
            raise ValueError("retrieval gating requires a prompt template name")
        template = self._read_template(template_name)
        fields = {
            "evolutionary_history": self._render_history(history),
            "parent_program": (
                parent.get("solution", "")
                if isinstance(parent, Mapping)
                else getattr(parent, "solution", "")
            )
            or "",
            "search_database": self._render_search_database(search_store),
        }
        return re.sub(r"\{(\w+)\}", lambda m: fields.get(m.group(1), m.group(0)), template)

    async def decide_with_details(
        self, *, parent, history, search_store, iteration=None, task_context=""
    ):
        prompt = self._build_prompt(parent, history, search_store)
        text = await self._generate("", prompt, label="wk-retrieval-gate", iteration=iteration)
        return parse_gate_result(text)
