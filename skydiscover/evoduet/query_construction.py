"""Generate and validate one bounded search query per model sample."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConstructedQuery:

    query: Optional[str] = None
    keywords: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    query_type: Optional[str] = None

    id: Optional[str] = None  # stable id for this query
    # Usage history — same shape as SearchResult; appended each time this query's
    # results are injected and the resulting child solution is evaluated.
    visit_count: int = 0  # times this query's results were used in prompt construction
    evolution_iterations: List[int] = field(default_factory=list)
    evolution_scores: List[float] = field(default_factory=list)
    evolution_feedbacks: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return ((self.query or "").strip() or " ".join(self.keywords or []).strip())[:500]


class QueryConstruction:
    """Text generation and strict parsing for the observed-search loop."""

    def __init__(self, llm):
        self.llm = llm

    async def _generate(self, system: str, prompt: str, *, label: str, iteration=None) -> str:
        response = await self.llm.generate(
            system,
            [{"role": "user", "content": prompt}],
            llm_context={"iteration": iteration, "phase": label},
        )
        return getattr(response, "text", "") or ""

    @staticmethod
    def _parse_query_sample(text: str) -> ConstructedQuery:
        """Parse one sample strictly, never extracting an object from an array."""

        def reject_constant(value):
            raise ValueError(f"nonstandard JSON constant: {value}")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        if not isinstance(text, str):
            raise ValueError("query sample must be text containing one JSON object")
        text = text.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            text = fence[1]
        try:
            row = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
        except (ValueError, TypeError) as exc:
            raise ValueError("query sample must contain one valid JSON object") from exc
        if not isinstance(row, dict):
            raise ValueError("query sample must contain one JSON object")
        query = QueryConstruction._from_object(row)
        if not query.text:
            raise ValueError("query sample contains no usable query")
        return query

    @staticmethod
    def _from_object(obj: dict) -> ConstructedQuery:
        def strings(value):
            if not isinstance(value, list):
                return None
            items = [
                item.strip()[:100] for item in value[:20] if isinstance(item, str) and item.strip()
            ]
            return items or None

        query = obj.get("query")
        query_type = obj.get("query_type") or obj.get("query_intent")
        return ConstructedQuery(
            query=(query.strip()[:500] or None) if isinstance(query, str) else None,
            keywords=strings(obj.get("keywords")),
            resources=strings(obj.get("resources")),
            query_type=(query_type.strip()[:500] or None) if isinstance(query_type, str) else None,
        )
