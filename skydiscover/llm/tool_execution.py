"""Shared execution helpers for LLM function tools."""

from __future__ import annotations

import json
from typing import Any, Dict


def limit_tool_output(content: Any, max_chars: int) -> str:
    """Bound a tool result before appending it to the LLM conversation."""
    limit = max(500, int(max_chars))
    serialized = (
        content
        if isinstance(content, str)
        else json.dumps(content, ensure_ascii=False, default=str)
    )
    if len(serialized) <= limit:
        return serialized

    try:
        parsed = json.loads(serialized)
    except (json.JSONDecodeError, TypeError):
        suffix = f"\n… [tool output truncated from {len(serialized)} characters]"
        return serialized[: max(0, limit - len(suffix))] + suffix

    if isinstance(parsed, dict):
        compacted = _fit_json_object(parsed, limit)
        if compacted is not None:
            return compacted

    return _json_preview(serialized, limit)


def _fit_json_object(output: Dict[str, Any], max_chars: int) -> str | None:
    """Compact common search-result fields while preserving valid JSON."""
    compact = dict(output)
    original_results = compact.get("results")
    if isinstance(original_results, list):
        compact_results = [
            dict(item) if isinstance(item, dict) else item for item in original_results
        ]
        compact["results"] = compact_results

        # Raw pages and images are typically much larger than result snippets.
        for item in compact_results:
            if isinstance(item, dict):
                item.pop("raw_content", None)
                item.pop("images", None)

        per_result_chars = max(300, max_chars // max(2 * len(compact_results), 1))
        for item in compact_results:
            if not isinstance(item, dict):
                continue
            item_content = item.get("content")
            if isinstance(item_content, str) and len(item_content) > per_result_chars:
                item["content"] = item_content[:per_result_chars] + "…"
    else:
        compact_results = None

    answer = compact.get("answer")
    if isinstance(answer, str) and len(answer) > max_chars // 4:
        compact["answer"] = answer[: max_chars // 4] + "…"

    compact["truncated"] = True
    serialized = _json_result(compact)
    while compact_results and len(serialized) > max_chars:
        compact_results.pop()
        serialized = _json_result(compact)

    return serialized if len(serialized) <= max_chars else None


def _json_preview(serialized: str, max_chars: int) -> str:
    """Return a valid JSON fallback for an otherwise unshrinkable result."""
    fallback = {
        "error": "Tool result exceeded the configured output limit.",
        "truncated": True,
        "result_preview": serialized[: max(1, max_chars - 200)],
    }
    fallback_json = _json_result(fallback)
    while len(fallback_json) > max_chars and fallback["result_preview"]:
        overflow = len(fallback_json) - max_chars
        fallback["result_preview"] = fallback["result_preview"][: -max(overflow, 1)]
        fallback_json = _json_result(fallback)
    return fallback_json


def _json_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
