"""JSON-safe capture helpers for provider response metadata and reasoning."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def json_safe(value: Any) -> Any:
    """Convert SDK/Pydantic response objects into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(mode="json", exclude_none=True))
        except TypeError:
            return json_safe(value.model_dump(exclude_none=True))
    if hasattr(value, "__dict__"):
        return {
            str(key): json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def response_reasoning(message: Any) -> Dict[str, Any]:
    """Return every reasoning field exposed by an OpenAI-compatible message."""
    if message is None:
        return {}

    dumped = json_safe(message)
    dumped = dumped if isinstance(dumped, dict) else {}
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        dumped = {**json_safe(model_extra), **dumped}

    reasoning: Dict[str, Any] = {}
    for key in ("reasoning", "reasoning_content", "reasoning_details"):
        value = dumped.get(key)
        if value not in (None, "", [], {}):
            reasoning[key] = value
    return reasoning


def reasoning_content(calls: Iterable[Dict[str, Any]]) -> str | None:
    """Flatten available plaintext reasoning while retaining structured data elsewhere."""
    parts: List[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text and text not in seen:
                parts.append(text)
                seen.add(text)
            return
        if isinstance(value, dict):
            for key in ("text", "content", "summary", "reasoning", "reasoning_content"):
                if key in value:
                    add(value[key])
            return
        if isinstance(value, list):
            for item in value:
                add(item)

    for call in calls:
        if not isinstance(call, dict):
            continue
        for response in call.get("responses", []):
            if not isinstance(response, dict):
                continue
            add(response.get("reasoning"))
            add(response.get("reasoning_content"))
            add(response.get("reasoning_details"))

    return "\n\n".join(parts) or None
