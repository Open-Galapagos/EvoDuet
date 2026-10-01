"""Read a stated probability without guessing or repairing the model's answer."""

import math
import re
from typing import Any

_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def parse_probability(text: Any) -> float:
    """Accept one finite number in [0, 1], optionally surrounded by whitespace."""
    if not isinstance(text, str) or not _NUMBER.fullmatch(text.strip()):
        raise ValueError("Expected exactly one numeric probability between 0 and 1.")
    probability = float(text.strip())
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Verbalized probability must be finite and between 0 and 1.")
    return probability


def chat_verbalized_probability(response: Any) -> float:
    """Require a complete, tool-free numeric answer with no generated reasoning."""
    choices = _value(response, "choices", []) or []
    if len(choices) != 1:
        raise ValueError("Provider must return exactly one probability answer.")
    choice = choices[0]
    if _value(choice, "finish_reason") not in (None, "stop"):
        raise ValueError("Provider did not complete the probability answer normally.")
    usage = _value(response, "usage", {})
    details = _value(usage, "completion_tokens_details", {})
    if (_value(details, "reasoning_tokens", 0) or 0) > 0:
        raise ValueError("Provider used hidden reasoning for the probability answer.")
    message = _value(choice, "message", {})
    if any(
        _value(message, name) for name in ("reasoning", "reasoning_content", "reasoning_details")
    ):
        raise ValueError("Provider generated reasoning for the probability answer.")
    if _value(message, "tool_calls") or _value(message, "function_call"):
        raise ValueError("Provider returned a tool call instead of a probability.")
    if _value(message, "refusal"):
        raise ValueError("Provider refused the probability assessment.")
    return parse_probability(_value(message, "content"))
