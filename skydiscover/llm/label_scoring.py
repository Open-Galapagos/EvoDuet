"""Extract label likelihoods without substituting absent top-k probabilities."""

import math
from typing import Any, Dict


def finite_json(value: Any) -> Any:
    """Keep unavailable provider logprobs JSON null, never NaN or Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def confidence_from_logprobs(logprob_true: float, logprob_false: float) -> Dict[str, float]:
    """Normalize the two full-label likelihoods in log space."""
    values = (logprob_true, logprob_false)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value > 0
        or value == -9999.0  # OpenAI's sentinel for an unavailable probability.
        for value in values
    ):
        raise ValueError("Both label logprobs must be finite, available log probabilities.")
    maximum = max(values)
    true_weight = math.exp(logprob_true - maximum)
    false_weight = math.exp(logprob_false - maximum)
    log_mass = maximum + math.log(true_weight + false_weight)
    if log_mass > 1e-6:
        raise ValueError("True and False probabilities sum to more than one.")
    return {
        "confidence": true_weight / (true_weight + false_weight),
        "logprob_true": float(logprob_true),
        "logprob_false": float(logprob_false),
        "label_probability_mass": min(1.0, math.exp(log_mass)),
        "log_label_probability_mass": log_mass,
    }


def _value(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def chat_label_logprobs(response: Any) -> Dict[str, float]:
    """Read two exact single-token labels from the first output position only.

    Seeing a complete label in one token's decoded text establishes its
    single-token representation. A missing label can be outside top-k or span
    multiple tokens; neither case permits treating its probability as zero.
    """
    choices = _value(response, "choices", []) or []
    if not choices:
        raise ValueError("Provider returned no choices.")
    usage = _value(response, "usage", {})
    details = _value(usage, "completion_tokens_details", {})
    if (_value(details, "reasoning_tokens", 0) or 0) > 0:
        raise ValueError("Provider used hidden reasoning before the fixed label position.")
    message = _value(choices[0], "message", {})
    if any(
        _value(message, name) for name in ("reasoning", "reasoning_content", "reasoning_details")
    ):
        raise ValueError("Provider generated reasoning before the fixed label position.")
    if _value(message, "tool_calls"):
        raise ValueError("Provider returned a tool call instead of label scores.")
    content = _value(_value(choices[0], "logprobs", {}), "content", []) or []
    if len(content) != 1:
        raise ValueError("Expected logprobs at exactly one fixed output-token position.")
    first = content[0]
    entries = [first, *(_value(first, "top_logprobs", []) or [])]
    scores = {}
    for entry in entries:
        token = _value(entry, "token")
        # Do not strip whitespace or combine alternate surface forms: these are
        # the likelihoods of the exact labels specified by the judge prompt.
        if token in ("True", "False"):
            scores[token] = _value(entry, "logprob")
    if set(scores) != {"True", "False"}:
        raise ValueError("Both exact True/False labels were not present in first-token logprobs.")
    return scores
