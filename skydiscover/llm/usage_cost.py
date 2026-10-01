"""Preserve provider usage and estimate missing costs for known UMN model routes."""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlsplit

from skydiscover.llm.response_metadata import json_safe

# USD per million tokens, supplied from the UMN models UI on 2026-09-10:
# https://api.aigateway.umn.edu/ui/models-and-endpoints/
# These are gateway-specific rates, not general provider list prices.
_UMN_PRICES = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gemini-3.8-flash": (0.75, 3.75),
}
_MODEL_PREFIXES = {"openai", "azure", "azure_ai", "vertex_ai", "gemini"}


def usage_with_cost(usage: Any, *, api_base: str | None, model: str) -> Any:
    """Copy usage, adding a USD estimate only for the two known UMN routes.

    Provider-reported costs (including zero) are authoritative. The fallback uses
    the requested gateway model's rates and the full input/output token counts;
    cache discounts and other billing adjustments are unknown, so it is marked
    as an estimate. Reasoning tokens are already part of output tokens and must
    not be charged twice. Missing usage never implies a zero-cost request.
    """
    result = json_safe(usage)
    if not isinstance(result, dict) or result.get("cost") is not None:
        return result
    try:
        hostname = urlsplit(api_base or "").hostname
    except ValueError:
        return result
    if hostname != "api.aigateway.umn.edu":
        return result

    model = model.strip().lower()
    while "/" in model and model.split("/", 1)[0] in _MODEL_PREFIXES:
        model = model.split("/", 1)[1]
    prices = _UMN_PRICES.get(model)
    if prices is None:
        return result

    # Chat Completions and Responses APIs use different names for these totals.
    input_tokens = result.get("prompt_tokens", result.get("input_tokens"))
    output_tokens = result.get("completion_tokens", result.get("output_tokens"))
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in (input_tokens, output_tokens)
    ):
        return result
    input_price, output_price = prices
    try:
        cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    except OverflowError:
        return result
    if not math.isfinite(cost):
        return result

    result.update(
        cost=round(cost, 12),
        cost_source="umn_model_pricing",
        cost_is_estimate=True,
        cost_currency="USD",
        cost_pricing={
            "model": model,
            "input_per_million_tokens": input_price,
            "output_per_million_tokens": output_price,
        },
    )
    return result
