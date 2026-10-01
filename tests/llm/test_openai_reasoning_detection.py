"""OpenAI reasoning-model detection behind OpenAI-compatible gateways."""

from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.openai import OpenAILLM, is_openai_reasoning_model

GATEWAY = "https://api.aigateway.example.edu"


@pytest.mark.parametrize(
    ("model", "api_base", "expected"),
    [
        # Proprietary OpenAI reasoning models keep their parameter rules behind a
        # gateway (e.g. a LiteLLM proxy) that forwards to OpenAI or Azure.
        ("gpt-5.5", GATEWAY, True),
        ("gpt-5.6-luna", GATEWAY, True),
        ("gpt-6-astra", GATEWAY, True),
        ("o3-mini", "http://localhost:4000/v1", True),
        ("gpt-5.5", "https://api.openai.com/v1", True),
        ("gpt-6-astra", "https://api.openai.com/v1", True),
        ("gpt-5.6-luna", "https://res.services.ai.azure.com/openai/v1", True),
        # OpenRouter has its own reasoning surface and openai/... model ids.
        ("gpt-5.6-luna", "https://openrouter.ai/api/v1", False),
        # Open-weight gpt-oss may be served by anything; only OpenAI hosts are special.
        ("gpt-oss-120b", "http://localhost:8000/v1", False),
        ("gpt-oss-120b", GATEWAY, False),
        ("gpt-oss-120b", "https://res.services.ai.azure.com/openai/v1", True),
        # Other vendors' models served by the same gateway keep sampling params.
        ("gemini-3.1-pro-preview", GATEWAY, False),
        ("deepseek-v4-flash", GATEWAY, False),
        ("gpt-4.1", GATEWAY, False),
    ],
)
def test_reasoning_detection_follows_the_model_name_off_openai_hosts(model, api_base, expected):
    assert is_openai_reasoning_model(model, api_base) is expected


@pytest.mark.asyncio
async def test_gateway_hosted_gpt5_request_omits_sampling_params():
    """A gpt-5 model behind a gateway must not receive temperature/top_p/max_tokens."""
    cfg = LLMModelConfig(
        name="gpt-5.5",
        api_base=GATEWAY,
        api_key="gateway-key",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000,
        timeout=10,
        retries=0,
        retry_delay=0,
    )
    with patch("skydiscover.llm.openai.openai.OpenAI"):
        llm = OpenAILLM(cfg)
    llm._call_api = AsyncMock(return_value="response")

    await llm.generate(system_message="sys", messages=[{"role": "user", "content": "hi"}])

    params = llm._call_api.call_args[0][0]
    assert params["max_completion_tokens"] == 1000
    assert "max_tokens" not in params
    assert "temperature" not in params
    assert "top_p" not in params
