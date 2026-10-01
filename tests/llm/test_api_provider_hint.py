"""Provider routing survives model normalization and custom compatible endpoints."""

import json
import os
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import Config, LLMModelConfig, apply_overrides
from skydiscover.llm.openai import OpenAILLM

CUSTOM_BASE = "https://proxy.example.test/api/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENAI_BASE = "https://api.openai.com/v1"
MODEL_POOLS = ("models", "evaluator_models", "guide_models")


def _llm(*, model="gpt-5.5", api_base=CUSTOM_BASE, api_provider=None):
    config = LLMModelConfig(
        name=model,
        api_base=api_base,
        api_provider=api_provider,
        api_key="test-key",
        temperature=0.7,
        top_p=0.9,
        max_tokens=1000,
        reasoning_effort="medium",
        timeout=10,
        retries=0,
        retry_delay=0,
    )
    with patch("skydiscover.llm.openai.openai.OpenAI"):
        return OpenAILLM(config)


@pytest.mark.parametrize(
    ("model_spec", "provider", "request_model", "expected_key"),
    [
        (
            "openrouter/anthropic/claude-sonnet-4",
            "openrouter",
            "anthropic/claude-sonnet-4",
            "router-key",
        ),
        ("openai/gpt-5.5", "openai", "gpt-5.5", "openai-key"),
    ],
)
def test_model_override_propagates_provider_to_every_pool(
    model_spec, provider, request_model, expected_key
):
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "router-key", "OPENAI_API_KEY": "openai-key"},
        clear=True,
    ):
        config = Config()
        apply_overrides(config, model=model_spec, api_base=CUSTOM_BASE)

    for pool_name in MODEL_POOLS:
        model = getattr(config.llm, pool_name)[0]
        assert model.api_provider == provider
        assert model.name == request_model
        assert model.api_base == CUSTOM_BASE
        assert model.api_key == expected_key
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            llm = OpenAILLM(model)
        params = {}
        llm._apply_reasoning_request(params, "medium")
        if provider == "openrouter":
            assert params == {"extra_body": {"reasoning": {"effort": "medium", "exclude": False}}}
        else:
            assert params == {"reasoning_effort": "medium"}


def test_provider_hint_survives_serialized_model_configuration():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=True):
        config = Config()
        apply_overrides(
            config,
            model="openrouter/anthropic/claude-sonnet-4",
            api_base=CUSTOM_BASE,
        )
        serialized = json.loads(json.dumps({"llm": asdict(config.llm)}))
        restored = Config.from_dict(serialized)

    for pool_name in MODEL_POOLS:
        model = getattr(restored.llm, pool_name)[0]
        assert model.api_provider == "openrouter"
        assert model.api_base == CUSTOM_BASE
        assert model.name == "anthropic/claude-sonnet-4"
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            assert OpenAILLM(model)._is_openrouter is True


def test_base_only_override_preserves_hint_and_model_override_recomputes_it():
    with patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "router-key", "OPENAI_API_KEY": "openai-key"},
        clear=True,
    ):
        config = Config()
        apply_overrides(config, model="openrouter/openai/gpt-5.5")
        apply_overrides(config, api_base=CUSTOM_BASE)

        for pool_name in MODEL_POOLS:
            model = getattr(config.llm, pool_name)[0]
            assert model.api_base == CUSTOM_BASE
            assert model.api_provider == "openrouter"

        apply_overrides(config, model="openai/gpt-5.5", api_base=CUSTOM_BASE)

    for pool_name in MODEL_POOLS:
        assert getattr(config.llm, pool_name)[0].api_provider == "openai"


@pytest.mark.parametrize(
    ("api_base", "api_provider", "expected"),
    [
        (CUSTOM_BASE, "openrouter", True),
        (OPENROUTER_BASE, None, True),
        (OPENAI_BASE, None, False),
        (CUSTOM_BASE, None, False),
    ],
)
def test_openrouter_hint_supports_custom_hosts_and_preserves_automatic_detection(
    api_base, api_provider, expected
):
    assert _llm(api_base=api_base, api_provider=api_provider)._is_openrouter is expected


def test_custom_openrouter_reasoning_preserves_other_extra_body_fields():
    llm = _llm(api_provider="openrouter")
    params = {"extra_body": {"provider": {"allow_fallbacks": False}}}

    llm._apply_reasoning_request(params, "high")

    assert params["extra_body"] == {
        "provider": {"allow_fallbacks": False},
        "reasoning": {"effort": "high", "exclude": False},
    }
    assert "reasoning_effort" not in params


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.5", "openai/gpt-5.5"])
async def test_custom_openrouter_gpt5_uses_openrouter_request_parameters(model):
    llm = _llm(model=model, api_provider="openrouter")
    llm._call_api = AsyncMock(return_value="answer")

    await llm.generate("system", [{"role": "user", "content": "question"}])

    llm._call_api.assert_awaited_once()
    params = llm._call_api.call_args.args[0]
    assert params["model"] == model
    assert params["max_tokens"] == 1000
    assert params["temperature"] == 0.7
    assert params["top_p"] == 0.9
    assert params["extra_body"]["reasoning"] == {"effort": "medium", "exclude": False}
    assert "max_completion_tokens" not in params
    assert "reasoning_effort" not in params


@pytest.mark.asyncio
async def test_unhinted_gpt5_gateway_retains_openai_request_parameters():
    llm = _llm()
    llm._call_api = AsyncMock(return_value="answer")

    await llm.generate("system", [{"role": "user", "content": "question"}])

    params = llm._call_api.call_args.args[0]
    assert params["max_completion_tokens"] == 1000
    assert params["reasoning_effort"] == "medium"
    assert "extra_body" not in params
    assert "max_tokens" not in params
    assert "temperature" not in params
    assert "top_p" not in params
