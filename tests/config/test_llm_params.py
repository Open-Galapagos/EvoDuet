"""Tests for LLM config: optional temperature/top_p and api_base routing."""

import os
from dataclasses import fields
from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import LLMConfig, LLMModelConfig

_OPENAI_DEFAULT_API_BASE: str = next(
    f.default for f in fields(LLMConfig) if f.name == "api_base"
)


class TestLLMConfigDefaults:
    def test_default_temperature(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.temperature == 0.7

    def test_default_top_p_is_none(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.top_p is None

    def test_explicit_none_temperature(self):
        cfg = LLMConfig(name="test-model", temperature=None)
        assert cfg.temperature is None

    def test_explicit_none_top_p(self):
        cfg = LLMConfig(name="test-model", top_p=None)
        assert cfg.top_p is None

    def test_both_none(self):
        cfg = LLMConfig(name="test-model", temperature=None, top_p=None)
        assert cfg.temperature is None
        assert cfg.top_p is None


class TestApiBaseRouting:
    def test_unknown_model_preserves_local_api_base(self):
        local = "http://localhost:11434/v1"
        cfg = LLMConfig(
            name="my-custom-local-model",
            api_base=local,
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == local

    def test_unknown_model_gets_openai_default(self):
        cfg = LLMConfig(
            name="my-custom-local-model",
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == _OPENAI_DEFAULT_API_BASE

    def test_mixed_providers_with_local_api_base(self):
        cfg = LLMConfig(
            api_base="http://localhost:11434/v1",
            models=[
                LLMModelConfig(name="anthropic/claude-3-sonnet"),
                LLMModelConfig(name="my-local-model"),
            ],
        )
        assert cfg.models[0].api_base == "https://api.anthropic.com/v1/"
        assert cfg.models[1].api_base == "http://localhost:11434/v1"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gem-key", "DEEPSEEK_API_KEY": "ds-key"})
    def test_bare_vendor_names_follow_shared_gateway_api_base(self):
        """A gateway may serve gemini-*/deepseek-* too; vendor keys must not reach it."""
        gateway = "https://api.aigateway.example.edu"
        cfg = LLMConfig(
            api_base=gateway,
            api_key="gateway-key",
            models=[
                LLMModelConfig(name="gemini-3.1-pro-preview"),
                LLMModelConfig(name="deepseek-v4-flash"),
            ],
        )
        assert [m.api_base for m in cfg.models] == [gateway, gateway]
        assert [m.api_key for m in cfg.models] == ["gateway-key", "gateway-key"]

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gem-key"})
    def test_prefixed_vendor_name_keeps_its_provider_under_shared_api_base(self):
        cfg = LLMConfig(
            api_base="https://api.aigateway.example.edu",
            api_key="gateway-key",
            models=[LLMModelConfig(name="gemini/gemini-3-pro")],
        )
        assert "generativelanguage.googleapis.com" in cfg.models[0].api_base
        assert cfg.models[0].api_key == "gem-key"
        assert cfg.models[0].name == "gemini-3-pro"

    @patch.dict(
        os.environ, {"AZURE_OPENAI_API_KEY": "azure-key", "OPENAI_API_KEY": "sk-openai"}
    )
    def test_bare_name_under_shared_azure_api_base_uses_the_azure_key(self):
        """The endpoint decides the credential, not the bare model name."""
        cfg = LLMConfig(
            api_base="https://res.services.ai.azure.com/openai/v1",
            models=[LLMModelConfig(name="gpt-5.6-luna")],
        )
        assert cfg.models[0].api_key == "azure-key"


class TestOpenAILLMParams:
    def _make_llm(self, temperature=0.7, top_p=0.95):
        from skydiscover.llm.openai import OpenAILLM

        cfg = LLMModelConfig(
            name="test-model",
            temperature=temperature,
            top_p=top_p,
            api_base="http://localhost:1234/v1",
            api_key="fake",
            timeout=10,
            retries=0,
            retry_delay=0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            llm = OpenAILLM(cfg)
        return llm

    @pytest.mark.asyncio
    async def test_params_include_temperature_and_top_p(self):
        llm = self._make_llm(temperature=0.5, top_p=0.9)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
            temperature=0.5,
            top_p=0.9,
        )
        params = llm._call_api.call_args[0][0]
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_params_exclude_none_top_p(self):
        llm = self._make_llm(top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "top_p" not in params
        assert "temperature" in params

    @pytest.mark.asyncio
    async def test_params_exclude_none_temperature(self):
        llm = self._make_llm(temperature=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" in params

    @pytest.mark.asyncio
    async def test_params_exclude_both_none(self):
        llm = self._make_llm(temperature=None, top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" not in params

    @pytest.mark.asyncio
    async def test_response_format_fallback_json_schema_to_json_object(self):
        llm = self._make_llm()
        captured_params = []

        async def fake_call_api(params, **_kwargs):
            captured_params.append(dict(params))
            if len(captured_params) == 1:
                raise Exception("This response_format type is unavailable now")
            return "response"

        llm._call_api = fake_call_api

        result = await llm.generate(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema",
                    "schema": {"type": "object"},
                },
            },
        )

        assert result.text == "response"
        assert captured_params[0]["response_format"]["type"] == "json_schema"
        assert captured_params[1]["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_response_format_fallback_json_object_to_none(self):
        llm = self._make_llm()
        captured_params = []

        async def fake_call_api(params, **_kwargs):
            captured_params.append(dict(params))
            if len(captured_params) == 1:
                raise Exception("response_format json_object is unsupported")
            return "response"

        llm._call_api = fake_call_api

        result = await llm.generate(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
            response_format={"type": "json_object"},
        )

        assert result.text == "response"
        assert captured_params[0]["response_format"]["type"] == "json_object"
        assert "response_format" not in captured_params[1]
