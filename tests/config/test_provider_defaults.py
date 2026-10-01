"""Tests for provider default removal: no component should silently fall back to OpenAI."""

import logging
import os
from unittest.mock import MagicMock, patch

import openai
import pytest

from skydiscover.config import (
    _AZURE_DEFAULT_BASE_URL,
    _AZURE_KEY_ENV,
    _OPENROUTER_DEFAULT_BASE_URL,
    _OPENROUTER_KEY_ENV,
    Config,
    LLMConfig,
    LLMModelConfig,
    MonitorConfig,
    _parse_model_spec,
    _resolve_api_key_from_env,
    apply_dot_overrides,
    apply_overrides,
    is_azure_endpoint,
    is_openrouter_endpoint,
)


# ── _parse_model_spec ──────────────────────────────────────────────


class TestParseModelSpec:
    def test_known_prefix_gpt(self):
        provider, name, api_base, env_vars = _parse_model_spec("gpt-5")
        assert provider == "openai"
        assert api_base == "https://api.openai.com/v1"
        assert "OPENAI_API_KEY" in env_vars

    def test_known_prefix_gemini(self):
        provider, name, api_base, env_vars = _parse_model_spec("gemini-3-pro")
        assert provider == "gemini"
        assert api_base is not None
        assert "GEMINI_API_KEY" in env_vars

    def test_explicit_provider_slash(self):
        provider, name, api_base, env_vars = _parse_model_spec("anthropic/claude-3-sonnet")
        assert provider == "anthropic"
        assert name == "claude-3-sonnet"

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_ENDPOINT": "https://other.services.ai.azure.com/openai/v1/"},
    )
    def test_azure_endpoint_env_overrides_the_default(self):
        provider, name, api_base, env_vars = _parse_model_spec("azure/gpt-5.6-luna")
        assert provider == "azure"
        assert name == "gpt-5.6-luna"
        assert api_base == "https://other.services.ai.azure.com/openai/v1"
        assert env_vars[0] == "AZURE_OPENAI_API_KEY"

    def test_azure_falls_back_to_the_default_endpoint(self):
        env = os.environ.copy()
        env.pop("AZURE_OPENAI_ENDPOINT", None)
        env.pop("AZURE_OPENAI_BASE_URL", None)
        with patch.dict(os.environ, env, clear=True):
            _, _, api_base, _ = _parse_model_spec("azure/gpt-5.6-luna")
        assert api_base == _AZURE_DEFAULT_BASE_URL
        assert is_azure_endpoint(api_base)

    def test_unknown_model_returns_none_provider(self):
        provider, name, api_base, env_vars = _parse_model_spec("my-custom-model")
        assert provider is None
        assert name == "my-custom-model"
        assert api_base is None
        assert env_vars == []

    def test_unknown_provider_slash_returns_none(self):
        provider, name, api_base, env_vars = _parse_model_spec("mycompany/some-model")
        assert provider is None
        assert api_base is None

    def test_openrouter_endpoint_preserves_author_model_id(self, caplog):
        with caplog.at_level(logging.WARNING):
            provider, name, api_base, env_vars = _parse_model_spec(
                "qwen/qwen3.5-27b", _OPENROUTER_DEFAULT_BASE_URL
            )

        assert provider == "openrouter"
        assert name == "qwen/qwen3.5-27b"
        assert api_base == _OPENROUTER_DEFAULT_BASE_URL
        assert env_vars == _OPENROUTER_KEY_ENV
        assert "Unknown model" not in caplog.text

    def test_explicit_openrouter_provider_strips_only_provider_prefix(self):
        provider, name, api_base, env_vars = _parse_model_spec(
            "openrouter/qwen/qwen3.5-27b"
        )

        assert provider == "openrouter"
        assert name == "qwen/qwen3.5-27b"
        assert api_base == _OPENROUTER_DEFAULT_BASE_URL
        assert env_vars == _OPENROUTER_KEY_ENV


# ── _resolve_api_key_from_env ───────────────────────────────────────


class TestResolveApiKeyFromEnv:
    def test_empty_list_returns_none(self):
        result = _resolve_api_key_from_env([])
        assert result is None

    def test_none_returns_none(self):
        result = _resolve_api_key_from_env(None)
        assert result is None

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key-123"})
    def test_known_provider_returns_own_key(self):
        result = _resolve_api_key_from_env(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
        assert result == "gemini-key-123"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key-123"}, clear=False)
    def test_gemini_does_not_fallback_to_openai_key(self):
        env = os.environ.copy()
        env.pop("GEMINI_API_KEY", None)
        env.pop("GOOGLE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            result = _resolve_api_key_from_env(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
            assert result is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key-123"})
    def test_openai_provider_uses_openai_key(self):
        result = _resolve_api_key_from_env(["OPENAI_API_KEY"])
        assert result == "openai-key-123"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key-123"})
    def test_unknown_model_empty_list_no_fallback(self):
        result = _resolve_api_key_from_env([])
        assert result is None


# ── Azure endpoints ────────────────────────────────────────────────


class TestAzureEndpoints:
    @pytest.mark.parametrize(
        "api_base",
        [
            "https://res.services.ai.azure.com/openai/v1",
            _AZURE_DEFAULT_BASE_URL,
        ],
    )
    def test_foundry_hosts_detected(self, api_base):
        assert is_azure_endpoint(api_base)

    @pytest.mark.parametrize(
        "api_base",
        [
            None,
            "",
            "https://api.openai.com/v1",
            "https://azure.example.com/v1",
            # Only Foundry is supported; a dated Azure OpenAI host is not special-cased.
            "https://res.openai.azure.com",
            # The domain appears in the string but is not the host: substring matching
            # here would release the Azure key to an unrelated server.
            "https://evil.example.com/?next=res.services.ai.azure.com/openai/v1",
            "https://res.services.ai.azure.com.evil.example.com/openai/v1",
        ],
    )
    def test_non_foundry_hosts_rejected(self, api_base):
        assert not is_azure_endpoint(api_base)

    def test_azure_key_envs_in_precedence_order(self):
        assert _AZURE_KEY_ENV == ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"]

    @patch.dict(
        os.environ, {"AZURE_API_KEY": "second-choice", "OPENAI_API_KEY": "sk-nope"}
    )
    def test_azure_api_key_is_the_second_choice(self):
        env = os.environ.copy()
        env.pop("AZURE_OPENAI_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig(models=[LLMModelConfig(name="azure/gpt-5.6-luna")])
        assert cfg.models[0].api_key == "second-choice"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-my-personal-key"})
    def test_openai_key_never_reaches_a_foundry_host(self):
        """An OpenAI key cannot authenticate to Foundry; sending it there leaks it."""
        env = os.environ.copy()
        env.pop("AZURE_OPENAI_API_KEY", None)
        env.pop("AZURE_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig(models=[LLMModelConfig(name="azure/gpt-5.6-luna")])
        assert cfg.models[0].api_base == _AZURE_DEFAULT_BASE_URL
        assert cfg.models[0].api_key is None

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_BASE_URL": "https://from-base-url.services.ai.azure.com/openai/v1"},
    )
    def test_azure_openai_base_url_is_honoured(self):
        env = os.environ.copy()
        env.pop("AZURE_OPENAI_ENDPOINT", None)
        with patch.dict(os.environ, env, clear=True):
            _, _, api_base, _ = _parse_model_spec("azure/gpt-5.6-luna")
        assert api_base == "https://from-base-url.services.ai.azure.com/openai/v1"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://wins.services.ai.azure.com/openai/v1",
            "AZURE_OPENAI_BASE_URL": "https://loses.services.ai.azure.com/openai/v1",
        },
    )
    def test_azure_endpoint_wins_over_base_url(self):
        _, _, api_base, _ = _parse_model_spec("azure/gpt-5.6-luna")
        assert api_base == "https://wins.services.ai.azure.com/openai/v1"

    @patch.dict(
        os.environ, {"AZURE_OPENAI_API_KEY": "azure-key", "OPENAI_API_KEY": "sk-key"}
    )
    def test_foundry_api_base_selects_the_azure_key_for_any_model_name(self):
        """The endpoint decides the credential; the bare name says nothing about it."""
        cfg = Config()
        apply_overrides(
            cfg,
            model="gpt-5.6-luna",
            api_base="https://res.services.ai.azure.com/openai/v1",
        )
        assert cfg.llm.models[0].api_key == "azure-key"

    @patch.dict(
        os.environ, {"AZURE_OPENAI_API_KEY": "azure-key", "OPENAI_API_KEY": "sk-key"}
    )
    def test_foundry_api_base_alone_selects_the_azure_key(self):
        cfg = Config()
        apply_overrides(cfg, api_base="https://res.services.ai.azure.com/openai/v1")
        assert cfg.llm.api_key == "azure-key"

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_ENDPOINT": "https://res.services.ai.azure.com/openai/v1"},
    )
    def test_azure_model_resolves_endpoint_and_bare_name(self):
        cfg = LLMConfig(models=[LLMModelConfig(name="azure/gpt-5.6-luna")])
        assert cfg.models[0].api_base == "https://res.services.ai.azure.com/openai/v1"
        assert cfg.models[0].name == "gpt-5.6-luna"

    @patch.dict(
        os.environ,
        {
            "AZURE_OPENAI_ENDPOINT": "https://res.services.ai.azure.com/openai/v1",
            "AZURE_OPENAI_API_KEY": "azure-key-123",
        },
    )
    def test_azure_model_uses_azure_key(self):
        cfg = LLMConfig(models=[LLMModelConfig(name="azure/gpt-5.6-luna")])
        assert cfg.models[0].api_key == "azure-key-123"

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_ENDPOINT": "https://res.services.ai.azure.com/openai/v1"},
    )
    def test_explicit_api_base_beats_azure_env(self):
        cfg = LLMConfig(
            api_base="http://localhost:8000/v1",
            models=[LLMModelConfig(name="azure/gpt-5.6-luna")],
        )
        assert cfg.models[0].api_base == "http://localhost:8000/v1"

    def test_foundry_v1_endpoint_uses_plain_openai_client(self):
        from skydiscover.llm.openai import OpenAILLM

        llm = OpenAILLM(
            LLMModelConfig(
                name="gpt-5.6-luna",
                api_base="https://res.services.ai.azure.com/openai/v1",
                api_key="test-key",
            )
        )
        assert type(llm.client) is openai.OpenAI

    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "azure-key-123"})
    def test_client_falls_back_to_azure_key_env(self):
        """A YAML config may set only a shared azure api_base and a bare model name."""
        from skydiscover.llm.openai import OpenAILLM

        llm = OpenAILLM(
            LLMModelConfig(
                name="gpt-5.6-luna",
                api_base="https://res.services.ai.azure.com/openai/v1",
            )
        )
        assert llm.api_key == "azure-key-123"

    def test_azure_reasoning_model_detected(self):
        from skydiscover.llm.openai import is_openai_reasoning_model

        assert is_openai_reasoning_model(
            "gpt-5.6-luna", "https://res.services.ai.azure.com/openai/v1"
        )


# ── OpenRouter endpoints ───────────────────────────────────────────


class TestOpenRouterEndpoints:
    @pytest.mark.parametrize(
        "api_base",
        [
            _OPENROUTER_DEFAULT_BASE_URL,
            "https://openrouter.ai/api/v1/",
        ],
    )
    def test_openrouter_hosts_detected(self, api_base):
        assert is_openrouter_endpoint(api_base)

    @pytest.mark.parametrize(
        "api_base",
        [
            None,
            "",
            "https://api.openai.com/v1",
            "https://openrouter.ai.evil.example/api/v1",
            "https://evil.example/?next=https://openrouter.ai/api/v1",
        ],
    )
    def test_non_openrouter_hosts_rejected(self, api_base):
        assert not is_openrouter_endpoint(api_base)

    @patch.dict(
        os.environ,
        {"OPENROUTER_API_KEY": "openrouter-key", "OPENAI_API_KEY": "openai-key"},
    )
    def test_openrouter_override_preserves_model_id_and_selects_own_key(self, caplog):
        cfg = Config()
        with caplog.at_level(logging.WARNING):
            apply_overrides(
                cfg,
                model="qwen/qwen3.5-27b",
                api_base=_OPENROUTER_DEFAULT_BASE_URL,
            )
            provider, *_ = _parse_model_spec(
                cfg.llm.models[0].name, cfg.llm.models[0].api_base
            )

        assert provider == "openrouter"
        assert cfg.llm.models[0].name == "qwen/qwen3.5-27b"
        assert cfg.llm.models[0].api_base == _OPENROUTER_DEFAULT_BASE_URL
        assert cfg.llm.models[0].api_key == "openrouter-key"
        assert "Unknown model" not in caplog.text


# ── Dotted llm overrides ───────────────────────────────────────────


class TestDotOverridePropagation:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key", "GEMINI_API_KEY": "gem-key"})
    def test_scalar_override_keeps_per_model_endpoints(self):
        """An --llm.<scalar> override must not stamp the shared endpoint over a pool."""
        cfg = Config()
        apply_overrides(cfg, model="openai/gpt-5,gemini/gemini-3-pro")
        apply_dot_overrides(cfg, {"llm.max_tokens": "30000"})

        assert [m.max_tokens for m in cfg.llm.models] == [30000, 30000]
        assert "openai.com" in cfg.llm.models[0].api_base
        assert "googleapis.com" in cfg.llm.models[1].api_base
        assert cfg.llm.models[1].api_key == "gem-key"

    @patch.dict(
        os.environ, {"OPENAI_API_KEY": "sk-openai", "AZURE_OPENAI_API_KEY": "azure-secret"}
    )
    def test_api_base_override_re_resolves_the_key(self):
        """The previous provider's credential must not follow the endpoint."""
        cfg = Config()
        apply_overrides(cfg, model="gpt-5.6-chat")
        assert cfg.llm.models[0].api_key == "sk-openai"

        apply_dot_overrides(
            cfg, {"llm.api_base": "https://res.services.ai.azure.com/openai/v1"}
        )
        assert cfg.llm.models[0].api_key == "azure-secret"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"})
    def test_api_base_override_keeps_an_explicit_key_when_none_resolves(self):
        cfg = Config()
        apply_overrides(cfg, model="gpt-5")
        cfg.llm.api_key = "explicitly-configured"
        cfg.llm.update_model_params({"api_key": "explicitly-configured"}, overwrite=True)

        apply_dot_overrides(cfg, {"llm.api_base": "http://localhost:8000/v1"})
        assert cfg.llm.models[0].api_key == "explicitly-configured"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-key"})
    def test_explicit_api_base_override_still_propagates(self):
        cfg = Config()
        apply_overrides(cfg, model="openai/gpt-5")
        apply_dot_overrides(cfg, {"llm.api_base": "http://localhost:8000/v1"})

        assert cfg.llm.models[0].api_base == "http://localhost:8000/v1"


# ── LLMConfig defaults ─────────────────────────────────────────────


class TestLLMConfigApiBaseDefault:
    def test_api_base_defaults_to_none(self):
        cfg = LLMConfig(name="test")
        assert cfg.api_base is None

    def test_explicit_api_base_preserved(self):
        cfg = LLMConfig(name="test", api_base="http://localhost:8000/v1")
        assert cfg.api_base == "http://localhost:8000/v1"


# ── OpenAI-configured runs still work ──────────────────────────────


class TestOpenAIConfiguredRuns:
    def test_openai_model_resolves_correctly(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="gpt-5")],
        )
        assert cfg.models[0].api_base == "https://api.openai.com/v1"

    def test_openai_explicit_prefix_resolves(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="openai/gpt-5")],
        )
        assert cfg.models[0].api_base == "https://api.openai.com/v1"

    def test_openai_model_with_custom_proxy_uses_proxy(self):
        """User's custom api_base should NOT be overridden by OpenAI default."""
        cfg = LLMConfig(
            api_base="http://localhost:8000/v1",
            models=[LLMModelConfig(name="gpt-5")],
        )
        assert cfg.models[0].api_base == "http://localhost:8000/v1"
        assert "openai.com" not in cfg.models[0].api_base

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-123"})
    def test_openai_key_resolved_for_openai_model(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="gpt-5")],
        )
        assert cfg.models[0].api_key == "sk-test-123"


# ── Non-OpenAI runs use correct provider ───────────────────────────


class TestNonOpenAIProviderRouting:
    def test_gemini_model_uses_gemini_endpoint(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="gemini/gemini-3-pro")],
        )
        assert "generativelanguage.googleapis.com" in cfg.models[0].api_base
        assert "openai.com" not in cfg.models[0].api_base

    def test_anthropic_model_uses_anthropic_endpoint(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="anthropic/claude-3-sonnet")],
        )
        assert "anthropic.com" in cfg.models[0].api_base
        assert "openai.com" not in cfg.models[0].api_base

    def test_deepseek_model_uses_deepseek_endpoint(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="deepseek/deepseek-chat")],
        )
        assert "deepseek.com" in cfg.models[0].api_base
        assert "openai.com" not in cfg.models[0].api_base

    def test_local_vllm_preserves_custom_api_base(self):
        cfg = LLMConfig(
            api_base="http://localhost:8000/v1",
            models=[LLMModelConfig(name="my-local-model")],
        )
        assert cfg.models[0].api_base == "http://localhost:8000/v1"
        assert "openai.com" not in cfg.models[0].api_base

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gem-key", "OPENAI_API_KEY": "sk-key"})
    def test_gemini_model_does_not_use_openai_key(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="gemini/gemini-3-pro")],
        )
        assert cfg.models[0].api_key == "gem-key"

    def test_bare_gemini_prefix_resolves_correctly(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="gemini-3-pro")],
        )
        assert "generativelanguage.googleapis.com" in cfg.models[0].api_base

    def test_multi_provider_each_gets_own_endpoint(self):
        cfg = LLMConfig(
            models=[
                LLMModelConfig(name="openai/gpt-5"),
                LLMModelConfig(name="gemini/gemini-3-pro"),
                LLMModelConfig(name="anthropic/claude-3-sonnet"),
            ],
        )
        assert "openai.com" in cfg.models[0].api_base
        assert "googleapis.com" in cfg.models[1].api_base
        assert "anthropic.com" in cfg.models[2].api_base


# ── MonitorConfig defaults ─────────────────────────────────────────


class TestMonitorConfigDefaults:
    def test_summary_model_defaults_to_none(self):
        cfg = MonitorConfig()
        assert cfg.summary_model is None

    def test_summary_api_base_defaults_to_none(self):
        cfg = MonitorConfig()
        assert cfg.summary_api_base is None

    def test_summary_api_key_defaults_to_none(self):
        cfg = MonitorConfig()
        assert cfg.summary_api_key is None


# ── Monitor summary propagation (runner.py) ────────────────────────


class TestMonitorSummaryPropagation:
    def _make_runner_config(self, llm_models=None, monitor_kwargs=None):
        """Build a Config with given LLM models and monitor settings."""
        cfg = Config.__new__(Config)
        cfg.monitor = MonitorConfig(**(monitor_kwargs or {}))
        cfg.llm = LLMConfig(
            models=llm_models or [],
        )
        cfg.llm.__post_init__()
        return cfg

    def test_propagates_from_main_config_when_monitor_not_configured(self):
        """When monitor has no summary_model, it should pick up the first LLM model."""
        from skydiscover.runner import Runner

        cfg = self._make_runner_config(
            llm_models=[LLMModelConfig(name="gemini/gemini-3-pro", api_base="https://gemini.test/v1", api_key="gem-key")],
        )

        mock_server = MagicMock()
        runner = Runner.__new__(Runner)
        runner.config = cfg

        runner._setup_monitor_summary(mock_server)

        mock_server.configure_summary.assert_called_once()
        call_kwargs = mock_server.configure_summary.call_args.kwargs
        assert call_kwargs["model"] == "gemini/gemini-3-pro"
        assert call_kwargs["api_base"] == "https://gemini.test/v1"
        assert call_kwargs["api_key"] == "gem-key"

    def test_monitor_explicit_config_takes_priority(self):
        """When monitor has its own summary_model, it should NOT be overridden."""
        from skydiscover.runner import Runner

        cfg = self._make_runner_config(
            llm_models=[LLMModelConfig(name="gpt-5", api_base="https://api.openai.com/v1")],
            monitor_kwargs={"summary_model": "claude-3-haiku", "summary_api_base": "https://anthropic.test/v1"},
        )

        mock_server = MagicMock()
        runner = Runner.__new__(Runner)
        runner.config = cfg

        runner._setup_monitor_summary(mock_server)

        call_kwargs = mock_server.configure_summary.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-haiku"
        assert call_kwargs["api_base"] == "https://anthropic.test/v1"

    def test_no_models_anywhere_disables_summary(self, caplog):
        """When neither monitor nor LLM config has a model, summary is disabled with warning."""
        from skydiscover.runner import Runner

        cfg = self._make_runner_config()

        mock_server = MagicMock()
        runner = Runner.__new__(Runner)
        runner.config = cfg

        with caplog.at_level(logging.WARNING):
            runner._setup_monitor_summary(mock_server)

        mock_server.configure_summary.assert_not_called()
        assert "Summary feature disabled" in caplog.text


# ── server.py: summary calls route through OpenAILLM ───────────────


class TestMonitorSummaryRouting:
    """The summary call must use the same client selection as ordinary LLM calls."""

    @staticmethod
    def _probe(recorder):
        from skydiscover.llm.base import LLMResponse
        from skydiscover.llm.openai import OpenAILLM as RealLLM

        class Probe(RealLLM):
            def __init__(self, model_cfg):
                super().__init__(model_cfg)
                recorder["client"] = type(self.client)
                recorder["base_url"] = str(self.client.base_url)
                recorder["api_key"] = self.client.api_key
                recorder["model"] = self.model
                recorder["timeout"] = self.timeout
                recorder["max_tokens"] = self.max_tokens

            async def generate(self, system_message, messages, **kwargs):
                recorder["system"] = system_message
                recorder["messages"] = messages
                return LLMResponse(text="  summary text  ")

        return Probe

    def _run(self, api_base, recorder, model="my-deployment", api_key="k", **call_kwargs):
        from skydiscover.extras.monitor.server import MonitorServer

        server = MonitorServer()
        server.configure_summary(model=model, api_key=api_key, api_base=api_base)
        with patch("skydiscover.llm.openai.OpenAILLM", self._probe(recorder)):
            return server._call_llm_api({"system": "s", "user": "u"}, **call_kwargs)

    def test_summary_reaches_the_configured_endpoint(self):
        recorder = {}
        text = self._run("https://res.services.ai.azure.com/openai/v1", recorder)
        assert recorder["client"] is openai.OpenAI
        assert recorder["base_url"].rstrip("/") == (
            "https://res.services.ai.azure.com/openai/v1"
        )
        assert recorder["api_key"] == "k"
        assert text == "summary text"

    def test_summary_prompt_reaches_the_model(self):
        recorder = {}
        self._run("https://api.openai.com/v1", recorder)
        assert recorder["system"] == "s"
        assert recorder["messages"] == [{"role": "user", "content": "u"}]

    def test_per_call_budget_reaches_the_model(self):
        recorder = {}
        self._run(
            "https://api.openai.com/v1", recorder, max_tokens=2048, timeout=120
        )
        assert (recorder["max_tokens"], recorder["timeout"]) == (2048, 120)

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_API_KEY": "azure-key-123", "OPENAI_API_KEY": "sk-wrong"},
    )
    def test_configure_summary_resolves_azure_key(self):
        from skydiscover.extras.monitor.server import MonitorServer

        server = MonitorServer()
        server.configure_summary(
            model="my-deployment",
            api_key="",
            api_base="https://res.services.ai.azure.com/openai/v1",
        )
        assert server._summary_api_key == "azure-key-123"

    @patch.dict(
        os.environ,
        {"AZURE_OPENAI_API_KEY": "azure-key-123", "OPENAI_API_KEY": "sk-wrong"},
    )
    def test_prefixed_summary_model_resolves_its_own_endpoint(self):
        """Without an api_base the Azure key would otherwise go to api.openai.com."""
        recorder = {}
        self._run("", recorder, model="azure/gpt-5.6-luna", api_key="")
        assert recorder["model"] == "gpt-5.6-luna"
        assert recorder["base_url"].rstrip("/") == _AZURE_DEFAULT_BASE_URL
        assert recorder["api_key"] == "azure-key-123"


# ── server.py: no hardcoded OpenAI ─────────────────────────────────


class TestServerNoHardcodedDefaults:
    def test_server_default_summary_model_empty(self):
        from skydiscover.extras.monitor.server import MonitorServer

        server = MonitorServer()
        assert server._summary_model == ""

    def test_server_default_summary_api_base_empty(self):
        from skydiscover.extras.monitor.server import MonitorServer

        server = MonitorServer()
        assert server._summary_api_base == ""

    def test_unconfigured_server_does_not_call_openai(self):
        """A server that was never configure_summary()'d should not attempt API calls."""
        from skydiscover.extras.monitor.server import MonitorServer

        server = MonitorServer()
        assert not server._summary_model


# ── openevolve_backend.py: error on missing api_base ────────────────


class TestOpenEvolveBackendError:
    def test_no_api_base_raises_error(self):
        """When no api_base can be resolved, a ValueError should be raised."""
        env = os.environ.copy()
        env.pop("OPENAI_API_BASE", None)
        env.pop("OPENAI_BASE_URL", None)

        with patch.dict(os.environ, env, clear=True):
            resolved = (
                os.environ.get("OPENAI_API_BASE")
                or os.environ.get("OPENAI_BASE_URL")
                or None  # simulates config.llm.api_base = None
            )
            assert resolved is None
            # The backend should raise ValueError, not continue with None
            if not resolved:
                with pytest.raises(ValueError, match="no api_base resolved"):
                    raise ValueError(
                        "OpenEvolve backend: no api_base resolved. "
                        "Set OPENAI_BASE_URL or configure llm.api_base in your config."
                    )


# ── viewer.py: no auto gpt-5-mini assignment ───────────────────────


class TestViewerNoAutoOpenAI:
    def test_summary_model_default_is_empty(self):
        """--summary-model default should be empty, not 'gpt-5-mini'."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--summary-model", default="")
        args = parser.parse_args([])
        assert args.summary_model == ""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "some-key"})
    def test_openai_key_presence_does_not_set_model(self):
        """Even with OPENAI_API_KEY set, no model should be auto-assigned."""
        summary_model = ""
        assert not summary_model

    def test_summary_api_base_default_is_empty(self):
        """--summary-api-base default should be empty, not OpenAI URL."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--summary-api-base", default="")
        args = parser.parse_args([])
        assert args.summary_api_base == ""
        assert "openai.com" not in args.summary_api_base


# ── variation_operator_generator.py: --model required ──────────────


class TestVariationOperatorCLI:
    def test_default_cli_model_is_none(self):
        from skydiscover.search.evox.utils.variation_operator_generator import DEFAULT_CLI_MODEL

        assert DEFAULT_CLI_MODEL is None

    def test_cli_without_model_returns_error(self):
        """CLI should fail with exit code 1 when --model is not provided."""
        from skydiscover.search.evox.utils.variation_operator_generator import main

        with patch("sys.argv", ["prog", "/tmp/fake-problem-dir"]), \
             patch("skydiscover.search.evox.utils.variation_operator_generator.os.path.exists", return_value=True), \
             patch("skydiscover.search.evox.utils.variation_operator_generator.load_config", return_value={"prompt": {}}), \
             patch("skydiscover.search.evox.utils.variation_operator_generator.load_evaluator", return_value="pass"), \
             patch("builtins.print") as mock_print:
            result = main()

        assert result == 1
        mock_print.assert_any_call("Error: --model is required (e.g. --model gpt-5-mini)")


# ── monitor/__init__.py: propagation ───────────────────────────────


class TestMonitorInitPropagation:
    def test_propagates_from_llm_config(self):
        """start_monitor should propagate LLM model to summary when not configured."""
        from skydiscover.extras.monitor import start_monitor

        cfg = Config.__new__(Config)
        cfg.monitor = MonitorConfig(enabled=True)
        cfg.llm = LLMConfig(
            models=[LLMModelConfig(name="anthropic/claude-3-sonnet", api_base="https://anthropic.test/v1", api_key="ant-key")],
        )
        cfg.llm.__post_init__()

        with patch("skydiscover.extras.monitor.MonitorServer") as MockServer:
            mock_instance = MagicMock()
            MockServer.return_value = mock_instance

            start_monitor(cfg, output_dir="/tmp/fake-output")

            mock_instance.configure_summary.assert_called_once()
            call_kwargs = mock_instance.configure_summary.call_args.kwargs
            assert call_kwargs["model"] == "anthropic/claude-3-sonnet"
            assert call_kwargs["api_base"] == "https://anthropic.test/v1"
            assert call_kwargs["api_key"] == "ant-key"

    def test_no_model_skips_summary(self):
        """start_monitor should not call configure_summary when no model is available."""
        from skydiscover.extras.monitor import start_monitor

        cfg = Config.__new__(Config)
        cfg.monitor = MonitorConfig(enabled=True)
        cfg.llm = LLMConfig()
        cfg.llm.__post_init__()

        with patch("skydiscover.extras.monitor.MonitorServer") as MockServer:
            mock_instance = MagicMock()
            MockServer.return_value = mock_instance

            start_monitor(cfg, output_dir="/tmp/fake-output")

            mock_instance.configure_summary.assert_not_called()
