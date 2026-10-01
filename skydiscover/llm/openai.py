"""OpenAI-compatible LLM backend (Chat Completions + Responses API)."""

import asyncio
import base64
import json
import logging
import math
import os
import re
import tempfile
import time
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit, urlunsplit

import openai

from skydiscover.config import (
    LLMModelConfig,
    TavilyToolConfig,
    ToolExecutionConfig,
    is_azure_endpoint,
    is_openrouter_endpoint,
    resolve_azure_api_key,
)
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.label_scoring import chat_label_logprobs, confidence_from_logprobs, finite_json
from skydiscover.llm.response_metadata import json_safe, reasoning_content, response_reasoning
from skydiscover.llm.responses_utils import (
    convert_messages_to_responses_input,
    extract_responses_output,
)
from skydiscover.llm.tool_execution import limit_tool_output
from skydiscover.llm.tools import TavilyTool
from skydiscover.llm.usage_cost import usage_with_cost
from skydiscover.llm.verbalized_scoring import chat_verbalized_probability

logger = logging.getLogger("skydiscover.llm")

REASONING_MODEL_PREFIXES = (
    "o1-",
    "o1",
    "o3-",
    "o3",
    "o4-",
    "gpt-5-",
    "gpt-5",
    "gpt-6",
    "gpt-oss-120b",
    "gpt-oss-20b",
)

# Open-weight gpt-oss can be served by any backend (vLLM, Ollama, ...) with its own
# parameter rules, so it only counts as an OpenAI reasoning model on OpenAI's hosts.
OPEN_WEIGHT_REASONING_MODEL_PREFIXES = ("gpt-oss-",)

GOOGLE_AI_STUDIO_DOMAIN = "generativelanguage.googleapis.com"

_OPENAI_API_PREFIXES = (
    "https://api.openai.com",
    "https://eu.api.openai.com",
    "https://apac.api.openai.com",
)


def is_openai_reasoning_model(model_name: str, api_base: str) -> bool:
    """Check if a model is an OpenAI reasoning model requiring special parameters.

    Proprietary OpenAI reasoning models (o-series, gpt-5+) keep OpenAI's parameter
    rules (no temperature/top_p, max_completion_tokens) behind any OpenAI-compatible
    gateway that forwards to OpenAI or Azure, such as a LiteLLM proxy, so they are
    matched by name on every endpoint except OpenRouter, which has its own surface.
    """
    normalized_model = model_name.lower()
    if not normalized_model.startswith(REASONING_MODEL_PREFIXES):
        return False
    api_base_lower = (api_base or "").lower()
    is_openai_api = any(
        api_base_lower.startswith(p) for p in _OPENAI_API_PREFIXES
    ) or is_azure_endpoint(api_base)
    if is_openai_api:
        return True
    if normalized_model.startswith(OPEN_WEIGHT_REASONING_MODEL_PREFIXES):
        return False
    return not is_openrouter_endpoint(api_base)


def requires_responses_api_for_reasoning_tools(
    model_name: str,
    api_base: str,
    reasoning_effort: Optional[str],
) -> bool:
    """Return whether a Luna reasoning+tools request must bypass Chat Completions."""
    if reasoning_effort is None:
        return False
    normalized_effort = str(reasoning_effort).strip().lower()
    if normalized_effort in {"", "none"}:
        return False

    normalized_model = str(model_name or "").strip().lower()
    is_luna = normalized_model == "gpt-5.6-luna" or normalized_model.startswith("gpt-5.6-luna-")
    return is_luna and is_openai_reasoning_model(normalized_model, api_base)


def is_chat_completions_unsupported_error(error: Exception) -> bool:
    """Recognize provider errors that explicitly direct callers away from Chat Completions."""
    error_text = str(error).lower()
    return any(marker in error_text for marker in ("unsupported", "not supported", "not found"))


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _validate_sample_count(n: Any) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    return n


class _InvalidMultiSamplingResponse(ValueError):
    """A completed request failed the native multi-choice response contract."""


def _multi_sampling_unsupported_error(error: Exception) -> bool:
    body = _value(error, "body", {})
    parameter = (
        _value(error, "param")
        or _value(body, "param")
        or _value(_value(body, "error", {}), "param")
    )
    if parameter == "n":
        return True
    text = str(error).lower()
    return bool(re.search(r"\bn\b", text)) and any(
        marker in text
        for marker in ("not supported", "unsupported", "only", "must be", "not allowed", "unknown")
    )


def _normalize_chat_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Convert SDK tool-call objects into plain Chat Completions dictionaries."""
    normalized: List[Dict[str, Any]] = []
    for tool_call in tool_calls or []:
        function = _value(tool_call, "function", {}) or {}
        arguments = _value(function, "arguments", "{}")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        normalized.append(
            {
                "id": _value(tool_call, "id", "") or "",
                "type": "function",
                "function": {
                    "name": _value(function, "name", "") or "",
                    "arguments": arguments or "{}",
                },
            }
        )
    return normalized


def _to_responses_tool_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a Chat Completions function schema for the Responses API."""
    if schema.get("type") != "function" or "function" not in schema:
        return schema
    function = schema["function"]
    return {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object", "properties": {}}),
    }


def _to_responses_text_format(response_format: Any) -> Any:
    """Translate Chat Completions structured-output format to Responses format."""
    if not isinstance(response_format, dict):
        return response_format

    converted = dict(response_format)
    json_schema = converted.pop("json_schema", None)
    if converted.get("type") == "json_schema" and isinstance(json_schema, dict):
        return {**json_schema, "type": "json_schema"}
    return converted


class OpenAILLM(LLMInterface):
    """LLM backend using OpenAI-compatible APIs (Chat Completions + Responses)."""

    def __init__(self, model_cfg: Optional[LLMModelConfig] = None):
        self.model = model_cfg.name
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self._is_openrouter = (
            getattr(model_cfg, "api_provider", None) == "openrouter"
            or "openrouter.ai" in (self.api_base or "").lower()
        )
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self.tool_choice = getattr(model_cfg, "tool_choice", None) or "auto"
        configured_tool_rounds = getattr(model_cfg, "max_tool_rounds", None)
        self.max_tool_rounds = 3 if configured_tool_rounds is None else int(configured_tool_rounds)
        tool_execution = getattr(model_cfg, "tool_execution", None)
        if tool_execution is None:
            tool_execution = ToolExecutionConfig()
        elif isinstance(tool_execution, dict):
            tool_execution = ToolExecutionConfig(**tool_execution)
        self.tool_execution = tool_execution
        self.tools = self._initialize_tools(model_cfg)
        # A controller can point this at its run-level unattached collector so
        # Tavily calls from guide/judge helpers are not silently discarded.
        self.default_web_search_result_sink: Optional[List[Dict[str, Any]]] = None
        # The same fallback captures reasoning from guide/judge/helper calls
        # that do not directly produce a Program.
        self.default_reasoning_result_sink: Optional[List[Dict[str, Any]]] = None

        max_retries = self.retries if self.retries is not None else 0
        is_azure = is_azure_endpoint(self.api_base)
        if is_azure and not self.api_key:
            # Azure credentials live in Azure-specific env vars the OpenAI SDK never reads.
            self.api_key = resolve_azure_api_key()
        # Azure AI Foundry's /openai/v1 surface is plain OpenAI-compatible, so it needs no
        # Azure-specific client.
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=max_retries,
        )

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            api_base_str = (self.api_base or "").lower()
            if is_azure:
                provider = "Azure"
            elif GOOGLE_AI_STUDIO_DOMAIN in api_base_str:
                provider = "Gemini"
            elif "api.anthropic.com" in api_base_str:
                provider = "Anthropic"
            elif "api.deepseek.com" in api_base_str:
                provider = "DeepSeek"
            elif "api.mistral.ai" in api_base_str:
                provider = "Mistral"
            else:
                provider = "OpenAI"
            logger.debug(f"{provider} LLM: {self.model}")
            logger._initialized_models.add(self.model)

    def _initialize_tools(self, model_cfg: LLMModelConfig) -> Dict[str, Any]:
        configured = getattr(model_cfg, "tools", None) or []
        if isinstance(configured, str):
            configured = [configured]
        if not isinstance(configured, (list, tuple)):
            raise ValueError("LLM tools must be a list of built-in tool names")

        tools: Dict[str, Any] = {}
        for raw_name in configured:
            name = str(raw_name).strip()
            if name == "tavily":
                tool_cfg = getattr(model_cfg, "tavily_tool", None)
                if tool_cfg is None:
                    tool_cfg = TavilyToolConfig()
                elif isinstance(tool_cfg, dict):
                    tool_cfg = TavilyToolConfig(**tool_cfg)
                tools[name] = TavilyTool(tool_cfg)
                continue
            raise ValueError(f"Unknown LLM tool {name!r}. Supported built-in tools: tavily")

        if tools and self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1 when LLM tools are enabled")
        return tools

    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Union[LLMResponse, List[LLMResponse]]:
        """Generate one response, or native independent Chat samples with n > 1."""
        n = _validate_sample_count(kwargs.get("n", 1))
        if n > 1:
            has_nontext_input = any(
                isinstance(message.get("content"), list)
                and any(_value(part, "type") != "text" for part in message["content"])
                for message in messages
            )
            if self.tools or kwargs.get("tools") or kwargs.get("image_output") or has_nontext_input:
                raise NotImplementedError(
                    "Native n > 1 generation supports text-only Chat Completions without tools; "
                    "disable tools/images or use n=1."
                )
        llm_call_id = _uuid.uuid4().hex
        llm_context = kwargs.get("llm_context") or {}
        web_search_result_sink = kwargs.get("web_search_result_sink")
        if not isinstance(web_search_result_sink, list):
            web_search_result_sink = self.default_web_search_result_sink
        if not isinstance(web_search_result_sink, list):
            web_search_result_sink = []
        reasoning_result_sink = kwargs.get("reasoning_result_sink")
        uses_default_reasoning_sink = not isinstance(reasoning_result_sink, list)
        if uses_default_reasoning_sink:
            reasoning_result_sink = self.default_reasoning_result_sink
        if not isinstance(reasoning_result_sink, list):
            reasoning_result_sink = []

        started_at = datetime.now(timezone.utc).isoformat()
        reasoning_call: Dict[str, Any] = {
            "llm_call_id": llm_call_id,
            "iteration": llm_context.get("iteration"),
            "attempt": llm_context.get("attempt"),
            "phase": llm_context.get("phase"),
            "program_id": None,
            "source_program_id": llm_context.get("source_program_id"),
            "association": ("unattached_auxiliary_call" if uses_default_reasoning_sink else None),
            "status": "running",
            "started_at": started_at,
            "completed_at": None,
            "duration_ms": None,
            "model": self.model,
            "api_base": self.api_base,
            "request": {
                "system_message": json_safe(system_message),
                "messages": json_safe(messages),
                "parameters": json_safe(
                    {
                        "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                        "temperature": kwargs.get("temperature", self.temperature),
                        "top_p": kwargs.get("top_p", self.top_p),
                        "reasoning_effort": kwargs.get("reasoning_effort", self.reasoning_effort),
                        "response_format": kwargs.get("response_format"),
                        "verbosity": kwargs.get("verbosity"),
                        "image_output": bool(kwargs.get("image_output")),
                        "tool_choice": self.tool_choice if self.tools else None,
                        "max_tool_rounds": self.max_tool_rounds if self.tools else 0,
                        "tools": list(self.tools),
                        **({"n": n} if "n" in kwargs else {}),
                    }
                ),
            },
            "responses": [],
            "tool_executions": [],
            "error": None,
        }
        reasoning_result_sink.append(reasoning_call)
        call_kwargs = dict(kwargs)
        call_kwargs["_llm_call_id"] = llm_call_id
        call_kwargs["_llm_context"] = llm_context
        call_kwargs["_reasoning_call"] = reasoning_call
        call_kwargs["_web_search_result_sink"] = web_search_result_sink

        started = time.perf_counter()
        try:
            if kwargs.get("image_output"):
                result = await self._generate_with_image(system_message, messages, **call_kwargs)
            else:
                text = await self._generate_text(system_message, messages, **call_kwargs)
                if n > 1:
                    if not isinstance(text, list) or len(text) != n:
                        raise _InvalidMultiSamplingResponse(
                            f"Native sampling requested n={n}, but the backend did not return {n} samples."
                        )
                    result = [LLMResponse(text=sample) for sample in text]
                else:
                    result = LLMResponse(text=text)
        except asyncio.CancelledError:
            reasoning_call.update(
                {
                    "status": "cancelled",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": "LLM call was cancelled.",
                }
            )
            raise
        except Exception as exc:
            reasoning_call.update(
                {
                    "status": "error",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "error": str(exc),
                }
            )
            raise

        reasoning_call.update(
            {
                "status": "success",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": None,
            }
        )
        call_search_results = [
            item
            for item in web_search_result_sink
            if isinstance(item, dict) and item.get("llm_call_id") == llm_call_id
        ]
        generation_context = reasoning_call.pop("_generation_context", None)
        safe_reasoning_call = json_safe(reasoning_call)
        if n > 1:
            # The request is billed once. Sibling samples retain a call reference,
            # not copies of the aggregate usage that downstream collectors could sum.
            recorded_responses = safe_reasoning_call.get("responses", [])
            choices = recorded_responses[-1].get("choices", []) if recorded_responses else []
            for index, sample in enumerate(result):
                sample.generation_context = generation_context
                sample.llm_reasoning = {
                    "calls": [safe_reasoning_call] if index == 0 else [],
                    "llm_call_id": llm_call_id,
                    "sample_index": index,
                }
                sample.llm_reasoning_content = reasoning_content(
                    [{"responses": [choice for choice in choices if choice.get("index") == index]}]
                )
                sample.generation_model = self
        else:
            if call_search_results:
                result.web_search_results = {"tavily_searches": json_safe(call_search_results)}
            result.generation_context = generation_context
            result.llm_reasoning = {"calls": [safe_reasoning_call]}
            result.llm_reasoning_content = reasoning_content([safe_reasoning_call])
            result.generation_model = self
        return result

    async def score_labels(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        *,
        timeout: float = 120.0,
        top_logprobs: int = 20,
        openrouter_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score a separate, tool-free True/False assessment on this exact client.

        vLLM first reads single-token labels from one next-token distribution,
        with teacher-forced full-label likelihoods as a fallback for missing
        or multi-token labels. Other compatible backends require both exact
        labels in first-token Chat logprobs.
        Generation parameters and the pool's model-selection RNG are untouched.
        """
        record: Dict[str, Any] = {
            "status": "unavailable",
            "confidence": None,
            "logprob_true": None,
            "logprob_false": None,
            "label_probability_mass": None,
            "model": self.model,
            "api_base": self.api_base,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": {
                "system_message": json_safe(system_message),
                "messages": json_safe(messages),
            },
            "responses": [],
            "error": None,
        }
        started = time.perf_counter()
        try:
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Label-scoring timeout must be finite and positive.")
            if (
                isinstance(top_logprobs, bool)
                or not isinstance(top_logprobs, int)
                or not 1 <= top_logprobs <= 20
            ):
                raise ValueError("Label-scoring top_logprobs must be an integer from 1 to 20.")
            if openrouter_provider is not None and (
                not isinstance(openrouter_provider, str) or not openrouter_provider.strip()
            ):
                raise ValueError("Label-scoring OpenRouter provider must be a nonempty string.")
            scores = await asyncio.wait_for(
                self._score_labels_request(
                    system_message, messages, record, timeout, top_logprobs, openrouter_provider
                ),
                timeout=timeout,
            )
            record.update(confidence_from_logprobs(scores["True"], scores["False"]))
            record["status"] = "success"
        except asyncio.TimeoutError:
            record["status"] = "timeout"
            record["error"] = f"Label scoring exceeded {timeout} seconds."
        except (openai.BadRequestError, openai.NotFoundError) as exc:
            record["status"] = "unsupported"
            record["error"] = str(exc)
        except Exception as exc:
            record["error"] = str(exc)
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        record["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return finite_json(record)

    async def _score_labels_request(
        self, system_message, messages, record, timeout, top_logprobs=20, openrouter_provider=None
    ):
        # A derived SDK client overrides request policy without modifying this
        # generator's retries, sampling, reasoning, tools, or provenance sinks.
        client = self.client.with_options(timeout=timeout, max_retries=0)
        loop = asyncio.get_running_loop()
        conversation = [{"role": "system", "content": system_message}] if system_message else []
        conversation.extend(json_safe(messages))
        is_vllm = await self._confidence_backend_metadata(client)
        if is_vllm:
            return await self._score_labels_vllm(client, conversation, record, top_logprobs)

        record["method"] = "chat_first_token_logprobs"
        params = {
            "model": self.model,
            "messages": conversation,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "max_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        if not self._is_openrouter and is_openai_reasoning_model(self.model, self.api_base):
            params.pop("max_tokens")
            params.pop("temperature")
            params.pop("top_p")
            # Some Azure/GPT routes need more budget than the visible label and
            # return an empty, length-limited result when it is just one.
            # Keep reasoning off; the parser still requires one label position.
            params.update(max_completion_tokens=16, reasoning_effort="none")
        elif self._is_openrouter:
            # Route assessment only to providers that advertise support for the
            # requested logprob parameters instead of silently ignoring them.
            params["extra_body"] = self._confidence_openrouter_parameters(openrouter_provider)
        record["request"]["parameters"] = json_safe(params)
        response = await loop.run_in_executor(
            None, lambda: client.chat.completions.create(**params)
        )
        record["responses"].append(self._response_with_usage_cost(response))
        return chat_label_logprobs(response)

    async def _confidence_backend_metadata(self, client) -> bool:
        """Share capability discovery across concurrent confidence assessments."""

        def cached():
            return getattr(self, "_confidence_is_vllm", None) is not None and (
                not self._is_openrouter or hasattr(self, "_confidence_supported_parameters")
            )

        if cached():
            return self._confidence_is_vllm
        lock = getattr(self, "_confidence_metadata_lock", None)
        if lock is None:
            lock = self._confidence_metadata_lock = asyncio.Lock()
        async with lock:
            if cached():
                return self._confidence_is_vllm
            model_metadata = {}
            try:
                models = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: client.get("/models", cast_to=Dict[str, Any])
                )
                items = models.get("data", []) if isinstance(models, dict) else []
                if isinstance(items, list):
                    model_metadata = next(
                        (
                            item
                            for item in items
                            if isinstance(item, dict) and item.get("id") == self.model
                        ),
                        {},
                    )
            except openai.OpenAIError:
                pass
            if getattr(self, "_confidence_is_vllm", None) is None:
                self._confidence_is_vllm = model_metadata.get("owned_by") == "vllm"
            if self._is_openrouter:
                supported = model_metadata.get("supported_parameters")
                self._confidence_supported_parameters = (
                    frozenset(item for item in supported if isinstance(item, str))
                    if isinstance(supported, list)
                    else frozenset()
                )
            return self._confidence_is_vllm

    def _confidence_openrouter_parameters(self, provider: Optional[str]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"provider": {"require_parameters": True}}
        # Strict routing considers optional controls too. Non-reasoning models
        # must not receive an unsupported reasoning parameter.
        if "reasoning" in self._confidence_supported_parameters:
            body["reasoning"] = {"enabled": False}
        if provider is not None:
            body["provider"].update(only=[provider], allow_fallbacks=False)
        return body

    async def score_verbalized(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        *,
        timeout: float = 120.0,
        openrouter_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request a separate numeric probability on this exact generating client."""
        record: Dict[str, Any] = {
            "status": "unavailable",
            "confidence": None,
            "method": "verbalized_probability",
            "raw_text": None,
            "model": self.model,
            "api_base": self.api_base,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request": {
                "system_message": json_safe(system_message),
                "messages": json_safe(messages),
            },
            "responses": [],
            "error": None,
        }
        started = time.perf_counter()
        try:
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Verbalized-scoring timeout must be finite and positive.")
            if openrouter_provider is not None and (
                not isinstance(openrouter_provider, str) or not openrouter_provider.strip()
            ):
                raise ValueError(
                    "Verbalized-scoring OpenRouter provider must be a nonempty string."
                )
            record["confidence"] = await asyncio.wait_for(
                self._score_verbalized_request(
                    system_message, messages, record, timeout, openrouter_provider
                ),
                timeout=timeout,
            )
            record["status"] = "success"
        except asyncio.TimeoutError:
            record["status"] = "timeout"
            record["error"] = f"Verbalized scoring exceeded {timeout} seconds."
        except (openai.BadRequestError, openai.NotFoundError) as exc:
            record["status"] = "unsupported"
            record["error"] = str(exc)
        except Exception as exc:
            record["error"] = str(exc)
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        record["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return finite_json(record)

    async def _score_verbalized_request(
        self, system_message, messages, record, timeout, openrouter_provider
    ):
        client = self.client.with_options(timeout=timeout, max_retries=0)
        loop = asyncio.get_running_loop()
        conversation = [{"role": "system", "content": system_message}] if system_message else []
        conversation.extend(json_safe(messages))
        is_vllm = await self._confidence_backend_metadata(client)
        params = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": 64,
            "temperature": 1.0,
            "top_p": 1.0,
        }
        if is_vllm:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif not self._is_openrouter and is_openai_reasoning_model(self.model, self.api_base):
            params.pop("max_tokens")
            params.pop("temperature")
            params.pop("top_p")
            params.update(max_completion_tokens=64, reasoning_effort="none")
        elif self._is_openrouter:
            params["extra_body"] = self._confidence_openrouter_parameters(openrouter_provider)
        record["request"]["parameters"] = json_safe(params)
        response = await loop.run_in_executor(
            None, lambda: client.chat.completions.create(**params)
        )
        record["responses"].append(self._response_with_usage_cost(response))
        choices = _value(response, "choices", []) or []
        if choices:
            record["raw_text"] = json_safe(_value(_value(choices[0], "message", {}), "content"))
        return chat_verbalized_probability(response)

    async def _score_labels_vllm_next_token(self, client, prefix_ids, labels, record, top_logprobs):
        """Read both exact label IDs from the same full-vocabulary distribution."""
        method = "vllm_next_token_logprobs"
        record["method"] = method
        params = {
            "model": self.model,
            "prompt": prefix_ids,
            "max_tokens": 1,
            "echo": False,
            "logprobs": top_logprobs,
            "temperature": 1.0,
            "top_p": 1.0,
            "extra_body": {
                "add_special_tokens": False,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            },
        }
        parameters = json_safe({k: v for k, v in params.items() if k != "prompt"})
        record["request"]["parameters"] = parameters
        attempt = {"method": method, "parameters": parameters}
        record["request"].setdefault("attempts", []).append(attempt)
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, lambda: client.completions.create(**params))
        except (openai.BadRequestError, openai.UnprocessableEntityError) as exc:
            # Only an explicit parameter incompatibility permits another request.
            # Transport failures, OOMs, and malformed successful responses do not.
            detail = str(exc).lower()
            unsupported = any(
                marker in detail
                for marker in (
                    "not support",
                    "unsupported",
                    "unknown",
                    "unrecognized",
                    "extra inputs",
                    "extra fields",
                    "not permitted",
                    "greater than max allowed",
                )
            ) and any(name in detail for name in ("return_tokens_as_token_ids", "logprobs"))
            record["responses"].append({"method": method, "error": str(exc)})
            if not unsupported:
                raise
            attempt["fallback_reason"] = "unsupported_next_token_parameters"
            return None

        choices = _value(response, "choices", []) or []
        record["responses"].append(
            {
                "method": method,
                "id": _value(response, "id"),
                "model": _value(response, "model"),
                "usage": self._response_usage(response),
                "choices": json_safe(choices),
            }
        )
        if len(choices) != 1 or _value(choices[0], "index") != 0:
            raise ValueError("vLLM must return exactly one next-token choice at index zero.")
        choice = choices[0]
        if _value(choice, "prompt_token_ids") != prefix_ids:
            raise ValueError("vLLM did not preserve the full prompt token IDs for label scoring.")
        logprobs = _value(choice, "logprobs", {})
        positions = _value(logprobs, "top_logprobs", [])
        if (
            not isinstance(positions, list)
            or len(positions) != 1
            or not isinstance(positions[0], dict)
            or len(_value(logprobs, "token_logprobs", []) or []) != 1
            or len(_value(choice, "token_ids", []) or []) != 1
        ):
            raise ValueError("vLLM did not return exactly one output-token logprob position.")
        keys = {label: f"token_id:{ids[0]}" for label, ids in labels.items()}
        if any(key not in positions[0] for key in keys.values()):
            attempt["fallback_reason"] = "exact_labels_missing_from_top_logprobs"
            return None
        scores = {label: positions[0][key] for label, key in keys.items()}
        # Validate without replacing the original, full-vocabulary logprobs.
        confidence_from_logprobs(scores["True"], scores["False"])
        return scores

    async def _score_labels_vllm(self, client, conversation, record, top_logprobs=20):
        """Tokenize the fixed answer position, then score exact label tokens."""
        loop = asyncio.get_running_loop()
        url = urlsplit(str(client.base_url))
        prefix = url.path.rstrip("/")
        if prefix.endswith("/v1"):
            prefix = prefix[:-3]
        tokenizer_url = urlunsplit((url.scheme, url.netloc, prefix + "/tokenize", "", ""))

        async def tokenize(payload):
            return await loop.run_in_executor(
                None, lambda: client.post(tokenizer_url, cast_to=Dict[str, Any], body=payload)
            )

        tokenize_request = {
            "model": self.model,
            "messages": conversation,
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tokenized = await tokenize(tokenize_request)
        prefix_ids = tokenized.get("tokens")
        if not isinstance(prefix_ids, list) or not prefix_ids:
            raise ValueError("vLLM did not return token IDs for the assessment prompt.")
        labels = {}
        for label in ("True", "False"):
            tokenized_label = await tokenize(
                {"model": self.model, "prompt": label, "add_special_tokens": False}
            )
            ids = tokenized_label.get("tokens")
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"vLLM did not return token IDs for {label}.")
            labels[label] = ids
        if labels["True"] == labels["False"]:
            raise ValueError("True and False unexpectedly have identical tokenizations.")
        record["label_token_ids"] = labels
        record["request"]["tokenization"] = {
            "endpoint": tokenizer_url,
            "chat_template_kwargs": {"enable_thinking": False},
            "prompt_token_ids": prefix_ids,
        }
        if all(len(ids) == 1 for ids in labels.values()):
            scores = await self._score_labels_vllm_next_token(
                client, prefix_ids, labels, record, top_logprobs
            )
            if scores is not None:
                return scores
        else:
            record["request"]["fallback_reason"] = "multi_token_labels"

        record["method"] = "vllm_forced_label_prompt_logprobs"
        params = {
            "model": self.model,
            "prompt": [prefix_ids + label_ids for label_ids in labels.values()],
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "extra_body": {"add_special_tokens": False, "return_token_ids": True},
        }
        # max_tokens=0 + echo requests prompt likelihoods only. vLLM internally
        # computes one unused output token per sequence; none is used as reasoning
        # or substituted for the likelihood of a supplied label token.
        record["request"]["parameters"] = json_safe(
            {k: v for k, v in params.items() if k != "prompt"}
        )
        record["request"].setdefault("attempts", []).append(
            {
                "method": record["method"],
                "parameters": record["request"]["parameters"],
            }
        )
        response = await loop.run_in_executor(None, lambda: client.completions.create(**params))
        response_record = {
            "method": record["method"],
            "id": _value(response, "id"),
            "model": _value(response, "model"),
            "usage": self._response_usage(response),
            "labels": {},
        }
        record["responses"].append(response_record)
        choices = {_value(choice, "index"): choice for choice in _value(response, "choices", [])}
        scores = {}
        for index, (label, label_ids) in enumerate(labels.items()):
            choice = choices.get(index)
            expected_ids = prefix_ids + label_ids
            if choice is None or _value(choice, "prompt_token_ids") != expected_ids:
                raise ValueError(f"vLLM did not preserve the full prompt and {label} token IDs.")
            probabilities = _value(_value(choice, "logprobs", {}), "token_logprobs", [])
            if len(probabilities) != len(expected_ids):
                raise ValueError(f"vLLM did not return full prompt logprobs for {label}.")
            label_logprobs = probabilities[len(prefix_ids) :]
            if any(
                value is None or not math.isfinite(value) or value > 0 or value == -9999.0
                for value in label_logprobs
            ):
                raise ValueError(f"vLLM returned unavailable logprobs for {label}.")
            scores[label] = sum(label_logprobs)
            response_record["labels"][label] = {
                "token_ids": label_ids,
                "token_logprobs": label_logprobs,
                "logprob": scores[label],
            }
        return scores

    def _apply_reasoning_request(
        self, params: Dict[str, Any], reasoning_effort: Optional[str]
    ) -> None:
        """Use OpenRouter's unified reasoning object while retaining OpenAI compatibility."""
        if reasoning_effort is None:
            return
        if self._is_openrouter:
            extra_body = dict(params.get("extra_body") or {})
            extra_body["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": False,
            }
            params["extra_body"] = extra_body
        else:
            params["reasoning_effort"] = reasoning_effort

    # ------------------------------------------------------------------
    # Text generation (Chat Completions API)
    # ------------------------------------------------------------------

    async def _generate_text(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> Union[str, List[str]]:
        # An empty system message is omitted rather than sent as an empty system turn.
        formatted_messages = (
            [{"role": "system", "content": system_message}] if system_message else []
        )
        formatted_messages.extend(messages)

        is_reasoning = not self._is_openrouter and is_openai_reasoning_model(
            self.model, self.api_base
        )

        if is_reasoning:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            self._apply_reasoning_request(params, reasoning_effort)
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                params["temperature"] = temperature
            top_p = kwargs.get("top_p", self.top_p)
            if top_p is not None:
                params["top_p"] = top_p
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            self._apply_reasoning_request(params, reasoning_effort)

        # Add response_format if requested (e.g. {"type": "json_object"})
        response_format = kwargs.get("response_format")
        if response_format is not None:
            params["response_format"] = response_format
        if "n" in kwargs:
            params["n"] = _validate_sample_count(kwargs["n"])

        if self.tools:
            params["tools"] = [tool.chat_completions_schema for tool in self.tools.values()]
            params["tool_choice"] = self.tool_choice

        retries, retry_delay, timeout = self._resolve_retry_options(**kwargs)
        attempt = 0

        while attempt <= retries:
            try:
                call_context: Dict[str, Any] = {
                    "llm_call_id": kwargs.get("_llm_call_id"),
                    "llm_context": kwargs.get("_llm_context"),
                    "reasoning_call": kwargs.get("_reasoning_call"),
                    "web_search_result_sink": kwargs.get("_web_search_result_sink"),
                }
                api_call = self._call_api(params, **call_context)
                return await asyncio.wait_for(
                    api_call,
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout attempt {attempt + 1}/{retries + 1}, retrying...")
                    attempt += 1
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                if kwargs.get("n", 1) > 1 and isinstance(
                    e, (NotImplementedError, _InvalidMultiSamplingResponse)
                ):
                    # Neither unsupported native n nor a malformed completed
                    # response warrants hidden replacement sampling requests.
                    raise
                downgrade_action = self._maybe_downgrade_response_format(params, e)
                if downgrade_action is not None:
                    logger.warning(
                        f"response_format downgrade applied ({downgrade_action}) after API error: {e}"
                    )
                    continue
                if attempt < retries:
                    logger.warning(f"Error attempt {attempt + 1}/{retries + 1}: {e}, retrying...")
                    attempt += 1
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    def _error_mentions_response_format(self, error: Exception) -> bool:
        error_text_parts = [str(error)]

        body = getattr(error, "body", None)
        if body is not None:
            error_text_parts.append(str(body))

        response = getattr(error, "response", None)
        if response is not None:
            response_text = getattr(response, "text", None)
            if response_text:
                error_text_parts.append(str(response_text))

        error_text = " ".join(error_text_parts).lower()
        return any(
            marker in error_text
            for marker in ("response_format", "response format", "text.format", "text format")
        )

    def _maybe_downgrade_response_format(
        self, params: Dict[str, Any], error: Exception
    ) -> Optional[str]:
        if not self._error_mentions_response_format(error):
            return None

        response_format = params.get("response_format")
        if not isinstance(response_format, dict):
            return None

        format_type = response_format.get("type")
        if format_type == "json_schema":
            params["response_format"] = {"type": "json_object"}
            return "json_schema->json_object"

        if format_type == "json_object":
            params.pop("response_format", None)
            return "json_object->none"

        return None

    def _response_usage(self, response: Any) -> Any:
        return usage_with_cost(_value(response, "usage"), api_base=self.api_base, model=self.model)

    def _response_with_usage_cost(self, response: Any) -> Any:
        """Include the same cost metadata in full confidence-response records."""
        payload = json_safe(response)
        if isinstance(payload, dict) and "usage" in payload:
            payload["usage"] = self._response_usage(response)
        return payload

    def _record_chat_response(
        self,
        response: Any,
        *,
        reasoning_call: Optional[Dict[str, Any]],
        round_index: int,
        phase: str,
        all_choices: bool = False,
    ) -> None:
        if not isinstance(reasoning_call, dict):
            return
        if all_choices:
            # Keep aggregate usage once, alongside every choice and its reasoning.
            payload = {
                "round": round_index,
                "phase": phase,
                "response_id": _value(response, "id"),
                "model": _value(response, "model", self.model),
                "usage": self._response_usage(response),
                "choices": [
                    {
                        "index": _value(choice, "index"),
                        "finish_reason": _value(choice, "finish_reason"),
                        "content": _value(_value(choice, "message", {}), "content", "") or "",
                        "tool_calls": _normalize_chat_tool_calls(
                            _value(_value(choice, "message", {}), "tool_calls", [])
                        ),
                        **response_reasoning(_value(choice, "message", {})),
                    }
                    for choice in _value(response, "choices", []) or []
                ],
            }
            reasoning_call.setdefault("responses", []).append(json_safe(payload))
            return
        choice = response.choices[0]
        message = choice.message
        payload: Dict[str, Any] = {
            "round": round_index,
            "phase": phase,
            "response_id": _value(response, "id"),
            "model": _value(response, "model", self.model),
            "finish_reason": _value(choice, "finish_reason"),
            "usage": self._response_usage(response),
            "content": _value(message, "content", "") or "",
            "tool_calls": _normalize_chat_tool_calls(_value(message, "tool_calls", [])),
        }
        payload.update(response_reasoning(message))
        reasoning_call.setdefault("responses", []).append(json_safe(payload))

    def _record_responses_response(
        self,
        response: Any,
        *,
        reasoning_call: Optional[Dict[str, Any]],
        round_index: int,
        phase: str,
        text: str,
        tool_calls: List[Dict[str, Any]],
    ) -> None:
        if not isinstance(reasoning_call, dict):
            return
        output = json_safe(_value(response, "output", []))
        reasoning_items = [
            item for item in output if isinstance(item, dict) and item.get("type") == "reasoning"
        ]
        payload: Dict[str, Any] = {
            "round": round_index,
            "phase": phase,
            "response_id": _value(response, "id"),
            "model": _value(response, "model", self.model),
            "status": _value(response, "status"),
            "usage": self._response_usage(response),
            "content": text,
            "tool_calls": tool_calls,
        }
        if reasoning_items:
            payload["reasoning_details"] = reasoning_items
        reasoning_call.setdefault("responses", []).append(json_safe(payload))

    def _assistant_tool_message(
        self, message: Any, tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": _value(message, "content", "") or "",
            "tool_calls": tool_calls,
        }
        if self._is_openrouter:
            assistant_message.update(response_reasoning(message))
        return assistant_message

    @staticmethod
    def _remember_generation_context(reasoning_call, params, *, responses_api=False):
        """Keep the last actual input on this call, never on shared client state."""
        if not isinstance(reasoning_call, dict):
            return
        if responses_api:
            context = {
                "api": "responses",
                "system_message": params.get("instructions", ""),
                "input": params.get("input", []),
            }
        else:
            messages = params.get("messages", [])
            context = {
                "api": "chat_completions",
                "system_message": next(
                    (item.get("content", "") for item in messages if item.get("role") == "system"),
                    "",
                ),
                "messages": [item for item in messages if item.get("role") != "system"],
            }
        reasoning_call["_generation_context"] = json_safe(context)

    async def _call_api(
        self,
        params: Dict[str, Any],
        *,
        llm_call_id: Optional[str] = None,
        llm_context: Optional[Dict[str, Any]] = None,
        reasoning_call: Optional[Dict[str, Any]] = None,
        web_search_result_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[str, List[str]]:
        n = _validate_sample_count(params.get("n", 1))
        if n > 1 and params.get("tools"):
            raise NotImplementedError("Native n > 1 generation does not support tool execution.")
        if (
            not self._is_openrouter
            and params.get("tools")
            and requires_responses_api_for_reasoning_tools(
                params.get("model", self.model),
                self.api_base,
                params.get("reasoning_effort"),
            )
        ):
            logger.debug("Routing Luna reasoning + function tools directly to the Responses API")
            return await self._call_api_via_responses(
                params,
                llm_call_id=llm_call_id,
                llm_context=llm_context,
                reasoning_call=reasoning_call,
                web_search_result_sink=web_search_result_sink,
            )

        try:
            if params.get("tools"):
                return await self._call_chat_api_with_tools(
                    params,
                    llm_call_id=llm_call_id,
                    llm_context=llm_context,
                    reasoning_call=reasoning_call,
                    web_search_result_sink=web_search_result_sink,
                )
            self._remember_generation_context(reasoning_call, params)
            response = await self._create_chat_completion(params)
            self._record_chat_response(
                response,
                reasoning_call=reasoning_call,
                round_index=0,
                phase="final",
                **({"all_choices": True} if n > 1 else {}),
            )
            if n > 1:
                choices = _value(response, "choices", []) or []
                indices = [_value(choice, "index") for choice in choices]
                if (
                    len(choices) != n
                    or any(
                        isinstance(index, bool) or not isinstance(index, int) for index in indices
                    )
                    or set(indices) != set(range(n))
                ):
                    raise _InvalidMultiSamplingResponse(
                        f"Native sampling requested n={n}, but received choice indices {indices!r}; "
                        f"expected each index from 0 to {n - 1} exactly once. "
                        "Use an endpoint supporting native n sampling, or set n=1."
                    )
                ordered = sorted(choices, key=lambda choice: _value(choice, "index"))
                return [
                    _value(_value(choice, "message", {}), "content", "") or "" for choice in ordered
                ]
            return response.choices[0].message.content
        except (openai.BadRequestError, openai.APIStatusError) as exc:
            if n > 1:
                # A format downgrade may still succeed with native n. Responses
                # cannot preserve n, so unsupported sampling must fail explicitly.
                if self._error_mentions_response_format(exc):
                    raise
                if _multi_sampling_unsupported_error(exc) or is_chat_completions_unsupported_error(
                    exc
                ):
                    raise NotImplementedError(
                        f"This endpoint/model cannot provide native Chat Completions n={n} samples. "
                        "Use a provider supporting n, or set n=1. The Responses API does not "
                        "support n; no singleton fallback was performed."
                    ) from exc
                raise
            # Some Azure deployments only expose the Responses API.
            # Fall back transparently when Chat Completions is unsupported.
            if not is_chat_completions_unsupported_error(exc):
                raise
            logger.debug("Chat Completions unsupported; falling back to Responses API")
            return await self._call_api_via_responses(
                params,
                llm_call_id=llm_call_id,
                llm_context=llm_context,
                reasoning_call=reasoning_call,
                web_search_result_sink=web_search_result_sink,
            )

    async def _call_api_via_responses(
        self,
        params: Dict[str, Any],
        *,
        llm_call_id: Optional[str] = None,
        llm_context: Optional[Dict[str, Any]] = None,
        reasoning_call: Optional[Dict[str, Any]] = None,
        web_search_result_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Translate a Chat-Completions-style *params* dict into a Responses API
        call and return the assistant text, including local function-tool execution."""
        if _validate_sample_count(params.get("n", 1)) > 1:
            raise NotImplementedError(
                "The Responses API does not support native n > 1 sampling; "
                "use a Chat Completions endpoint supporting n, or set n=1."
            )
        messages = params.get("messages", [])
        conversation = [m for m in messages if m.get("role") != "system"]
        input_items = convert_messages_to_responses_input(conversation)
        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        resp_params: Dict[str, Any] = {
            "model": params.get("model", self.model),
            "input": input_items,
        }
        if system_msg:
            resp_params["instructions"] = system_msg
        if params.get("max_tokens"):
            resp_params["max_output_tokens"] = params["max_tokens"]
        if params.get("max_completion_tokens"):
            resp_params["max_output_tokens"] = params["max_completion_tokens"]
        if params.get("temperature") is not None:
            resp_params["temperature"] = params["temperature"]
        if params.get("top_p") is not None:
            resp_params["top_p"] = params["top_p"]
        if params.get("reasoning_effort") is not None:
            resp_params["reasoning"] = {"effort": params["reasoning_effort"]}
        openrouter_reasoning = (params.get("extra_body") or {}).get("reasoning")
        if isinstance(openrouter_reasoning, dict):
            resp_params["reasoning"] = dict(openrouter_reasoning)
        text_config: Dict[str, Any] = {}
        if params.get("response_format") is not None:
            text_config["format"] = _to_responses_text_format(params["response_format"])
        if params.get("verbosity") is not None:
            text_config["verbosity"] = params["verbosity"]
        if text_config:
            resp_params["text"] = text_config
        if params.get("tools"):
            resp_params["tools"] = [_to_responses_tool_schema(schema) for schema in params["tools"]]
            resp_params["tool_choice"] = params.get("tool_choice", "auto")

        tool_round_limit = self.max_tool_rounds if resp_params.get("tools") else 1
        for tool_round in range(tool_round_limit):
            request_params = dict(resp_params)
            request_params["input"] = list(input_items)
            self._remember_generation_context(reasoning_call, request_params, responses_api=True)
            response = await self._create_responses_completion(request_params)
            text, _, tool_calls = extract_responses_output(response)
            self._record_responses_response(
                response,
                reasoning_call=reasoning_call,
                round_index=tool_round,
                phase="tool_request" if tool_calls else "final",
                text=text or "",
                tool_calls=tool_calls,
            )
            if not tool_calls:
                return text or ""

            tool_outputs = await self._execute_tool_calls(
                tool_calls,
                llm_call_id=llm_call_id,
                llm_context=llm_context,
                tool_round=tool_round,
                reasoning_call=reasoning_call,
                web_search_result_sink=web_search_result_sink,
            )
            # Reasoning models require every output item (including hidden reasoning
            # items) to be carried into the next tool round. Reconstructing only the
            # function call would drop that state and can make the continuation fail.
            response_output = list(_value(response, "output", []) or [])
            input_items = [
                *input_items,
                *response_output,
                *convert_messages_to_responses_input(tool_outputs),
            ]
            resp_params["input"] = input_items

            if tool_round + 1 == tool_round_limit:
                final_params = dict(resp_params)
                final_params.pop("tools", None)
                final_params.pop("tool_choice", None)
                self._remember_generation_context(reasoning_call, final_params, responses_api=True)
                final_response = await self._create_responses_completion(final_params)
                final_text, _, _ = extract_responses_output(final_response)
                self._record_responses_response(
                    final_response,
                    reasoning_call=reasoning_call,
                    round_index=tool_round + 1,
                    phase="forced_final",
                    text=final_text or "",
                    tool_calls=[],
                )
                return final_text or ""

        return ""

    async def _call_chat_api_with_tools(
        self,
        params: Dict[str, Any],
        *,
        llm_call_id: Optional[str] = None,
        llm_context: Optional[Dict[str, Any]] = None,
        reasoning_call: Optional[Dict[str, Any]] = None,
        web_search_result_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Run the Chat Completions function-calling loop."""
        call_params = dict(params)
        conversation = list(params.get("messages", []))

        for tool_round in range(self.max_tool_rounds):
            call_params["messages"] = conversation
            self._remember_generation_context(reasoning_call, call_params)
            response = await self._create_chat_completion(call_params)
            message = response.choices[0].message
            tool_calls = _normalize_chat_tool_calls(getattr(message, "tool_calls", None))
            self._record_chat_response(
                response,
                reasoning_call=reasoning_call,
                round_index=tool_round,
                phase="tool_request" if tool_calls else "final",
            )
            if not tool_calls:
                return getattr(message, "content", None) or ""

            conversation.append(self._assistant_tool_message(message, tool_calls))
            conversation.extend(
                await self._execute_tool_calls(
                    tool_calls,
                    llm_call_id=llm_call_id,
                    llm_context=llm_context,
                    tool_round=tool_round,
                    reasoning_call=reasoning_call,
                    web_search_result_sink=web_search_result_sink,
                )
            )

            if tool_round + 1 == self.max_tool_rounds:
                # Give the model one final text-only turn so repeated searches
                # cannot leave the caller with an empty tool-call response.
                final_params = dict(call_params)
                final_params["messages"] = conversation
                final_params.pop("tools", None)
                final_params.pop("tool_choice", None)
                self._remember_generation_context(reasoning_call, final_params)
                final_response = await self._create_chat_completion(final_params)
                self._record_chat_response(
                    final_response,
                    reasoning_call=reasoning_call,
                    round_index=tool_round + 1,
                    phase="forced_final",
                )
                return final_response.choices[0].message.content or ""

        return ""

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        llm_call_id: Optional[str] = None,
        llm_context: Optional[Dict[str, Any]] = None,
        tool_round: Optional[int] = None,
        reasoning_call: Optional[Dict[str, Any]] = None,
        web_search_result_sink: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        async def execute_one(tool_call: Dict[str, Any]) -> Dict[str, Any]:
            function = tool_call.get("function", {})
            name = function.get("name", "")
            raw_arguments = function.get("arguments", "{}")
            started_at = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            arguments: Dict[str, Any] = {}
            parse_error: Optional[Exception] = None
            tavily_execution = None
            tavily_record: Optional[Dict[str, Any]] = None
            provisional_request: Optional[Dict[str, Any]] = None
            execution_record: Optional[Dict[str, Any]] = None
            if isinstance(reasoning_call, dict):
                execution_record = {
                    "tool_call_id": tool_call.get("id", ""),
                    "name": name,
                    "round": tool_round,
                    "status": "running",
                    "started_at": started_at,
                    "completed_at": None,
                    "duration_ms": None,
                    "raw_arguments": json_safe(raw_arguments),
                    "arguments": None,
                    "result": None,
                    "error": None,
                }
                reasoning_call.setdefault("tool_executions", []).append(execution_record)
            try:
                arguments = (
                    raw_arguments
                    if isinstance(raw_arguments, dict)
                    else json.loads(raw_arguments or "{}")
                )
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must decode to an object")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                parse_error = exc
                tool = None
            else:
                tool = self.tools.get(name)

            if name == "tavily" and web_search_result_sink is not None:
                context = llm_context or {}
                if isinstance(tool, TavilyTool) and parse_error is None:
                    query = arguments.get("query")
                    requested_max_results = arguments.get("max_results")
                    if (
                        isinstance(query, str)
                        and query.strip()
                        and not isinstance(requested_max_results, bool)
                        and (
                            requested_max_results is None or isinstance(requested_max_results, int)
                        )
                    ):
                        provisional_request = tool._build_payload(
                            query.strip(), max_results=requested_max_results
                        )
                tavily_record = {
                    "tool_call_id": tool_call.get("id", ""),
                    "llm_call_id": llm_call_id,
                    "iteration": context.get("iteration"),
                    "attempt": context.get("attempt"),
                    "round": tool_round,
                    "program_id": None,
                    "source_program_id": context.get("source_program_id"),
                    "status": "running",
                    "started_at": started_at,
                    "completed_at": None,
                    "duration_ms": None,
                    "config": (tool.provenance_config() if isinstance(tool, TavilyTool) else None),
                    "request": provisional_request,
                    "response": None,
                    "llm_tool_result": None,
                    "error": None,
                }
                web_search_result_sink.append(tavily_record)

            if parse_error is not None:
                result = json.dumps({"error": f"Invalid tool arguments: {parse_error}"})
            elif tool is None:
                result = json.dumps({"error": f"Unknown tool: {name}"})
            else:
                logger.info("LLM requested tool=%s", name)
                try:
                    if isinstance(tool, TavilyTool):
                        # Preserve instance-level test/custom overrides of execute().
                        if "execute" in vars(tool):
                            result = await tool.execute(arguments)
                        else:
                            tavily_execution = await tool.execute_with_metadata(arguments)
                            result = tavily_execution.llm_result
                    else:
                        result = await tool.execute(arguments)
                except asyncio.CancelledError:
                    if execution_record is not None:
                        execution_record.update(
                            {
                                "status": "cancelled",
                                "completed_at": datetime.now(timezone.utc).isoformat(),
                                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                                "arguments": json_safe(arguments),
                                "error": f"Tool {name!r} execution was cancelled.",
                            }
                        )
                    if tavily_record is not None:
                        tavily_record.update(
                            {
                                "status": "cancelled",
                                "completed_at": datetime.now(timezone.utc).isoformat(),
                                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                                "error": "Tavily tool execution was cancelled.",
                            }
                        )
                    raise
                except Exception as exc:
                    logger.exception("LLM tool %s failed unexpectedly", name)
                    result = json.dumps(
                        {"error": f"Tool {name!r} failed: {exc}"},
                        ensure_ascii=False,
                    )

            max_output_chars = self.tool_execution.max_output_chars
            if max_output_chars is not None:
                result = limit_tool_output(result, max_output_chars)

            try:
                parsed_result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                parsed_result = result
            status = (
                "error"
                if isinstance(parsed_result, dict) and parsed_result.get("error")
                else "success"
            )
            completed_at = datetime.now(timezone.utc).isoformat()
            duration_ms = round((time.perf_counter() - started) * 1000, 3)

            if execution_record is not None:
                execution_record.update(
                    json_safe(
                        {
                            "status": status,
                            "completed_at": completed_at,
                            "duration_ms": duration_ms,
                            "arguments": arguments,
                            "result": parsed_result,
                            "error": (
                                parsed_result.get("error")
                                if isinstance(parsed_result, dict)
                                else None
                            ),
                        }
                    )
                )

            if tavily_record is not None:
                error = (
                    tavily_execution.error
                    if tavily_execution is not None
                    else (parsed_result.get("error") if isinstance(parsed_result, dict) else None)
                )
                tavily_record.update(
                    json_safe(
                        {
                            "status": status,
                            "completed_at": completed_at,
                            "duration_ms": duration_ms,
                            "request": (
                                tavily_execution.request
                                if tavily_execution is not None
                                else provisional_request
                            ),
                            "response": (
                                tavily_execution.response if tavily_execution is not None else None
                            ),
                            "llm_tool_result": parsed_result,
                            "error": error,
                        }
                    )
                )

            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": result,
            }

        return list(await asyncio.gather(*(execute_one(call) for call in tool_calls)))

    async def _create_chat_completion(self, params: Dict[str, Any]):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )

    async def _create_responses_completion(self, params: Dict[str, Any]):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.responses.create(**params))

    def _resolve_retry_options(self, **kwargs) -> Tuple[int, int, int]:
        """Resolve retry/timeout options from kwargs, falling back to instance defaults."""
        retries = kwargs.get("retries", self.retries)
        if retries is None:
            retries = 0
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        if retry_delay is None:
            retry_delay = 2
        timeout = kwargs.get("timeout", self.timeout)
        if timeout is None:
            timeout = 300
        return retries, retry_delay, timeout

    # ------------------------------------------------------------------
    # Image generation (OpenAI Responses API)
    # ------------------------------------------------------------------

    async def _generate_with_image(
        self,
        system_message: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> LLMResponse:
        output_dir = kwargs.get("output_dir", tempfile.gettempdir())
        program_id = kwargs.get("program_id", "")

        input_items = convert_messages_to_responses_input(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": [
                {
                    "type": "image_generation",
                    "quality": kwargs.get("image_quality", "medium"),
                    "size": kwargs.get("image_size", "1024x1024"),
                    "output_format": "png",
                }
            ],
        }
        if system_message:
            params["instructions"] = system_message
        is_reasoning = self.model.lower().startswith(REASONING_MODEL_PREFIXES)
        if not is_reasoning and self.temperature is not None:
            params["temperature"] = kwargs.get("temperature", self.temperature)
        if self.max_tokens is not None:
            params["max_output_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        retries, retry_delay, timeout = self._resolve_retry_options(**kwargs)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_responses_api(params), timeout=timeout)
                text, image_b64, tool_calls = extract_responses_output(response)
                self._record_responses_response(
                    response,
                    reasoning_call=kwargs.get("_reasoning_call"),
                    round_index=attempt,
                    phase="image_generation",
                    text=text or "",
                    tool_calls=tool_calls,
                )

                image_path = None
                if image_b64:
                    os.makedirs(output_dir, exist_ok=True)
                    fname = f"{program_id or _uuid.uuid4().hex[:12]}.png"
                    image_path = os.path.join(output_dir, fname)
                    with open(image_path, "wb") as f:
                        f.write(base64.b64decode(image_b64))
                    logger.debug(f"Image saved: {image_path}")

                return LLMResponse(text=text, image_path=image_path)

            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Image timeout attempt {attempt + 1}/{retries + 1}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Image error attempt {attempt + 1}/{retries + 1}: {e}, retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def _call_responses_api(self, params: Dict[str, Any]):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.client.responses.create(**params))
