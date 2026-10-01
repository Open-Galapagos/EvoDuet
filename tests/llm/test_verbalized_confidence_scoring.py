"""Numeric confidence uses its own request and never depends on token logprobs."""

import asyncio
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.openai import OpenAILLM
from skydiscover.llm.verbalized_scoring import chat_verbalized_probability, parse_probability


def _llm(*, vllm=False):
    config = LLMModelConfig(
        name="test-model",
        api_base="http://localhost:8001/v1",
        api_key="test-secret",
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        reasoning_effort="medium",
        timeout=30,
        retries=2,
    )
    with patch("skydiscover.llm.openai.openai.OpenAI"):
        llm = OpenAILLM(config)
    llm._confidence_is_vllm = vllm
    llm.client.with_options.return_value = llm.client
    return llm


def _response(text="0.73", *, finish_reason="stop"):
    return {
        "model": "test-model",
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0", 0), ("1.0", 1), (" 0.73\n", 0.73), (".25", 0.25), ("7.3e-1", 0.73)],
)
def test_probability_accepts_one_numeric_value(text, expected):
    assert parse_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        None,
        True,
        0.73,
        "True",
        "False",
        "NaN",
        "Infinity",
        "1e999",
        "-0.1",
        "1.1",
        "73%",
        '{"confidence": 0.73}',
        "[0.73]",
        '"0.73"',
        "```0.73```",
        "0.73 because it should improve",
        "Probability: 0.73",
        "0.7 0.3",
    ],
)
def test_probability_does_not_repair_or_clip_answers(text):
    with pytest.raises(ValueError):
        parse_probability(text)


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls", "error"])
def test_incomplete_numeric_answer_is_not_accepted(finish_reason):
    with pytest.raises(ValueError, match="complete"):
        chat_verbalized_probability(_response("0.7", finish_reason=finish_reason))


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "reasoning_details"])
def test_generated_reasoning_is_not_accepted(field):
    response = _response()
    response["choices"][0]["message"][field] = "reasoning before the number"
    with pytest.raises(ValueError, match="reasoning"):
        chat_verbalized_probability(response)


def test_hidden_reasoning_is_not_accepted():
    response = _response()
    response["usage"] = {"completion_tokens_details": {"reasoning_tokens": 2}}
    with pytest.raises(ValueError, match="hidden reasoning"):
        chat_verbalized_probability(response)


@pytest.mark.parametrize("field", ["tool_calls", "function_call", "refusal"])
def test_tool_calls_and_refusals_are_not_accepted(field):
    response = _response()
    response["choices"][0]["message"][field] = {"name": "tool"}
    with pytest.raises(ValueError):
        chat_verbalized_probability(response)


@pytest.mark.parametrize("choices", [[], [None, None]])
def test_missing_or_multiple_answers_are_not_accepted(choices):
    with pytest.raises(ValueError, match="exactly one"):
        chat_verbalized_probability({"choices": choices})


def test_sdk_style_response_is_supported():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="0.73"), finish_reason="stop")]
    )
    assert chat_verbalized_probability(response) == pytest.approx(0.73)


@pytest.mark.asyncio
async def test_verbalized_scoring_preserves_generator_settings_and_sinks():
    llm = _llm()
    llm.tools = {"tavily": MagicMock()}
    llm.default_reasoning_result_sink = []
    llm.default_web_search_result_sink = []
    keys = ("temperature", "top_p", "max_tokens", "reasoning_effort", "retries", "tool_choice")
    before = {key: getattr(llm, key) for key in keys}
    llm.client.chat.completions.create.return_value = _response(" 0.73\n")
    messages = [{"role": "user", "content": "candidate"}]

    result = await llm.score_verbalized("", messages, timeout=5)

    assert result["status"] == "success"
    assert result["confidence"] == pytest.approx(0.73)
    assert result["method"] == "verbalized_probability"
    assert result["raw_text"] == " 0.73\n"
    assert result["responses"] == [_response(" 0.73\n")]
    assert before == {key: getattr(llm, key) for key in keys}
    llm.client.with_options.assert_called_once_with(timeout=5, max_retries=0)
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params == {
        "model": llm.model,
        "messages": messages,
        "max_tokens": 64,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    assert result["request"]["parameters"] == params
    assert llm.default_reasoning_result_sink == llm.default_web_search_result_sink == []
    assert messages == [{"role": "user", "content": "candidate"}]
    assert "test-secret" not in json.dumps(result, allow_nan=False)
    assert result["duration_ms"] >= 0
    assert result["started_at"] <= result["completed_at"]


@pytest.mark.asyncio
async def test_invalid_response_keeps_provenance_without_confidence():
    llm = _llm()
    llm.client.chat.completions.create.return_value = _response("73%")
    result = await llm.score_verbalized("judge", [])
    assert result["status"] == "unavailable"
    assert result["confidence"] is None
    assert result["raw_text"] == "73%"
    assert result["responses"] == [_response("73%")]
    assert result["request"]["parameters"]["messages"] == [{"role": "system", "content": "judge"}]


@pytest.mark.asyncio
async def test_reasoning_gateway_uses_completion_budget_and_disables_reasoning():
    llm = _llm()
    llm.model = "gpt-5.4-nano"
    llm.api_base = "https://api.aigateway.umn.edu"
    llm.client.chat.completions.create.return_value = _response()
    result = await llm.score_verbalized("", [])
    assert result["status"] == "success"
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params["max_completion_tokens"] == 64
    assert params["reasoning_effort"] == "none"
    assert not {"max_tokens", "temperature", "top_p", "logprobs", "top_logprobs"} & params.keys()
    assert llm.max_tokens == 100 and llm.reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_vllm_disables_thinking_without_forcing_labels():
    llm = _llm()
    del llm._confidence_is_vllm
    llm.client.get.return_value = {"data": [{"id": llm.model, "owned_by": "vllm"}]}
    llm.client.chat.completions.create.return_value = _response()
    result = await llm.score_verbalized("", [])
    assert result["status"] == "success"
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert not {"logprobs", "top_logprobs"} & params.keys()
    llm.client.post.assert_not_called()
    llm.client.completions.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("supported_parameters", [["reasoning"], [], None])
async def test_openrouter_does_not_require_logprobs_and_pins_provider(supported_parameters):
    llm = _llm()
    llm._is_openrouter = True
    llm.client.get.return_value = {
        "data": [{"id": llm.model, "supported_parameters": supported_parameters}]
    }
    llm.client.chat.completions.create.return_value = _response()
    result = await llm.score_verbalized("", [], openrouter_provider="alibaba")
    assert result["status"] == "success"
    params = llm.client.chat.completions.create.call_args.kwargs
    expected_body = {
        "provider": {"require_parameters": True, "only": ["alibaba"], "allow_fallbacks": False}
    }
    if supported_parameters:
        expected_body["reasoning"] = {"enabled": False}
    assert params["extra_body"] == expected_body
    assert not {"logprobs", "top_logprobs", "tools"} & params.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {"data": None}, openai.OpenAIError("unavailable")])
async def test_missing_metadata_is_cached_without_optional_reasoning(metadata):
    llm = _llm()
    del llm._confidence_is_vllm
    llm._is_openrouter = True
    if isinstance(metadata, Exception):
        llm.client.get.side_effect = metadata
    else:
        llm.client.get.return_value = metadata
    llm.client.chat.completions.create.return_value = _response()
    for _ in range(2):
        result = await llm.score_verbalized("", [])
        assert result["status"] == "success"
        assert llm.client.chat.completions.create.call_args.kwargs["extra_body"] == {
            "provider": {"require_parameters": True}
        }
    llm.client.get.assert_called_once()


@pytest.mark.asyncio
async def test_concurrent_assessments_share_capability_discovery():
    llm = _llm()
    del llm._confidence_is_vllm
    llm._is_openrouter = True
    llm.client.get.return_value = {"data": [{"id": llm.model, "supported_parameters": []}]}

    def respond(**params):
        if "logprobs" not in params:
            return _response()
        response = _response("True")
        response["choices"][0]["logprobs"] = {
            "content": [
                {
                    "token": "True",
                    "logprob": math.log(0.72),
                    "top_logprobs": [{"token": "False", "logprob": math.log(0.18)}],
                }
            ]
        }
        return response

    llm.client.chat.completions.create.side_effect = respond
    label, verbalized = await asyncio.gather(llm.score_labels("", []), llm.score_verbalized("", []))
    assert label["confidence"] == pytest.approx(0.8)
    assert verbalized["confidence"] == pytest.approx(0.73)
    llm.client.get.assert_called_once()
    assert llm.client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_timeout_leaves_no_numeric_probability():
    llm = _llm()

    async def slow(*args):
        await asyncio.sleep(1)

    llm._score_verbalized_request = AsyncMock(side_effect=slow)
    result = await llm.score_verbalized("", [], timeout=0.01)
    assert result["status"] == "timeout"
    assert result["confidence"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs", [{"timeout": 0}, {"timeout": math.nan}, {"openrouter_provider": ""}]
)
async def test_invalid_options_are_rejected_before_request(kwargs):
    llm = _llm()
    result = await llm.score_verbalized("", [], **kwargs)
    assert result["status"] == "unavailable"
    assert result["error"]
    llm.client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_provider_response_is_recorded():
    llm = _llm()
    llm.client.chat.completions.create.side_effect = openai.BadRequestError(
        "unsupported option",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.com")),
        body=None,
    )
    result = await llm.score_verbalized("", [])
    assert result["status"] == "unsupported"
    assert result["confidence"] is None
    assert "unsupported option" in result["error"]


@pytest.mark.asyncio
async def test_base_backend_reports_unsupported_without_using_generation():
    class UnsupportedBackend(LLMInterface):
        async def generate(self, *args, **kwargs):
            raise AssertionError("Confidence must not call ordinary generation")

    result = await UnsupportedBackend().score_verbalized("", [])
    assert result["status"] == "unsupported"
    assert result["confidence"] is None
    assert result["method"] == "verbalized_probability"
