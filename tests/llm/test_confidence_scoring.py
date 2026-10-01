"""Confidence must use both label probabilities on the generating model."""

import asyncio
import copy
import json
import math
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import httpx
import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMInterface, LLMResponse
from skydiscover.llm.label_scoring import confidence_from_logprobs
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.llm.openai import OpenAILLM
from skydiscover.llm.response_metadata import json_safe


def _llm(*, vllm=False):
    cfg = LLMModelConfig(
        name="Qwen/Qwen3.5-9B" if vllm else "test-model",
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
        llm = OpenAILLM(cfg)
    llm._confidence_is_vllm = vllm
    llm.client.with_options.return_value = llm.client
    llm.client.base_url = cfg.api_base + "/"
    return llm


def _chat_response(*, false=True, reasoning=None):
    entries = [{"token": "True", "logprob": math.log(0.72)}]
    if false:
        entries.append({"token": "False", "logprob": math.log(0.18)})
    return {
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"content": "True", "reasoning_content": reasoning},
                "logprobs": {"content": [{**entries[0], "top_logprobs": entries}]},
            }
        ],
    }


def test_confidence_normalizes_stably_and_records_mass():
    normal = confidence_from_logprobs(math.log(0.72), math.log(0.18))
    assert normal["confidence"] == pytest.approx(0.8)
    assert normal["label_probability_mass"] == pytest.approx(0.9)
    tiny = confidence_from_logprobs(-1000.0, -1001.0)
    assert tiny["confidence"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert tiny["log_label_probability_mass"] < -999
    json.dumps(tiny, allow_nan=False)


@pytest.mark.parametrize("bad", [None, True, math.nan, math.inf, -math.inf, 0.1, -9999.0])
def test_invalid_logprobs_are_not_imputed(bad):
    with pytest.raises(ValueError):
        confidence_from_logprobs(bad, -1.0)


@pytest.mark.asyncio
async def test_chat_scoring_preserves_generator_settings_and_sinks():
    llm = _llm()
    llm.tools = {"tavily": MagicMock()}
    llm.default_reasoning_result_sink = []
    llm.default_web_search_result_sink = []
    keys = ("temperature", "top_p", "max_tokens", "reasoning_effort", "retries", "tool_choice")
    before = {key: getattr(llm, key) for key in keys}
    llm.client.chat.completions.create.return_value = _chat_response()
    messages = [{"role": "user", "content": "candidate"}]

    result = await llm.score_labels("judge", messages, timeout=5)

    assert result["status"] == "success"
    assert result["confidence"] == pytest.approx(0.8)
    assert before == {key: getattr(llm, key) for key in keys}
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params["max_tokens"] == 1
    assert params["temperature"] == params["top_p"] == 1
    assert not {"tools", "tool_choice", "logit_bias", "response_format"} & params.keys()
    assert llm.default_reasoning_result_sink == llm.default_web_search_result_sink == []
    assert messages == [{"role": "user", "content": "candidate"}]
    assert "test-secret" not in json.dumps(result, allow_nan=False)


@pytest.mark.asyncio
async def test_gateway_reasoning_model_allows_label_output_overhead():
    llm = _llm()
    llm.model = "gpt-5.4-nano"
    llm.api_base = "https://api.aigateway.umn.edu"

    def gateway_response(**params):
        if params["max_completion_tokens"] == 1:
            return {
                "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 0},
            }
        response = _chat_response()
        response["usage"] = {
            "completion_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        return response

    llm.client.chat.completions.create.side_effect = gateway_response
    result = await llm.score_labels("", [{"role": "user", "content": "candidate"}], top_logprobs=5)

    assert result["status"] == "success"
    assert result["confidence"] == pytest.approx(0.8)
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params["reasoning_effort"] == "none"
    assert params["top_logprobs"] == 5
    assert not {"max_tokens", "temperature", "top_p", "tools"} & params.keys()
    assert llm.max_tokens == 100 and llm.reasoning_effort == "medium"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "supported_parameters", "expected_reasoning"),
    [
        ("qwen/qwen3.5-27b", ["logprobs", "top_logprobs", "reasoning"], True),
        ("openai/gpt-4o-mini", ["logprobs", "top_logprobs"], False),
    ],
)
async def test_openrouter_scoring_requires_logprob_capable_routing(
    model, supported_parameters, expected_reasoning
):
    llm = _llm()
    llm.model = model
    llm.api_base = "https://openrouter.ai/api/v1"
    llm._is_openrouter = True
    llm.client.get.return_value = {
        "data": [{"id": model, "supported_parameters": supported_parameters}]
    }
    llm.client.chat.completions.create.return_value = _chat_response()

    result = await llm.score_labels("", [{"role": "user", "content": "candidate"}])

    assert result["status"] == "success"
    assert result["confidence"] == pytest.approx(0.8)
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params["logprobs"] is True
    assert params["top_logprobs"] == 20
    expected_body = {"provider": {"require_parameters": True}}
    if expected_reasoning:
        expected_body["reasoning"] = {"enabled": False}
    assert params["extra_body"] == expected_body
    assert result["request"]["parameters"]["extra_body"] == params["extra_body"]
    assert llm.reasoning_effort == "medium"
    await llm.score_labels("", [{"role": "user", "content": "another candidate"}])
    llm.client.get.assert_called_once()
    assert llm.client.chat.completions.create.call_args.kwargs["extra_body"] == expected_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"data": None},
        {"data": [{"id": "test-model"}]},
        {"data": [{"id": "test-model", "supported_parameters": "reasoning"}]},
        {"data": [{"id": "other-model", "supported_parameters": ["reasoning"]}]},
        openai.OpenAIError("model metadata unavailable"),
    ],
)
async def test_openrouter_missing_metadata_omits_optional_reasoning_and_caches(metadata):
    llm = _llm()
    llm._is_openrouter = True
    del llm._confidence_is_vllm
    if isinstance(metadata, Exception):
        llm.client.get.side_effect = metadata
    else:
        llm.client.get.return_value = metadata
    llm.client.chat.completions.create.return_value = _chat_response()

    for _ in range(2):
        result = await llm.score_labels("", [{"role": "user", "content": "candidate"}])
        assert result["status"] == "success"
        params = llm.client.chat.completions.create.call_args.kwargs
        assert params["extra_body"] == {"provider": {"require_parameters": True}}
    llm.client.get.assert_called_once()


@pytest.mark.asyncio
async def test_openrouter_missing_metadata_still_rejects_generated_reasoning():
    llm = _llm()
    llm._is_openrouter = True
    llm.client.get.return_value = {"data": []}
    llm.client.chat.completions.create.return_value = _chat_response(reasoning="thinking")

    result = await llm.score_labels("", [{"role": "user", "content": "candidate"}])

    assert result["status"] == "unavailable"
    assert result["confidence"] is None
    assert "reasoning" in result["error"]


@pytest.mark.asyncio
async def test_openrouter_scoring_can_pin_provider_and_limit_top_logprobs():
    llm = _llm()
    llm.model = "qwen/qwen3.5-27b"
    llm._is_openrouter = True
    llm.client.get.return_value = {
        "data": [{"id": llm.model, "supported_parameters": ["reasoning", "logprobs"]}]
    }
    llm.client.chat.completions.create.return_value = _chat_response()

    result = await llm.score_labels(
        "",
        [{"role": "user", "content": "candidate"}],
        top_logprobs=5,
        openrouter_provider="alibaba",
    )

    assert result["status"] == "success"
    assert result["confidence"] == pytest.approx(0.8)
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params["top_logprobs"] == 5
    assert params["extra_body"] == {
        "reasoning": {"enabled": False},
        "provider": {
            "require_parameters": True,
            "only": ["alibaba"],
            "allow_fallbacks": False,
        },
    }
    assert result["request"]["parameters"]["extra_body"] == params["extra_body"]
    assert result["request"]["parameters"]["top_logprobs"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("top_logprobs", [True, 0, 21, 5.5, "5"])
async def test_invalid_top_logprobs_is_rejected_before_call(top_logprobs):
    llm = _llm()

    result = await llm.score_labels("", [], top_logprobs=top_logprobs)

    assert result["status"] == "unavailable"
    assert "top_logprobs" in result["error"]
    llm.client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [_chat_response(false=False), _chat_response(reasoning="thinking")]
)
async def test_missing_label_or_generated_reasoning_is_unavailable(response):
    llm = _llm()
    llm.client.chat.completions.create.return_value = response

    result = await llm.score_labels("judge", [])

    assert result["status"] == "unavailable"
    assert result["confidence"] is None
    assert result["logprob_false"] is None
    assert result["error"]


@pytest.mark.asyncio
async def test_unavailable_provider_logprob_remains_strictly_json_safe():
    llm = _llm()
    response = _chat_response()
    response["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["logprob"] = -math.inf
    llm.client.chat.completions.create.return_value = response
    result = await llm.score_labels("judge", [])
    assert result["status"] == "unavailable"
    assert result["confidence"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.asyncio
async def test_hidden_reasoning_usage_is_rejected_without_visible_reasoning():
    llm = _llm()
    response = _chat_response()
    response["usage"] = {"completion_tokens_details": {"reasoning_tokens": 4}}
    llm.client.chat.completions.create.return_value = response
    result = await llm.score_labels("judge", [])
    assert result["status"] == "unavailable"
    assert result["confidence"] is None
    assert "hidden reasoning" in result["error"]


@pytest.mark.asyncio
async def test_vllm_reads_both_exact_labels_from_one_next_token_distribution():
    llm = _llm(vllm=True)
    llm.client.post.side_effect = [{"tokens": [10, 11]}, {"tokens": [20]}, {"tokens": [30]}]
    response = _vllm_next_token_response()
    llm.client.completions.create.return_value = response

    result = await llm.score_labels("judge", [], top_logprobs=5)

    assert result["status"] == "success"
    assert result["method"] == "vllm_next_token_logprobs"
    assert result["logprob_true"] == math.log(0.72)
    assert result["logprob_false"] == math.log(0.18)
    assert result["confidence"] == pytest.approx(0.8)
    assert result["label_probability_mass"] == pytest.approx(0.9)
    assert result["label_token_ids"] == {"True": [20], "False": [30]}
    assert result["responses"][0]["choices"] == response["choices"]
    params = llm.client.completions.create.call_args.kwargs
    assert params["prompt"] == [10, 11]
    assert params["max_tokens"] == 1 and params["echo"] is False
    assert params["logprobs"] == 5
    assert params["temperature"] == params["top_p"] == 1
    assert params["extra_body"] == {
        "add_special_tokens": False,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    assert not {"logit_bias", "allowed_token_ids", "tools"} & params.keys()
    llm.client.completions.create.assert_called_once()
    assert "test-secret" not in json.dumps(result, allow_nan=False)


def _vllm_next_token_response():
    # The sampled token is neither label: both scores must come from top_logprobs.
    return {
        "id": "next-score",
        "choices": [
            {
                "index": 0,
                "prompt_token_ids": [10, 11],
                "token_ids": [99],
                "logprobs": {
                    "tokens": ["token_id:99"],
                    "token_logprobs": [-5.0],
                    "top_logprobs": [
                        {"token_id:20": math.log(0.72), "token_id:30": math.log(0.18)}
                    ],
                },
            }
        ],
    }


def _vllm_forced_response():
    return {
        "id": "forced-score",
        "choices": [
            {
                "index": 0,
                "prompt_token_ids": [10, 11, 20],
                "logprobs": {"token_logprobs": [None, -2.0, -30.0]},
            },
            {
                "index": 1,
                "prompt_token_ids": [10, 11, 30],
                "logprobs": {"token_logprobs": [None, -2.0, -31.0]},
            },
        ],
    }


@pytest.mark.asyncio
async def test_vllm_missing_exact_label_falls_back_and_preserves_both_attempts():
    llm = _llm(vllm=True)
    llm.client.post.side_effect = [{"tokens": [10, 11]}, {"tokens": [20]}, {"tokens": [30]}]
    first = _vllm_next_token_response()
    first["choices"][0]["logprobs"]["top_logprobs"][0] = {
        "token_id:20": -0.1,
        "False": -3.0,  # Decoded text cannot establish token identity.
    }
    llm.client.completions.create.side_effect = [first, _vllm_forced_response()]

    result = await llm.score_labels("judge", [])

    assert result["status"] == "success"
    assert result["method"] == "vllm_forced_label_prompt_logprobs"
    assert result["logprob_true"] == -30 and result["logprob_false"] == -31
    assert len(result["responses"]) == len(result["request"]["attempts"]) == 2
    assert result["responses"][0]["choices"] == first["choices"]
    assert (
        result["request"]["attempts"][0]["fallback_reason"]
        == "exact_labels_missing_from_top_logprobs"
    )
    assert result["request"]["attempts"][0]["parameters"]["logprobs"] == 20
    assert result["request"]["parameters"]["max_tokens"] == 0
    assert llm.client.completions.create.call_args.kwargs["prompt"] == [[10, 11, 20], [10, 11, 30]]


@pytest.mark.asyncio
async def test_vllm_explicitly_unsupported_parameter_allows_audited_fallback():
    llm = _llm(vllm=True)
    llm.client.post.side_effect = [{"tokens": [10, 11]}, {"tokens": [20]}, {"tokens": [30]}]
    request = httpx.Request("POST", "http://localhost:8001/v1/completions")
    error = openai.BadRequestError(
        "Unsupported parameter: return_tokens_as_token_ids",
        response=httpx.Response(400, request=request),
        body=None,
    )
    llm.client.completions.create.side_effect = [error, _vllm_forced_response()]

    result = await llm.score_labels("judge", [])

    assert result["status"] == "success"
    assert len(result["responses"]) == 2
    assert "Unsupported parameter" in result["responses"][0]["error"]
    assert (
        result["request"]["attempts"][0]["fallback_reason"] == "unsupported_next_token_parameters"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "timeout", "oom", "bad_request"])
async def test_vllm_operational_errors_never_trigger_another_scoring_request(failure):
    llm = _llm(vllm=True)
    llm.client.post.side_effect = [{"tokens": [10, 11]}, {"tokens": [20]}, {"tokens": [30]}]
    request = httpx.Request("POST", "http://localhost:8001/v1/completions")
    errors = {
        "connection": openai.APIConnectionError(request=request),
        "timeout": openai.APITimeoutError(request=request),
        "oom": openai.InternalServerError(
            "CUDA out of memory",
            response=httpx.Response(500, request=request),
            body=None,
        ),
        "bad_request": openai.BadRequestError(
            "Maximum context length exceeded",
            response=httpx.Response(400, request=request),
            body=None,
        ),
    }
    llm.client.completions.create.side_effect = errors[failure]

    result = await llm.score_labels("judge", [])

    assert result["status"] != "success" and result["confidence"] is None
    assert result["error"]
    llm.client.completions.create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformation", ["index", "duplicate", "prompt", "positions", "output_ids", "mass", "nan"]
)
async def test_vllm_invalid_same_position_response_is_rejected_without_fallback(malformation):
    llm = _llm(vllm=True)
    llm.client.post.side_effect = [{"tokens": [10, 11]}, {"tokens": [20]}, {"tokens": [30]}]
    response = _vllm_next_token_response()
    choice = response["choices"][0]
    if malformation == "index":
        choice["index"] = 1
    elif malformation == "duplicate":
        response["choices"].append(copy.deepcopy(choice))
    elif malformation == "prompt":
        choice["prompt_token_ids"] = [11]
    elif malformation == "positions":
        choice["logprobs"]["top_logprobs"] *= 2
    elif malformation == "output_ids":
        choice["token_ids"] = []
    elif malformation == "mass":
        choice["logprobs"]["top_logprobs"][0] = {"token_id:20": -0.1, "token_id:30": -0.1}
    else:
        choice["logprobs"]["top_logprobs"][0]["token_id:20"] = math.nan
    llm.client.completions.create.return_value = response

    result = await llm.score_labels("judge", [])

    assert result["status"] == "unavailable" and result["confidence"] is None
    assert result["error"]
    llm.client.completions.create.assert_called_once()
    json.dumps(result, allow_nan=False)


@pytest.mark.asyncio
async def test_vllm_forces_both_full_labels_outside_top_k():
    llm = _llm(vllm=True)
    # True deliberately spans two tokens; its full logprob is their sum.
    llm.client.post.side_effect = [
        {"tokens": [10, 11, 12]},
        {"tokens": [20, 21]},
        {"tokens": [30]},
    ]

    def completion(**kwargs):
        prompts = kwargs["prompt"]
        assert prompts == [[10, 11, 12, 20, 21], [10, 11, 12, 30]]
        return SimpleNamespace(
            id="score-1",
            model=llm.model,
            usage={"completion_tokens": 2},
            choices=[
                SimpleNamespace(
                    index=1,
                    prompt_token_ids=prompts[1],
                    logprobs={"token_logprobs": [None, -1, -2, -31.0]},
                ),
                SimpleNamespace(
                    index=0,
                    prompt_token_ids=prompts[0],
                    logprobs={"token_logprobs": [None, -1, -2, -10.0, -20.0]},
                ),
            ],
        )

    llm.client.completions.create.side_effect = completion
    result = await llm.score_labels("judge", [{"role": "user", "content": "candidate"}])

    assert result["status"] == "success"
    assert result["logprob_true"] == -30
    assert result["logprob_false"] == -31
    assert result["confidence"] == pytest.approx(1 / (1 + math.exp(-1)))
    assert result["label_token_ids"] == {"True": [20, 21], "False": [30]}
    assert result["method"] == "vllm_forced_label_prompt_logprobs"
    assert result["request"]["fallback_reason"] == "multi_token_labels"
    llm.client.completions.create.assert_called_once()
    token_call = llm.client.post.call_args_list[0]
    assert token_call.args[0] == "http://localhost:8001/tokenize"
    assert token_call.kwargs["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    params = llm.client.completions.create.call_args.kwargs
    assert params["max_tokens"] == 0 and params["echo"] is True and params["logprobs"] == 0
    assert "tools" not in params
    assert "test-secret" not in json.dumps(result, allow_nan=False)


@pytest.mark.asyncio
async def test_vllm_rejects_truncated_or_mismatched_prompt():
    llm = _llm(vllm=True)
    llm.client.post.side_effect = [{"tokens": [10, 11]}, {"tokens": [20]}, {"tokens": [30]}]
    llm.client.completions.create.return_value = {
        "choices": [
            {"index": 0, "prompt_token_ids": [11, 20], "logprobs": {"token_logprobs": [-1, -2]}}
        ],
    }

    result = await llm.score_labels("judge", [])

    assert result["status"] == "unavailable"
    assert "preserve the full prompt" in result["error"]
    assert result["confidence"] is None


@pytest.mark.asyncio
async def test_timeout_is_explicit_missing_data():
    llm = _llm()

    async def stalled(*args):
        await asyncio.sleep(1)

    llm._score_labels_request = stalled
    result = await llm.score_labels("judge", [], timeout=0.001)
    assert result["status"] == "timeout"
    assert result["confidence"] is None


@pytest.mark.asyncio
async def test_pool_pins_each_concurrent_response_without_extra_sampling():
    class Client(LLMInterface):
        async def generate(self, system_message, messages, **kwargs):
            await asyncio.sleep(0)
            return LLMResponse(text=system_message)

    clients = [Client(), Client()]
    configs = [
        LLMModelConfig(name=f"client-{i}", init_client=lambda cfg, i=i: clients[i])
        for i in range(2)
    ]
    pool = LLMPool(configs)
    pool._sample_model = MagicMock(side_effect=clients)
    state = copy.deepcopy(pool.random_state.getstate())

    first, second = await asyncio.gather(pool.generate("a", []), pool.generate("b", []))

    assert first.generation_model is clients[0]
    assert second.generation_model is clients[1]
    await first.generation_model.score_labels("judge", [])
    assert pool._sample_model.call_count == 2
    assert pool.random_state.getstate() == state
    assert "generation_model" not in asdict(first)
    assert "generation_model" not in json.dumps(json_safe(first))


@pytest.mark.asyncio
async def test_runtime_generation_context_includes_final_tool_outputs():
    llm = _llm()
    llm.tools = {"lookup": SimpleNamespace(chat_completions_schema={"type": "function"})}
    tool_call = {
        "id": "tool-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }
    llm._create_chat_completion = AsyncMock(
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call]))
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="candidate", tool_calls=[]))
                ]
            ),
        ]
    )
    llm._execute_tool_calls = AsyncMock(
        return_value=[
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": "knowledge from tool",
            }
        ]
    )

    response = await llm.generate("system", [{"role": "user", "content": "task"}])

    assert response.generation_model is llm
    assert response.generation_context["system_message"] == "system"
    assert response.generation_context["messages"][-1]["content"] == "knowledge from tool"
    assert response.generation_context["messages"][1]["tool_calls"][0]["id"] == "tool-1"
    assert "_generation_context" not in response.llm_reasoning["calls"][0]
    assert "generation_context" not in asdict(response)
    assert "generation_context" not in json.dumps(json_safe(response))
