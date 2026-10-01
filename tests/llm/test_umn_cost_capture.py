"""UMN token pricing is preserved alongside actual LLM call provenance."""

import copy
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from skydiscover.config import DatabaseConfig, LLMModelConfig
from skydiscover.llm.openai import OpenAILLM
from skydiscover.llm.usage_cost import usage_with_cost
from skydiscover.search.base_database import Program
from skydiscover.search.utils.checkpoint_manager import CheckpointManager

UMN_API_BASE = "https://api.aigateway.umn.edu/v1"


@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "expected"),
    [
        ("gpt-5.6-luna", 0.20, 1.20, 0.0000588),
        ("gemini-3.8-flash", 0.75, 3.75, 0.00018525),
    ],
)
@pytest.mark.parametrize(
    "token_keys", [("prompt_tokens", "completion_tokens"), ("input_tokens", "output_tokens")]
)
def test_umn_cost_uses_total_input_and_output_tokens_without_mutating_usage(
    model, input_rate, output_rate, expected, token_keys
):
    usage = {
        token_keys[0]: 12,
        token_keys[1]: 47,
        "total_tokens": 59,
        "completion_tokens_details": {"reasoning_tokens": 40},
        "prompt_tokens_details": {"cached_tokens": 4},
    }
    original = copy.deepcopy(usage)

    result = usage_with_cost(usage, api_base=UMN_API_BASE, model=model)

    assert usage == original
    assert result is not usage
    assert result["cost"] == pytest.approx(expected)
    assert result["cost_source"] == "umn_model_pricing"
    assert result["cost_is_estimate"] is True
    assert result["cost_currency"] == "USD"
    assert result["cost_pricing"] == {
        "model": model,
        "input_per_million_tokens": input_rate,
        "output_per_million_tokens": output_rate,
    }
    assert result["completion_tokens_details"] == {"reasoning_tokens": 40}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("model", "canonical"),
    [
        ("openai/gpt-5.6-luna", "gpt-5.6-luna"),
        ("azure/gpt-5.6-luna", "gpt-5.6-luna"),
        ("openai/azure_ai/gpt-5.6-luna", "gpt-5.6-luna"),
        ("vertex_ai/gemini-3.8-flash", "gemini-3.8-flash"),
        ("openai/vertex_ai/gemini-3.8-flash", "gemini-3.8-flash"),
        ("gemini/gemini-3.8-flash", "gemini-3.8-flash"),
    ],
)
def test_umn_model_aliases_use_canonical_pricing(model, canonical):
    result = usage_with_cost(
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        api_base=UMN_API_BASE,
        model=model,
    )

    assert result["cost_pricing"]["model"] == canonical
    assert result["cost"] == pytest.approx(1.40 if canonical == "gpt-5.6-luna" else 4.50)


@pytest.mark.parametrize("cost", [0, 0.002, "0.002"])
def test_existing_provider_cost_is_preserved(cost):
    usage = {"prompt_tokens": 12, "completion_tokens": 47, "cost": cost}

    result = usage_with_cost(usage, api_base=UMN_API_BASE, model="gpt-5.6-luna")

    assert result == usage
    assert "cost_is_estimate" not in result


def test_null_cost_and_sdk_usage_support_fallback():
    result = usage_with_cost(
        SimpleNamespace(prompt_tokens=12, completion_tokens=47, cost=None),
        api_base=UMN_API_BASE,
        model="gpt-5.6-luna",
    )

    assert result["cost"] == pytest.approx(0.0000588)


@pytest.mark.parametrize(
    "api_base",
    [
        "https://openrouter.ai/api/v1",
        "http://localhost:8000/v1",
        "https://api.openai.com/v1",
        "https://api.aigateway.umn.edu.evil.example/v1",
        "https://evil.example/api.aigateway.umn.edu",
        "https://api.aigateway.umn.edu@evil.example/v1",
        None,
    ],
)
def test_other_endpoints_never_get_umn_estimates(api_base):
    usage = {"prompt_tokens": 12, "completion_tokens": 47}

    assert usage_with_cost(usage, api_base=api_base, model="gpt-5.6-luna") == usage


def test_unknown_gateway_model_does_not_get_a_guessed_price():
    usage = {"prompt_tokens": 12, "completion_tokens": 47}

    assert usage_with_cost(usage, api_base=UMN_API_BASE, model="unknown-model") == usage


@pytest.mark.parametrize("bad", [None, True, -1, 1.5, "12", math.nan, math.inf])
@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
def test_invalid_token_counts_do_not_create_a_cost(bad, field):
    usage = {"prompt_tokens": 12, "completion_tokens": 47, field: bad}

    result = usage_with_cost(usage, api_base=UMN_API_BASE, model="gpt-5.6-luna")

    assert "cost" not in result


@pytest.mark.parametrize(
    "usage", [None, {}, {"total_tokens": 59}, {"prompt_tokens": 12}, {"completion_tokens": 47}]
)
def test_missing_usage_counts_do_not_create_a_cost(usage):
    assert usage_with_cost(usage, api_base=UMN_API_BASE, model="gpt-5.6-luna") == usage


def test_zero_token_usage_is_a_valid_zero_estimate():
    result = usage_with_cost(
        {"prompt_tokens": 0, "completion_tokens": 0},
        api_base=UMN_API_BASE,
        model="gpt-5.6-luna",
    )

    assert result["cost"] == 0.0
    assert result["cost_is_estimate"] is True


def _llm(model):
    config = LLMModelConfig(
        name=model,
        api_base=UMN_API_BASE,
        api_key="test-placeholder",
        max_tokens=100,
        timeout=10,
        retries=0,
    )
    with patch("skydiscover.llm.openai.openai.OpenAI"):
        llm = OpenAILLM(config)
    llm._confidence_is_vllm = False
    llm.client.with_options.return_value = llm.client
    return llm


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_generated_cost_and_call_duration_survive_checkpoint_and_trace(api, tmp_path):
    model = "gpt-5.6-luna" if api == "chat" else "gemini-3.8-flash"
    llm = _llm(model)
    if api == "chat":
        provider_usage = SimpleNamespace(prompt_tokens=12, completion_tokens=47)
        response = SimpleNamespace(
            id="response-1",
            model="upstream-deployment-name",
            usage=provider_usage,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=[]),
                    finish_reason="stop",
                )
            ],
        )
        llm._create_chat_completion = AsyncMock(return_value=response)
    else:
        provider_usage = SimpleNamespace(input_tokens=12, output_tokens=47)
        response = SimpleNamespace(
            id="response-1",
            model="upstream-deployment-name",
            status="completed",
            usage=provider_usage,
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(text="answer")])],
        )
        unsupported = openai.BadRequestError(
            "Chat Completions not supported",
            response=httpx.Response(400, request=httpx.Request("POST", UMN_API_BASE)),
            body=None,
        )
        llm._create_chat_completion = AsyncMock(side_effect=unsupported)
        llm._create_responses_completion = AsyncMock(return_value=response)

    result = await llm.generate("system", [{"role": "user", "content": "question"}])
    call = result.llm_reasoning["calls"][0]
    usage = call["responses"][0]["usage"]
    expected = 0.0000588 if api == "chat" else 0.00018525
    assert usage["cost"] == pytest.approx(expected)
    assert call["responses"][0]["model"] == "upstream-deployment-name"
    assert usage["cost_pricing"]["model"] == model
    assert call["status"] == "success"
    assert call["duration_ms"] >= 0
    assert not hasattr(provider_usage, "cost")

    program = Program(id="child", solution="pass", llm_reasoning=result.llm_reasoning)
    checkpoint = tmp_path / "checkpoints" / "checkpoint_1"
    manager = CheckpointManager(DatabaseConfig())
    manager.save(
        {program.id: program},
        prompts_by_program=None,
        best_program_id=program.id,
        last_iteration=1,
        path=str(checkpoint),
    )
    restored, best_id, last_iteration = manager.load(str(checkpoint))
    assert best_id == "child" and last_iteration == 1
    assert restored["child"].llm_reasoning == result.llm_reasoning
    manager._write_evolution_trace(restored, best_id, last_iteration, str(tmp_path))
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    saved_program = json.loads((checkpoint / "programs" / "child.json").read_text())
    for record in [saved_program, trace["programs"][0]]:
        assert record["llm_reasoning"]["calls"][0] == call


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["score_labels", "score_verbalized"])
async def test_confidence_raw_response_usage_gets_umn_cost(method):
    llm = _llm("gpt-5.6-luna")
    entries = [
        {"token": "True", "logprob": math.log(0.72)},
        {"token": "False", "logprob": math.log(0.18)},
    ]
    response = {
        "model": llm.model,
        "usage": {"prompt_tokens": 12, "completion_tokens": 47},
        "choices": [
            {
                "index": 0,
                "message": {"content": "True" if method == "score_labels" else "0.73"},
                "finish_reason": "stop",
                "logprobs": {"content": [{**entries[0], "top_logprobs": entries}]},
            }
        ],
    }
    llm.client.chat.completions.create.return_value = response

    result = await getattr(llm, method)("", [{"role": "user", "content": "candidate"}])

    assert result["status"] == "success"
    assert result["responses"][0]["usage"]["cost"] == pytest.approx(0.0000588)
    assert result["responses"][0]["usage"]["cost_is_estimate"] is True
    assert "cost" not in response["usage"]
