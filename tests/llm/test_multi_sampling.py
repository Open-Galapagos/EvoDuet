"""Native multi-choice sampling keeps request counts, provenance, and usage correct."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.llm.openai import OpenAILLM


def _llm(**overrides):
    settings = dict(
        name="test-model",
        api_base="http://localhost:1234/v1",
        api_key="test",
        temperature=0.7,
        top_p=0.9,
        max_tokens=120,
        timeout=2,
        retries=0,
        retry_delay=0,
    )
    settings.update(overrides)
    with patch("skydiscover.llm.openai.openai.OpenAI"):
        return OpenAILLM(LLMModelConfig(**settings))


def _completion(indices=(1, 0)):
    return SimpleNamespace(
        id="request-samples",
        model="test-model",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=19),
        choices=[
            SimpleNamespace(
                index=index,
                finish_reason="stop",
                message=SimpleNamespace(
                    content=f"sample {index}",
                    tool_calls=[],
                    reasoning_content=f"reasoning {index}",
                ),
            )
            for index in indices
        ],
    )


def _bad_request(message, parameter=None):
    return openai.BadRequestError(
        message,
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://provider.test/v1/chat/completions")
        ),
        body={"message": message, "param": parameter},
    )


def _pool(*clients):
    return LLMPool(
        [
            LLMModelConfig(name=f"custom-{index}", init_client=lambda config, client=client: client)
            for index, client in enumerate(clients)
        ]
    )


@pytest.mark.asyncio
async def test_native_n_uses_one_sdk_request_and_returns_all_choices_in_index_order():
    llm = _llm()
    llm.client.chat.completions.create.return_value = _completion((2, 0, 1))
    sink = []
    llm.default_reasoning_result_sink = sink

    results = await llm.generate(
        "shared system",
        [{"role": "user", "content": "shared query prompt"}],
        n=3,
        llm_context={"iteration": 7, "phase": "evoduet-query", "source_program_id": "parent"},
    )

    assert [result.text for result in results] == ["sample 0", "sample 1", "sample 2"]
    llm.client.chat.completions.create.assert_called_once()
    params = llm.client.chat.completions.create.call_args.kwargs
    assert params["n"] == 3
    assert params["messages"] == [
        {"role": "system", "content": "shared system"},
        {"role": "user", "content": "shared query prompt"},
    ]
    assert (params["temperature"], params["top_p"], params["max_tokens"]) == (0.7, 0.9, 120)
    assert [result.llm_reasoning_content for result in results] == [
        "reasoning 0",
        "reasoning 1",
        "reasoning 2",
    ]
    assert all(result.generation_model is llm for result in results)
    assert all(result.generation_context["api"] == "chat_completions" for result in results)
    assert len(sink) == 1
    call = sink[0]
    assert call["status"] == "success"
    assert call["association"] == "unattached_auxiliary_call"
    assert (call["iteration"], call["phase"], call["source_program_id"]) == (
        7,
        "evoduet-query",
        "parent",
    )
    assert call["request"]["parameters"]["n"] == 3
    assert len(call["responses"]) == 1
    recorded = call["responses"][0]
    assert recorded["usage"] == {"prompt_tokens": 11, "completion_tokens": 19}
    assert sorted(choice["index"] for choice in recorded["choices"]) == [0, 1, 2]
    # A request has one billable record even when samples are serialized together.
    sample_calls = [entry for result in results for entry in result.llm_reasoning["calls"]]
    assert len(sample_calls) == 1
    assert all(result.llm_reasoning["llm_call_id"] == call["llm_call_id"] for result in results)
    assert [result.llm_reasoning["sample_index"] for result in results] == [0, 1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, {"n": 1}])
async def test_singleton_contract_and_metadata_are_unchanged(options):
    llm = _llm()
    llm.client.chat.completions.create.return_value = _completion((0,))

    result = await llm.generate("system", [], **options)

    assert isinstance(result, LLMResponse)
    assert result.text == "sample 0"
    assert result.llm_reasoning_content == "reasoning 0"
    assert result.llm_reasoning["calls"][0]["responses"][0]["content"] == "sample 0"
    params = llm.client.chat.completions.create.call_args.kwargs
    assert ("n" in params) == ("n" in options)


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [0, -1, True, False, 1.5, 2.0, "2", None])
async def test_invalid_n_is_rejected_before_sampling_or_provider_calls(n):
    llm = _llm()
    client = SimpleNamespace(generate=AsyncMock())
    pool = _pool(client)

    for target in (llm, pool):
        with pytest.raises(ValueError, match="positive integer"):
            await target.generate("system", [], n=n)

    llm.client.chat.completions.create.assert_not_called()
    client.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", ["tools", "image_output", "image_input", "tool_argument"])
async def test_multi_sampling_rejects_nontext_or_tools_before_provider_calls(unsupported):
    llm = _llm()
    messages = [{"role": "user", "content": "query"}]
    kwargs = {}
    if unsupported == "tools":
        llm.tools = {"fake": object()}
    elif unsupported == "image_output":
        kwargs["image_output"] = True
    elif unsupported == "tool_argument":
        kwargs["tools"] = [{"type": "function"}]
    else:
        messages[0]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png,..."}}]

    with pytest.raises(NotImplementedError, match="text-only"):
        await llm.generate("system", messages, n=2, **kwargs)

    llm.client.chat.completions.create.assert_not_called()
    llm.client.responses.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "indices", [(), (0,), (0, 0), (0, 2), (0, "1"), (0, True), (0, None), (0, 1, 2)]
)
async def test_malformed_multi_choices_fail_without_retry_and_retain_usage(indices):
    llm = _llm(retries=3)
    llm._create_chat_completion = AsyncMock(return_value=_completion(indices))
    sink = []

    with pytest.raises(ValueError, match="choice indices"):
        await llm.generate("system", [], n=2, reasoning_result_sink=sink)

    llm._create_chat_completion.assert_awaited_once()
    assert sink[0]["status"] == "error"
    assert len(sink[0]["responses"]) == 1
    assert sink[0]["responses"][0]["usage"]["completion_tokens"] == 19
    llm.client.responses.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message,parameter",
    [
        ("Chat Completions is not supported; use Responses", None),
        ("n must be 1", "n"),
        ("Only n=1 is allowed", None),
        ("Unsupported parameter: n", None),
    ],
)
async def test_unsupported_native_n_fails_without_responses_fallback_or_retry(message, parameter):
    llm = _llm(retries=3)
    llm._create_chat_completion = AsyncMock(side_effect=_bad_request(message, parameter))
    llm._call_api_via_responses = AsyncMock(return_value="one response")
    sink = []

    with pytest.raises(NotImplementedError, match="no singleton fallback"):
        await llm.generate("system", [], n=2, reasoning_result_sink=sink)

    llm._create_chat_completion.assert_awaited_once()
    llm._call_api_via_responses.assert_not_awaited()
    assert sink[0]["status"] == "error"


@pytest.mark.asyncio
async def test_direct_responses_path_cannot_silently_drop_n():
    llm = _llm()
    with pytest.raises(NotImplementedError, match="Responses API does not support"):
        await llm._call_api_via_responses({"model": llm.model, "messages": [], "n": 2})
    llm.client.responses.create.assert_not_called()


@pytest.mark.asyncio
async def test_singleton_retains_responses_fallback():
    llm = _llm()
    llm._create_chat_completion = AsyncMock(
        side_effect=_bad_request("Chat Completions unsupported")
    )
    llm._call_api_via_responses = AsyncMock(return_value="single fallback")

    result = await llm.generate("system", [], n=1)

    assert result.text == "single fallback"
    llm._call_api_via_responses.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_failure", [RuntimeError("temporary transport error"), asyncio.TimeoutError()]
)
async def test_transient_retries_keep_native_n_and_record_successful_usage_once(first_failure):
    llm = _llm(retries=1)
    llm._create_chat_completion = AsyncMock(side_effect=[first_failure, _completion()])
    sink = []

    results = await llm.generate("system", [], n=2, reasoning_result_sink=sink)

    assert [result.text for result in results] == ["sample 0", "sample 1"]
    assert llm._create_chat_completion.await_count == 2
    assert all(call.args[0]["n"] == 2 for call in llm._create_chat_completion.call_args_list)
    assert len(sink) == 1
    assert sink[0]["status"] == "success"
    assert len(sink[0]["responses"]) == 1


@pytest.mark.asyncio
async def test_multi_request_timeout_uses_existing_budget_and_marks_call_error():
    llm = _llm()

    async def delayed(params):
        await asyncio.sleep(10)

    llm._create_chat_completion = AsyncMock(side_effect=delayed)
    sink = []
    with pytest.raises(asyncio.TimeoutError):
        await llm.generate("system", [], n=2, timeout=0.01, reasoning_result_sink=sink)

    llm._create_chat_completion.assert_awaited_once()
    assert sink[0]["status"] == "error"
    assert sink[0]["completed_at"] is not None


@pytest.mark.asyncio
async def test_format_downgrade_preserves_native_n_and_reasoning_parameters():
    llm = _llm(name="gpt-5.5", reasoning_effort="medium")
    requests = []

    async def create(params):
        requests.append(copy.deepcopy(params))
        if len(requests) == 1:
            raise _bad_request("response_format json_schema unsupported", "response_format")
        return _completion()

    llm._create_chat_completion = AsyncMock(side_effect=create)
    results = await llm.generate(
        "system",
        [],
        n=2,
        response_format={"type": "json_schema", "json_schema": {"name": "sample", "schema": {}}},
        max_tokens=80,
    )

    assert len(results) == 2
    assert len(requests) == 2
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert requests[1]["response_format"] == {"type": "json_object"}
    for params in requests:
        assert params["n"] == 2
        assert params["max_completion_tokens"] == 80
        assert params["reasoning_effort"] == "medium"
        assert "temperature" not in params
        assert "top_p" not in params
    llm.client.responses.create.assert_not_called()


@pytest.mark.asyncio
async def test_pool_forwards_native_n_once_and_associates_every_sample():
    samples = [LLMResponse(text="first"), LLMResponse(text="second")]
    client = SimpleNamespace(generate=AsyncMock(return_value=samples))
    pool = _pool(client)

    results = await pool.generate("same system", [], n=2)

    assert results is samples
    client.generate.assert_awaited_once_with("same system", [], n=2)
    assert all(result.generation_model is client for result in results)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [LLMResponse(text="collapsed"), [], [LLMResponse()], ["a", "b"]]
)
async def test_pool_rejects_custom_clients_that_drop_samples_or_return_wrong_types(response):
    client = SimpleNamespace(generate=AsyncMock(return_value=response))
    with pytest.raises(ValueError, match="exactly 2 LLMResponse"):
        await _pool(client).generate("system", [], n=2)
    client.generate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{}, {"n": 1}])
async def test_pool_singleton_contract_is_unchanged(options):
    response = LLMResponse(text="one")
    client = SimpleNamespace(generate=AsyncMock(return_value=response))
    result = await _pool(client).generate("system", [], **options)
    assert result is response
    assert result.generation_model is client


@pytest.mark.asyncio
async def test_generate_all_groups_native_samples_by_model_and_associates_each():
    first = SimpleNamespace(
        generate=AsyncMock(return_value=[LLMResponse(text="a0"), LLMResponse(text="a1")])
    )
    second = SimpleNamespace(
        generate=AsyncMock(return_value=[LLMResponse(text="b0"), LLMResponse(text="b1")])
    )

    results = await _pool(first, second).generate_all("system", [], n=2)

    assert [[sample.text for sample in group] for group in results] == [["a0", "a1"], ["b0", "b1"]]
    assert all(sample.generation_model is first for sample in results[0])
    assert all(sample.generation_model is second for sample in results[1])
    first.generate.assert_awaited_once_with("system", [], n=2)
    second.generate.assert_awaited_once_with("system", [], n=2)
