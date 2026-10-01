import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import (
    Config,
    LLMModelConfig,
    TavilyToolConfig,
    ToolExecutionConfig,
    apply_dot_overrides,
)
from skydiscover.llm.openai import (
    OpenAILLM,
    is_chat_completions_unsupported_error,
    requires_responses_api_for_reasoning_tools,
)
from skydiscover.llm.tool_execution import limit_tool_output
from skydiscover.llm.tools.tavily import (
    TAVILY_TOOL_SCHEMA,
    TavilyTool,
    _format_response,
)
from skydiscover.utils.tavily import TavilyResponseDecodeError


def test_tavily_tool_schema_exposes_optional_max_results():
    function = TAVILY_TOOL_SCHEMA["function"]
    parameters = function["parameters"]
    max_results = parameters["properties"]["max_results"]

    assert function["name"] == "tavily"
    assert max_results["type"] == "integer"
    assert max_results["minimum"] == 0
    assert max_results["maximum"] == 20
    assert max_results["default"] == 5
    assert parameters["required"] == ["query"]


def test_tavily_tool_config_matches_official_search_defaults():
    config = asdict(TavilyToolConfig())

    # query is supplied at tool-call time; transport settings are SkyDiscover-owned.
    for key in ("api_key", "base_url", "timeout"):
        config.pop(key)

    assert config == {
        "search_depth": "basic",
        "chunks_per_source": 3,
        "max_results": 5,
        "topic": "general",
        "time_range": None,
        "start_date": None,
        "end_date": None,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
        "include_image_descriptions": False,
        "include_favicon": False,
        "include_domains": [],
        "exclude_domains": [],
        "country": None,
        "language": None,
        "filter_by_language": False,
        "auto_parameters": False,
        "exact_match": False,
        "include_usage": False,
        "safe_search": False,
    }


def test_tavily_tool_default_payload_uses_official_defaults():
    payload = TavilyTool(TavilyToolConfig())._build_payload("test query")

    assert payload == {
        "query": "test query",
        "search_depth": "basic",
        "max_results": 5,
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
        "include_usage": False,
    }


def test_tavily_tool_sends_chunks_for_advanced_depth():
    payload = TavilyTool(
        TavilyToolConfig(search_depth="advanced", chunks_per_source=2)
    )._build_payload("test query")

    assert payload["chunks_per_source"] == 2


@pytest.mark.parametrize("search_depth", ["basic", "fast", "ultra-fast"])
def test_tavily_tool_omits_chunks_for_non_advanced_depth(search_depth):
    payload = TavilyTool(TavilyToolConfig(search_depth=search_depth))._build_payload("test query")

    assert "chunks_per_source" not in payload


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _chat_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(query="latest circle-packing record"):
    return SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="tavily",
            arguments=json.dumps({"query": query}),
        ),
    )


def _make_llm(
    *,
    model_name="test-model",
    max_tool_rounds=3,
    max_output_chars=None,
    api_base="http://localhost:1234/v1",
    reasoning_effort=None,
):
    config = LLMModelConfig(
        name=model_name,
        api_base=api_base,
        api_key="fake-llm-key",
        temperature=0.1,
        max_tokens=1000,
        timeout=10,
        retries=0,
        retry_delay=0,
        reasoning_effort=reasoning_effort,
        tools=["tavily"],
        max_tool_rounds=max_tool_rounds,
        tool_execution=ToolExecutionConfig(max_output_chars=max_output_chars),
        tavily_tool=TavilyToolConfig(
            api_key="tvly-test",
            max_results=3,
            search_depth="basic",
        ),
    )
    with patch("skydiscover.llm.openai.openai.OpenAI"):
        return OpenAILLM(config)


@pytest.mark.parametrize(
    ("model", "api_base", "effort", "expected"),
    [
        (
            "gpt-5.6-luna",
            "https://unit.services.ai.azure.com/openai/v1",
            "medium",
            True,
        ),
        ("gpt-5.6-luna", "https://api.openai.com/v1", "medium", True),
        ("gpt-5.6-luna", "https://openrouter.ai/api/v1", "medium", False),
        (
            "gpt-5.6-terra",
            "https://unit.services.ai.azure.com/openai/v1",
            "medium",
            False,
        ),
        (
            "gpt-5.6-luna",
            "https://unit.services.ai.azure.com/openai/v1",
            "none",
            False,
        ),
    ],
)
def test_reasoning_tool_responses_routing_is_provider_and_model_scoped(
    model, api_base, effort, expected
):
    assert requires_responses_api_for_reasoning_tools(model, api_base, effort) is expected


def test_chat_unsupported_error_matches_provider_wording():
    error = RuntimeError("Function tools with reasoning_effort are not supported for gpt-5.6-luna")

    assert is_chat_completions_unsupported_error(error)


def test_config_parses_and_propagates_tavily_tool():
    config = Config.from_dict(
        {
            "llm": {
                "models": [
                    {
                        "name": "test-model",
                        "api_base": "http://localhost:1234/v1",
                        "api_key": "fake",
                    }
                ],
                "tools": ["tavily"],
                "tool_choice": "auto",
                "max_tool_rounds": 4,
                "tool_execution": {"max_output_chars": 12_000},
                "tavily_tool": {
                    "api_key": "tvly-config",
                    "max_results": 7,
                    "include_domains": ["arxiv.org"],
                    "language": "en",
                    "filter_by_language": True,
                },
            }
        }
    )

    model = config.llm.models[0]
    assert model.tools == ["tavily"]
    assert model.tool_choice == "auto"
    assert model.max_tool_rounds == 4
    assert isinstance(model.tool_execution, ToolExecutionConfig)
    assert model.tool_execution.max_output_chars == 12_000
    assert isinstance(model.tavily_tool, TavilyToolConfig)
    assert model.tavily_tool.max_results == 7
    assert model.tavily_tool.include_domains == ["arxiv.org"]
    assert model.tavily_tool.language == "en"
    assert model.tavily_tool.filter_by_language is True
    assert not hasattr(model, "tavily_search_tool")


def test_legacy_tavily_search_tool_name_is_rejected():
    config = LLMModelConfig(
        name="test-model",
        api_base="http://localhost:1234/v1",
        api_key="fake-llm-key",
        tools=["tavily_search"],
    )

    with pytest.raises(ValueError, match="Unknown LLM tool 'tavily_search'"):
        OpenAILLM(config)


def test_legacy_tavily_search_config_path_is_rejected():
    config = Config.from_dict({"llm": {"models": [{"name": "test-model"}]}})

    with pytest.raises(ValueError, match="unknown config section"):
        apply_dot_overrides(
            config,
            {"llm.tavily_search_tool.max_results": "5"},
        )


def test_cli_tavily_override_reaches_shared_model_config():
    config = Config.from_dict(
        {
            "llm": {
                "models": [{"name": "test-model"}],
                "tools": ["tavily"],
            }
        }
    )

    apply_dot_overrides(
        config,
        {
            "llm.tool_choice": "auto",
            "llm.max_tool_rounds": "2",
            "llm.tool_execution.max_output_chars": "12000",
            "llm.tavily_tool.max_results": "0",
        },
    )

    model = config.llm.models[0]
    assert model.tool_choice == "auto"
    assert model.max_tool_rounds == 2
    assert model.tool_execution.max_output_chars == 12_000
    assert model.tavily_tool.max_results == 0


@pytest.mark.asyncio
async def test_tavily_tool_executes_configured_search(monkeypatch):
    captured = {}

    async def fake_search(payload, *, api_key, base_url, timeout):
        captured.update(
            payload=payload,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        return {
            "answer": "A useful result.",
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "Relevant content",
                    "score": 0.9,
                }
            ],
        }

    monkeypatch.setattr(
        "skydiscover.llm.tools.tavily.tavily_search",
        fake_search,
    )
    tool = TavilyTool(
        TavilyToolConfig(
            api_key="tvly-key",
            max_results=2,
            search_depth="advanced",
            chunks_per_source=2,
            include_domains=["example.com"],
            language="en",
            filter_by_language=True,
        )
    )

    output = json.loads(await tool.execute({"query": "  test query  "}))

    assert captured["api_key"] == "tvly-key"
    assert captured["payload"]["query"] == "test query"
    assert captured["payload"]["max_results"] == 2
    assert captured["payload"]["chunks_per_source"] == 2
    assert captured["payload"]["include_domains"] == ["example.com"]
    assert captured["payload"]["language"] == "en"
    assert captured["payload"]["filter_by_language"] is True
    assert output == {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com",
                "content": "Relevant content",
            }
        ]
    }


@pytest.mark.asyncio
async def test_tavily_tool_uses_model_requested_max_results(monkeypatch):
    captured = {}

    async def fake_search(payload, **kwargs):
        captured["payload"] = payload
        return {"results": []}

    monkeypatch.setattr(
        "skydiscover.llm.tools.tavily.tavily_search",
        fake_search,
    )
    tool = TavilyTool(
        TavilyToolConfig(
            api_key="tvly-key",
            max_results=5,
            search_depth="basic",
        )
    )

    await tool.execute({"query": "test query", "max_results": 0})

    assert captured["payload"]["max_results"] == 0


@pytest.mark.asyncio
async def test_tavily_tool_preserves_non_json_success_body(monkeypatch):
    async def invalid_json_response(*args, **kwargs):
        raise TavilyResponseDecodeError(200, "upstream proxy returned HTML")

    monkeypatch.setattr(
        "skydiscover.llm.tools.tavily.tavily_search",
        invalid_json_response,
    )
    execution = await TavilyTool(TavilyToolConfig(api_key="tvly-key")).execute_with_metadata(
        {"query": "test query"}
    )

    assert execution.response == {
        "status_code": 200,
        "body": "upstream proxy returned HTML",
    }
    assert execution.error == "Tavily returned non-JSON HTTP 200 response"


@pytest.mark.parametrize(
    ("requested_max_results", "expected_max_results"),
    [(-1, 0), (21, 20)],
)
def test_tavily_tool_bounds_max_results(
    requested_max_results,
    expected_max_results,
):
    tool = TavilyTool(TavilyToolConfig(api_key="tvly-key"))

    payload = tool._build_payload(
        "test query",
        max_results=requested_max_results,
    )

    assert payload["max_results"] == expected_max_results


@pytest.mark.asyncio
async def test_tavily_tool_rejects_non_integer_max_results(monkeypatch):
    search = AsyncMock(return_value={"results": []})
    monkeypatch.setattr(
        "skydiscover.llm.tools.tavily.tavily_search",
        search,
    )
    tool = TavilyTool(TavilyToolConfig(api_key="tvly-key"))

    output = json.loads(await tool.execute({"query": "test query", "max_results": "five"}))

    assert output == {"error": "'max_results' must be an integer."}
    search.assert_not_awaited()


@pytest.mark.asyncio
async def test_tavily_tool_reports_missing_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    tool = TavilyTool(TavilyToolConfig(api_key=None))
    output = json.loads(await tool.execute({"query": "test"}))
    assert "API key" in output["error"]


def test_tavily_tool_output_is_valid_json_within_limit():
    serialized = _format_response(
        {
            "answer": "a" * 2_000,
            "results": [
                {
                    "title": f"Result {index}",
                    "url": f"https://example.com/{index}",
                    "content": "content " * 2_000,
                    "raw_content": "raw " * 5_000,
                }
                for index in range(10)
            ],
        },
        query="large response",
    )
    output = limit_tool_output(serialized, max_chars=500)

    assert len(output) <= 500
    assert json.loads(output)["truncated"] is True


@pytest.mark.asyncio
async def test_tool_execution_limits_result_before_returning_it_to_llm():
    llm = _make_llm(max_output_chars=500)
    llm.tools["tavily"].execute = AsyncMock(
        return_value=json.dumps(
            {
                "results": [
                    {
                        "title": "large result",
                        "url": "https://example.com",
                        "content": "content " * 2_000,
                    }
                ]
            }
        )
    )

    messages = await llm._execute_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "tavily",
                    "arguments": '{"query":"test"}',
                },
            }
        ]
    )

    assert len(messages[0]["content"]) <= 500
    assert json.loads(messages[0]["content"])["truncated"] is True


@pytest.mark.asyncio
async def test_tool_execution_output_limit_is_disabled_by_default():
    llm = _make_llm()
    full_result = json.dumps({"content": "x" * 40_000})
    llm.tools["tavily"].execute = AsyncMock(return_value=full_result)

    messages = await llm._execute_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "tavily",
                    "arguments": '{"query":"test"}',
                },
            }
        ]
    )

    assert llm.tool_execution.max_output_chars is None
    assert messages[0]["content"] == full_result


@pytest.mark.asyncio
async def test_cancelled_tavily_call_remains_in_search_sink():
    llm = _make_llm()
    started = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_cancellation(arguments):
        started.set()
        await release.wait()
        return json.dumps({"results": []})

    llm.tools["tavily"].execute = wait_for_cancellation
    sink = []
    task = asyncio.create_task(
        llm._execute_tool_calls(
            [
                {
                    "id": "cancelled-tool",
                    "type": "function",
                    "function": {
                        "name": "tavily",
                        "arguments": '{"query":"test"}',
                    },
                }
            ],
            llm_call_id="cancelled-llm",
            web_search_result_sink=sink,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sink[0]["tool_call_id"] == "cancelled-tool"
    assert sink[0]["status"] == "cancelled"
    assert sink[0]["request"]["query"] == "test"


@pytest.mark.asyncio
async def test_chat_completions_runs_tool_loop(monkeypatch):
    async def fake_search(payload, **kwargs):
        return {
            "results": [
                {
                    "title": "Search hit",
                    "url": "https://example.com/hit",
                    "content": "The searched fact.",
                    "score": 0.95,
                }
            ]
        }

    monkeypatch.setattr(
        "skydiscover.llm.tools.tavily.tavily_search",
        fake_search,
    )
    llm = _make_llm()
    fallback_sink = []
    llm.default_web_search_result_sink = fallback_sink
    calls = []
    responses = [
        _chat_response(_message(tool_calls=[_tool_call()])),
        _chat_response(_message(content="Final answer grounded in search.")),
    ]

    def create(**params):
        calls.append(params)
        return responses.pop(0)

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = await llm.generate("system", [{"role": "user", "content": "question"}])

    assert result.text == "Final answer grounded in search."
    search = result.web_search_results["tavily_searches"][0]
    assert search["response"]["results"][0]["score"] == 0.95
    assert search["llm_tool_result"] == {
        "results": [
            {
                "title": "Search hit",
                "url": "https://example.com/hit",
                "content": "The searched fact.",
            }
        ]
    }
    assert search["request"]["query"] == "latest circle-packing record"
    assert search["program_id"] is None
    execution = result.llm_reasoning["calls"][0]["tool_executions"][0]
    assert execution["tool_call_id"] == "call-1"
    assert execution["status"] == "success"
    assert execution["result"] == search["llm_tool_result"]
    assert fallback_sink[0]["tool_call_id"] == search["tool_call_id"]
    assert calls[0]["tools"] == [TAVILY_TOOL_SCHEMA]
    assert calls[0]["tool_choice"] == "auto"
    tool_message = calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-1"
    assert json.loads(tool_message["content"])["results"][0]["url"] == ("https://example.com/hit")


@pytest.mark.asyncio
async def test_tavily_captures_reasoning_and_program_search_results(monkeypatch):
    async def fake_search(payload, **kwargs):
        return {
            "results": [
                {
                    "title": "Search hit",
                    "url": "https://example.com/hit",
                    "content": "The searched fact.",
                    "score": 0.95,
                }
            ]
        }

    monkeypatch.setattr(
        "skydiscover.llm.tools.tavily.tavily_search",
        fake_search,
    )
    llm = _make_llm(
        api_base="https://openrouter.ai/api/v1",
        reasoning_effort="medium",
    )
    first_message = _message(tool_calls=[_tool_call()])
    first_message.reasoning = "Search before proposing a construction."
    first_message.reasoning_details = [
        {"type": "reasoning.text", "text": "Need external evidence first."}
    ]
    responses = [
        _chat_response(first_message),
        _chat_response(_message(content="Final answer grounded in search.")),
    ]
    requests = []

    def create(**params):
        requests.append(params)
        return responses.pop(0)

    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = await llm.generate(
        "system",
        [{"role": "user", "content": "question"}],
        llm_context={"iteration": 2, "attempt": 1},
    )

    assert result.text == "Final answer grounded in search."
    assert requests[0]["extra_body"]["reasoning"] == {
        "effort": "medium",
        "exclude": False,
    }
    assert "reasoning_effort" not in requests[0]
    assert requests[1]["messages"][-2]["reasoning_details"][0]["text"] == (
        "Need external evidence first."
    )

    reasoning_call = result.llm_reasoning["calls"][0]
    assert reasoning_call["iteration"] == 2
    assert reasoning_call["attempt"] == 1
    assert reasoning_call["status"] == "success"
    assert reasoning_call["responses"][0]["reasoning"] == (
        "Search before proposing a construction."
    )
    assert reasoning_call["responses"][0]["reasoning_details"] == [
        {"type": "reasoning.text", "text": "Need external evidence first."}
    ]
    assert result.llm_reasoning_content == (
        "Search before proposing a construction.\n\nNeed external evidence first."
    )
    stored = result.web_search_results["tavily_searches"][0]
    assert stored["request"]["query"] == "latest circle-packing record"
    assert stored["config"]["max_results"] == 3
    assert stored["response"]["results"][0]["score"] == 0.95
    assert set(stored["llm_tool_result"]["results"][0]) == {
        "title",
        "url",
        "content",
    }


@pytest.mark.asyncio
async def test_responses_api_runs_tool_loop():
    llm = _make_llm()
    llm.tools["tavily"].execute = AsyncMock(
        return_value=json.dumps({"results": [{"title": "hit"}]})
    )

    reasoning_item = SimpleNamespace(type="reasoning", id="rs-1", summary=[])
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="tavily",
        arguments='{"query":"test"}',
    )
    final_message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(text="Responses final answer")],
    )
    llm._create_responses_completion = AsyncMock(
        side_effect=[
            SimpleNamespace(output=[reasoning_item, function_call]),
            SimpleNamespace(output=[final_message]),
        ]
    )

    result = await llm._call_api_via_responses(
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "tools": [TAVILY_TOOL_SCHEMA],
            "tool_choice": "auto",
        }
    )

    assert result == "Responses final answer"
    first_params = llm._create_responses_completion.call_args_list[0].args[0]
    assert first_params["tools"][0]["name"] == "tavily"
    assert "function" not in first_params["tools"][0]
    second_params = llm._create_responses_completion.call_args_list[1].args[0]
    assert reasoning_item in second_params["input"]
    assert second_params["input"][-1]["type"] == "function_call_output"
    assert second_params["input"][-1]["call_id"] == "call-1"


@pytest.mark.asyncio
async def test_luna_reasoning_with_tavily_routes_directly_to_responses_api():
    llm = _make_llm(
        model_name="gpt-5.6-luna",
        api_base="https://unit.services.ai.azure.com/openai/v1",
        reasoning_effort="medium",
    )
    final_message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(text="Responses final answer")],
    )
    llm._create_chat_completion = AsyncMock(
        side_effect=AssertionError("Chat Completions must not be called")
    )
    llm._create_responses_completion = AsyncMock(
        return_value=SimpleNamespace(output=[final_message])
    )

    result = await llm.generate(
        "system",
        [{"role": "user", "content": "question"}],
        verbosity="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {"type": "object"},
            },
        },
    )

    assert result.text == "Responses final answer"
    llm._create_chat_completion.assert_not_awaited()
    request = llm._create_responses_completion.await_args.args[0]
    assert request["reasoning"] == {"effort": "medium"}
    assert request["text"] == {
        "verbosity": "low",
        "format": {
            "type": "json_schema",
            "name": "answer",
            "strict": True,
            "schema": {"type": "object"},
        },
    }
    assert request["tools"][0]["name"] == "tavily"
    assert "function" not in request["tools"][0]


@pytest.mark.asyncio
async def test_luna_without_reasoning_keeps_chat_completions():
    llm = _make_llm(
        model_name="gpt-5.6-luna",
        api_base="https://unit.services.ai.azure.com/openai/v1",
        reasoning_effort="none",
    )
    llm._create_chat_completion = AsyncMock(
        return_value=_chat_response(_message(content="Chat final answer"))
    )
    llm._create_responses_completion = AsyncMock(
        side_effect=AssertionError("Responses API must not be called")
    )

    result = await llm.generate(
        "system",
        [{"role": "user", "content": "question"}],
    )

    assert result.text == "Chat final answer"
    llm._create_chat_completion.assert_awaited_once()
    llm._create_responses_completion.assert_not_awaited()
    request = llm._create_chat_completion.await_args.args[0]
    assert request["reasoning_effort"] == "none"
