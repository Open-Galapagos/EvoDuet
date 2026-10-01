from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.openai import OpenAILLM


def _llm() -> OpenAILLM:
    config = LLMModelConfig(
        name="reasoning-model",
        api_base="http://localhost:1234/v1",
        api_key="test",
        max_tokens=100,
        timeout=10,
        retries=0,
    )
    with patch("skydiscover.llm.openai.openai.OpenAI"):
        return OpenAILLM(config)


@pytest.mark.asyncio
async def test_chat_reasoning_is_returned_without_tools():
    llm = _llm()
    message = SimpleNamespace(
        content="answer",
        tool_calls=[],
        reasoning_content="provider-visible reasoning",
        reasoning_details=[{"type": "reasoning.text", "text": "structured detail"}],
    )
    response = SimpleNamespace(
        id="response-1",
        model="reasoning-model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
    )
    llm._create_chat_completion = AsyncMock(return_value=response)

    result = await llm.generate(
        "system",
        [{"role": "user", "content": "question"}],
        llm_context={"iteration": 4, "attempt": 2, "phase": "generation"},
    )

    call = result.llm_reasoning["calls"][0]
    assert call["iteration"] == 4
    assert call["status"] == "success"
    assert call["request"]["system_message"] == "system"
    assert call["request"]["messages"] == [{"role": "user", "content": "question"}]
    assert call["responses"][0]["reasoning_content"] == "provider-visible reasoning"
    assert call["responses"][0]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    assert result.llm_reasoning_content == ("provider-visible reasoning\n\nstructured detail")


@pytest.mark.asyncio
async def test_failed_call_updates_the_supplied_checkpoint_sink():
    llm = _llm()
    sink = []
    llm._create_chat_completion = AsyncMock(side_effect=RuntimeError("provider failed"))

    with pytest.raises(RuntimeError, match="provider failed"):
        await llm.generate(
            "system",
            [{"role": "user", "content": "question"}],
            reasoning_result_sink=sink,
            llm_context={"iteration": 5, "attempt": 1},
        )

    assert len(sink) == 1
    assert sink[0]["status"] == "error"
    assert sink[0]["error"] == "provider failed"
    assert sink[0]["completed_at"]
