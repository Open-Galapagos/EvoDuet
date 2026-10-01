from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skydiscover.llm.agentic_generator import AgenticGenerator
from skydiscover.llm.responses_utils import RESPONSES_OUTPUT_ITEMS_KEY


@pytest.mark.asyncio
async def test_agentic_luna_reasoning_uses_responses_and_preserves_output_items():
    reasoning_item = SimpleNamespace(type="reasoning", id="rs-1", summary=[])
    function_call = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="search",
        arguments='{"pattern":"target"}',
    )
    final_message = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(text="Final answer")],
    )
    responses_create = Mock(
        side_effect=[
            SimpleNamespace(output=[reasoning_item, function_call]),
            SimpleNamespace(output=[final_message]),
        ]
    )
    chat_create = Mock(side_effect=AssertionError("Chat Completions must not be called"))
    model = SimpleNamespace(
        model="gpt-5.6-luna",
        api_base="https://unit.services.ai.azure.com/openai/v1",
        reasoning_effort="medium",
        max_tokens=1000,
        temperature=0.1,
        top_p=0.9,
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
            responses=SimpleNamespace(create=responses_create),
        ),
        _record_chat_response=Mock(),
        _record_responses_response=Mock(),
    )
    pool = SimpleNamespace(
        models=[model],
        weights=[1],
        random_state=SimpleNamespace(choices=lambda *args, **kwargs: [0]),
        get_model_for_context=lambda context: model,
    )
    generator = AgenticGenerator(pool, SimpleNamespace())

    first = await generator._call_llm(
        "system",
        [{"role": "user", "content": "question"}],
    )
    second = await generator._call_llm(
        "system",
        [
            {"role": "user", "content": "question"},
            first,
            {"role": "tool", "tool_call_id": "call-1", "content": "search result"},
        ],
    )

    assert model._use_responses_api is True
    assert first[RESPONSES_OUTPUT_ITEMS_KEY] == [reasoning_item, function_call]
    assert second["content"] == "Final answer"
    chat_create.assert_not_called()
    first_request = responses_create.call_args_list[0].kwargs
    assert first_request["reasoning"] == {"effort": "medium"}
    second_input = responses_create.call_args_list[1].kwargs["input"]
    assert reasoning_item in second_input
    assert second_input[-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "search result",
    }
