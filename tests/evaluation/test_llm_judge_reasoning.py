from types import SimpleNamespace

import pytest

from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.base import LLMResponse


class _Templates:
    templates = {
        "judge-system": "Judge the candidate",
        "evaluator_user_message": "{current_program}",
    }

    def get_template(self, name):
        return self.templates[name]


class _Pool:
    weights = [1.0]

    def __init__(self, text):
        self.text = text
        self.kwargs = None

    async def generate_all(self, system_message, messages, **kwargs):
        self.kwargs = kwargs
        kwargs["reasoning_result_sink"].append(
            {
                "llm_call_id": "judge-call",
                "phase": "evaluation_judge",
                "status": "success",
                "responses": [{"reasoning_content": "judge reasoning", "content": self.text}],
            }
        )
        return [LLMResponse(text=self.text)]


def _context_builder():
    return SimpleNamespace(
        template_manager=_Templates(),
        config=SimpleNamespace(evaluator_system_message="judge-system"),
    )


@pytest.mark.asyncio
async def test_llm_judge_returns_reasoning_with_metrics():
    pool = _Pool('{"score": 0.75}')
    result = await LLMJudge(pool, _context_builder()).evaluate("candidate", "program-1")

    assert result.metrics == {"score": 0.75}
    assert result.llm_reasoning_content == "judge reasoning"
    assert result.llm_reasoning["calls"][0]["llm_call_id"] == "judge-call"
    assert pool.kwargs["llm_context"] == {
        "phase": "evaluation_judge",
        "source_program_id": "program-1",
    }


@pytest.mark.asyncio
async def test_llm_judge_preserves_reasoning_when_output_is_invalid():
    result = await LLMJudge(_Pool("not json"), _context_builder()).evaluate(
        "candidate", "program-1"
    )

    assert result is not None
    assert result.metrics == {}
    assert result.llm_reasoning_content == "judge reasoning"
