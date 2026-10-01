"""Failed evaluations remain inspectable without becoming search candidates."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skydiscover.config import Config, DatabaseConfig
from skydiscover.evaluation import EvaluationResult
from skydiscover.llm.base import LLMResponse
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.default_discovery_controller import DiscoveryController


class _Database(ProgramDatabase):
    def add(self, program, iteration=None, **kwargs):
        self.programs[program.id] = program
        return program.id

    def sample(self, num_context_programs=4, **kwargs):
        return self.programs["parent"], []


def _controller(output):
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.config = Config()
    controller.config.checkpoint_interval = 1
    controller.config.diff_based_generation = False
    controller.database = _Database("test", DatabaseConfig(db_path=None, log_prompts=True))
    controller.database.add(
        Program(id="parent", solution="seed()", metrics={"combined_score": 0.5})
    )
    controller.database.best_program_id = "parent"
    controller.database.add = Mock(wraps=controller.database.add)
    controller.output_dir = str(output)
    controller.shutdown_event = threading.Event()
    controller.num_context_programs = 0
    controller.feedback_reader = None
    controller.monitor_callback = Mock()
    controller.evoduet = SimpleNamespace(
        staged_search_record_id=Mock(return_value="search-record-1"),
        record_usage=Mock(),
    )
    controller._maybe_run_evoduet = AsyncMock(return_value="Retrieved evidence")
    controller._build_prompt = Mock(
        return_value={"system": "Improve the program", "user": "seed()\n# Task\nImprove it"}
    )
    controller._parse_llm_response = Mock(return_value=("broken_child()", "a change", None))
    controller._call_llm = AsyncMock(
        return_value=LLMResponse(
            text="```python\nbroken_child()\n```",
            llm_reasoning={
                "calls": [
                    {
                        "llm_call_id": "generation-call",
                        "attempt": 1,
                        "responses": [{"reasoning_content": "Try this implementation"}],
                    }
                ]
            },
            web_search_results={
                "tavily_searches": [
                    {
                        "llm_call_id": "generation-call",
                        "tool_call_id": "search-tool",
                        "attempt": 1,
                        "response": {"results": [{"content": "Full search result"}]},
                    }
                ]
            },
        )
    )
    controller.evaluator = SimpleNamespace(
        evaluate_program=AsyncMock(
            return_value=EvaluationResult(
                metrics={"combined_score": -100.0, "error": "correctness check failed"},
                artifacts={"stderr": "AssertionError", "feedback": "The result was incorrect"},
            )
        )
    )
    return controller


@pytest.mark.parametrize("parallel", [False, True])
def test_failed_child_survives_iteration_checkpoint_resume_and_trace(tmp_path, parallel):
    controller = _controller(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "checkpoint_1"

    def save(iteration):
        controller.database.save(str(checkpoint), iteration=iteration, write_evolution_trace=True)
        controller.database.write_evolution_trace(str(tmp_path), iteration=iteration)

    checkpoint_callback = Mock(side_effect=save)
    loop = controller._run_discovery_parallel if parallel else controller._run_discovery_sequential
    best = asyncio.run(loop(1, 1, checkpoint_callback, retry_times=1))

    assert best.id == "parent"
    assert list(controller.database.programs) == ["parent"]
    controller.database.add.assert_not_called()
    controller.monitor_callback.assert_not_called()
    checkpoint_callback.assert_called_once_with(1)
    (child_id,) = controller.database._evolution_archive
    failed = controller.database._evolution_archive[child_id]
    controller.evoduet.record_usage.assert_called_once_with(
        iteration=1,
        score=None,
        feedback="Evaluator failed after 1 attempts: correctness check failed",
        result_program_id=child_id,
    )
    assert failed.metadata["evoduet_search_record_id"] == "search-record-1"
    assert failed.metadata["evaluation_status"] == "failed"
    assert controller.database.programs["parent"].llm_reasoning == {}
    assert controller.database.programs["parent"].web_search_results == {}

    for trace_path in (tmp_path / "evolution_trace.json", checkpoint / "evolution_trace.json"):
        trace = json.loads(trace_path.read_text())
        assert trace["best_program_id"] == "parent"
        entry = next(program for program in trace["programs"] if program["id"] == child_id)
        assert entry["solution"] == "broken_child()"
        assert entry["parent_id"] == "parent"
        assert entry["metrics"] == failed.metrics
        assert entry["artifacts"] == failed.artifacts
        assert entry["llm_response"] == failed.llm_response
        assert entry["llm_reasoning"] == failed.llm_reasoning
        assert entry["web_search_results"] == failed.web_search_results
        assert entry["score"] is None
        assert "improvement_delta" not in entry
        prompt = entry["prompts"]["full_rewrite_user_message"]
        assert "Retrieved evidence" in prompt["user"]
        assert prompt["responses"] == [failed.llm_response]
        assert entry["llm_reasoning"]["calls"][0]["program_id"] == child_id
        assert entry["web_search_results"]["tavily_searches"][0]["program_id"] == child_id

    resumed = _Database("test", DatabaseConfig(db_path=None, log_prompts=True))
    resumed.load(str(checkpoint))
    assert list(resumed.programs) == ["parent"]
    assert not (checkpoint / "programs" / f"{child_id}.json").exists()
    archived = resumed._evolution_archive[child_id]
    assert archived.solution == failed.solution
    assert archived.metrics == failed.metrics
    assert archived.artifacts == failed.artifacts
    assert archived.llm_reasoning == failed.llm_reasoning
    assert archived.web_search_results == failed.web_search_results
    assert archived.prompts["full_rewrite_user_message"] == prompt
    resumed.write_evolution_trace(str(tmp_path), iteration=1)
    resumed_entry = next(
        program
        for program in json.loads((tmp_path / "evolution_trace.json").read_text())["programs"]
        if program["id"] == child_id
    )
    assert resumed_entry["prompts"] == entry["prompts"]
    assert resumed_entry["score"] is None
    assert "improvement_delta" not in resumed_entry


def test_generation_failure_without_child_keeps_parent_call_records(tmp_path):
    controller = _controller(tmp_path)
    controller._parse_llm_response.return_value = (None, None, "No valid code")

    result = asyncio.run(controller._run_iteration(1, retry_times=1))
    controller._process_iteration_result(result, 1, verbose=False)

    assert result.child_program_dict is None
    assert controller.database._evolution_archive == {}
    parent = controller.database.programs["parent"]
    assert parent.llm_reasoning["calls"][0]["program_id"] == parent.id
    assert parent.web_search_results["tavily_searches"][0]["program_id"] == parent.id
    assert parent.llm_reasoning["calls"][0]["association"] == "failed_generation_attempt"
    controller.evoduet.record_usage.assert_called_once_with(
        iteration=1, score=None, feedback=result.error
    )
    controller.evaluator.evaluate_program.assert_not_awaited()
    controller.database.add.assert_not_called()
