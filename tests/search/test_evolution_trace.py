import json
from copy import deepcopy

import pytest

from skydiscover.config import DatabaseConfig, OpenEvolveNativeDatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.openevolve_native.database import OpenEvolveNativeDatabase
from skydiscover.search.utils.checkpoint_manager import CheckpointManager


def test_evolution_trace_embeds_complete_program_web_search_results(tmp_path):
    raw_response = {
        "query": "circle packing",
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": [
            {
                "title": "Packing result",
                "url": "https://example.com/packing",
                "content": "Relevant snippet",
                "raw_content": "Full extracted page",
                "score": 0.97,
                "provider_future_field": {"kept": True},
            }
        ],
        "response_time": 0.42,
        "usage": {"credits": 1},
        "request_id": "request-1",
        "provider_future_top_level": [1, 2, 3],
    }
    search = {
        "tool_call_id": "tool-1",
        "llm_call_id": "llm-1",
        "iteration": 1,
        "attempt": 1,
        "round": 0,
        "program_id": "child",
        "status": "success",
        "request": {"query": "circle packing", "max_results": 5},
        "response": raw_response,
        "llm_tool_result": {
            "results": [
                {
                    "title": "Packing result",
                    "url": "https://example.com/packing",
                    "content": "Relevant snippet",
                }
            ]
        },
        "error": None,
    }
    program = Program(
        id="child",
        solution="pass",
        metrics={"combined_score": 0.8},
        iteration_found=1,
        web_search_results={"tavily_searches": [search]},
    )

    manager = CheckpointManager(DatabaseConfig(db_path=str(tmp_path)))
    manager.save(
        {program.id: program},
        prompts_by_program=None,
        best_program_id=program.id,
        last_iteration=1,
        write_evolution_trace=True,
    )

    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    stored = trace["programs"][0]["web_search_results"]["tavily_searches"][0]
    assert stored["response"] == raw_response
    assert set(stored["llm_tool_result"]["results"][0]) == {
        "title",
        "url",
        "content",
    }


def test_program_checkpoint_resume_preserves_web_search_results(tmp_path):
    program = Program(
        id="p1",
        solution="pass",
        web_search_results={
            "tavily_searches": [{"tool_call_id": "tool-1", "response": {"results": []}}]
        },
    )
    manager = CheckpointManager(DatabaseConfig(db_path=str(tmp_path)))
    manager.save(
        {program.id: program},
        prompts_by_program=None,
        best_program_id=program.id,
        last_iteration=2,
    )

    programs, _, _ = manager.load(str(tmp_path))
    assert programs["p1"].web_search_results == program.web_search_results


def test_reasoning_round_trips_through_checkpoint_and_evolution_trace(tmp_path):
    reasoning = {
        "calls": [
            {
                "llm_call_id": "call-1",
                "iteration": 3,
                "attempt": 2,
                "phase": "generation",
                "program_id": "p1",
                "source_program_id": "parent",
                "association": "generated_program",
                "status": "success",
                "request": {
                    "system_message": "system",
                    "messages": [{"role": "user", "content": "question"}],
                    "parameters": {"reasoning_effort": "high"},
                },
                "responses": [
                    {
                        "round": 0,
                        "phase": "tool_request",
                        "content": "",
                        "reasoning": "first reasoning round",
                        "reasoning_details": [{"type": "reasoning.text", "text": "raw detail"}],
                    },
                    {
                        "round": 1,
                        "phase": "final",
                        "content": "final answer",
                        "reasoning_content": "second reasoning round",
                    },
                ],
                "tool_executions": [
                    {
                        "tool_call_id": "tool-1",
                        "name": "tavily",
                        "status": "success",
                        "result": {"results": []},
                    }
                ],
            }
        ]
    }
    program = Program(
        id="p1",
        solution="pass",
        llm_response="final answer",
        llm_reasoning_content="first reasoning round\n\nraw detail\n\nsecond reasoning round",
        llm_reasoning=reasoning,
    )
    manager = CheckpointManager(DatabaseConfig(db_path=str(tmp_path)))
    manager.save(
        {program.id: program},
        prompts_by_program=None,
        best_program_id=program.id,
        last_iteration=3,
        write_evolution_trace=True,
    )

    checkpoint_program = json.loads((tmp_path / "programs" / "p1.json").read_text())
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    programs, _, _ = manager.load(str(tmp_path))

    assert checkpoint_program["llm_reasoning"] == reasoning
    assert checkpoint_program["llm_reasoning_content"] == program.llm_reasoning_content
    assert trace["schema_version"] == 2
    assert trace["programs"][0]["llm_response"] == "final answer"
    assert trace["programs"][0]["llm_reasoning"] == reasoning
    assert programs["p1"].llm_reasoning == reasoning


def test_evolution_trace_uses_logged_response_when_direct_response_is_missing(tmp_path):
    program = Program(
        id="p1",
        solution="pass",
        prompts={"generation": {"responses": ["logged model response"]}},
    )
    manager = CheckpointManager(DatabaseConfig(db_path=str(tmp_path)))

    manager.save(
        {program.id: program},
        prompts_by_program=None,
        best_program_id=program.id,
        last_iteration=1,
        write_evolution_trace=True,
    )

    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert trace["programs"][0]["llm_response"] == "logged model response"


class _Database(ProgramDatabase):
    def add(self, program, iteration=None, **kwargs):
        self.programs[program.id] = program
        return program.id

    def sample(self, num_context_programs=4, **kwargs):
        program = next(iter(self.programs.values()))
        return program, []


def test_checkpoint_resume_preserves_trace_archive_and_unattached_searches(tmp_path):
    config = DatabaseConfig(db_path=None)
    database = _Database("test", config)
    archived = Program(
        id="archived",
        solution="pass",
        web_search_results={"tavily_searches": [{"tool_call_id": "archived-search"}]},
    )
    database._evolution_archive[archived.id] = archived
    database.unattached_web_search_results = {
        "tavily_searches": [{"tool_call_id": "unattached-search"}]
    }
    database.unattached_llm_reasoning = {
        "calls": [{"llm_call_id": "unattached-llm", "responses": []}]
    }
    database.save(str(tmp_path), iteration=4)

    resumed = _Database("test", config)
    resumed.load(str(tmp_path))

    assert resumed._evolution_archive[archived.id].web_search_results == (
        archived.web_search_results
    )
    assert resumed.unattached_web_search_results == (database.unattached_web_search_results)
    assert resumed.unattached_llm_reasoning == database.unattached_llm_reasoning

    resumed.write_evolution_trace(str(tmp_path), iteration=4)
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert trace["unattached_llm_reasoning"] == database.unattached_llm_reasoning


def test_evicted_program_prompts_survive_checkpoint_resume_without_mutating_archive(tmp_path):
    config = OpenEvolveNativeDatabaseConfig(population_size=2, num_islands=1)
    database = OpenEvolveNativeDatabase("openevolve_native", config)
    parent = Program(id="parent", solution="parent code", metrics={"combined_score": 3.0})
    evicted = Program(
        id="evicted",
        parent_id=parent.id,
        solution="evaluated child code",
        metrics={"combined_score": 1.0},
        artifacts={"feedback": "candidate feedback"},
    )
    database.add(parent, iteration=0)
    database.add(evicted, iteration=1)
    database.log_prompt(
        evicted.id,
        "diff_user_message",
        {"system": "system context", "user": "parent and retrieved evidence"},
        ["raw candidate response"],
    )
    database.add(
        Program(id="newest", solution="new code", metrics={"combined_score": 2.0}),
        iteration=2,
    )
    assert evicted.id not in database.programs
    assert database._evolution_archive[evicted.id] is evicted
    expected_prompts = deepcopy(database.prompts_by_program[evicted.id])
    database.write_evolution_trace(str(tmp_path), iteration=2)
    expected_trace = json.loads((tmp_path / "evolution_trace.json").read_text())

    checkpoint = tmp_path / "checkpoint"
    database.save(str(checkpoint), iteration=2)

    assert evicted.prompts is None
    assert database.prompts_by_program[evicted.id] == expected_prompts
    stored = json.loads((checkpoint / "evolution_trace_state.json").read_text())
    assert stored["archived_programs"][0]["prompts"] == expected_prompts

    resumed = OpenEvolveNativeDatabase("openevolve_native", config)
    resumed.load(str(checkpoint))
    assert resumed.prompts_by_program is None
    assert resumed._evolution_archive[evicted.id].prompts == expected_prompts
    resumed.write_evolution_trace(str(tmp_path), iteration=2)
    assert json.loads((tmp_path / "evolution_trace.json").read_text()) == expected_trace

    next_checkpoint = tmp_path / "next_checkpoint"
    resumed.save(str(next_checkpoint), iteration=2)
    restored = OpenEvolveNativeDatabase("openevolve_native", config)
    restored.load(str(next_checkpoint))
    assert restored._evolution_archive[evicted.id].prompts == expected_prompts


@pytest.mark.parametrize("log_prompts", [True, False])
def test_archive_checkpoint_preserves_embedded_prompts(log_prompts, tmp_path):
    embedded = {"generation": {"system": "captured system", "user": "captured user"}}
    program = Program(id="archived", solution="pass", prompts=deepcopy(embedded))
    manager = CheckpointManager(DatabaseConfig(log_prompts=log_prompts))
    manager.save(
        {},
        prompts_by_program={program.id: {"generation": {"user": "separate log"}}},
        best_program_id=None,
        last_iteration=1,
        path=str(tmp_path),
        evolution_archive={program.id: program},
    )

    archived, _, _ = manager.load_evolution_trace_state(str(tmp_path))
    assert archived[program.id].prompts == embedded
    assert program.prompts == embedded


@pytest.mark.parametrize("failed_program_id", [None, "parent", "child"])
def test_trace_deltas_require_valid_parent_and_child_evaluations(failed_program_id, tmp_path):
    parent = Program(id="parent", solution="parent code", metrics={"combined_score": -2.0})
    child = Program(
        id="child", parent_id=parent.id, solution="child code", metrics={"combined_score": -1.0}
    )
    programs = {parent.id: parent, child.id: child}
    if failed_program_id:
        programs[failed_program_id].artifacts["error"] = "evaluation failed"
    manager = CheckpointManager(DatabaseConfig())
    manager._write_evolution_trace(programs, None, 1, str(tmp_path))

    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    entries = {entry["id"]: entry for entry in trace["programs"]}
    if failed_program_id:
        assert entries[failed_program_id]["score"] is None
        assert "improvement_delta" not in entries[child.id]
        assert entries[failed_program_id]["artifacts"]["error"] == "evaluation failed"
    else:
        assert entries[parent.id]["score"] == -2.0
        assert entries[child.id]["score"] == -1.0
        assert entries[child.id]["improvement_delta"] == {"combined_score": 1.0}
