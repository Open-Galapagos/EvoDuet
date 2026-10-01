import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skydiscover.config import BeamSearchDatabaseConfig, DatabaseConfig, SolutionConfidenceConfig
from skydiscover.evaluation import EvaluationResult
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.beam_search.database import BeamSearchDatabase
from skydiscover.search.utils.checkpoint_manager import (
    EVOLUTION_TRACE_STATE_FILE,
    CheckpointManager,
)
from skydiscover.solution_confidence import assess, finish, freeze_prompt, new_record


class _Database(ProgramDatabase):
    def add(self, program, iteration=None, **kwargs):
        self.programs[program.id] = program
        return program.id

    def sample(self, num_context_programs=4, **kwargs):
        return next(iter(self.programs.values())), []


def _record(record_id, *, program_id=None, attempt=1, status="evaluation_failed"):
    return {
        "record_id": record_id,
        "iteration": 3,
        "attempt": attempt,
        "program_id": program_id,
        "parent_id": "parent",
        "confidence": 0.8,
        "confidence_verbalized": 0.6 if status == "success" else None,
        "y_t": 1 if status == "success" else None,
        "brier_score": 0.04 if status == "success" else None,
        "brier_score_verbalized": 0.16 if status == "success" else None,
        "assessment_status_verbalized": "success" if status == "success" else "not_measured",
        "assessment_verbalized": {"raw_text": "0.6"} if status == "success" else {},
        "status": status,
        "assessment": {"logprob_true": -0.2231435513142097},
    }


def test_confidence_trace_and_checkpoint_keep_retries_without_population_members(tmp_path):
    database = _Database("test", DatabaseConfig(db_path=None))
    records = [
        _record("attempt-1", status="generation_failed"),
        _record("attempt-2", program_id="failed-candidate", attempt=2),
        _record("attempt-3", program_id="evicted-candidate", attempt=3, status="success"),
    ]
    records[-1].update(
        web_document="The injected web document: {'bound': 0.7}",
        prompt_template="solution_confidence_user_web.txt",
        prompt_version="parent-improvement-oracle-v1",
        prompt_template_verbalized="solution_confidence_verbalized_user_web.txt",
        prompt_version_verbalized="parent-improvement-verbalized-web-v1",
    )
    database.solution_confidence_records = records
    archived = Program(
        id="evicted-candidate",
        solution="pass",
        metrics={"combined_score": 1.0},
        solution_confidence=deepcopy(records[-1]),
    )
    database._evolution_archive[archived.id] = archived

    checkpoint = tmp_path / "checkpoints" / "checkpoint_3"
    database.save(str(checkpoint), iteration=3, write_evolution_trace=True)
    database.write_evolution_trace(str(tmp_path), iteration=3)

    state = json.loads((checkpoint / EVOLUTION_TRACE_STATE_FILE).read_text())
    checkpoint_trace = json.loads((checkpoint / "evolution_trace.json").read_text())
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert state["solution_confidence_records"] == records
    assert checkpoint_trace["solution_confidence_records"] == records
    assert trace["solution_confidence_records"] == records
    assert trace["programs"][0]["solution_confidence"] == records[-1]
    assert [item["id"] for item in trace["programs"]] == ["evicted-candidate"]

    resumed = _Database("test", DatabaseConfig(db_path=None))
    resumed.load(str(checkpoint))
    assert resumed.programs == {}
    assert resumed.solution_confidence_records == records
    assert resumed._evolution_archive[archived.id].solution_confidence == records[-1]
    assert resumed.last_iteration == 3

    resumed.solution_confidence_records[0]["assessment"]["logprob_true"] = -1.0
    assert database.solution_confidence_records[0]["assessment"]["logprob_true"] != -1.0
    assert resumed.checkpoint_manager.load_solution_confidence_records(str(checkpoint)) == records

    resumed.solution_confidence_records.append(_record("attempt-4", attempt=4))
    resumed.write_evolution_trace(str(tmp_path), iteration=4)
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert [item["record_id"] for item in trace["solution_confidence_records"]] == [
        "attempt-1",
        "attempt-2",
        "attempt-3",
        "attempt-4",
    ]


def test_program_confidence_roundtrip_keeps_metrics_separate(tmp_path):
    assessment = _record("attempt-1", program_id="child", status="success")
    program = Program(
        id="child",
        solution="pass",
        metrics={"combined_score": 0.7},
        solution_confidence=assessment,
    )
    encoded = program.to_dict()
    encoded["solution_confidence"]["assessment"]["logprob_true"] = -1.0
    assert program.solution_confidence == assessment
    assert program.solution_confidence["assessment"]["logprob_true"] != -1.0

    manager = CheckpointManager(DatabaseConfig(db_path=str(tmp_path)))
    manager.save({program.id: program}, None, program.id, 3, write_evolution_trace=True)
    programs, best_id, iteration = manager.load(str(tmp_path))
    assert programs[program.id].solution_confidence == assessment
    assert programs[program.id].metrics == {"combined_score": 0.7}
    assert best_id == program.id
    assert iteration == 3
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert trace["programs"][0]["solution_confidence"] == assessment


@pytest.mark.parametrize("document", ["", "Selected source {'bound': 0.7}"])
def test_both_assessments_persist_under_selected_output_and_resume(tmp_path, document):
    output = tmp_path / "selected_output"
    checkpoint = output / "checkpoints" / "checkpoint_3"
    settings = SolutionConfidenceConfig(enabled=True)
    parent = Program(id="parent", solution="parent()", metrics={"combined_score": 0.5})
    record = new_record(
        settings,
        parent=parent,
        iteration=3,
        attempt=1,
        run_id=str(output),
        task="test",
        seed=42,
    )
    freeze_prompt(record, {"user": "parent()\n" + document}, web_document=document)
    model = SimpleNamespace(
        score_labels=AsyncMock(return_value={"status": "success", "confidence": 0.8}),
        score_verbalized=AsyncMock(
            return_value={"status": "success", "confidence": 0.6, "raw_text": "0.6"}
        ),
    )
    asyncio.run(
        assess(
            record,
            settings,
            generation_model=model,
            candidate="candidate()",
            program_id="child",
            seen_before=False,
        )
    )
    finish(record, EvaluationResult(metrics={"combined_score": 0.7}))
    assert record["confidence"] == 0.8
    assert record["confidence_verbalized"] == 0.6
    assert record["brier_score"] == pytest.approx(0.04)
    assert record["brier_score_verbalized"] == pytest.approx(0.16)
    assert record["status"] == "completed"

    database = _Database("test", DatabaseConfig(db_path=None))
    database.solution_confidence_records = [record]
    database.add(
        Program(
            id="child",
            parent_id=parent.id,
            solution="candidate()",
            metrics={"combined_score": 0.7},
            solution_confidence=deepcopy(record),
        )
    )
    database.save(str(checkpoint), iteration=3)
    database.write_evolution_trace(str(output), iteration=3)

    trace = json.loads((output / "evolution_trace.json").read_text())
    state = json.loads((checkpoint / EVOLUTION_TRACE_STATE_FILE).read_text())
    candidate = json.loads((checkpoint / "programs" / "child.json").read_text())
    assert trace["solution_confidence_records"] == state["solution_confidence_records"] == [record]
    assert trace["programs"][0]["solution_confidence"] == candidate["solution_confidence"] == record
    resumed = _Database("test", DatabaseConfig(db_path=None))
    resumed.load(str(checkpoint))
    assert resumed.solution_confidence_records == [record]
    assert resumed.programs["child"].solution_confidence == record


def test_checkpoint_and_trace_use_same_confidence_snapshot(tmp_path, monkeypatch):
    database = _Database("test", DatabaseConfig(db_path=None))
    database.add(Program(id="child", solution="pass"))
    database.solution_confidence_records = [_record("attempt-1")]
    expected = deepcopy(database.solution_confidence_records)
    save_program = database.checkpoint_manager._save_program

    def mutate_while_saving(*args, **kwargs):
        database.solution_confidence_records[0]["assessment"]["logprob_true"] = -100.0
        database.solution_confidence_records.append(_record("later-attempt"))
        return save_program(*args, **kwargs)

    monkeypatch.setattr(database.checkpoint_manager, "_save_program", mutate_while_saving)
    database.save(str(tmp_path), iteration=3, write_evolution_trace=True)

    state = json.loads((tmp_path / EVOLUTION_TRACE_STATE_FILE).read_text())
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert state["solution_confidence_records"] == expected
    assert trace["solution_confidence_records"] == expected
    assert len(database.solution_confidence_records) == 2


def test_legacy_checkpoint_without_confidence_loads_and_clears_previous_records(tmp_path):
    programs_path = tmp_path / "programs"
    programs_path.mkdir()
    (programs_path / "legacy.json").write_text(
        json.dumps({"id": "legacy", "solution": "pass", "metrics": {"combined_score": 0.5}})
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"best_program_id": "legacy", "last_iteration": 2})
    )
    database = _Database("test", DatabaseConfig(db_path=None))
    database.solution_confidence_records = [_record("previous-run")]
    database.load(str(tmp_path))
    assert database.solution_confidence_records == []
    assert database.programs["legacy"].solution_confidence == {}
    assert database.checkpoint_manager.load_evolution_trace_state(str(tmp_path)) == ({}, {}, {})

    database.write_evolution_trace(str(tmp_path), iteration=2)
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert "solution_confidence_records" not in trace


@pytest.mark.parametrize(
    "contents",
    [
        "{",
        "[]",
        "{}",
        '{"solution_confidence_records": null}',
        '{"solution_confidence_records": {}}',
    ],
)
def test_invalid_or_legacy_confidence_state_is_ignored(tmp_path, contents):
    (tmp_path / EVOLUTION_TRACE_STATE_FILE).write_text(contents)
    manager = CheckpointManager(DatabaseConfig(db_path=None))
    assert manager.load_solution_confidence_records(str(tmp_path)) == []


def test_legacy_trace_state_api_keeps_three_return_values(tmp_path):
    manager = CheckpointManager(DatabaseConfig(db_path=None))
    manager._save_evolution_trace_state(str(tmp_path), {}, {}, {})
    assert manager.load_evolution_trace_state(str(tmp_path)) == ({}, {}, {})
    assert manager.load_solution_confidence_records(str(tmp_path)) == []


def test_confidence_state_ignores_non_record_entries(tmp_path):
    record = _record("attempt-1")
    (tmp_path / EVOLUTION_TRACE_STATE_FILE).write_text(
        json.dumps({"solution_confidence_records": [None, record, "invalid", 42]})
    )
    manager = CheckpointManager(DatabaseConfig(db_path=None))
    assert manager.load_solution_confidence_records(str(tmp_path)) == [record]


def test_beam_search_checkpoint_resumes_attempt_ledger_and_program_assessment(tmp_path):
    config = BeamSearchDatabaseConfig(db_path=None, beam_width=1)
    database = BeamSearchDatabase("beam_search", config)
    parent = Program(id="parent", solution="pass", metrics={"combined_score": 0.5})
    database.add(parent, iteration=0)
    failure = _record("failed-attempt", program_id="failed-candidate")
    success = _record("final-attempt", program_id="child", attempt=2, status="success")
    child = Program(
        id="child",
        parent_id="parent",
        solution="pass",
        metrics={"combined_score": 1.0},
        solution_confidence=deepcopy(success),
        iteration_found=3,
    )
    database.add(child, iteration=3)
    database.solution_confidence_records = [failure, success]
    database.save(str(tmp_path), iteration=3)

    resumed = BeamSearchDatabase(
        "beam_search", BeamSearchDatabaseConfig(db_path=str(tmp_path), beam_width=1)
    )
    assert resumed.solution_confidence_records == [failure, success]
    assert resumed.programs["child"].solution_confidence == success
    assert "failed-candidate" not in resumed.programs
    assert resumed.beam == database.beam
    assert resumed.depth == database.depth
    assert resumed.last_iteration == 3

    resumed.write_evolution_trace(str(tmp_path), iteration=3)
    trace = json.loads((tmp_path / "evolution_trace.json").read_text())
    assert trace["solution_confidence_records"] == [failure, success]


def test_beam_search_legacy_checkpoint_clears_previous_attempt_records(tmp_path):
    config = BeamSearchDatabaseConfig(db_path=None)
    database = BeamSearchDatabase("beam_search", config)
    database.add(Program(id="parent", solution="pass", metrics={"combined_score": 0.5}))
    database.save(str(tmp_path), iteration=0)
    (tmp_path / EVOLUTION_TRACE_STATE_FILE).unlink()

    database.solution_confidence_records = [_record("previous-run")]
    database.load(str(tmp_path))
    assert database.solution_confidence_records == []
