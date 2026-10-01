"""EvoDuet state follows the selected program checkpoint exactly."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from skydiscover.config import DatabaseConfig
from skydiscover.evoduet.layer import EvoDuet
from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval import SearchResult, SearchStore
from skydiscover.runner import EVODUET_STATE_FILE, Runner
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.default_discovery_controller import DiscoveryController
from skydiscover.search.evox.controller import CoEvolutionController
from skydiscover.search.evox.utils.search_scorer import LogWindowScorer
from skydiscover.search.utils.discovery_utils import SerializableResult


def _document(name: str) -> SearchResult:
    return SearchResult(
        raw_content=f"raw {name}",
        content=f"summary {name}",
        id=name,
    )


def _layer(store: SearchStore) -> EvoDuet:
    layer = object.__new__(EvoDuet)
    layer.search_store = store
    layer.output_dir = None
    return layer


def test_layer_state_replaces_a_newer_live_search_database(tmp_path):
    live = SearchStore(output_dir=tmp_path)
    live.add(
        ConstructedQuery(query="checkpoint query"),
        [_document("checkpoint")],
        iteration=2,
        target_program_id="parent",
        target_score=0.4,
    )
    live.record_usage(2, score=0.7, result_program_id="child")
    checkpoint_state = _layer(live).state_dict()

    live.add(ConstructedQuery(query="future query"), [_document("future")], iteration=9)
    resumed = SearchStore(output_dir=tmp_path)
    assert resumed.records == []

    _layer(resumed).load_state_dict(checkpoint_state)

    assert [document.id for document in resumed.documents()] == ["checkpoint"]
    assert resumed.records[0].impacts[0].delta == pytest.approx(0.3)
    assert resumed._pending == {}
    rows = (tmp_path / "search_database" / "search_result.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["search_results"][0]["id"] == "checkpoint"


def test_layer_checkpoint_contains_only_current_state():
    state = _layer(SearchStore()).state_dict()
    assert set(state) == {"schema_version", "store", "query_optimization_history"}
    assert state["schema_version"] == 1


def test_layer_checkpoint_restores_staged_credit():
    original = SearchStore()
    record = original.add(
        ConstructedQuery(query="pending query"),
        [_document("pending")],
        iteration=3,
        target_program_id="parent",
        target_score=0.4,
    )

    restored = SearchStore()
    _layer(restored).load_state_dict(_layer(original).state_dict())
    restored.record_usage(3, score=0.7, result_program_id="child")

    impact = restored.records[0].impacts[0]
    assert restored.records[0].id == record.id
    assert impact.delta == pytest.approx(0.3)
    assert restored.records[0].search_results[0].visit_count == 1


@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_legacy_layer_checkpoint_is_rejected_without_replacing_state(version):
    store = SearchStore()
    store.add(ConstructedQuery(query="grounding"), [_document("seed")], iteration=None)
    layer = _layer(store)
    before = layer.state_dict()
    legacy = {"store": SearchStore().state_dict(), "grounded": True}
    if version is not None:
        legacy["schema_version"] = version
    with pytest.raises(ValueError, match="unsupported EvoDuet checkpoint format"):
        layer.load_state_dict(legacy)
    assert layer.state_dict() == before


class _Database:
    best_program_id = None

    def save(self, path: str, iteration: int) -> None:
        self.saved = (path, iteration)

    def get_best_program(self):
        return None


def _runner(tmp_path, layer: EvoDuet) -> Runner:
    runner = object.__new__(Runner)
    runner.output_dir = str(tmp_path)
    runner.file_extension = ".py"
    runner.database = _Database()
    runner.discovery_controller = SimpleNamespace(evoduet=layer)
    return runner


def test_runner_saves_and_restores_evoduet_with_program_checkpoint(tmp_path):
    store = SearchStore(output_dir=tmp_path)
    store.add(ConstructedQuery(query="kept"), [_document("kept")], iteration=4)
    runner = _runner(tmp_path, _layer(store))

    runner._save_checkpoint(4)
    checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_4"
    state = json.loads((checkpoint_dir / "evoduet.json").read_text())
    assert state["store"]["records"][0]["search_results"][0]["id"] == "kept"

    store.add(ConstructedQuery(query="newer"), [_document("newer")], iteration=5)
    resumed_store = SearchStore(output_dir=tmp_path)
    resumed = _runner(tmp_path, _layer(resumed_store))
    resumed._restore_evoduet_checkpoint(str(checkpoint_dir))

    assert [document.id for document in resumed_store.documents()] == ["kept"]


def test_missing_checkpoint_state_fails_without_rewriting_live_search_database(tmp_path):
    live = SearchStore(output_dir=tmp_path)
    live.add(ConstructedQuery(query="future"), [_document("future")], iteration=8)
    resumed_store = SearchStore(output_dir=tmp_path)
    checkpoint_dir = tmp_path / "checkpoints" / "checkpoint_2"
    checkpoint_dir.mkdir(parents=True)

    runner = _runner(tmp_path, _layer(resumed_store))
    before = live.path.read_bytes()
    with pytest.raises(RuntimeError, match="Could not load EvoDuet checkpoint state"):
        runner._restore_evoduet_checkpoint(str(checkpoint_dir))

    assert resumed_store.records == []
    assert live.path.read_bytes() == before


def test_evoduet_checkpoint_write_is_atomic_and_fail_closed(tmp_path, monkeypatch):
    store = SearchStore()
    store.add(ConstructedQuery(query="kept"), [_document("kept")], iteration=1)
    runner = _runner(tmp_path, _layer(store))
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    state_path = checkpoint_dir / EVODUET_STATE_FILE
    state_path.write_text("previous")

    def fail_dump(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("skydiscover.runner.json.dump", fail_dump)
    with pytest.raises(OSError, match="disk full"):
        runner._save_evoduet_checkpoint(str(checkpoint_dir))

    assert state_path.read_text() == "previous"
    assert list(checkpoint_dir.glob(f"{EVODUET_STATE_FILE}.tmp-*")) == []


def test_malformed_evoduet_checkpoint_fails_resume(tmp_path):
    store = SearchStore()
    store.add(ConstructedQuery(query="current"), [_document("current")], iteration=2)
    runner = _runner(tmp_path, _layer(store))
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / EVODUET_STATE_FILE).write_text("{not json")

    with pytest.raises(RuntimeError, match="Could not load EvoDuet"):
        runner._restore_evoduet_checkpoint(str(checkpoint_dir))

    assert [document.id for document in store.documents()] == ["current"]


def _evox_controller(layer, *, with_pending: bool):
    search_controller = DiscoveryController.__new__(DiscoveryController)
    search_controller.evoduet = layer
    search_controller.database = SimpleNamespace(
        programs={
            "search-parent": Program(
                id="search-parent",
                solution="search code",
                metrics={"combined_score": 0.2},
            )
        },
        best_program_id="search-parent",
        last_iteration=1,
        initial_program_id="search-parent",
        initial_program_score=0.2,
        prompts_by_program=None,
    )

    controller = CoEvolutionController.__new__(CoEvolutionController)
    controller.evoduet = None
    controller.search_controller = search_controller
    controller.search_scorer = LogWindowScorer("search-parent")
    controller.search_scorer.reset_window(0.2, start_iteration=2)
    controller.search_scorer.record_step(0.5)
    controller._pending_search_result = (
        SerializableResult(
            child_program_dict={
                "id": "search-child",
                "solution": "new search code",
                "metrics": {"combined_score": 0.5},
            },
            iteration=2,
        )
        if with_pending
        else None
    )
    controller._best_search_score = 0.2
    controller._num_search_evolutions = 2
    controller._switch_interval = 4
    controller._stagnant_count = 1
    controller._last_tracked_best_score = 0.5
    controller._diverge_label = "diverge"
    controller._refine_label = "refine"
    controller._search_initial_code = "initial search code"
    controller._active_search_algorithm_code = "new search code"
    return controller


def test_runner_restores_evox_inner_evoduet_and_delayed_credit(tmp_path):
    original_store = SearchStore()
    original_store.add(
        ConstructedQuery(query="improve search strategy"),
        [_document("search-evidence")],
        iteration=2,
        target_program_id="search-parent",
        target_score=0.2,
    )
    original = _evox_controller(_layer(original_store), with_pending=True)
    runner = object.__new__(Runner)
    runner.discovery_controller = original
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    runner._save_evoduet_checkpoint(str(checkpoint_dir))

    restored_store = SearchStore()
    restored = _evox_controller(_layer(restored_store), with_pending=False)
    restored.search_controller.database.programs = {}
    runner.discovery_controller = restored
    runner._restore_evoduet_checkpoint(str(checkpoint_dir))

    assert restored._pending_search_result.child_program_dict["id"] == "search-child"
    assert list(restored.search_controller.database.programs) == ["search-parent"]
    assert restored.search_scorer._best_scores == [0.5]
    restored.search_controller.evoduet.record_usage(
        iteration=2,
        score=0.5,
        result_program_id="search-child",
    )
    assert restored_store.records[0].impacts[0].delta == pytest.approx(0.3)


def test_evox_controller_state_is_checkpointed_when_evoduet_is_disabled():
    controller = _evox_controller(None, with_pending=True)

    state = controller.evoduet_checkpoint_state()

    assert controller.has_evoduet_checkpoint_state() is True
    assert state["format"] == "evox-v1"
    assert state["solution"] is None and state["search"] is None
    assert state["evox"]["tracking"]["active_search_algorithm_code"] == "new search code"
    assert state["evox"]["pending_search_result"]["iteration"] == 2


@pytest.mark.parametrize("legacy", [None, {}, {"store": {"records": []}}])
def test_evox_rejects_flat_or_missing_checkpoint_state(legacy):
    controller = _evox_controller(_layer(SearchStore()), with_pending=True)
    before = controller.evoduet_checkpoint_state()
    with pytest.raises(ValueError, match="unsupported EvoX checkpoint format"):
        controller.load_evoduet_checkpoint_state(legacy)
    assert controller.evoduet_checkpoint_state() == before


def test_evox_search_database_restore_preserves_dynamic_program_subclass():
    @dataclass
    class DynamicSearchProgram(Program):
        search_family: str = ""

    database = SimpleNamespace(_program_class=DynamicSearchProgram)
    controller = CoEvolutionController.__new__(CoEvolutionController)
    controller.search_controller = SimpleNamespace(database=database)
    row = DynamicSearchProgram(
        id="dynamic",
        solution="search code",
        search_family="novelty-guided",
    ).to_dict()

    controller._restore_search_database(
        {
            "programs": [row],
            "best_program_id": "dynamic",
            "last_iteration": 3,
            "initial_program_id": "dynamic",
            "initial_program_score": 0.4,
            "prompts_by_program": None,
        }
    )

    restored = database.programs["dynamic"]
    assert isinstance(restored, DynamicSearchProgram)
    assert restored.search_family == "novelty-guided"


def test_evox_resume_rebuilds_active_and_fallback_search_databases(monkeypatch):
    @dataclass
    class ActiveProgram(Program):
        strategy_tag: str = ""

    class ReplayDatabase(ProgramDatabase):
        marker = "base"

        def __init__(self, name, config):
            super().__init__(name, config)
            self.replayed = []

        def add(self, program, iteration=None, **_kwargs):
            self.programs[program.id] = program
            self.replayed.append(program.id)
            if iteration is not None:
                self.last_iteration = max(self.last_iteration, iteration)
            self._update_best_program(program)
            return program.id

        def sample(self, num_context_programs=4, **_kwargs):
            return next(iter(self.programs.values())), []

    class InitialDatabase(ReplayDatabase):
        marker = "initial"

    class ActiveDatabase(ReplayDatabase):
        marker = "active"

    class FallbackDatabase(ReplayDatabase):
        marker = "fallback"

    def load_saved_database(path):
        code = Path(path).read_text()
        if code == "new search code":
            return ActiveDatabase, ActiveProgram
        if code == "old search code":
            return FallbackDatabase, Program
        raise AssertionError(f"unexpected search code: {code}")

    monkeypatch.setattr(
        "skydiscover.search.evox.controller.load_database_from_file",
        load_saved_database,
    )
    database_config = DatabaseConfig()
    original = _evox_controller(_layer(SearchStore()), with_pending=True)
    original._fallback_search_code = "old search code"
    original.database = ActiveDatabase("evox", database_config)
    original.database.add(
        ActiveProgram(
            id="solution",
            solution="solution code",
            metrics={"combined_score": 0.7},
            iteration_found=4,
            strategy_tag="preserved active field",
        ),
        iteration=4,
    )
    original.database.initial_program_id = "solution"
    original.database.initial_program_score = 0.7
    state = original.evoduet_checkpoint_state()

    restored = _evox_controller(_layer(SearchStore()), with_pending=False)
    restored.config = SimpleNamespace(search=SimpleNamespace(type="evox", database=database_config))
    restored.evaluator = SimpleNamespace(llm_judge=None)
    restored.database = InitialDatabase("evox", database_config)
    restored.database.add(
        Program(
            id="solution",
            solution="solution code",
            metrics={"combined_score": 0.7},
            iteration_found=4,
        ),
        iteration=4,
    )
    restored.database.initial_program_id = "solution"
    restored.database.initial_program_score = 0.7

    restored.load_evoduet_checkpoint_state(state)

    assert restored.database.marker == "active"
    assert restored.database.replayed == ["solution"]
    assert isinstance(restored.database.programs["solution"], ActiveProgram)
    assert restored.database.programs["solution"].strategy_tag == "preserved active field"
    assert restored._fallback_database.marker == "fallback"
    assert restored._fallback_database.replayed == ["solution"]
    assert restored._fallback_search_code == "old search code"
    assert restored._pending_search_result.child_program_dict["id"] == "search-child"
