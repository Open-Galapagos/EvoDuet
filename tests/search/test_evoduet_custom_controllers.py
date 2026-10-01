import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skydiscover.search.adaevolve.controller import AdaEvolveController
from skydiscover.search.base_database import Program
from skydiscover.search.evox.controller import CoEvolutionController
from skydiscover.search.gepa_native.controller import GEPANativeController
from skydiscover.search.utils.discovery_utils import SerializableResult


def _program(program_id="parent", score=0.25):
    return Program(
        id=program_id,
        solution="seed",
        language="python",
        metrics={"combined_score": score},
        iteration_found=0,
    )


def _startup_probes(controller):
    controller.evoduet = object()
    controller._preflight_evoduet = Mock()


def test_custom_controller_entrypoints_preflight_before_other_work():
    parent = _program()

    ada = AdaEvolveController.__new__(AdaEvolveController)
    _startup_probes(ada)
    ada.database = SimpleNamespace(
        programs={parent.id: parent},
        num_islands=1,
        log_status=lambda: None,
        get_best_program=lambda: parent,
    )
    ada.shutdown_event = SimpleNamespace(is_set=lambda: False)
    ada._setup_iteration_stats_logging = lambda: None
    ada._ensure_all_islands_seeded = lambda: None
    ada._iteration_stats_log_path = None
    asyncio.run(ada.run_discovery(0, 0))

    gepa = GEPANativeController.__new__(GEPANativeController)
    _startup_probes(gepa)
    gepa.database = SimpleNamespace(
        programs={parent.id: parent},
        name="gepa",
        get_best_program=lambda: parent,
    )
    gepa.shutdown_event = SimpleNamespace(is_set=lambda: False)
    asyncio.run(gepa.run_discovery(0, 0))

    evox = CoEvolutionController.__new__(CoEvolutionController)
    _startup_probes(evox)
    evox.database = SimpleNamespace(
        programs={parent.id: parent},
        name="evox",
        get_statistics=lambda **kwargs: {},
        get_best_program=lambda: parent,
    )
    evox.shutdown_event = SimpleNamespace(is_set=lambda: False)
    evox._switch_interval = 1
    evox._reset_search_window = lambda: None
    evox._generate_variation_operators = AsyncMock()
    evox._pending_search_result = None
    asyncio.run(evox.run_discovery(0, 0))

    for controller in (ada, gepa, evox):
        controller._preflight_evoduet.assert_called_once_with()


def test_evox_preflights_inner_evoduet_before_model_work():
    controller = CoEvolutionController.__new__(CoEvolutionController)
    controller.evoduet = None
    controller._evoduet_preflighted = False
    controller.search_controller = SimpleNamespace(
        _preflight_evoduet=Mock(side_effect=RuntimeError("missing search key"))
    )
    controller._generate_variation_operators = AsyncMock()

    with pytest.raises(RuntimeError, match="missing search key"):
        asyncio.run(controller.run_discovery(0, 0))

    controller.search_controller._preflight_evoduet.assert_called_once_with()
    controller._generate_variation_operators.assert_not_awaited()


@pytest.mark.parametrize("iteration_error", [None, "failed"])
def test_evox_checkpoints_consumed_retry_boundary_after_recording_search_steps(iteration_error):
    events = []
    controller = CoEvolutionController.__new__(CoEvolutionController)
    controller.evoduet = None
    controller._preflight_evoduet = Mock()
    controller.database = SimpleNamespace(
        programs={},
        name="evox",
        get_statistics=lambda **_kwargs: {},
        get_best_program=lambda: None,
        log_status=lambda: events.append("status"),
    )
    controller.config = SimpleNamespace(
        checkpoint_interval=5,
        search=SimpleNamespace(switch_interval=1),
    )
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    controller._switch_interval = 1
    controller._pending_search_result = None
    controller._fallback_database = None
    controller._reset_search_window = lambda: None
    controller._generate_variation_operators = AsyncMock()

    async def run_iteration(iteration, retry_times):
        events.append(("run", iteration, retry_times))
        return SerializableResult(
            child_program_dict={"id": "child"},
            attempts_used=3,
            error=iteration_error,
        )

    controller._run_iteration = run_iteration
    controller._process_iteration_result = (
        lambda _result, _iteration, callback, **_kwargs: events.append(("processed", callback))
    )
    controller._record_search_window_step = lambda: events.append("scored")
    controller._should_evolve_search = lambda: False

    asyncio.run(
        controller.run_discovery(
            5,
            3,
            checkpoint_callback=lambda iteration: events.append(("checkpoint", iteration)),
        )
    )

    assert events == [
        ("run", 5, 3),
        ("processed", None),
        "scored",
        "scored",
        "scored",
        "status",
        ("checkpoint", 7),
    ]


def test_evox_caps_retries_at_remaining_iteration_budget():
    calls = []
    controller = CoEvolutionController.__new__(CoEvolutionController)
    controller.evoduet = None
    controller._preflight_evoduet = Mock()
    controller.database = SimpleNamespace(
        programs={},
        name="evox",
        get_statistics=lambda **_kwargs: {},
        get_best_program=lambda: None,
        log_status=lambda: None,
    )
    controller.config = SimpleNamespace(
        checkpoint_interval=10,
        search=SimpleNamespace(switch_interval=1),
    )
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    controller._switch_interval = 1
    controller._pending_search_result = None
    controller._reset_search_window = lambda: None
    controller._generate_variation_operators = AsyncMock()

    async def run_iteration(iteration, retry_times):
        calls.append((iteration, retry_times))
        return SerializableResult(child_program_dict={"id": "child"}, attempts_used=3)

    controller._run_iteration = run_iteration
    controller._process_iteration_result = lambda *_args, **_kwargs: None
    controller._record_search_window_step = lambda: calls.append("scored")
    controller._should_evolve_search = lambda: False

    asyncio.run(controller.run_discovery(7, 1))

    assert calls == [(7, 1), "scored"]


def _evox_loop_controller(events, checkpoint_interval=10):
    controller = CoEvolutionController.__new__(CoEvolutionController)
    controller.evoduet = None
    controller._preflight_evoduet = Mock()
    controller.database = SimpleNamespace(
        programs={},
        name="evox",
        get_statistics=lambda **_kwargs: {},
        get_best_program=lambda: None,
        log_status=lambda: events.append("status"),
    )
    controller.config = SimpleNamespace(
        checkpoint_interval=checkpoint_interval,
        search=SimpleNamespace(switch_interval=1),
    )
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    controller._switch_interval = 1
    controller._pending_search_result = None
    controller._fallback_database = None
    controller._reset_search_window = lambda: None
    controller._generate_variation_operators = AsyncMock()
    controller._process_iteration_result = lambda *_args, **_kwargs: None
    controller._record_search_window_step = lambda: None
    return controller


def test_evox_search_evolution_failure_does_not_skip_solution_iteration():
    calls = []
    controller = _evox_loop_controller(calls)

    async def run_iteration(iteration, retry_times):
        calls.append(iteration)
        return SerializableResult(child_program_dict={"id": f"child-{iteration}"})

    controller._run_iteration = run_iteration
    controller._should_evolve_search = lambda: True
    controller._evolve_search = AsyncMock(side_effect=RuntimeError("meta search failed"))

    asyncio.run(controller.run_discovery(1, 3))

    assert calls == [1, 2, 3]
    assert controller.last_processed_iteration == 3


def test_evox_checkpoint_failure_propagates_without_consuming_next_iteration():
    events = []
    iterations = []
    controller = _evox_loop_controller(events, checkpoint_interval=1)

    async def run_iteration(iteration, retry_times):
        iterations.append(iteration)
        return SerializableResult(child_program_dict={"id": f"child-{iteration}"})

    controller._run_iteration = run_iteration
    controller._should_evolve_search = lambda: False

    def fail_checkpoint(_iteration):
        raise OSError("checkpoint failed")

    with pytest.raises(OSError, match="checkpoint failed"):
        asyncio.run(controller.run_discovery(1, 2, checkpoint_callback=fail_checkpoint))

    assert iterations == [1]


def test_adaevolve_checkpoints_after_end_iteration():
    events = []
    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.evoduet = None
    controller._preflight_evoduet = Mock()
    controller.database = SimpleNamespace(
        programs={},
        num_islands=1,
        end_iteration=lambda iteration: events.append(("ended", iteration)),
        log_status=lambda: events.append("status"),
        get_best_program=lambda: None,
    )
    controller.config = SimpleNamespace(checkpoint_interval=1)
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    controller._setup_iteration_stats_logging = lambda: None
    controller._ensure_all_islands_seeded = lambda: None
    controller._iteration_stats_log_path = None

    async def run_iteration(iteration, callback):
        events.append(("ran", iteration, callback))
        return False

    controller._run_iteration = run_iteration
    asyncio.run(
        controller.run_discovery(
            1,
            1,
            checkpoint_callback=lambda iteration: events.append(("checkpoint", iteration)),
        )
    )

    assert events == [
        ("ran", 1, None),
        ("ended", 1),
        "status",
        ("checkpoint", 1),
        "status",
    ]


def test_adaevolve_reuses_one_evoduet_step_across_retries():
    parent = _program()

    class Database:
        programs = {parent.id: parent}
        use_paradigm_breakthrough = False
        current_island = 0
        use_adaptive_search = False
        fixed_intensity = 0.5
        _last_sampling_mode = "balanced"

        def __init__(self):
            self.sample_calls = 0

        def sample(self, *_args, **_kwargs):
            self.sample_calls += 1
            return {"parent": parent}, {"context": []}

        def get_children(self, _parent_id):
            return []

    database = Database()
    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.llms = SimpleNamespace(release_model=Mock())
    controller.database = database
    controller.num_context_programs = 1
    controller.enable_retry = True
    controller.max_retries = 1
    controller._prompt_context = {}
    controller.feedback_reader = SimpleNamespace(
        mode="replace",
        set_current_prompt=Mock(),
        read=Mock(return_value="replacement"),
        apply_feedback=Mock(
            side_effect=lambda _prompt: {
                "system": "replacement system",
                "user": "feedback replacement\n# Task\nwrite code",
            }
        ),
        log_usage=Mock(),
    )
    controller._ensure_all_islands_seeded = lambda: None
    controller.context_builder = SimpleNamespace(
        build_prompt=lambda *_args, **_kwargs: {
            "system": "system",
            "user": "mutation context\n# Task\nwrite code",
        }
    )
    controller._maybe_run_evoduet = AsyncMock(return_value="retrieved evidence")
    controller._execute_generation = AsyncMock(
        side_effect=[
            SerializableResult(error="retry", iteration=4),
            SerializableResult(child_program_dict={"id": "child"}, iteration=4),
        ]
    )

    result = asyncio.run(controller._run_normal_step(4))

    assert not result.error
    controller.llms.release_model.assert_called_once_with(4)
    assert database.sample_calls == 1
    # The stub builder records no history, so the layer receives an empty scaffold history.
    controller._maybe_run_evoduet.assert_awaited_once_with(parent, "", 4)
    prompts = [call.args[1]["user"] for call in controller._execute_generation.await_args_list]
    assert all("retrieved evidence" in prompt for prompt in prompts)
    assert all("feedback replacement" in prompt for prompt in prompts)
    assert all("mutation context" not in prompt for prompt in prompts)


def test_adaevolve_discards_unsent_evoduet_after_all_retries():
    discarded = []
    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.llms = SimpleNamespace(release_model=Mock())
    controller.enable_retry = True
    controller.max_retries = 1
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda iteration: discarded.append(iteration)
    )

    async def fail_before_send(iteration, *, error_context, iteration_context):
        controller.llms.release_model.assert_not_called()
        iteration_context["evoduet"] = "retrieved evidence"
        return SerializableResult(error=error_context or "prompt failed", iteration=iteration)

    controller._generate_child = fail_before_send

    result = asyncio.run(controller._run_normal_step(5))

    assert result.error.startswith("All 2 attempts failed")
    assert discarded == [5]
    controller.llms.release_model.assert_called_once_with(5)


def test_adaevolve_discards_unsent_evoduet_when_cancelled():
    discarded = []
    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.llms = SimpleNamespace(release_model=Mock())
    controller.enable_retry = False
    controller.max_retries = 0
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda iteration: discarded.append(iteration)
    )

    async def cancel_before_send(iteration, *, error_context, iteration_context):
        iteration_context["evoduet"] = "retrieved evidence"
        raise asyncio.CancelledError

    controller._generate_child = cancel_before_send

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller._run_normal_step(6))

    assert discarded == [6]
    controller.llms.release_model.assert_called_once_with(6)


def test_adaevolve_credits_sent_evoduet_when_cancelled():
    recorded = []
    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.enable_retry = False
    controller.max_retries = 0
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda _iteration: None,
        record_usage=lambda **kwargs: recorded.append(kwargs),
    )

    async def cancel_after_send(iteration, *, error_context, iteration_context):
        iteration_context["evoduet"] = "retrieved evidence"
        iteration_context["evoduet_sent"] = True
        raise asyncio.CancelledError

    controller._generate_child = cancel_after_send

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller._run_normal_step(7))

    assert recorded == [{"iteration": 7, "score": None, "feedback": "iteration cancelled"}]


def test_adaevolve_cancelled_retrieval_discards_pending_evidence_prediction():
    from skydiscover.evoduet.layer import EvoDuet
    from skydiscover.evoduet.retrieval import SearchStore

    layer = EvoDuet.__new__(EvoDuet)
    layer.search_store = SearchStore()
    layer.output_dir = None
    layer._pending_query_optimizations = {}
    trace = {"iteration": 8, "parent_score": 0.25, "status": "awaiting_outcome"}
    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.enable_retry = False
    controller.max_retries = 0
    controller.evoduet = layer

    async def cancel_before_prompt(iteration, *, error_context, iteration_context):
        layer._pending_query_optimizations[iteration] = trace
        iteration_context["evoduet"] = ""
        raise asyncio.CancelledError

    controller._generate_child = cancel_before_prompt
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller._run_normal_step(8))

    assert layer._pending_query_optimizations == {}
    assert trace["status"] == "discarded"
    assert "actual_improvement" not in trace


def test_custom_acceptance_records_evaluated_evoduet_outcomes():
    calls = []
    result = SerializableResult(
        child_program_dict={
            "id": "child",
            "metrics": {"combined_score": 0.1},
            "artifacts": {"feedback": "needs work"},
        }
    )

    ada = AdaEvolveController.__new__(AdaEvolveController)
    ada.evoduet = SimpleNamespace(record_usage=lambda **kwargs: calls.append(kwargs))
    ada._record_evoduet_outcome(result, 7)

    gepa = GEPANativeController.__new__(GEPANativeController)
    gepa.evoduet = SimpleNamespace(record_usage=lambda **kwargs: calls.append(kwargs))
    gepa._record_rejected_evoduet(result, 8)

    assert calls == [
        {
            "iteration": 7,
            "score": 0.1,
            "feedback": "needs work",
            "result_program_id": "child",
        },
        {
            "iteration": 8,
            "score": 0.1,
            "feedback": "needs work",
            "result_program_id": "child",
        },
    ]


def test_adaevolve_evoduet_uses_database_proxy_scores():
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"loss": 1.0},
        iteration_found=0,
    )
    child = Program(
        id="child",
        solution="improved",
        language="python",
        metrics={"loss": 0.5},
        iteration_found=1,
        parent_id=parent.id,
    )
    evicted = Program(
        id="evicted",
        solution="stale",
        language="python",
        metrics={"loss": 0.1},
        iteration_found=2,
    )

    class Database:
        programs = {parent.id: parent, evicted.id: evicted}
        active_programs = {parent.id: parent}

        @staticmethod
        def get_program_proxy_score(program):
            return -program.metrics["loss"]

    run_calls = []
    credit_calls = []

    class WorldKnowledge:
        grounded = False

        @staticmethod
        def preflight():
            return None

        async def run(self, **kwargs):
            run_calls.append(kwargs)
            return []

        @staticmethod
        def record_usage(**kwargs):
            credit_calls.append(kwargs)

    ada = AdaEvolveController.__new__(AdaEvolveController)
    ada.database = Database()
    ada.evoduet = WorldKnowledge()
    ada.config = SimpleNamespace(evoduet=SimpleNamespace(max_document_chars=2_000))
    ada.context_builder = SimpleNamespace(_get_system_message=lambda: "task")
    ada._evoduet_preflighted = False

    scaffold_history = "## Previous Attempts\n\n### Attempt 1\n- Changes: rendered by the builder"
    asyncio.run(ada._maybe_run_evoduet(parent, scaffold_history, 1))
    ada._record_evoduet_outcome(
        SerializableResult(child_program_dict=child.to_dict()),
        1,
        child,
    )

    assert run_calls[0]["target_score"] == -1.0
    assert [program.id for program in run_calls[0]["population"]] == ["parent"]
    assert run_calls[0]["population"][0].metrics["evoduet_score"] == -1.0
    # The scaffold's rendered history reaches the layer exactly as given.
    assert run_calls[0]["history"] == scaffold_history
    assert "evoduet_score" not in parent.metrics
    assert credit_calls[0]["score"] == -0.5


def test_adaevolve_database_add_failure_finalizes_evoduet_credit():
    child = Program(
        id="child",
        solution="candidate",
        language="python",
        metrics={"loss": 0.25},
        parent_id="parent",
    )
    credit_calls = []

    class Database:
        @staticmethod
        def add(*_args, **_kwargs):
            raise RuntimeError("archive write failed")

        @staticmethod
        def get_program_proxy_score(program):
            return -program.metrics["loss"]

    controller = AdaEvolveController.__new__(AdaEvolveController)
    controller.database = Database()
    controller.evoduet = SimpleNamespace(record_usage=lambda **kwargs: credit_calls.append(kwargs))
    result = SerializableResult(
        child_program_dict=child.to_dict(),
        parent_id="parent",
    )

    with pytest.raises(RuntimeError, match="archive write failed"):
        controller._process_result(result, 8, None)

    assert credit_calls == [
        {
            "iteration": 8,
            "score": -0.25,
            "feedback": "database add failed: archive write failed",
            "result_program_id": "child",
        }
    ]


def test_evox_clears_failed_search_controller_evoduet(monkeypatch):
    processed = []

    class SearchController:
        _prompt_context = {}

        async def run_discovery(self, **_kwargs):
            return SerializableResult(error="generation failed", iteration=3)

        async def postprocess_result(self, result, iteration, verbose=True):
            processed.append((result.error, iteration, verbose))

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr("skydiscover.search.evox.controller.handle_generation_failure", no_op)

    controller = CoEvolutionController.__new__(CoEvolutionController)
    controller._num_search_evolutions = 3
    controller.search_controller = SearchController()
    controller._active_search_algorithm_code = "old"
    controller.search_outputs_dir = "."
    controller._build_search_stats = lambda _iteration: {
        "search_algorithm_stats": {},
        "db_stats": {},
    }

    asyncio.run(controller._generate_and_validate_search_algorithm(12))

    assert processed == [("generation failed", 3, False)]
