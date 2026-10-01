"""Small integration contracts around the current EvoDuet core."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skydiscover.config import Config
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.evoduet.layer import (
    _program_target,
)
from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval import SearchResult, SearchStore
from skydiscover.evoduet.retrieval_gating import HeuristicRetrievalGating
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    _insert_evoduet,
)
from skydiscover.search.utils.discovery_utils import SerializableResult


def _document() -> SearchResult:
    return SearchResult(raw_content="raw", content="content", id="doc")


def test_live_search_database_is_only_an_output_mirror(tmp_path):
    first = SearchStore(output_dir=tmp_path)
    first.add(ConstructedQuery(query="stale"), [_document()])

    fresh = SearchStore(output_dir=tmp_path)

    assert fresh.records == []
    assert (tmp_path / "search_database" / "search_result.jsonl").read_text() == ""


def test_evidence_is_inserted_before_the_final_task_section():
    prompt = "# Context\nbase\n# Task\nsolve it"

    combined = _insert_evoduet(prompt, "# Retrieved external knowledge\nevidence")

    assert combined.index("Retrieved external knowledge") < combined.index("# Task")
    assert combined.endswith("# Task\nsolve it")


def test_evoduet_receives_the_scaffold_history_the_prompt_builder_rendered():
    """The layer sees the mutation prompt's own history sections, untouched."""
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"combined_score": 0.3},
    )
    context = Program(
        id="context",
        solution="other approach",
        language="python",
        metrics={"combined_score": 0.2},
        metadata={"changes": "Change 1: tried another approach"},
    )
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.config = Config()
    controller.config.language = "python"
    controller.context_builder = DefaultContextBuilder(controller.config)
    controller._prompt_context = {}
    controller.database = SimpleNamespace(
        programs={parent.id: parent, context.id: context},
        get_statistics=lambda **_kwargs: {"previous_programs": [context]},
    )

    rendered = controller._scaffold_evolutionary_history(parent, {"": [context]})

    expected = controller.context_builder.build_prompt(
        current_program=parent,
        context={
            "program_metrics": parent.metrics,
            "other_context_programs": {"": [context]},
            "previous_programs": [context],
        },
    )["user"]
    assert rendered
    assert rendered in expected
    assert "### Attempt 1" in rendered
    assert "Change 1: tried another approach" in rendered
    assert "### Program 1 (combined_score: 0.2000)" in rendered
    assert "other approach" in rendered
    assert "evoduet_score" not in rendered
    assert "iteration" not in rendered


def test_scaffold_history_is_empty_without_a_context_builder():
    controller = DiscoveryController.__new__(DiscoveryController)

    assert controller._scaffold_evolutionary_history(SimpleNamespace(id="p"), []) == ""


def test_evoduet_population_uses_database_live_set():
    controller = DiscoveryController.__new__(DiscoveryController)
    old, live = SimpleNamespace(id="old"), SimpleNamespace(id="live")
    controller.database = SimpleNamespace(
        programs={"old": old, "live": live},
        get_beam_programs=lambda: [live],
    )

    assert controller._evoduet_population() == [live]

    controller.database = SimpleNamespace(
        programs={"old": old, "live": live},
        elite_pool=["live"],
    )
    assert controller._evoduet_population() == [live]


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
def test_evoduet_score_fallbacks_reject_non_finite_values(invalid_score):
    program = Program(
        id="invalid",
        solution="seed",
        metrics={"combined_score": invalid_score},
    )
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = SimpleNamespace()

    assert controller._evoduet_program_score(program) is None
    assert _program_target(program) == ("invalid", None)


def test_parallel_checkpoint_waits_for_every_iteration_through_boundary():
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    controller.config = SimpleNamespace(checkpoint_interval=2)
    controller.database = SimpleNamespace(
        name="test", log_status=lambda: None, get_best_program=lambda: None
    )
    started = []
    processed = []
    checkpoints = []

    async def run_iteration(iteration, retry_times):
        started.append(iteration)
        await asyncio.sleep(0)
        return SerializableResult(iteration=iteration)

    controller._run_iteration = run_iteration
    controller._process_iteration_result = (
        lambda result, iteration, checkpoint_callback: processed.append(iteration)
    )

    def checkpoint(iteration):
        checkpoints.append((iteration, list(started), sorted(processed)))

    asyncio.run(
        controller._run_discovery_parallel(
            start_iteration=1,
            max_iterations=3,
            checkpoint_callback=checkpoint,
            max_parallel=3,
        )
    )

    assert checkpoints == [(2, [1, 2], [1, 2])]
    assert started == [1, 2, 3]


def test_sequential_failure_still_checkpoints_completed_boundary():
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    controller.config = SimpleNamespace(checkpoint_interval=1)
    controller.database = SimpleNamespace(
        name="test", log_status=lambda: None, get_best_program=lambda: None
    )
    controller.evoduet = None
    controller._run_iteration = AsyncMock(
        return_value=SerializableResult(error="failed", iteration=1)
    )
    checkpoints = []

    asyncio.run(
        controller._run_discovery_sequential(
            start_iteration=1,
            max_iterations=1,
            checkpoint_callback=checkpoints.append,
        )
    )

    assert checkpoints == [1]
    assert controller.last_processed_iteration == 1


def test_parallel_cancellation_cancels_inflight_work_and_clears_credit():
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    controller.config = SimpleNamespace(checkpoint_interval=10)
    controller.database = SimpleNamespace(get_best_program=lambda: None)
    started = []
    processed = []
    credited = []
    all_started = asyncio.Event()

    async def run_iteration(iteration, retry_times):
        started.append(iteration)
        if len(started) == 2:
            all_started.set()
        await asyncio.Event().wait()

    controller._run_iteration = run_iteration
    controller._process_iteration_result = lambda *args: processed.append(args)
    controller.evoduet = SimpleNamespace(record_usage=lambda **kwargs: credited.append(kwargs))

    async def cancel_run():
        task = asyncio.create_task(
            controller._run_discovery_parallel(
                start_iteration=1,
                max_iterations=4,
                max_parallel=2,
            )
        )
        await all_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_run())

    assert started == [1, 2]
    assert processed == []
    assert {entry["iteration"] for entry in credited} == {1, 2}
    assert all(entry["feedback"] == "iteration cancelled" for entry in credited)


def test_autonomous_claude_search_rejects_unsupported_evoduet_combo():
    from skydiscover.search.route import get_discovery_controller

    controller_input = SimpleNamespace(
        config=SimpleNamespace(
            search=SimpleNamespace(type="claude_code_local"),
            evoduet=SimpleNamespace(enabled=True),
            validate_generation_mode=lambda: None,
        )
    )

    with pytest.raises(ValueError, match="not supported"):
        get_discovery_controller(controller_input)


def test_evoduet_config_is_serialized_without_credentials():
    config = Config()
    config.evoduet.enabled = True
    config.evoduet.tavily_retrieval.api_key = "do-not-save"

    saved = config.to_dict()["evoduet"]

    assert saved["enabled"] is True
    assert saved["query_optimization_max_rounds"] == 3
    assert "api_key" not in saved["tavily_retrieval"]
    restored = Config.from_dict({"evoduet": saved})
    assert restored.evoduet.search_selection.policy == "delta"


def test_heuristic_gate_is_seeded_per_iteration():
    config = SimpleNamespace(
        evoduet=SimpleNamespace(
            gating_retrieve_probability=0.5,
            random_seed=41,
        )
    )
    first = HeuristicRetrievalGating(config)
    second = HeuristicRetrievalGating(config)

    async def decisions(gate):
        return [
            await gate.decide(parent=None, history=None, search_store=None, iteration=iteration)
            for iteration in range(8)
        ]

    assert asyncio.run(decisions(first)) == asyncio.run(decisions(second))


def test_failed_iteration_clears_staged_evoduet_credit():
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.shutdown_event = SimpleNamespace(is_set=lambda: False)
    calls = []
    controller.evoduet = SimpleNamespace(record_usage=lambda **kwargs: calls.append(kwargs))
    controller._finalize_discovery = lambda: None

    async def run_iteration(iteration, retry_times):
        return SerializableResult(error="generation failed", iteration=iteration)

    controller._run_iteration = run_iteration
    asyncio.run(
        controller._run_discovery_sequential(
            start_iteration=3,
            max_iterations=1,
            post_process_result=True,
        )
    )

    assert calls == [{"iteration": 3, "score": None, "feedback": "generation failed"}]


def test_prompt_render_failure_discards_staged_evoduet(monkeypatch):
    controller = DiscoveryController.__new__(DiscoveryController)
    parent = SimpleNamespace(id="parent", metrics={"combined_score": 0.3})
    discarded = []

    class WorldKnowledge:
        grounded = False

        async def run(self, **kwargs):
            return [_document()]

        def discard_usage(self, iteration):
            discarded.append(iteration)

    controller.evoduet = WorldKnowledge()
    controller._evoduet_preflighted = True
    controller.config = SimpleNamespace(evoduet=SimpleNamespace(max_document_chars=2_000))
    controller.database = SimpleNamespace(programs={"parent": parent})
    monkeypatch.setattr(
        "skydiscover.evoduet.format_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    evidence = asyncio.run(controller._maybe_run_evoduet(parent, [], 7))

    assert evidence == ""
    assert discarded == [7]


def test_generation_prompt_failure_discards_evidence_that_was_never_sent():
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"combined_score": 0.3},
    )
    discarded = []
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **_kwargs: (parent, []),
    )
    controller.num_context_programs = 0
    controller.feedback_reader = None
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda iteration: discarded.append(iteration)
    )
    controller._maybe_run_evoduet = AsyncMock(return_value="retrieved evidence")
    controller._build_prompt = Mock(side_effect=RuntimeError("prompt failed"))

    result = asyncio.run(controller._run_iteration(9))

    assert result.error == "prompt failed"
    assert discarded == [9]


def test_feedback_replacement_happens_before_evoduet_is_inserted():
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"combined_score": 0.3},
    )
    discarded = []

    class Feedback:
        mode = "replace"

        @staticmethod
        def set_current_prompt(_prompt):
            return None

        @staticmethod
        def read():
            return "replace it"

        @staticmethod
        def apply_feedback(_prompt):
            return {"system": "replacement system", "user": "replacement\n# Task\nsolve"}

        @staticmethod
        def log_usage(*_args):
            return None

    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **_kwargs: (parent, []),
    )
    controller.num_context_programs = 0
    controller.config = SimpleNamespace(language="python")
    controller.feedback_reader = Feedback()
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda iteration: discarded.append(iteration)
    )
    controller._maybe_run_evoduet = AsyncMock(return_value="retrieved evidence")
    controller._build_prompt = Mock(
        return_value={"system": "original system", "user": "original\n# Task\nsolve"}
    )
    controller._call_llm = AsyncMock(side_effect=RuntimeError("stop after send"))

    result = asyncio.run(controller._run_iteration(10))

    sent_system, sent_user = controller._call_llm.await_args.args[:2]
    assert result.error == "LLM generation failed: stop after send"
    assert sent_system == "replacement system"
    assert "replacement" in sent_user and "retrieved evidence" in sent_user
    assert "original" not in sent_user
    assert sent_user.index("retrieved evidence") < sent_user.index("# Task")
    assert discarded == []


def test_feedback_failure_discards_evoduet_before_model_send():
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"combined_score": 0.3},
    )
    discarded = []
    feedback = SimpleNamespace(
        set_current_prompt=lambda _prompt: None,
        read=lambda: "replace it",
        apply_feedback=Mock(side_effect=RuntimeError("feedback failed")),
    )
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **_kwargs: (parent, []),
    )
    controller.num_context_programs = 0
    controller.feedback_reader = feedback
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda iteration: discarded.append(iteration)
    )
    controller._maybe_run_evoduet = AsyncMock(return_value="retrieved evidence")
    controller._build_prompt = Mock(return_value={"system": "system", "user": "prompt"})
    controller._call_llm = AsyncMock()

    result = asyncio.run(controller._run_iteration(11))

    assert result.error == "feedback failed"
    controller._call_llm.assert_not_awaited()
    assert discarded == [11]


def test_exception_after_retry_preserves_consumed_attempt_count():
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"combined_score": 0.3},
    )
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **_kwargs: (parent, []),
    )
    controller.num_context_programs = 0
    controller.config = SimpleNamespace(language="python")
    controller.feedback_reader = None
    controller.evoduet = None
    controller._maybe_run_evoduet = AsyncMock(return_value="")
    controller._build_prompt = Mock(
        side_effect=[
            {"system": "system", "user": "prompt"},
            RuntimeError("second prompt failed"),
        ]
    )
    controller._call_llm = AsyncMock(return_value=SimpleNamespace(text="invalid"))
    controller._parse_llm_response = Mock(return_value=(None, None, "parse failed"))

    result = asyncio.run(controller._run_iteration(7, retry_times=3))

    assert result.error == "second prompt failed"
    assert result.attempts_used == 2


def test_iteration_cancellation_before_generation_discards_staged_evidence():
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"combined_score": 0.3},
    )
    staged = asyncio.Event()
    discarded = []
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **_kwargs: (parent, []),
    )
    controller.num_context_programs = 0
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda iteration: discarded.append(iteration)
    )
    # The layer runs after the first prompt build, so the build must succeed first.
    controller._build_prompt = Mock(return_value={"system": "system", "user": "prompt"})

    async def retrieve_then_wait(*_args):
        staged.set()
        await asyncio.Event().wait()

    controller._maybe_run_evoduet = retrieve_then_wait

    async def cancel_iteration():
        task = asyncio.create_task(controller._run_iteration(10))
        await staged.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_iteration())

    assert discarded == [10]


def test_iteration_cancellation_after_generation_send_credits_staged_evidence():
    parent = Program(
        id="parent",
        solution="seed",
        language="python",
        metrics={"combined_score": 0.3},
    )
    recorded = []
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = SimpleNamespace(
        programs={parent.id: parent},
        sample=lambda **_kwargs: (parent, []),
    )
    controller.num_context_programs = 0
    controller.config = SimpleNamespace(language="python")
    controller.feedback_reader = None
    controller.evoduet = SimpleNamespace(
        discard_usage=lambda _iteration: None,
        record_usage=lambda **kwargs: recorded.append(kwargs),
    )
    controller._maybe_run_evoduet = AsyncMock(return_value="retrieved evidence")
    controller._build_prompt = Mock(return_value={"system": "system", "user": "prompt"})
    controller._call_llm = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(controller._run_iteration(12))

    assert recorded == [{"iteration": 12, "score": None, "feedback": "iteration cancelled"}]


def test_database_add_failure_finalizes_evoduet_credit():
    controller = DiscoveryController.__new__(DiscoveryController)
    calls = []

    class Database:
        _program_class = Program

        def add(self, child, iteration):
            raise RuntimeError("write failed")

    controller.database = Database()
    controller.evoduet = SimpleNamespace(record_usage=lambda **kwargs: calls.append(kwargs))
    result = SerializableResult(
        child_program_dict={
            "id": "child",
            "solution": "code",
            "language": "python",
            "metrics": {"combined_score": 0.7},
            "iteration_found": 4,
        }
    )

    with pytest.raises(RuntimeError, match="write failed"):
        controller._process_iteration_result(result, 4)

    assert calls == [
        {
            "iteration": 4,
            "score": pytest.approx(0.7),
            "feedback": "database add failed: write failed",
            "result_program_id": "child",
        }
    ]
