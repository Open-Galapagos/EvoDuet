"""Same-prompt candidate selection, failure isolation, and persistence without API calls."""

import asyncio
import copy
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from skydiscover.config import Config, DatabaseConfig, LLMConfig, LLMModelConfig
from skydiscover.evaluation import EvaluationResult
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.openevolve_native.database import OpenEvolveNativeDatabase
from skydiscover.search.topk.database import TopKDatabase


class _Backend:
    def __init__(self, name, state):
        self.model = name
        self.state = state
        self.score_labels = AsyncMock(return_value={"status": "success", "confidence": 0.8})
        self.score_verbalized = AsyncMock(return_value={"status": "success", "confidence": 0.6})

    async def generate(self, system, messages, **kwargs):
        index = len(self.state.requests)
        request = {
            "model": self.model,
            "system": system,
            "messages": copy.deepcopy(messages),
            **kwargs,
        }
        self.state.requests.append(request)
        call = {
            "llm_call_id": f"call-{index}",
            **kwargs["llm_context"],
            "usage": {"total_tokens": 10},
            "responses": [{"reasoning_content": f"reasoning-{index}"}],
        }
        search = {
            "llm_call_id": f"call-{index}",
            "tool_call_id": f"tool-{index}",
            **kwargs["llm_context"],
            "response": {"results": [{"content": f"evidence-{index}"}]},
        }
        kwargs["reasoning_result_sink"].append(call)
        kwargs["web_search_result_sink"].append(search)
        text = (
            await self.state.generate(index, request)
            if self.state.generate is not None
            else f"```python\nvalue = {index}\n```"
        )
        return LLMResponse(
            text=text,
            llm_reasoning={"calls": [copy.deepcopy(call)]},
            web_search_results={"tavily_searches": [copy.deepcopy(search)]},
        )


def _controller(tmp_path, n=3, *, parent=True, **settings):
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.config = Config(
        num_generations=n, language="python", diff_based_generation=False, **settings
    )
    controller.config.checkpoint_interval = 1
    controller.database = TopKDatabase("topk", DatabaseConfig(db_path=None, log_prompts=True))
    if parent:
        controller.database.add(
            Program(id="parent", solution="value = -10", metrics={"combined_score": -10})
        )
    controller.database.add = Mock(wraps=controller.database.add)
    controller.database.sample = Mock(wraps=controller.database.sample)
    controller.database.log_prompt = Mock(wraps=controller.database.log_prompt)
    controller.output_dir = str(tmp_path)
    controller.evaluation_file = str(tmp_path / "evaluator.py")
    controller.shutdown_event = threading.Event()
    controller.num_context_programs = 0
    controller.feedback_reader = None
    controller.monitor_callback = Mock()
    controller.evoduet = None
    controller.agentic_generator = None
    controller._maybe_run_evoduet = AsyncMock(return_value="")

    def prompt(**kwargs):
        capture = kwargs.get("confidence_context")
        if capture is not None:
            capture.update(parent_program="value = -10", evolutionary_history="prior score -10")
        return {
            "system": "Optimize value",
            "user": "value = -10\n# Task\nImprove it\n"
            + json.dumps(kwargs.get("failed_attempts", [])),
        }

    controller._build_prompt = Mock(side_effect=prompt)
    controller.context_builder = SimpleNamespace(build_prompt=Mock(side_effect=prompt))
    controller.test_state = SimpleNamespace(requests=[], generate=None)
    controller.backends = [_Backend(name, controller.test_state) for name in ("model-a", "model-b")]
    controller.llms = LLMPool(
        [
            LLMModelConfig(name=backend.model, init_client=lambda config, backend=backend: backend)
            for backend in controller.backends
        ]
    )
    controller.llms._sample_model = Mock(return_value=controller.backends[0])

    async def evaluate(solution, program_id):
        return EvaluationResult(metrics={"combined_score": float(solution.split("=")[1])})

    controller.evaluator = SimpleNamespace(evaluate_program=AsyncMock(side_effect=evaluate))
    return controller


@pytest.mark.asyncio
async def test_default_single_generation_preserves_existing_path(tmp_path):
    controller = _controller(tmp_path, n=1)
    result = await controller._run_iteration(1)

    assert result.error is None
    assert len(controller.test_state.requests) == 1
    assert "n" not in controller.test_state.requests[0]
    assert not hasattr(controller, "_generation_batch_runner")
    assert "generation_batches" not in result.child_program_dict["metadata"]
    controller.llms._sample_model.assert_called_once()
    controller.evaluator.evaluate_program.assert_awaited_once()


@pytest.mark.asyncio
async def test_parallel_candidates_share_prompt_model_and_select_only_top_one(tmp_path):
    controller = _controller(tmp_path)
    controller.evoduet = SimpleNamespace(
        staged_search_record_id=Mock(return_value="retrieval-1"), record_usage=Mock()
    )
    controller._maybe_run_evoduet.return_value = "ORACLE evidence"
    all_started = asyncio.Event()
    finished = []

    async def generate(index, request):
        if len(controller.test_state.requests) == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), 2)
        # Completion order must not determine sample IDs or the winner.
        await asyncio.sleep((2 - index) * 0.001)
        finished.append(index)
        return f"```python\nvalue = {[2, 9, 4][index]}\n```"

    async def evaluate(solution, program_id):
        assert len(finished) == 3
        return EvaluationResult(metrics={"combined_score": float(solution.split("=")[1])})

    controller.test_state.generate = generate
    controller.evaluator.evaluate_program.side_effect = evaluate
    result = await controller._run_iteration(7)
    controller._process_iteration_result(result, 7, verbose=False)

    assert result.error is None
    controller.database.sample.assert_called_once()
    controller._build_prompt.assert_called_once()
    controller._maybe_run_evoduet.assert_awaited_once()
    controller.llms._sample_model.assert_called_once()
    assert controller.llms._iteration_model.get() is None
    assert controller.evaluator.evaluate_program.await_count == 3
    requests = controller.test_state.requests
    assert {request["model"] for request in requests} == {"model-a"}
    assert len({request["system"] for request in requests}) == 1
    assert all(request["messages"] == requests[0]["messages"] for request in requests)
    assert "ORACLE evidence" in requests[0]["messages"][0]["content"]
    assert all("n" not in request for request in requests)
    child = controller.database.programs[result.child_program_dict["id"]]
    assert child.metrics == {"combined_score": 9.0}
    assert child.solution == "value = 9"
    assert child.parent_id == "parent"
    assert child.llm_reasoning_content == "reasoning-1"
    controller.database.add.assert_called_once()
    controller.database.log_prompt.assert_called_once()
    controller.evoduet.record_usage.assert_called_once_with(
        iteration=7, score=9.0, feedback=None, result_program_id=child.id
    )
    batch = child.metadata["generation_batches"][0]
    assert batch["selected_index"] == 1
    assert batch["num_evaluated"] == batch["num_valid"] == 3
    assert [row["selected"] for row in batch["candidates"]] == [False, True, False]
    assert len({row["program_id"] for row in batch["candidates"]}) == 3
    assert len(controller.database.programs) == 2
    assert (
        len(child.llm_reasoning["calls"]) == len(child.web_search_results["tavily_searches"]) == 3
    )
    assert sum(call["usage"]["total_tokens"] for call in child.llm_reasoning["calls"]) == 30
    for index, call in enumerate(child.llm_reasoning["calls"]):
        assert call["candidate_program_id"] == batch["candidates"][index]["program_id"]
        assert call["program_id"] == child.id
        assert call["candidate_selected"] is (index == 1)
        assert call["association"] == (
            "generated_program" if index == 1 else "unselected_candidate_for_generated_program"
        )
    checkpoint = tmp_path / "checkpoint"
    controller.database.save(str(checkpoint), iteration=7, write_evolution_trace=True)
    resumed = TopKDatabase("topk", DatabaseConfig(db_path=None))
    resumed.load(str(checkpoint))
    assert resumed.programs[child.id].metadata == child.metadata
    assert resumed.programs[child.id].llm_reasoning == child.llm_reasoning
    trace = json.loads((checkpoint / "evolution_trace.json").read_text())
    assert len(trace["programs"]) == 2
    json.dumps(child.to_dict(), allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("evaluation_limit", [1, 2])
async def test_limits_are_shared_across_concurrent_iterations(tmp_path, evaluation_limit):
    controller = _controller(
        tmp_path, max_parallel_generations=2, max_parallel_evaluations=evaluation_limit
    )
    controller.llms._sample_model.side_effect = controller.backends
    running = {"generate": 0, "evaluate": 0}
    peak = dict(running)

    async def work(kind):
        running[kind] += 1
        peak[kind] = max(peak[kind], running[kind])
        await asyncio.sleep(0.01)
        running[kind] -= 1

    async def generate(index, request):
        await work("generate")
        return f"```python\nvalue = {index}\n```"

    async def evaluate(solution, program_id):
        await work("evaluate")
        return EvaluationResult(metrics={"combined_score": float(solution.split("=")[1])})

    controller.test_state.generate = generate
    controller.evaluator.evaluate_program.side_effect = evaluate
    results = await asyncio.gather(controller._run_iteration(1), controller._run_iteration(2))
    assert all(result.error is None for result in results)
    assert peak == {"generate": 2, "evaluate": evaluation_limit}
    assert running == {"generate": 0, "evaluate": 0}
    assert len(controller.test_state.requests) == 6
    assert controller.evaluator.evaluate_program.await_count == 6
    assert controller.llms._sample_model.call_count == 2
    for iteration, model in ((1, "model-a"), (2, "model-b")):
        assert {
            r["model"]
            for r in controller.test_state.requests
            if r["llm_context"]["iteration"] == iteration
        } == {model}


@pytest.mark.asyncio
async def test_failures_cannot_beat_valid_negative_scores(tmp_path):
    controller = _controller(tmp_path, n=7)

    async def generate(index, request):
        if index == 0:
            raise RuntimeError("provider unavailable")
        if index == 1:
            return ""
        return f"```python\nvalue = {index}\n```"

    async def evaluate(solution, program_id):
        index = int(solution.split("=")[1])
        if index == 2:
            raise TimeoutError("evaluation timed out")
        if index == 3:
            return EvaluationResult(metrics={"combined_score": 0, "error": "invalid solution"})
        return EvaluationResult(metrics={"combined_score": {4: float("nan"), 5: -8, 6: -2}[index]})

    controller.test_state.generate = generate
    controller.evaluator.evaluate_program.side_effect = evaluate
    result = await controller._run_iteration(1, retry_times=3)
    assert result.error is None
    assert result.attempts_used == 1
    assert result.child_program_dict["metrics"] == {"combined_score": -2}
    assert len(controller.test_state.requests) == 7
    assert controller.evaluator.evaluate_program.await_count == 5
    batch = result.child_program_dict["metadata"]["generation_batches"][0]
    assert [c["status"] for c in batch["candidates"]] == [
        "generation_error",
        "generation_error",
        "evaluation_error",
        "evaluation_error",
        "evaluation_error",
        "evaluated",
        "evaluated",
    ]
    assert batch["candidates"][4]["metrics"]["combined_score"] is None
    json.dumps(result.child_program_dict, allow_nan=False)


@pytest.mark.asyncio
async def test_diff_candidates_all_use_the_original_parent(tmp_path):
    controller = _controller(tmp_path)
    controller.config.diff_based_generation = True

    async def generate(index, request):
        original = "value = missing" if index == 1 else "value = -10"
        return f"<<<<<<< SEARCH\n{original}\n=======\nvalue = {index}\n>>>>>>> REPLACE"

    controller.test_state.generate = generate
    result = await controller._run_iteration(1)
    assert result.error is None
    assert result.child_program_dict["solution"] == "value = 2"
    assert controller.database.programs["parent"].solution == "value = -10"
    assert controller.evaluator.evaluate_program.await_count == 2
    assert (
        result.child_program_dict["metadata"]["generation_batches"][0]["candidates"][1]["status"]
        == "parse_error"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_all_failed_batches_retry_and_preserve_every_attempt(tmp_path, recover):
    controller = _controller(tmp_path, n=2)
    controller._maybe_run_evoduet.return_value = "ORACLE evidence"

    async def evaluate(solution, program_id):
        index = int(solution.split("=")[1])
        return EvaluationResult(
            metrics=(
                {"combined_score": index}
                if recover and index >= 2
                else {"combined_score": 0, "error": "bad candidate"}
            )
        )

    controller.evaluator.evaluate_program.side_effect = evaluate
    result = await controller._run_iteration(1, retry_times=2)
    controller._process_iteration_result(result, 1, verbose=False)
    assert bool(result.error) is (not recover)
    assert result.attempts_used == 2
    assert controller.evaluator.evaluate_program.await_count == 4
    assert len(controller.test_state.requests) == 4
    controller.database.sample.assert_called_once()
    controller._maybe_run_evoduet.assert_awaited_once()
    controller.llms._sample_model.assert_called_once()
    requests = controller.test_state.requests
    assert requests[0]["messages"] == requests[1]["messages"]
    assert requests[2]["messages"] == requests[3]["messages"]
    assert "bad candidate" in requests[2]["messages"][0]["content"]
    assert "bad candidate" not in requests[0]["messages"][0]["content"]
    child = result.child_program_dict
    assert len(child["metadata"]["generation_batches"]) == 2
    assert len(child["llm_reasoning"]["calls"]) == 4
    if recover:
        assert child["metrics"] == {"combined_score": 3}
        controller.database.add.assert_called_once()
        assert (
            child["llm_reasoning"]["calls"][0]["association"]
            == "failed_retry_for_generated_program"
        )
    else:
        controller.database.add.assert_not_called()
        assert list(controller.database.programs) == ["parent"]
        failed = controller.database._evolution_archive[child["id"]]
        assert failed.metadata["evaluation_status"] == "failed"
        checkpoint = tmp_path / "checkpoint"
        controller.database.save(str(checkpoint), iteration=1)
        resumed = TopKDatabase("topk", DatabaseConfig(db_path=None))
        resumed.load(str(checkpoint))
        assert resumed._evolution_archive[child["id"]].metadata == failed.metadata
        assert list(resumed.programs) == ["parent"]


@pytest.mark.asyncio
async def test_from_scratch_selects_one_and_releases_reserved_model(tmp_path):
    controller = _controller(tmp_path, parent=False)
    result = await controller._run_from_scratch_iteration(1)
    assert result.error is None
    assert result.child_program_dict["parent_id"] is None
    assert result.child_program_dict["metrics"] == {"combined_score": 2}
    assert controller.llms._iteration_model.get() is None
    assert controller.evaluator.evaluate_program.await_count == 3
    controller._process_iteration_result(result, 1, verbose=False)
    controller.database.add.assert_called_once()
    assert len(controller.database.programs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["generation", "oversize", "nan"])
async def test_all_failed_boundary_cases_keep_a_readable_archive(tmp_path, failure):
    controller = _controller(tmp_path, n=2)
    if failure == "generation":

        async def generate(index, request):
            raise RuntimeError("provider unavailable")

        controller.test_state.generate = generate
    elif failure == "oversize":
        controller.config.max_solution_length = 1
    else:
        controller.evaluator.evaluate_program.side_effect = [
            EvaluationResult(metrics={"combined_score": float("nan")}) for _ in range(2)
        ]

    result = await controller._run_iteration(1, retry_times=1)
    assert result.error is not None
    controller._process_iteration_result(result, 1, verbose=False)
    controller.database.add.assert_not_called()
    assert controller.evaluator.evaluate_program.await_count == (2 if failure == "nan" else 0)
    failed = controller.database._evolution_archive[result.child_program_dict["id"]]
    assert len(failed.llm_reasoning["calls"]) == 2
    assert len(failed.metadata["generation_batches"][0]["candidates"]) == 2
    json.dumps(failed.to_dict(), allow_nan=False)


@pytest.mark.asyncio
async def test_native_database_comparator_excludes_features_and_ties_keep_first(tmp_path):
    controller = _controller(tmp_path)
    config = DatabaseConfig(db_path=None)
    config.feature_dimensions = ["complexity", "diversity"]
    database = OpenEvolveNativeDatabase("openevolve_native", config)
    parent = controller.database.programs["parent"]
    database.add(parent, iteration=0)
    database.sample = Mock(return_value=(parent, []))
    controller.database = database
    controller.evaluator.evaluate_program.side_effect = [
        EvaluationResult(metrics={"quality": 2, "complexity": 1000}),
        EvaluationResult(metrics={"quality": 8, "complexity": 0}),
        EvaluationResult(metrics={"quality": 8, "complexity": 1}),
    ]
    result = await controller._run_iteration(1)
    assert result.error is None
    assert result.child_program_dict["solution"] == "value = 1"
    batch = result.child_program_dict["metadata"]["generation_batches"][0]
    assert [c["score"] for c in batch["candidates"]] == [2, 8, 8]


@pytest.mark.asyncio
async def test_multiobjective_selection_uses_configured_minimization(tmp_path):
    controller = _controller(tmp_path)
    controller.database.config.pareto_objectives = ["loss"]
    controller.database.config.higher_is_better = {"loss": False}
    controller.evaluator.evaluate_program.side_effect = [
        EvaluationResult(metrics={"loss": 4}),
        EvaluationResult(metrics={"loss": 1}),
        EvaluationResult(metrics={"loss": 3}),
    ]
    result = await controller._run_iteration(1)
    assert result.error is None
    assert result.child_program_dict["solution"] == "value = 1"
    batch = result.child_program_dict["metadata"]["generation_batches"][0]
    assert [c["score"] for c in batch["candidates"]] == [-4, -1, -3]


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [False, True])
async def test_real_controller_prompt_and_evaluator_keep_one_program_per_iteration(
    tmp_path, parallel
):
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        "from pathlib import Path\n"
        "evaluated_paths = []\n"
        "def evaluate(path):\n"
        "    evaluated_paths.append(path)\n"
        "    return {'combined_score': float(Path(path).read_text().split('=')[1])}\n"
    )
    state = SimpleNamespace(requests=[], generate=None)
    backend = _Backend("local-test", state)
    config = Config(num_generations=3, language="python", diff_based_generation=False)
    config.max_parallel_iterations = 2 if parallel else 1
    config.llm = LLMConfig(
        models=[LLMModelConfig(name="local-test", init_client=lambda cfg: backend)]
    )
    config.evaluator.cascade_evaluation = False
    database = TopKDatabase("topk", DatabaseConfig(db_path=None))
    database.add(Program(id="parent", solution="value = -10", metrics={"combined_score": -10}))
    controller = DiscoveryController(
        DiscoveryControllerInput(config, str(evaluator), database, output_dir=str(tmp_path))
    )
    try:
        loop = (
            controller._run_discovery_parallel if parallel else controller._run_discovery_sequential
        )
        best = await loop(1, 2, retry_times=1)
        assert best.metrics == {"combined_score": 5}
        assert len(state.requests) == 6
        assert len(database.programs) == 3
        paths = controller.evaluator._eval_module.evaluated_paths
        assert len(paths) == len(set(paths)) == 6
        for iteration in (1, 2):
            requests = [r for r in state.requests if r["llm_context"]["iteration"] == iteration]
            assert len(requests) == 3
            assert all(r["messages"] == requests[0]["messages"] for r in requests)
        assert database.unattached_llm_reasoning["calls"] == []
    finally:
        controller.close()


@pytest.mark.asyncio
async def test_confidence_is_assessed_for_each_candidate_before_its_evaluation(tmp_path):
    controller = _controller(tmp_path)
    controller.config.solution_confidence.enabled = True
    controller._maybe_run_evoduet.return_value = "ORACLE evidence"

    async def evaluate(solution, program_id):
        record = next(
            r
            for r in controller.database.solution_confidence_records
            if r["program_id"] == program_id
        )
        assert record["assessment_status"] == "success"
        assert record["assessment_status_verbalized"] == "success"
        assert record["candidate_score"] is None
        return EvaluationResult(metrics={"combined_score": float(solution.split("=")[1])})

    controller.evaluator.evaluate_program.side_effect = evaluate
    result = await controller._run_iteration(1)
    assert result.error is None
    records = controller.database.solution_confidence_records
    assert len(records) == len({r["record_id"] for r in records}) == 3
    assert {r["sample_index"] for r in records} == {0, 1, 2}
    assert all(r["phase"] == "completed" and r["parent_score"] == -10 for r in records)
    assert all(r["generation_prompt"] == result.prompt for r in records)
    assert all(r["web_document"] == "ORACLE evidence" for r in records)
    assert [r["candidate_solution"] for r in records] == ["value = 0", "value = 1", "value = 2"]
    assert result.child_program_dict["solution_confidence"] == records[2]
    assert controller.backends[0].score_labels.await_count == 3
    assert controller.backends[0].score_verbalized.await_count == 3
    json.dumps(records, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["generate", "evaluate"])
async def test_cancellation_drains_all_siblings_and_retains_call_records(tmp_path, phase):
    controller = _controller(tmp_path, max_parallel_evaluations=3)
    controller.config.solution_confidence.enabled = True
    all_started = asyncio.Event()
    running = set()
    cancelled = set()

    async def block(index):
        running.add(index)
        if len(running) == 3:
            all_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(index)
            raise
        finally:
            running.remove(index)

    if phase == "generate":

        async def generate(index, request):
            await block(index)

        controller.test_state.generate = generate
    else:

        async def evaluate(solution, program_id):
            await block(program_id)

        controller.evaluator.evaluate_program.side_effect = evaluate

    task = asyncio.create_task(controller._run_iteration(1))
    await asyncio.wait_for(all_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert running == set()
    assert len(cancelled) == 3
    controller.database.add.assert_not_called()
    assert all(r["phase"] == "completed" for r in controller.database.solution_confidence_records)
    parent = controller.database.programs["parent"]
    assert (
        len(parent.llm_reasoning["calls"]) == len(parent.web_search_results["tavily_searches"]) == 3
    )
    assert all(
        c["association"] == "cancelled_generation_attempt" for c in parent.llm_reasoning["calls"]
    )
