"""
Discovery controller for running discovery processes.

Provides the default execution loop for discovery processes (sample → prompt → LLM → evaluate).
Subclasses only need to override ``run_discovery`` to change orchestration
(e.g. co-evolution interleaves solution and search-algorithm evolution).
"""

import asyncio
import copy
import logging
import math
import multiprocessing as mp
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from skydiscover.config import Config
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.context_builder.evox import EvoxContextBuilder
from skydiscover.evaluation import create_evaluator, evaluation_failure_reason
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.llm.base import LLMResponse
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.llm.label_scoring import finite_json
from skydiscover.llm.response_metadata import json_safe, reasoning_content
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.utils.discovery_utils import SerializableResult, build_image_content
from skydiscover import solution_confidence
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


def _insert_evoduet(prompt: str, evidence: str) -> str:
    """Insert untrusted evidence before the final task section when present."""
    marker = "\n# Task"
    index = prompt.rfind(marker)
    block = f"\n\n{evidence}"
    return prompt[:index] + block + prompt[index:] if index >= 0 else prompt + block


@dataclass
class DiscoveryControllerInput:
    """Input to the discovery controller"""

    config: Config
    evaluation_file: str
    database: ProgramDatabase
    file_suffix: str = ".py"
    output_dir: Optional[str] = None
    evaluator_env_vars: Optional[Dict[str, str]] = None


class DiscoveryController:
    """
    Discovery controller with a default sequential execution strategy.

    Handles the full generate-evaluate cycle: prompt building, LLM calls,
    response parsing, evaluation, and result processing.

    The default ``run_discovery`` runs iterations sequentially.  Subclasses
    (e.g. CoEvolutionController) can override it for different orchestration
    while reusing the shared iteration primitives.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        self.config = controller_input.config
        self.evaluation_file = controller_input.evaluation_file
        self.database = controller_input.database
        self.file_suffix = controller_input.file_suffix
        self.output_dir = controller_input.output_dir
        self.evaluator_env_vars = controller_input.evaluator_env_vars

        self.config.validate_generation_mode()
        if self.config.num_generations > 1 and type(self) is not DiscoveryController:
            raise ValueError("num_generations > 1 requires the standard discovery controller")

        if self._solution_confidence_enabled():
            if type(self) is not DiscoveryController:
                raise ValueError(
                    "solution_confidence currently supports the standard discovery controller "
                    "(openevolve_native, topk, best_of_n, beam_search)."
                )
            if self.config.language == "image" or self.config.agentic.enabled:
                raise ValueError("solution_confidence requires non-agentic text/code generation.")

        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False
        self.last_processed_iteration: Optional[int] = None

        self.llms = LLMPool(self.config.llm.models)
        self.evaluator_llms = LLMPool(self.config.llm.evaluator_models)
        self.guide_llms = LLMPool(self.config.llm.guide_models)
        self._configure_llm_result_fallback_sinks()

        self._init_context_builder()
        self._configure_llm_result_fallback_sinks()

        self.config.evaluator.evaluation_file = self.evaluation_file
        self.config.evaluator.file_suffix = self.file_suffix
        self.config.evaluator.is_image_mode = self.config.language == "image"

        llm_judge = None
        if self.config.evaluator.llm_as_judge:
            ctx = DefaultContextBuilder(self.config)
            ctx.set_templates("evaluator_system_message")
            llm_judge = LLMJudge(self.evaluator_llms, ctx, self.database)

        self.evaluator = create_evaluator(
            self.config.evaluator,
            llm_judge=llm_judge,
            max_concurrent=max(self.config.max_parallel_iterations, 4),
            env_vars=controller_input.evaluator_env_vars,
        )

        self.agentic_generator = None
        if self.config.agentic.enabled:
            from skydiscover.llm.agentic_generator import AgenticGenerator

            self.agentic_generator = AgenticGenerator(self.llms, self.config.agentic)
            logger.debug(f"Agentic mode enabled (codebase: {self.config.agentic.codebase_root})")

        # Optional EvoDuet (built only when enabled; lazy import).
        self.evoduet = None
        if self.config.evoduet.enabled:
            from skydiscover.evoduet import EvoDuet

            self.evoduet = EvoDuet(self.config, output_dir=self.output_dir)
            logger.info("EvoDuet enabled (Tavily retrieval)")
            self._configure_llm_result_fallback_sinks()
        self._evoduet_preflighted = False

        self.num_context_programs = controller_input.config.search.num_context_programs

        self.monitor_callback: Optional[Callable] = None
        self.feedback_reader: Optional[Any] = None
        self._prompt_context: Dict[str, Any] = {}

        # Load evaluator/task description and inject into system message so
        # the LLM knows what problem to solve (especially for from-scratch).
        self._inject_evaluator_context()

        logger.debug(
            f"DiscoveryController initialized: num_context_programs={self.num_context_programs}"
        )

    def close(self):
        """Release resources held by the evaluator (e.g. Docker containers)."""
        if hasattr(self.evaluator, "close"):
            self.evaluator.close()

    def _evaluation_failure_reason(self, metrics, artifacts=None):
        """Use an explicitly registered evaluator formatter, else legacy feedback."""
        evaluate_function = getattr(self.evaluator, "evaluate_function", None)
        formatter = getattr(evaluate_function, "__dict__", {}).get(
            "_skydiscover_failure_reason"
        )
        if formatter is None:
            return evaluation_failure_reason(metrics, artifacts)
        return formatter(metrics, artifacts)

    def _configure_llm_result_fallback_sinks(self) -> None:
        """Retain auxiliary Tavily and reasoning results without a Program owner."""
        container = self.database.unattached_web_search_results
        searches = container.setdefault("tavily_searches", [])
        if not isinstance(searches, list):
            searches = []
            container["tavily_searches"] = searches
        reasoning_container = self.database.unattached_llm_reasoning
        reasoning_calls = reasoning_container.setdefault("calls", [])
        if not isinstance(reasoning_calls, list):
            reasoning_calls = []
            reasoning_container["calls"] = reasoning_calls

        pools = [self.llms, self.evaluator_llms, self.guide_llms]
        context_builder = getattr(self, "context_builder", None)
        summary_pool = getattr(context_builder, "summary_llm", None)
        if summary_pool is not None:
            pools.append(summary_pool)
        evoduet = getattr(self, "evoduet", None)
        if evoduet is not None:
            for component_name in ("analyzer", "gating", "query_construction", "retrieval"):
                component = getattr(evoduet, component_name, None)
                pool = getattr(component, "llm", None)
                if pool is not None:
                    pools.append(pool)

        seen: set[int] = set()
        for pool in pools:
            for model in getattr(pool, "models", []):
                if id(model) in seen:
                    continue
                seen.add(id(model))
                if "tavily" in getattr(model, "tools", {}):
                    model.default_web_search_result_sink = searches
                model.default_reasoning_result_sink = reasoning_calls

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _inject_evaluator_context(self):
        """Load evaluator/task description and prepend to the system message.

        For Harbor tasks this loads instruction.md; for containerized benchmarks
        it loads the evaluator source files. The content gives the LLM essential
        context about the problem it needs to solve.

        Controlled by ``evaluator.inject_evaluator_context`` (default False).
        """
        if not self.config.evaluator.inject_evaluator_context:
            return

        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        task_description = load_evaluator_code(self.evaluation_file)
        if not task_description:
            return

        ctx = self.config.context_builder
        existing = ctx.system_message or ""
        # Prepend the task description so the LLM always sees it.
        ctx.system_message = (
            f"# Task Description\n\n{task_description}\n\n{existing}"
            if existing
            else f"# Task Description\n\n{task_description}"
        )

    def _init_context_builder(self):
        """Initialize the appropriate context builder based on config."""
        if getattr(self.config.context_builder, "template", "default") == "evox":
            self.context_builder = EvoxContextBuilder(self.config)
            template_name = "search_evolution_user_message"
            self.context_builder.set_templates(user_template=template_name)
        else:
            self.context_builder = DefaultContextBuilder(self.config)

    async def _call_llm(self, system_message: str, user_message: str, **kwargs) -> LLMResponse:
        """Call the LLM, using agentic mode if enabled (text-only)."""
        if self.agentic_generator and not kwargs.get("image_output"):
            sink = kwargs.get("reasoning_result_sink")
            before = len(sink) if isinstance(sink, list) else 0
            text = await self.agentic_generator.generate(
                system_message,
                user_message,
                reasoning_result_sink=sink,
                llm_context=kwargs.get("llm_context") or {},
            )
            if text:
                calls = sink[before:] if isinstance(sink, list) else []
                return LLMResponse(
                    text=text,
                    llm_reasoning={"calls": copy.deepcopy(calls)},
                    llm_reasoning_content=reasoning_content(calls),
                )
        return await self.llms.generate(
            system_message, [{"role": "user", "content": user_message}], **kwargs
        )

    def _get_generation_batch_runner(self):
        """Allocate shared concurrency limits only when multi-generation is enabled."""
        if not hasattr(self, "_generation_batch_runner"):
            from skydiscover.search.generation_batch import CandidateBatchRunner

            self._generation_batch_runner = CandidateBatchRunner(self.config)
        return self._generation_batch_runner

    @staticmethod
    def _merge_web_search_results(target: Dict[str, Any], source: Optional[Dict[str, Any]]) -> None:
        """Merge Tavily records without duplicating one tool invocation."""
        if not isinstance(source, dict):
            return
        incoming = source.get("tavily_searches")
        if not isinstance(incoming, list) or not incoming:
            return
        current = target.setdefault("tavily_searches", [])
        existing = {
            (item.get("llm_call_id"), item.get("tool_call_id"))
            for item in current
            if isinstance(item, dict)
        }
        for item in incoming:
            if not isinstance(item, dict):
                continue
            key = (item.get("llm_call_id"), item.get("tool_call_id"))
            if key in existing:
                continue
            current.append(copy.deepcopy(item))
            existing.add(key)

    @staticmethod
    def _bind_web_search_results(
        web_search_results: Dict[str, Any],
        *,
        program_id: Optional[str],
        source_program_id: Optional[str],
        association: str,
        successful_attempt: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bind call-time Tavily records to their eventual trace owner."""
        bound = copy.deepcopy(web_search_results or {})
        searches = bound.get("tavily_searches")
        if not isinstance(searches, list):
            return {}
        for search in searches:
            if not isinstance(search, dict):
                continue
            search["program_id"] = program_id
            search["source_program_id"] = source_program_id
            if (
                association == "generated_program"
                and successful_attempt is not None
                and search.get("attempt") != successful_attempt
            ):
                search["association"] = "failed_retry_for_generated_program"
            elif association == "generated_program" and search.get("candidate_selected") is False:
                search["association"] = "unselected_candidate_for_generated_program"
            else:
                search["association"] = association
        return bound

    def _store_unowned_web_search_results(
        self,
        web_search_results: Dict[str, Any],
        *,
        source_program_id: Optional[str],
        association: str,
    ) -> None:
        """Keep searches from an attempt that produced no persisted child."""
        if not web_search_results:
            return
        source_program = self.database.get(source_program_id) if source_program_id else None
        if source_program is None and source_program_id:
            source_program = getattr(self.database, "_evolution_archive", {}).get(source_program_id)
        bound = self._bind_web_search_results(
            web_search_results,
            program_id=source_program.id if source_program is not None else None,
            source_program_id=source_program_id if source_program is not None else None,
            association=(
                association
                if source_program is not None or association.startswith("unattached_")
                else f"unattached_{association}"
            ),
        )
        if source_program is not None:
            if not isinstance(source_program.web_search_results, dict):
                source_program.web_search_results = {}
            self._merge_web_search_results(source_program.web_search_results, bound)
            return

        unattached = getattr(self.database, "unattached_web_search_results", None)
        if not isinstance(unattached, dict):
            unattached = {}
            self.database.unattached_web_search_results = unattached
        self._merge_web_search_results(unattached, bound)

    @staticmethod
    def _merge_llm_reasoning(target: Dict[str, Any], source: Optional[Dict[str, Any]]) -> None:
        """Merge complete LLM call records without duplicating a call."""
        if not isinstance(source, dict):
            return
        incoming = source.get("calls")
        if not isinstance(incoming, list) or not incoming:
            return
        current = target.setdefault("calls", [])
        if not isinstance(current, list):
            current = []
            target["calls"] = current
        existing = {item.get("llm_call_id") for item in current if isinstance(item, dict)}
        for item in incoming:
            if not isinstance(item, dict):
                continue
            call_id = item.get("llm_call_id")
            if call_id in existing:
                continue
            current.append(copy.deepcopy(item))
            existing.add(call_id)

    @staticmethod
    def _bind_llm_reasoning(
        llm_reasoning: Dict[str, Any],
        *,
        program_id: Optional[str],
        source_program_id: Optional[str],
        association: str,
        successful_attempt: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bind LLM call records to their eventual Program or failed attempt."""
        bound = copy.deepcopy(llm_reasoning or {})
        calls = bound.get("calls")
        if not isinstance(calls, list):
            return {}
        for call in calls:
            if not isinstance(call, dict):
                continue
            call["program_id"] = program_id
            call["source_program_id"] = source_program_id
            if (
                association == "generated_program"
                and successful_attempt is not None
                and call.get("attempt") != successful_attempt
            ):
                call["association"] = "failed_retry_for_generated_program"
            elif association == "generated_program" and call.get("candidate_selected") is False:
                call["association"] = "unselected_candidate_for_generated_program"
            else:
                call["association"] = association
        return bound

    def _store_unowned_llm_reasoning(
        self,
        llm_reasoning: Dict[str, Any],
        *,
        source_program_id: Optional[str],
        association: str,
    ) -> None:
        """Keep reasoning from an LLM call that produced no persisted child."""
        if not isinstance(llm_reasoning, dict) or not llm_reasoning.get("calls"):
            return
        source_program = self.database.get(source_program_id) if source_program_id else None
        if source_program is None and source_program_id:
            source_program = getattr(self.database, "_evolution_archive", {}).get(source_program_id)
        bound = self._bind_llm_reasoning(
            llm_reasoning,
            program_id=source_program.id if source_program is not None else None,
            source_program_id=source_program_id if source_program is not None else None,
            association=(
                association
                if source_program is not None or association.startswith("unattached_")
                else f"unattached_{association}"
            ),
        )
        if source_program is not None:
            if not isinstance(source_program.llm_reasoning, dict):
                source_program.llm_reasoning = {}
            self._merge_llm_reasoning(source_program.llm_reasoning, bound)
            source_program.llm_reasoning_content = reasoning_content(
                source_program.llm_reasoning.get("calls", [])
            )
            return

        unattached = getattr(self.database, "unattached_llm_reasoning", None)
        if not isinstance(unattached, dict):
            unattached = {}
            self.database.unattached_llm_reasoning = unattached
        self._merge_llm_reasoning(unattached, bound)

    # ------------------------------------------------------------------
    # Main discovery loop
    # ------------------------------------------------------------------

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        """
        Run the discovery process.

        When ``config.max_parallel_iterations == 1`` (default), iterations
        run sequentially — same behaviour as before.

        When ``> 1``, up to *N* iterations run concurrently as asyncio
        tasks, bounded by a semaphore.  Generation and evaluation naturally
        overlap across iterations: while iteration *i* evaluates, iteration
        *i+1* can generate, and iteration *i+2* can sample.

        Args:
            start_iteration: The iteration to start from.
            max_iterations: The number of iterations to run.
            checkpoint_callback: Optional callback for checkpointing.
            post_process_result: If True, add results to the database and
                return the best Program.  If False, return the raw
                ``SerializableResult`` from the last iteration.
            retry_times: Number of retry attempts per iteration.

        Returns:
            Best ``Program`` found (post_process_result=True) or raw
            ``SerializableResult`` (post_process_result=False).
        """
        self._preflight_evoduet()
        max_parallel = self.config.max_parallel_iterations

        if max_parallel > 1:
            return await self._run_discovery_parallel(
                start_iteration,
                max_iterations,
                checkpoint_callback,
                post_process_result,
                retry_times,
                max_parallel,
            )

        return await self._run_discovery_sequential(
            start_iteration,
            max_iterations,
            checkpoint_callback,
            post_process_result,
            retry_times,
        )

    # ------------------------------------------------------------------
    # Sequential loop (original behaviour, max_parallel_iterations=1)
    # ------------------------------------------------------------------

    async def _run_discovery_sequential(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        total_iterations = start_iteration + max_iterations

        result = None
        for iteration in range(start_iteration, total_iterations):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested, stopping discovery loop early")
                break

            try:
                result = await self._run_iteration(iteration, retry_times=retry_times)
                if result.error:
                    if post_process_result:
                        self._process_iteration_result(result, iteration, None, verbose=False)
                    logger.warning(f"Iteration {iteration} failed: {result.error}")
                    continue

                if post_process_result:
                    self._process_iteration_result(result, iteration, None)

            except Exception as e:
                logger.exception(f"Error in iteration {iteration}: {e}")
            finally:
                self.last_processed_iteration = iteration
                if post_process_result:
                    self._checkpoint_after_iteration(iteration, checkpoint_callback)
                    if not (
                        checkpoint_callback is not None
                        and iteration > 0
                        and iteration % self.config.checkpoint_interval == 0
                    ):
                        self._write_solution_confidence_trace(iteration)

        if not post_process_result:
            return result

        return self._finalize_discovery()

    # ------------------------------------------------------------------
    # Parallel loop (max_parallel_iterations > 1)
    # ------------------------------------------------------------------

    async def _run_discovery_parallel(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
        max_parallel: int = 4,
    ) -> Optional[Union[Program, SerializableResult]]:
        total_iterations = start_iteration + max_iterations
        sem = asyncio.Semaphore(max_parallel)
        pending: set = set()
        last_result: Optional[SerializableResult] = None
        last_result_iteration = -1

        logger.debug(
            f"Parallel discovery: up to {max_parallel} iterations in flight "
            f"({start_iteration}..{total_iterations - 1})"
        )

        async def _bounded_iteration(iteration: int) -> Tuple[int, Optional[SerializableResult]]:
            """Run one iteration under the semaphore, then process its result.

            Result processing (database.add) happens here rather than being
            collected later so that subsequent iterations see the latest DB
            state as soon as the ``await`` inside ``_run_iteration`` yields.
            """
            async with sem:
                if self.shutdown_event.is_set():
                    return iteration, None
                try:
                    result = await self._run_iteration(iteration, retry_times=retry_times)
                except asyncio.CancelledError:
                    if self.evoduet is not None:
                        self.evoduet.record_usage(
                            iteration=iteration,
                            score=None,
                            feedback="iteration cancelled",
                        )
                    raise
                except Exception as e:
                    logger.exception(f"Error in parallel iteration {iteration}: {e}")
                    result = SerializableResult(error=str(e), iteration=iteration)

            # Process outside the semaphore — database.add() is sync and
            # completes atomically between await-points, so no lock needed.
            if result and post_process_result:
                # Checkpoints are emitted only after every task through their
                # boundary has drained, so no future/uncredited WK record leaks in.
                self._process_iteration_result(result, iteration, None)
                self._write_solution_confidence_trace(iteration)
            elif result and result.error:
                logger.warning(f"Iteration {iteration} failed: {result.error}")

            return iteration, result

        def collect(done) -> None:
            nonlocal last_result, last_result_iteration
            for task in done:
                try:
                    completed_iteration, result = task.result()
                    if result is not None and completed_iteration >= last_result_iteration:
                        last_result = result
                        last_result_iteration = completed_iteration
                        self.last_processed_iteration = completed_iteration
                except Exception as e:
                    logger.warning(f"A task in parallel discovery failed: {e}")

        try:
            for iteration in range(start_iteration, total_iterations):
                if self.shutdown_event.is_set():
                    break

                task = asyncio.create_task(_bounded_iteration(iteration), name=f"iter_{iteration}")
                pending.add(task)

                # When the pipeline is full, wait for at least one to finish
                # before scheduling more — this provides backpressure.
                if len(pending) >= max_parallel:
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    collect(done)

                is_checkpoint = (
                    post_process_result
                    and checkpoint_callback is not None
                    and iteration > 0
                    and iteration % self.config.checkpoint_interval == 0
                )
                if is_checkpoint:
                    if pending:
                        done, _ = await asyncio.wait(pending)
                        collect(done)
                        pending = set()
                    logger.debug(f"Checkpoint interval reached at iteration {iteration}")
                    self.database.log_status()
                    checkpoint_callback(iteration)

            # Drain remaining tasks.
            if pending:
                done, _ = await asyncio.wait(pending)
                collect(done)
                pending = set()
        finally:
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        if not post_process_result:
            return last_result

        return self._finalize_discovery()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _finalize_discovery(self) -> Optional[Program]:
        if self.shutdown_event.is_set():
            logger.info(
                f"✅ Discovery process completed "
                f"(search strategy = {self.database.name}) - Shutdown requested"
            )
        else:
            logger.info(
                f"✅ Discovery process completed "
                f"(search strategy = {self.database.name}) - Maximum iterations reached"
            )
        return self.database.get_best_program()

    # ------------------------------------------------------------------
    # Single-iteration primitives (shared by all controllers)
    # ------------------------------------------------------------------

    def _solution_confidence_enabled(self) -> bool:
        settings = getattr(getattr(self, "config", None), "solution_confidence", None)
        return bool(getattr(settings, "enabled", False))

    def _write_solution_confidence_trace(self, iteration: int) -> None:
        if self._solution_confidence_enabled() and getattr(self, "output_dir", None):
            # Concurrent iterations may complete out of order. The trace can
            # include pending attempts; resumable checkpoints still drain them.
            completed = [
                record["iteration"]
                for record in self.database.solution_confidence_records
                if record.get("phase") == "completed"
            ]
            self.database.write_evolution_trace(self.output_dir, max([iteration, *completed]))

    def _start_solution_confidence(self, parent, iteration: int, attempt: int):
        if not self._solution_confidence_enabled():
            return None
        database_config = getattr(getattr(self.config, "search", None), "database", None)
        record = solution_confidence.new_record(
            self.config.solution_confidence,
            parent=parent,
            iteration=iteration,
            attempt=attempt,
            run_id=getattr(self, "output_dir", None),
            task=getattr(self, "evaluation_file", None),
            seed=getattr(database_config, "random_seed", None),
        )
        self.database.solution_confidence_records.append(record)
        return record

    async def _assess_solution_confidence(self, record, result, candidate, program_id):
        if record is None:
            return
        runtime_context = getattr(result, "generation_context", None)
        if isinstance(runtime_context, dict) and runtime_context:
            record["generation_context"] = copy.deepcopy(finite_json(json_safe(runtime_context)))
        seen_before = any(
            prior.get("candidate_solution") == candidate
            and prior.get("candidate_score") is not None
            for prior in self.database.solution_confidence_records
        ) or any(program.solution == candidate for program in self.database.programs.values())
        await solution_confidence.assess(
            record,
            self.config.solution_confidence,
            generation_model=getattr(result, "generation_model", None),
            candidate=candidate,
            program_id=program_id,
            seen_before=seen_before,
        )

    def _finish_solution_confidence(self, record, evaluation_result):
        solution_confidence.finish(record, evaluation_result)
        if record is not None:
            logger.info(
                "Iteration %s attempt %s: confidence=%s confidence_verbalized=%s "
                "y_t=%s brier_score=%s brier_score_verbalized=%s (%s/%s/%s)",
                record["iteration"],
                record["attempt"],
                record["confidence"],
                record.get("confidence_verbalized"),
                record["y_t"],
                record["brier_score"],
                record.get("brier_score_verbalized"),
                record["assessment_status"],
                record.get("assessment_status_verbalized"),
                record["outcome_status"],
            )

    async def _run_from_scratch_iteration(
        self, iteration: int, attempt_number: int = 1
    ) -> SerializableResult:
        """Generate a first solution from scratch when the database is empty."""
        tavily_searches: List[Dict[str, Any]] = []
        web_search_results: Dict[str, Any] = {}
        llm_reasoning_calls: List[Dict[str, Any]] = []
        llm_reasoning: Dict[str, Any] = {"calls": llm_reasoning_calls}
        confidence_record = self._start_solution_confidence(None, iteration, attempt_number)
        try:
            iteration_start = time.time()

            prompt = self.context_builder.build_prompt(current_program=None, context={})

            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)

            solution_confidence.freeze_prompt(confidence_record, prompt)
            if getattr(self.config, "num_generations", 1) > 1:
                runner = self._get_generation_batch_runner()
                batch = await runner.run(
                    self,
                    prompt=prompt,
                    parent=None,
                    iteration=iteration,
                    attempt=attempt_number,
                    retry_times=attempt_number,
                    confidence_template=confidence_record,
                    llm_reasoning=llm_reasoning,
                    web_search_results=web_search_results,
                )
                return runner.result(
                    self,
                    [batch],
                    parent=None,
                    iteration=iteration,
                    iteration_start=iteration_start,
                    llm_reasoning=llm_reasoning,
                    web_search_results=web_search_results,
                )
            llm_generation_time = 0.0
            llm_start = time.time()
            result = await self._call_llm(
                prompt["system"],
                prompt["user"],
                llm_context={
                    "iteration": iteration,
                    "attempt": attempt_number,
                    "phase": "generation",
                    "source_program_id": None,
                },
                web_search_result_sink=tavily_searches,
                reasoning_result_sink=llm_reasoning_calls,
            )
            if tavily_searches:
                web_search_results["tavily_searches"] = tavily_searches
            self._merge_web_search_results(
                web_search_results, getattr(result, "web_search_results", {})
            )
            self._merge_llm_reasoning(llm_reasoning, getattr(result, "llm_reasoning", {}))
            llm_generation_time = time.time() - llm_start
            llm_response = result.text
            if not llm_response:
                solution_confidence.abort(
                    confidence_record, "generation_error", "Empty LLM response"
                )
                return SerializableResult(
                    error="Empty LLM response",
                    iteration=iteration,
                    web_search_results=web_search_results,
                    llm_reasoning=llm_reasoning,
                    llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                )

            child_solution = parse_full_rewrite(llm_response, self.config.language)
            if not child_solution:
                solution_confidence.abort(
                    confidence_record, "parse_error", "No valid solution in response"
                )
                return SerializableResult(
                    error="No valid solution in response",
                    iteration=iteration,
                    prompt=prompt,
                    llm_response=llm_response,
                    web_search_results=web_search_results,
                    llm_reasoning=llm_reasoning,
                    llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                )

            child_id = str(uuid.uuid4())
            await self._assess_solution_confidence(
                confidence_record, result, child_solution, child_id
            )
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(child_solution, child_id)
            eval_time = time.time() - eval_start
            self._finish_solution_confidence(confidence_record, eval_result)
            self._merge_llm_reasoning(llm_reasoning, getattr(eval_result, "llm_reasoning", {}))

            child = Program(
                id=child_id,
                solution=child_solution,
                language=self.config.language,
                parent_id=None,
                metrics=eval_result.metrics,
                iteration_found=iteration,
                metadata={"changes": "Generated from scratch"},
                artifacts=eval_result.artifacts or {},
                solution_confidence=copy.deepcopy(confidence_record or {}),
                llm_response=llm_response,
                llm_reasoning=self._bind_llm_reasoning(
                    llm_reasoning,
                    program_id=child_id,
                    source_program_id=None,
                    association="generated_program",
                    successful_attempt=attempt_number,
                ),
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                web_search_results=self._bind_web_search_results(
                    web_search_results,
                    program_id=child_id,
                    source_program_id=None,
                    association="generated_program",
                    successful_attempt=attempt_number,
                ),
            )

            return SerializableResult(
                child_program_dict=child.to_dict(),
                parent_id=None,
                other_context_ids=[],
                iteration_time=time.time() - iteration_start,
                llm_generation_time=llm_generation_time,
                eval_time=eval_time,
                prompt=prompt,
                llm_response=llm_response,
                web_search_results=child.web_search_results,
                llm_reasoning=child.llm_reasoning,
                llm_reasoning_content=child.llm_reasoning_content,
                iteration=iteration,
            )
        except asyncio.CancelledError:
            solution_confidence.abort(confidence_record, "cancelled", "Iteration cancelled")
            if tavily_searches:
                web_search_results["tavily_searches"] = tavily_searches
            self._store_unowned_web_search_results(
                web_search_results,
                source_program_id=None,
                association="cancelled_generation_attempt",
            )
            self._store_unowned_llm_reasoning(
                llm_reasoning,
                source_program_id=None,
                association="cancelled_generation_attempt",
            )
            raise
        except Exception as e:
            solution_confidence.abort(confidence_record, "iteration_error", str(e))
            logger.exception(f"From-scratch generation failed: {e}")
            if tavily_searches:
                web_search_results["tavily_searches"] = tavily_searches
            return SerializableResult(
                error=str(e),
                iteration=iteration,
                web_search_results=web_search_results,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )
        finally:
            if getattr(self.config, "num_generations", 1) > 1:
                self.llms.release_model(iteration)

    async def _run_iteration(
        self,
        iteration: int,
        retry_times: int = 1,
    ) -> SerializableResult:
        """Run a single generate-evaluate iteration."""
        evoduet_sent = False
        attempts_used = 1
        confidence_record = None
        parent: Optional[Program] = None
        tavily_searches: List[Dict[str, Any]] = []
        web_search_results: Dict[str, Any] = {}
        llm_reasoning_calls: List[Dict[str, Any]] = []
        llm_reasoning: Dict[str, Any] = {"calls": llm_reasoning_calls}
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(iteration)

            raw_parent, raw_context_programs = self.database.sample(
                num_context_programs=self.num_context_programs
            )

            # Normalize sample() result — databases may return plain or dict-wrapped
            if isinstance(raw_parent, dict):
                if len(raw_parent) != 1:
                    raise ValueError(
                        f"sample() must return exactly one parent, got {len(raw_parent)}"
                    )
                parent_info_key = list(raw_parent.keys())[0]
                parent = list(raw_parent.values())[0]
            else:
                parent_info_key = ""
                parent = raw_parent

            # Other context programs that are relevant
            if isinstance(raw_context_programs, dict):
                context_programs_dict = raw_context_programs
            else:
                context_programs_dict = {"": raw_context_programs}

            parent_info = (parent_info_key, parent.id)
            context_info = [
                (key, p.id) for key, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            logger.debug(
                f"Iteration {iteration}: parent {parent.id} ({parent_info_key}), "
                f"other_context_programs keys: {list(context_programs_dict.keys())}"
            )

            iteration_start = time.time()

            failed_attempts = []
            child_solution, child_id, child_metrics, llm_response, changes_summary = (
                None,
                None,
                None,
                None,
                None,
            )

            image_path = None  # set by image mode or evaluator
            eval_time = 0.0

            # EvoDuet: runs once per iteration (reused across retries)
            # right after the first prompt build, so it reads the scaffold's own
            # rendered evolutionary history verbatim. "" when disabled, the gate
            # declines, or it errors out.
            evoduet = ""
            evoduet_ready = False
            evoduet_sent = False
            generation_batches = []

            # Build prompt with parent and context programs
            for retry in range(retry_times):
                attempts_used = retry + 1
                confidence_record = self._start_solution_confidence(
                    parent, iteration, attempts_used
                )
                try:
                    # The builder records the rendered history and parent here;
                    # solution confidence reuses the same sink when enabled.
                    prompt_capture: Dict[str, Any] = (
                        confidence_record if confidence_record is not None else {}
                    )
                    prompt = self._build_prompt(
                        current_program=raw_parent,
                        context_programs=context_programs_dict,
                        failed_attempts=failed_attempts,
                        confidence_context=prompt_capture,
                    )
                    if not evoduet_ready:
                        evoduet = await self._maybe_run_evoduet(
                            parent,
                            str(prompt_capture.get("evolutionary_history") or ""),
                            iteration,
                        )
                        evoduet_ready = True

                    if failed_attempts:
                        logger.debug(
                            f"Retry {retry + 1}/{retry_times}: rebuilding prompt with "
                            f"{len(failed_attempts)} failed attempt(s)"
                        )

                    # Feedback may replace the user prompt, so insert evidence only
                    # after feedback has produced the final mutation prompt.
                    if self.feedback_reader:
                        self.feedback_reader.set_current_prompt(prompt["system"])
                        feedback = self.feedback_reader.read()
                        if feedback:
                            prompt = self.feedback_reader.apply_feedback(prompt)
                            self.feedback_reader.log_usage(
                                iteration, feedback, self.feedback_reader.mode
                            )
                    if evoduet:
                        prompt["user"] = _insert_evoduet(prompt["user"], evoduet)
                    solution_confidence.freeze_prompt(
                        confidence_record, prompt, web_document=evoduet
                    )
                except Exception:
                    if not evoduet_sent:
                        self._discard_evoduet_usage(iteration)
                    raise

                if getattr(self.config, "num_generations", 1) > 1:
                    evoduet_sent = bool(evoduet) or evoduet_sent
                    runner = self._get_generation_batch_runner()
                    batch = await runner.run(
                        self,
                        prompt=prompt,
                        parent=parent,
                        iteration=iteration,
                        attempt=attempts_used,
                        retry_times=retry_times,
                        confidence_template=confidence_record,
                        llm_reasoning=llm_reasoning,
                        web_search_results=web_search_results,
                    )
                    generation_batches.append(batch)
                    if batch.winner is not None or attempts_used == retry_times:
                        return runner.result(
                            self,
                            generation_batches,
                            parent=parent,
                            iteration=iteration,
                            iteration_start=iteration_start,
                            llm_reasoning=llm_reasoning,
                            web_search_results=web_search_results,
                            context_program_ids=context_program_ids,
                            parent_info=parent_info,
                            context_info=context_info,
                        )
                    failed_attempts.extend(batch.failed_attempts(parent))
                    continue

                try:
                    llm_generation_time = 0.0
                    llm_start = time.time()
                    evoduet_sent = bool(evoduet) or evoduet_sent
                    if self.config.language == "image":
                        child_id = str(uuid.uuid4())
                        user_content = build_image_content(
                            prompt["user"], parent, context_programs_dict
                        )
                        result = await self._call_llm(
                            prompt["system"],
                            user_content,
                            image_output=True,
                            output_dir=self._get_image_output_dir(),
                            program_id=child_id,
                            llm_context={
                                "iteration": iteration,
                                "attempt": retry + 1,
                                "phase": "generation",
                                "source_program_id": parent.id,
                            },
                            web_search_result_sink=tavily_searches,
                            reasoning_result_sink=llm_reasoning_calls,
                        )
                        if tavily_searches:
                            web_search_results["tavily_searches"] = tavily_searches
                        self._merge_web_search_results(
                            web_search_results, getattr(result, "web_search_results", {})
                        )
                        self._merge_llm_reasoning(
                            llm_reasoning, getattr(result, "llm_reasoning", {})
                        )
                        llm_response = result.text or ""
                        image_path = result.image_path
                        if image_path:
                            child_solution = result.text or "(image generated)"
                            changes_summary = "Image generation"
                            parse_error = None
                        else:
                            child_solution = None
                            changes_summary = None
                            parse_error = "VLM did not generate an image"
                    else:
                        result = await self._call_llm(
                            prompt["system"],
                            prompt["user"],
                            llm_context={
                                "iteration": iteration,
                                "attempt": retry + 1,
                                "phase": "generation",
                                "source_program_id": parent.id,
                            },
                            web_search_result_sink=tavily_searches,
                            reasoning_result_sink=llm_reasoning_calls,
                        )
                        if tavily_searches:
                            web_search_results["tavily_searches"] = tavily_searches
                        self._merge_web_search_results(
                            web_search_results, getattr(result, "web_search_results", {})
                        )
                        self._merge_llm_reasoning(
                            llm_reasoning, getattr(result, "llm_reasoning", {})
                        )
                        llm_response = result.text
                    llm_generation_time = time.time() - llm_start
                except Exception as e:
                    solution_confidence.abort(confidence_record, "generation_error", str(e))
                    logger.error(f"LLM generation failed: {e}")
                    if tavily_searches:
                        web_search_results["tavily_searches"] = tavily_searches
                    return SerializableResult(
                        error=f"LLM generation failed: {str(e)}",
                        iteration=iteration,
                        attempts_used=retry + 1,
                        parent_id=parent.id,
                        web_search_results=web_search_results,
                        llm_reasoning=llm_reasoning,
                        llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                    )

                if self.config.language != "image":
                    # Text/code mode: parse LLM response
                    if llm_response is None:
                        solution_confidence.abort(
                            confidence_record, "generation_error", "LLM returned None response"
                        )
                        return SerializableResult(
                            error="LLM returned None response",
                            iteration=iteration,
                            attempts_used=retry + 1,
                            parent_id=parent.id,
                            web_search_results=web_search_results,
                            llm_reasoning=llm_reasoning,
                            llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                        )

                    child_solution, changes_summary, parse_error = self._parse_llm_response(
                        llm_response, parent.solution, iteration, retry + 1, retry_times
                    )

                    if child_solution and len(child_solution) > self.config.max_solution_length:
                        logger.warning(
                            "Generated solution exceeds maximum length (iteration=%s, attempt %s/%s): %s > %s",
                            iteration,
                            retry + 1,
                            retry_times,
                            len(child_solution),
                            self.config.max_solution_length,
                        )
                        parse_error = f"Generated solution exceeds maximum length ({len(child_solution)} > {self.config.max_solution_length})"
                        child_solution = None

                if parse_error:
                    solution_confidence.abort(confidence_record, "parse_error", parse_error)
                    failed_attempts.append(
                        {
                            "solution": child_solution or "",
                            "llm_response": llm_response,
                            "metrics": {},
                            "metadata": {
                                "error": parse_error,
                                "attempt_number": retry + 1,
                            },
                        }
                    )
                    if retry < retry_times - 1:
                        continue
                    logger.error(
                        "All %s retry attempts failed due to parse/validation error: %s",
                        retry_times,
                        parse_error,
                    )
                    return SerializableResult(
                        error=f"{parse_error} (after {retry_times} attempts)",
                        iteration=iteration,
                        prompt=prompt,
                        llm_response=llm_response,
                        attempts_used=retry_times,
                        parent_id=parent.id,
                        web_search_results=web_search_results,
                        llm_reasoning=llm_reasoning,
                        llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                    )

                if self.config.language != "image":
                    child_id = str(uuid.uuid4())

                await self._assess_solution_confidence(
                    confidence_record, result, child_solution, child_id
                )
                eval_input = image_path if self.config.language == "image" else child_solution
                eval_start = time.time()
                child_eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
                eval_time = time.time() - eval_start
                self._finish_solution_confidence(confidence_record, child_eval_result)
                self._merge_llm_reasoning(
                    llm_reasoning, getattr(child_eval_result, "llm_reasoning", {})
                )
                child_metrics = child_eval_result.metrics
                # Extract image_path from evaluator metrics (non-image mode fallback)
                if not image_path:
                    image_path = (
                        child_metrics.pop("image_path", None)
                        if isinstance(child_metrics.get("image_path"), str)
                        else None
                    )

                child_artifacts = child_eval_result.artifacts or {}
                error_msg = self._evaluation_failure_reason(child_metrics, child_artifacts)
                if error_msg is not None:

                    logger.warning(
                        "Evaluation failed (attempt %s/%s): %s",
                        retry + 1,
                        retry_times,
                        error_msg,
                    )
                    logger.debug(
                        "Failed solution (attempt %s/%s):\n%s",
                        retry + 1,
                        retry_times,
                        child_solution,
                    )

                    failed_attempts.append(
                        {
                            "solution": child_solution,
                            "metrics": child_metrics,
                            "metadata": {
                                "changes": changes_summary,
                                "parent_metrics": parent.metrics,
                                "error": error_msg,
                                "attempt_number": retry + 1,
                            },
                        }
                    )

                    if retry < retry_times - 1:
                        continue
                    logger.error(
                        "All %s retry attempts failed. Final error: %s", retry_times, error_msg
                    )
                    iteration_time = time.time() - iteration_start
                    failed_extra = {"failed_attempts": failed_attempts}
                    if self.evoduet is not None:
                        record_id = self.evoduet.staged_search_record_id(iteration)
                        if record_id:
                            failed_extra["evoduet_search_record_id"] = record_id
                    if image_path:
                        failed_extra["image_path"] = image_path
                    failed_child_program = self._create_child_program(
                        child_id=child_id,
                        child_solution=child_solution,
                        parent=parent,
                        context_program_ids=context_program_ids,
                        parent_info=parent_info,
                        context_info=context_info,
                        child_metrics=child_metrics or {},
                        iteration=iteration,
                        changes_summary=changes_summary,
                        extra_metadata=failed_extra,
                        solution_confidence=confidence_record,
                        artifacts=child_eval_result.artifacts,
                        llm_response=llm_response,
                        llm_reasoning=self._bind_llm_reasoning(
                            llm_reasoning,
                            program_id=child_id,
                            source_program_id=parent.id,
                            association="failed_generation_attempt",
                        ),
                        llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                        web_search_results=self._bind_web_search_results(
                            web_search_results,
                            program_id=child_id,
                            source_program_id=parent.id,
                            association="failed_generation_attempt",
                        ),
                    )
                    return SerializableResult(
                        error=f"Evaluator failed after {retry_times} attempts: {error_msg}",
                        iteration=iteration,
                        child_program_dict=failed_child_program.to_dict(),
                        parent_id=parent.id,
                        other_context_ids=context_program_ids,
                        iteration_time=iteration_time,
                        llm_generation_time=llm_generation_time,
                        eval_time=eval_time,
                        prompt=prompt,
                        llm_response=llm_response,
                        attempts_used=retry_times,
                        web_search_results=failed_child_program.web_search_results,
                        llm_reasoning=failed_child_program.llm_reasoning,
                        llm_reasoning_content=failed_child_program.llm_reasoning_content,
                    )
                break

            extra_meta = {}
            if image_path:
                extra_meta["image_path"] = image_path
            if self.evoduet is not None:
                record_id = self.evoduet.staged_search_record_id(iteration)
                if record_id:
                    extra_meta["evoduet_search_record_id"] = record_id
            child_program = self._create_child_program(
                child_id=child_id,
                child_solution=child_solution,
                parent=parent,
                context_program_ids=context_program_ids,
                parent_info=parent_info,
                context_info=context_info,
                child_metrics=child_metrics,
                iteration=iteration,
                changes_summary=changes_summary,
                extra_metadata=extra_meta if extra_meta else None,
                solution_confidence=confidence_record,
                artifacts=child_eval_result.artifacts,
                llm_response=llm_response,
                llm_reasoning=self._bind_llm_reasoning(
                    llm_reasoning,
                    program_id=child_id,
                    source_program_id=parent.id,
                    association="generated_program",
                    successful_attempt=retry + 1,
                ),
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                web_search_results=self._bind_web_search_results(
                    web_search_results,
                    program_id=child_id,
                    source_program_id=parent.id,
                    association="generated_program",
                    successful_attempt=retry + 1,
                ),
            )
            iteration_time = time.time() - iteration_start

            return SerializableResult(
                child_program_dict=child_program.to_dict(),
                parent_id=parent.id,
                other_context_ids=context_program_ids,
                iteration_time=iteration_time,
                llm_generation_time=llm_generation_time,
                eval_time=eval_time,
                prompt=prompt,
                llm_response=llm_response,
                iteration=iteration,
                attempts_used=retry + 1,
                web_search_results=child_program.web_search_results,
                llm_reasoning=child_program.llm_reasoning,
                llm_reasoning_content=child_program.llm_reasoning_content,
            )
        except asyncio.CancelledError:
            solution_confidence.abort(confidence_record, "cancelled", "Iteration cancelled")
            if tavily_searches:
                web_search_results["tavily_searches"] = tavily_searches
            self._store_unowned_web_search_results(
                web_search_results,
                source_program_id=parent.id if parent is not None else None,
                association="cancelled_generation_attempt",
            )
            self._store_unowned_llm_reasoning(
                llm_reasoning,
                source_program_id=parent.id if parent is not None else None,
                association="cancelled_generation_attempt",
            )
            if evoduet_sent and self.evoduet is not None:
                self.evoduet.record_usage(
                    iteration=iteration,
                    score=None,
                    feedback="iteration cancelled",
                )
            else:
                self._discard_evoduet_usage(iteration)
            raise
        except Exception as e:
            solution_confidence.abort(confidence_record, "iteration_error", str(e))
            logger.exception(f"Error in iteration {iteration}")
            if tavily_searches:
                web_search_results["tavily_searches"] = tavily_searches
            return SerializableResult(
                error=str(e),
                iteration=iteration,
                attempts_used=attempts_used,
                parent_id=parent.id if parent is not None else None,
                web_search_results=web_search_results,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )
        finally:
            release = getattr(getattr(self, "llms", None), "release_model", None)
            if callable(release):
                release(iteration)

    # ------------------------------------------------------------------
    # Prompt / parsing / program creation helpers
    # ------------------------------------------------------------------

    async def _maybe_run_evoduet(
        self,
        parent: Program,
        history: Any,
        iteration: int,
    ) -> str:
        """Run the EvoDuet for this iteration; return a prompt block.

        ``history`` is the scaffold's own evolutionary history exactly as its
        context builder rendered it for the mutation prompt; it is handed to the
        layer unchanged.

        Returns "" when the layer is disabled, the gate declines, or it errors
        out (best-effort: never breaks the discovery loop).
        """
        if self.evoduet is None:
            return ""
        # Reserve the actual solution model before retrieval. All later solution
        # attempts (including agentic tool steps) reuse it for this iteration.
        pool = getattr(self, "llms", None)
        reserve = getattr(pool, "reserve_model", None)
        solution_model = reserve(iteration) if callable(reserve) else None
        self._preflight_evoduet()
        try:
            from skydiscover.evoduet import format_documents

            documents = await self.evoduet.run(
                parent=parent,
                history=history,
                population=[
                    self._evoduet_scored_program(program)
                    for program in self._evoduet_population()
                ],
                iteration=iteration,
                task_context=self._evoduet_task_context(),
                target_score=self._evoduet_program_score(parent),
                **({"solution_model": solution_model} if solution_model is not None else {}),
            )
            return format_documents(
                documents,
                getattr(self.config.evoduet, "max_document_chars", 10_000),
            )
        except Exception:
            self._discard_evoduet_usage(iteration)
            logger.exception("EvoDuet failed; continuing without it")
            return ""

    def _discard_evoduet_usage(self, iteration: int) -> None:
        discard = getattr(self.evoduet, "discard_usage", None)
        if callable(discard):
            discard(iteration)

    def _preflight_evoduet(self) -> None:
        if self.evoduet is None or getattr(self, "_evoduet_preflighted", False):
            return
        preflight = getattr(self.evoduet, "preflight", None)
        if callable(preflight):
            preflight()
        self._evoduet_preflighted = True

    def has_evoduet_checkpoint_state(self) -> bool:
        """Whether this controller owns EvoDuet state worth checkpointing."""
        return self.evoduet is not None

    def evoduet_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """Return this controller's EvoDuet checkpoint payload."""
        state_dict = getattr(self.evoduet, "state_dict", None)
        return state_dict() if callable(state_dict) else None

    def load_evoduet_checkpoint_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore EvoDuet from a complete current payload."""
        load_state_dict = getattr(self.evoduet, "load_state_dict", None)
        if not callable(load_state_dict):
            return
        store = state.get("store") if isinstance(state, dict) else None
        if not isinstance(store, dict) or not isinstance(store.get("records"), list):
            raise ValueError("checkpoint does not contain valid EvoDuet records")
        load_state_dict(state)

    def _evoduet_task_context(self) -> str:
        builder = getattr(self, "context_builder", None)
        resolve = getattr(builder, "_get_system_message", None)
        if callable(resolve):
            return resolve() or ""
        config = getattr(self, "config", None)
        context_config = getattr(config, "context_builder", None)
        return getattr(context_config, "system_message", "") or ""

    def _evoduet_program_score(self, program: Program) -> Optional[float]:
        """Use the database's fitness proxy when it exposes one."""
        proxy = getattr(self.database, "get_program_proxy_score", None)
        if callable(proxy) and not isinstance(program, dict):
            try:
                score = float(proxy(program))
                if math.isfinite(score):
                    return score
            except (TypeError, ValueError, OverflowError):
                logger.debug("Could not compute database proxy score", exc_info=True)
        metrics = (
            program.get("metrics", {})
            if isinstance(program, dict)
            else getattr(program, "metrics", {})
        ) or {}
        if not metrics:
            return None
        try:
            score = float(get_score(metrics))
        except (TypeError, ValueError, OverflowError):
            return None
        return score if math.isfinite(score) else None

    def _scaffold_evolutionary_history(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context_programs: Union[List[Program], Dict[str, List[Program]]],
    ) -> str:
        """Render the sampled history exactly as the scaffold's mutation prompt shows it.

        Builds the prompt through the configured context builder and returns the
        history sections it recorded (previous attempts and context programs); no
        EvoDuet re-rendering is applied. Empty when no builder is set up.
        """
        if getattr(self, "context_builder", None) is None:
            return ""
        capture: Dict[str, Any] = {}
        self._build_prompt(
            current_program=current_program,
            context_programs=context_programs,
            failed_attempts=[],
            confidence_context=capture,
        )
        return str(capture.get("evolutionary_history") or "")

    def _evoduet_population(self) -> List[Program]:
        """Return the live population when a database exposes one."""
        database = getattr(self, "database", None)
        active = getattr(database, "active_programs", None)
        if callable(active):
            active = active()
        if active is None:
            beam = getattr(database, "get_beam_programs", None)
            if callable(beam):
                active = beam()
        if active is None:
            elite_ids = getattr(database, "elite_pool", None)
            registry = getattr(database, "programs", None)
            if elite_ids is not None and isinstance(registry, Mapping):
                active = [
                    registry[program_id] for program_id in elite_ids if program_id in registry
                ]
        programs = active if active is not None else getattr(database, "programs", None)
        if isinstance(programs, Mapping):
            return list(programs.values())
        return list(programs or [])

    def _evoduet_scored_program(self, program: Program) -> Program:
        """Copy a program and expose the database's higher-is-better fitness proxy."""
        score = self._evoduet_program_score(program)
        metrics = (
            program.get("metrics", {})
            if isinstance(program, dict)
            else getattr(program, "metrics", {})
        ) or {}
        metrics = dict(metrics)
        if score is not None:
            metrics["evoduet_score"] = score

        if isinstance(program, dict):
            return {**program, "metrics": metrics}
        view = copy.copy(program)
        view.metrics = metrics
        return view

    def _build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context_programs: Union[List[Program], Dict[str, List[Program]]],
        failed_attempts: list,
        confidence_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Build the prompt for LLM generation."""
        parent = (
            list(current_program.values())[0]
            if isinstance(current_program, dict)
            else current_program
        )
        db_stats = self._prompt_context.get("db_stats") or self.database.get_statistics()

        # Build context with parent program and any other relevant information
        context = {
            "program_metrics": parent.metrics,
            "other_context_programs": context_programs,
            "previous_programs": db_stats.get("previous_programs", []),
            "db_stats": db_stats,
        }
        for k, v in self._prompt_context.items():
            if k not in context:
                context[k] = v

        if failed_attempts:
            context["errors"] = failed_attempts

        capture = (
            {"confidence_context": confidence_context} if confidence_context is not None else {}
        )
        return self.context_builder.build_prompt(
            current_program=current_program, context=context, **capture
        )

    def _parse_llm_response(
        self,
        llm_response: str,
        parent_solution: str,
        iteration: int,
        attempt: int,
        retry_times: int,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse LLM response to extract child solution.

        Returns:
            Tuple of (child_solution, changes_summary, parse_error)
        """
        if self.config.diff_based_generation:
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                preview = llm_response[:2000] + (
                    "\n... (truncated) ..." if len(llm_response) > 2000 else ""
                )
                logger.warning(
                    "No valid diffs found in LLM response (iteration=%s, attempt %s/%s). "
                    "Expected SEARCH/REPLACE blocks. Preview:\n%s",
                    iteration,
                    attempt,
                    retry_times,
                    preview,
                )
                return None, None, "No valid diffs found in response"

            child_solution = apply_diff(parent_solution, llm_response)
            changes_summary = format_diff_summary(diff_blocks)

            if child_solution == parent_solution:
                logger.warning(
                    "Diff blocks found but none matched parent solution (iteration=%s, attempt %s/%s).",
                    iteration,
                    attempt,
                    retry_times,
                )
                return (
                    None,
                    None,
                    "Diff SEARCH blocks did not match parent solution - no changes applied",
                )

            return child_solution, changes_summary, None
        else:
            new_solution = parse_full_rewrite(llm_response, self.config.language)
            if not new_solution:
                logger.warning(
                    "No valid solution found in LLM response (iteration=%s, attempt %s/%s).",
                    iteration,
                    attempt,
                    retry_times,
                )
                return None, None, "No valid solution found in response"
            return new_solution, "Full rewrite", None

    def _create_child_program(
        self,
        child_id: str,
        child_solution: str,
        parent: Program,
        context_program_ids: list,
        parent_info: tuple,
        context_info: list,
        child_metrics: Dict[str, Any],
        iteration: int,
        changes_summary: Optional[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        llm_response: Optional[str] = None,
        llm_reasoning: Optional[Dict[str, Any]] = None,
        llm_reasoning_content: Optional[str] = None,
        web_search_results: Optional[Dict[str, Any]] = None,
        solution_confidence: Optional[Dict[str, Any]] = None,
    ) -> Program:
        """Create a child program with the given attributes."""
        metadata = {
            "changes": changes_summary,
            "parent_metrics": parent.metrics,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata=metadata,
            artifacts=artifacts or {},
            llm_response=llm_response,
            llm_reasoning=llm_reasoning or {},
            llm_reasoning_content=llm_reasoning_content,
            web_search_results=web_search_results or {},
            solution_confidence=copy.deepcopy(solution_confidence or {}),
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    async def postprocess_result(
        self, result: SerializableResult, iteration_number: int, verbose: bool = True
    ):
        """
        Process the iteration result and return the best program from the database.

        Used by co-evolution where evaluation can be delayed.
        """
        self._process_iteration_result(
            result, iteration_number, checkpoint_callback=None, verbose=verbose
        )
        return self.database.get_best_program()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_image_output_dir(self) -> str:
        """Return the directory for saving VLM-generated images."""
        base = self.output_dir or "."
        d = os.path.join(base, "generated_images")
        os.makedirs(d, exist_ok=True)
        return d

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _process_iteration_result(
        self,
        result: Any,
        iteration: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Process the result from a single iteration.

        Args:
            result: The iteration result to process.
            iteration: Current iteration number.
            checkpoint_callback: Optional callback for checkpoint intervals.
            verbose: If True, log progress and metrics; if False, suppress logging.
        """
        if result.error:
            failed_child = None
            if getattr(result, "child_program_dict", None):
                program_class = getattr(self.database, "_program_class", Program)
                failed_child = program_class(**result.child_program_dict)
                failed_child.metadata = copy.deepcopy(failed_child.metadata or {})
                failed_child.metadata.update(
                    evaluation_status="failed",
                    error=result.error,
                    attempts_used=getattr(result, "attempts_used", 1),
                )
                # Preserve failed evaluations for inspection and resume without
                # admitting their error sentinel scores into the search population.
                archive = getattr(self.database, "_evolution_archive", None)
                if not isinstance(archive, dict):
                    archive = self.database._evolution_archive = {}
                archive[failed_child.id] = failed_child
                if result.prompt:
                    self.database.log_prompt(
                        template_key=(
                            "full_rewrite_user_message"
                            if not self.config.diff_based_generation
                            else "diff_user_message"
                        ),
                        program_id=failed_child.id,
                        prompt=copy.deepcopy(result.prompt),
                        responses=[result.llm_response] if result.llm_response else [],
                    )
            else:
                failed_web_search_results = getattr(result, "web_search_results", {}) or {}
                if failed_web_search_results:
                    self._store_unowned_web_search_results(
                        failed_web_search_results,
                        source_program_id=getattr(result, "parent_id", None),
                        association="failed_generation_attempt",
                    )
                failed_llm_reasoning = getattr(result, "llm_reasoning", {}) or {}
                if failed_llm_reasoning:
                    self._store_unowned_llm_reasoning(
                        failed_llm_reasoning,
                        source_program_id=getattr(result, "parent_id", None),
                        association="failed_generation_attempt",
                    )
            if self.evoduet is not None:
                self.evoduet.record_usage(
                    iteration=iteration,
                    score=None,
                    feedback=result.error,
                    **({"result_program_id": failed_child.id} if failed_child is not None else {}),
                )
            if verbose:
                logger.warning(f"Iteration {iteration} failed: {result.error}")
            self._checkpoint_after_iteration(iteration, checkpoint_callback, verbose)
            return

        program_class = getattr(self.database, "_program_class", Program)
        child_program = program_class(**result.child_program_dict)

        try:
            self.database.add(child_program, iteration=iteration)
        except Exception as exc:
            archive = getattr(self.database, "_evolution_archive", None)
            if isinstance(archive, dict):
                archive[child_program.id] = child_program
            if self.evoduet is not None:
                self.evoduet.record_usage(
                    iteration=iteration,
                    score=self._evoduet_program_score(child_program),
                    feedback=f"database add failed: {exc}",
                    result_program_id=child_program.id,
                )
            raise

        # EvoDuet: attribute this child's score/feedback to the query+results
        # it used this iteration (bumps their visit_count). No-op if nothing injected.
        if self.evoduet is not None:
            artifacts = child_program.artifacts if isinstance(child_program.artifacts, dict) else {}
            feedback = artifacts.get("feedback") or artifacts.get("text_feedback")
            self.evoduet.record_usage(
                iteration=iteration,
                score=self._evoduet_program_score(child_program),
                feedback=feedback if isinstance(feedback, str) else None,
                result_program_id=child_program.id,
            )

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child_program, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child_program.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        if verbose:
            logger.info(
                f"Iteration {iteration}: "
                f"Program {child_program.id} "
                f"(parent: {result.parent_id}) "
                f"completed in {result.iteration_time:.2f}s"
                f" (llm: {result.llm_generation_time:.2f}s,"
                f" eval: {result.eval_time:.2f}s)"
            )

        self._checkpoint_after_iteration(iteration, checkpoint_callback, verbose)

        if child_program.metrics:
            if verbose:
                metrics_str = ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in child_program.metrics.items()
                )
                logger.debug(f"Metrics: {metrics_str}")

            if not hasattr(self, "_warned_about_combined_score"):
                self._warned_about_combined_score = False

            if (
                "combined_score" not in child_program.metrics
                and not self._warned_about_combined_score
            ):
                if verbose:
                    logger.warning(
                        "⚠️  No 'combined_score' metric found in evaluation results. "
                        "Using 0.0 for discovery process guidance. "
                        "For better solution discovery results, please modify your evaluator to return a 'combined_score' "
                        "metric that properly weights different aspects of program performance."
                    )
                self._warned_about_combined_score = True

        if self.database.best_program_id == child_program.id and verbose:
            logger.info(f"🌟 New best solution found at iteration {iteration}")

    def _checkpoint_after_iteration(
        self,
        iteration: int,
        checkpoint_callback: Optional[Callable[[int], None]],
        verbose: bool = True,
    ) -> None:
        """Persist every completed iteration boundary, including failures."""
        if (
            checkpoint_callback is not None
            and iteration > 0
            and iteration % self.config.checkpoint_interval == 0
        ):
            if verbose:
                logger.debug(f"Checkpoint interval reached at iteration {iteration}")

            self.database.log_status()
            checkpoint_callback(iteration)
