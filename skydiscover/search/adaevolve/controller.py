"""
AdaEvolve Controller - Evolution loop with adaptive search intensity.

A clean implementation that uses the adaptive database for all
exploration/exploitation decisions. No explicit stagnation tracking -
search intensity handles exploration automatically.

Features:
- Adaptive sampling based on accumulated improvement signal
- Mode-aware prompting (exploration vs exploitation)
- Paradigm breakthrough for high-level strategy shifts
- Sibling context for learning from previous attempts
- Comprehensive JSON logging of all AdaEvolve signals
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from skydiscover.context_builder.adaevolve import AdaEvolveContextBuilder
from skydiscover.llm.llm_pool import LLMPool
from skydiscover.llm.response_metadata import reasoning_content
from skydiscover.search.adaevolve.paradigm import ParadigmGenerator
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
    _insert_evoduet,
)
from skydiscover.search.utils.discovery_utils import SerializableResult
from skydiscover.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)


class AdaEvolveController(DiscoveryController):
    """
    AdaEvolve evolution controller with adaptive search intensity.

    Key Features:
    1. Adaptive sampling: Search intensity per island determines exploration/exploitation
    2. Mode-aware prompting: Different guidance for exploration vs exploitation
    3. Sibling context: Shows previous mutations for learning
    4. Error retry: Retries failed generations with error context
    5. Island rotation: UCB-based selection via database.end_iteration()
    6. Paradigm breakthrough: High-level strategy shifts when globally stuck

    No explicit stagnation tracking - search intensity handles exploration
    automatically based on accumulated improvement signal.
    """

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        # Configuration
        db_config = self.config.search.database
        self.enable_retry = getattr(db_config, "enable_error_retry", True)
        self.max_retries = getattr(db_config, "max_error_retries", 2)
        self.num_context_programs = self.config.search.num_context_programs

        # Components
        self.llms = LLMPool(self.config.llm.models)
        self._configure_llm_result_fallback_sinks()
        self.context_builder = AdaEvolveContextBuilder(self.config)

        # Paradigm generator (if paradigm breakthrough is enabled)
        # Note: We check database.use_paradigm_breakthrough at runtime, not this init-time flag
        # This ensures correct behavior after checkpoint load
        if self.database.use_paradigm_breakthrough:
            model_names = ", ".join(m.name for m in self.guide_llms.models_cfg)
            logger.debug(f"Paradigm LLM: using guide_models [{model_names}]")

            self.paradigm_generator = ParadigmGenerator(
                llm_pool=self.guide_llms,
                system_message=self.config.context_builder.system_message or "",
                evaluator_code=self._load_evaluator_code(),
                num_paradigms=self.database.get_paradigm_num_to_generate(),
                eval_timeout=self.config.evaluator.timeout,
                language=self.config.language or "python",
                objective_names=getattr(db_config, "pareto_objectives", []),
                higher_is_better=getattr(db_config, "higher_is_better", {}),
                fitness_key=getattr(db_config, "fitness_key", None),
            )
        else:
            self.paradigm_generator = None

        # JSON logging for comprehensive AdaEvolve stats
        self._iteration_stats_log_path: Optional[str] = None
        self._iteration_stats_file = None
        self._last_sampling_mode: Optional[str] = None
        self._last_sampling_intensity: Optional[float] = None

        logger.debug(
            f"AdaEvolveController initialized "
            f"(language={self.config.language}, "
            f"paradigm_breakthrough={self.database.use_paradigm_breakthrough})"
        )

    def _load_evaluator_code(self) -> str:
        """Load evaluator source code for paradigm generation context."""
        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        return load_evaluator_code(self.evaluation_file)

    # =========================================================================
    # JSON Logging for AdaEvolve Stats
    # =========================================================================

    def _setup_iteration_stats_logging(self, output_dir: Optional[str] = None) -> None:
        """
        Set up JSON logging for comprehensive iteration statistics.

        Creates a JSONL file that records all AdaEvolve signals at each iteration.
        This enables detailed post-hoc analysis of the discovery process.

        Args:
            output_dir: Directory to write the log file. If None, uses database.config.db_path
        """
        # Determine output directory
        if output_dir is None:
            output_dir = self.output_dir
        if output_dir is None:
            output_dir = getattr(self.database.config, "db_path", None)
        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._iteration_stats_log_path = os.path.join(
            output_dir, f"adaevolve_iteration_stats_{timestamp}.jsonl"
        )

        logger.debug(
            f"AdaEvolve iteration stats will be logged to: {self._iteration_stats_log_path}"
        )

    def _log_iteration_stats(
        self,
        iteration: int,
        sampling_mode: Optional[str] = None,
        sampling_intensity: Optional[float] = None,
        child_program: Optional[Dict] = None,
        iteration_time: Optional[float] = None,
        llm_generation_time: Optional[float] = None,
        eval_time: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log comprehensive iteration statistics to JSON file.

        This method collects all AdaEvolve signals and writes them as a single
        JSON line to the log file for easy post-processing.

        Args:
            iteration: Current iteration number
            sampling_mode: The mode used for sampling (exploration/exploitation/balanced)
            sampling_intensity: The search intensity value used
            child_program: The child program dict if successfully generated
            iteration_time: Time taken for this iteration
            error: Error message if iteration failed
        """
        if self._iteration_stats_log_path is None:
            return

        try:
            # Get comprehensive stats from database
            stats = self.database.get_comprehensive_iteration_stats(
                iteration=iteration,
                sampling_mode=(
                    sampling_mode if sampling_mode is not None else self._last_sampling_mode
                ),
                sampling_intensity=(
                    sampling_intensity
                    if sampling_intensity is not None
                    else self._last_sampling_intensity
                ),
            )

            # Add timestamp
            stats["timestamp"] = datetime.now().isoformat()

            # Add iteration-specific info
            stats["iteration_result"] = {
                "success": error is None,
                "error": error,
                "iteration_time_seconds": iteration_time,
                "llm_generation_time_seconds": llm_generation_time,
                "eval_time_seconds": eval_time,
            }

            # Add child program info if available
            if child_program:
                stats["iteration_result"]["child_program"] = {
                    "id": child_program.get("id"),
                    "metrics": child_program.get("metrics"),
                    "generation": child_program.get("generation"),
                    "parent_id": child_program.get("parent_id"),
                }

            # Write to JSONL file
            with open(self._iteration_stats_log_path, "a") as f:
                f.write(json.dumps(stats, default=str) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log iteration stats: {e}")

    def get_iteration_stats_log_path(self) -> Optional[str]:
        """Get the path to the iteration stats log file."""
        return self._iteration_stats_log_path

    # =========================================================================
    # Main Evolution Loop
    # =========================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
    ) -> Optional[Program]:
        """Run evolution with adaptive search intensity and island rotation."""
        self._preflight_evoduet()
        total = start_iteration + max_iterations
        logger.info(
            f"AdaEvolve: Running {max_iterations} iterations "
            f"across {self.database.num_islands} islands"
        )

        # Set up comprehensive JSON logging for iteration stats
        self._setup_iteration_stats_logging()

        # Ensure all islands are seeded
        self._ensure_all_islands_seeded()

        for iteration in range(start_iteration, total):
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break

            try:
                await self._run_iteration(iteration, None)
            except Exception as e:
                logger.exception(f"Iteration {iteration} failed: {e}")
                self._process_iteration_result(
                    SerializableResult(error=str(e), iteration=iteration),
                    iteration,
                    None,
                    verbose=False,
                )
            finally:
                # CRITICAL: Tell database iteration is complete
                # This handles island rotation (UCB) and migration
                self.database.end_iteration(iteration)
                self.last_processed_iteration = iteration
                self._checkpoint_after_iteration(iteration, checkpoint_callback)

        logger.info("AdaEvolve completed")
        self.database.log_status()

        # Log final summary and stats file location
        if self._iteration_stats_log_path:
            logger.debug(f"AdaEvolve iteration stats saved to: {self._iteration_stats_log_path}")

        return self.database.get_best_program()

    def _checkpoint_after_iteration(self, iteration: int, checkpoint_callback) -> None:
        """Checkpoint after island rotation, spawning, and migration complete."""
        if (
            checkpoint_callback is not None
            and iteration > 0
            and iteration % self.config.checkpoint_interval == 0
        ):
            logger.debug(f"Checkpoint interval reached at iteration {iteration}")
            self.database.log_status()
            checkpoint_callback(iteration)

    def _ensure_all_islands_seeded(self) -> None:
        """Ensure all islands have at least one program."""
        # Find a seed program
        seed_program = None
        for i in range(self.database.num_islands):
            size = self.database.get_island_size(i)
            if size > 0 and seed_program is None:
                population = self.database.get_island_population(i)
                if population:
                    seed_program = population[0]
                    break

        if seed_program is None:
            logger.warning("No seed program found")
            return

        # Seed empty islands
        for i in range(self.database.num_islands):
            if self.database.get_island_size(i) == 0:
                copy = Program(
                    id=str(uuid.uuid4()),
                    solution=seed_program.solution,
                    language=seed_program.language,
                    metrics=seed_program.metrics.copy() if seed_program.metrics else {},
                    iteration_found=seed_program.iteration_found,
                    parent_id=None,
                    generation=0,
                    metadata={"seeded_to_island": i},
                )
                self.database.add(copy, iteration=0, target_island=i)
                logger.debug(f"Seeded island {i}")

    async def _run_iteration(self, iteration: int, checkpoint_callback) -> bool:
        """Execute one evolution iteration."""
        iteration_start_time = time.time()

        # Check for global paradigm stagnation
        # Use database flag directly to stay in sync after checkpoint load
        if self.database.use_paradigm_breakthrough and self.database.is_paradigm_stagnating():
            await self._generate_paradigms_if_needed(iteration)

        result = await self._run_normal_step(iteration)

        iteration_time = time.time() - iteration_start_time

        if result.error:
            self._process_iteration_result(result, iteration, checkpoint_callback, verbose=False)
            logger.warning(f"Iteration {iteration}: {result.error}")
            # Log failed iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=None,
                iteration_time=iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=result.error,
            )
            return False
        else:
            self._process_result(result, iteration, checkpoint_callback)
            # Log successful iteration stats
            self._log_iteration_stats(
                iteration=iteration,
                sampling_mode=self._last_sampling_mode,
                sampling_intensity=self._last_sampling_intensity,
                child_program=result.child_program_dict,
                iteration_time=result.iteration_time,
                llm_generation_time=result.llm_generation_time,
                eval_time=result.eval_time,
                error=None,
            )
            return True

    async def _generate_paradigms_if_needed(self, iteration: int) -> None:
        """Generate new paradigms if stagnating and none active."""
        if self.paradigm_generator is None:
            return

        if self.database.has_active_paradigm():
            return  # Already have paradigms to use

        logger.debug("Global paradigm stagnation detected, generating breakthrough ideas...")

        # Get current best program for context
        best_program = self.database.get_best_program()
        best_solution = best_program.solution if best_program else ""
        best_score = self.database.get_program_proxy_score(best_program)

        # Extract evaluator feedback from the best program's artifacts
        evaluator_feedback = None
        if best_program and best_program.artifacts:
            fb = best_program.artifacts.get("feedback")
            if fb and isinstance(fb, str):
                evaluator_feedback = fb

        # Get previously tried ideas for feedback
        previously_tried = self.database.get_previously_tried_ideas()

        # Generate new paradigms
        paradigms = await self.paradigm_generator.generate(
            current_program_solution=best_solution,
            current_best_score=best_score,
            previously_tried_ideas=previously_tried,
            evaluator_feedback=evaluator_feedback,
            llm_context={
                "iteration": iteration,
                "phase": "paradigm_generation",
                "source_program_id": best_program.id if best_program else None,
            },
        )

        if paradigms:
            self.database.set_paradigms(paradigms)
            logger.debug(f"Generated {len(paradigms)} breakthrough paradigms")
        else:
            logger.warning("Failed to generate paradigms")

    async def _run_normal_step(self, iteration: int) -> SerializableResult:
        """Run a normal iteration with optional retry."""
        last_error = None
        last_result: Optional[SerializableResult] = None
        aggregate_web_search_results: Dict[str, Any] = {}
        aggregate_llm_reasoning: Dict[str, Any] = {"calls": []}
        attempts = 1 + (self.max_retries if self.enable_retry else 0)
        iteration_context: Dict[str, Any] = {}
        cancelled = False

        try:
            for attempt in range(attempts):
                iteration_context["attempt_number"] = attempt + 1
                result = await self._generate_child(
                    iteration,
                    error_context=last_error,
                    iteration_context=iteration_context,
                )
                last_result = result
                self._merge_web_search_results(
                    aggregate_web_search_results,
                    getattr(result, "web_search_results", {}),
                )
                self._merge_llm_reasoning(
                    aggregate_llm_reasoning,
                    getattr(result, "llm_reasoning", {}),
                )
                if not result.error:
                    child = result.child_program_dict or {}
                    bound = self._bind_web_search_results(
                        aggregate_web_search_results,
                        program_id=child.get("id"),
                        source_program_id=result.parent_id,
                        association="generated_program",
                        successful_attempt=attempt + 1,
                    )
                    child["web_search_results"] = bound
                    result.web_search_results = bound
                    bound_reasoning = self._bind_llm_reasoning(
                        aggregate_llm_reasoning,
                        program_id=child.get("id"),
                        source_program_id=result.parent_id,
                        association="generated_program",
                        successful_attempt=attempt + 1,
                    )
                    child["llm_reasoning"] = bound_reasoning
                    child["llm_reasoning_content"] = reasoning_content(
                        bound_reasoning.get("calls", [])
                    )
                    child["llm_response"] = result.llm_response
                    result.llm_reasoning = bound_reasoning
                    result.llm_reasoning_content = child["llm_reasoning_content"]
                    return result
                last_error = result.error
                logger.debug(f"Attempt {attempt + 1}/{attempts} failed: {last_error}")

            return SerializableResult(
                error=f"All {attempts} attempts failed: {last_error}",
                iteration=iteration,
                parent_id=last_result.parent_id if last_result is not None else None,
                web_search_results=aggregate_web_search_results,
                llm_response=last_result.llm_response if last_result is not None else None,
                llm_reasoning=aggregate_llm_reasoning,
                llm_reasoning_content=reasoning_content(aggregate_llm_reasoning.get("calls", [])),
            )
        except asyncio.CancelledError:
            cancelled = True
            source_program_id = last_result.parent_id if last_result is not None else None
            if source_program_id is None:
                selection = iteration_context.get("selection")
                if isinstance(selection, tuple) and len(selection) > 3:
                    source_program_id = getattr(selection[3], "id", None)
            self._store_unowned_llm_reasoning(
                aggregate_llm_reasoning,
                source_program_id=source_program_id,
                association="cancelled_generation_attempt",
            )
            if iteration_context.get("evoduet_sent") and self.evoduet is not None:
                self.evoduet.record_usage(
                    iteration=iteration,
                    score=None,
                    feedback="iteration cancelled",
                )
            raise
        finally:
            release = getattr(getattr(self, "llms", None), "release_model", None)
            if callable(release):
                release(iteration)
            # An empty retrieval still stages a policy prediction. Cancelled
            # iterations cannot produce its outer outcome, even with no evidence.
            if (cancelled or iteration_context.get("evoduet")) and not (
                iteration_context.get("evoduet_sent", False)
            ):
                self._discard_evoduet_usage(iteration)

    def _process_result(
        self,
        result: SerializableResult,
        iteration: int,
        checkpoint_callback,
    ) -> None:
        """Process a successful result by adding to database."""
        child = Program(**result.child_program_dict)

        # Add to database (database handles which island)
        try:
            self.database.add(child, iteration=iteration, parent_id=result.parent_id)
        except Exception as exc:
            archive = getattr(self.database, "_evolution_archive", None)
            if isinstance(archive, dict):
                archive[child.id] = child
            self._record_evoduet_outcome(
                result,
                iteration,
                child,
                feedback_override=f"database add failed: {exc}",
            )
            raise

        self._record_evoduet_outcome(result, iteration, child)

        # Fire monitor callback (live dashboard)
        if self.monitor_callback:
            try:
                self.monitor_callback(child, iteration)
            except Exception:
                logger.debug("Monitor callback error", exc_info=True)

        # Log prompt
        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user_message"
                    if not self.config.diff_based_generation
                    else "diff_user_message"
                ),
                program_id=child.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
            )

        # Log progress
        logger.info(
            f"Iteration {iteration}: Program {child.id[:8]} "
            f"(parent: {result.parent_id[:8] if result.parent_id else 'None'}) "
            f"completed in {result.iteration_time:.2f}s"
            f" (llm: {result.llm_generation_time:.2f}s,"
            f" eval: {result.eval_time:.2f}s)"
        )

        # Log metrics
        if child.metrics:
            metrics_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in child.metrics.items()
            )
            logger.debug(f"Metrics: {metrics_str}")

        # Check for new best
        if self.database.is_multiobjective_enabled():
            pareto_front_ids = {program.id for program in self.database.get_pareto_front()}
            if child.id in pareto_front_ids:
                logger.info(f"Program entered the global Pareto front at iteration {iteration}")
            if self.database.best_program_id == child.id:
                logger.info(f"New representative Pareto solution found at iteration {iteration}")
        elif self.database.best_program_id == child.id:
            logger.info(f"New best solution found at iteration {iteration}")

    def _record_evoduet_outcome(
        self,
        result: SerializableResult,
        iteration: int,
        evaluated_program: Optional[Program] = None,
        feedback_override: Optional[str] = None,
    ) -> None:
        """Credit evidence without changing AdaEvolve's database semantics."""
        if self.evoduet is None:
            return
        child = result.child_program_dict or {}
        artifacts = child.get("artifacts") or {}
        feedback = (
            feedback_override
            if feedback_override is not None
            else artifacts.get("feedback") or artifacts.get("text_feedback")
        )
        metrics = child.get("metrics") or {}
        self.evoduet.record_usage(
            iteration=iteration,
            score=(
                self._evoduet_program_score(evaluated_program)
                if evaluated_program is not None
                else get_score(metrics) if metrics else None
            ),
            feedback=feedback if isinstance(feedback, str) else None,
            result_program_id=child.get("id") or "",
        )

    # =========================================================================
    # Child Generation
    # =========================================================================

    async def _generate_child(
        self,
        iteration: int,
        error_context: Optional[str] = None,
        force_exploration: bool = False,
        iteration_context: Optional[Dict[str, Any]] = None,
        attempt_number: int = 1,
    ) -> SerializableResult:
        """Generate and evaluate a single child program."""
        attempt_number = int((iteration_context or {}).get("attempt_number", attempt_number))
        try:
            if not self.database.programs:
                return await self._run_from_scratch_iteration(
                    iteration, attempt_number=attempt_number
                )

            # Ensure all islands are seeded (needed after from-scratch bootstrap)
            self._ensure_all_islands_seeded()

            selection = (iteration_context or {}).get("selection")
            if selection is None:
                # Sample once so retries use the same parent and retrieved evidence.
                parent_dict, context_programs_dict = self.database.sample(
                    self.num_context_programs,
                    force_exploration=force_exploration,
                )
                if not parent_dict:
                    logger.error("sample() returned empty parent dict")
                    return SerializableResult(
                        error="Empty parent dict from sample()", iteration=iteration
                    )
                parent_label = list(parent_dict.keys())[0]
                parent = list(parent_dict.values())[0]
                sampling_mode = getattr(self.database, "_last_sampling_mode", None) or "balanced"

                paradigm = (
                    self.database.get_current_paradigm()
                    if self.database.use_paradigm_breakthrough
                    else None
                )
                if paradigm:
                    best_program = self.database.get_best_program()
                    if best_program:
                        parent_dict = {parent_label: best_program}
                        parent = best_program

                selection = (
                    parent_dict,
                    context_programs_dict,
                    parent_label,
                    parent,
                    sampling_mode,
                    paradigm,
                )
                if iteration_context is not None:
                    iteration_context["selection"] = selection
            else:
                (
                    parent_dict,
                    context_programs_dict,
                    parent_label,
                    parent,
                    sampling_mode,
                    paradigm,
                ) = selection

            # Capture sampling mode and intensity for logging
            self._last_sampling_mode = sampling_mode
            current_island = self.database.current_island
            if self.database.use_adaptive_search:
                self._last_sampling_intensity = self.database.adapter.get_search_intensity(
                    current_island
                )
            else:
                self._last_sampling_intensity = self.database.fixed_intensity

            # Gather siblings for sibling context
            siblings = []
            if hasattr(self.database, "get_children"):
                try:
                    siblings = self.database.get_children(parent.id)
                except (AttributeError, NotImplementedError):
                    pass

            # Build context for prompt generation
            # Only database-derived data — config values are read by the
            # context builder from self.config directly.
            context = {
                "program_metrics": parent.metrics,
                "other_context_programs": context_programs_dict,
                # AdaEvolve-specific keys (consumed by AdaEvolveContextBuilder)
                "paradigm": paradigm,
                "siblings": siblings,
                "error_context": error_context,
            }
            # Include any extra prompt context
            for k, v in self._prompt_context.items():
                if k not in context:
                    context[k] = v

            # Build prompt (AdaEvolveContextBuilder handles paradigm/sibling/error formatting).
            # The builder records the rendered history it showed, which the World
            # Knowledge Layer then receives verbatim.
            prompt_capture: Dict[str, Any] = {}
            prompt = self.context_builder.build_prompt(
                parent_dict, context, confidence_context=prompt_capture
            )

            if iteration_context is None or "evoduet" not in iteration_context:
                evoduet = await self._maybe_run_evoduet(
                    parent,
                    str(prompt_capture.get("evolutionary_history") or ""),
                    iteration,
                )
                if iteration_context is not None:
                    iteration_context["evoduet"] = evoduet
            else:
                evoduet = iteration_context["evoduet"]
            # Mark paradigm as used after prompt is built
            if paradigm and not (iteration_context or {}).get("paradigm_used"):
                self.database.use_paradigm()
                if iteration_context is not None:
                    iteration_context["paradigm_used"] = True

            # Build tracking info for child program
            parent_info = (parent_label, parent.id)
            context_info = [
                (label, p.id) for label, programs in context_programs_dict.items() for p in programs
            ]
            context_program_ids = [
                p.id for programs in context_programs_dict.values() for p in programs
            ]

            # Apply human feedback (append or replace mode)
            if self.feedback_reader:
                self.feedback_reader.set_current_prompt(prompt["system"])
                feedback = self.feedback_reader.read()
                if feedback:
                    prompt = self.feedback_reader.apply_feedback(prompt)
                    self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

            # Feedback may replace the prompt; evidence belongs in the final request.
            if evoduet:
                prompt["user"] = _insert_evoduet(prompt["user"], evoduet)

            if iteration_context is not None:
                iteration_context["evoduet_sent"] = bool(
                    evoduet
                ) or iteration_context.get("evoduet_sent", False)

            # Generate and evaluate
            return await self._execute_generation(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=context_programs_dict,
                attempt_number=attempt_number,
            )

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return SerializableResult(error=str(e), iteration=iteration)

    # =========================================================================
    # LLM Generation
    # =========================================================================

    async def _execute_generation(
        self,
        parent: Program,
        prompt: Dict[str, str],
        iteration: int,
        parent_info: Optional[tuple] = None,
        context_info: Optional[List[tuple]] = None,
        context_program_ids: Optional[List[str]] = None,
        other_context_programs: Optional[Dict] = None,
        attempt_number: int = 1,
    ) -> SerializableResult:
        """Execute one generation while retaining LLM metadata on every exit path."""
        web_search_results: Dict[str, Any] = {}
        llm_reasoning_calls: List[Dict[str, Any]] = []
        llm_reasoning: Dict[str, Any] = {"calls": llm_reasoning_calls}
        try:
            return await self._execute_generation_impl(
                parent,
                prompt,
                iteration,
                parent_info=parent_info,
                context_info=context_info,
                context_program_ids=context_program_ids,
                other_context_programs=other_context_programs,
                attempt_number=attempt_number,
                trace_web_search_results=web_search_results,
                trace_llm_reasoning=llm_reasoning,
            )
        except asyncio.CancelledError:
            self._store_unowned_web_search_results(
                web_search_results,
                source_program_id=parent.id,
                association="cancelled_generation_attempt",
            )
            self._store_unowned_llm_reasoning(
                llm_reasoning,
                source_program_id=parent.id,
                association="cancelled_generation_attempt",
            )
            raise
        except Exception as exc:
            logger.exception("Generation failed after LLM execution: %s", exc)
            return SerializableResult(
                error=str(exc),
                iteration=iteration,
                parent_id=parent.id,
                web_search_results=web_search_results,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )

    async def _execute_generation_impl(
        self,
        parent: Program,
        prompt: Dict[str, str],
        iteration: int,
        parent_info: Optional[tuple] = None,
        context_info: Optional[List[tuple]] = None,
        context_program_ids: Optional[List[str]] = None,
        other_context_programs: Optional[Dict] = None,
        attempt_number: int = 1,
        trace_web_search_results: Optional[Dict[str, Any]] = None,
        trace_llm_reasoning: Optional[Dict[str, Any]] = None,
    ) -> SerializableResult:
        """Execute LLM generation and evaluation."""
        start_time = time.time()

        image_path = None
        child_id = str(uuid.uuid4())
        tavily_searches: List[Dict[str, Any]] = []
        web_search_results = (
            trace_web_search_results if trace_web_search_results is not None else {}
        )
        llm_reasoning = trace_llm_reasoning if trace_llm_reasoning is not None else {}
        llm_reasoning_calls = llm_reasoning.setdefault("calls", [])
        if not isinstance(llm_reasoning_calls, list):
            llm_reasoning_calls = []
            llm_reasoning["calls"] = llm_reasoning_calls

        # Generate
        llm_generation_time = 0.0
        try:
            llm_start = time.time()
            if self.config.language == "image":
                from skydiscover.search.utils.discovery_utils import build_image_content

                user_content = build_image_content(
                    prompt["user"], parent, other_context_programs or {}
                )
                result = await self._call_llm(
                    prompt["system"],
                    user_content,
                    image_output=True,
                    output_dir=self._get_image_output_dir(),
                    program_id=child_id,
                    llm_context={
                        "iteration": iteration,
                        "attempt": attempt_number,
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
                self._merge_llm_reasoning(llm_reasoning, getattr(result, "llm_reasoning", {}))
                response = result.text or ""
                image_path = result.image_path
                if not image_path:
                    return SerializableResult(
                        error="VLM did not generate an image",
                        iteration=iteration,
                        parent_id=parent.id,
                        web_search_results=web_search_results,
                        llm_response=response,
                        llm_reasoning=llm_reasoning,
                        llm_reasoning_content=reasoning_content(llm_reasoning_calls),
                    )
            else:
                result = await self._call_llm(
                    prompt["system"],
                    prompt["user"],
                    llm_context={
                        "iteration": iteration,
                        "attempt": attempt_number,
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
                self._merge_llm_reasoning(llm_reasoning, getattr(result, "llm_reasoning", {}))
                response = result.text
            llm_generation_time = time.time() - llm_start
        except Exception as e:
            if tavily_searches:
                web_search_results["tavily_searches"] = tavily_searches
            return SerializableResult(
                error=f"LLM error: {e}",
                iteration=iteration,
                parent_id=parent.id,
                web_search_results=web_search_results,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )

        if not response and self.config.language != "image":
            return SerializableResult(
                error="Empty LLM response",
                iteration=iteration,
                parent_id=parent.id,
                web_search_results=web_search_results,
                llm_response=response,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )

        # Parse code from response
        if self.config.language == "image":
            child_solution = response or "(image generated)"
            changes = "Image generation"
        elif self.config.diff_based_generation:
            diffs = extract_diffs(response)
            if diffs:
                child_solution = apply_diff(parent.solution, response)
                changes = format_diff_summary(diffs)
            else:
                # No diffs found, try full rewrite
                child_solution = parse_full_rewrite(response, self.config.language)
                changes = "Full rewrite"
        else:
            child_solution = parse_full_rewrite(response, self.config.language)
            changes = "Full rewrite"

        if not child_solution:
            return SerializableResult(
                error="No valid solution in response",
                iteration=iteration,
                parent_id=parent.id,
                web_search_results=web_search_results,
                llm_response=response,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )

        # Evaluate
        try:
            eval_input = image_path if self.config.language == "image" else child_solution
            eval_start = time.time()
            eval_result = await self.evaluator.evaluate_program(eval_input, child_id)
            eval_time = time.time() - eval_start
            self._merge_llm_reasoning(llm_reasoning, getattr(eval_result, "llm_reasoning", {}))
        except Exception as e:
            return SerializableResult(
                error=f"Evaluation error: {e}",
                iteration=iteration,
                parent_id=parent.id,
                web_search_results=web_search_results,
                llm_response=response,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )

        metrics = eval_result.metrics
        artifacts = eval_result.artifacts

        # Check for eval failure (e.g., constraint violation, malformed solution)
        error_msg = self._evaluation_failure_reason(metrics, artifacts)
        if error_msg is not None:
            return SerializableResult(
                error=f"Eval failure: {error_msg}",
                iteration=iteration,
                parent_id=parent.id,
                web_search_results=web_search_results,
                llm_response=response,
                llm_reasoning=llm_reasoning,
                llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            )

        # Extract image_path from evaluator metrics (non-image mode fallback)
        if not image_path:
            image_path = (
                metrics.pop("image_path", None)
                if isinstance(metrics.get("image_path"), str)
                else None
            )

        # Build child program with full tracking info
        child_metadata = {"changes": changes, "parent_metrics": parent.metrics}
        if image_path:
            child_metadata["image_path"] = image_path
        if self.evoduet is not None:
            record_id = self.evoduet.staged_search_record_id(iteration)
            if record_id:
                child_metadata["evoduet_search_record_id"] = record_id
        child = Program(
            id=child_id,
            solution=child_solution,
            language=self.config.language,
            metrics=metrics,
            iteration_found=iteration,
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            parent_info=parent_info,
            context_info=context_info,
            generation=parent.generation + 1,
            metadata=child_metadata,
            artifacts=artifacts,
            llm_response=response,
            llm_reasoning=self._bind_llm_reasoning(
                llm_reasoning,
                program_id=child_id,
                source_program_id=parent.id,
                association="generated_program",
                successful_attempt=attempt_number,
            ),
            llm_reasoning_content=reasoning_content(llm_reasoning_calls),
            web_search_results=self._bind_web_search_results(
                web_search_results,
                program_id=child_id,
                source_program_id=parent.id,
                association="generated_program",
                successful_attempt=attempt_number,
            ),
        )

        iteration_time = time.time() - start_time

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id,
            other_context_ids=context_program_ids,
            iteration_time=iteration_time,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=response,
            llm_reasoning=child.llm_reasoning,
            llm_reasoning_content=child.llm_reasoning_content,
            iteration=iteration,
            web_search_results=child.web_search_results,
        )
