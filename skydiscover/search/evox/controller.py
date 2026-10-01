"""
Co-evolution controller.

Runs evolution for the main *solution* database while also evolving a
separate *search* program/database in the same process.
"""

import logging
import os
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.evox.utils.coevolve_logging import (
    handle_generation_failure,
    log_search_algorithm_generated,
    make_json_serializable,
    update_saved_search_algorithm_score,
)
from skydiscover.search.evox.utils.search_scorer import LogWindowScorer
from skydiscover.search.evox.utils.variation_operator_generator import generate_variation_operators
from skydiscover.search.registry import get_program, setup_search
from skydiscover.search.utils.discovery_utils import SerializableResult, load_database_from_file

logger = logging.getLogger(__name__)


class CoEvolutionController(DiscoveryController):
    """
    Co-evolves solution programs alongside search algorithms.

    The solution database uses an evolving search algorithm for sampling,
    while the search algorithm itself is scored based on solution improvements.
    """

    # Adaptive mode defaults
    DEFAULT_SWITCH_RATIO = 0.10  # Evolve search after 10% of total iterations stagnate
    DEFAULT_IMPROVEMENT_THRESHOLD = 0.01

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        self._init_search_evolution_controller()
        self._init_output_dir(controller_input)

    def _init_search_evolution_controller(self) -> None:
        """Initialize search controller, scorer, and load initial algorithm."""
        db_cfg = self.config.search.database
        if not db_cfg.database_file_path:
            raise ValueError(
                "config.search.database.database_file_path is required for co-evolution"
            )

        controller_input, self._search_initial_code = setup_search(
            initial_program_path=db_cfg.database_file_path,
            evaluation_file=db_cfg.evaluation_file,
            config_path=db_cfg.config_path,
            output_dir=self.config.search.output_dir,
            evaluator_env_vars=self.evaluator_env_vars,
            parent_llm_config=self.config.llm if self.config.search.share_llm else None,
        )
        self.search_controller = DiscoveryController(controller_input)
        self.search_scorer = LogWindowScorer()
        self._active_search_algorithm_code = self._search_initial_code

        self._log_coevolution_setup(db_cfg)
        self._init_search_tracking()

    def _init_search_tracking(self) -> None:
        """Initialize search evolution tracking state."""
        self._pending_search_result: Optional[SerializableResult] = None
        self._best_search_score: Optional[float] = None
        self._num_search_evolutions = 0

        self._switch_interval = getattr(self.config.search, "switch_interval", None)
        self._stagnant_count = 0
        self._meta_evolution_failures = 0
        self._guide_llm_available = True
        self._meta_llm_available = True
        self._last_tracked_best_score: Optional[float] = None

        self._diverge_label = ""
        self._refine_label = ""

        self._fallback_database = None
        self._fallback_search_code = None

    def _preflight_evoduet(self) -> None:
        """Validate both solution- and search-side retrieval before model work."""
        super()._preflight_evoduet()
        self.search_controller._preflight_evoduet()

    def has_evoduet_checkpoint_state(self) -> bool:
        # This file also carries EvoX's active algorithm/database/scorer state.
        return True

    def evoduet_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """Capture both knowledge layers and the delayed search-side outcome."""
        database = self.search_controller.database
        scorer = self.search_scorer
        return {
            "format": "evox-v1",
            "solution": super().evoduet_checkpoint_state(),
            "search": self.search_controller.evoduet_checkpoint_state(),
            "evox": {
                "solution_database": self._solution_database_checkpoint_state(),
                "search_database": {
                    "programs": [program.to_dict() for program in database.programs.values()],
                    "best_program_id": database.best_program_id,
                    "last_iteration": database.last_iteration,
                    "initial_program_id": database.initial_program_id,
                    "initial_program_score": database.initial_program_score,
                    "prompts_by_program": database.prompts_by_program,
                },
                "pending_search_result": (
                    asdict(self._pending_search_result)
                    if self._pending_search_result is not None
                    else None
                ),
                "search_scorer": {
                    "algorithm_id": scorer.algorithm_id,
                    "start_score": scorer._start_score,
                    "start_iteration": scorer._start_iteration,
                    "best_scores": list(scorer._best_scores),
                },
                "tracking": {
                    "best_search_score": self._best_search_score,
                    "num_search_evolutions": self._num_search_evolutions,
                    "switch_interval": self._switch_interval,
                    "stagnant_count": self._stagnant_count,
                    "last_tracked_best_score": self._last_tracked_best_score,
                    "diverge_label": self._diverge_label,
                    "refine_label": self._refine_label,
                    "active_search_algorithm_code": self._active_search_algorithm_code,
                    "fallback_search_code": getattr(self, "_fallback_search_code", None),
                },
            },
        }

    def _solution_database_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        database = getattr(self, "database", None)
        if database is None:
            return None
        return {
            "programs": [program.to_dict() for program in database.programs.values()],
            "best_program_id": database.best_program_id,
            "last_iteration": database.last_iteration,
            "initial_program_id": database.initial_program_id,
            "initial_program_score": database.initial_program_score,
            "prompts_by_program": database.prompts_by_program,
            "language": getattr(database, "language", None),
        }

    def load_evoduet_checkpoint_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore EvoX's nested layer and enough state to finish delayed credit."""
        if not isinstance(state, dict) or state.get("format") != "evox-v1":
            raise ValueError("unsupported EvoX checkpoint format; start a new run")

        if self.evoduet is not None and not isinstance(state.get("solution"), dict):
            raise ValueError("EvoX checkpoint is missing solution EvoDuet state")
        if self.search_controller.evoduet is not None and not isinstance(state.get("search"), dict):
            raise ValueError("EvoX checkpoint is missing search EvoDuet state")
        super().load_evoduet_checkpoint_state(state.get("solution"))
        self.search_controller.load_evoduet_checkpoint_state(state.get("search"))

        evox = state.get("evox")
        if not isinstance(evox, dict):
            raise ValueError("EvoX checkpoint is missing controller state")
        self._restore_search_database(evox.get("search_database"))

        pending = evox.get("pending_search_result")
        if pending is not None and not isinstance(pending, dict):
            raise ValueError("EvoX pending search result must be an object")
        self._pending_search_result = SerializableResult(**pending) if pending else None

        scorer = evox.get("search_scorer")
        if not isinstance(scorer, dict) or not isinstance(scorer.get("best_scores"), list):
            raise ValueError("EvoX checkpoint is missing search scorer state")
        self.search_scorer.algorithm_id = scorer.get("algorithm_id") or "unknown"
        self.search_scorer._start_score = scorer.get("start_score")
        self.search_scorer._start_iteration = scorer.get("start_iteration")
        self.search_scorer._best_scores = [float(score) for score in scorer["best_scores"]]

        tracking = evox.get("tracking")
        if not isinstance(tracking, dict):
            raise ValueError("EvoX checkpoint is missing search tracking state")
        for field, default in (
            ("best_search_score", None),
            ("num_search_evolutions", 0),
            ("switch_interval", None),
            ("stagnant_count", 0),
            ("last_tracked_best_score", None),
            ("diverge_label", ""),
            ("refine_label", ""),
            ("active_search_algorithm_code", self._search_initial_code),
            ("fallback_search_code", None),
        ):
            setattr(self, f"_{field}", tracking.get(field, default))

        # Runner restores the persisted programs through the configured (initial)
        # database class before this hook runs. Replaying them through the saved
        # classes rebuilds any algorithm-specific indexes and sampling state.
        if hasattr(self, "database"):
            source = self.database
            solution_state = evox.get("solution_database")
            if solution_state is not None and (
                not isinstance(solution_state, dict)
                or not isinstance(solution_state.get("programs"), list)
            ):
                raise ValueError("EvoX checkpoint has invalid solution database state")
            self.database, _ = self._database_from_search_code(
                self._active_search_algorithm_code,
                source,
                source_state=solution_state,
            )
            self._fallback_database = None
            if self._fallback_search_code:
                self._fallback_database, _ = self._database_from_search_code(
                    self._fallback_search_code, self.database
                )
            if self.evaluator.llm_judge:
                self.evaluator.llm_judge.database = self.database

    def _restore_search_database(self, state: Any) -> None:
        if not isinstance(state, dict) or not isinstance(state.get("programs"), list):
            raise ValueError("EvoX checkpoint is missing search database state")
        database = self.search_controller.database
        program_class = getattr(database, "_program_class", Program)
        from_dict = getattr(program_class, "from_dict", None)
        programs = {}
        for row in state["programs"]:
            if not isinstance(row, dict):
                raise ValueError("EvoX search database contains an invalid program")
            program = from_dict(row) if callable(from_dict) else program_class(**row)
            programs[program.id] = program
        database.programs = programs
        database.best_program_id = state.get("best_program_id")
        database.last_iteration = int(state.get("last_iteration") or 0)
        database.initial_program_id = state.get("initial_program_id")
        database.initial_program_score = state.get("initial_program_score")
        database.prompts_by_program = state.get("prompts_by_program")

    def _init_output_dir(self, controller_input: DiscoveryControllerInput) -> None:
        base_dir = controller_input.output_dir or os.path.join(
            os.path.dirname(controller_input.evaluation_file),
            "outputs",
            self.config.search.type,
        )
        self.search_outputs_dir = os.path.join(base_dir, "search")
        os.makedirs(self.search_outputs_dir, exist_ok=True)

    async def _check_meta_llm_availability(self) -> None:
        """Probe guide and meta-search LLM pools at startup and warn if unreachable."""
        import asyncio

        search_controller = getattr(self, "search_controller", None)
        guide_pool = getattr(search_controller, "guide_llms", None)
        meta_pool = getattr(search_controller, "llms", None)
        if guide_pool is None or meta_pool is None:
            # Custom subclasses and lightweight test controllers may omit the
            # nested meta-search controller entirely.
            self._guide_llm_available = True
            self._meta_llm_available = True
            return

        guide_endpoint = guide_pool.models_cfg[0].api_base if guide_pool.models_cfg else "unknown"
        meta_endpoint = meta_pool.models_cfg[0].api_base if meta_pool.models_cfg else "unknown"

        guide_ok, meta_ok = await asyncio.gather(
            guide_pool.check_availability(),
            meta_pool.check_availability(),
        )

        self._guide_llm_available = guide_ok
        self._meta_llm_available = meta_ok

        if guide_ok and meta_ok:
            logger.info(
                "Meta-search LLM connectivity verified (guide: %s, meta: %s)",
                guide_endpoint,
                meta_endpoint,
            )
            return

        if not guide_ok:
            logger.warning(
                "Guide LLM (label generation) at %s is not reachable. "
                "Variation operator labels will fall back to generic defaults. "
                "The meta-search can still evolve strategies, but without "
                "problem-specific labels. To fix: set search.share_llm: true "
                "to use the main discovery endpoint.",
                guide_endpoint,
            )

        if not meta_ok:
            logger.warning(
                "Meta-search LLM (search strategy evolution) at %s is not "
                "reachable. Search strategy evolution will be skipped; the "
                "initial search algorithm will be used throughout the run. "
                "Solution evolution is unaffected. To fix: set "
                "search.share_llm: true to use the main discovery endpoint.",
                meta_endpoint,
            )

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback=None,
        post_process_result: Optional[bool] = True,
    ):
        """Run co-evolution of solution programs and search algorithms."""
        self._preflight_evoduet()
        self.total_solution_iterations = start_iteration + max_iterations
        self._max_solution_iterations = max_iterations

        if self._switch_interval is None:
            self._switch_interval = max(1, int(max_iterations * self.DEFAULT_SWITCH_RATIO))
            logger.debug(f"Switch if {self._switch_interval} iterations of stagnation detected")

        self.start_db_stats = self.database.get_statistics(
            improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD
        )

        # Check meta-search LLM availability before starting
        await self._check_meta_llm_availability()

        # A pending candidate owns the restored scoring window until delayed credit.
        if self._pending_search_result is None:
            self._reset_search_window()

        # Generate variation labels for the search algorithm
        await self._generate_variation_operators()

        # Run co-evolution
        iteration = start_iteration
        while iteration < self.total_solution_iterations:
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested")
                break

            current_iteration = iteration
            try:
                # Run solution iteration
                remaining = self.total_solution_iterations - current_iteration
                result = await self._run_iteration(current_iteration, retry_times=min(3, remaining))
                attempts_used = min(remaining, max(1, int(getattr(result, "attempts_used", 1))))

                if result.error:
                    self._process_iteration_result(result, current_iteration, None, verbose=False)
                    logger.warning(
                        f"Iteration {current_iteration} failed "
                        f"(used {attempts_used} attempts): {result.error}"
                    )
                    # Database error after a switch: fall back and retry
                    if self._fallback_database is not None and result.prompt is None:
                        self._restore_fallback_database()
                        continue  # Retry same iteration with restored database
                else:
                    self._process_iteration_result(result, current_iteration, None)

            except Exception as e:
                logger.error(f"Error in iteration {current_iteration}: {e}", exc_info=True)
                self._process_iteration_result(
                    SerializableResult(error=str(e), iteration=current_iteration),
                    current_iteration,
                    None,
                    verbose=False,
                )
                # Exception from database.add() after a switch: fall back and retry
                if self._fallback_database is not None:
                    self._restore_fallback_database()
                    continue  # Retry same iteration
                iteration = current_iteration + 1
                self.last_processed_iteration = current_iteration
                self._checkpoint_after_solution_step(
                    current_iteration,
                    current_iteration,
                    checkpoint_callback,
                )
                continue

            for _ in range(attempts_used):
                self._record_search_window_step()

            completed_solution_iter = current_iteration + attempts_used - 1
            next_iteration = completed_solution_iter + 1

            # Search-strategy evolution must not consume a solution iteration.
            if next_iteration < self.total_solution_iterations and self._should_evolve_search():
                logger.debug(
                    "Stagnation detected -> evolving search strategy "
                    f"(solution_iter={completed_solution_iter})"
                )
                try:
                    await self._evolve_search(completed_solution_iter)
                except Exception as e:
                    logger.warning(
                        "Search-strategy evolution failed (%s); continuing with the "
                        "current strategy. Solution evolution is unaffected.",
                        e,
                        exc_info=True,
                    )
                    self._meta_evolution_failures = (
                        getattr(self, "_meta_evolution_failures", 0) + 1
                    )

            iteration = next_iteration
            self.last_processed_iteration = completed_solution_iter
            self._checkpoint_after_solution_step(
                current_iteration,
                completed_solution_iter,
                checkpoint_callback,
            )

        if self._pending_search_result:
            await self._finalize_pending_search()

        logger.info(f"[SOLUTION EVOLUTION] Evolution completed: {self.database.name}")
        if getattr(self, "_meta_evolution_failures", 0):
            logger.warning(
                "Meta-search evolution failed %d time(s) during this run.",
                self._meta_evolution_failures,
            )
        return self.database.get_best_program()

    def _checkpoint_after_solution_step(
        self, first_iteration: int, completed_iteration: int, checkpoint_callback
    ) -> None:
        """Checkpoint only after EvoX's scorer and strategy transition are durable."""
        interval = self.config.checkpoint_interval
        if (
            checkpoint_callback is not None
            and completed_iteration > 0
            and completed_iteration // interval > (first_iteration - 1) // interval
        ):
            logger.debug(f"Checkpoint interval reached at iteration {completed_iteration}")
            self.database.log_status()
            checkpoint_callback(completed_iteration)

    def _should_evolve_search(self) -> bool:
        """Check if it's time to evolve the search algorithm (stagnation-based)."""
        if not self._meta_llm_available:
            return False

        current = self._get_best_score()

        if self._last_tracked_best_score is None:
            self._stagnant_count = 0
        elif (current - self._last_tracked_best_score) > self.DEFAULT_IMPROVEMENT_THRESHOLD:
            self._stagnant_count = 0
        else:
            self._stagnant_count += 1

        self._last_tracked_best_score = current

        if self._stagnant_count >= self._switch_interval:
            self._stagnant_count = 0
            return True

        return False

    async def _evolve_search(self, solution_iter: int) -> None:
        """Handle search evolution: score previous algorithm, generate and switch to new one."""

        if not self.search_controller.database.programs:
            await self._initialize_first_search_program(solution_iter)
            return

        # If there is a pending search result, finalize it (as search window is reset)
        if self._pending_search_result:
            await self._finalize_pending_search()

        self._reset_search_window()
        await self._generate_and_validate_search_algorithm(solution_iter)

    async def _finalize_pending_search(self) -> None:
        """Score the pending search algorithm and add it to the search strategy database."""
        pending_iteration = self._num_search_evolutions
        is_new_best = self._assign_search_score()

        await update_saved_search_algorithm_score(
            self.search_outputs_dir,
            pending_iteration,
            self._pending_search_result,
            is_new_best=is_new_best,
            db_stats=self.database.get_statistics(),
        )
        await self.search_controller.postprocess_result(
            self._pending_search_result, self._num_search_evolutions, verbose=False
        )

        self._pending_search_result = None
        self._num_search_evolutions += 1
        self._fallback_database = None
        self._fallback_search_code = None

    async def _initialize_first_search_program(self, solution_iter: int) -> None:
        """Initialize and score the first (file-based) search program."""
        start_score = (
            self.search_scorer.get_start_score()
            or getattr(self.database, "initial_program_score", None)
            or 0.0
        )
        metrics = self._compute_search_metrics(
            start_score=start_score,
            best_scores=None,
            horizon=self._switch_interval,
            start_iteration=0,
        )
        search_score = float(metrics.get("combined_score", 0.0) or 0.0)

        initial_program = get_program(
            self.search_controller.config,
            self._search_initial_code,
            str(uuid.uuid4()),
            metrics,
            self._num_search_evolutions,
        )
        initial_program.metadata = initial_program.metadata or {}
        initial_program.metadata["start_db_stats"] = make_json_serializable(self.start_db_stats)
        initial_program.metadata["end_db_stats"] = make_json_serializable(
            self.database.get_statistics(improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD)
        )

        initial_result = SerializableResult(
            child_program_dict=initial_program.to_dict(), iteration=self._num_search_evolutions
        )
        self._best_search_score = search_score
        await self.search_controller.postprocess_result(
            initial_result, self._num_search_evolutions, verbose=False
        )

        self.search_controller.database.initial_program_id = initial_program.id
        self.search_controller.database.initial_program_score = search_score
        self._num_search_evolutions += 1

        self._reset_search_window()
        await self._generate_and_validate_search_algorithm(solution_iter)

    async def _generate_variation_operators(self) -> None:
        """Generate diverge/refine labels once and assign to the current database."""
        if self._diverge_label and self._refine_label:
            self._assign_labels_to_db(self.database)
            return

        db_cfg = self.config.search.database
        if not getattr(db_cfg, "auto_generate_variation_operators", True):
            from skydiscover.search.evox.utils.template import (
                DEFAULT_DIVERGE_TEMPLATE,
                DEFAULT_REFINE_TEMPLATE,
            )

            self._diverge_label = DEFAULT_DIVERGE_TEMPLATE
            self._refine_label = DEFAULT_REFINE_TEMPLATE
            logger.debug(
                "Using default variation operators (auto_generate_variation_operators=false)"
            )
            self._assign_labels_to_db(self.database)
            return

        if not self._guide_llm_available:
            logger.info("Skipping label generation (guide LLM unavailable at startup)")
            self._assign_labels_to_db(self.database)
            return

        system_message = self.config.context_builder.system_message or ""
        from skydiscover.search.utils.discovery_utils import load_evaluator_code

        evaluator_code = load_evaluator_code(self.evaluation_file)

        try:
            problem_dir = Path(self.evaluation_file).parent if self.evaluation_file else None
            label_llms = self.search_controller.guide_llms
            model_names = ", ".join(m.name for m in label_llms.models_cfg)
            logger.debug(f"Label generation: using guide_model = [{model_names}]")
            self._diverge_label, self._refine_label = await generate_variation_operators(
                system_message,
                evaluator_code,
                problem_dir=problem_dir,
                llm_pool=label_llms,
            )
            logger.debug(
                f"Generated variation operator labels ({len(self._diverge_label)}/{len(self._refine_label)} chars)"
            )
        except Exception as e:
            self._diverge_label = ""
            self._refine_label = ""
            logger.warning(
                "Guide LLM unreachable during label generation (%s); using "
                "default variation operators. Meta-search evolution is "
                "unaffected.",
                e,
            )

        self._assign_labels_to_db(self.database)

    def _assign_labels_to_db(self, db) -> None:
        """Assign the variation operators to a database instance."""
        db.DIVERGE_LABEL = self._diverge_label
        db.REFINE_LABEL = self._refine_label

    async def _generate_and_validate_search_algorithm(self, solution_iter: int) -> None:
        """Generate a new search algorithm and switch to it if valid."""
        iteration = self._num_search_evolutions
        search_stats = self._build_search_stats(solution_iter)

        self.search_controller._prompt_context = {
            "search_stats": search_stats["search_algorithm_stats"],
            "db_stats": search_stats["db_stats"],
        }
        result = await self.search_controller.run_discovery(
            start_iteration=iteration,
            max_iterations=1,
            post_process_result=False,
        )

        if not result or result.error:
            failed_result = result or SerializableResult(
                error="Search algorithm generation returned no result", iteration=iteration
            )
            await self.search_controller.postprocess_result(failed_result, iteration, verbose=False)
            await handle_generation_failure(
                self.search_outputs_dir,
                self._active_search_algorithm_code,
                iteration,
                result,
                solution_iter,
            )
            self._meta_evolution_failures = getattr(self, "_meta_evolution_failures", 0) + 1
            self._num_search_evolutions += 1
            return

        result.child_program_dict.setdefault("metadata", {})["start_db_stats"] = (
            make_json_serializable(search_stats["db_stats"])
        )
        await log_search_algorithm_generated(
            self.search_outputs_dir,
            result,
            iteration,
            diverge_label=self._diverge_label,
            refine_label=self._refine_label,
        )

        if not self._switch_to_new_search_algorithm(result):
            await self.search_controller.postprocess_result(
                SerializableResult(error="Search algorithm validation failed", iteration=iteration),
                iteration,
                verbose=False,
            )
            await handle_generation_failure(
                self.search_outputs_dir,
                self._active_search_algorithm_code,
                iteration,
                result,
                solution_iter,
                "validation",
            )
            self._meta_evolution_failures = getattr(self, "_meta_evolution_failures", 0) + 1
            self._num_search_evolutions += 1
            return

        self._pending_search_result = result
        self._reset_search_window(start_iteration=solution_iter)

    def _build_search_stats(self, solution_iter: int) -> Dict[str, Any]:
        """Build statistics dict for search algorithm generation."""
        return {
            "search_algorithm_stats": {
                "window_start_iteration": solution_iter,
                "total_iterations": self._max_solution_iterations,
                "search_window_horizon": self._switch_interval,
                "problem_description": self.config.context_builder.system_message,
                "evaluator_context": self.evaluation_file,
                "improvement_threshold": self.DEFAULT_IMPROVEMENT_THRESHOLD,
            },
            "db_stats": self.database.get_statistics(
                improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD
            ),
        }

    def _switch_to_new_search_algorithm(self, result: SerializableResult) -> bool:
        """Switch solution database to use the new search algorithm."""
        child_dict = result.child_program_dict or {}
        search_code = child_dict.get("solution")
        if not search_code:
            logger.warning("No solution in search result; skipping transition")
            return False

        search_program_id = child_dict.get("id", "unknown")
        try:
            new_db, migrated_count = self._database_from_search_code(search_code, self.database)

            self._fallback_database = self.database
            self._fallback_search_code = self._active_search_algorithm_code

            self.database = new_db
            if self.evaluator.llm_judge:
                self.evaluator.llm_judge.database = new_db
            logger.debug(
                f"Switched to search algorithm {search_program_id} ({migrated_count} programs migrated)"
            )

            self._active_search_algorithm_code = search_code
            return True

        except Exception as e:
            logger.error(f"Failed to load search algorithm {search_program_id}: {e}")
            return False

    def _restore_fallback_database(self) -> None:
        """Restore the previous search strategy after a failed switch."""
        if self._pending_search_result:
            self.search_controller._process_iteration_result(
                SerializableResult(
                    error="Search algorithm rejected after database failure",
                    iteration=self._num_search_evolutions,
                ),
                self._num_search_evolutions,
                verbose=False,
            )
        broken_db = self.database
        old_db = self._fallback_database

        # Migrate new programs found during the broken strategy's successful runs
        old_ids = set(old_db.programs)
        migrated = 0
        for pid, program in broken_db.programs.items():
            if pid not in old_ids:
                try:
                    program_class = getattr(old_db, "_program_class", Program)
                    from_dict = getattr(program_class, "from_dict", None)
                    row = program.to_dict()
                    converted = from_dict(row) if callable(from_dict) else program_class(**row)
                    old_db.add(converted, iteration=converted.iteration_found)
                    migrated += 1
                except Exception:
                    logger.debug("Migration failed for program %s", program.id, exc_info=True)

        logger.warning(
            "New search strategy caused database error — "
            f"restoring previous search strategy ({migrated} new programs preserved)"
        )
        self.database = old_db
        if self.evaluator.llm_judge:
            self.evaluator.llm_judge.database = old_db
        self._active_search_algorithm_code = self._fallback_search_code
        self._pending_search_result = None
        self._num_search_evolutions += 1  # Count the failed attempt
        self._fallback_database = None
        self._fallback_search_code = None

    def _database_from_search_code(
        self, search_code: str, source_database, *, source_state: Optional[Dict[str, Any]] = None
    ):
        """Instantiate a saved search algorithm and replay the solution population."""
        if not isinstance(search_code, str) or not search_code.strip():
            raise ValueError("EvoX checkpoint has no active search algorithm code")
        fd, file_path = tempfile.mkstemp(suffix=".py", prefix="evox_search_")
        try:
            with os.fdopen(fd, "w") as file:
                file.write(search_code)
            database_class, program_class = load_database_from_file(file_path)
            for label in ("DIVERGE_LABEL", "REFINE_LABEL"):
                if not hasattr(database_class, label):
                    setattr(database_class, label, "")
            database = database_class(self.config.search.type, self.config.search.database)
            database._program_class = program_class
            self._assign_labels_to_db(database)
            migrated = self._migrate_to_db(database, source_database, source_state=source_state)
            self._wrap_add_method(database)
            database.get_best_program()
            return database, migrated
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def _migrate_to_db(
        self,
        new_db,
        source_database=None,
        *,
        source_state: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Migrate all programs and prompts from one database to another."""
        source = source_database or self.database
        prog_class = getattr(new_db, "_program_class", None)
        from_dict = getattr(prog_class, "from_dict", None)
        rows = source_state.get("programs") if source_state is not None else None
        programs = (
            [row for row in rows if isinstance(row, dict)]
            if rows is not None
            else [program.to_dict() for program in source.programs.values()]
        )
        programs.sort(key=lambda row: int(row.get("iteration_found") or 0))
        for row in programs:
            if callable(from_dict):
                converted = from_dict(row)
            else:
                converted = prog_class(**row) if prog_class else Program.from_dict(row)
            new_db.add(converted, iteration=converted.iteration_found)
        migrated = len(programs)

        state = source_state or {}
        new_db.last_iteration = max(
            new_db.last_iteration,
            source.last_iteration,
            int(state.get("last_iteration") or 0),
        )
        new_db.initial_program_id = state.get("initial_program_id", source.initial_program_id)
        new_db.initial_program_score = state.get(
            "initial_program_score", source.initial_program_score
        )
        language = state.get("language", getattr(source, "language", None))
        if language is not None:
            new_db.language = language
        best_program_id = state.get("best_program_id", source.best_program_id)
        if best_program_id in new_db.programs:
            new_db.best_program_id = best_program_id

        # Migrate prompts
        if source.config.log_prompts:
            if new_db.prompts_by_program is None:
                new_db.prompts_by_program = {}

            old_prompts = state.get("prompts_by_program", source.prompts_by_program) or {}
            new_db.prompts_by_program.update(
                {k: v for k, v in old_prompts.items() if k not in new_db.prompts_by_program}
            )

            for p in new_db.programs.values():
                if p.prompts and p.id not in new_db.prompts_by_program:
                    new_db.prompts_by_program[p.id] = p.prompts

        return migrated

    def _get_best_score(self) -> float:
        """Get the current best solution score (proxy / combined_score)."""

        best = self.database.get_best_program()

        if best and best.metrics:
            if getattr(self.database, "is_multiobjective_enabled", lambda: False)():
                return float(self.database._proxy_score(best))
            score = best.metrics.get("combined_score")
            return float(score) if isinstance(score, (int, float)) else 0.0
        return getattr(self.database, "initial_program_score", None) or 0.0

    def _reset_search_window(self, start_iteration: Optional[int] = None) -> None:
        """Start a fresh scoring window for the active search algorithm."""
        self.search_scorer.reset_window(self._get_best_score(), start_iteration=start_iteration)

    def _record_search_window_step(self) -> None:
        """Record current best score for search algorithm scoring."""

        if self.search_scorer.get_start_score() is None:
            self._reset_search_window()

        self.search_scorer.record_step(self._get_best_score())

    def _compute_search_metrics(
        self,
        start_score: Optional[float] = None,
        best_scores: Optional[List[float]] = None,
        horizon: Optional[int] = None,
        start_iteration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compute scoring metrics using the configured scorer."""

        return self.search_scorer.compute_metrics(
            start_score=start_score,
            best_scores=best_scores,
            horizon=horizon,
            total_iterations=self._max_solution_iterations,
            start_iteration=start_iteration,
        )

    def _wrap_add_method(self, db) -> None:
        """Wrap database.add() to ensure _update_best_program is always called."""
        original_add = db.add

        def wrapped_add(program, iteration=None, **kwargs):
            result = original_add(program, iteration=iteration, **kwargs)
            db._update_best_program(program)  # Idempotent safety for LLM-generated databases
            return result

        db.add = wrapped_add

    def _assign_search_score(self) -> bool:
        """Assign score to pending search algorithm. Returns True if new best."""
        if not self._pending_search_result:
            return False

        if self.search_scorer.get_window_size() > 0:
            metrics = self._compute_search_metrics(horizon=self._switch_interval)
        else:
            start = self.search_scorer.get_start_score() or 0.0
            metrics = self._compute_search_metrics(
                start_score=start,
                best_scores=[self._get_best_score()],
                horizon=self._switch_interval,
            )

        score = float(metrics.get("combined_score", 0.0) or 0.0)

        child_dict = self._pending_search_result.child_program_dict or {}
        child_dict.setdefault("metrics", {}).update(metrics)
        child_dict.setdefault("metadata", {})["end_db_stats"] = make_json_serializable(
            self.database.get_statistics(improvement_threshold=self.DEFAULT_IMPROVEMENT_THRESHOLD)
        )
        self._pending_search_result.child_program_dict = child_dict

        is_new_best = self._best_search_score is not None and score > self._best_search_score
        if is_new_best:
            logger.debug(
                f"New best search score: {score:.6f} (+{score - self._best_search_score:.6f})"
            )
        if is_new_best or self._best_search_score is None:
            self._best_search_score = score
        return is_new_best

    def _log_coevolution_setup(self, db_cfg) -> None:
        logger.debug("=" * 70)
        logger.debug("[EVOX CO-EVOLUTION SETUP]")
        logger.debug("-" * 70)
        logger.debug(f"  [SOLUTION EVOLUTION]")
        logger.debug(f"    Initial search strategy file : {db_cfg.database_file_path}")
        logger.debug(f"    Solution database class      : {self.database.__class__.__name__}")
        logger.debug(f"  [META EVOLUTION OF SEARCH STRATEGY]")
        logger.debug(
            f"    Search strategy database class: {self.search_controller.database.__class__.__name__}"
        )
        logger.debug(f"    Search strategy evaluator     : {db_cfg.evaluation_file}")
        logger.debug(f"    Search strategy config        : {db_cfg.config_path}")
        logger.debug("=" * 70)
