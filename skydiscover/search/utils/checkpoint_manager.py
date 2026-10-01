"""
Checkpoint management for program databases.

Handles saving and loading database state to/from disk.
"""

import json
import logging
import os
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)

EVOLUTION_TRACE_STATE_FILE = "evolution_trace_state.json"


class SafeJSONEncoder(json.JSONEncoder):
    """
    JSON encoder that handles non-serializable types gracefully.

    This is important for evolved databases where LLM-generated code may
    store non-serializable types (like sets) in program metadata.
    """

    def default(self, obj):
        # Convert numpy arrays/scalars to Python types
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
        except ImportError:
            pass
        # Convert sets to sorted lists for consistency
        if isinstance(obj, set):
            return sorted(list(obj))
        # Convert frozensets to sorted lists
        if isinstance(obj, frozenset):
            return sorted(list(obj))
        # Let the base class raise TypeError for other non-serializable types
        return super().default(obj)


class CheckpointManager:
    """
    Manages database checkpointing (save/load operations).
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config

    def save(
        self,
        programs: Dict[str, Program],
        prompts_by_program: Optional[Dict[str, Dict[str, Dict[str, str]]]],
        best_program_id: Optional[str],
        last_iteration: int,
        path: Optional[str] = None,
        write_evolution_trace: bool = False,
        evolution_archive: Optional[Dict[str, Program]] = None,
        unattached_web_search_results: Optional[Dict[str, Any]] = None,
        unattached_llm_reasoning: Optional[Dict[str, Any]] = None,
        solution_confidence_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Save the database to disk

        Args:
            programs: Dictionary of program ID to Program
            prompts_by_program: Optional prompts by program ID
            best_program_id: ID of the best program
            last_iteration: Last iteration number
            path: Path to save to (uses config.db_path if None)
            write_evolution_trace: Also write evolution_trace.json at the save path.
            evolution_archive: Programs evicted from a bounded live population.
            unattached_web_search_results: Calls made before a Program could exist.
            solution_confidence_records: Assessments for all attempts, including failures.
        """
        save_path = path or self.config.db_path
        if not save_path:
            logger.warning("No database path specified, skipping save")
            return

        # Checkpoint and trace must describe the same snapshot of mutable attempt records.
        confidence_records = deepcopy(solution_confidence_records or [])

        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save each program
        for program in programs.values():
            prompts = None
            if self.config.log_prompts and prompts_by_program and program.id in prompts_by_program:
                prompts = prompts_by_program[program.id]
            self._save_program(program, save_path, prompts=prompts)

        # Save metadata
        metadata = {
            "best_program_id": best_program_id,
            "last_iteration": last_iteration,
        }

        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

        self._save_evolution_trace_state(
            save_path,
            evolution_archive or {},
            unattached_web_search_results or {},
            unattached_llm_reasoning or {},
            solution_confidence_records=confidence_records,
            prompts_by_program=prompts_by_program,
        )

        if write_evolution_trace:
            self._write_evolution_trace(
                programs={**(evolution_archive or {}), **programs},
                best_program_id=best_program_id,
                last_iteration=last_iteration,
                save_path=save_path,
                prompts_by_program=prompts_by_program,
                unattached_web_search_results=unattached_web_search_results,
                unattached_llm_reasoning=unattached_llm_reasoning,
                solution_confidence_records=confidence_records,
            )

        logger.debug(f"Saved database with {len(programs)} programs to {save_path}")

    def load(self, path: str) -> Tuple[Dict[str, Program], Optional[str], int]:
        """
        Load the database from disk

        Args:
            path: Path to load from

        Returns:
            Tuple of (programs_dict, best_program_id, last_iteration)
        """
        # Import here to avoid circular import
        from skydiscover.search.base_database import Program

        programs: Dict[str, Program] = {}
        best_program_id: Optional[str] = None
        last_iteration: int = 0

        if not os.path.exists(path):
            logger.warning(f"Database path {path} does not exist, skipping load")
            return programs, best_program_id, last_iteration

        # Load metadata first
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            best_program_id = metadata.get("best_program_id")
            last_iteration = metadata.get("last_iteration", 0)

            logger.debug(f"Loaded database metadata with last_iteration={last_iteration}")

        # Load programs
        programs_dir = os.path.join(path, "programs")
        if os.path.exists(programs_dir):
            for program_file in os.listdir(programs_dir):
                if program_file.endswith(".json"):
                    program_path = os.path.join(programs_dir, program_file)
                    try:
                        with open(program_path, "r") as f:
                            program_data = json.load(f)

                        program = Program.from_dict(program_data)
                        programs[program.id] = program
                    except Exception as e:
                        logger.warning(f"Error loading program {program_file}: {str(e)}")

        logger.debug(f"Loaded database with {len(programs)} programs from {path}")

        return programs, best_program_id, last_iteration

    def _save_evolution_trace_state(
        self,
        save_path: str,
        evolution_archive: Dict[str, Program],
        unattached_web_search_results: Dict[str, Any],
        unattached_llm_reasoning: Dict[str, Any],
        solution_confidence_records: Optional[List[Dict[str, Any]]] = None,
        prompts_by_program: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
    ) -> None:
        """Persist non-population trace state so checkpoint resume is lossless."""
        archived_programs = []
        for program in evolution_archive.values():
            entry = program.to_dict()
            # Prompt logging keeps a separate map until checkpoint serialization.
            # Use the same fallback as the trace, without changing the archived Program.
            if self.config.log_prompts and not entry.get("prompts") and prompts_by_program:
                prompts = prompts_by_program.get(program.id)
                if prompts:
                    entry["prompts"] = deepcopy(prompts)
            archived_programs.append(entry)
        state = {
            "archived_programs": archived_programs,
            "unattached_web_search_results": unattached_web_search_results,
            "unattached_llm_reasoning": unattached_llm_reasoning,
        }
        if solution_confidence_records:
            state["solution_confidence_records"] = deepcopy(solution_confidence_records)
        path = os.path.join(save_path, EVOLUTION_TRACE_STATE_FILE)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)

    def load_evolution_trace_state(
        self, path: str
    ) -> Tuple[Dict[str, Program], Dict[str, Any], Dict[str, Any]]:
        """Load optional trace-only state from a checkpoint."""
        state_path = os.path.join(path, EVOLUTION_TRACE_STATE_FILE)
        try:
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            return {}, {}, {}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not load evolution trace state %s: %s", state_path, exc)
            return {}, {}, {}

        archived: Dict[str, Program] = {}
        records = state.get("archived_programs") if isinstance(state, dict) else []
        for item in records if isinstance(records, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                program = Program.from_dict(item)
            except (TypeError, ValueError) as exc:
                logger.warning("Could not restore archived Program: %s", exc)
                continue
            archived[program.id] = program

        unattached = (
            state.get("unattached_web_search_results", {}) if isinstance(state, dict) else {}
        )
        unattached_reasoning = (
            state.get("unattached_llm_reasoning", {}) if isinstance(state, dict) else {}
        )
        return (
            archived,
            unattached if isinstance(unattached, dict) else {},
            unattached_reasoning if isinstance(unattached_reasoning, dict) else {},
        )

    def load_solution_confidence_records(self, path: str) -> List[Dict[str, Any]]:
        """Restore attempt records without changing the existing trace-state tuple API."""
        state_path = os.path.join(path, EVOLUTION_TRACE_STATE_FILE)
        try:
            with open(state_path, encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as exc:
            logger.warning("Could not load solution confidence records %s: %s", state_path, exc)
            return []

        records = state.get("solution_confidence_records", []) if isinstance(state, dict) else []
        if not isinstance(records, list):
            logger.warning("Ignoring invalid solution confidence records in %s", state_path)
            return []
        return [record for record in records if isinstance(record, dict)]

    def _write_evolution_trace(
        self,
        programs: Dict[str, Program],
        best_program_id: Optional[str],
        last_iteration: int,
        save_path: str,
        prompts_by_program: Optional[Dict[str, Dict[str, Dict[str, str]]]] = None,
        unattached_web_search_results: Optional[Dict[str, Any]] = None,
        unattached_llm_reasoning: Optional[Dict[str, Any]] = None,
        solution_confidence_records: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Write all Program records, including embedded Tavily provenance."""
        from skydiscover.evaluation.evaluation_result import evaluation_failure_reason
        from skydiscover.utils.metrics import get_score

        def has_failed_evaluation(program: Program) -> bool:
            return (program.metadata or {}).get("evaluation_status") == "failed" or (
                evaluation_failure_reason(program.metrics, program.artifacts) is not None
            )

        confidence_records = deepcopy(solution_confidence_records or [])
        entries = []
        for program in programs.values():
            entry = program.to_dict()
            metadata = program.metadata or {}
            evaluation_failed = has_failed_evaluation(program)
            entry["score"] = (
                get_score(program.metrics) if program.metrics and not evaluation_failed else None
            )

            prompts = program.prompts
            if not prompts and prompts_by_program and program.id in prompts_by_program:
                prompts = prompts_by_program[program.id]
            if prompts:
                entry["prompts"] = prompts

            parent = programs.get(program.parent_id)
            if not evaluation_failed and parent is not None and not has_failed_evaluation(parent):
                parent_metrics = parent.metrics or {}
                delta: Dict[str, float] = {}
                for key, value in (program.metrics or {}).items():
                    parent_value = parent_metrics.get(key)
                    if (
                        isinstance(value, (int, float))
                        and isinstance(parent_value, (int, float))
                        and not isinstance(value, bool)
                        and not isinstance(parent_value, bool)
                    ):
                        delta[key] = value - parent_value
                if delta:
                    entry["improvement_delta"] = delta

            if "island" in metadata:
                entry.setdefault("island_id", metadata["island"])

            if prompts:
                for prompt_data in prompts.values():
                    if not isinstance(prompt_data, dict):
                        continue
                    responses = prompt_data.get("responses")
                    if responses and not entry.get("llm_response"):
                        entry["llm_response"] = responses[0]
            entries.append(entry)

        entries.sort(
            key=lambda entry: (
                entry.get("iteration_found", 0),
                entry.get("timestamp", 0.0),
            )
        )
        trace: Dict[str, Any] = {
            "schema_version": 2,
            "last_iteration": last_iteration,
            "best_program_id": best_program_id,
            "total_programs": len(entries),
            "programs": entries,
        }
        if isinstance(unattached_web_search_results, dict) and unattached_web_search_results.get(
            "tavily_searches"
        ):
            trace["unattached_web_search_results"] = unattached_web_search_results
        if isinstance(unattached_llm_reasoning, dict) and unattached_llm_reasoning.get("calls"):
            trace["unattached_llm_reasoning"] = unattached_llm_reasoning
        if confidence_records:
            trace["solution_confidence_records"] = confidence_records

        os.makedirs(save_path, exist_ok=True)
        trace_path = os.path.join(save_path, "evolution_trace.json")
        temporary_path = f"{trace_path}.tmp-{uuid.uuid4().hex}"
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(trace, handle, indent=2, ensure_ascii=False, cls=SafeJSONEncoder)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, trace_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a program to disk

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program, in the format {template_key: { 'system': str, 'user': str }}
        """
        save_path = base_path or self.config.db_path
        if not save_path:
            return

        # Create programs directory if it doesn't exist
        programs_dir = os.path.join(save_path, "programs")
        os.makedirs(programs_dir, exist_ok=True)

        # Save program
        program_dict = program.to_dict()
        if prompts:
            program_dict["prompts"] = prompts
        program_path = os.path.join(programs_dir, f"{program.id}.json")

        with open(program_path, "w") as f:
            json.dump(program_dict, f, cls=SafeJSONEncoder)
