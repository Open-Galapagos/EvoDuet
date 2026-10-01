"""One task-directory launcher for EvoDuet and its no-retrieval baseline."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from skydiscover import cli
from skydiscover.config import _SECTION_ALIASES, apply_dot_overrides, apply_overrides, load_config
from skydiscover.evoduet.selective_generation import selective_generation_scope
from skydiscover.runner import (
    CHECKPOINT_COMPLETE_FILE,
    CHECKPOINT_SCHEMA_VERSION,
    EVODUET_STATE_FILE,
)

# Shared settings from the original default / default_ours_v7 OpenEvolve runs.
# Explicit dotted flags below can override these experiment defaults.
EXPERIMENT_DEFAULTS = {
    "llm.temperature": "0.7",
    "llm.top_p": "0.95",
    "llm.max_tokens": "32768",
    "llm.reasoning_effort": "medium",
    "llm.timeout": "1800",
    "llm.retries": "3",
    "llm.tools": "",
    "checkpoint_interval": "1",
    "solution_confidence.enabled": "false",
    "evaluator.cascade_evaluation": "false",
    "evaluator.inject_evaluator_context": "false",
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _checkpoint_iteration(checkpoint, *, evoduet_enabled):
    """Read a complete checkpoint without losing its saved search memory."""
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")

    def read_object(name):
        path = checkpoint / name
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read checkpoint file {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise TypeError(f"checkpoint file must contain an object: {path}")
        return value

    if evoduet_enabled:
        # Missing state must not silently turn a resumed experiment into a fresh
        # search-memory run. The public checkpoint format uses evoduet.json.
        read_object(EVODUET_STATE_FILE)
    metadata = read_object("metadata.json")
    marker = read_object(CHECKPOINT_COMPLETE_FILE)
    iteration = metadata.get("last_iteration")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("checkpoint metadata has an invalid last_iteration")
    if (
        type(marker.get("schema_version")) is not int
        or marker["schema_version"] != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported checkpoint format; start a new run")
    if type(marker.get("iteration")) is not int or marker["iteration"] != iteration:
        raise ValueError("checkpoint completion marker does not match its metadata")
    if checkpoint.name.startswith("checkpoint_") and checkpoint.name != f"checkpoint_{iteration}":
        raise ValueError("checkpoint directory name does not match its metadata")
    return iteration


def prepare_run(argv=None):
    """Resolve task files and validate settings without constructing API clients."""
    parser = argparse.ArgumentParser(
        description="EvoDuet: evolve programs with web-search evidence.",
        epilog="Advanced settings accept dotted flags, e.g. --llm.temperature 0.7.",
    )
    parser.add_argument("task", type=Path, help="Directory containing a benchmark task")
    parser.add_argument("--config", type=Path, help="Override the task's config.yaml")
    parser.add_argument("--initial-program", type=Path, help="Override initial_program.*")
    parser.add_argument("--evaluator", type=Path, help="Override the task's evaluator")
    parser.add_argument("--model", help="Model name, e.g. openrouter/author/model")
    parser.add_argument("--api-base", help="Optional OpenAI-compatible endpoint")
    parser.add_argument(
        "--search",
        default="openevolve_native",
        choices=["openevolve_native", "evox", "adaevolve", "topk", "best_of_n", "beam_search"],
    )
    parser.add_argument("--output", type=Path, help="Run output directory")
    parser.add_argument(
        "--checkpoint", type=Path, help="Checkpoint directory to resume (EvoDuet uses evoduet.json)"
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        help="Total iteration target, including a resumed checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edit", choices=["diff", "full"], default="diff")
    parser.add_argument(
        "--num-generations",
        type=positive_int,
        default=1,
        help="Candidates after retrieve/look-up; no-op uses one candidate",
    )
    parser.add_argument("--inner-rounds", type=positive_int, default=3)
    parser.add_argument("--query-count", type=positive_int, default=1)
    parser.add_argument("--documents-per-query", type=positive_int, default=5)
    parser.add_argument("--search-top-k", type=positive_int, default=3)
    parser.add_argument("--baseline", action="store_true", help="Disable retrieval")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print validated settings without API calls"
    )
    args, extra = parser.parse_known_args(argv)
    task = args.task.resolve()
    if not task.is_dir():
        parser.error(f"task directory does not exist: {task}")
    initial = args.initial_program
    if initial is None:
        candidates = sorted(p for p in task.glob("initial_program.*") if p.is_file())
        if len(candidates) != 1:
            parser.error(
                "task must contain one initial_program.*; use --initial-program to select it"
            )
        initial = candidates[0]
    evaluator = args.evaluator or next(
        (
            task / name
            for name in ("skydiscover_evaluator.py", "evaluator.py", "evaluator")
            if (task / name).exists()
        ),
        None,
    )
    config_path = args.config or task / "config.yaml"
    for label, path in (
        ("initial program", initial),
        ("evaluator", evaluator),
        ("config", config_path),
    ):
        if path is None or not (path.is_file() or (label == "evaluator" and path.is_dir())):
            parser.error(f"{label} file does not exist: {path}")
    initial, evaluator, config_path = (p.resolve() for p in (initial, evaluator, config_path))
    overrides = {
        **EXPERIMENT_DEFAULTS,
        "evoduet.enabled": str(not args.baseline).lower(),
        "evoduet.retrieval_gating_prompt_template_name": "retrieval_gating",
        "evoduet.knowledge_state_analysis_prompt_template_name": "knowledge_analysis",
        "evoduet.search_selection.policy": "recency",
        "evoduet.search_selection.num": "10",
        "evoduet.query_optimization_max_rounds": str(args.inner_rounds),
        "evoduet.query_optimization_queries_per_round": str(args.query_count),
        "evoduet.tavily_retrieval.max_results": str(args.documents_per_query),
        "evoduet.search_result_top_k": str(args.search_top_k),
        "evoduet.random_seed": str(args.seed),
        "search.database.random_seed": str(args.seed),
        "diff_based_generation": str(args.edit == "diff").lower(),
        "num_generations": str(args.num_generations),
        "max_parallel_generations": str(args.num_generations),
        "max_parallel_evaluations": "1" if args.baseline else str(args.num_generations),
        "max_parallel_iterations": "1",
        "monitor.enabled": "false",
        "human_feedback_enabled": "false",
    }
    try:
        overrides.update(cli._parse_dot_overrides(extra))
        config = load_config(str(config_path))
        apply_overrides(config, model=args.model, api_base=args.api_base, search=args.search)
        for dotted in overrides:
            parts = dotted.split(".")
            parts[0] = _SECTION_ALIASES.get(parts[0], parts[0])
            target = config
            for part in parts:
                if not hasattr(target, part):
                    raise ValueError(f"unknown config field: --{dotted}")
                target = getattr(target, part)
        apply_dot_overrides(config, overrides)
        config.validate_generation_mode()
        args.iterations = args.iterations if args.iterations is not None else config.max_iterations
        args.resume_iteration = 0
        if args.checkpoint is not None:
            args.checkpoint = args.checkpoint.resolve()
            args.resume_iteration = _checkpoint_iteration(
                args.checkpoint, evoduet_enabled=config.evoduet.enabled
            )
            if args.resume_iteration > args.iterations:
                raise ValueError(
                    "checkpoint iteration exceeds the requested total iteration target"
                )
            if args.output is None and args.checkpoint.parent.name == "checkpoints":
                args.output = args.checkpoint.parent.parent
        args.remaining_iterations = args.iterations - args.resume_iteration
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    forwarded = [
        str(initial),
        str(evaluator),
        "--config",
        str(config_path),
        "--search",
        args.search,
    ]
    for flag in ("model", "api_base", "output", "checkpoint"):
        value = getattr(args, flag)
        if value is not None:
            forwarded.extend(["--" + flag.replace("_", "-"), str(value)])
    # The framework runner accepts additional iterations; the experiment CLI
    # accepts a total target, as the original experiment launchers did.
    forwarded.extend(["--iterations", str(args.remaining_iterations)])
    for key, value in overrides.items():
        forwarded.extend(["--" + key, value])
    return args, config, forwarded


def main(argv=None):
    load_dotenv(find_dotenv(usecwd=True) or find_dotenv())
    args, config, forwarded = prepare_run(argv)
    if args.dry_run:
        # Deliberately print only public settings, never resolved credentials.
        print(
            json.dumps(
                {
                    "method": "evoduet" if config.evoduet.enabled else "baseline",
                    "initial_program": forwarded[0],
                    "evaluator": forwarded[1],
                    "config": forwarded[3],
                    "search": config.search.type,
                    "models": [model.name for model in config.llm.models],
                    "iterations": args.iterations,
                    "resume_iteration": args.resume_iteration,
                    "remaining_iterations": args.remaining_iterations,
                    "output": str(args.output) if args.output else None,
                    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                    "checkpoint_state_file": EVODUET_STATE_FILE if config.evoduet.enabled else None,
                    "checkpoint_interval": config.checkpoint_interval,
                    "num_generations": config.num_generations,
                    "max_parallel_generations": config.max_parallel_generations,
                    "max_parallel_evaluations": config.max_parallel_evaluations,
                    "llm": {
                        name: getattr(config.llm, name)
                        for name in (
                            "temperature",
                            "top_p",
                            "max_tokens",
                            "reasoning_effort",
                            "timeout",
                            "retries",
                        )
                    },
                    "inner_rounds": config.evoduet.query_optimization_max_rounds,
                    "query_count": config.evoduet.query_optimization_queries_per_round,
                    "documents_per_query": config.evoduet.tavily_retrieval.max_results,
                    "search_top_k": config.evoduet.search_result_top_k,
                },
                indent=2,
            )
        )
        return 0
    original_argv = sys.argv
    routing = (
        selective_generation_scope()
        if config.evoduet.enabled and config.num_generations > 1
        else nullcontext()
    )
    try:
        sys.argv = ["evoduet-run", *forwarded]
        with routing:
            return cli.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
