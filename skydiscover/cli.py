"""Command-line interface for SkyDiscover."""

import argparse
import asyncio
import json
import logging
import multiprocessing
import os
import sys
import traceback
from typing import Optional

from dotenv import find_dotenv, load_dotenv

from skydiscover import Runner
from skydiscover.benchmarks.resolution import resolve_benchmark_problem
from skydiscover.config import (
    _parse_model_spec,
    apply_dot_overrides,
    apply_overrides,
    is_azure_endpoint,
    load_config,
)
from skydiscover.runner import CHECKPOINT_COMPLETE_FILE

try:
    multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass

logger = logging.getLogger(__name__)

_SEARCH_CHOICES = [
    "evox",
    "adaevolve",
    "best_of_n",
    "beam_search",
    "topk",
    "openevolve_native",
    "openevolve",
    "shinkaevolve",
    "gepa",
    "gepa_native",
    "claude_code",
    "claude_code_local",
]


def _parse_dot_overrides(tokens: list) -> dict:
    """Parse leftover CLI tokens into dotted-path config overrides.

    Accepts ``--section.field value`` and ``--section.field=value`` forms
    (including non-dotted top-level keys like ``--checkpoint_interval``).
    """
    overrides: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            raise ValueError(f"unexpected argument: {tok!r}")
        key = tok[2:]
        if "=" in key:
            name, value = key.split("=", 1)
            overrides[name] = value
            i += 1
        else:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
                raise ValueError(f"missing value for override '--{key}'")
            overrides[key] = tokens[i + 1]
            i += 2
    return overrides


def parse_args() -> argparse.Namespace:
    """Build and parse the CLI argument parser.

    Dotted config overrides (e.g. ``--llm.temperature 0.2``) are collected from
    arguments not explicitly defined below and attached to the private
    ``_dot_overrides`` namespace attribute for ``main_async``.
    """
    parser = argparse.ArgumentParser(
        description="SkyDiscover - AI-Driven Scientific and Algorithmic Discovery",
    )

    parser.add_argument(
        "paths",
        nargs="+",
        metavar="path",
        help=(
            "Positional paths: either '<evaluation_file>' or "
            "'<initial_program> <evaluation_file>'"
        ),
    )
    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)
    parser.add_argument("--output", "-o", help="Output directory for results", default=None)
    parser.add_argument(
        "--iterations", "-i", type=int, default=None, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Logging level",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint directory to resume from",
    )
    parser.add_argument("--api-base", default=None, help="Base URL for the LLM API")
    parser.add_argument(
        "--agentic",
        action="store_true",
        default=False,
        help="Enable agentic mode (codebase root derived from initial program location)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="LLM model(s) for solution generation, comma-separated (e.g. 'gpt-5', 'gpt-5,gemini/gemini-3-pro')",
    )
    parser.add_argument(
        "--search",
        "-s",
        choices=_SEARCH_CHOICES,
        default=None,
        help="Search algorithm to use",
    )

    args, unknown = parser.parse_known_args()
    args._dot_overrides = _parse_dot_overrides(unknown)

    if len(args.paths) == 1:
        args.initial_program = None
        args.evaluation_file = args.paths[0]
    elif len(args.paths) == 2:
        args.initial_program, args.evaluation_file = args.paths
    else:
        parser.error("Expected either '<evaluation_file>' or '<initial_program> <evaluation_file>'")

    delattr(args, "paths")
    return args


def main() -> int:
    """Synchronous entry point for the skydiscover console script."""
    # Load environment variables from a .env file, searched from the working directory
    # and then from this package. Already-exported variables take precedence.
    load_dotenv(find_dotenv(usecwd=True) or find_dotenv())
    return asyncio.run(main_async())


async def main_async() -> int:
    """Async entry point for the CLI. Returns exit code."""
    try:
        args = parse_args()
        dot_overrides = args._dot_overrides
        delattr(args, "_dot_overrides")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _configure_logging(args.log_level)

    if args.initial_program and not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found", file=sys.stderr)
        return 1
    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found", file=sys.stderr)
        return 1

    has_overrides = any((args.api_base, args.model, args.agentic, args.search, dot_overrides))
    config = None
    evaluator_env_vars: Optional[dict[str, str]] = None

    # Load the configuration
    if args.config or has_overrides:
        config = load_config(args.config)

        evaluator_env_vars = None

        try:
            apply_overrides(
                config,
                model=args.model,
                api_base=args.api_base,
                agentic=args.agentic,
                search=args.search,
            )
            apply_dot_overrides(config, dot_overrides)
            config.validate_generation_mode()
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        # Resolve benchmark problem if configured and no initial_program provided
        if args.initial_program is None and config.benchmark and config.benchmark.enabled:
            try:
                resolution = resolve_benchmark_problem(config.benchmark)
                args.initial_program = resolution.initial_program_path
                args.evaluation_file = resolution.evaluator_path
                evaluator_env_vars = resolution.evaluator_env_vars
                print(
                    f"[Benchmark Loader] Benchmark: {config.benchmark.name}, Initial program: {args.initial_program}, Evaluator: {args.evaluation_file}"
                )
            except Exception as exc:
                print(f"Error: Failed to load benchmark problem: {exc}", file=sys.stderr)
                traceback.print_exc()
                return 1

        if args.model:
            print("Active models:")
            for i, m in enumerate(config.llm.models):
                provider, *_ = _parse_model_spec(m.name, m.api_base)
                # The bare deployment name carries no provider, so trust the endpoint.
                if is_azure_endpoint(m.api_base):
                    provider = "azure"
                print(f"  {i + 1}. {m.name} (provider: {provider}, weight: {m.weight})")
        if args.api_base:
            print(f"Using API base: {config.llm.api_base}")
        if args.agentic:
            if not config.agentic.codebase_root and args.initial_program:
                config.agentic.codebase_root = os.path.dirname(
                    os.path.abspath(args.initial_program)
                )
            print(f"Agentic mode enabled (codebase: {config.agentic.codebase_root})")
        if args.search:
            print(f"Using search algorithm: {args.search}")

    # Run the discovery
    try:
        search_type = config.search.type if config and hasattr(config, "search") else None

        if search_type:
            from skydiscover.extras.external import (
                KNOWN_EXTERNAL,
                get_package_name,
                get_runner,
                is_external,
            )

            # External backends (openevolve, shinkaevolve, gepa)
            if is_external(search_type):
                if evaluator_env_vars:
                    env_var_names = ", ".join(sorted(evaluator_env_vars))
                    print(
                        "Error: Passing evaluator environment variables to external backends "
                        "is not yet supported. "
                        f"External backend '{search_type}' cannot be used with evaluator env vars: "
                        f"{env_var_names}",
                        file=sys.stderr,
                    )
                    return 1

                from skydiscover.config import build_output_dir

                output_dir = args.output or build_output_dir(
                    search_type, args.initial_program or "scratch"
                )
                os.makedirs(output_dir, exist_ok=True)

                from skydiscover.extras.monitor import start_monitor, stop_monitor

                # Start monitor for external backends as well
                monitor_server, monitor_callback, feedback_reader = start_monitor(
                    config, output_dir
                )
                try:
                    result = await get_runner(search_type)(
                        program_path=args.initial_program,
                        evaluator_path=args.evaluation_file,
                        config_obj=config,
                        iterations=args.iterations or config.max_iterations,
                        output_dir=output_dir,
                        monitor_callback=monitor_callback,
                        feedback_reader=feedback_reader,
                    )
                except ModuleNotFoundError as exc:
                    pkg = get_package_name(search_type)
                    print(f"Error: {exc}", file=sys.stderr)
                    print(f"\nThe '{search_type}' backend requires its package.", file=sys.stderr)
                    print(f"Install with:  pip install {pkg}", file=sys.stderr)
                    return 1
                finally:
                    stop_monitor(monitor_server)

                from skydiscover.evaluation.final_eval import run_final_evaluation
                from skydiscover.utils.metrics import format_metrics

                test_metrics = await run_final_evaluation(
                    result.best_solution,
                    config,
                    output_dir,
                    evaluator_path=args.evaluation_file,
                    program_path=args.initial_program,
                )
                result.metrics.update(test_metrics)

                print(f"\nDiscovery complete! Best score: {result.best_score:.4f}")
                if test_metrics:
                    print(f"Final evaluation: {format_metrics(test_metrics)}")
                return 0

            if search_type in KNOWN_EXTERNAL:
                pkg = get_package_name(search_type)
                print(
                    f"Error: Search type '{search_type}' requires the '{pkg}' package. "
                    f"Install with: pip install {pkg}",
                    file=sys.stderr,
                )
                return 1

        # Initialize the runner
        runner = Runner(
            initial_program_path=args.initial_program,
            evaluation_file=args.evaluation_file,
            config=config,
            config_path=args.config if config is None else None,
            output_dir=args.output,
            evaluator_env_vars=evaluator_env_vars,
        )

        # Load the checkpoint if provided
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found", file=sys.stderr)
                return 1
            print(f"Will resume from checkpoint: {args.checkpoint}")

        # Run the discovery
        best_program = await runner.run(
            iterations=args.iterations,
            checkpoint_path=args.checkpoint,
        )

        checkpoint_dir = os.path.join(runner.output_dir, "checkpoints")
        latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)

        print("\nDiscovery complete!")
        if best_program is None:
            print("No valid programs were found.")
        else:
            print("Best program metrics:")
            for name, value in best_program.metrics.items():
                formatted = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                print(f"  {name}: {formatted}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint: {latest_checkpoint}")
            print(f"To resume: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


def _configure_logging(level_name: Optional[str]) -> None:
    """Set up the root logger with the SkyDiscover console format."""
    from skydiscover.search.utils.logging_utils import _ConsoleFilter, _ConsoleFormatter

    log_level = getattr(logging, level_name) if level_name else logging.WARNING
    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_ConsoleFormatter())
        handler.addFilter(_ConsoleFilter())
        root.addHandler(handler)
    logging.getLogger("skydiscover").setLevel(logging.INFO)


def _find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path of the latest checkpoint directory named like ``checkpoint_<n>``."""
    if not os.path.isdir(checkpoint_dir):
        return None

    def parse_iteration(name: str) -> Optional[int]:
        try:
            return int(name.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return None

    complete = []
    legacy = []
    for name in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(full_path):
            continue
        iteration = parse_iteration(name)
        if iteration is None:
            continue
        marker = os.path.join(full_path, CHECKPOINT_COMPLETE_FILE)
        if not os.path.exists(marker):
            legacy.append((iteration, full_path))
            continue
        try:
            with open(marker) as file:
                payload = json.load(file)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("iteration") == iteration:
            complete.append((iteration, full_path))

    candidates = complete or legacy
    if not candidates:
        return None

    return max(candidates, key=lambda item: item[0])[1]


if __name__ == "__main__":
    sys.exit(main())
