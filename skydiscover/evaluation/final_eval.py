"""Post-search final (test-mode) evaluation for the external backends.

``Runner`` re-evaluates its best program with ``mode="test"`` on its own.  The
external backends (openevolve, shinkaevolve, gepa, alphaevolve) never build a
SkyDiscover evaluator, so they get the same stage here — driven from the two
places that consume their ``DiscoveryResult``.
"""

import json
import logging
import os
from dataclasses import replace
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

FINAL_EVALUATION_FILENAME = "final_evaluation.json"


def _file_suffix(config, program_path: Optional[str]) -> str:
    """Mirror Runner's rule: the seed program's extension wins over the default."""
    suffix = config.evaluator.file_suffix
    if suffix and suffix != ".py":
        return suffix
    if config.file_suffix and config.file_suffix != ".py":
        return config.file_suffix
    ext = os.path.splitext(program_path)[1] if program_path else ""
    return ext or ".py"


async def run_final_evaluation(
    best_solution: str,
    config,
    output_dir: Optional[str] = None,
    evaluator_path: Optional[str] = None,
    program_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Score ``best_solution`` in test mode and return its ``test_``-prefixed metrics.

    ``evaluator_path`` and ``program_path`` are what the external backends were
    invoked with.  Only the controllers populate ``config.evaluator`` from them,
    and the external backends never build one, so fill it in here.

    Returns an empty dict when the stage is disabled, has nothing to score, or
    fails — a finished search must never be lost to its final evaluation.
    """
    if not config.evaluator.final_evaluation:
        return {}

    evaluation_file = evaluator_path or config.evaluator.evaluation_file
    if not best_solution or not evaluation_file:
        return {}

    from skydiscover.evaluation import create_evaluator

    evaluator_config = replace(
        config.evaluator,
        evaluation_file=evaluation_file,
        file_suffix=_file_suffix(config, program_path),
        is_image_mode=config.language == "image",
    )

    evaluator = None
    try:
        evaluator = create_evaluator(evaluator_config)
        result = await evaluator.evaluate_program(best_solution, "best", mode="test")
    except Exception as e:
        logger.warning(f"Final test-mode evaluation failed: {e}")
        return {}
    finally:
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception:
                pass

    metrics = {f"test_{k}": v for k, v in result.metrics.items()}
    if output_dir:
        path = os.path.join(output_dir, FINAL_EVALUATION_FILENAME)
        try:
            with open(path, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
        except OSError as e:
            logger.warning(f"Could not write {path}: {e}")
    return metrics
