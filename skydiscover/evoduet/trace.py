"""Iteration-level, human-readable trace of the EvoDuet layer.

Everything for one iteration lands in its own dir,
``<output_dir>/evoduet/checkpoint_<N>/``, alongside that iteration's trajectories
and search snapshot:

  - ``gate_decision.json`` — the gate decision, knowledge report, and requested IDs.
  - ``query_evolution.json`` — candidate queries, their documents, and final selection.
  - ``query_optimization.json`` — document forecasts and the evaluated evidence outcome.

Writes are best-effort — a failure never breaks the discovery loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("skydiscover.evoduet")


def _doc_summary(documents: Any) -> dict:
    """Provider IDs/titles and registered search-document IDs for retrieved evidence."""
    if not documents:
        return {"num_docs": 0, "doc_ids": [], "doc_titles": [], "search_document_ids": []}
    ids, titles, search_ids = [], [], []
    for d in documents:
        ids.append(getattr(d, "id", "") or getattr(d, "url", "") or "")
        titles.append(getattr(d, "title", "") or getattr(d, "url", "") or "")
        if identifier := getattr(d, "search_document_id", ""):
            search_ids.append(identifier)
    return {
        "num_docs": len(documents),
        "doc_ids": ids,
        "doc_titles": titles,
        "search_document_ids": search_ids,
    }


def make_step(step: int, kind: str, query: Any, documents: Any) -> dict:
    """One retrieval or selection step with its query and documents."""
    rec: dict = {"step": step, "kind": kind}
    if query is None or isinstance(query, str):
        rec["query"] = query
    else:
        rec["query"] = getattr(query, "text", None) or getattr(query, "query", None)
        keywords = getattr(query, "keywords", None)
        query_type = getattr(query, "query_type", None)
        if keywords:
            rec["keywords"] = keywords
        if query_type:
            rec["query_type"] = query_type
    rec.update(_doc_summary(documents))
    return rec


def record_gate_decision(
    output_dir, iteration, decision: str, *, details: dict | None = None
) -> None:
    """Write ``checkpoint_<N>/gate_decision.json`` for this iteration (best-effort)."""
    d = iteration_dir(output_dir, iteration)
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        payload = dict(details or {}, iteration=iteration, decision=decision)
        (d / "gate_decision.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        logger.exception("EvoDuet: failed to record gate decision")


def record_query_evolution(
    output_dir,
    iteration: Union[int, str],
    *,
    decision: str,
    mode: str,
    steps: List[dict],
) -> None:
    """Write query candidates and selected evidence for one iteration.

    ``iteration`` is the integer step, or a label such as ``"grounding"`` for the
    online-RAG pre-loop seed. Best-effort — never raises into the discovery loop.
    """
    d = iteration_dir(output_dir, iteration if isinstance(iteration, int) else None)
    if d is None:
        return
    try:
        d.mkdir(parents=True, exist_ok=True)
        final_query = next((s.get("query") for s in reversed(steps) if s.get("query")), None)
        committed = steps[-1] if steps else {}
        payload = {
            "iteration": iteration,
            "decision": decision,
            "mode": mode,
            "num_steps": len(steps),
            "steps": steps,
            "final_query": final_query,
            "committed_doc_ids": committed.get("doc_ids", []),
            "committed_search_document_ids": committed.get("search_document_ids", []),
        }
        (d / "query_evolution.json").write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        logger.exception("EvoDuet: failed to record query evolution")


def iteration_dir(output_dir, iteration) -> Optional[Path]:
    """``<output_dir>/evoduet/checkpoint_<iteration>`` — the per-iteration dir
    that holds its gate decision, query evolution, search snapshot, and trajectories.
    ``iteration=None`` keeps the historical ``checkpoint_grounding`` directory name.
    Returns None when there is no output dir."""
    if not output_dir:
        return None
    label = "grounding" if iteration is None else iteration
    return Path(output_dir) / "evoduet" / f"checkpoint_{label}"


def persist_query_trace(output_dir, trace: dict) -> None:
    """Write an auditable prediction trace after search, selection, and evaluation updates."""
    directory = iteration_dir(output_dir, trace.get("iteration"))
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "query_optimization.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.exception("EvoDuet: failed to persist query optimization")
