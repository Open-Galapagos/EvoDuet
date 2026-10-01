"""Population analysis and knowledge synthesis for fresh retrieval."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from skydiscover.evoduet.history import verbalize_history
from skydiscover.evoduet.population_stats import serialize_population_statistics

logger = logging.getLogger("skydiscover.evoduet")
_PROMPTS = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class KnowledgeContext:
    """The database text plus analyses derived from the current search state."""

    search_database: str
    population_analysis: str = ""
    search_database_analysis: str = ""
    knowledge_state_analysis: str = ""

    def __str__(self) -> str:
        return self.search_database


def _render(name, **fields):
    template = (_PROMPTS / f"{name.removesuffix('.txt')}.txt").read_text()
    return re.sub(r"\{(\w+)\}", lambda match: fields.get(match.group(1), match.group(0)), template)


class KnowledgeAnalyzer:
    """EvoX population report, plus a knowledge fallback for non-LLM gates."""

    def __init__(self, generate, config):
        self.generate = generate
        self.population_template_name = config.population_state_prompt_template_name
        self.knowledge_state_template_name = config.knowledge_state_analysis_prompt_template_name
        self.population_max_chars = config.population_state_max_chars
        self.population_recent_k = config.population_state_recent_k
        self.search_database_max_chars = config.search_database_analysis_max_chars
        self.analysis_max_chars = config.analysis_max_chars
        # LLM gating already produces the knowledge report from the three source
        # inputs. Heuristic gates retain the downstream synthesis they need for Q.
        self.analyze_knowledge_state = config.retrieval_gating_backend_type != "llm"

    async def _report(self, system, prompt, *, label, iteration):
        try:
            outcome = await self.generate(system, prompt, label=label, iteration=iteration)
            return str(outcome or "").strip()[: self.analysis_max_chars]
        except Exception as exc:
            # Preserve the standard layer's best-effort analysis behavior.
            logger.warning("EvoDuet %s failed: %s", label, exc)
            return ""

    async def analyze(
        self,
        population,
        search_database,
        iteration=None,
        task_context="",
        *,
        parent=None,
        history=None,
    ):
        database = "" if search_database is None else str(search_database)
        population_payload = serialize_population_statistics(
            population,
            self.population_max_chars,
            recent_k=self.population_recent_k,
            parent=parent,
        )
        # Exact EvoX instructions in system; deterministic statistics alone in
        # user. The caller invokes this only after the gate selects retrieve.
        population_report = (
            await self._report(
                _render(self.population_template_name),
                f"Population Statistics:\n\n{population_payload}",
                label="wk-population-analysis",
                iteration=iteration,
            )
            if population_payload
            else ""
        )
        if not self.analyze_knowledge_state:
            return KnowledgeContext(database, population_analysis=population_report)
        program = (
            parent.get("solution", "")
            if isinstance(parent, Mapping)
            else getattr(parent, "solution", "")
        )
        program = str(program or "")[: self.population_max_chars]
        # Substitute only explicit slots: no hidden task, history or DB appendices.
        # In the knowledge template, web evidence is not a direct input.
        knowledge_report = await self._report(
            "",
            _render(
                self.knowledge_state_template_name,
                parent_program=program,
                current_program=program,
                evolutionary_history=verbalize_history(history),
                population_state_analysis=population_report,
                population_analysis=population_report,
                search_database=database[: self.search_database_max_chars],
                search_database_analysis="",
                task_context=task_context,
            ),
            label="wk-knowledge-state-analysis",
            iteration=iteration,
        )
        return KnowledgeContext(
            search_database=database,
            population_analysis=population_report,
            search_database_analysis="",
            knowledge_state_analysis=knowledge_report,
        )
