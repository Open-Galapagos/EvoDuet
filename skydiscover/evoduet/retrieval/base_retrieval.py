"""Retrieval records, measured search memory, and the retrieval backend base."""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

if TYPE_CHECKING:
    from skydiscover.evoduet.query_construction import ConstructedQuery

logger = logging.getLogger("skydiscover.evoduet")


@dataclass
class SearchResult:
    """A single retrieved document; the unit accumulated in ``D_search``."""

    raw_content: str  # full document content
    content: str  # A short description of the search result
    id: str  # stable id (e.g. url/DOI-based)
    title: str = ""
    url: str = ""
    source: str = ""  # backend that produced it ("tavily")
    published_date: Optional[str] = None  # the document's own publication date
    query: Optional[str] = None  # the actual query string sent to the backend
    keywords: Optional[List[str]] = None  # the search keywords/terms
    resources: Optional[List[str]] = None  # source kinds consulted (paper, github, blog, ...)
    query_type: Optional[str] = None  # the intent behind the query
    rank: Optional[int] = None  # local rank within the retrieval batch
    relevance_score: Optional[float] = None  # relevance score
    iteration: Optional[int] = None  # iteration at which it was retrieved
    favicon: Optional[str] = None  # source favicon URL (e.g. Tavily include_favicon)
    images: Optional[List[str]] = None  # related image URLs (e.g. Tavily include_images)
    metadata: Dict[str, Any] = field(default_factory=dict)  # extra / backend-specific fields

    retrieval_iteration: Optional[int] = None  # iteration the retrieval was activated
    # Usage history — filled after the child generated with this retrieval's
    # evidence is evaluated.
    visit_count: int = 0  # times this result was used in prompt construction
    evolution_iterations: List[int] = field(default_factory=list)  # iterations it was used at
    evolution_scores: List[float] = field(default_factory=list)  # child evaluator scores when used
    evolution_feedbacks: List[str] = field(default_factory=list)  # child feedbacks when used
    search_document_id: str = ""  # short, stable SearchStore ID; provider ``id`` stays unchanged


@dataclass
class SearchResultAssessment:
    """Retrieval-time assessment for one result, kept separate from provider data."""

    document_ref: str = ""
    relevance: Optional[float] = None
    novelty: Optional[float] = None
    estimated_improvement_score: Optional[float] = None
    estimated_score: Optional[float] = None
    reason: str = ""
    helpfulness: Optional[float] = None


def _finite_float(value: Any) -> Optional[float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass
class SearchImpact:
    """One measured child outcome produced with a stored search record."""

    iteration: Optional[int] = None
    target_program_id: str = ""
    target_score: Optional[float] = None
    result_program_id: str = ""
    result_score: Optional[float] = None
    delta: Optional[float] = None
    feedback: Optional[str] = None


def _new_search_record_id() -> str:
    return f"sr_{uuid.uuid4().hex}"


@dataclass
class SearchRecord:
    """One evidence event stored in ``D_search``.

    ``query`` is None for stored-document look-up.
    Multiple ``source_queries`` identify combined retrieval evidence:
    its measured outcome belongs to the combined evidence, not ``query`` alone.
    """

    query: Optional["ConstructedQuery"] = None
    search_results: List[SearchResult] = field(default_factory=list)
    id: str = field(default_factory=_new_search_record_id)
    iteration: Optional[int] = None
    retrieved_for_program_id: str = ""
    retrieved_for_score: Optional[float] = None
    assessments: List[SearchResultAssessment] = field(default_factory=list)
    relevance: Optional[float] = None
    novelty: Optional[float] = None
    estimated_improvement_score: Optional[float] = None
    estimated_score: Optional[float] = None
    impacts: List[SearchImpact] = field(default_factory=list)
    helpfulness: Optional[float] = None
    source_queries: List[str] = field(default_factory=list)
    operation: str = "retrieve"  # "look-up" records reuse documents already in D_search
    # Source identity/history for reused documents. Full source-record snapshots
    # live in the iteration trace, keeping persisted lookup links non-recursive.
    lookup_provenance: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def delta(self) -> Optional[float]:
        """Mean measured improvement; an unused record remains unrated (``None``)."""
        measured = [
            value for impact in self.impacts if (value := _finite_float(impact.delta)) is not None
        ]
        return sum(measured) / len(measured) if measured else None


# Persisted location of D_search, relative to the run's output_dir.
SEARCH_DB_SUBDIR = "search_database"
SEARCH_DB_FILE = "search_result.jsonl"
SEARCH_RECORD_ID_METADATA_KEY = "evoduet_search_record_id"


def _from_dict(cls, data: dict):
    """Read the complete current dataclass format without aliases or defaults."""
    names = {f.name for f in fields(cls)}
    if not isinstance(data, dict) or data.keys() != names:
        raise ValueError(f"invalid {cls.__name__} checkpoint fields")
    return cls(**copy.deepcopy(data))


def _record_from_dict(data: dict) -> SearchRecord:
    """Reconstruct one current record and its nested credit data."""
    from skydiscover.evoduet.query_construction import ConstructedQuery

    record = _from_dict(SearchRecord, data)
    if not isinstance(record.id, str) or not record.id:
        raise ValueError("checkpoint search record must have an id")
    if record.query is not None:
        record.query = _from_dict(ConstructedQuery, record.query)
    for name, item_type in (
        ("search_results", SearchResult),
        ("assessments", SearchResultAssessment),
        ("impacts", SearchImpact),
    ):
        items = getattr(record, name)
        if not isinstance(items, list):
            raise ValueError(f"checkpoint {name} must be a list")
        setattr(record, name, [_from_dict(item_type, item) for item in items])
    return record


class SearchStore:
    """``D_search`` — accumulated queries and their results across iterations.

    Empty by default. When ``output_dir`` is given,
    the store is (re)written after each ``add``/``record_usage`` to
    ``<output_dir>/search_database/search_result.jsonl`` (one JSON line per
    retrieval event, including its accumulated usage).
    """

    def __init__(
        self,
        records: Optional[List[SearchRecord]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        search_selection_policy: str = "top_k",
        search_selection_num: Optional[int] = None,
        search_selection_criterion: str = "relevance_score",
        documents_per_entry: int = 3,
        max_document_chars: int = 10_000,
    ) -> None:
        self.records: List[SearchRecord] = list(records or [])
        live_path = Path(output_dir) / SEARCH_DB_SUBDIR / SEARCH_DB_FILE if output_dir else None
        # The live JSONL is an output mirror, never resume state. Exact resume is
        # restored from the selected program checkpoint by ``load_state_dict``.
        self._document_ids_by_identity: Dict[tuple[str, str], str] = {}
        self._document_identities: Dict[str, Optional[tuple[str, str]]] = {}
        self._next_document_id = 1
        self._register_document_ids(self.documents())
        # Search-memory selection: "top_k" (rank by search_selection_criterion) | "recency"
        # (most recently added) | "full".
        self.search_selection_policy: str = search_selection_policy
        self.search_selection_num: Optional[int] = search_selection_num
        self.search_selection_criterion: str = search_selection_criterion
        self.documents_per_entry = max(1, int(documents_per_entry))
        self.max_document_chars = max(1, int(max_document_chars))
        # iteration -> {results, queries} injected that turn, so record_usage() can
        # attribute the child's score/feedback back. Keyed by iteration (unique per
        # turn) — safe under asyncio's single-threaded concurrency.
        self._pending: Dict[int, Dict[str, Any]] = {}
        self.path: Optional[Path] = live_path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._persist()

    def add(
        self,
        query: Optional["ConstructedQuery"],
        search_results: List[SearchResult],
        *,
        iteration: Optional[int] = None,
        target_program_id: str = "",
        target_score: Optional[float] = None,
        assessments: Optional[List[SearchResultAssessment]] = None,
        relevance: Optional[float] = None,
        novelty: Optional[float] = None,
        estimated_improvement_score: Optional[float] = None,
        estimated_score: Optional[float] = None,
        retrieved_for_program_id: Optional[str] = None,
        retrieved_for_score: Optional[float] = None,
        helpfulness: Optional[float] = None,
        source_queries: Optional[List[str]] = None,
        operation: str = "retrieve",
        lookup_provenance: Optional[List[Dict[str, Any]]] = None,
    ) -> SearchRecord:
        """Accumulate a ``(query, search_results)`` record (``query`` may be None).

        Retrieval-time target data is copied into the pending use, so later credit is
        measured against that exact score even if the population has since changed.
        Empty retrievals are remembered but not staged because no evidence was used.
        Pooled retrievals credit the record and documents, not a representative query.
        """
        results = list(search_results)
        self._register_document_ids(results)
        assessed = list(assessments or [])
        program_id = (
            retrieved_for_program_id if retrieved_for_program_id is not None else target_program_id
        )
        retrieval_score = retrieved_for_score if retrieved_for_score is not None else target_score
        assessment_fields = (
            "relevance",
            "novelty",
        )
        # Preserve imported rating formats without treating a partially assessed
        # batch as an assessed record. The observed-search loop stores forecasts
        # in its query trace, separately from these checkpoint compatibility fields.
        complete_assessments = bool(assessed) and all(
            all(
                _finite_float(getattr(assessment, name, None)) is not None
                for name in assessment_fields
            )
            and (
                _finite_float(assessment.helpfulness) is not None
                or (
                    _finite_float(assessment.estimated_improvement_score) is not None
                    and _finite_float(assessment.estimated_score) is not None
                )
            )
            for assessment in assessed
        )

        def aggregate(name: str, explicit: Optional[float]) -> Optional[float]:
            if explicit is not None:
                return _finite_float(explicit)
            if not complete_assessments:
                return None
            values = [
                value
                for assessment in assessed
                if (value := _finite_float(getattr(assessment, name, None))) is not None
            ]
            return sum(values) / len(values) if values else None

        record = SearchRecord(
            query=query,
            search_results=results,
            iteration=iteration,
            retrieved_for_program_id=program_id or "",
            retrieved_for_score=retrieval_score,
            assessments=assessed,
            relevance=aggregate("relevance", relevance),
            novelty=aggregate("novelty", novelty),
            estimated_improvement_score=aggregate(
                "estimated_improvement_score", estimated_improvement_score
            ),
            estimated_score=aggregate("estimated_score", estimated_score),
            helpfulness=aggregate("helpfulness", helpfulness),
            source_queries=list(source_queries or []),
            operation=operation,
            lookup_provenance=copy.deepcopy(lookup_provenance or []),
        )
        self.records.append(record)
        if iteration is not None and results:
            self._pending[iteration] = {
                "records": [record],
                "results": list(record.search_results),
                # The pooled child measures the joint record, not its representative query.
                "queries": [query] if query is not None and len(record.source_queries) <= 1 else [],
                "target_program_id": program_id or "",
                "target_score": retrieval_score,
            }
        self._persist()
        return record

    def documents(self) -> List[SearchResult]:
        """All accumulated search results, flattened across records."""
        return [r for record in self.records for r in record.search_results]

    @staticmethod
    def _document_identity(document: SearchResult) -> Optional[tuple[str, str]]:
        """Use exact provenance, without URL normalization or content-based inference."""
        if isinstance(document.url, str) and document.url:
            return ("url", document.url)
        if isinstance(document.id, str) and document.id:
            return ("provider_id", document.id)
        return None

    def _register_document_ids(
        self,
        documents: Iterable[SearchResult],
        *,
        reset: bool = False,
        next_document_id: int = 1,
    ) -> None:
        """Assign only the new ID field, preserving caller-owned document objects.

        Reserve all supplied IDs before allocating IDs for new documents.
        Validate conflicting provenance before
        changing any document or registry. Anonymous documents share an ID only if
        they are the same object or already carry the same registered ID.
        """
        documents = list(documents)
        identities = {} if reset else dict(self._document_ids_by_identity)
        owners = {} if reset else dict(self._document_identities)
        if reset:
            if (
                isinstance(next_document_id, bool)
                or not isinstance(next_document_id, int)
                or next_document_id < 1
            ):
                raise ValueError("checkpoint next_document_id must be a positive integer")
            counter = next_document_id
        else:
            counter = self._next_document_id

        for document in documents:
            assigned = document.search_document_id
            if not isinstance(assigned, str):
                raise ValueError("search_document_id must be a string")
            if not assigned:
                continue
            identity = self._document_identity(document)
            previous_id = identities.get(identity) if identity is not None else None
            if previous_id is not None and previous_id != assigned:
                raise ValueError(
                    f"Conflicting search_document_id values for the same document: "
                    f"{previous_id!r} and {assigned!r}"
                )
            previous_identity = owners.get(assigned)
            if (
                previous_identity is not None
                and identity is not None
                and previous_identity != identity
            ):
                raise ValueError(f"search_document_id {assigned!r} belongs to different documents")
            owners[assigned] = identity if identity is not None else previous_identity
            if identity is not None:
                identities[identity] = assigned
            suffix = re.fullmatch(r"doc_([0-9]+)", assigned)
            if suffix:
                counter = max(counter, int(suffix.group(1)) + 1)

        assignments = []
        planned: Dict[int, str] = {}
        for document in documents:
            if document.search_document_id:
                continue
            identity = self._document_identity(document)
            assigned = planned.get(id(document)) or (
                identities.get(identity) if identity is not None else None
            )
            if assigned is None:
                assigned = f"doc_{counter:06d}"
                while assigned in owners:
                    counter += 1
                    assigned = f"doc_{counter:06d}"
                counter += 1
                owners[assigned] = identity
                if identity is not None:
                    identities[identity] = assigned
            planned[id(document)] = assigned
            assignments.append((document, assigned))

        for document, assigned in assignments:
            document.search_document_id = assigned
        self._document_ids_by_identity = identities
        self._document_identities = owners
        self._next_document_id = counter

    def lookup_documents(self, document_ids: Iterable[str]) -> List[SearchResult]:
        """Resolve short IDs first, then exact legacy provider IDs, without mutation.

        IDs are opaque strings; URL inference and normalization are not performed.
        Aliases resolving to the same document are deduplicated in requested order,
        unknown/non-string IDs are logged and skipped, and the last stored version
        wins. Returned documents are the stored objects; callers creating a new
        reuse record must copy them before changing fields or usage history.
        """
        if isinstance(document_ids, (str, bytes)):
            raise TypeError("document_ids must be an iterable of string IDs, not one string")
        latest = {
            document.search_document_id: document
            for record in self.records
            for document in record.search_results
            if document.search_document_id
        }
        legacy = {
            document.id: latest.get(document.search_document_id, document)
            for record in self.records
            for document in record.search_results
            if isinstance(document.id, str) and document.id
        }
        seen_requests = set()
        seen_documents = set()
        results = []
        for document_id in document_ids:
            if not isinstance(document_id, str):
                logger.warning("Search lookup: skipping non-string document ID %r", document_id)
                continue
            if document_id in seen_requests:
                continue
            seen_requests.add(document_id)
            document = latest.get(document_id) or legacy.get(document_id)
            if document is None:
                logger.warning(
                    "Search lookup: unknown search_document_id %r; skipping", document_id
                )
                continue
            resolved_id = document.search_document_id or id(document)
            if resolved_id in seen_documents:
                continue
            seen_documents.add(resolved_id)
            results.append(document)
        return results

    def select_store(self) -> List[SearchResult]:
        """Apply the selection policy on the fly and return the selected results.

        ``"full"`` returns every cached result; ``"top_k"`` ranks results by
        ``search_selection_criterion`` (e.g. ``relevance_score``) across all
        records — missing values rank last — and keeps the top
        ``search_selection_num``; ``"recency"`` keeps the most recently added
        ``search_selection_num`` results (``D_search`` appends per retrieval, so the
        tail is newest) and returns them newest-first. The store keeps accumulating;
        selection is computed per call. The search-context renderer associates
        each selected result with its recorded outcome.
        """
        if self.search_selection_policy == "full":
            return self.documents()
        results = self.documents()
        k = self.search_selection_num if self.search_selection_num else len(results)
        if self.search_selection_policy == "recency":
            return list(reversed(results[-k:]))  # last-added kept, newest-first
        crit = self.search_selection_criterion
        results.sort(
            key=lambda r: (
                getattr(r, crit, None) if getattr(r, crit, None) is not None else float("-inf")
            ),
            reverse=True,
        )
        return results[:k]

    def select_records(
        self,
        policy: Optional[str] = None,
        num: Optional[int] = None,
    ) -> List[SearchRecord]:
        """Select whole search experiences without changing legacy document selection.

        ``delta`` includes both successful and harmful directions, then fills with
        recent unrated records. ``estimated_score`` ranks assessed records and uses
        that same recent-unrated fallback.
        """
        policy = policy or self.search_selection_policy
        requested = self.search_selection_num if num is None else num
        k = (
            len(self.records)
            if not requested or requested <= 0
            else min(requested, len(self.records))
        )
        if policy == "full":
            return list(self.records)
        if policy == "recency":
            return list(reversed(self.records))[:k]
        if policy == "estimated_score":
            newest = list(reversed(self.records))
            rated = [r for r in newest if _finite_float(r.estimated_score) is not None]
            rated.sort(key=lambda r: _finite_float(r.estimated_score), reverse=True)
            unrated = [r for r in newest if _finite_float(r.estimated_score) is None]
            return (rated + unrated)[:k]
        if policy == "delta":
            rated = sorted(
                (r for r in self.records if r.delta is not None),
                key=lambda r: r.delta,
                reverse=True,
            )
            unrated = list(reversed([r for r in self.records if r.delta is None]))
            best_count = min(len(rated), (k + 1) // 2)
            chosen = rated[:best_count]
            worst_count = min(k - len(chosen), len(rated) - best_count)
            if worst_count:
                chosen.extend(rated[-worst_count:])
            chosen.extend(unrated[: max(0, k - len(chosen))])
            return chosen
        raise ValueError(
            f"unknown search record selection policy {policy!r}; "
            "expected delta | estimated_score | recency | full"
        )

    def _prompt_text(self, value: Any) -> str:
        return " ".join(str(value or "").split()).replace("```", "~~~")[: self.max_document_chars]

    def record_usage(
        self,
        iteration: int,
        score: Optional[float] = None,
        feedback: Optional[str] = None,
        result_program_id: str = "",
    ) -> None:
        """Attribute iteration ``iteration``'s child outcome to the query/results it used.

        For each query and result injected that turn: ``visit_count += 1`` and the
        iteration / evaluator ``score`` / ``feedback`` are appended to its history.
        No-op if that iteration injected nothing (e.g. the gate said no-op).
        """
        pending = self._pending.pop(iteration, None)
        if not pending:
            return
        target_score = _finite_float(pending.get("target_score"))
        result_score = _finite_float(score)
        delta = (
            result_score - target_score
            if result_score is not None and target_score is not None
            else None
        )
        bounded_feedback = self._prompt_text(feedback) if feedback else None
        for record in pending.get("records", []):
            record.impacts.append(
                SearchImpact(
                    iteration=iteration,
                    target_program_id=pending.get("target_program_id") or "",
                    target_score=target_score,
                    result_program_id=result_program_id or "",
                    result_score=result_score,
                    delta=delta,
                    feedback=bounded_feedback,
                )
            )
        for item in (*pending["results"], *pending["queries"]):
            item.visit_count += 1
            item.evolution_iterations.append(iteration)
            if result_score is not None:
                item.evolution_scores.append(result_score)
        self._persist()

    def staged_record_id(self, iteration: int) -> Optional[str]:
        """Return the record whose evidence is pending for ``iteration``."""
        records = (self._pending.get(iteration) or {}).get("records") or []
        return records[0].id if records else None

    def discard_usage(self, iteration: int) -> None:
        """Forget staged evidence that never reached a mutation prompt."""
        self._pending.pop(iteration, None)

    def state_dict(self) -> Dict[str, Any]:
        """Return the learned search memory for an exact checkpoint resume."""
        pending = {}
        for iteration, staged in self._pending.items():
            entries = []
            staged_results = {id(result) for result in staged.get("results", [])}
            for record in staged.get("records", []):
                entries.append(
                    {
                        "record_id": record.id,
                        "result_indices": [
                            index
                            for index, result in enumerate(record.search_results)
                            if id(result) in staged_results
                        ],
                    }
                )
            pending[str(iteration)] = {
                "records": entries,
                "target_program_id": staged.get("target_program_id") or "",
                "target_score": staged.get("target_score"),
            }
        return {
            "records": [asdict(record) for record in self.records],
            "pending": pending,
            "next_document_id": self._next_document_id,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Replace search memory with checkpoint state and refresh its live mirror."""
        if not isinstance(state, dict) or state.keys() != {
            "records",
            "pending",
            "next_document_id",
        }:
            raise ValueError("invalid SearchStore checkpoint fields")
        if not isinstance(state["records"], list) or not isinstance(state["pending"], dict):
            raise ValueError("invalid SearchStore checkpoint records or pending credit")
        counter = state["next_document_id"]
        if type(counter) is not int or counter < 1:
            raise ValueError("checkpoint next_document_id must be a positive integer")
        records = [_record_from_dict(row) for row in state["records"]]
        records_by_id = {record.id: record for record in records}
        if len(records_by_id) != len(records):
            raise ValueError("duplicate search record ids in checkpoint")
        documents = [document for record in records for document in record.search_results]
        for document in documents:
            if not isinstance(document.search_document_id, str) or not document.search_document_id:
                raise ValueError("checkpoint document must have a search_document_id")
            suffix = re.fullmatch(r"doc_([0-9]+)", document.search_document_id)
            if suffix and int(suffix.group(1)) >= counter:
                raise ValueError("checkpoint next_document_id would reuse an existing id")

        pending = {}
        for raw_iteration, staged in state["pending"].items():
            if not isinstance(raw_iteration, str) or not re.fullmatch(
                r"0|[1-9][0-9]*", raw_iteration
            ):
                raise ValueError("invalid pending checkpoint iteration")
            if not isinstance(staged, dict) or staged.keys() != {
                "records",
                "target_program_id",
                "target_score",
            }:
                raise ValueError("invalid pending checkpoint credit fields")
            if not isinstance(staged["records"], list) or not staged["records"]:
                raise ValueError("pending checkpoint credit must reference records")
            selected_records = []
            results = []
            for entry in staged["records"]:
                if not isinstance(entry, dict) or entry.keys() != {"record_id", "result_indices"}:
                    raise ValueError("invalid pending checkpoint record fields")
                record_id = entry["record_id"]
                record = records_by_id.get(record_id) if isinstance(record_id, str) else None
                if record is None:
                    raise ValueError("pending checkpoint credit references a missing record")
                indices = entry["result_indices"]
                if (
                    not isinstance(indices, list)
                    or not indices
                    or any(
                        type(index) is not int or not 0 <= index < len(record.search_results)
                        for index in indices
                    )
                    or len(set(indices)) != len(indices)
                ):
                    raise ValueError("invalid pending checkpoint result indices")
                selected_records.append(record)
                results.extend(record.search_results[index] for index in indices)
            pending[int(raw_iteration)] = {
                "records": selected_records,
                "results": results,
                "queries": [
                    record.query
                    for record in selected_records
                    if record.query is not None and len(record.source_queries) <= 1
                ],
                "target_program_id": staged["target_program_id"],
                "target_score": staged["target_score"],
            }
        # All deserialization and reference validation precede changes to live state.
        self._register_document_ids(documents, reset=True, next_document_id=counter)
        self.records = records
        self._pending = pending
        self._persist()

    def _write_records(self, path: Union[str, Path]) -> None:
        """Write the current records to ``path`` (one JSON line per record, incl. usage)."""
        with open(path, "w") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), default=str) + "\n")

    def _persist(self) -> None:
        """Rewrite ``search_result.jsonl`` from the current records. Rewrite — not append —
        because record_usage mutates earlier records. No-op when persistence is off."""
        if self.path is None:
            return
        try:
            self._write_records(self.path)
        except OSError:
            logger.exception("EvoDuet: could not persist search database to %s", self.path)

    def snapshot(self, dir_path: Union[str, Path]) -> Path:
        """Write a checkpoint copy of D_search to ``<dir_path>/search_result.jsonl``.

        Used for the per-iteration ``evoduet/checkpoint_<N>/`` snapshots so each
        checkpoint preserves the search database (records + accumulated usage) as of that
        iteration, even though the live file is continually rewritten. Returns the path.
        """
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        out = d / SEARCH_DB_FILE
        self._write_records(out)
        return out


class BaseRetrieval:
    """Common base for retrieve backends.

    Subclasses implement ``retrieve``; ``SearchStore`` preserves their search history.
    """

    def __init__(self, config):
        self.config = config

    async def retrieve(self, *args, **kwargs) -> List[SearchResult]:
        raise NotImplementedError
