"""Tavily retrieval backend — the structured ``Search(q)`` over the Tavily REST API.

Given a constructed query ``q`` (a :class:`ConstructedQuery` or a plain string), it calls
Tavily's ``/search`` endpoint and maps its source-backed ranked hits into
:class:`SearchResult`s for the EvoDuet retrieval pipeline.

Only ``/search`` is used: one call returns query-ranked documents with snippet
content (and, with ``include_raw_content``, the full page) plus a relevance score —
exactly the shape this layer needs. ``/extract``, ``/crawl`` and ``/map`` operate
on already-known URLs/domains and are intentionally not used here (see README).

No SDK dependency: this talks to the documented REST API with ``httpx`` (already a
core dependency).
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Any, List, Optional, Union
from urllib.parse import urlsplit

import httpx

from skydiscover.evoduet.retrieval.base_retrieval import BaseRetrieval, SearchResult
from skydiscover.utils.tavily import tavily_search

if TYPE_CHECKING:
    from skydiscover.evoduet.query_construction import ConstructedQuery

logger = logging.getLogger("skydiscover.evoduet")


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _http_url(value: Any) -> str:
    url = value.strip()[:2_048] if isinstance(value, str) else ""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except ValueError:
        return ""
    return url if parsed.scheme in {"http", "https"} and hostname else ""


def _usage_metadata(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    output = {}
    for key, item in list(value.items())[:16]:
        name = str(key)[:64]
        if item is None or isinstance(item, (bool, int, float)):
            output[name] = item
        else:
            output[name] = str(item)[:256]
    return output


def _to_documents(
    data: dict,
    *,
    query_str: str,
    keywords: Optional[List[str]],
    query_type: Optional[str],
    iteration: Optional[int],
    max_document_chars: int = 10_000,
    max_results: int = 20,
) -> List[SearchResult]:
    """Map a Tavily ``/search`` response into ranked ``SearchResult``s.

    ``results[i].content`` is Tavily's query-relevant snippet (-> condensed);
    ``raw_content`` (present only with ``include_raw_content``) is the full page,
    and falls back to the snippet when absent so ``raw_content`` is never empty.
    The top-level synthesized ``answer`` is deliberately ignored: unlike
    ``results[]`` it has no source URL or provider score. ``favicon`` / ``images``
    map to dedicated fields; call metadata is copied to each result.
    """
    docs: List[SearchResult] = []
    # Call-level fields shared by every doc from this response — credit usage (when
    # include_usage=true) is handy for cost tracking the gate; request_id for support.
    call_meta = {
        "response_time": _optional_float(data.get("response_time")),
        "usage": _usage_metadata(data.get("usage")),
        "request_id": str(data.get("request_id") or "")[:256],
    }

    results = data.get("results")
    if not isinstance(results, list):
        return []
    for i, item in enumerate(results[:max_results], start=1):
        if not isinstance(item, dict):
            continue
        url = _http_url(item.get("url"))
        content = item.get("content")
        if not url or not isinstance(content, str) or not content.strip():
            continue
        snippet = content[:max_document_chars]
        raw_content = item.get("raw_content")
        raw = (
            raw_content[:max_document_chars]
            if isinstance(raw_content, str) and raw_content
            else snippet
        )
        images = item.get("images")
        bounded_images = (
            [_http_url(image) for image in images[:10] if _http_url(image)]
            if isinstance(images, list)
            else None
        )
        docs.append(
            SearchResult(
                raw_content=raw,
                content=snippet,
                id=url,
                title=str(item.get("title") or "")[:500],
                url=url,
                source="tavily",
                published_date=str(item.get("published_date") or "")[:100] or None,
                favicon=_http_url(item.get("favicon")) or None,
                images=bounded_images,
                query=query_str,
                keywords=keywords,
                query_type=query_type,
                rank=i,
                relevance_score=_optional_float(item.get("score")),
                iteration=iteration,
                metadata=call_meta,
            )
        )
    return docs


class TavilyRetrieval(BaseRetrieval):
    """Structured retrieve backend: ``Search(q)`` via the Tavily ``/search`` API."""

    def __init__(self, config):
        super().__init__(config)
        tv = config.evoduet.tavily_retrieval
        self.api_key: Optional[str] = tv.api_key or os.environ.get("TAVILY_API_KEY")
        self.base_url: str = tv.base_url
        self.timeout = tv.timeout
        # search behaviour
        self.search_depth: str = tv.search_depth
        self.topic: str = tv.topic
        self.chunks_per_source = tv.chunks_per_source
        self.auto_parameters = tv.auto_parameters
        self.exact_match = tv.exact_match
        # Documents kept per retrieval: explicit max_results, else the shared top-k.
        self.max_results = (
            tv.max_results if tv.max_results is not None else config.evoduet.top_k_for_retrieval
        )
        # date filtering
        self.time_range = tv.time_range
        self.start_date = tv.start_date
        self.end_date = tv.end_date
        # domain / geo filtering
        self.include_domains = list(tv.include_domains or [])
        self.exclude_domains = list(tv.exclude_domains or [])
        self.country = tv.country
        # response content
        self.include_answer = tv.include_answer
        self.include_raw_content = tv.include_raw_content
        self.include_images = tv.include_images
        self.include_image_descriptions = tv.include_image_descriptions
        self.include_favicon = tv.include_favicon
        self.include_usage = tv.include_usage
        self.safe_search = tv.safe_search
        self.max_document_chars = config.evoduet.max_document_chars

    def preflight(self) -> None:
        """Fail before model calls if this run cannot issue a web search."""
        if not self.api_key:
            raise RuntimeError(
                "evoduet retrieval needs Tavily; set TAVILY_API_KEY or "
                "evoduet.tavily_retrieval.api_key"
            )

    def _build_payload(self, query_str: str, *, max_results: Optional[int] = None) -> dict:
        # Core params, always sent. Cost is set by search_depth alone (1 credit for
        # basic/fast/ultra-fast, 2 for advanced) unless auto_parameters overrides it.
        payload: dict = {
            "query": query_str,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "max_results": self.max_results if max_results is None else max_results,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "include_usage": self.include_usage,
        }
        # chunks_per_source only applies to advanced search.
        if self.search_depth == "advanced":
            payload["chunks_per_source"] = self.chunks_per_source
        if self.auto_parameters:
            payload["auto_parameters"] = True
        if self.exact_match:
            payload["exact_match"] = True
        # date filtering
        if self.time_range:
            payload["time_range"] = self.time_range
        if self.start_date:
            payload["start_date"] = self.start_date
        if self.end_date:
            payload["end_date"] = self.end_date
        # domain / geo filtering
        if self.include_domains:
            payload["include_domains"] = self.include_domains
        if self.exclude_domains:
            payload["exclude_domains"] = self.exclude_domains
        if self.country:
            payload["country"] = self.country
        # optional response content (default off)
        if self.include_images:
            payload["include_images"] = True
            if self.include_image_descriptions:
                payload["include_image_descriptions"] = True
        if self.include_favicon:
            payload["include_favicon"] = True
        if self.safe_search:
            payload["safe_search"] = True
        return payload

    async def retrieve(
        self,
        query: Union["ConstructedQuery", str],
        *,
        iteration: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """``Search(q)`` — fetch documents with an optional limit for this call.

        ``query`` may be a :class:`ConstructedQuery` or a plain string. A missing
        API key raises during preflight. An empty query or API/network error
        returns an empty list. ``max_results`` overrides the configured limit
        for this request without changing later requests.
        """
        self.preflight()
        limit = self.max_results if max_results is None else max_results

        if isinstance(query, str):
            query_str, keywords, query_type = query, None, None
        else:
            query_str = query.text
            keywords, query_type = query.keywords, query.query_type

        if not query_str.strip():
            logger.warning("tavily retrieval: empty query; returning no documents.")
            return []

        try:
            data = await tavily_search(
                self._build_payload(query_str, max_results=limit),
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300] if e.response is not None else ""
            logger.error("tavily retrieval HTTP %s: %s", e.response.status_code, body)
            return []
        except (httpx.HTTPError, ValueError) as e:
            logger.error("tavily retrieval failed: %s", e)
            return []

        docs = _to_documents(
            data,
            query_str=query_str,
            keywords=keywords,
            query_type=query_type,
            iteration=iteration,
            max_document_chars=self.max_document_chars,
            max_results=limit,
        )
        logger.info("tavily retrieval: %d documents for query %r", len(docs), query_str[:80])
        return docs


if __name__ == "__main__":
    # Standalone smoke test against the live API:
    #   TAVILY_API_KEY=tvly-... python -m skydiscover.evoduet.retrieval.tavily_retrieval "your query"
    import asyncio
    import sys

    async def _main() -> None:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            sys.exit("set TAVILY_API_KEY to run the smoke test")
        q = " ".join(sys.argv[1:]) or "circle packing 26 circles unit square maximize sum of radii"
        data = await tavily_search(
            {"query": q, "search_depth": "advanced", "max_results": 5, "include_answer": True},
            api_key=key,
            base_url="https://api.tavily.com",
            timeout=30.0,
        )
        docs = _to_documents(data, query_str=q, keywords=None, query_type=None, iteration=0)
        print(f"{len(docs)} documents for: {q}\n")
        for d in docs:
            print(
                f"  [{d.rank}] score={d.relevance_score} {d.title}\n      {d.url}\n      {d.content[:160]}\n"
            )

    asyncio.run(_main())
