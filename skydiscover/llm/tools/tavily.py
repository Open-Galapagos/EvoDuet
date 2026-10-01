"""Tavily web-search function tool for ordinary LLM generation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import httpx

from skydiscover.config import TavilyToolConfig
from skydiscover.utils.tavily import TavilyResponseDecodeError, tavily_search

logger = logging.getLogger("skydiscover.llm")

TAVILY_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "tavily",
        "description": (
            "Search the public web for current or external information. "
            "Returns relevant page titles, URLs, and content snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused natural-language web search query.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 20,
                    "default": 5,
                    "description": "The maximum number of search results to return.",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass
class TavilyExecutionResult:
    """One Tavily execution split into provider and model-facing payloads."""

    llm_result: str
    request: Dict[str, Any] | None = None
    response: Any = None
    error: Any = None


class TavilyTool:
    """Execute model-issued ``tavily`` calls through Tavily's REST API."""

    name = "tavily"
    chat_completions_schema = TAVILY_TOOL_SCHEMA

    def __init__(self, config: TavilyToolConfig):
        self.config = config

    @property
    def responses_schema(self) -> Dict[str, Any]:
        function = self.chat_completions_schema["function"]
        return {
            "type": "function",
            "name": function["name"],
            "description": function["description"],
            "parameters": function["parameters"],
        }

    def _build_payload(self, query: str, *, max_results: int | None = None) -> Dict[str, Any]:
        cfg = self.config
        requested_max_results = cfg.max_results if max_results is None else max_results
        bounded_max_results = max(0, min(int(requested_max_results), 20))
        payload: Dict[str, Any] = {
            "query": query,
            "search_depth": cfg.search_depth,
            "topic": cfg.topic,
            "max_results": bounded_max_results,
            "include_answer": cfg.include_answer,
            "include_raw_content": cfg.include_raw_content,
            "include_usage": cfg.include_usage,
        }
        # Tavily documents chunks_per_source as an advanced-search-only input.
        if cfg.search_depth == "advanced":
            payload["chunks_per_source"] = max(1, min(int(cfg.chunks_per_source), 3))
        if cfg.auto_parameters:
            payload["auto_parameters"] = True
        if cfg.exact_match:
            payload["exact_match"] = True
        if cfg.time_range:
            payload["time_range"] = cfg.time_range
        if cfg.start_date:
            payload["start_date"] = cfg.start_date
        if cfg.end_date:
            payload["end_date"] = cfg.end_date
        if cfg.include_domains:
            payload["include_domains"] = list(cfg.include_domains)
        if cfg.exclude_domains:
            payload["exclude_domains"] = list(cfg.exclude_domains)
        if cfg.country:
            payload["country"] = cfg.country
        if cfg.language:
            payload["language"] = cfg.language
        if cfg.filter_by_language:
            payload["filter_by_language"] = True
        if cfg.include_images:
            payload["include_images"] = True
            if cfg.include_image_descriptions:
                payload["include_image_descriptions"] = True
        if cfg.include_favicon:
            payload["include_favicon"] = True
        if cfg.safe_search:
            payload["safe_search"] = True
        return payload

    def provenance_config(self) -> Dict[str, Any]:
        """Return the complete effective tool config without credentials."""
        cfg = self.config
        return {
            "base_url": cfg.base_url,
            "timeout": cfg.timeout,
            "search_depth": cfg.search_depth,
            "chunks_per_source": cfg.chunks_per_source,
            "max_results": cfg.max_results,
            "topic": cfg.topic,
            "time_range": cfg.time_range,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "include_answer": cfg.include_answer,
            "include_raw_content": cfg.include_raw_content,
            "include_images": cfg.include_images,
            "include_image_descriptions": cfg.include_image_descriptions,
            "include_favicon": cfg.include_favicon,
            "include_domains": list(cfg.include_domains),
            "exclude_domains": list(cfg.exclude_domains),
            "country": cfg.country,
            "language": cfg.language,
            "filter_by_language": cfg.filter_by_language,
            "auto_parameters": cfg.auto_parameters,
            "exact_match": cfg.exact_match,
            "include_usage": cfg.include_usage,
            "safe_search": cfg.safe_search,
        }

    async def execute_with_metadata(self, arguments: Dict[str, Any]) -> TavilyExecutionResult:
        """Run Tavily while preserving the exact request and provider response."""
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            error = "'query' must be a non-empty string."
            return TavilyExecutionResult(llm_result=_json_result({"error": error}), error=error)

        requested_max_results = arguments.get("max_results")
        if requested_max_results is not None and (
            isinstance(requested_max_results, bool) or not isinstance(requested_max_results, int)
        ):
            error = "'max_results' must be an integer."
            return TavilyExecutionResult(llm_result=_json_result({"error": error}), error=error)

        api_key = self.config.api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            error = (
                "Tavily API key is not configured. Set TAVILY_API_KEY or "
                "llm.tavily_tool.api_key."
            )
            return TavilyExecutionResult(llm_result=_json_result({"error": error}), error=error)

        query = query.strip()
        request = self._build_payload(query, max_results=requested_max_results)
        try:
            data = await tavily_search(
                request,
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text if exc.response is not None else ""
            logger.warning("tavily tool HTTP %s: %s", status, body[:300])
            error = f"Tavily search HTTP {status}."
            llm_result = _json_result({"error": error, "details": body[:300]})
            response: Any = {"status_code": status, "body": body}
            try:
                if exc.response is not None:
                    response = exc.response.json()
            except (ValueError, json.JSONDecodeError):
                pass
            return TavilyExecutionResult(
                llm_result=llm_result,
                request=request,
                response=response,
                error=error,
            )
        except TavilyResponseDecodeError as exc:
            logger.warning("tavily tool returned invalid JSON: %s", exc)
            error = str(exc)
            return TavilyExecutionResult(
                llm_result=_json_result({"error": error}),
                request=request,
                response={"status_code": exc.status_code, "body": exc.body},
                error=error,
            )
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("tavily tool failed: %s", exc)
            error = f"Tavily search failed: {exc}"
            return TavilyExecutionResult(
                llm_result=_json_result({"error": error}),
                request=request,
                error=error,
            )

        logger.info(
            "tavily tool returned %d result(s) for %r",
            len(data.get("results") or []),
            query[:100],
        )
        return TavilyExecutionResult(
            llm_result=_format_response(data, query=query),
            request=request,
            response=data,
        )

    async def execute(self, arguments: Dict[str, Any]) -> str:
        """Run one tool invocation and return the compact JSON seen by the model."""
        return (await self.execute_with_metadata(arguments)).llm_result


def _format_response(data: Dict[str, Any], *, query: str) -> str:
    """Serialize only title, URL, and query-relevant content for tool context."""
    results = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        result = {
            key: item[key]
            for key in ("title", "url", "content")
            if item.get(key) not in (None, "", [])
        }
        results.append(result)

    return _json_result({"results": results})


def _json_result(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
