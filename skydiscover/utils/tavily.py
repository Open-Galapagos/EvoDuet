"""Shared low-level client for Tavily's REST search endpoint."""

from __future__ import annotations

import httpx

_SEARCH_PATH = "/search"


class TavilyResponseDecodeError(ValueError):
    """A successful HTTP response whose body was not valid Tavily JSON."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"Tavily returned non-JSON HTTP {status_code} response")
        self.status_code = status_code
        self.body = body


async def tavily_search(payload: dict, *, api_key: str, base_url: str, timeout: float) -> dict:
    """POST one request to Tavily ``/search`` and return the parsed JSON."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            base_url.rstrip("/") + _SEARCH_PATH,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise TavilyResponseDecodeError(response.status_code, response.text) from exc
