"""Tests for the Tavily retrieval backend.

The unit tests mock the HTTP call (no network / no API key). A single live test
is marked ``integration`` and skips unless ``TAVILY_API_KEY`` is set.
"""

import asyncio
import os
from types import SimpleNamespace

import pytest

from skydiscover.config import EvoDuetConfig
from skydiscover.evoduet.query_construction import ConstructedQuery
from skydiscover.evoduet.retrieval import tavily_retrieval as tv
from skydiscover.evoduet.retrieval.tavily_retrieval import (
    TavilyRetrieval,
    _http_url,
    _to_documents,
)

# A canned Tavily /search response (including an unsourced answer, plus two results).
SAMPLE = {
    "query": "circle packing 26 unit square",
    "response_time": 1.23,
    "usage": {"requests": 1},
    "request_id": "req-123",
    "answer": "The known record is about 2.6359.",
    "results": [
        {
            "title": "OpenEvolve circle packing",
            "url": "https://github.com/codelion/openevolve",
            "content": "SLSQP over centers and radii.",
            "raw_content": "Full page: decision vector = 3n vars ...",
            "score": 0.97,
            "favicon": "https://github.com/favicon.ico",
            "images": ["https://img/one.png"],
        },
        {
            "title": "Packomania n=26",
            "url": "http://packomania.com/26",
            "content": "Tables of optimal packings.",
            "score": 0.81,
        },
    ],
}


def _backend(api_key="tvly-test", top_k=10, **tavily_kw):
    """A TavilyRetrieval wired to a minimal in-memory config."""
    wk = EvoDuetConfig(top_k_for_retrieval=top_k)
    wk.tavily_retrieval.api_key = api_key
    wk.tavily_retrieval.max_results = None
    for k, v in tavily_kw.items():
        setattr(wk.tavily_retrieval, k, v)
    return TavilyRetrieval(SimpleNamespace(evoduet=wk))


def test_to_documents_ignores_unsourced_answer_and_maps_results():
    docs = _to_documents(SAMPLE, query_str="q", keywords=["pack"], query_type="find", iteration=3)
    assert len(docs) == 2
    top = docs[0]
    assert top.rank == 1 and top.relevance_score == 0.97
    assert top.title == "OpenEvolve circle packing"
    assert top.url == "https://github.com/codelion/openevolve"
    assert top.content == "SLSQP over centers and radii."  # results[].content
    assert top.raw_content.startswith("Full page:")  # results[].raw_content
    assert top.query == "q" and top.keywords == ["pack"] and top.iteration == 3
    assert top.favicon == "https://github.com/favicon.ico"  # results[].favicon
    assert top.images == ["https://img/one.png"]  # results[].images
    assert top.metadata["usage"] == {"requests": 1}
    assert top.metadata["request_id"] == "req-123"  # call-level -> metadata

    second = docs[1]
    assert second.favicon is None and second.images is None  # absent -> None


def test_to_documents_raw_content_falls_back_to_snippet():
    # second result has no raw_content -> raw_content should fall back to content
    docs = _to_documents(SAMPLE, query_str="q", keywords=None, query_type=None, iteration=0)
    second = docs[1]
    assert second.raw_content == "Tables of optimal packings."


def test_to_documents_rejects_non_finite_provider_score():
    data = {
        "results": [{"url": "https://example.test", "content": "evidence", "score": float("nan")}]
    }
    document = _to_documents(data, query_str="q", keywords=None, query_type=None, iteration=0)[0]
    assert document.relevance_score is None


def test_to_documents_without_answer():
    data = {k: v for k, v in SAMPLE.items() if k != "answer"}
    docs = _to_documents(data, query_str="q", keywords=None, query_type=None, iteration=0)
    assert len(docs) == 2 and docs[0].rank == 1


def test_to_documents_caps_stored_content():
    data = {
        "results": [{"url": "https://example.test", "content": "s" * 20, "raw_content": "r" * 30}]
    }
    document = _to_documents(
        data,
        query_str="q",
        keywords=None,
        query_type=None,
        iteration=0,
        max_document_chars=7,
    )[0]
    assert document.content == "s" * 7
    assert document.raw_content == "r" * 7


def test_to_documents_caps_response_and_rejects_unsourced_entries():
    data = {
        "results": [
            {"url": "", "content": "missing URL"},
            {"url": "ftp://example.test", "content": "wrong scheme"},
            {"url": "https://example.test/empty", "content": ""},
            {"url": "http://[", "content": "malformed URL"},
            {"url": "https://example.test/late", "content": "outside provider top-k"},
        ]
    }

    assert (
        _to_documents(
            data,
            query_str="q",
            keywords=None,
            query_type=None,
            iteration=0,
            max_results=4,
        )
        == []
    )


@pytest.mark.parametrize("url", ["http://[", "http://:", "http://user@", "ftp://example.test"])
def test_http_url_rejects_malformed_or_unsupported_sources(url):
    assert _http_url(url) == ""


def test_build_payload_uses_top_k_and_explicit_depth():
    backend = _backend(top_k=7, search_depth="advanced", chunks_per_source=2)
    payload = backend._build_payload("my query")
    assert payload["query"] == "my query"
    assert payload["max_results"] == 7  # from top_k_for_retrieval
    assert payload["search_depth"] == "advanced"  # explicit, never auto
    assert payload["chunks_per_source"] == 2  # advanced only
    assert payload["include_usage"] is True
    assert "auto_parameters" not in payload


def test_build_payload_basic_depth_omits_chunks():
    payload = _backend(search_depth="basic")._build_payload("q")
    assert "chunks_per_source" not in payload  # basic search ignores chunks


def test_explicit_max_results_overrides_top_k():
    # max_results set explicitly wins over top_k_for_retrieval
    assert _backend(top_k=10, max_results=3)._build_payload("q")["max_results"] == 3
    # unset -> falls back to top_k
    assert _backend(top_k=10)._build_payload("q")["max_results"] == 10


def test_build_payload_full_config_flags():
    backend = _backend(
        search_depth="advanced",
        topic="news",
        country="united states",
        time_range="year",
        start_date="2024-01-01",
        end_date="2025-01-01",
        exact_match=True,
        auto_parameters=True,
        include_images=True,
        include_image_descriptions=True,
        include_favicon=True,
        safe_search=True,
        exclude_domains=["reddit.com"],
    )
    p = backend._build_payload("q")
    assert p["topic"] == "news"
    assert p["country"] == "united states"
    assert p["time_range"] == "year"
    assert p["start_date"] == "2024-01-01" and p["end_date"] == "2025-01-01"
    assert p["exact_match"] is True and p["auto_parameters"] is True
    assert p["include_images"] is True and p["include_image_descriptions"] is True
    assert p["include_favicon"] is True and p["safe_search"] is True
    assert p["exclude_domains"] == ["reddit.com"]


def test_build_payload_omits_off_flags_by_default():
    p = _backend()._build_payload("q")  # all new flags default off / None
    for k in (
        "country",
        "time_range",
        "start_date",
        "end_date",
        "exact_match",
        "auto_parameters",
        "include_images",
        "include_favicon",
        "safe_search",
    ):
        assert k not in p


def test_retrieve_calls_search_and_maps(monkeypatch):
    captured = {}

    async def fake_search(payload, *, api_key, base_url, timeout):
        captured["payload"] = payload
        captured["api_key"] = api_key
        return SAMPLE

    monkeypatch.setattr(tv, "tavily_search", fake_search)
    backend = _backend(api_key="tvly-xyz", top_k=5)
    docs = asyncio.run(backend.retrieve(ConstructedQuery(query="circle packing"), iteration=8))

    assert captured["api_key"] == "tvly-xyz"
    assert captured["payload"]["query"] == "circle packing"
    assert captured["payload"]["max_results"] == 5
    assert [d.rank for d in docs] == [1, 2]
    assert all(d.iteration == 8 for d in docs)


def test_per_call_limit_controls_payload_and_results_without_changing_concurrent_defaults(
    monkeypatch,
):
    requests = {}
    response = {
        "results": [
            {"title": f"Source {i}", "url": f"https://example.test/{i}", "content": "Evidence"}
            for i in range(4)
        ]
    }

    async def search(payload, **kwargs):
        requests[payload["query"]] = payload["max_results"]
        await asyncio.sleep(0)
        return response

    monkeypatch.setattr(tv, "tavily_search", search)
    backend = _backend(max_results=2)

    async def retrieve_all():
        return await asyncio.gather(
            backend.retrieve("one", max_results=1),
            backend.retrieve("three", max_results=3),
            backend.retrieve("default"),
        )

    batches = asyncio.run(retrieve_all())

    assert requests == {"one": 1, "three": 3, "default": 2}
    assert [len(documents) for documents in batches] == [1, 3, 2]
    assert [documents[0].query for documents in batches] == ["one", "three", "default"]
    assert backend.max_results == 2
    assert backend._build_payload("still default")["max_results"] == 2


def test_retrieve_without_key_fails_before_silent_closed_world_run(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    backend = _backend(api_key=None)
    with pytest.raises(RuntimeError, match="needs Tavily"):
        asyncio.run(backend.retrieve(ConstructedQuery(query="anything")))


def test_retrieve_empty_query_returns_empty():
    backend = _backend()
    docs = asyncio.run(backend.retrieve(ConstructedQuery(query="   ")))
    assert docs == []


def test_retrieve_accepts_plain_string(monkeypatch):
    async def fake_search(payload, **kw):
        assert payload["query"] == "plain string query"
        return {"results": []}

    monkeypatch.setattr(tv, "tavily_search", fake_search)
    docs = asyncio.run(_backend().retrieve("plain string query"))
    assert docs == []


@pytest.mark.integration
def test_live_search():
    """Hits the real Tavily API; skipped unless TAVILY_API_KEY is set."""
    if not os.environ.get("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set")
    backend = _backend(api_key=None, top_k=3)  # picks up the env key
    docs = asyncio.run(
        backend.retrieve(ConstructedQuery(query="circle packing 26 circles maximize sum of radii"))
    )
    assert docs, "expected at least one document from the live API"
    assert all(d.source == "tavily" for d in docs)
    assert any(d.url for d in docs)
