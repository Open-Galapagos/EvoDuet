"""Tests for query construction, prompt rendering, and retrieval routing."""

import pytest

from skydiscover.evoduet.query_construction import (
    ConstructedQuery,
    QueryConstruction,
)


def test_parse_query_bare_json():
    q = QueryConstruction._parse_query_sample(
        '{"query": "sparse attention", "keywords": ["attention", "sparsity"], '
        '"resources": ["paper"], "query_type": "technique"}'
    )
    assert q.query == "sparse attention"
    assert q.keywords == ["attention", "sparsity"]
    assert q.resources == ["paper"]
    assert q.query_type == "technique"


def test_keyword_only_query_has_bounded_search_text():
    q = ConstructedQuery(keywords=[str(index) * 100 for index in range(6)])

    assert q.query is None
    assert q.text.startswith("0" * 100)
    assert len(q.text) == 500


@pytest.mark.parametrize("fenced", [False, True])
def test_individual_sample_accepts_object_and_keyword_only_query(fenced):
    text = '{"keywords": ["feasible", "projection"], "resources": ["paper"]}'
    if fenced:
        text = f"```json\n{text}\n```"
    query = QueryConstruction._parse_query_sample(text)
    assert query.text == "feasible projection"
    assert query.resources == ["paper"]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "just search for projection methods",
        "[]",
        '[{"query": "one"}]',
        '[{"query": "one"}, {"query": "two"}]',
        '{"queries": [{"query": "one"}, {"query": "two"}]}',
        "{}",
        '{"query": "  ", "keywords": []}',
        '{"query": 2}',
        '{"query": NaN}',
        '{"query": "two", "query": "three"}',
        'Here is the query: {"query": "one"}',
        '```json\n[{"query": "one"}, {"query": "two"}]\n```',
        None,
        ["not text"],
    ],
)
def test_invalid_individual_sample_never_becomes_raw_search_text(response):
    with pytest.raises(ValueError, match="query sample"):
        QueryConstruction._parse_query_sample(response)
