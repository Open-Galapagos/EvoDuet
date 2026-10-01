"""Exact cached-document lookup and model-visible document identifiers."""

import json
import logging
from dataclasses import asdict

import pytest

from skydiscover.evoduet.retrieval.base_retrieval import (
    SearchRecord,
    SearchResult,
    SearchStore,
)


def doc(identifier, *, content="document summary", **fields):
    return SearchResult(
        id=identifier,
        title="Document title",
        content=content,
        raw_content="full document body",
        **fields,
    )


def rendered_ids(text):
    return [
        json.loads(line.strip().split(":", 1)[1])
        for line in text.splitlines()
        if line.strip().startswith("search_document_id:")
    ]


def test_lookup_preserves_request_order_and_deduplicates():
    first, second, third = doc("one"), doc("two"), doc("three")
    store = SearchStore(records=[SearchRecord(search_results=[first, second, third])])

    results = store.lookup_documents(["three", "one", "three", "two", "one"])

    assert results == [third, first, second]
    assert results[0] is third
    assert store.lookup_documents([]) == []


def test_lookup_uses_exact_opaque_ids_without_url_or_title_inference(caplog):
    identifier = ' https://EXAMPLE.org/A%2Fb?q="value"#frag '
    document = doc(identifier, url="https://canonical.example/document")
    numeric_string = doc("42")
    store = SearchStore(records=[SearchRecord(search_results=[document, numeric_string])])

    with caplog.at_level(logging.WARNING, logger="skydiscover.evoduet"):
        results = store.lookup_documents(
            [
                identifier.strip(),
                identifier.lower(),
                identifier.replace("%2F", "/"),
                document.url,
                document.title,
                42,
                True,
                None,
                ["42"],
                identifier,
                "42",
            ]
        )

    assert results == [document, numeric_string]
    assert "unknown search_document_id" in caplog.text
    assert "non-string document ID" in caplog.text


def test_lookup_rejects_a_bare_string_instead_of_interpreting_characters():
    store = SearchStore(records=[SearchRecord(search_results=[doc("a"), doc("b")])])
    with pytest.raises(TypeError):
        store.lookup_documents("ab")


def test_lookup_unknown_ids_are_skipped_and_logged_once_per_request(caplog):
    store = SearchStore(records=[SearchRecord(search_results=[doc("known")])])
    with caplog.at_level(logging.WARNING, logger="skydiscover.evoduet"):
        results = store.lookup_documents(["unknown", "known", "unknown", "also unknown"])
    assert [result.id for result in results] == ["known"]
    assert len(caplog.records) == 2


def test_lookup_latest_record_wins_without_ranking_by_relevance_or_iteration():
    old = doc("same", content="old", relevance_score=1.0, iteration=100)
    new = doc("same", content="new", relevance_score=0.1, iteration=2, query="source query")
    store = SearchStore(
        records=[
            SearchRecord(search_results=[old], iteration=100),
            SearchRecord(search_results=[new], iteration=2),
        ]
    )
    assert store.lookup_documents(["same"]) == [new]
    assert store.lookup_documents(["same"])[0].query == "source query"


def test_lookup_does_not_mutate_or_persist_store_or_stage_credit(tmp_path, monkeypatch):
    store = SearchStore(output_dir=tmp_path)
    document = doc("pending", visit_count=3, evolution_iterations=[1], evolution_scores=[0.8])
    store.add(None, [document], iteration=7, target_score=0.5)
    before = json.dumps(store.state_dict(), sort_keys=True)
    disk_before = store.path.read_bytes()

    def unexpected_persist():
        raise AssertionError("Lookup must not persist or mutate the store")

    monkeypatch.setattr(store, "_persist", unexpected_persist)
    assert store.lookup_documents(["pending", "unknown"])[0] is document
    assert store.lookup_documents(["pending"])[0] is document
    assert json.dumps(store.state_dict(), sort_keys=True) == before
    assert store.path.read_bytes() == disk_before
    assert list(store._pending) == [7]


def test_lookup_survives_checkpoint_reload(tmp_path):
    store = SearchStore(output_dir=tmp_path)
    store.add(None, [doc("same", content="old")], iteration=1)
    latest = doc("same", content="latest", query="source query", metadata={"source_field": 1})
    store.add(None, [latest, doc("other")], iteration=2, target_score=0.2)
    restored = SearchStore()
    restored.load_state_dict(json.loads(json.dumps(store.state_dict())))

    results = restored.lookup_documents(["other", "same"])
    assert [result.id for result in results] == ["other", "same"]
    assert asdict(results[1]) == asdict(latest)
