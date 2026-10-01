"""Stable SearchStore IDs, legacy migration and read-only alias lookup."""

import copy
import json
from dataclasses import asdict

import pytest

from skydiscover.evoduet.retrieval.base_retrieval import (
    SearchImpact,
    SearchRecord,
    SearchResult,
    SearchResultAssessment,
    SearchStore,
)


def document(provider_id="", url="", **kwargs):
    return SearchResult(
        id=provider_id,
        url=url,
        content="query-relevant snippet",
        raw_content="stored body",
        **kwargs,
    )


def test_add_assigns_short_ids_without_replacing_objects_or_provider_fields():
    store = SearchStore()
    first = document("provider-a", "https://example.org/a", metadata={"raw": "provider data"})
    second = document("provider-b", "https://example.org/b")
    before = asdict(first)

    record = store.add(None, [first, second])

    assert record.search_results[0] is first
    assert [item.search_document_id for item in record.search_results] == [
        "doc_000001",
        "doc_000002",
    ]
    after = asdict(first)
    after.pop("search_document_id")
    before.pop("search_document_id")
    assert after == before
    assert store.state_dict()["next_document_id"] == 3


def test_exact_url_is_identity_across_retrievals_and_legacy_aliases():
    original = document("provider-old", "https://example.org/a", query="first search")
    newer = document("provider-new", "https://example.org/a", query="second search")
    store = SearchStore(records=[SearchRecord(search_results=[original])])
    store.add(None, [newer])

    assert original.search_document_id == newer.search_document_id == "doc_000001"
    assert store.lookup_documents(["doc_000001"])[0] is newer
    assert store.lookup_documents(["provider-old"])[0] is newer
    assert store.lookup_documents(["provider-new"])[0] is newer
    assert newer.query == "second search"
    assert store.state_dict()["next_document_id"] == 2


def test_url_identity_is_not_normalized_and_is_separate_from_provider_identity():
    urls = ["https://EXAMPLE.org/A%2Fb", "https://example.org/A%2Fb", "https://EXAMPLE.org/A/b"]
    members = [document("shared-provider-id", url) for url in urls]
    members.append(document(urls[0]))
    store = SearchStore(records=[SearchRecord(search_results=members)])

    assert len({item.search_document_id for item in members}) == 4
    assert store.lookup_documents(["shared-provider-id"])[0] is members[2]
    assert store.lookup_documents([urls[0]])[0] is members[3]


def test_missing_urls_use_provider_id_and_anonymous_documents_are_distinct():
    first, repeated = document("provider"), document("provider")
    anonymous_a, anonymous_b = document(), document()
    store = SearchStore()
    store.add(None, [first, repeated, anonymous_a, anonymous_a, anonymous_b])

    assert first.search_document_id == repeated.search_document_id == "doc_000001"
    assert anonymous_a.search_document_id == "doc_000002"
    assert anonymous_b.search_document_id == "doc_000003"
    assert store.state_dict()["next_document_id"] == 4
    assert store.lookup_documents(["doc_000002", "doc_000003"]) == [anonymous_a, anonymous_b]


@pytest.mark.parametrize("anonymous", [False, True])
def test_lookup_reuse_record_preserves_id_and_returns_latest_copy(anonymous):
    original = document() if anonymous else document("provider", "https://example.org/a")
    store = SearchStore()
    store.add(None, [original], iteration=1)
    reused = copy.deepcopy(store.lookup_documents([original.search_document_id])[0])
    reused.content = "new prompt body"
    record = store.add(None, [reused], operation="look-up", iteration=2, target_score=0.2)

    assert record.search_results[0] is reused
    assert reused.search_document_id == original.search_document_id == "doc_000001"
    assert store.lookup_documents([original.search_document_id])[0] is reused
    assert store.state_dict()["next_document_id"] == 2


def test_lookup_deduplicates_resolved_aliases_in_requested_order():
    first, second = document("first"), document("second")
    store = SearchStore(records=[SearchRecord(search_results=[first, second])])

    assert store.lookup_documents(
        ["second", "doc_000001", "doc_000002", "first", "second", "doc_000001"]
    ) == [second, first]


def test_short_id_has_priority_over_a_colliding_legacy_provider_id():
    first = document("ordinary-provider")
    legacy = document("doc_000001")
    store = SearchStore(records=[SearchRecord(search_results=[first, legacy])])

    assert store.lookup_documents(["doc_000001"])[0] is first
    assert store.lookup_documents(["doc_000002"])[0] is legacy


def test_unknown_ids_are_not_inferred_from_urls_and_lookup_never_mutates(tmp_path, monkeypatch):
    item = document("provider", "https://example.org/document")
    store = SearchStore(output_dir=tmp_path)
    store.add(None, [item], iteration=3, target_score=0.5)
    before = json.dumps(store.state_dict(), sort_keys=True)
    disk_before = store.path.read_bytes()

    def unexpected_mutation(*args, **kwargs):
        raise AssertionError("Lookup must not register, persist or stage documents")

    monkeypatch.setattr(store, "_register_document_ids", unexpected_mutation)
    monkeypatch.setattr(store, "_persist", unexpected_mutation)
    assert (
        store.lookup_documents(["https://example.org/document", "doc_999999", "DOC_000001"]) == []
    )
    assert store.lookup_documents(["doc_000001", "provider"]) == [item]
    assert json.dumps(store.state_dict(), sort_keys=True) == before
    assert store.path.read_bytes() == disk_before


def test_existing_ids_are_reserved_before_assigning_earlier_legacy_rows():
    legacy = document("legacy")
    existing = document("known", search_document_id="doc_000007")
    same_existing = document("known")
    store = SearchStore(records=[SearchRecord(search_results=[legacy, existing, same_existing])])

    assert existing.search_document_id == same_existing.search_document_id == "doc_000007"
    assert legacy.search_document_id == "doc_000008"
    assert store.state_dict()["next_document_id"] == 9


def test_new_batch_reserves_all_explicit_ids_before_allocation():
    store = SearchStore()
    legacy, assigned = document("legacy"), document("assigned", search_document_id="doc_000001")
    store.add(None, [legacy, assigned])
    assert assigned.search_document_id == "doc_000001"
    assert legacy.search_document_id == "doc_000002"


def test_checkpoint_preserves_ids_pending_credit_and_next_counter():
    store = SearchStore()
    item = document("provider", "https://example.org/a")
    store.add(None, [item], iteration=3, target_program_id="parent", target_score=0.4)
    state = json.loads(json.dumps(store.state_dict()))
    state["next_document_id"] = 50
    restored = SearchStore()
    restored.load_state_dict(state)

    assert restored.lookup_documents(["doc_000001"])[0].id == "provider"
    assert restored.state_dict() == state
    assert restored._pending[3]["results"][0] is restored.records[0].search_results[0]
    new = document("new-provider")
    restored.add(None, [new])
    assert new.search_document_id == "doc_000050"
    assert restored.state_dict()["next_document_id"] == 51
    restored.record_usage(3, score=0.7)
    assert restored.records[0].impacts[0].delta == pytest.approx(0.3)


def test_checkpoint_without_document_ids_is_rejected():
    provider_url = "https://example.org/legacy"
    record = SearchRecord(
        search_results=[
            document(provider_url, provider_url, visit_count=2, evolution_scores=[0.7])
        ],
        assessments=[SearchResultAssessment(document_ref=provider_url, relevance=0.9)],
        impacts=[SearchImpact(target_score=0.2, result_score=0.7, delta=0.5)],
    )
    serialized = asdict(record)
    serialized["search_results"][0].pop("search_document_id")
    state = {"records": [serialized], "pending": {}, "next_document_id": 2}
    before = copy.deepcopy(state)
    store = SearchStore()
    with pytest.raises(ValueError, match="SearchResult checkpoint fields"):
        store.load_state_dict(state)

    assert state == before
    assert store.records == []


def test_checkpoint_reload_keeps_assigned_ids_and_counter(tmp_path):
    store = SearchStore(output_dir=tmp_path)
    store.add(None, [document("one"), document("two")])
    loaded = SearchStore()
    loaded.load_state_dict(json.loads(json.dumps(store.state_dict())))
    assert [item.search_document_id for item in loaded.documents()] == ["doc_000001", "doc_000002"]
    newer = document("three")
    loaded.add(None, [newer])
    assert newer.search_document_id == "doc_000003"


@pytest.mark.parametrize("counter", [1, 0, -1, True, "10", None])
def test_invalid_checkpoint_counter_is_rejected(counter):
    item = document("existing", search_document_id="doc_000020")
    state = {
        "records": [asdict(SearchRecord(search_results=[item]))],
        "pending": {},
        "next_document_id": counter,
    }
    restored = SearchStore()
    with pytest.raises(ValueError, match="next_document_id"):
        restored.load_state_dict(state)
    assert restored.records == []


@pytest.mark.parametrize("same_identity", [False, True])
def test_conflicting_persisted_ids_fail_before_mutating_store_or_incoming_documents(same_identity):
    existing = document("existing", "https://example.org/a", search_document_id="doc_000001")
    store = SearchStore(records=[SearchRecord(search_results=[existing])])
    before = store.state_dict()
    fresh = document("fresh")
    conflicting = (
        document("other", "https://example.org/a", search_document_id="doc_000002")
        if same_identity
        else document("other", "https://example.org/b", search_document_id="doc_000001")
    )

    with pytest.raises(ValueError, match="search_document_id"):
        store.add(None, [fresh, conflicting], iteration=5)

    assert fresh.search_document_id == ""
    assert store.state_dict() == before
    assert store.lookup_documents(["doc_000001"])[0] is existing


def test_invalid_checkpoint_id_conflict_does_not_replace_existing_store():
    store = SearchStore(records=[SearchRecord(search_results=[document("original")])])
    before = store.state_dict()
    conflicting = [
        document("one", "url-one", search_document_id="doc_000004"),
        document("two", "url-two", search_document_id="doc_000004"),
    ]
    state = {
        "records": [asdict(SearchRecord(search_results=conflicting))],
        "pending": {},
        "next_document_id": 5,
    }
    with pytest.raises(ValueError, match="search_document_id"):
        store.load_state_dict(state)
    assert store.state_dict() == before
