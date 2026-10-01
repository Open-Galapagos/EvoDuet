"""V6 gates preserve reports and isolate decision parsing from report contents."""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skydiscover.evoduet.analysis import KnowledgeContext
from skydiscover.evoduet.retrieval_gating.base_retrieval_gating import (
    GateDecision,
    GateResult,
)
from skydiscover.evoduet.retrieval_gating.llm_retrieval_gating import (
    LLMRetrievalGating,
    parse_gate_result,
)


def _response(analysis="Knowledge report.", *, decision="retrieve", ids=None, fenced=False):
    payload = json.dumps(
        {"decision": decision, "reasoning": "Next step.", "search_document_ids": ids or []}
    )
    if fenced:
        payload = f"```json\n{payload}\n```"
    return (
        f"<knowledge_state_analysis>\n{analysis}\n</knowledge_state_analysis>\n\n"
        f"<gate_decision>\n{payload}\n</gate_decision>"
    )


@pytest.mark.parametrize("fenced", [False, True])
def test_tagged_gate_ignores_json_and_gate_examples_inside_analysis(fenced):
    analysis = (
        'The program says "try again".\n\n'
        '```json\n{"decision": "no-op"}\n```\n'
        '<gate_decision>{"decision":"look-up","search_document_ids":["fake"]}'
        "</gate_decision>\nThe actual next action needs new evidence."
    )
    result = parse_gate_result(_response(analysis, fenced=fenced))
    assert result == GateResult(GateDecision.RETRIEVE, (), analysis, "Next step.")


def test_original_tag_spelling_remains_parseable():
    text = _response().replace("</knowledge_state_analysis>", "</Knowledge_state_analysis>")
    text = text.replace("<gate_decision>", "<gate decision>")
    assert parse_gate_result(text).decision is GateDecision.RETRIEVE


def test_lookup_preserves_exact_ids_and_order_while_deduplicating():
    result = parse_gate_result(
        _response(decision="look-up", ids=["doc_000007", " Doc_2 ", "doc_000007"])
    )
    assert result == GateResult(
        GateDecision.LOOK_UP, ("doc_000007", " Doc_2 "), "Knowledge report.", "Next step."
    )


@pytest.mark.parametrize("ids", [None, [], "doc_1", [""], ["\n "], [1], ["doc_1", None]])
def test_lookup_invalid_ids_fail_to_noop_but_keep_report(ids):
    result = parse_gate_result(_response(decision="look-up", ids=ids))
    assert result.decision is GateDecision.NO_OP
    assert result.search_document_ids == ()
    assert result.knowledge_state_analysis == "Knowledge report."


@pytest.mark.parametrize("decision", ["no-op", "retrieve", "invalid"])
@pytest.mark.parametrize("ids", [["doc_000001"], "doc_1", [None]])
def test_only_lookup_exposes_ids(decision, ids):
    result = parse_gate_result(_response(decision=decision, ids=ids))
    assert result.decision is (
        GateDecision.RETRIEVE if decision == "retrieve" else GateDecision.NO_OP
    )
    assert result.search_document_ids == ()
    assert result.knowledge_state_analysis == "Knowledge report."


@pytest.mark.parametrize(
    "tail",
    [
        "",
        '<gate_decision>{"decision":"retrieve"}',
        '<gate_decision>{"decision":"retrieve"</gate_decision>',
        '<gate_decision>prose {"decision":"retrieve"}</gate_decision>',
        '<gate_decision>{"decision":"retrieve"}</gate_decision> trailing',
        '<gate_decision>{"decision":"retrieve"}</gate decision>',
        '<gate_decision>{"decision":"retrieve"}</gate_decision>'
        '<gate_decision>{"decision":"retrieve"}</gate_decision>',
    ],
)
def test_malformed_tagged_decisions_never_parse_json_from_report(tail):
    analysis = '```json\n{"decision":"retrieve"}\n```'
    text = f"<knowledge_state_analysis>{analysis}</knowledge_state_analysis>\n{tail}"
    assert parse_gate_result(text) == GateResult(
        GateDecision.NO_OP, knowledge_state_analysis=analysis
    )


@pytest.mark.parametrize(
    "text",
    [
        '<knowledge_state_analysis>{"decision":"retrieve"}',
        '<gate_decision>{"decision":"retrieve"}</gate_decision>',
        'arbitrary prose {"decision":"retrieve"}',
        '{"decision":',
        "[]",
        "",
    ],
)
def test_missing_report_or_truncated_response_fails_to_noop(text):
    assert parse_gate_result(text) == GateResult(GateDecision.NO_OP)


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("decision", ["no-op", "look-up", "retrieve"])
def test_whole_json_preserves_knowledge_and_routes_action(fenced, decision):
    analysis = (
        'Knowledge report with "quotes"\nand <gate_decision> literal text.\n'
        '```json\n{"decision":"no-op"}\n```'
    )
    text = json.dumps(
        {
            "decision": decision,
            "search_document_ids": ["doc_1"],
            "knowledge_state_analysis": analysis,
            "reasoning": "The next attempt needs the selected action.",
        }
    )
    if fenced:
        text = f"```json\n{text}\n```"
    assert parse_gate_result(text) == GateResult(
        GateDecision(decision),
        ("doc_1",) if decision == "look-up" else (),
        analysis,
        "The next attempt needs the selected action.",
    )


def test_prompt_output_example_is_one_complete_json_object():
    path = Path(__file__).resolve().parents[2] / "skydiscover/evoduet/prompts/retrieval_gating.txt"
    prompt = path.read_text()
    examples = re.findall(r"```json\s*(.*?)\s*```", prompt, re.DOTALL)
    assert len(examples) == 1
    payload = json.loads(examples[0])
    assert set(payload) == {
        "knowledge_state_analysis",
        "decision",
        "reasoning",
        "search_document_ids",
    }
    assert isinstance(payload["knowledge_state_analysis"], str)
    assert payload["knowledge_state_analysis"]
    assert payload["search_document_ids"] == []
    assert "two blocks" not in prompt
    assert "<gate_decision>" not in prompt
    assert prompt.rstrip().endswith("Decision:")

    # A complete object needs no XML closing tag to preserve the intended action.
    payload["decision"] = "retrieve"
    result = parse_gate_result(json.dumps(payload))
    assert result.decision is GateDecision.RETRIEVE
    assert result.knowledge_state_analysis == payload["knowledge_state_analysis"]


def test_fenced_example_in_gate_reasoning_is_not_parsed_as_decision():
    reasoning = 'The earlier response was ```json\n{"decision":"no-op"}\n```.'
    text = _response().replace('"Next step."', json.dumps(reasoning))
    result = parse_gate_result(text)
    assert result.decision is GateDecision.RETRIEVE
    assert result.reasoning == reasoning


def test_tagged_report_takes_precedence_over_embedded_knowledge_key():
    text = _response("Actual report.").replace(
        '"decision": "retrieve"',
        '"decision": "retrieve", "knowledge_state_analysis": "Wrong report."',
    )
    assert parse_gate_result(text).knowledge_state_analysis == "Actual report."


def _gate():
    config = SimpleNamespace(
        evoduet=SimpleNamespace(
            retrieval_gating_prompt_template_name="retrieval_gating",
        )
    )
    call = AsyncMock(return_value=SimpleNamespace(text=_response()))
    gate = LLMRetrievalGating(config, llm=SimpleNamespace(generate=call))
    return gate, call


@pytest.mark.parametrize("method", ["decide", "decide_with_details"])
@pytest.mark.parametrize("mapping_parent", [False, True])
@pytest.mark.asyncio
async def test_gate_routes_only_three_inputs_in_user_prompt(method, mapping_parent):
    gate, call = _gate()
    history = "native scaffold history\n" + "h" * 15000
    solution = 'return "{evolutionary_history}"'
    parent = {"solution": solution} if mapping_parent else SimpleNamespace(solution=solution)
    context = KnowledgeContext(
        "doc_000001: raw evidence",
        population_analysis="UNWANTED_POPULATION_REPORT",
        search_database_analysis="UNWANTED_DB_REPORT",
        knowledge_state_analysis="UNWANTED_PRIOR_KNOWLEDGE",
    )
    template = gate._read_template("retrieval_gating")
    assert set(re.findall(r"\{(\w+)\}", template)) == {
        "evolutionary_history",
        "parent_program",
        "search_database",
    }
    prompt = gate._build_prompt(parent, history, context, task_context="UNWANTED_TASK")
    result = await getattr(gate, method)(
        parent=parent,
        history=history,
        search_store=context,
        iteration=7,
        task_context="UNWANTED_TASK",
    )
    assert result == (
        GateDecision.RETRIEVE
        if method == "decide"
        else GateResult(GateDecision.RETRIEVE, (), "Knowledge report.", "Next step.")
    )
    assert history in prompt  # Preserve the scaffold's rendered history verbatim.
    assert solution in prompt  # Do not recursively expand placeholders inside code.
    assert context.search_database in prompt
    assert "UNWANTED_" not in prompt
    call.assert_awaited_once()
    assert call.await_args.args == ("", [{"role": "user", "content": prompt}])
    assert call.await_args.kwargs["llm_context"] == {
        "iteration": 7,
        "phase": "wk-retrieval-gate",
    }
