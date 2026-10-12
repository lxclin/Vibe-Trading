"""Replay the bank report that lost its analysis during quote recovery."""
import json
from datetime import datetime
from pathlib import Path

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.figures import parse_figures_block, strip_figures_block


@pytest.fixture
def bank(tmp_path, monkeypatch):
    class SavedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 12, 12, tzinfo=tz)

    for module in ("decision", "equity_research", "policies"):
        monkeypatch.setattr(f"src.agent.grounding.{module}.datetime", SavedDateTime)
    data = json.loads((Path(__file__).parent / "fixtures/research_20261012/bank_derivation.json").read_text())
    ledger = GroundingLedger(run_dir=tmp_path, user_message=
        "招商银行600036.SH现在是否值得买入？尚未持有，持有期限6—12个月，比较银行估值与盈利质量。")
    for receipt in data["receipts"]:
        ledger.ingest_tool_result(tool_name=receipt["tool"], arguments=receipt["arguments"],
                                  result=receipt["result"], call_id=receipt["call_id"], success=receipt["success"])
    return ledger, data["content"]


def test_missing_dividend_operand_does_not_refetch_existing_quote(bank):
    ledger, draft = bank
    check = ledger.validate_final_answer(draft)
    assert [(i["value"], i["role"], i["reason"]) for i in check.issues] == [
        ("4.75%", "derived", "no_evidence")]
    assert ledger.recovery_action(check) is None


def test_local_omission_preserves_every_other_character_and_passed_figure(bank):
    ledger, draft = bank
    check = ledger.validate_final_answer(draft)
    repaired = ledger.repair_isolated_derivation(draft, check)
    assert repaired is not None
    result = ledger.revalidate(repaired)
    assert result.valid, result.issues
    assert set(check.passed_figures) <= set(result.passed_figures)
    # Compare the entire report, including tables, citations and judgments.
    original_body = strip_figures_block(draft, parse_figures_block(draft))
    start, end = check.issues[0]["span"]
    assert result.released_text.startswith(original_body[:start] + "（略※）" + original_body[end:])
    assert "4.75%" not in result.released_text
    assert "证据缺口" in result.released_text


@pytest.mark.parametrize("damage", ["conflict", "citation", "checklist", "second_value"])
def test_local_omission_cannot_bypass_other_failures(bank, damage):
    ledger, draft = bank
    check = ledger.validate_final_answer(draft)
    extra = dict(check.issues[0])
    if damage == "conflict":
        extra.update(code="numeric_claim_conflict", reason="derivation_result_mismatch")
    elif damage == "citation":
        extra.update(code="numeric_claim_conflict", role="cited", reason="citation_not_visible")
    elif damage == "checklist":
        extra.update(code="decision_income_not_checked", value=None)
    else:
        extra.update(value="9.99%")
    check.issues.append(extra)
    assert ledger.repair_isolated_derivation(draft, check) is None


def test_local_omission_rechecks_full_report(bank):
    ledger, draft = bank
    check = ledger.validate_final_answer(draft)
    # The stale verdict cannot bless an additional unsupported claim.
    assert ledger.repair_isolated_derivation(draft + "\n历史收益率99.99%。", check) is None


def test_loop_delivers_original_research_without_another_model_or_tool_turn(bank, tmp_path, monkeypatch):
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry
    from src.agent.trace import TraceWriter
    from tests.test_grounding_release import _Response, _ScriptedLLM

    ledger, draft = bank
    monkeypatch.setattr("src.agent.loop.GroundingLedger", lambda **kwargs: ledger)
    repaired = ledger.repair_isolated_derivation(draft, ledger.revalidate(draft))
    assert repaired is not None
    expected = ledger.revalidate(repaired).released_text
    llm = _ScriptedLLM([_Response(content=draft)])
    agent = AgentLoop(registry=ToolRegistry(), llm=llm, max_iterations=5)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    result = agent.run("招商银行600036.SH现在是否值得买入？尚未持有，持有期限6—12个月。")
    assert result["status"] == "success"
    assert llm.calls == 1
    assert result["content"] == expected
    assert "4.75%" not in result["content"]
    events = TraceWriter.read(run_dir)
    assert any(e["type"] == "answer_derivation_omitted" for e in events)
    assert not any(e["type"] in {"grounding_recovery", "tool_call"} for e in events)
    costs = json.loads((run_dir / "research_cost.json").read_text())
    assert costs["events"]["answer_derivation_omitted"] == 1
