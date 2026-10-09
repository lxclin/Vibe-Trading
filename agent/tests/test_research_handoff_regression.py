"""Executed team evidence survives transport, compaction and host validation."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.equity_research import equity_research_input_issues
from src.agent.grounding.research_handoff import (
    MAX_RECEIPTS, capture_receipt, merge_receipts, scoped_receipts, swarm_context_result,
)

SYMBOL = "601899.SH"
NOW = datetime.now(ZoneInfo("Asia/Shanghai"))
PERIOD = f"{NOW.year}-06-30"


def receipt(tool, args, payload, call_id="original", success=True, symbols=(SYMBOL,)):
    return capture_receipt(tool, args, json.dumps(payload), call_id, success, symbols)


def financial(statement, values, call_id=None):
    return receipt("get_financial_statements", {"code": SYMBOL, "statement": statement, "period": "quarter"},
                   {"ok": True, "source": "eastmoney", "statement": statement, "period": "quarter",
                    "data": {SYMBOL: {"periods": [{"REPORT_DATE": PERIOD, **values}]}}},
                   call_id or statement)


def quote(call_id="quote", stamp=None, price=30, pe=12):
    return receipt("get_a_share_valuation", {"code": SYMBOL},
                   {"ok": True, "source": "eastmoney_quote", "data": {
                       "symbol": SYMBOL, "as_of": stamp or NOW.isoformat(),
                       "last_price": price, "pe_ttm": pe, "pb": 4, "price_adjustment": "raw",
                   }}, call_id)


def host(tmp_path):
    return GroundingLedger(run_dir=tmp_path, user_message=f"使用investment_committee对{SYMBOL}做多还是做空")


def test_swarm_replays_income_and_quote_without_requiring_duplicate_reads(tmp_path):
    ledger = host(tmp_path)
    receipts = [quote(), financial("indicators", {"PARENTNETPROFIT": 100, "KCFJCXSYJLR": 95, "EPSJB": 1}),
                financial("cashflow", {"NETCASH_OPERATE": 120, "CONSTRUCT_LONG_ASSET": 20}),
                financial("income", {"PARENT_NETPROFIT": 100, "OPERATE_INCOME": 500, "INVEST_INCOME": 2})]
    ledger.ingest_tool_result(tool_name="run_swarm", arguments={}, call_id="team",
                             result=json.dumps({"status": "completed", "research_receipts": receipts,
                                                "final_report": "Claimed market price 99999"}), success=True)
    sheet = ledger.equity_research_worksheet()["companies"][0]
    assert sheet["research_progress"]["income_statement"] == "observed_analysis_pending"
    assert sheet["valuation_snapshot"]["fields"]["pe_ttm"]["value"] == 12
    assert sheet["valuation_snapshot"]["fields"]["last_price"]["ref"] == "quote::data.last_price"
    issues = equity_research_input_issues(ledger._evidence, {SYMBOL}, ledger._decision_tool_attempts)
    assert not any(issue["code"] == "decision_income_not_checked" for issue in issues)
    assert not any(record.value == 99999 for record in ledger._evidence)
    assert {record.call_id for record in ledger._evidence} == {"quote", "indicators", "cashflow", "income"}
    saved = json.loads((tmp_path / "artifacts/grounding_evidence.json").read_text())
    assert saved["research_handoffs"][0]["source_call_id"] == "team"


def test_receipts_are_deduplicated_and_failures_remain_attempts(tmp_path):
    ledger = host(tmp_path)
    failed = receipt("get_financial_statements", {"code": SYMBOL, "statement": "income", "period": "quarter"},
                     {"ok": False, "error": "offline"}, "failed", False)
    for _ in range(2):
        ledger.ingest_research_handoff([quote(), failed], source_call_id="team")
    assert len(ledger._decision_tool_attempts) == 2
    assert ledger.equity_research_worksheet()["companies"][0]["research_progress"]["income_statement"] == "unavailable"
    assert len(ledger._research_handoffs) == 1


@pytest.mark.parametrize("mutate", [
    lambda item: item.update(tool="run_swarm"),
    lambda item: item.update(tool="read_file"),
    lambda item: item.update(scope_symbols=["600519.SH"]),
    lambda item: item.update(scope_symbols=[]),
    lambda item: item.update(success="true"),
    lambda item: item.update(result="not JSON"),
    lambda item: item.update(result='{"ok":false}'),
    lambda item: item["arguments"].update(code="600519.SH"),
])
def test_invalid_or_out_of_scope_receipts_are_rejected(mutate):
    item = quote()
    mutate(item)
    assert scoped_receipts([item], {SYMBOL}) == []


def test_company_quote_scope_is_narrowed_after_peer_resolution():
    item = receipt("get_a_share_valuation", {"code": SYMBOL}, {"ok": True}, symbols=(SYMBOL, "600519.SH"))
    assert item["scope_symbols"] == [SYMBOL]
    assert len(scoped_receipts([item], {SYMBOL})) == 1


def test_name_resolution_does_not_create_a_host_identity():
    item = receipt("search_symbol", {"query": "紫金矿业"}, {"ok": True})
    assert scoped_receipts([item], {SYMBOL}) == []


def test_packet_cap_does_not_duplicate_original_call_ids():
    items = [quote(str(index)) for index in range(MAX_RECEIPTS + 10)]
    merged = merge_receipts([items[0]], items)
    assert len(merged) == MAX_RECEIPTS
    assert len({item["call_id"] for item in merged}) == MAX_RECEIPTS


def test_late_arriving_old_quote_does_not_overwrite_recent_snapshot(tmp_path):
    ledger = host(tmp_path)
    day = NOW.date().isoformat()
    ledger.ingest_research_handoff([
        quote("recent", f"{day}T11:00:00+08:00", 31, None),
        quote("old", f"{day}T10:00:00+08:00", 30, 12),
    ], source_call_id="team")
    snapshot = ledger.equity_research_worksheet()["companies"][0]["valuation_snapshot"]["fields"]
    assert snapshot["last_price"]["value"] == 31
    assert "pe_ttm" not in snapshot


def test_context_view_keeps_full_typical_report_without_duplicate_receipt_bodies():
    report = "完整研究结论、估值解释和字段引用。" * 650
    result = swarm_context_result(json.dumps({"status": "completed", "final_report": report,
                                              "tasks": [{"id": "evidence", "summary": "duplicate" * 10000}],
                                              "research_receipts": [quote()]}))
    view = json.loads(result)
    assert view["final_report"] == report
    assert not view["report_truncated"]
    assert "research_receipts" not in view and "duplicate" not in result


def test_context_view_explicitly_marks_oversized_report():
    view = json.loads(swarm_context_result(json.dumps({"final_report": "x" * 23000})))
    assert view["report_truncated"] and len(view["final_report"]) == 22000


def test_worker_keeps_original_receipt_before_context_truncation(monkeypatch, tmp_path):
    from src.config.limits import TOOL_RESULT_LIMIT
    from src.providers.chat import LLMResponse, ToolCallRequest
    from src.swarm.models import SwarmAgentSpec, SwarmTask
    from src.swarm import worker
    from tests.test_swarm_worker_tool_result_truncation import _ScriptedChatLLM, FINAL_TEXT

    original = quote("worker-quote")
    payload = json.loads(original["result"])
    payload["description"] = "x" * (TOOL_RESULT_LIMIT + 100)
    raw = json.dumps(payload)

    class Registry:
        def get_definitions(self):
            return [{"type": "function", "function": {"name": "get_a_share_valuation", "parameters": {}}}]

        def get(self, name):
            return None

        def execute(self, name, args):
            return raw

    llm = _ScriptedChatLLM([
        LLMResponse(tool_calls=[ToolCallRequest(id="worker-quote", name="get_a_share_valuation", arguments={"code": SYMBOL})]),
        LLMResponse(content=FINAL_TEXT),
    ])
    monkeypatch.setattr(worker, "build_swarm_registry", lambda *args, **kwargs: Registry())
    monkeypatch.setattr(worker, "ChatLLM", llm)
    result = worker.run_worker(
        agent_spec=SwarmAgentSpec(id="analyst", role="Analyst", system_prompt="Research",
                                 tools=["get_a_share_valuation"], research_worksheet=True, max_iterations=3),
        task=SwarmTask(id="t1", agent_id="analyst", prompt_template=f"Research {SYMBOL}"),
        upstream_summaries={}, user_vars={}, run_dir=tmp_path / "worker",
    )
    assert len(result.research_receipts) == 1
    assert result.research_receipts[0]["result"] == raw
    assert any("[TRUNCATED:" in message.get("content", "")
               for message in llm.received_messages[-1] if message.get("role") == "tool")
    ledger = host(tmp_path / "host")
    ledger.ingest_research_handoff(result.research_receipts, source_call_id="worker")
    assert ledger.equity_research_worksheet()["companies"][0]["valuation_snapshot"]["fields"]["pe_ttm"]["value"] == 12
