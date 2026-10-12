"""Saved real drafts and receipts, not a simulated successful model answer."""
import json
from datetime import datetime
from pathlib import Path

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.decision import decision_issues, requested_horizon
from src.agent.grounding.figures import parse_figures_block
from src.agent.grounding.query_budget import SessionResearchCache
from src.swarm.worker import _classify_deliverable, _uses_inherited_evidence

FIXTURES = Path(__file__).parent / "fixtures/research_20261010"


@pytest.fixture(autouse=True)
def saved_run_clock(monkeypatch):
    class SavedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 10, 12, tzinfo=tz)
    for module in ("decision", "equity_research", "policies"):
        monkeypatch.setattr(f"src.agent.grounding.{module}.datetime", SavedDateTime)


def replay(tmp_path, name, index):
    data = json.loads((FIXTURES / (name + ".json")).read_text())
    draft = data["drafts"][index]
    symbol = {"etf": "518850.SH", "bank": "600036.SH", "zijin": "601899.SH"}[name]
    question = (f"{symbol}黄金ETF现在是否值得买入，期限3—6个月？" if name == "etf"
                else f"{symbol}请比较同行估值和盈利质量，分析6—12个月买入价值")
    ledger = GroundingLedger(run_dir=tmp_path, user_message=question)
    for item in data["receipts"][draft["receipt_start"]:draft["receipt_count"]]:
        ledger.ingest_tool_result(tool_name=item["tool"], arguments=item["arguments"], result=item["result"],
                                  call_id=item["call_id"], success=item["success"])
    return ledger, draft["content"]


@pytest.mark.parametrize("index", range(4))
def test_saved_etf_drafts_release_without_cutting_analysis(tmp_path, index):
    ledger, content = replay(tmp_path, "etf", index)
    check = ledger.validate_final_answer(content)
    repaired = ledger.repair_references(content, check) if not check.valid else content
    assert repaired is not None, check.issues
    result = ledger.revalidate(repaired)
    assert result.valid, result.issues
    assert "黄金主要驱动" in result.released_text
    assert "条件性偏积极" in result.released_text


@pytest.mark.parametrize("duration", ["3—6个月", "沿用3—6个月", "6-12月", "半年"])
def test_explicit_horizon_survives_chinese_month_classifier(duration):
    text = f"结论：研究未完成；期限：{duration}；信心：低。主要依据：数据缺失。改变判断的条件：比较净值。"
    assert not any(i["code"] == "decision_horizon_missing" for i in decision_issues(text, [], set()))


def test_horizon_inherits_only_user_referential_request():
    history = [{"role": "user", "content": "黄金ETF持有3—6个月，能买吗？"},
               {"role": "assistant", "content": "期限6—12个月"}]
    assert requested_horizon("那现在能买吗？请沿用刚才的期限", history) == "3—6个月"
    assert requested_horizon("招商银行能买吗？", history) is None


def test_quantity_parser_does_not_accept_arbitrary_suffix():
    assert not parse_figures_block("```figures\n4,256吨 | cited | 黄金持有量 | https://example.org\n```").malformed
    assert parse_figures_block("```figures\n39吨错误 | cited | 数量 | https://example.org\n```").malformed


def test_formula_exemption_does_not_release_standalone_percentage(tmp_path):
    ledger, content = replay(tmp_path, "etf", 1)
    content = content.replace("100 | count | 百分比换算系数 | financial_rigor\n", "")
    check = ledger.validate_final_answer(content + "\n实际收益率为100%。")
    assert not check.valid
    assert any(i.get("value") == "100%" for i in check.issues)


def test_inherited_contract_requires_cited_observations(tmp_path):
    ledger, _ = replay(tmp_path, "etf", 3)
    observed = next(r for r in ledger._evidence if r.value is not None)
    ledger._imported_research_calls.add(observed.call_id)
    summary = "已分析可核验的上游证据及风险边界。" * 50 + observed.call_id
    assert _uses_inherited_evidence(summary, ledger)
    assert _classify_deliverable(summary, is_data_agent=True, report_written=False, data_tool_calls=0,
                                 inherited_evidence_used=True) is None
    assert not _uses_inherited_evidence("上游已有数据。" * 150, ledger)
    assert not _uses_inherited_evidence(observed.call_id, ledger)


def test_session_cache_retains_original_calls_and_rejects_other_subjects(tmp_path):
    path = tmp_path / "cache.json"
    args = {"url": "https://example.org/nav", "offset": 0}
    cache = SessionResearchCache(path, now=1000, enabled=False)
    cache.record("read_url", args, '{"ok":true,"web_financial":{"unit_nav":8.6915}}', "nav-original", {"518850.SH"})
    assert cache.lookup("read_url", args, {"518850.SH"}) is None
    cache = SessionResearchCache(path, now=1100, enabled=True)
    assert cache.lookup("read_url", args, {"518850.SH"})["call_id"] == "nav-original"
    assert cache.lookup("read_url", args, {"600036.SH"}) is None
    assert cache.lookup("read_url", {**args, "offset": 8000}, {"518850.SH"}) is None
    assert cache.lookup("read_url", {**args, "fresh": True}, {"518850.SH"}) is None
    assert SessionResearchCache(path, now=2801, enabled=True).lookup("read_url", args, {"518850.SH"}) is None
    cache.record("get_a_share_valuation", {"code": "518850.SH"}, '{"ok":true}', "quote", {"518850.SH"})
    assert cache.lookup("get_a_share_valuation", {"code": "518850.SH"}, {"518850.SH"}) is None


def test_bank_wrong_period_eps_still_rejected_but_ratio_rounding_survives(tmp_path):
    ledger, text = replay(tmp_path, "bank", 1)
    issues = ledger.validate_final_answer(text).issues
    assert not any(i.get("value") == "−0.25%" for i in issues)
    quarter = next(r for r in ledger._evidence if r.symbol == "600036.SH"
                   and r.field.endswith(".EPSJB") and r.timestamp.startswith("2026-06-30"))
    rows = text.splitlines()
    for i, row in enumerate(rows):
        if row.startswith("5.70 | observed"):
            rows[i] = row.rsplit(" | ", 1)[0] + f" | {quarter.call_id}::{quarter.field}"
    issues = ledger.revalidate("\n".join(rows)).issues
    assert any(i.get("value") == "5.70" and i.get("reason") == "not_in_referenced_call" for i in issues)
    wrong_sign = text.replace("−0.25%", "+0.25%")
    assert any(i.get("value") == "+0.25%" for i in ledger.revalidate(wrong_sign).issues)


def test_zijin_unresolved_ttm_sources_stay_rejected(tmp_path):
    ledger, text = replay(tmp_path, "zijin", 1)
    assert any(i.get("reason") == "additive_operand_not_observed" for i in ledger.validate_final_answer(text).issues)


def test_etf_fallback_keeps_user_horizon_and_fund_framework(tmp_path):
    data = json.loads((FIXTURES / "etf.json").read_text())
    ledger = GroundingLedger(run_dir=tmp_path, user_message="那现在能买吗？请沿用刚才的期限", history=[
        {"role": "user", "content": "518850.SH黄金ETF持有3—6个月，能买吗？"}])
    for item in data["receipts"][:data["drafts"][0]["receipt_count"]]:
        ledger.ingest_tool_result(tool_name=item["tool"], arguments=item["arguments"], result=item["result"],
                                  call_id=item["call_id"], success=item["success"])
    text = ledger.safe_fallback()
    assert "3—6个月" in text and "官方净值" in text
    assert "财报" not in text and "六至十二" not in text


def test_loop_reuses_prior_session_receipt_without_second_execution(tmp_path):
    from types import SimpleNamespace
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry
    from tests.test_agent_loop_dedup_arguments import _PagedTool, _drive

    tool = _PagedTool()
    registry = ToolRegistry()
    registry.register(tool)
    path = tmp_path / "session/research_cache.json"
    args = {"code": "600036.SH", "statement": "income", "period": "quarter"}
    for index in range(2):
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        agent = AgentLoop(registry=registry, llm=SimpleNamespace(), max_iterations=4,
                          event_callback=lambda *_: None)
        agent.memory.run_dir = str(run_dir)
        agent._grounding = GroundingLedger(run_dir=run_dir, user_message="600036.SH研究")
        agent._session_research_cache = SessionResearchCache(path, now=1000 + index, enabled=bool(index))
        messages, trace = _drive(agent, tool.name, run_dir, [args])
        if index:
            assert json.loads(messages[0]["content"])["_research_reuse"]["original_call_id"] == "call_1"
            assert any(item["type"] == "tool_result_reused" for item in trace)
    assert len(tool.calls) == 1


def test_real_escaped_upstream_call_index_satisfies_contract():
    from types import SimpleNamespace
    data = json.loads((FIXTURES / "bull_escaped_calls.json").read_text())
    ledger = SimpleNamespace(_evidence=[SimpleNamespace(**r) for r in data["evidence"]],
                             _imported_research_calls=set(data["imported_calls"]))
    assert _uses_inherited_evidence(data["summary"], ledger)
    assert _classify_deliverable(data["summary"], is_data_agent=True, report_written=False,
                                 data_tool_calls=0, inherited_evidence_used=True) is None


def test_new_bank_middle_draft_preserves_complete_analysis_with_period_metadata(tmp_path):
    data = json.loads((FIXTURES / "bank_next.json").read_text())
    draft = data["drafts"][1]
    ledger = GroundingLedger(run_dir=tmp_path, user_message="招商银行600036.SH现在是否值得买入，期限6—12个月，比较银行估值？")
    for item in data["receipts"][:draft["receipt_count"]]:
        ledger.ingest_tool_result(tool_name=item["tool"], arguments=item["arguments"], result=item["result"],
                                  call_id=item["call_id"], success=item["success"])
    check = ledger.validate_final_answer(draft["content"])
    assert check.valid, check.issues
    assert "公司质量" in check.released_text and "价格吸引力" in check.released_text


def test_session_cache_selects_latest_original_receipt(tmp_path):
    path = tmp_path / "cache.json"
    cache = SessionResearchCache(path, now=1000, enabled=True)
    args = {"url": "https://example.org/nav"}
    cache.record("read_url", args, '{"ok":true,"web_financial":{"unit_nav":1}}', "old", {"518850.SH"})
    cache.record("read_url", args, '{"ok":true,"web_financial":{"unit_nav":2}}', "new", {"518850.SH"})
    assert SessionResearchCache(path, now=1100, enabled=True).lookup("read_url", args, {"518850.SH"})["call_id"] == "new"


def test_fund_contract_uses_nav_and_price_without_company_financials():
    from types import SimpleNamespace
    records = [SimpleNamespace(status="observed", symbol="518850.SH", tool="get_a_share_valuation",
                               field="data.last_price", value=8.71),
               SimpleNamespace(status="observed", symbol="518850.SH", tool="read_url",
                               field="web_financial.unit_nav", value=8.6915)]
    text = "结论：暂不买入（等待）；期限：3—6个月；信心：低。主要依据：黄金实际利率上行。改变判断的条件：比较同步IOPV与市价，利率回落。"
    assert not decision_issues(text, records, {"518850.SH"})


def test_real_bank_provenance_round_can_be_repaired_locally(tmp_path):
    data = json.loads((FIXTURES / "bank_provenance.json").read_text())
    ledger = GroundingLedger(run_dir=tmp_path, user_message=(
        "招商银行600036.SH现在是否值得买入？持有期限6—12个月。比较同口径银行估值。"))
    for item in data["receipts"]:
        ledger.ingest_tool_result(tool_name=item["tool"], arguments=item["arguments"], result=item["result"],
                                  call_id=item["call_id"], success=item["success"])
    draft = data["draft"]
    check = ledger.validate_final_answer(draft)
    assert not check.valid
    assert {i["code"] for i in check.issues} == {"data_source_not_surfaced", "currency_not_surfaced"}
    repaired = ledger.repair_metadata(draft, check)
    assert repaired is not None and repaired.startswith(draft)
    assert ledger.revalidate(repaired).valid
    assert "tencent" in repaired and "CNY" in repaired
    assert ledger.validation_count == 1
