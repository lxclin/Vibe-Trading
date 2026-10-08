"""Regression coverage for quote fallback and auditable equity entry answers."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.decision import decision_coverage, decision_issues, is_single_buy_question
from src.agent.grounding.equity_research import (
    equity_entry_answer_issues, equity_research_input_issues, equity_worksheet, needs_entry_inputs,
)
from src.agent.grounding.figures import parse_figures_block
from src.tools import a_share_valuation_tool as quotes

SYMBOL = "601899.SH"


def _financials(ledger, call_id="call_financial|fc_financial"):
    ledger.ingest_tool_result(
        tool_name="get_financial_statements", arguments={"code": SYMBOL, "statement": "indicators", "period": "quarter"},
        result=json.dumps({"ok": True, "source": "eastmoney", "statement": "indicators", "period": "quarter",
                           "data": {SYMBOL: {"periods": [
                               {"REPORT_DATE": "2026-06-30", "PARENTNETPROFITTZ": 68.170288138556, "ROEJQ": 19.6},
                               {"REPORT_DATE": "2026-03-31", "PARENTNETPROFITTZ": 97.500066337394, "ROEJQ": 10.2},
                           ]}}}), call_id=call_id, success=True,
    )


@pytest.mark.parametrize("role", ["observed", "derived"])
def test_report_date_resolves_bare_field_and_rounded_role(tmp_path, role):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}财报数据")
    _financials(ledger)
    result = ledger.validate_final_answer(
        f"{SYMBOL}（东方财富，CNY），2026-06-30归母净利润同比增长68.17%。\n"
        f"```figures\n68.17% | {role} | {SYMBOL}，2026-06-30，四舍五入 | PARENTNETPROFITTZ\n```"
    )
    assert result.valid, result.issues


def test_saved_draft_style_rounding_does_not_need_arithmetic(tmp_path):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}财报数据")
    _financials(ledger)
    result = ledger.validate_final_answer(
        f"{SYMBOL}（东方财富，CNY），归母净利润同比增长68.17%。\n"
        "```figures\n68.17% | derived | 2026年上半年PARENTNETPROFITTZ返回68.170288138556%，四舍五入 "
        "| get_financial_statements::PARENTNETPROFITTZ\n```"
    )
    assert result.valid, result.issues


@pytest.mark.parametrize("value,ref,date", [
    ("69.17%", "PARENTNETPROFITTZ", "2026-06-30"),
    ("68.17%", "PARENTNETPROFITTZ", "2026-03-31"),
    ("68.17%", "MISSING", "2026-06-30"),
    ("68.17%", "fake_call::PARENTNETPROFITTZ", "2026-06-30"),
    ("-68.17%", "PARENTNETPROFITTZ", "2026-06-30"),
])
def test_rounding_repair_does_not_bless_wrong_values_or_refs(tmp_path, value, ref, date):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}财报数据")
    _financials(ledger)
    result = ledger.validate_final_answer(
        f"{SYMBOL}（东方财富，CNY）利润增长{value}。\n"
        f"```figures\n{value} | derived | {SYMBOL}，{date}，四舍五入 | {ref}\n```"
    )
    assert not result.valid


def test_rounding_role_repair_preserves_actual_arithmetic(tmp_path):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}财报数据")
    _financials(ledger)
    block = parse_figures_block(
        f"```figures\n68.17% | derived | {SYMBOL}，2026-06-30，四舍五入，68.170288 + 1 "
        "| get_financial_statements::PARENTNETPROFITTZ\n```"
    )
    assert ledger._normalize_provider_refs(block).declarations[0].role == "derived"


@pytest.mark.parametrize("tool,path,value", [
    ("get_a_share_valuation", "data.pe_ttm", 11.54),
    ("get_research_reports", "data.reports[0].eps_forecast.this_year", 3.11),
])
def test_unique_full_tool_field_reference_is_expanded(tmp_path, tool, path, value):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}资料")
    payload = {"ok": True, "source": "eastmoney", "data": {"symbol": SYMBOL, "pe_ttm": value}}
    if tool == "get_research_reports":
        payload["data"] = {"reports": [{"eps_forecast": {"this_year": value}}]}
    ledger.ingest_tool_result(tool_name=tool, arguments={"code": SYMBOL},
                              result=json.dumps(payload), call_id="call_read|fc_read", success=True)
    result = ledger.validate_final_answer(
        f"{SYMBOL}（东方财富，CNY）资料中的观测数值为{value}。\n"
        f"```figures\n{value} | observed | {SYMBOL} | {tool}::{path}\n```"
    )
    assert result.valid, result.issues


def test_full_tool_field_reference_cannot_choose_between_calls(tmp_path):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}资料")
    for call, value in (("call_first", 11.54), ("call_second", 12.54)):
        ledger.ingest_tool_result(
            tool_name="get_a_share_valuation", arguments={"code": SYMBOL},
            result=json.dumps({"ok": True, "source": "eastmoney", "data": {"symbol": SYMBOL, "pe_ttm": value}}),
            call_id=call, success=True,
        )
    result = ledger.validate_final_answer(
        f"{SYMBOL}（东方财富，CNY）市盈率为11.54倍。\n"
        f"```figures\n11.54 | observed | {SYMBOL} | get_a_share_valuation::data.pe_ttm\n```"
    )
    assert not result.valid


@pytest.mark.parametrize("field,valid", [("eps_forecast", True), ("pe_forecast", False)])
def test_broker_eps_can_have_currency_but_pe_cannot(tmp_path, field, valid):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}资料")
    ledger.ingest_tool_result(
        tool_name="get_research_reports", arguments={"code": SYMBOL},
        result=json.dumps({"ok": True, "source": "eastmoney", "data": {
            "reports": [{field: {"this_year": 3.11}}],
        }}), call_id="call_reports|fc_reports", success=True,
    )
    result = ledger.validate_final_answer(
        f"{SYMBOL}（东方财富，CNY）研报中的预测数值为3.11元/股。\n"
        f"```figures\n3.11 | observed | {SYMBOL} | get_research_reports::data.reports[0].{field}.this_year\n```"
    )
    assert result.valid is valid, result.issues
    assert not ledger._price_records(), "Forecast EPS must not widen the traded-price evidence pool"


@pytest.mark.parametrize("refs", [
    "get_financial_statements::NETCASH_OPERATE, get_financial_statements::CONSTRUCT_LONG_ASSET",
    "call_cash::data.601899.SH.periods[0].NETCASH_OPERATE; call_cash::data.601899.SH.periods[0].CONSTRUCT_LONG_ASSET",
])
def test_cash_flow_arithmetic_uses_multiple_monetary_field_refs(tmp_path, refs):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}现金流")
    ledger.ingest_tool_result(
        tool_name="get_financial_statements", arguments={"code": SYMBOL, "statement": "cashflow", "period": "quarter"},
        result=json.dumps({"ok": True, "source": "eastmoney", "statement": "cashflow", "period": "quarter", "data": {
            SYMBOL: {"periods": [{"REPORT_DATE": "2026-06-30", "NETCASH_OPERATE": 55471692934,
                                   "CONSTRUCT_LONG_ASSET": 12942927636}]},
        }}), call_id="call_cash", success=True,
    )
    result = ledger.validate_final_answer(
        f"{SYMBOL}（东方财富，CNY）经营现金流减购建长期资产现金支出约425.29亿元，不等同完整自由现金流。\n"
        f"```figures\n425.29 | derived | (55471692934 - 12942927636) / 100000000；{SYMBOL}，2026-06-30，亿元 | {refs}\n```"
    )
    assert result.valid, result.issues
    assert not ledger._price_records(), "Monetary accounts must not become traded-price observations"


def _eastmoney(now, price=29.5):
    return {"data": {"f57": "601899", "f58": "紫金矿业", "f43": price,
                     "f86": int(now.timestamp()), "f164": 11.59, "f167": 4.06, "f116": 1e11}}


def _tencent_text(now, code="601899", price="29.37"):
    fields = [""] * 88
    for index, value in {1: "紫金矿业", 2: code, 3: price, 30: now.strftime("%Y%m%d%H%M%S"), 39: "11.54", 46: "4.09"}.items():
        fields[index] = value
    return 'v_sh601899="' + "~".join(fields) + '";'


def _fail(*args, **kwargs):
    raise OSError("offline")


def test_quote_uses_eastmoney_without_extra_network(monkeypatch):
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(quotes.eastmoney_client, "get_json", lambda *args, **kwargs: _eastmoney(now))
    monkeypatch.setattr(quotes.requests, "get", _fail)
    result = json.loads(quotes.AShareValuationTool().execute(code=SYMBOL))
    assert result["ok"] and not result["fallback_used"]
    assert result["data"]["pe_ttm"] == 11.59


@pytest.mark.parametrize("primary", ["offline", "invalid", "stale"])
def test_quote_falls_back_without_inventing_ttm_multiples(monkeypatch, primary):
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    calls = []
    def get_json(url, **kwargs):
        calls.append(url)
        if primary == "offline":
            return _fail()
        return _eastmoney(now - timedelta(days=30) if primary == "stale" else now, price=0 if primary == "invalid" else 29.5)
    monkeypatch.setattr(quotes.eastmoney_client, "get_json", get_json)
    monkeypatch.setattr(quotes.requests, "get", lambda *args, **kwargs: SimpleNamespace(
        text=_tencent_text(now), raise_for_status=lambda: None,
    ))
    result = json.loads(quotes.AShareValuationTool().execute(code=SYMBOL))
    assert result["ok"] and result["fallback_used"] and result["source"] == "tencent"
    assert len(calls) == 2
    assert result["data"]["last_price"] == 29.37
    assert result["data"]["price_adjustment"] == "raw"
    assert result["data"]["pe_ttm"] is None and result["data"]["pb"] is None
    assert set(result["unavailable_fields"]) == {"pe_ttm", "pb", "market_cap_cny"}


@pytest.mark.parametrize("kind", ["wrong_symbol", "stale", "future", "bad_price", "bad_envelope"])
def test_quote_rejects_invalid_fallback(monkeypatch, kind):
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    text = _tencent_text(now)
    if kind == "wrong_symbol":
        text = _tencent_text(now, code="600547")
    if kind == "stale":
        text = _tencent_text(now - timedelta(days=30))
    if kind == "future":
        text = _tencent_text(now + timedelta(days=1))
    if kind == "bad_price":
        text = _tencent_text(now, price="NaN")
    if kind == "bad_envelope":
        text = "upstream failure"
    monkeypatch.setattr(quotes.eastmoney_client, "get_json", _fail)
    monkeypatch.setattr(quotes.requests, "get", lambda *args, **kwargs: SimpleNamespace(text=text, raise_for_status=lambda: None))
    result = json.loads(quotes.AShareValuationTool().execute(code=SYMBOL))
    assert not result["ok"] and len(result["attempted_sources"]) == 3


@pytest.mark.parametrize("question,entry", [
    ("你推荐我现在买什么，紫金矿业吗", True),
    ("什么时候，什么指标出现后值得考虑买入这只股票", True),
    ("紫金矿业目前是否值得买入", True),
    ("紫金矿业代码601899.SH，目前是否值得买入", True),
    ("推荐A股低价高增长股票并给出买入价", False),
    ("请分析 BYN.V 并推荐买入价", False),
    ("601899.SH当前市盈率是多少", False),
    ("帮我写提示词问是否值得买入", False),
    ("是否值得买入的代码怎么写", False),
])
def test_entry_research_scope(question, entry):
    assert needs_entry_inputs(question) is entry


def test_price_level_request_is_not_full_buy_verdict():
    assert not is_single_buy_question("请分析 BYN.V 并推荐买入价")
    assert is_single_buy_question("你推荐我现在买什么，紫金矿业吗")
    assert is_single_buy_question("紫金矿业代码601899.SH，目前是否值得买入")
    assert not is_single_buy_question("是否值得买入的代码怎么写")


def _fact(field, value, call="new", tool="get_a_share_valuation", statement=None, stamp="2026-06-30"):
    return SimpleNamespace(tool=tool, symbol=SYMBOL, status="observed", field=field, value=value,
                           call_id=call, timestamp=stamp, currency="CNY", source="eastmoney",
                           report_period="quarter", statement=statement)


def test_latest_quote_cannot_reuse_old_snapshot_multiples():
    records = [_fact("data.last_price", 29.5, "old"), _fact("data.pe_ttm", 11.59, "old"),
               _fact("data.last_price", 29.37)]
    coverage = decision_coverage(records, SYMBOL)
    assert coverage["raw_quote"]["status"] == "observed"
    assert coverage["valuation_multiple"]["status"] == "unavailable"
    assert coverage["entry_evidence"]["status"] == "incomplete"


def test_missing_inputs_cannot_be_a_wait_verdict():
    assert equity_entry_answer_issues("结论：暂不买入（等待）", [], {SYMBOL}, [])
    assert not equity_entry_answer_issues("结论：研究未完成。未取得报价。", [], {SYMBOL}, [])


@pytest.mark.parametrize("stance,accepted", [("研究未完成", True), ("暂不买入（等待）", False), ("回避", False)])
def test_actual_buy_contract_distinguishes_data_failure_from_market_view(stance, accepted):
    records = [_fact("data.last_price", 29.37),
               _fact("data.601899.SH.periods[0].PARENTNETPROFIT", 100, tool="get_financial_statements"),
               _fact("data.601899.SH.periods[0].NETCASH_OPERATE", 200, tool="get_financial_statements")]
    attempts = [{"symbol": SYMBOL, "tool": "get_a_share_valuation", "success": True}]
    answer = (f"结论：{stance}；期限：未来六至十二个月；信心：低。\n"
              "主要依据：财报截至2026-06-30，未取得市盈率与市净率，暂无法评价入场价格。\n"
              "改变判断的条件：取得同期估值后与可比公司比较。")
    issues = decision_issues(answer, records, {SYMBOL}, attempts=attempts)
    assert (not issues) is accepted, issues


def test_followup_contract_accepts_equivalent_explicit_headings():
    records = [_fact("data.last_price", 29.37),
               _fact("data.601899.SH.periods[0].PARENTNETPROFIT", 100, tool="get_financial_statements"),
               _fact("data.601899.SH.periods[0].NETCASH_OPERATE", 200, tool="get_financial_statements")]
    answer = ("补充核验后，结论仍为：研究未完成；期限：未来六至十二个月；信心：低。\n"
              "## 盈利质量：已有证据\n财报截至2026-06-30。\n"
              "## 估值与价格：买入条件仍缺哪一环？\n未取得市盈率；需要与同业同口径估值比较。")
    assert not decision_issues(answer, records, {SYMBOL})


@pytest.mark.parametrize("stamp", ["2099-06-30", "2020-06-30", "not-a-date"])
def test_invalid_or_stale_financial_dates_do_not_complete_entry_evidence(stamp):
    records = [_fact("data.last_price", 29.5), _fact("data.pe_ttm", 11.59),
               _fact("data.601899.SH.periods[0].PARENTNETPROFIT", 100, tool="get_financial_statements", stamp=stamp),
               _fact("data.601899.SH.periods[0].NETCASH_OPERATE", 200, tool="get_financial_statements", stamp=stamp)]
    coverage = decision_coverage(records, SYMBOL)
    assert coverage["entry_evidence"]["status"] == "incomplete"
    assert "profitability" in coverage["entry_evidence"]["gaps"]


def test_profit_summary_requires_income_attempt_but_does_not_repeat_failed_read():
    records = [_fact("data.601899.SH.periods[0].PARENTNETPROFIT", 100, tool="get_financial_statements")]
    issues = equity_research_input_issues(records, {SYMBOL}, [])
    assert any(issue["code"] == "decision_income_not_checked" for issue in issues)
    attempts = [{"symbol": SYMBOL, "tool": "get_financial_statements", "statement": "income", "success": False}]
    issues = equity_research_input_issues(records, {SYMBOL}, attempts)
    assert not any(issue["code"] == "decision_income_not_checked" for issue in issues)
    sheet = equity_worksheet(records, {SYMBOL}, attempts)["companies"][0]
    assert sheet["research_progress"]["income_statement"] == "unavailable"
    assert sheet["research_progress"]["normalized_earnings_and_valuation"] == "not_verified_by_worksheet"
