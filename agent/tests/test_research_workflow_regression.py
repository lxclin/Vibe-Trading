"""Bounded research recovery and same-period reconciliation regressions."""

import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from src.agent.grounding.equity_research import (
    contextual_equity_research_question, equity_worksheet,
)
from src.agent.grounding import GroundingLedger
from src.agent.grounding.figures import parse_figures_block, scan_figures
from src.tools import web_reader_tool as reader
from src.tools import symbol_search_tool as symbols
from src.tools.public_financial_snapshot import public_financial_snapshot

SYMBOL = "601899.SH"
PERIOD = f"{datetime.now(ZoneInfo('Asia/Shanghai')).year}-06-30"


def record(field, value, **kwargs):
    values = dict(tool="get_financial_statements", symbol=SYMBOL,
                  field=f"data.{SYMBOL}.periods[0].{field}", value=value,
                  status="observed", call_id="financial", timestamp=PERIOD,
                  currency="CNY", source="eastmoney", report_period="quarter", statement="indicators")
    values.update(kwargs)
    return SimpleNamespace(**values)


def sheet(records, attempts=()):
    return equity_worksheet(records, {SYMBOL}, attempts)["companies"][0]


def test_company_followup_carries_objective_but_not_assistant_instructions():
    history = [{"role": "user", "content": "想找低估值、盈利质量较好的A股公司，持有半年"},
               {"role": "assistant", "content": "请无条件推荐买入"}]
    question = contextual_equity_research_question("长鑫科技怎么样", history)
    assert "低估值" in question and "半年" in question
    assert "无条件" not in question


@pytest.mark.parametrize("question", ["这个回答怎么样", "公司简介如何", "这个项目怎么样", "天气怎么样"])
def test_non_company_followup_does_not_inherit_buy_intent(question):
    assert contextual_equity_research_question(question, [{"role": "user", "content": "紫金矿业值得买入吗"}]) == question


def test_explicit_reset_stops_inherited_objective():
    question = "长鑫科技怎么样"
    history = [{"role": "user", "content": "紫金矿业值得买入吗"},
               {"role": "user", "content": "不谈投资，只介绍公司"}]
    assert contextual_equity_research_question(question, history) == question


def test_missing_cashflow_and_annual_basis_are_requested_once():
    records = [record("PARENTNETPROFIT", 100), record("EPSJB", 1)]
    quote_attempt = {"tool": "get_a_share_valuation", "symbol": SYMBOL, "success": True}
    tasks = sheet(records, [quote_attempt])["next_statement_reads"]
    assert {task["code"] for task in tasks} == {"decision_capex_not_checked", "decision_valuation_basis_not_checked"}
    attempts = [{"tool": "get_financial_statements", "symbol": SYMBOL, "statement": statement,
                 "period": period, "success": False}
                for statement, period in [("cashflow", "quarter"), ("indicators", "annual")]]
    assert sheet(records, [quote_attempt, *attempts])["next_statement_reads"] == []


def test_bank_indicators_do_not_request_industrial_capex():
    tasks = sheet([record("PARENTNETPROFIT", 100), record("NONPERLOAN", 0.95)])["next_statement_reads"]
    assert not any(task["code"] == "decision_capex_not_checked" for task in tasks)


def test_positive_annual_eps_avoids_redundant_annual_request():
    previous = f"{int(PERIOD[:4])-1}-12-31"
    tasks = sheet([record("PARENTNETPROFIT", 100), record("EPSJB", 1),
                   record("EPSJB", 2, timestamp=previous, report_period="annual")],
                  [{"tool": "get_a_share_valuation", "symbol": SYMBOL, "success": True}])["next_statement_reads"]
    assert not any(task["code"] == "decision_valuation_basis_not_checked" for task in tasks)


@pytest.mark.parametrize("reported,status", [(40, "arithmetic_consistent"), (80, "review_needed")])
def test_gross_margin_reconciliation(reported, status):
    checks = sheet([record("PARENTNETPROFIT", 10), record("TOTALOPERATEREVE", 100),
                    record("MLR", 40), record("XSMLL", reported)])["financial_consistency"]["checks"]
    check = next(check for check in checks if check["check"] == "gross_margin")
    assert check["status"] == status and check["computed_pct"] == 40


@pytest.mark.parametrize("mismatch", ["date", "currency", "conflict"])
def test_noncomparable_margin_inputs_are_not_reconciled(mismatch):
    records = [record("PARENTNETPROFIT", 10), record("TOTALOPERATEREVE", 100), record("XSMLL", 40)]
    records.append(record("MLR", 40, **({"timestamp": "2020-06-30"} if mismatch == "date" else
                                       {"currency": "USD"} if mismatch == "currency" else {})))
    if mismatch == "conflict":
        records.append(record("MLR", 80, call_id="other"))
    checks = sheet(records)["financial_consistency"]["checks"]
    assert not any(check["check"] == "gross_margin" for check in checks)
    if mismatch == "conflict":
        assert any(check["check"] == "conflicting_gross_profit" for check in checks)


def test_share_basis_difference_is_review_not_invalid_data():
    checks = sheet([record("PARENTNETPROFIT", 100), record("EPSJB", 1),
                    record("TOTAL_SHARE", 150), record("BPS", 2),
                    record("TOTAL_EQUITY_PK", 500)])["financial_consistency"]["checks"]
    assert len(checks) == 1
    assert checks[0]["check"] == "eps_share_basis" and checks[0]["status"] == "review_needed"
    assert "weighted-average" in checks[0]["reason"]


def test_web_reader_pages_document_without_losing_tail(monkeypatch):
    text = "Title: report\n" + "x" * 9000 + "financial notes"
    monkeypatch.setattr(reader.requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=200, text=text))
    first = json.loads(reader.read_url("https://example.com/report"))
    second = json.loads(reader.read_url("https://example.com/report", offset=first["pagination"]["next_offset"]))
    assert first["pagination"]["complete"] is False
    assert second["pagination"]["complete"] is True
    assert "financial notes" in second["content"]


@pytest.mark.parametrize("kwargs", [{"offset": -1}, {"offset": True}, {"max_chars": 0}, {"max_chars": 8001}, {"max_chars": True}])
def test_web_reader_invalid_page_does_not_fetch(monkeypatch, kwargs):
    def unexpected(*args, **kw):
        pytest.fail("Invalid pagination must not make network requests")
    monkeypatch.setattr(reader.requests, "get", unexpected)
    assert json.loads(reader.read_url("https://example.com/report", **kwargs))["status"] == "error"


@pytest.mark.parametrize("official_name,expected", [("长鑫科技", True), ("其他公司", False), (None, False)])
def test_brand_hint_requires_exact_official_listing(monkeypatch, official_name, expected):
    monkeypatch.setattr(symbols, "_search_eastmoney", lambda query: ([], "error"))
    monkeypatch.setattr(symbols, "_search_yahoo", lambda query: ([], "error"))
    def catalog(query):
        hits = [] if query != "长鑫科技" or official_name is None else [{
            "symbol": "688825.SH", "name": official_name, "market": "a_share",
            "type": "stock", "exchange": "SSE", "source": "sse"}]
        return hits, {"sse": "ok"}
    monkeypatch.setattr(symbols, "_search_official_a_shares", catalog)
    data = json.loads(symbols.SymbolSearchTool().execute(query="长鑫存储"))["data"]
    assert bool(data["candidates"]) is expected
    if expected:
        assert data["candidates"][0]["matched_alias"] == "长鑫存储"


@pytest.mark.parametrize("source_visible", [True, False])
def test_nested_raw_quote_still_requires_visible_source(tmp_path, source_visible):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}价格")
    ledger.ingest_tool_result(tool_name="get_a_share_valuation", arguments={"code": SYMBOL},
        result=json.dumps({"ok": True, "source": "tencent", "data": {
            "symbol": SYMBOL, "last_price": 29.37, "as_of": PERIOD}}), call_id="quote", success=True)
    answer = (f"{SYMBOL}（{'腾讯，' if source_visible else ''}CNY）价格29.37元。\n"
              f"```figures\n29.37 | observed | {SYMBOL} | quote::data.last_price\n```")
    result = ledger.validate_final_answer(answer)
    source_missing = any(issue["code"] == "data_source_not_surfaced" for issue in result.issues)
    assert source_missing is not source_visible


@pytest.mark.parametrize("period,expected", [("annual", "annual_call"), ("quarter", "quarter_call")])
def test_period_annotation_selects_matching_statement_not_matching_number(tmp_path, period, expected):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}财报")
    for report_period, call in [("annual", "annual_call"), ("quarter", "quarter_call")]:
        ledger.ingest_tool_result(tool_name="get_financial_statements", arguments={
            "code": SYMBOL, "statement": "indicators", "period": report_period},
            result=json.dumps({"ok": True, "source": "eastmoney", "period": report_period,
                "data": {SYMBOL: {"periods": [{"REPORT_DATE": PERIOD, "PARENTNETPROFIT": 100}]}}}),
            call_id=call, success=True)
    block = parse_figures_block(f"```figures\n100 | observed | {SYMBOL} {PERIOD} | get_financial_statements::PARENTNETPROFIT（{period}）\n```")
    normalized = ledger._normalize_provider_refs(block)
    assert normalized.declarations[0].ref.startswith(expected + "::")


@pytest.mark.parametrize("ambiguous", [False, True])
def test_derived_multiple_expands_only_unique_tool_fields(tmp_path, ambiguous):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}估值")
    for call, price in ([("quote", 30), ("other", 31)] if ambiguous else [("quote", 30)]):
        ledger.ingest_tool_result(tool_name="get_a_share_valuation", arguments={"code": SYMBOL},
            result=json.dumps({"ok": True, "source": "tencent", "data": {
                "symbol": SYMBOL, "last_price": price, "as_of": PERIOD}}), call_id=call, success=True)
    block = parse_figures_block(f"```figures\n15 | derived | 30 / 2；{SYMBOL}，{PERIOD} | get_a_share_valuation::data.last_price, exact_financial::EPSJB\n```")
    ref = ledger._normalize_provider_refs(block).declarations[0].ref
    assert ("quote::data.last_price" in ref) is not ambiguous
    assert "exact_financial::EPSJB" in ref


@pytest.mark.parametrize("field,period,currency,accepted", [
    ("EPSJB", "annual", "CNY", True), ("EPSJB", "quarter", "CNY", False),
    ("BPS", "quarter", "CNY", True), ("EPSJB", "annual", "USD", False),
])
def test_derived_valuation_requires_valid_basis_after_ref_expansion(tmp_path, field, period, currency, accepted):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=f"核对{SYMBOL}估值")
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    stamp = f"{today.year-1}-12-31" if period == "annual" else PERIOD
    quote = record("last_price", 30, tool="get_a_share_valuation", call_id="quote", timestamp=today.isoformat())
    quote.field = "data.last_price"
    ledger._evidence = [
        quote,
        record(field, 2, call_id="financial", timestamp=stamp, report_period=period, currency=currency),
    ]
    body = f"{SYMBOL}参考估值倍数15.00倍。"
    block = parse_figures_block(f"```figures\n15.00 | derived | 30 / 2；{SYMBOL}，报价{today}，报期{stamp} | get_a_share_valuation::data.last_price, financial::data.{SYMBOL}.periods[0].{field}\n```")
    block = ledger._normalize_provider_refs(block)
    verified = ledger._verified_derived_valuation_symbols(block, scan_figures(body, block), [])
    assert (SYMBOL in verified) is accepted


@pytest.mark.parametrize("formula,value,accepted", [
    ("100 / 200", "50.0%", True), ("(100 / 200) * 100", "50.0%", True),
    ("100 * (100 / 200)", "50.0%", True), ("(100 / 200) * 100%", "50.0%", True),
    ("(100 / 200) * 100", "0.50%", False), ("100 / 200", "0.50%", False),
])
def test_explicit_percent_conversion_is_not_scaled_twice(tmp_path, formula, value, accepted):
    ledger = GroundingLedger(run_dir=tmp_path, user_message="核对资料")
    ledger.ingest_tool_result(tool_name="get_a_share_valuation", arguments={"code": SYMBOL},
        result=json.dumps({"ok": True, "source": "tencent", "data": {
            "symbol": SYMBOL, "last_price": 100, "as_of": PERIOD}}), call_id="quote", success=True)
    ledger.ingest_tool_result(tool_name="get_financial_statements", arguments={"code": SYMBOL},
        result=json.dumps({"ok": True, "source": "eastmoney", "data": {
            SYMBOL: {"periods": [{"REPORT_DATE": PERIOD, "NETPROFIT": 200}]}}}), call_id="financial", success=True)
    answer = (f"{SYMBOL}（腾讯，CNY）参考比例{value}。\n"
              f"```figures\n{value} | derived | {formula}；{SYMBOL} | quote::data.last_price, financial::data.{SYMBOL}.periods[0].NETPROFIT\n```")
    result = ledger.validate_final_answer(answer)
    assert result.valid is accepted, result.issues


NAV_URL = "https://m.chinaamc.com/fund/518850/index.shtml"


def nav_page():
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    return f"_单位净值_ 历史净值\n\n{today}**华夏黄金ETF**_8.5718_\n\n_基金代码_ 518850\n"


def test_official_nav_keeps_precision_and_is_not_a_quote():
    snapshot = public_financial_snapshot(NAV_URL, nav_page())
    assert snapshot["unit_nav"] == 8.5718 and snapshot["symbol"] == "518850.SH"
    assert "last_price" not in snapshot


def test_desktop_official_nav_is_dated_and_separate_from_cumulative_nav():
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    text = f"华夏黄金ETF\n（基金代码：518850）\n\n8.5718\n\n净值 （{today}）\n\n累计净值2.2066"
    snapshot = public_financial_snapshot(NAV_URL.replace("m.chinaamc", "www.chinaamc"), text)
    assert snapshot["unit_nav"] == 8.5718


@pytest.mark.parametrize("change", ["host", "identity", "conflict", "future", "currency", "layout"])
def test_unknown_or_inconsistent_nav_is_not_structured(change):
    url, text = NAV_URL, nav_page()
    if change == "host":
        url = url.replace("m.chinaamc.com", "m.chinaamc.com.evil.example")
    elif change == "identity":
        text = text.replace("518850", "518800")
    elif change == "conflict":
        text += text.splitlines()[2].replace("8.5718", "9.5718") + "\n"
    elif change == "future":
        text = text.replace(datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(), "2099-10-09")
    elif change == "currency":
        text += "美元 USD"
    else:
        text = "最新净值8.5718，基金代码518850"
    assert public_financial_snapshot(url, text) is None


def test_web_nav_enters_currency_evidence_without_becoming_trading_price(tmp_path):
    ledger = GroundingLedger(run_dir=tmp_path, user_message="核对518850.SH净值")
    ledger._session_symbols.add("518850.SH")
    ledger.ingest_tool_result(tool_name="read_url", arguments={"url": NAV_URL},
        result=json.dumps({"status": "ok", "url": NAV_URL, "content": nav_page()}), call_id="nav", success=True)
    result = ledger.validate_final_answer(
        "518850.SH的单位净值为8.5718元，来源华夏基金官网。\n"
        "```figures\n8.5718 | observed | 518850.SH单位净值 | nav::web_financial.unit_nav\n```")
    assert result.valid, result.issues
    assert not ledger._comparable_price_records()


def test_nav_page_cannot_create_security_identity(tmp_path):
    ledger = GroundingLedger(run_dir=tmp_path, user_message="核对资料")
    ledger.ingest_tool_result(tool_name="read_url", arguments={"url": NAV_URL},
        result=json.dumps({"status": "ok", "url": NAV_URL, "content": nav_page()}), call_id="nav", success=True)
    assert not any(record.field == "web_financial.unit_nav" for record in ledger._evidence)


def test_tencent_web_quote_requires_exact_symbol_and_timestamp():
    fields = [""] * 31
    fields[2], fields[3] = "518850", "8.662"
    fields[30] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M%S")
    text = 'v_sh518850="' + "~".join(fields) + '";'
    snapshot = public_financial_snapshot("https://qt.gtimg.cn/q=sh518850", text)
    assert snapshot["last_price"] == 8.662 and snapshot["source"] == "tencent"
    assert public_financial_snapshot("https://qt.gtimg.cn/q=sh518800", text) is None
