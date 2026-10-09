"""Analysis inputs stay distinct from recommendations and source verification."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.agent.grounding.analysis import delivery_review, multiple_comparisons, valuation_workbench


def fact(value, ref, currency="CNY", date="2026-06-30"):
    return {"value": value, "ref": ref, "currency": currency, "as_of": date, "source": "provider"}


def inputs():
    return {"parent_profit": fact(100, "f::parent"), "core_parent_profit": fact(80, "f::core"),
            "operating_cash_flow": fact(120, "c::ocf"), "capex_cash_paid": fact(50, "c::capex")}


def test_sustainability_inputs_have_formulas_but_no_automatic_verdict():
    result = valuation_workbench(inputs(), {"pe_ttm": fact(12, "q::pe")}, "industrial")
    metrics = {row["metric"]: row for row in result["calculations"]}
    assert metrics["non_core_profit_share_pct"]["value"] == 20
    assert metrics["operating_cash_after_disclosed_capex"]["value"] == 70
    assert "f::core" in metrics["non_core_profit_share_pct"]["formula"]
    assert result["sustainable_earnings"]["status"] == "analysis_required"
    assert result["benchmark_comparison"]["status"] == "analysis_required"
    assert set(result["conclusion_dimensions"]) == {"company_quality", "price_attractiveness", "direction", "current_action"}


@pytest.mark.parametrize("field,value", [("currency", "USD"), ("as_of", "2025-06-30"), ("ref", ""), ("value", float("nan")), ("value", True)])
def test_incompatible_or_missing_inputs_do_not_produce_a_ratio(field, value):
    data = inputs()
    data["core_parent_profit"][field] = value
    assert not any(row["metric"] == "non_core_profit_share_pct" for row in valuation_workbench(data, {}, "industrial")["calculations"])


@pytest.mark.parametrize("sector", ["bank", "insurance", "financial"])
def test_financial_companies_do_not_get_industrial_free_cash_flow(sector):
    assert len(valuation_workbench(inputs(), {}, sector)["calculations"]) == 1


def test_negative_capex_or_nonpositive_profit_is_not_silently_normalized():
    data = inputs()
    data["capex_cash_paid"]["value"] = -50
    data["parent_profit"]["value"] = 0
    assert valuation_workbench(data, {}, "industrial")["calculations"] == []


def test_delivery_review_requests_four_distinct_dimensions_and_analysis():
    requests = delivery_review("利润增长，PE不高。建议等待。")
    assert len(requests) == 3
    assert all(label in requests[0] for label in ("公司质量", "价格吸引力", "方向倾向", "当前行动"))


def test_honest_incomplete_analysis_passes_editorial_review():
    text = "公司质量：现金流较强。价格吸引力：历史估值基准未核实，暂未评估。方向倾向：未确定。当前行动：研究未完成。可持续盈利尚需查原始附注。"
    assert delivery_review(text) == []
    assert delivery_review("Company quality: cash positive. Price attractiveness: unassessed; peer basis missing. "
                           "Directional view: undetermined. Current action: research incomplete. Sustainable earnings not assessed.") == []


def peer_records():
    return [SimpleNamespace(tool="get_a_share_valuation", symbol="600036.SH", status="observed", call_id="peer",
                            field="data." + field, value=value, timestamp="2026-10-09T12:00:00+08:00", currency="CNY", source="provider")
            for field, value in [("last_price", 30), ("pe_ttm", 10), ("pb", 1)]]


def test_peer_multiple_difference_is_not_fair_value():
    quote = {"last_price": fact(30, "q::price", date="2026-10-09"), "pe_ttm": fact(12, "q::pe", date="2026-10-09")}
    rows = multiple_comparisons(peer_records(), "601899.SH", {"601899.SH", "600036.SH"}, quote)
    assert len(rows) == 1
    assert rows[0]["relative_multiple_difference_pct"] == pytest.approx(20)
    assert rows[0]["status"] == "same_day_multiple_difference_only"
    assert "not a verified comparable" in rows[0]["adoption_requires"]
    assert "peer::data.pe_ttm" in rows[0]["formula"]


@pytest.mark.parametrize("field,value", [("timestamp", "2026-10-08"), ("currency", "USD"), ("source", "another")])
def test_mismatched_peer_snapshots_do_not_generate_comparison(field, value):
    records = deepcopy(peer_records())
    for record in records:
        setattr(record, field, value)
    quote = {"last_price": fact(30, "q::price", date="2026-10-09"), "pe_ttm": fact(12, "q::pe", date="2026-10-09")}
    row = multiple_comparisons(records, "601899.SH", {"601899.SH", "600036.SH"}, quote)[0]
    assert row["relative_multiple_difference_pct"] is None
    assert row["formula"] is None


def test_unique_tool_reference_repairs_metadata_without_changing_report(tmp_path):
    from tests.test_grounding_role_hardening import _ledger, MARKET_A, XRAY, HDR, ROW, _block
    ledger = _ledger(tmp_path, MARKET_A, XRAY, message="Compare portfolio tail risk.")
    prose = HDR + " VaR 95%: 1.57%。"
    draft = prose + _block(ROW, "95% | count | confidence",
                           "1.57% | observed | VaR 95% | portfolio_risk_xray::data.tail_risk.var_95")
    validation = ledger.validate_final_answer(draft)
    repaired = ledger.repair_references(draft, validation)
    assert repaired is not None
    assert repaired.startswith(prose)
    assert "x1::data.tail_risk.var_95" in repaired
    assert ledger.revalidate(repaired).valid
    assert ledger.validation_count == 1


@pytest.mark.parametrize("mode", ["ambiguous", "alias", "wrong_value", "unrelated_bad_value"])
def test_reference_repair_does_not_guess_or_bypass_other_failures(tmp_path, mode):
    from tests.test_grounding_role_hardening import _ledger, MARKET_A, XRAY, XRAY_SECOND, HDR, ROW, _block
    calls = [MARKET_A, XRAY, *([XRAY_SECOND] if mode == "ambiguous" else [])]
    ledger = _ledger(tmp_path, *calls, message="Compare portfolio tail risk.")
    reference = "invented::data.tail_risk.var_95" if mode == "alias" else "portfolio_risk_xray::data.tail_risk.var_95"
    value = "9.99%" if mode == "wrong_value" else "1.57%"
    draft = HDR + f" VaR 95%: {value}。" + (" Price 9999 CNY." if mode == "unrelated_bad_value" else "")
    draft += _block(ROW, "95% | count | confidence", f"{value} | observed | VaR 95% | {reference}")
    assert ledger.repair_references(draft, ledger.validate_final_answer(draft)) is None


@pytest.mark.parametrize("budget,expected_calls", [(1, 1), (2, 1), (4, 2)])
def test_editorial_review_is_one_text_only_round_and_never_a_release_gate(tmp_path, monkeypatch, budget, expected_calls):
    from src.agent.grounding import GroundingLedger
    from src.agent.grounding.policies import ValidationResult
    from tests.test_agent_loop_no_assistant_prefill import _StubLLM, _build_agent

    class LLM(_StubLLM):
        def __init__(self):
            super().__init__()
            self.tool_sets = []

        def stream_chat(self, messages, **kwargs):
            self.tool_sets.append(kwargs.get("tools"))
            return super().stream_chat(messages, **kwargs)

    # Isolate the editorial review from numeric/identity gates, covered separately.
    monkeypatch.setattr(GroundingLedger, "validate_final_answer",
                        lambda self, text: ValidationResult(valid=True, released_text=text))
    llm = LLM()
    result = _build_agent(llm, max_iter=budget, tmp_run_dir=tmp_path / "run").run("601899.SH是否值得买入")
    assert result["status"] == "success"
    assert llm.call_count == expected_calls
    if budget > 2:
        assert llm.tool_sets[-1] is None
        assert any("[RESEARCH DELIVERY REVIEW]" in message.get("content", "")
                   for message in llm.seen_messages[-1])


def test_verified_summary_preserves_facts_without_preserving_rejected_advice(tmp_path):
    from tests.test_research_handoff_regression import host, quote, financial
    ledger = host(tmp_path)
    ledger.ingest_research_handoff([
        quote(), financial("indicators", {"PARENTNETPROFIT": 100, "KCFJCXSYJLR": 80}),
        financial("cashflow", {"NETCASH_OPERATE": 120, "CONSTRUCT_LONG_ASSET": 50}),
        financial("income", {"PARENT_NETPROFIT": 100, "OPERATE_INCOME": 500}),
    ], source_call_id="team")
    summary = ledger.verified_research_summary()
    assert summary is not None
    assert "研究未完成" in summary
    assert "归母净利润: CNY 100" in summary
    assert "经营现金流: CNY 120" in summary
    assert "公司质量" in summary and "价格吸引力" in summary and "方向倾向" in summary and "当前行动" in summary
    assert "```figures" not in summary
    assert ledger.validation_count == 0


def test_evidence_summary_does_not_release_unresolved_or_unchecked_research(tmp_path):
    from tests.test_research_handoff_regression import host, quote
    ledger = host(tmp_path)
    assert ledger.verified_research_summary() is None
    ledger.ingest_research_handoff([quote()], source_call_id="team")
    assert ledger.verified_research_summary() is None


@pytest.mark.parametrize("english", [False, True])
def test_partial_summary_keeps_missing_valuation_separate_from_wait(tmp_path, english):
    import json
    from src.agent.grounding import GroundingLedger
    from tests.test_research_handoff_regression import SYMBOL, NOW, quote, financial
    question = f"Should I buy {SYMBOL}?" if english else f"{SYMBOL}是否值得买入"
    ledger = GroundingLedger(run_dir=tmp_path, user_message=question)
    missing = quote()
    payload = json.loads(missing["result"])
    payload["data"].update(pe_ttm=None, pb=None)
    missing["result"] = json.dumps(payload)
    annual = financial("indicators", {"EPSJB": 2}, "annual")
    annual["arguments"]["period"] = "annual"
    data = json.loads(annual["result"])
    data["period"] = "annual"
    data["data"][SYMBOL]["periods"][0]["REPORT_DATE"] = f"{NOW.year-1}-12-31"
    annual["result"] = json.dumps(data)
    ledger.ingest_research_handoff([
        missing, annual, financial("indicators", {"PARENTNETPROFIT": 100, "KCFJCXSYJLR": 80}),
        financial("cashflow", {"NETCASH_OPERATE": 120, "CONSTRUCT_LONG_ASSET": 50}),
        financial("income", {"PARENT_NETPROFIT": 100, "OPERATE_INCOME": 500}),
    ], source_call_id="team")
    summary = ledger.verified_research_summary()
    assert summary is not None
    assert ("Current action: research incomplete" if english else "当前行动：研究未完成") in summary
    assert ("valuation_multiple" if english else "PE/PB估值依据") in summary


def test_undated_peer_multiples_are_not_compared_even_if_both_dates_missing():
    records = peer_records()
    for record in records:
        record.timestamp = None
    quote = {"last_price": fact(30, "q::price", date=None), "pe_ttm": fact(12, "q::pe", date=None)}
    row = multiple_comparisons(records, "601899.SH", {"601899.SH", "600036.SH"}, quote)[0]
    assert row["relative_multiple_difference_pct"] is None
