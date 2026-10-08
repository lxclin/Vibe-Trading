"""The local stock checklist must coexist with upstream text-only correction."""

import json
from pathlib import Path

from src.agent.grounding import GroundingLedger
from src.agent.grounding.decision import decision_coverage, is_single_buy_question

WAIT = (
    "结论：暂不买入（等待）；期限：未来六至十二个月；信心：低。\n"
    "主要依据：未取得本轮行情、估值及财报数据。\n"
    "改变判断的条件：核对最新财报同比盈利与同业估值的比较结果后重新评估。"
)


def test_checklist_recovers_unchecked_tools_then_stops_after_failures(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="601899.SH现在是否值得买入")
    validation = ledger.validate_final_answer(WAIT)
    assert ledger.recovery_action(validation) == "get_a_share_valuation"
    assert 'get_a_share_valuation' in ledger.recovery_prompt("get_a_share_valuation", validation)
    ledger.ingest_tool_result(
        tool_name="get_a_share_valuation", arguments={"code": "601899.SH"},
        result='{"ok":false,"error":"provider unavailable"}', call_id="quote", success=False,
    )
    validation = ledger.validate_final_answer(WAIT)
    assert ledger.recovery_action(validation) == "get_financial_statements"
    assert 'statement="cashflow"' in ledger.recovery_prompt("get_financial_statements", validation)
    for statement in ("indicators", "cashflow"):
        ledger.ingest_tool_result(
            tool_name="get_financial_statements",
            arguments={"code": "601899.SH", "statement": statement, "period": "quarter"},
            result='{"ok":false,"error":"provider unavailable"}', call_id=statement, success=False,
        )
    validation = ledger.validate_final_answer(WAIT)
    assert ledger.recovery_action(validation) is None
    assert validation.valid, validation.issues


def test_uppercase_report_dates_survive_with_upstream_evidence_fields(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="601899.SH是否值得买入")
    for statement, metric in (("indicators", "PARENTNETPROFIT"), ("cashflow", "NETCASH_OPERATE")):
        ledger.ingest_tool_result(
            tool_name="get_financial_statements",
            arguments={"code": "601899.SH", "statement": statement, "period": "quarter"},
            result=json.dumps({"ok": True, "source": "eastmoney", "statement": statement,
                               "data": {"601899.SH": {"periods": [{"REPORT_DATE": "2026-06-30", metric: 123456789}]}}}),
            call_id=statement, success=True,
        )
    coverage = decision_coverage(ledger._evidence, "601899.SH", ledger._decision_tool_attempts)
    assert coverage["profitability"]["status"] == "observed"
    assert coverage["operating_cash_flow"]["status"] == "observed"
    assert coverage["profitability"]["as_of"] == "2026-06-30"
    assert all(record.statement in {"indicators", "cashflow"} for record in ledger._evidence)


def test_bare_fund_codes_are_excluded_from_company_financial_checklist() -> None:
    for code in ("518850", "513330.SH", "159915.SZ"):
        assert not is_single_buy_question(f"{code}目前是否值得买入")
    assert is_single_buy_question("601899.SH目前是否值得买入")
