"""Responses transport item IDs must not invalidate real tool-call references."""

import json
from pathlib import Path

import pytest

from src.agent.grounding import GroundingLedger


def _quote(ledger: GroundingLedger, symbol: str, price: float, pe: float, call_id: str) -> None:
    ledger.ingest_tool_result(
        tool_name="get_a_share_valuation", arguments={"code": symbol},
        result=json.dumps({"ok": True, "source": "eastmoney_quote", "price_adjustment": "raw",
                           "as_of": "2026-10-08T11:23:10+08:00",
                           "data": {"last_price": price, "pe_ttm": pe}}),
        call_id=call_id, success=True,
    )


@pytest.mark.parametrize("ref", ["call_zijin", "call_zijin|fc_item"])
def test_provider_call_and_transport_pair_refer_to_same_observation(tmp_path: Path, ref: str) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="比较601899.SH和600547.SH的估值")
    _quote(ledger, "601899.SH", 29.48, 11.59, "call_zijin|fc_item")
    _quote(ledger, "600547.SH", 26.48, 22.3, "call_shandong|fc_other")
    answer = (
        "601899.SH（eastmoney_quote，CNY）价格29.48元，PE为11.59倍。\n"
        "600547.SH（eastmoney_quote，CNY）价格26.48元，PE为22.3倍。\n"
        "```figures\n"
        f"29.48 | observed | 601899.SH | {ref}::data.last_price\n"
        f"11.59 | observed | 601899.SH | {ref}::data.pe_ttm\n"
        "26.48 | observed | 600547.SH | call_shandong::data.last_price\n"
        "22.3 | observed | 600547.SH | call_shandong::data.pe_ttm\n```"
    )
    result = ledger.validate_final_answer(answer)
    assert result.valid, result.issues


@pytest.mark.parametrize("ref,written", [
    ("r1::data.pe_ttm", "11.59"),
    ("call_zijin::data.missing", "11.59"),
    ("call_zijin::data.pe_ttm", "19.59"),
    ("call_shandong::data.pe_ttm", "11.59"),
])
def test_provider_compatibility_does_not_allow_aliases_wrong_fields_or_values(
    tmp_path: Path, ref: str, written: str,
) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="比较601899.SH和600547.SH估值")
    _quote(ledger, "601899.SH", 29.48, 11.59, "call_zijin|fc_item")
    _quote(ledger, "600547.SH", 26.48, 22.3, "call_shandong|fc_other")
    result = ledger.validate_final_answer(
        f"601899.SH（eastmoney_quote，CNY）PE为{written}倍。\n"
        f"```figures\n{written} | observed | 601899.SH | {ref}\n```"
    )
    assert not result.valid


def test_duplicate_provider_call_prefix_is_not_guessed(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="601899.SH估值如何")
    _quote(ledger, "601899.SH", 29.48, 11.59, "call_duplicate|fc_first")
    _quote(ledger, "601899.SH", 30.48, 12.59, "call_duplicate|fc_second")
    result = ledger.validate_final_answer(
        "601899.SH（eastmoney_quote，CNY）PE为11.59倍。\n"
        "```figures\n11.59 | observed | 601899.SH | call_duplicate::data.pe_ttm\n```"
    )
    assert not result.valid
    assert result.issues[0]["reason"] == "unknown_call_id"
    assert "已取得数据" in ledger.safe_fallback()
    assert "未通过工具获取的价格" not in ledger.safe_fallback()
