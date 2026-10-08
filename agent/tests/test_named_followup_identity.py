"""Named follow-ups reuse only a uniquely verified instrument in session history."""

import json
from pathlib import Path

from src.agent.grounding.ledger import GroundingLedger


_LOCKED = {
    "status": "locked",
    "authorized_symbols": ["601899.SH", "518850.SH"],
    "primary_symbols": ["601899.SH", "518850.SH"],
    "records": [
        {"status": "locked", "symbol": "601899.SH", "currency": "CNY"},
        {"status": "locked", "symbol": "518850.SH", "currency": "CNY"},
    ],
}
_ANSWER = "紫金矿业 601899.SH 收于29.46元；华夏黄金ETF 518850.SH 收于8.607元。"

_ALTERNATIVES = "你是觉得这只股票的标的不怎么样吗，是否有别的好的标的，胜率更高，比如黄金相关的标的、紫晶矿业、山东黄金等"
_HOLDING = {
    "status": "locked", "authorized_symbols": ["513330.SH"],
    "primary_symbols": ["513330.SH"],
    "records": [{"status": "locked", "symbol": "513330.SH", "currency": "CNY"}],
}


def _failed_name(ledger: GroundingLedger) -> None:
    ledger.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "紫晶矿业"},
        result=json.dumps({"ok": True, "data": {"query": "紫晶矿业", "candidates": [],
                           "sources": {"eastmoney": "search failed", "sse": "ok", "szse": "ok"}}}),
        call_id="failed_peer", success=True,
    )


def test_alternatives_preserve_unique_current_subject_and_continue_other_candidates(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message=_ALTERNATIVES, history=[
        {"role": "assistant", "content": "持仓513330.SH", "grounding_identity": _HOLDING},
    ])
    assert ledger.inherited_symbols == {"513330.SH"}
    _failed_name(ledger)
    assert not ledger.should_request_user_confirmation
    assert ledger.authorized_symbols == {"513330.SH"}
    ledger.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "山东黄金"},
        result=json.dumps({"ok": True, "data": {"query": "山东黄金", "candidates": [
            {"symbol": "600547.SH", "name": "山东黄金", "exchange": "SH", "market": "cn", "source": "sse"},
        ], "sources": {"sse": "ok"}}}), call_id="other_peer", success=True,
    )
    assert ledger.authorized_symbols == {"513330.SH", "600547.SH"}
    assert "601899.SH" not in ledger.authorized_symbols  # No silent typo correction.


def test_alternatives_without_history_do_not_stop_after_one_failed_candidate(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message=_ALTERNATIVES)
    _failed_name(ledger)
    assert not ledger.should_request_user_confirmation
    assert not ledger.authorized_symbols


def test_retry_of_failed_alternatives_recovers_original_subject(tmp_path: Path) -> None:
    failed = {"status": "invalidated", "authorized_symbols": [], "records": [
        {"status": "invalidated", "query": "紫晶矿业", "candidates": []},
    ]}
    ledger = GroundingLedger(run_dir=tmp_path, user_message=_ALTERNATIVES, history=[
        {"role": "assistant", "grounding_identity": _HOLDING},
        {"role": "user", "content": _ALTERNATIVES},
        {"role": "assistant", "content": "查询失败", "grounding_identity": failed},
    ])
    assert ledger.inherited_symbols == {"513330.SH"}


def test_alternatives_cannot_choose_among_multiple_prior_primary_symbols(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message=_ALTERNATIVES, history=[
        {"role": "assistant", "content": _ANSWER, "grounding_identity": _LOCKED},
    ])
    assert not ledger.inherited_symbols


def test_named_followup_inherits_only_the_named_instrument(tmp_path: Path) -> None:
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="紫金矿业目前的机构评价、财报分析以及估值是否偏高",
        history=[
            {"role": "user", "content": "比较紫金矿业和华夏黄金ETF"},
            {"role": "assistant", "content": _ANSWER, "grounding_identity": _LOCKED},
        ],
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"601899.SH"}
    assert ledger.inherited_symbols == {"601899.SH"}


def test_named_followup_recovers_after_same_name_source_failure(tmp_path: Path) -> None:
    failed = {
        "status": "invalidated",
        "authorized_symbols": [],
        "records": [{"status": "invalidated", "query": "紫金矿业", "candidates": []}],
    }
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="紫金矿业目前的机构评价、财报分析以及估值是否偏高",
        history=[
            {"role": "user", "content": "比较紫金矿业和华夏黄金ETF"},
            {"role": "assistant", "content": _ANSWER, "grounding_identity": _LOCKED},
            {"role": "user", "content": "紫金矿业目前的机构评价、财报分析以及估值是否偏高"},
            {"role": "assistant", "content": "请确认标的", "grounding_identity": failed},
        ],
    )

    assert ledger.authorized_symbols == {"601899.SH"}


def test_new_company_name_cannot_inherit_old_symbol(tmp_path: Path) -> None:
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="山东黄金目前的财报和估值如何",
        history=[{"role": "assistant", "content": _ANSWER, "grounding_identity": _LOCKED}],
    )

    assert ledger.identity_status == "unresolved"
    assert ledger.inherited_symbols == set()


def test_longer_company_name_cannot_inherit_prefix_name(tmp_path: Path) -> None:
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="紫金矿业国际目前的财报如何",
        history=[{"role": "assistant", "content": _ANSWER, "grounding_identity": _LOCKED}],
    )

    assert ledger.inherited_symbols == set()


def test_search_source_failure_is_not_reported_as_ambiguity(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="紫金矿业估值如何")
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "紫金矿业"},
        result=json.dumps({
            "ok": True,
            "data": {
                "query": "紫金矿业",
                "candidates": [],
                "sources": {
                    "eastmoney": "search failed",
                    "yahoo": "skipped: non-ASCII query is not supported",
                },
            },
        }, ensure_ascii=False),
        call_id="resolve",
        success=True,
    )

    assert ledger.identity_status == "invalidated"
    assert "数据源暂时不可用" in ledger.clarification_prompt()
    assert "唯一的交易标的" not in ledger.clarification_prompt()
