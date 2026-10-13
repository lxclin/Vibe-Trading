import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.grounding.fund_research import fund_holdings_context
from src.agent.grounding.research_handoff import capture_receipt, scoped_receipts

PAYLOAD = json.loads((Path(__file__).parent / "fixtures/research_20261012/fund_holdings.json").read_text())


def _receipt(payload=None, arguments=None):
    return capture_receipt("etf_holdings", arguments or {"mode": "holdings", "symbol": "513330.SH", "market": "CN"},
                           json.dumps(payload or PAYLOAD), "holding-original", True, {"513330.SH", "00700.HK"})


@pytest.mark.parametrize("symbol", ["513330.SH", "513330.SS", "513330"])
def test_symbol_parameter_restores_only_verified_fund_without_authorizing_constituents(symbol):
    receipt = _receipt(arguments={"mode": "holdings", "symbol": symbol, "market": "CN"})
    assert receipt["scope_symbols"] == ["513330.SH"]
    restored = scoped_receipts([receipt], {"513330.SH"})
    assert len(restored) == 1
    assert not scoped_receipts([receipt], {"600036.SH"})
    assert "00700.HK" not in restored[0]["scope_symbols"]


def test_legacy_overbroad_receipt_is_narrowed_only_to_existing_fund_identity():
    receipt = _receipt()
    receipt["scope_symbols"] = ["513330.SH", "00700.HK"]
    assert scoped_receipts([receipt], {"513330.SH"})[0]["scope_symbols"] == ["513330.SH"]


@pytest.mark.parametrize("args", [
    {"symbol": "600036.SH", "mode": "holdings"},
    {"symbol": "513330.SH", "mode": "lookup"},
    {"symbol": "513330.SH", "market": "US"},
])
def test_unverified_fund_lookup_and_wrong_market_are_not_restored(args):
    assert _receipt(arguments=args) is None


def test_mismatched_response_subject_is_not_fund_evidence():
    payload = copy.deepcopy(PAYLOAD)
    payload["data"]["symbol"] = "600036.SH"
    assert not scoped_receipts([_receipt(payload)], {"513330.SH"})


def test_real_top_five_snapshot_exposes_weights_and_partial_scope():
    ledger = SimpleNamespace(authorized_symbols={"513330.SH"}, inherited_symbols={"513330.SH"})
    context = fund_holdings_context(ledger, [_receipt()])
    row = json.loads(context.split("\n")[-1])[0]
    assert row["as_of"] == "2026-06-30"
    assert row["returned_rows"] == 5 and row["holdings_in_report"] == 30
    assert row["rows_withheld_by_top_n"] == 25
    assert row["source_coverage_label"] == "full_portfolio"
    assert row["analysis_coverage"].startswith("unchecked")
    assert row["sum_is_derived"] is True
    assert float(row["returned_weight_sum_pct"]) < row["report_disclosed_weight_pct"]
    assert row["constituents_preview"][0]["weight_ref"] == "holding-original::data.holdings[0].pct_of_net_assets"
    assert "No constituent is authorized" in context


def test_missing_weights_are_unknown_and_disable_sum():
    payload = copy.deepcopy(PAYLOAD)
    payload["data"]["holdings"][0]["pct_of_net_assets"] = None
    ledger = SimpleNamespace(authorized_symbols={"513330.SH"}, inherited_symbols={"513330.SH"})
    row = json.loads(fund_holdings_context(ledger, [_receipt(payload)]).split("\n")[-1])[0]
    assert row["returned_weight_sum_pct"] is None and not row["all_returned_weights_known"]
    assert row["constituents_preview"][0]["pct_of_net_assets"] is None


def test_newest_report_is_used_not_newest_fetch_of_an_old_period():
    payload = copy.deepcopy(PAYLOAD)
    payload["as_of"] = "2026-03-31"
    old = _receipt(payload)
    old["call_id"] = "old-period-new-fetch"
    old["fetched_at"] = 9999999999
    ledger = SimpleNamespace(authorized_symbols={"513330.SH"}, inherited_symbols={"513330.SH"})
    row = json.loads(fund_holdings_context(ledger, [old, _receipt()]).split("\n")[-1])[0]
    assert row["as_of"] == "2026-06-30" and row["call_id"] == "holding-original"
