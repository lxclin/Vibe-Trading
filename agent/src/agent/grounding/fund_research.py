"""Dated holdings context for research coverage, never constituent authorization."""
from __future__ import annotations

import json
import math
from decimal import Decimal

from src.agent.grounding.research_handoff import etf_receipt_subject, scoped_receipts


def fund_holdings_context(ledger, receipts):
    latest = {}
    for receipt in scoped_receipts(receipts, ledger.authorized_symbols):
        if receipt["tool"] != "etf_holdings" or not receipt["success"]:
            continue
        subject = etf_receipt_subject(receipt["arguments"], ledger.inherited_symbols)
        payload = json.loads(receipt["result"])
        data = payload.get("data")
        if not subject or not isinstance(data, dict) or not isinstance(data.get("holdings"), list):
            continue
        declared = str(data.get("symbol") or "").upper()
        if declared not in {subject, subject.split(".")[0], subject.replace(".SH", ".SS")}:
            continue
        date = str(payload.get("as_of") or "")
        if subject in latest and date <= latest[subject]["as_of"]:
            continue
        rows, weights = [], []
        for index, row in enumerate(data["holdings"]):
            if not isinstance(row, dict):
                continue
            weight = row.get("pct_of_net_assets")
            valid = isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(weight) and 0 <= weight <= 100
            if valid:
                weights.append(Decimal(str(weight)))
            rows.append({"name": row.get("name"), "source_symbol": row.get("symbol"),
                         "pct_of_net_assets": weight if valid else None,
                         "weight_ref": f"{receipt['call_id']}::data.holdings[{index}].pct_of_net_assets" if valid else None})
        paging = payload.get("paging") or {}
        all_weights = len(weights) == len(rows) and bool(rows)
        weight_sum = sum(weights, Decimal(0)) if all_weights else None
        latest[subject] = {
            "fund": subject, "as_of": date, "call_id": receipt["call_id"],
            "source_coverage_label": payload.get("coverage"), "returned_rows": len(rows),
            "holdings_in_report": data.get("holdings_in_period"),
            "rows_withheld_by_top_n": data.get("holdings_withheld_by_top_n"),
            "next_offset": paging.get("next_offset"),
            "all_returned_weights_known": all_weights,
            "returned_weight_sum_pct": str(weight_sum) if weight_sum is not None else None,
            "sum_is_derived": True,
            "sum_formula": " + ".join(str(r["pct_of_net_assets"]) for r in rows) if all_weights and len(rows) <= 12 else None,
            "report_disclosed_weight_pct": data.get("pct_of_net_assets_disclosed"),
            "constituents_preview": rows[:12],
            "analysis_coverage": "unchecked; holdings disclosure is not financial analysis",
        }
    if not latest:
        return ""
    return (
        "\n\n[DATED FUND RESEARCH COVERAGE] Use disclosed holdings weights to prioritize material "
        "company evidence, not to claim a few companies represent the whole index. The sum below is "
        "derived from returned rows only, NOT analyzed coverage or live weights; cite operands and "
        "formula if publishing it. A full_portfolio source label or complete page does not establish "
        "all rows were returned: check top_n withholding and pagination. Missing weights remain "
        "unknown, never zero. No constituent is authorized by this list: resolve its actual venue "
        "before financial calls. Verify the tracked index from an official fund document, then "
        "discover the matching index provider's published valuation links; never guess a code or "
        "substitute another similarly named index. Weight-specific earnings/cash-flow findings must "
        "name covered and unexamined holdings, periods and accounting basis. Price attraction still "
        "requires actual index valuation evidence, not the weighted average of constituent PEs.\n"
        + json.dumps(list(latest.values()), ensure_ascii=False)
    )
