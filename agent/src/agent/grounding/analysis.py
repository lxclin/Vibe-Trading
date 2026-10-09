"""Evidence-backed analysis inputs and a bounded, non-gating delivery review.

Arithmetic aids analysis; they never turn provider data into a buy rating.
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Any, Mapping


def valuation_workbench(facts: Mapping[str, Any], quote: Mapping[str, Any], sector: str,
                        comparisons: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    calculations = []

    def compute(label, left, right, operation, boundary):
        a, b = facts.get(left, {}), facts.get(right, {})
        x, y = a.get("value"), b.get("value")
        if (not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y))
                or not a.get("as_of") or a.get("as_of") != b.get("as_of") or not a.get("currency") or a.get("currency") != b.get("currency")
                or not a.get("ref") or not b.get("ref")):
            return
        if (operation == "ratio" and x <= 0) or (operation == "subtract" and y < 0):
            return
        value = (x - y) / x * 100 if operation == "ratio" else x - y
        if not math.isfinite(value):
            return
        formula = f"({a['ref']} - {b['ref']})" + (f" / {a['ref']} * 100" if operation == "ratio" else "")
        calculations.append({"metric": label, "value": value, "formula": formula,
                             "input_refs": [a["ref"], b["ref"]], "date": a.get("as_of"),
                             "unit": "percent" if operation == "ratio" else a.get("currency"),
                             "status": "derived_not_independently_verified", "boundary": boundary})

    compute("non_core_profit_share_pct", "parent_profit", "core_parent_profit", "ratio",
            "Difference between parent and deducted profit, not a normalized earnings estimate. Inspect filing notes and signs.")
    if sector not in {"bank", "insurance", "financial"}:
        compute("operating_cash_after_disclosed_capex", "operating_cash_flow", "capex_cash_paid", "subtract",
                "Consolidated OCF minus disclosed purchase/construction cash payments; not owner free cash flow or parent cash conversion.")
    return {
        "calculations": calculations,
        "current_multiples": {key: quote[key] for key in ("pe_ttm", "pb") if key in quote},
        "multiple_comparison_candidates": comparisons or [],
        "benchmark_comparison": {
            "status": "analysis_required",
            "required_columns": ["metric", "current_value_ref", "benchmark_value_and_ref", "benchmark_date",
                                 "earnings_or_book_basis", "peer_business_and_cycle_differences", "comparison_conclusion"],
            "rule": "Compare like-for-like PE/TTM/forward or PB, with dates and accounting basis. Do not average unverified peers. "
                    "A retrieved page or provider multiple does not itself establish a benchmark. State unavailable cells explicitly.",
        },
        "sustainable_earnings": {
            "status": "analysis_required",
            "bridge": ["reported parent profit", "documented non-recurring items", "cycle/volume/price/cost changes",
                       "tax and minority interests", "share basis", "sustainable range or explicit inability to estimate"],
            "rule": "Deducted profit is a starting input, not normalized earnings. Do not annualize interim EPS to claim cheapness. "
                    "Scenarios require sourced inputs or explicit analyst assumptions and formulas; otherwise remain qualitative.",
        },
        "conclusion_dimensions": {
            "company_quality": "Profitability, cash, sustainability and material risks; preserve supported findings despite price gaps.",
            "price_attractiveness": "Current multiple versus a comparable benchmark; unavailable comparison means not assessed, not expensive.",
            "direction": "Bullish/bearish/neutral/undetermined with evidence; data failure is not bearish evidence.",
            "current_action": "Buy consideration/wait/avoid/research incomplete. Wait requires a market or valuation reason, "
                              "not just an unavailable endpoint. Keep research status separate from market stance.",
        },
    }


def multiple_comparisons(records, symbol: str, symbols: set[str], quote: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute observed same-day multiple differences, not fair value or peer selection.

    Other authorized companies are candidates only: no automatic median or
    industry classification. A date/source/currency mismatch is not comparable.
    """
    from src.agent.grounding.decision import decision_coverage

    current_price = quote.get("last_price", {})
    rows = []
    for peer in sorted(symbols - {symbol})[:4]:
        coverage = decision_coverage(records, peer)
        peer_ref = coverage["raw_quote"].get("ref") or ""
        peer_call = peer_ref.partition("::")[0]
        peer_fields = {record.field.removeprefix("data."): record for record in records
                       if record.symbol == peer and record.call_id == peer_call
                       and record.tool == "get_a_share_valuation" and record.status == "observed"}
        peer_price = peer_fields.get("last_price")
        for metric in ("pe_ttm", "pb"):
            current, other = quote.get(metric, {}), peer_fields.get(metric)
            if not current.get("ref") or other is None:
                continue
            x, y = current.get("value"), other.value
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in (x, y)):
                continue
            current_day = str(current.get("as_of") or "")[:10]
            peer_day = str(other.timestamp or "")[:10]
            try:
                dated = len(current_day) == 10 and len(peer_day) == 10
                if dated:
                    date.fromisoformat(current_day)
                    date.fromisoformat(peer_day)
            except ValueError:
                dated = False
            comparable = bool(dated and peer_price and current_price.get("currency")
                              and current_price.get("currency") == peer_price.currency
                              and current_day == peer_day
                              and current.get("source") and current.get("source") == other.source)
            difference = (x / y - 1) * 100 if comparable else None
            if difference is not None and not math.isfinite(difference):
                comparable, difference = False, None
            ref = f"{other.call_id}::{other.field}"
            rows.append({"candidate_symbol": peer, "metric": metric,
                         "current_value": x, "current_ref": current["ref"], "current_as_of": current.get("as_of"),
                         "candidate_value": y, "candidate_ref": ref, "candidate_as_of": other.timestamp,
                         "status": "same_day_multiple_difference_only" if comparable else "noncomparable_snapshot",
                         "relative_multiple_difference_pct": difference,
                         "formula": f"({current['ref']} / {ref} - 1) * 100" if comparable else None,
                         "adoption_requires": "Confirm business mix, accounting/share basis, earnings cycle and returns on capital. "
                                              "This is not a verified comparable, fair-value discount or buy signal."})
    return rows


_DIMENSIONS = {
    "公司质量": r"公司质量|企业质量|company quality|business quality",
    "价格吸引力": r"价格吸引力|现价吸引力|price attractiveness|entry valuation",
    "方向倾向": r"方向倾向|方向偏好|directional (?:view|bias)|direction:",
    "当前行动": r"当前行动|current action",
}


def delivery_review(content: str) -> list[str]:
    """One editorial revision, never grounds for withholding a safe report."""
    text = re.sub(r"```figures\b.*?```", "", content, flags=re.S | re.I)
    missing = [name for name, pattern in _DIMENSIONS.items() if not re.search(pattern, text, re.I)]
    requests = []
    if missing:
        requests.append("Separate these missing dimensions with explicit labels: " + ", ".join(missing)
                        + ". Give each its evidence, confidence and gaps; do not repeat one wait verdict in all four.")
    if not re.search(r"历史|同业|同行|可比|benchmark|historical|peer", text, re.I):
        requests.append("Add a valuation benchmark comparison with dates, metric/accounting basis and source refs, "
                        "or explicitly state that none was verified and price attractiveness remains unassessed.")
    if not re.search(r"可持续|正常化|周期|一次性|非经常|sustainab|normaliz|cyclic|non.recurring", text, re.I):
        requests.append("Explain sustainable earnings versus reported growth, or the exact missing inputs preventing that assessment.")
    return requests
