"""Bounded transport of actual research tool returns between team stages.

Reports and worksheets are not tool receipts. Capture happens at execution,
before context compaction, and receivers replay receipts through the ordinary
evidence parser instead of trusting model-written numeric declarations.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

RESEARCH_TOOLS = frozenset({
    "search_symbol", "get_a_share_valuation", "get_financial_statements",
    "get_market_data", "financial_rigor", "web_search", "read_url",
})
MAX_RECEIPTS = 64
MAX_RECEIPT_CHARS = 180000
MAX_PACKET_CHARS = 1500000


def capture_receipt(
    tool: str, arguments: Mapping[str, Any], result: str, call_id: str,
    success: bool, symbols: Iterable[str],
) -> dict[str, Any] | None:
    if tool not in RESEARCH_TOOLS or not call_id or len(result) > MAX_RECEIPT_CHARS:
        return None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    scope = sorted(set(symbols))
    code = str(arguments.get("code") or "").upper()
    query = str(arguments.get("query") or "").upper()
    if tool in {"get_a_share_valuation", "get_financial_statements"} and code in scope:
        scope = [code]
    elif tool == "search_symbol" and query in scope:
        scope = [query]
    elif tool == "get_market_data" and isinstance(arguments.get("codes"), list):
        requested = arguments["codes"]
        if all(isinstance(item, str) and item.upper() in scope for item in requested):
            scope = sorted({item.upper() for item in requested})
    elif tool == "web_search":
        mentioned = [symbol for symbol in scope if symbol in query or symbol.split(".")[0] in query]
        if mentioned:
            scope = mentioned
    if not scope:
        return None
    return {"tool": tool, "arguments": dict(arguments), "result": result,
            "call_id": call_id, "success": success, "scope_symbols": scope}


def merge_receipts(existing: list[dict[str, Any]], incoming: Iterable[Any]) -> list[dict[str, Any]]:
    """Retain first-seen original call IDs within a bounded packet."""
    merged, seen, size = [], set(), 0
    for item in [*existing, *incoming]:
        if not isinstance(item, dict) or not isinstance(item.get("call_id"), str):
            continue
        key = item["call_id"]
        if key in seen or item.get("tool") not in RESEARCH_TOOLS:
            continue
        encoded = json.dumps(item, ensure_ascii=False)
        if len(merged) >= MAX_RECEIPTS or size + len(encoded) > MAX_PACKET_CHARS:
            continue
        merged.append(item)
        seen.add(key)
        size += len(encoded)
    return merged


def scoped_receipts(items: Any, authorized_symbols: set[str]) -> list[dict[str, Any]]:
    """Accept same-subject tool returns, never grant a new identity via handoff."""
    if not isinstance(items, list):
        return []
    accepted = []
    for item in merge_receipts([], items[:MAX_RECEIPTS]):
        scope = item.get("scope_symbols")
        args, result = item.get("arguments"), item.get("result")
        if (not isinstance(scope, list) or not scope or not all(isinstance(s, str) for s in scope)
                or not set(scope).issubset(authorized_symbols)
                or not isinstance(args, dict) or not isinstance(result, str)
                or len(result) > MAX_RECEIPT_CHARS or not isinstance(item.get("success"), bool)):
            continue
        tool = item["tool"]
        if tool in {"get_a_share_valuation", "get_financial_statements"}:
            if str(args.get("code") or "").upper() not in authorized_symbols:
                continue
        elif tool == "search_symbol":
            # Name-based identity resolution stays with the original resolver.
            if str(args.get("query") or "").upper() not in authorized_symbols:
                continue
        elif tool == "get_market_data":
            codes = args.get("codes")
            if not isinstance(codes, list) or not codes or not all(
                isinstance(code, str) and code.upper() in authorized_symbols for code in codes
            ):
                continue
        try:
            payload = json.loads(result)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        failed = payload.get("ok") is False or payload.get("success") is False or payload.get("status") == "error"
        if item["success"] and failed:
            continue
        accepted.append(item)
    return accepted


def swarm_context_result(result: str) -> str:
    """Keep the committee report visible without copying raw receipts or reports twice."""
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("final_report"), str):
        return result
    view = {key: payload[key] for key in (
        "status", "wait_budget_exhausted", "run_id", "preset", "error", "token_usage",
    ) if key in payload}
    report = payload["final_report"]
    view["final_report"] = report[:22000]
    view["report_truncated"] = len(report) > 22000
    tasks = payload.get("tasks")
    view["tasks"] = [{key: task.get(key) for key in ("id", "status", "iterations", "error")}
                     for task in (tasks if isinstance(tasks, list) else [])[:16] if isinstance(task, dict)]
    view["evidence_handoff"] = (
        "Original tool receipts are delivered separately to the host evidence ledger. "
        "Consult the current worksheet for accepted values, exact refs and unresolved gaps. "
        "The report is a research synthesis, not independent numeric evidence. "
        "Deliver a complete answer integrating company quality, valuation and direction; do not only narrate a supplementary lookup."
    )
    return json.dumps(view, ensure_ascii=False)
