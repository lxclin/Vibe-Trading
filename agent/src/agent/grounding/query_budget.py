"""Reuse exact research receipts and bound failures within one team run.

Current quotes and bars are never cached by this helper. No global/disk cache
or changed-argument guesses: receipt references remain the original call IDs.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

REUSABLE_TOOLS = frozenset({"get_financial_statements", "web_search", "read_url", "financial_rigor"})
BUDGETED_TOOLS = REUSABLE_TOOLS | {"search_symbol", "get_a_share_valuation", "get_market_data"}
MAX_FAILURES = 2


def query_key(tool: str, arguments: dict) -> tuple[str, str] | None:
    if tool not in BUDGETED_TOOLS:
        return None
    try:
        return tool, json.dumps({key: value for key, value in arguments.items() if key != "run_dir"},
                                ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return None


class ResearchQueryBudget:
    def __init__(self, receipts: Iterable[dict[str, Any]] = ()):
        self.successes: dict[tuple[str, str], dict[str, Any]] = {}
        self.failures: dict[tuple[str, str], int] = {}
        self.seen: set[str] = set()
        for receipt in receipts:
            self.record(receipt)

    def record(self, receipt: dict[str, Any]) -> None:
        key = query_key(receipt.get("tool", ""), receipt.get("arguments", {}))
        call_id = receipt.get("call_id")
        if key is None or not call_id or call_id in self.seen:
            return
        self.seen.add(call_id)
        if receipt.get("success") is True:
            try:
                cacheable = isinstance(json.loads(receipt.get("result", "")), dict)
            except (TypeError, ValueError):
                cacheable = False
            if (cacheable and key[0] in REUSABLE_TOOLS and (key in self.successes or len(self.successes) < 64)
                    and len(receipt.get("result", "")) <= 180000
                    and sum(len(item.get("result", "")) for item in self.successes.values())
                        + len(receipt.get("result", "")) <= 1500000):
                self.successes[key] = receipt
        elif receipt.get("success") is False:
            self.failures[key] = self.failures.get(key, 0) + 1

    def lookup(self, tool: str, arguments: dict) -> dict[str, Any] | None:
        if arguments.get("no_cache") or arguments.get("fresh"):
            return None
        return self.successes.get(query_key(tool, arguments))

    def blocked(self, tool: str, arguments: dict) -> bool:
        return self.failures.get(query_key(tool, arguments), 0) >= MAX_FAILURES


def reused_result(receipt: dict[str, Any]) -> str:
    try:
        payload = json.loads(receipt["result"])
    except (TypeError, ValueError):
        payload = {"result": receipt["result"]}
    if not isinstance(payload, dict):
        payload = {"result": payload}
    return json.dumps({**payload, "_research_reuse": {
        "original_call_id": receipt["call_id"],
        "instruction": "Same-run existing result, not a fresh fetch. Cite original_call_id::field. Keep original dates and source.",
    }}, ensure_ascii=False)
