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


class SessionResearchCache:
    """Bounded original receipts for explicitly requested same-session reuse.

    Thirty-minute retention is a snapshot policy, never a promise of freshness.
    Quotes/bars/search identity are excluded. Existing authorization and normal
    evidence ingestion still apply when a receipt is reused.
    """
    def __init__(self, path, *, now: float, enabled: bool):
        from pathlib import Path
        self.path = Path(path) if path is not None else None
        self.now = now
        self.enabled = enabled
        self.items = []
        if self.path is not None:
            try:
                if self.path.stat().st_size <= 1600000:
                    packet = json.loads(self.path.read_text())
                    self.items = [item for item in packet if isinstance(item, dict)
                                  and isinstance(item.get("fetched_at"), (int, float))
                                  and 0 <= now - item["fetched_at"] <= 1800][:64]
            except (OSError, TypeError, ValueError):
                self.items = []

    def lookup(self, tool, arguments, symbols):
        from src.agent.grounding.research_handoff import scoped_receipts
        if not self.enabled or arguments.get("fresh") or arguments.get("no_cache"):
            return None
        key = query_key(tool, arguments)
        return next((item for item in scoped_receipts(self.items, symbols)
                     if item.get("success") is True and tool in REUSABLE_TOOLS
                     and query_key(item["tool"], item["arguments"]) == key), None)

    def record(self, tool, arguments, result, call_id, symbols):
        from src.agent.grounding.research_handoff import capture_receipt, merge_receipts
        if self.path is None or tool not in REUSABLE_TOOLS:
            return
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return
        if (not isinstance(payload, dict) or payload.get("ok") is False
                or payload.get("success") is False or payload.get("status") == "error"
                or payload.get("_research_reuse")
                or (tool == "read_url" and isinstance(payload.get("web_financial"), dict)
                    and "last_price" in payload["web_financial"])):
            return
        receipt = capture_receipt(tool, arguments, result, call_id, True, symbols)
        if receipt is None:
            return
        receipt["fetched_at"] = self.now
        self.items = merge_receipts([receipt], self.items)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.items, ensure_ascii=False))
            temporary.replace(self.path)
        except OSError:
            pass  # Research can proceed if this optional cache cannot persist.
