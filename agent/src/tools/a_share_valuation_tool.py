"""Unadjusted A-share quote and provider valuation snapshot.

Historical OHLCV loaders can return dividend-adjusted prices. An A-share
price-to-earnings or price-to-book calculation needs a contemporaneous raw
quote (or market value) instead. This tool keeps that distinction explicit.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from backtest.loaders import eastmoney_client
from src.agent.tools import BaseTool

logger = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^(\d{6})\.(SH|SZ)$", re.IGNORECASE)
_QUOTE_HOSTS = ("push2delay.eastmoney.com", "push2.eastmoney.com")
_QUOTE_PATH = "/api/qt/stock/get"
_FIELDS = "f43,f57,f58,f86,f116,f164,f167"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, "", "-"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _quote_time(value: Any) -> str | None:
    try:
        stamp = int(value)
        if stamp <= 0:
            return None
        return datetime.fromtimestamp(stamp, ZoneInfo("Asia/Shanghai")).isoformat()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


class AShareValuationTool(BaseTool):
    """Fetch a raw quote and same-source PE/PB for one mainland equity."""

    name = "get_a_share_valuation"
    description = (
        "Fetch one Shanghai/Shenzhen A-share's unadjusted quote, market value, "
        "TTM PE and PB from the same Eastmoney quote snapshot. Use before a "
        "current-price valuation or buy/avoid judgment; get_market_data may "
        "return dividend-adjusted historical prices that must not be divided "
        "by unadjusted EPS or book value. Returns source, as-of time, and "
        "individual unavailable fields. This is a delayed public quote, not "
        "an executable real-time price."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Canonical A-share equity code, e.g. 601899.SH or 000001.SZ.",
            },
        },
        "required": ["code"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        code = str(kwargs.get("code") or "").strip().upper()
        match = _SYMBOL_RE.fullmatch(code)
        if match is None:
            return json.dumps({"ok": False, "error": "code must be a .SH or .SZ six-digit A-share symbol"})

        bare, venue = match.groups()
        secid = f"{'1' if venue == 'SH' else '0'}.{bare}"
        last_error: Exception | None = None
        data: dict[str, Any] | None = None
        for host in _QUOTE_HOSTS:
            try:
                payload = eastmoney_client.get_json(
                    f"https://{host}{_QUOTE_PATH}",
                    params={"secid": secid, "fltt": "2", "invt": "2", "fields": _FIELDS},
                )
                candidate = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(candidate, dict):
                    raise ValueError("quote payload has no data object")
                if str(candidate.get("f57") or "").strip() != bare:
                    raise ValueError("quote returned a different security code")
                data = candidate
                break
            except Exception as exc:  # noqa: BLE001 - another quote edge may answer
                logger.debug("A-share quote failed on %s for %s: %s", host, code, exc)
                last_error = exc

        if data is None:
            return json.dumps({
                "ok": False,
                "source": "eastmoney_quote",
                "error": f"A-share quote unavailable: {last_error}",
            }, ensure_ascii=False)

        last_price = _number(data.get("f43"))
        as_of = _quote_time(data.get("f86"))
        if last_price is None or last_price <= 0 or as_of is None:
            return json.dumps({
                "ok": False,
                "source": "eastmoney_quote",
                "error": "A-share quote has no usable price or quote timestamp",
            }, ensure_ascii=False)
        if datetime.now(ZoneInfo("Asia/Shanghai")) - datetime.fromisoformat(as_of) > timedelta(days=14):
            return json.dumps({
                "ok": False,
                "source": "eastmoney_quote",
                "error": f"A-share quote is stale (as_of={as_of})",
            }, ensure_ascii=False)

        result = {
            "ok": True,
            "market": "a_share",
            "source": "eastmoney_quote",
            "data": {
                "symbol": code,
                "name": str(data.get("f58") or "").strip() or None,
                "as_of": as_of,
                "price_adjustment": "raw",
                "last_price": last_price,
                "market_cap_cny": _number(data.get("f116")),
                "pe_ttm": _number(data.get("f164")),
                "pb": _number(data.get("f167")),
            },
        }
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
