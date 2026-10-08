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

import requests

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


def _validate_snapshot(data: dict[str, Any]) -> None:
    price = data.get("last_price")
    as_of = data.get("as_of")
    if price is None or price <= 0 or not as_of:
        raise ValueError("quote has no usable price or quote timestamp")
    age = datetime.now(ZoneInfo("Asia/Shanghai")) - datetime.fromisoformat(as_of)
    if age > timedelta(days=14) or age < -timedelta(minutes=5):
        raise ValueError(f"quote timestamp is stale or in the future (as_of={as_of})")


def _tencent_snapshot(code: str) -> dict[str, Any]:
    """Use only the raw last-price fields; do not infer Tencent PE's basis."""
    bare, venue = code.split(".")
    ticker = venue.lower() + bare
    response = requests.get(
        f"https://qt.gtimg.cn/q={ticker}", timeout=8,
        headers={"Referer": "https://gu.qq.com/", "User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    match = re.fullmatch(rf'\s*v_{ticker}="([^"\r\n]*)";?\s*', response.text)
    if match is None:
        raise ValueError("Tencent returned an unexpected quote envelope")
    fields = match.group(1).split("~")
    if len(fields) <= 30 or fields[2] != bare:
        raise ValueError("Tencent returned a different or incomplete security")
    stamp = datetime.strptime(fields[30], "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    data = {
        "symbol": code, "name": fields[1] or None,
        "as_of": stamp.isoformat(), "price_adjustment": "raw",
        "last_price": _number(fields[3]),
        "market_cap_cny": None, "pe_ttm": None, "pb": None,
    }
    _validate_snapshot(data)
    return data


class AShareValuationTool(BaseTool):
    """Fetch a raw quote and same-source PE/PB for one mainland equity."""

    name = "get_a_share_valuation"
    description = (
        "Fetch one Shanghai/Shenzhen A-share's unadjusted quote, market value, "
        "TTM PE and PB from the same Eastmoney quote snapshot; automatically "
        "falls back to a Tencent raw quote when Eastmoney fails. Tencent fallback "
        "does not supply verified TTM PE/PB. Use before a "
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
        failed_sources: list[str] = []
        source = "eastmoney_quote"
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
                snapshot = {
                    "symbol": code,
                    "name": str(candidate.get("f58") or "").strip() or None,
                    "as_of": _quote_time(candidate.get("f86")),
                    "price_adjustment": "raw",
                    "last_price": _number(candidate.get("f43")),
                    "market_cap_cny": _number(candidate.get("f116")),
                    "pe_ttm": _number(candidate.get("f164")),
                    "pb": _number(candidate.get("f167")),
                }
                _validate_snapshot(snapshot)
                data = snapshot
                break
            except Exception as exc:  # noqa: BLE001 - another quote edge may answer
                logger.debug("A-share quote failed on %s for %s: %s", host, code, exc)
                last_error = exc
                failed_sources.append(host)

        if data is None:
            try:
                data = _tencent_snapshot(code)
                source = "tencent"
            except Exception as exc:  # noqa: BLE001 - report bounded provider failures
                return json.dumps({
                    "ok": False, "source": "eastmoney_quote/tencent",
                    "attempted_sources": [*failed_sources, "tencent"],
                    "error": f"Eastmoney unavailable: {last_error}; Tencent unavailable: {exc}",
                }, ensure_ascii=False)

        result = {
            "ok": True,
            "market": "a_share",
            "source": source,
            "fallback_used": source != "eastmoney_quote",
            "failed_sources": failed_sources,
            "unavailable_fields": [key for key in ("market_cap_cny", "pe_ttm", "pb") if data[key] is None],
            "data": data,
        }
        return json.dumps(result, ensure_ascii=False, allow_nan=False)
