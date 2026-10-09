"""Strict readers for a small set of public financial page formats.

Do not extract arbitrary prose numbers into the quote pool. Unknown layouts,
conflicting values, stale dates and mismatched identities return no snapshot.
"""

import math
import re
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo


def public_financial_snapshot(url: str, content: str) -> dict | None:
    try:
        target = urlsplit(url)
        if target.port not in (None, 443):
            return None
    except ValueError:
        return None
    if target.scheme != "https" or target.username or target.password:
        return None
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    symbol = source = stamp = None
    values = {}
    if target.hostname == "qt.gtimg.cn":
        if target.path.startswith("/q=") and not target.query:
            query = [target.path[3:]]
        elif target.path in {"", "/"}:
            query = parse_qs(target.query).get("q", [])
        else:
            return None
        if len(query) != 1 or not re.fullmatch(r"(?:sh|sz)\d{6}", query[0]):
            return None
        ticker = query[0]
        rows = re.findall(rf'(?m)^\s*v_{ticker}="([^"\r\n]+)";?\s*$', content)
        if len(rows) != 1:
            return None
        fields = rows[0].split("~")
        if len(fields) <= 30 or fields[2] != ticker[2:]:
            return None
        try:
            stamp = datetime.strptime(fields[30], "%Y%m%d%H%M%S").replace(tzinfo=now.tzinfo)
            values["last_price"] = float(fields[3])
        except (ValueError, OverflowError):
            return None
        symbol, source = ticker[2:] + "." + ticker[:2].upper(), "tencent"
    elif target.hostname in {"m.chinaamc.com", "www.chinaamc.com"}:
        match = re.fullmatch(r"/fund/([15]\d{5})/(?:index|jijinfeilv).shtml", target.path)
        if not match or target.query or re.search(r"美元|港币|USD|HKD", content, re.I):
            return None
        code = match[1]
        # Require the dated NAV row itself, not the rounded headline, returns,
        # cumulative NAV or a quote sidebar elsewhere on the page.
        if target.hostname == "m.chinaamc.com":
            if not re.search(rf"_基金代码_\s+{code}\b", content) or "_单位净值_" not in content:
                return None
            rows = re.findall(r"(?m)^\s*(\d{4}-\d{2}-\d{2})\*\*[^*\n]+ETF\*\*_([0-9]+\.[0-9]+)_\s*$", content)
        else:
            if not re.search(rf"基金代码[：:]\s*{code}\b", content) or "ETF" not in content:
                return None
            desktop_rows = re.findall(r"(?m)^\s*([0-9]+\.[0-9]+)\s*\n\s*净值\s*[（(](\d{4}-\d{2}-\d{2})[）)]", content)
            rows = [(stamp, nav) for nav, stamp in desktop_rows]
        if len(set(rows)) != 1:
            return None
        try:
            stamp = datetime.strptime(rows[0][0], "%Y-%m-%d").replace(tzinfo=now.tzinfo)
            values["unit_nav"] = float(rows[0][1])
        except (ValueError, OverflowError):
            return None
        symbol, source = code + (".SH" if code.startswith("5") else ".SZ"), "chinaamc"
    else:
        return None
    if not stamp or not all(math.isfinite(value) and value > 0 for value in values.values()):
        return None
    if now - stamp > timedelta(days=14) or stamp - now > timedelta(minutes=5):
        return None
    return {"symbol": symbol, "source": source, "as_of": stamp.isoformat(),
            "currency": "CNY", **values,
            "boundary": "Dated public snapshot; unit_nav is a daily fund NAV, not a trade price or intraday IOPV."}
