"""Strict dated fundamentals from the official English HSI factsheet layout."""
import math
import re
from datetime import datetime
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo


def public_index_snapshot(url, content):
    try:
        target = urlsplit(url)
        params = parse_qs(target.query)
        if (target.scheme != "https" or target.hostname != "www.hsi.com.hk"
                or target.port not in {None, 443} or target.username or target.password
                or target.path != "/api/wsit-hsi-ddoc-ea-public-website-proxy/v1/content/gcs/download"
                or set(params) != {"resource"} or len(params["resource"]) != 1):
            return None
        path = re.fullmatch(r"/public-website/webapp/factsheets/en-hk/([a-z0-9_-]+)\.pdf", params["resource"][0])
        if not path:
            return None
        code = path[1].upper()
        if not re.search(r"\bBloomberg\s+" + re.escape(code) + r"\b", content):
            return None
        names = set(re.findall(r"(?m)^# (Hang Seng[^\n]+Index)\s*$", content))
        dates = set(re.findall(r"All data as at\s+(\d{1,2} [A-Za-z]{3} \d{4})", content))
        if len(names) != 1 or len(dates) != 1:
            return None
        stamp = datetime.strptime(next(iter(dates)), "%d %b %Y").date()
        if stamp > datetime.now(ZoneInfo("Asia/Shanghai")).date():
            return None
        sections = re.findall(r"INDEX FUNDAMENTALS\s+Index Dividend Yield \(%\) PE Ratio \(Times\) Annual Volatility\*?\(%\)"
                            r"\s+" + re.escape(code) + r"\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\b", content)
        if len(set(sections)) != 1:
            return None
        dy, pe, volatility = map(float, sections[0])
        if not all(math.isfinite(v) for v in (dy, pe, volatility)) or dy < 0 or volatility < 0:
            return None
        return {"index_code": code, "index_name": next(iter(names)), "source": "hang_seng_indexes",
                "as_of": stamp.isoformat(), "dividend_yield_pct": dy, "pe_ratio": pe,
                "annual_volatility_pct": volatility,
                "boundary": "Official dated monthly index fundamentals, not ETF PE, fund NAV, "
                            "intraday data or historical percentile. PE earnings methodology and "
                            "comparability are not inferred from this table."}
    except (ValueError, TypeError, OverflowError):
        return None
