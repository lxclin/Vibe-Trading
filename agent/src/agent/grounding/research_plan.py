"""Bounded source-retrieval plans, separate from investment judgments.

Queries and page reads are progress events, never verified earnings or fair
value. Match activity to a locked company before reusing it in its plan.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urldefrag

_TOPICS = {
    "valuation_benchmark": (
        re.compile(r"历史|同业|同行|可比|分位|中位|\bpeer|historical|percentile|median", re.I),
        re.compile(r"估值|市盈率|市净率|\bp/?e\b|\bp/?b\b|valuation", re.I),
    ),
    "earnings_drivers": (
        re.compile(r"正常化|可持续|盈利驱动|利润驱动|产量|单位成本|净息差|拨备|"
                   r"一次性损益|非经常|股本变动|normalized|sustainable|earnings drivers", re.I),
    ),
}


def _url(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 2000:
        return None
    value = value.strip()
    try:
        target = urlsplit(value)
        if target.scheme not in {"http", "https"} or not target.hostname or target.username or target.password:
            return None
    except ValueError:
        return None
    return urldefrag(value)[0]


def _mentions(query: str, symbol: str, names: Sequence[str]) -> bool:
    code = symbol.split(".", 1)[0]
    return bool(re.search(rf"(?<!\d){re.escape(code)}(?!\d)", query)) or any(
        len(name) >= 2 and name.casefold() in query.casefold() for name in names
    )


def research_attempt(
    tool: str, arguments: Mapping[str, Any], payload: Mapping[str, Any] | None,
    call_id: str, success: bool, companies: Mapping[str, Sequence[str]],
    previous: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Record scoped source activity without copying page contents into policy."""
    payload = payload or {}
    if tool == "web_search":
        query = str(arguments.get("query") or "")[:600]
        symbols = [symbol for symbol, names in companies.items() if _mentions(query, symbol, names)]
        topics = [topic for topic, patterns in _TOPICS.items() if all(pattern.search(query) for pattern in patterns)]
        if not symbols or not topics:
            return None
        rows = payload.get("results")
        urls = list(dict.fromkeys(
            url for row in (rows if isinstance(rows, list) else []) if isinstance(row, dict)
            if (url := _url(row.get("url") or row.get("href")))
        ))[:3]
        return {"tool": tool, "call_id": call_id, "symbols": symbols, "topics": topics,
                "query": query, "urls": urls if success else [],
                "status": "source_candidates_found" if success and urls else "unavailable"}
    if tool == "read_url":
        url = _url(arguments.get("url"))
        if not url:
            return None
        linked = [attempt for attempt in previous if url and url in attempt.get("urls", [])]
        symbols = sorted({symbol for attempt in linked for symbol in attempt.get("symbols", []) if symbol in companies})
        topics = sorted({topic for attempt in linked for topic in attempt.get("topics", [])})
        content = payload.get("content")
        readable = success and payload.get("status") == "ok" and isinstance(content, str) and bool(content.strip())
        return {"tool": tool, "call_id": call_id, "symbols": symbols, "topics": topics,
                "urls": [url], "status": "source_read_analysis_pending" if readable else "unavailable"}
    return None


def company_research_plan(
    symbol: str, names: Sequence[str], report_date: str | None,
    coverage: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]], *, sector: str,
) -> dict[str, Any]:
    """Offer at most one search and one source read for each research topic."""
    ready = all(coverage[key]["status"] == "observed" and coverage[key].get("freshness") != "stale"
                for key in ("raw_quote", "profitability", "operating_cash_flow"))
    ready = ready and coverage.get("financial_periods_match") is True
    subject = symbol + (" " + names[0][:60] if names else "")
    period = report_date[:4] if report_date else "最新"
    quote_year = str(coverage["raw_quote"].get("as_of") or period)[:4]
    # Use resolved company names only to choose retrieval terms, never to infer
    # financial facts or a valuation conclusion.
    name_text = " ".join(names)
    focus = sector
    if sector == "industrial":
        focus = ("semiconductor" if re.search(r"半导体|存储|芯片|长鑫|中芯", name_text)
                 else "resources" if re.search(r"矿业|黄金|铜业|铝业|煤业|煤炭", name_text)
                 else "industrial")
    drivers = {
        "bank": "净息差 不良贷款 拨备 非息收入",
        "insurance": "保险服务利润 新业务价值 投资收益 偿付能力",
        "financial": "经常性收入 投资收益 风险成本 资本充足",
        "semiconductor": "产品价格 需求 库存 产能利用率 研发 折旧 资本开支",
        "resources": "金属或能源价格 产量 单位成本 储量 项目投产 资本开支",
    }.get(focus, "收入结构 量价变化 毛利率 成本 一次性损益 资本开支")
    queries = {
        "valuation_benchmark": f"{subject} {quote_year} 历史估值分位 同业可比 市盈率 市净率 统计日期",
        "earnings_drivers": f"{subject} {period} 财报 可持续盈利 一次性损益 {drivers}",
    }
    topics, actions = [], []
    for topic in _TOPICS:
        scoped = [attempt for attempt in attempts
                  if symbol in attempt.get("symbols", []) and topic in attempt.get("topics", [])]
        searches = [attempt for attempt in scoped if attempt.get("tool") == "web_search"]
        candidates = list(dict.fromkeys(url for attempt in searches for url in attempt.get("urls", [])))[:3]
        # A page may have been read before a scoped search returned its URL.
        # Reuse that exact page without guessing its subject from arbitrary text.
        reads = [attempt for attempt in attempts if attempt.get("tool") == "read_url" and (
            (symbol in attempt.get("symbols", []) and topic in attempt.get("topics", []))
            or any(url in candidates for url in attempt.get("urls", []))
        )]
        status = ("source_read_analysis_pending" if any(attempt["status"] == "source_read_analysis_pending" for attempt in reads)
                  else "unavailable" if reads else "source_candidates_found" if candidates
                  else "unavailable" if searches else "unchecked")
        topics.append({"topic": topic, "status": status,
                       "call_ids": list(dict.fromkeys(attempt["call_id"] for attempt in [*scoped, *reads]))[-4:],
                       "candidate_urls": candidates})
        if not ready:
            continue
        if not searches:
            actions.append({"topic": topic, "tool": "web_search",
                            "arguments": {"query": queries[topic], "max_results": 3}})
        elif candidates and not reads:
            # Prefer original disclosures, while keeping other pages explicitly
            # labelled as candidates. Reading any page still requires analysis.
            preferred = sorted(candidates, key=lambda url: urlsplit(url).hostname not in {
                "www.sse.com.cn", "www.szse.cn", "static.cninfo.com.cn", "www.cninfo.com.cn",
            })
            actions.append({"topic": topic, "tool": "read_url", "arguments": {"url": preferred[0]}})
    return {"prerequisites_ready": ready, "driver_search_focus": focus, "topics": topics, "priority_actions": actions[:2],
            "stop_rule": "Recommend one scoped search and one candidate read per topic; paginate that document only if needed. "
                         "A failed read is a disclosed gap, "
                         "not an instruction to retry. No extra final-answer rejection is created by this plan.",
            "boundary": "Found URLs and read pages are not verified benchmarks or normalized earnings. "
                        "Check dates, accounting scope, share basis and source contents before adopting a number."}
