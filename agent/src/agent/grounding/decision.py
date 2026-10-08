"""A narrow output contract for single-instrument buy questions.

The model still evaluates the investment thesis. This module only makes it
answer the question directly and prevents an actionable positive conclusion
when this run never obtained either prices or company financials.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Sequence
from zoneinfo import ZoneInfo


_BUY_QUESTION_RE = re.compile(
    r"值得(?:现在)?买入|值得买吗|能否买入|能买吗|适合买入|是否(?:值得)?买入|"
    r"现在(?:能|该|要不要)买|买入时机|"
    r"\b(?:should I buy|worth buying|buy now|good time to buy)\b",
    re.IGNORECASE,
)
_META_OR_COMPARISON_RE = re.compile(
    r"怎么问|如何提问|提示词|prompt|项目|系统|模型|代码|ETF|基金|"
    r"哪个更|哪只更|对比|比较|\b(?:compare|versus|vs\.?|how to ask)\b",
    re.IGNORECASE,
)
_STANCE_RE = re.compile(r"(?:研究)?结论\s*[:：]\s*([^\n。；;]{1,100})", re.IGNORECASE)
_ENGLISH_STANCE_RE = re.compile(r"\b(?:research )?(?:verdict|conclusion)\s*:\s*([^\n.;]{1,100})", re.IGNORECASE)
_HORIZON_RE = re.compile(
    r"(?:持有|分析|判断)?期限\s*[:：]\s*[*_`~\s]*(?:未来|短线|中线|长线|\d+\s*(?:[—–-]\s*\d+\s*)?(?:天|周|月|年))|"
    r"\b(?:horizon|timeframe)\s*:\s*(?:the requested|\d+|short|medium|long|six|one)",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(r"(?:信心|置信度)\s*[:：]\s*(?:高|中|低)|\bconfidence\s*:\s*(?:high|medium|low)\b", re.IGNORECASE)
_RATIONALE_RE = re.compile(r"(?:主要|核心)?依据\s*[:：]|\b(?:key evidence|rationale)\s*:", re.IGNORECASE)
_CHANGE_RE = re.compile(r"改变判断的条件|判断.{0,8}(?:改变|失效)|失效条件|\b(?:what changes the view|invalidation conditions?)\b", re.IGNORECASE)
_MISSING_RE = re.compile(r"缺少|缺失|未取得|未获取|未核验|无法获取|数据不足|\b(?:missing|unavailable|not retrieved)\b", re.IGNORECASE)
_ADJUSTMENT_RE = re.compile(r"复权|调整后|adjusted|dividend.adjusted", re.IGNORECASE)
_HISTORICAL_PRICE_RE = re.compile(r"收盘价|历史价格|日线价|closing price|historical close", re.IGNORECASE)
_VALUATION_RATIO_RE = re.compile(r"(?:PE|P/E|PB|P/B|市盈率|市净率)\s*(?:约|为|=|:|：)?\s*\d", re.IGNORECASE)
_VALUATION_DISCLOSURE_RE = re.compile(
    r"(?:PE|P/E|PB|P/B|市盈率|市净率).{0,40}(?:\d|未取得|不可用|不采用|无法|not usable|unavailable)",
    re.IGNORECASE,
)
_OBSERVABLE_CONDITION_RE = re.compile(
    r"高于|低于|不超过|不低于|跌破|站上|收复|中位数|分位|同比|环比|"
    r"连续|至少|比较|对照|threshold|above|below|median|percentile|compare",
    re.IGNORECASE,
)
_CRITICAL_GAPS = (
    re.compile(r"金价|铜价|金铜价格|commodity prices?", re.IGNORECASE),
    re.compile(r"正常化盈利|可持续盈利|normalized earnings?", re.IGNORECASE),
    re.compile(r"可比.{0,8}估值|同业.{0,8}估值|peer valuation", re.IGNORECASE),
)
_VALUATION_BENCHMARK_RE = re.compile(
    r"同业|可比公司|历史分位|历史均值|中位数|正常化盈利|可持续盈利|"
    r"peer|historical (?:range|median|percentile)|normalized earnings",
    re.IGNORECASE,
)

_WAIT = ("暂不买入", "暂缓买入", "等待", "观望", "暂不考虑买入", "wait", "watch")
_AVOID = ("不值得买入", "回避", "避免买入", "avoid", "unfavorable")
_FAVORABLE = ("可考虑买入", "值得买入", "可以买入", "favorable", "consider buying")
_PROFIT_FIELDS = frozenset({"PARENTNETPROFIT", "PARENT_NETPROFIT", "NETPROFIT", "EPSJB"})
_OPERATING_CASH_FIELDS = frozenset({"NETCASH_OPERATE", "NETCASH_OPERATE_PK"})
_MAX_REPORT_AGE_DAYS = 270


def _latest_financial_fact(records: Sequence[Any], symbol: str, fields: frozenset[str]) -> Any | None:
    """Return a relevant numeric fact from the latest reported period."""
    candidates = [
        record for record in records
        if record.tool == "get_financial_statements"
        and record.symbol == symbol
        and record.status == "observed"
        and isinstance(record.value, (int, float))
        and re.search(r"\.periods\[\d+\]\.[^.]+$", record.field)
        and record.field.rsplit(".", 1)[-1].upper() in fields
    ]
    return max(candidates, key=lambda record: record.timestamp or "") if candidates else None


def decision_coverage(
    records: Sequence[Any], symbol: str, attempts: Sequence[Any] = (),
) -> dict[str, Any]:
    """Summarize observed research inputs separately from attempted retrievals."""
    quotes = [
        record for record in records
        if record.tool == "get_a_share_valuation" and record.symbol == symbol
        and record.status == "observed"
    ]
    quote = next((record for record in reversed(quotes) if record.field == "data.last_price"
                  and isinstance(record.value, (int, float)) and record.value > 0), None)
    multiple = next((record for record in reversed(quotes) if record.field in {"data.pe_ttm", "data.pb"}
                     and isinstance(record.value, (int, float)) and record.value > 0), None)
    profit = _latest_financial_fact(records, symbol, _PROFIT_FIELDS)
    cash = _latest_financial_fact(records, symbol, _OPERATING_CASH_FIELDS)
    relevant_attempts = [
        attempt for attempt in attempts
        if isinstance(attempt, dict) and attempt.get("symbol") == symbol
    ]
    def slot(record: Any | None, tool: str, statement: str | None = None) -> dict[str, Any]:
        matching = [attempt for attempt in relevant_attempts if attempt.get("tool") == tool
                    and (statement is None or attempt.get("statement") == statement)]
        return {
            "status": "observed" if record is not None else ("unavailable" if matching else "unchecked"),
            "field": record.field if record is not None else None,
            "as_of": record.timestamp if record is not None else None,
            "source": record.source if record is not None else None,
            "report_period": getattr(record, "report_period", None) if record is not None else None,
            "attempted": bool(matching) or record is not None,
            "quarter_attempted": any(attempt.get("period") == "quarter" for attempt in matching),
        }
    coverage = {
        "symbol": symbol,
        "raw_quote": slot(quote, "get_a_share_valuation"),
        "valuation_multiple": slot(multiple, "get_a_share_valuation"),
        "profitability": slot(profit, "get_financial_statements", "indicators"),
        "operating_cash_flow": slot(cash, "get_financial_statements", "cashflow"),
        "financial_periods_match": (
            bool(profit.timestamp and cash.timestamp)
            and profit.timestamp[:10] == cash.timestamp[:10]
            if profit is not None and cash is not None else None
        ),
    }
    for key in ("profitability", "operating_cash_flow"):
        as_of = coverage[key]["as_of"]
        if not as_of:
            continue
        try:
            report_date = date.fromisoformat(as_of[:10])
        except ValueError:
            coverage[key]["freshness"] = "unknown"
            continue
        age = (datetime.now(ZoneInfo("Asia/Shanghai")).date() - report_date).days
        coverage[key]["freshness"] = "stale" if age > _MAX_REPORT_AGE_DAYS else "current"
    return coverage


def is_single_buy_question(question: str) -> bool:
    """Select a direct one-instrument entry question, excluding meta prompts."""
    # Bare mainland fund codes need the ETF workflow too, even when the user
    # omits the word ETF. A fund has no issuer operating cash-flow statement.
    if re.search(r"(?<!\d)(?:5\d{5}|1[568]\d{4})(?:\.(?:SH|SZ))?(?!\d)", question, re.IGNORECASE):
        return False
    return bool(_BUY_QUESTION_RE.search(question)) and not bool(
        _META_OR_COMPARISON_RE.search(question)
    )


def decision_guidance(chinese: bool) -> str:
    """Tell the model the exact, auditable research-answer shape."""
    if chinese:
        return (
            "[单一标的买入研究] 先确认标的和持有期限；用户未给期限时明确采用未来6—12个月作为分析假设。"
            "沪深A股先调用get_a_share_valuation取得未复权报价、时点、同口径PE/PB和市值，再取最新财报；"
            "财报优先取get_financial_statements的quarter/indicators与quarter/cashflow，核对报告期、归母利润和经营现金流。"
            "如果同一工具结果已经包含这些字段，可以复用；若缺少要如实列为证据缺口。"
            "get_market_data复权日线只用于走势，不能直接除以未复权EPS或每股净资产。"
            "评估估值、盈利/现金流与主要风险；算过但不采用的估值要说明口径原因。"
            "周期股不能只凭半年EPS乘二认定便宜；核对商品价格、正常化盈利及可比估值，取不到就列明缺口。"
            "正面买入结论还需说明所用的历史或同业估值基准。"
            "第一段按固定字段回答：结论：可考虑买入/暂不买入（等待）/回避；期限：…；信心：高/中/低。"
            "随后写‘主要依据：’和‘改变判断的条件：’，条件写出可核对指标和比较基准。"
            "关键估值报价取不到时只能暂不买入、信心低，并明确说明数据缺口。"
            "这是基于现有证据的研究判断，"
            "不是个人化交易指令；不要用三种情景代替当前判断。"
            "若行情或财报缺失，只能给‘暂不买入（等待）’，明确指出缺少什么数据。"
        )
    return (
        "[Single-instrument entry research] State the instrument and assumed horizon "
        "(default 6-12 months if unspecified). For a Shanghai/Shenzhen equity, call "
        "get_a_share_valuation for an unadjusted quote, as-of time, PE/PB and market cap. "
        "Retrieve quarterly indicators and cash-flow statements, checking report dates, "
        "parent profit and operating cash flow. Reuse fields already returned by a tool; "
        "name genuinely missing items as evidence gaps. "
        "Use adjusted get_market_data bars for trends only, never as the numerator "
        "of PE/PB against unadjusted per-share accounts. Obtain the latest financial "
        "statements; assess valuation, earnings/cash flow and key risks. "
        "For cyclicals, do not infer cheapness from annualizing a half-year EPS alone. "
        "A favorable verdict also needs a historical or peer valuation benchmark. "
        "Begin with Verdict: favorable / wait / avoid; Horizon: ...; Confidence: high / "
        "medium / low. Add Key evidence: and What changes the view: with an observable "
        "metric and comparison benchmark. If a necessary unadjusted quote is unavailable, "
        "choose wait with low confidence and name the gap. This is a research "
        "assessment, not a personalized trade order. If price or financial evidence is "
        "missing, choose wait and identify the missing data."
    )


def _stance(content: str) -> str | None:
    """Read the declared position near the start without inferring it from prose."""
    opening = content[:900]
    found = _STANCE_RE.search(opening) or _ENGLISH_STANCE_RE.search(opening)
    if not found:
        return None
    value = re.sub(r"^[\s*_`~]+", "", found.group(1)).casefold()
    value = re.sub(r"^(?:(?:当前|目前|现阶段|研究上|at present|currently)\s*)+", "", value)
    if re.search(r"(?:/|／|或|\bor\b).{0,20}(?:买入|等待|观望|回避|wait|avoid|favorable)", value):
        return None
    if value.startswith(_WAIT):
        return "wait"
    if value.startswith(_AVOID):
        return "avoid"
    if value.startswith(_FAVORABLE):
        return "favorable"
    return None


def decision_issues(
    content: str,
    records: Sequence[Any],
    primary_symbols: set[str],
    tool_failures: Sequence[Any] = (),
    attempts: Sequence[Any] = (),
) -> list[dict[str, Any]]:
    """Require a directional, time-bound research view with minimum evidence."""
    issues: list[dict[str, Any]] = []

    def issue(code: str, message: str) -> None:
        issues.append({
            "code": code, "value": None, "role": None, "span": None,
            "symbol": next(iter(primary_symbols)) if len(primary_symbols) == 1 else None,
            "reason": code, "message": message,
        })

    stance = _stance(content)
    if stance is None:
        issue(
            "decision_stance_missing",
            "Begin with an explicit research verdict: 结论：可考虑买入 / 暂不买入（等待） / 回避. Do not substitute balanced scenarios.",
        )
    if not _HORIZON_RE.search(content[:900]):
        issue("decision_horizon_missing", "State the investment horizon explicitly as 期限：… / Horizon: … .")
    if not _CONFIDENCE_RE.search(content[:900]):
        issue("decision_confidence_missing", "State confidence explicitly as 信心：高/中/低 or Confidence: high/medium/low.")
    if not _RATIONALE_RE.search(content):
        issue("decision_evidence_missing", "Add 主要依据： / Key evidence: supporting the chosen verdict.")
    if not _CHANGE_RE.search(content):
        issue("decision_invalidation_missing", "Add 改变判断的条件： / What changes the view: with observable conditions.")
    else:
        match = _CHANGE_RE.search(content)
        assert match is not None
        if not _OBSERVABLE_CONDITION_RE.search(content[match.end():match.end() + 500]):
            issue(
                "decision_condition_not_observable",
                "Give at least one measurable comparison or threshold in the change condition (for example PE versus a named peer median), without inventing unavailable numbers.",
            )

    confidence_match = re.search(
        r"(?:信心|置信度)\s*[:：]\s*(高|中|低)|\bconfidence\s*:\s*(high|medium|low)\b",
        content[:900], re.IGNORECASE,
    )
    confidence = (confidence_match.group(1) or confidence_match.group(2)).casefold() if confidence_match else None

    if stance is not None and len(primary_symbols) == 1:
        symbol = next(iter(primary_symbols))
        has_price = any(
            record.status == "observed"
            and record.symbol == symbol
            and record.field in {"open", "high", "low", "close", "adj_close", "price"}
            and record.value is not None
            for record in records
        )
        has_financials = any(
            record.status == "observed" and record.symbol == symbol
            and record.tool == "get_financial_statements" and record.value is not None
            and ".periods[" in record.field
            for record in records
        )
        mainland_equity = symbol.endswith((".SH", ".SZ", ".BJ")) and not symbol.startswith(
            ("5", "1")
        )
        quote_records = [
            record for record in records
            if record.tool == "get_a_share_valuation" and record.symbol == symbol
        ]
        quote_attempted = bool(quote_records) or (
            not attempts and any(
                failure.get("tool") == "get_a_share_valuation"
                for failure in tool_failures if isinstance(failure, dict)
            )
        )
        has_raw_quote = any(
            record.status == "observed"
            and record.field == "data.last_price"
            and isinstance(record.value, (int, float))
            and record.value > 0
            for record in quote_records
        )
        has_price = has_price or has_raw_quote
        has_valuation = any(
            record.status == "observed"
            and record.field in {"data.pe_ttm", "data.pb"}
            and isinstance(record.value, (int, float))
            and record.value > 0
            for record in quote_records
        )
        coverage = decision_coverage(records, symbol, attempts) if mainland_equity else None
        if coverage is not None:
            profit = coverage["profitability"]
            cash = coverage["operating_cash_flow"]
            has_financials = profit["status"] == "observed"
            quote_attempted = coverage["raw_quote"]["attempted"] or quote_attempted
            if profit["status"] == "unchecked":
                issue(
                    "decision_profitability_not_checked",
                    f"Call get_financial_statements(code=\"{symbol}\", statement=\"indicators\", period=\"quarter\") to check latest profitability and its report date.",
                )
            if cash["status"] == "unchecked":
                issue(
                    "decision_cash_flow_not_checked",
                    f"Call get_financial_statements(code=\"{symbol}\", statement=\"cashflow\", period=\"quarter\") to check operating cash flow and its report date.",
                )
            if profit["status"] != "observed":
                if stance == "favorable":
                    issue("decision_profitability_missing", "A favorable buy view requires observed latest-period profitability, not just any financial field.")
                if confidence not in {"低", "low"}:
                    issue("decision_confidence_overstated", "Profitability was not observed; use low confidence and name the gap.")
            if cash["status"] != "observed":
                if stance == "favorable":
                    issue("decision_cash_flow_missing", "A favorable buy view requires observed operating cash flow or a clearly sourced substitute.")
                if confidence not in {"低", "low"}:
                    issue("decision_confidence_overstated", "Operating cash flow was not observed; use low confidence and name the gap.")
            for label, slot in (("profitability", profit), ("operating cash flow", cash)):
                if slot["report_period"] == "annual" and not slot["quarter_attempted"]:
                    issue(
                        "decision_quarterly_financials_not_checked",
                        f"Only annual {label} was observed. Check the latest quarterly report before an entry judgment.",
                    )
                if slot.get("freshness") == "stale":
                    if stance == "favorable":
                        issue("decision_stale_financials", f"The latest observed {label} is more than {_MAX_REPORT_AGE_DAYS} days old; do not give a favorable entry verdict without current financials.")
                    if confidence not in {"低", "low"}:
                        issue("decision_confidence_overstated", f"The latest observed {label} is stale; use low confidence.")
                if slot["status"] == "observed" and slot["as_of"]:
                    date_text = slot["as_of"][:10]
                    localized = date_text.replace("-", "年", 1).replace("-", "月", 1) + "日"
                    if date_text not in content and localized not in content:
                        issue("decision_report_date_omitted", f"State the report period {date_text} for observed {label} figures.")
            if (profit["status"] != "observed" or cash["status"] != "observed") and not _MISSING_RE.search(content):
                issue("decision_missing_data_unspecified", "Name the missing profitability or cash-flow evidence explicitly.")
            if coverage["financial_periods_match"] is False:
                if stance == "favorable":
                    issue(
                        "decision_financial_period_mismatch",
                        "Profit and operating cash flow refer to different report periods. Reconcile the periods before a favorable entry view; do not compare them as same-period evidence.",
                    )
                if confidence not in {"低", "low"}:
                    issue("decision_confidence_overstated", "Financial report periods differ; use low confidence and disclose the mismatch.")
        if mainland_equity and not quote_attempted:
            issue(
                "decision_raw_valuation_not_checked",
                f"Before answering a buy question for {symbol}, call get_a_share_valuation(code=\"{symbol}\") to obtain an unadjusted quote and valuation basis.",
            )
        if mainland_equity and (not has_raw_quote or not has_valuation):
            if stance != "wait":
                issue(
                    "decision_valuation_basis_missing",
                    "Without an observed unadjusted quote and valuation basis, choose 暂不买入（等待） / wait.",
                )
            if confidence not in {"低", "low"}:
                issue(
                    "decision_confidence_overstated",
                    "A missing unadjusted quote or valuation basis limits this buy judgment to low confidence.",
                )
            if not _MISSING_RE.search(content):
                issue("decision_missing_data_unspecified", "Name the unavailable quote or valuation field.")
            if _VALUATION_RATIO_RE.search(content):
                issue(
                    "decision_adjusted_valuation_unsupported",
                    "Do not report a numeric PE/PB from adjusted OHLC prices and unadjusted per-share financials.",
                )
        if mainland_equity and has_raw_quote and has_valuation and not _VALUATION_DISCLOSURE_RE.search(content):
            issue(
                "decision_valuation_omitted",
                "The raw quote returned PE/PB. Report at least one observed multiple with its source and as-of time, or explain specifically why the multiple is unusable.",
            )
        if mainland_equity and stance == "favorable" and not _VALUATION_BENCHMARK_RE.search(content):
            issue(
                "decision_valuation_benchmark_missing",
                "A favorable A-share entry view needs a named historical or peer valuation benchmark, or a normalized-earnings comparison; an isolated PE/PB is insufficient.",
            )
        adjusted_prices = any(
            record.tool == "get_market_data"
            and record.symbol == symbol
            and record.field in {"open", "high", "low", "close"}
            and getattr(record, "adjustment", None) in {
                "split", "split_dividend", "split_dividend_additive"
            }
            for record in records
        )
        if adjusted_prices and _HISTORICAL_PRICE_RE.search(content) and not _ADJUSTMENT_RE.search(content):
            issue(
                "decision_adjusted_price_unlabeled",
                "If citing a historical OHLC price, state that the returned series is adjusted and cannot be treated as a raw executable quote.",
            )
        if confidence not in {"低", "low"} and _MISSING_RE.search(content) and sum(
            bool(pattern.search(content)) for pattern in _CRITICAL_GAPS
        ) >= 2:
            issue(
                "decision_confidence_overstated",
                "When multiple key valuation or earnings drivers are explicitly missing, use low confidence or obtain the missing evidence before claiming medium/high confidence.",
            )
        if not (has_price and has_financials):
            if stance != "wait":
                issue(
                    "decision_evidence_insufficient",
                    "Without both observed price and company financials in this run, choose 暂不买入（等待） / wait rather than a favorable or avoid verdict.",
                )
            elif not _MISSING_RE.search(content):
                issue("decision_missing_data_unspecified", "Name the missing price or financial data behind the wait verdict.")
    return issues
