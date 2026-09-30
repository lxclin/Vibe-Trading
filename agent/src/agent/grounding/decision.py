"""A narrow output contract for single-instrument buy questions.

The model still evaluates the investment thesis. This module only makes it
answer the question directly and prevents an actionable positive conclusion
when this run never obtained either prices or company financials.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


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


def is_single_buy_question(question: str) -> bool:
    """Select a direct one-instrument entry question, excluding meta prompts."""
    return bool(_BUY_QUESTION_RE.search(question)) and not bool(
        _META_OR_COMPARISON_RE.search(question)
    )


def decision_guidance(chinese: bool) -> str:
    """Tell the model the exact, auditable research-answer shape."""
    if chinese:
        return (
            "[单一标的买入研究] 先确认标的和持有期限；用户未给期限时明确采用未来6—12个月作为分析假设。"
            "沪深A股先调用get_a_share_valuation取得未复权报价、时点、同口径PE/PB和市值，再取最新财报；"
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
            record.status == "observed"
            and record.symbol == symbol
            and record.tool == "get_financial_statements"
            and record.value is not None
            and ".periods[" in record.field
            and not record.field.endswith((".REPORT_YEAR", ".REPORT_DATE"))
            for record in records
        )
        mainland_equity = symbol.endswith((".SH", ".SZ")) and not symbol.startswith(
            ("5", "1")
        )
        quote_records = [
            record for record in records
            if record.tool == "get_a_share_valuation" and record.symbol == symbol
        ]
        quote_attempted = bool(quote_records) or any(
            failure.get("tool") == "get_a_share_valuation"
            for failure in tool_failures if isinstance(failure, dict)
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
