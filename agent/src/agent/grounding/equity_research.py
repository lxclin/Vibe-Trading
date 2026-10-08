"""Evidence-backed worksheets for equity entry and comparison research.

The worksheet is a data inventory, not an investment score. It never marks
earnings as sustainable merely because net profit or cash flow is positive.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from src.agent.grounding.decision import decision_coverage

_RESEARCH_RE = re.compile(
    r"买入|值得买|能买吗|买什么|推荐.{0,6}买|投资价值|估值|市盈率|市净率|盈利质量|正常化盈利|"
    r"胜率|更好.{0,8}标的|别的.{0,8}标的|"
    r"\b(?:buy|valuation|undervalued|overvalued|earnings quality|better investment)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(r"提示词|怎么问|如何提问|编写代码|代码实现|代码怎么写|修改代码|修复代码|程序|项目|prompt|bug", re.IGNORECASE)
_ENTRY_RE = re.compile(
    r"买入|值得买|能买吗|买什么|推荐我.{0,6}买(?:什么|哪)|胜率|(?:更好|别的)[^。！？\n]{0,8}标的|"
    r"\b(?:buy|worth buying|better investment)\b",
    re.IGNORECASE,
)
_FIELDS = {
    "parent_profit": ("PARENTNETPROFIT", "PARENT_NETPROFIT"),
    "core_parent_profit": ("KCFJCXSYJLR",),
    "revenue": ("TOTALOPERATEREVE", "OPERATE_INCOME"),
    "operating_cash_flow": ("NETCASH_OPERATE", "NETCASH_OPERATE_PK"),
    "capex_cash_paid": ("CONSTRUCT_LONG_ASSET",),
    "basic_eps": ("EPSJB",),
    "investment_income": ("INVEST_INCOME",),
    "fair_value_change": ("FAIRVALUE_CHANGE_INCOME",),
    "impairment": ("ASSET_IMPAIRMENT_INCOME",),
}


def is_equity_entry_research(question: str) -> bool:
    """Include comparisons with funds, without asking funds for issuer accounts."""
    return bool(_RESEARCH_RE.search(question)) and not bool(_META_RE.search(question))


def is_mainland_company(symbol: str) -> bool:
    return bool(re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", symbol)) and not symbol.startswith(("5", "1"))


def needs_entry_inputs(question: str) -> bool:
    """A quote/ratio lookup alone should not trigger full investment research."""
    # A request to display a price/level or a screening shortlist is not itself
    # a thesis about buying a named company at today's price.
    intent = re.sub(r"买入价(?:格)?|买入点位|入场价(?:格)?|入场点位", "", question)
    return is_equity_entry_research(question) and bool(_ENTRY_RE.search(intent))


def _observed(record: Any) -> bool:
    if record.status != "observed" or isinstance(record.value, bool) or not isinstance(record.value, (int, float)):
        return False
    try:
        return math.isfinite(record.value)
    except OverflowError:
        return False


def _fact(record: Any) -> dict[str, Any]:
    return {
        "value": record.value, "currency": record.currency,
        "as_of": record.timestamp, "source": record.source,
        "ref": f"{record.call_id}::{record.field}",
    }


def equity_worksheet(records: Sequence[Any], symbols: set[str], attempts: Sequence[Any]) -> dict[str, Any]:
    """Select same-period facts and expose missing work without fabricating it."""
    companies = sorted(symbol for symbol in symbols if is_mainland_company(symbol))
    sheets = []
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    for symbol in companies[:4]:
        financials = []
        rejected_dates = False
        for record in records:
            if record.symbol != symbol or record.tool != "get_financial_statements" or not _observed(record):
                continue
            if not re.search(r"\.periods\[\d+\]\.[^.]+$", record.field):
                continue
            try:
                stamp = date.fromisoformat(str(record.timestamp)[:10])
            except ValueError:
                continue
            if stamp > today:
                rejected_dates = True
                continue
            financials.append(record)
        # Do not let a newer balance sheet silently replace the latest profit
        # period. Every comparison below uses the selected profitability date.
        profit_dates = [record.timestamp[:10] for record in financials
                        if record.field.rsplit(".", 1)[-1].upper() in _FIELDS["parent_profit"]]
        period = max(profit_dates) if profit_dates else None
        facts: dict[str, Any] = {}
        for label, fields in _FIELDS.items():
            selected = [record for record in financials
                        if period and record.timestamp[:10] == period
                        and record.field.rsplit(".", 1)[-1].upper() in fields]
            # Latest retrieved version wins only within an identical field.
            # Different sources or conflicting values require reconciliation.
            if selected:
                currencies = {record.currency for record in selected}
                values = {record.value for record in selected}
                if len(values) == 1 and len(currencies) == 1:
                    facts[label] = _fact(selected[-1])
                else:
                    facts[label] = {"status": "conflicting", "refs": [
                        f"{record.call_id}::{record.field}" for record in selected
                    ][:4]}

        coverage = decision_coverage(records, symbol, attempts)
        income_attempted = any(
            isinstance(attempt, dict) and attempt.get("symbol") == symbol
            and attempt.get("tool") == "get_financial_statements"
            and attempt.get("statement") == "income" for attempt in attempts
        )
        income_observed = any(record.statement == "income" for record in financials)
        missing = [key for key in ("parent_profit", "core_parent_profit", "operating_cash_flow")
                   if key not in facts or "value" not in facts[key]]
        warnings = [
            "Core/deducted profit is not automatically sustainable profit. Check filing notes, "
            "investment/fair-value changes, impairments, tax effects and minority interests.",
            "Operating cash flow is consolidated; parent profit excludes minority earnings. "
            "Do not call their ratio a like-for-like cash conversion rate.",
        ]
        if period and period[5:] != "12-31":
            warnings.append("This is a cumulative interim period, not annual EPS or one standalone quarter. Do not multiply EPS by two to declare cheapness.")
        if rejected_dates:
            warnings.append("Future-dated financial records were excluded from this worksheet.")
        if facts.get("parent_profit", {}).get("value", 0) > 0 and facts.get("operating_cash_flow", {}).get("value", 0) < 0:
            warnings.append("Positive parent profit accompanies negative operating cash flow; investigate working capital and consolidation scope.")
        sheets.append({
            "symbol": symbol, "report_date": period, "facts": facts,
            "coverage": coverage, "missing_same_period_inputs": missing,
            "research_progress": {
                "quote": coverage["raw_quote"]["status"],
                "provider_multiples": coverage["valuation_multiple"]["status"],
                "financial_inputs": "incomplete" if missing else "observed_analysis_pending",
                "income_statement": "observed_analysis_pending" if income_observed else (
                    "unavailable" if income_attempted else "unchecked"
                ),
                "original_filing_notes": "not_verified_by_worksheet",
                "industry_drivers": "not_verified_by_worksheet",
                "normalized_earnings_and_valuation": "not_verified_by_worksheet",
                "entry_conditions": "not_verified_by_worksheet",
            },
            "earnings_normalization": "not_assessed_by_worksheet",
            "fair_value_and_margin_of_safety": "not_assessed_by_worksheet",
            "warnings": warnings,
        })
    return {
        "kind": "evidence_inventory_not_investment_verdict", "companies": sheets,
        "omitted_companies": companies[4:],
        "boundary": "Facts come from this run only. A successful query is not proof of sustainable earnings or a favorable entry price."
    }


def equity_research_guidance(chinese: bool) -> str:
    """A bounded research sequence used for both named stocks and comparisons."""
    if chinese:
        return (
            "[企业买入与比较研究流程] 先区分公司质量、相对排序和当前价格是否值得买；"
            "用户未给期限时说明采用未来6—12个月。比较问题逐个确认标的；基金只检查指数、净值、费用、折溢价和风险，不能索取基金公司的经营财报。"
            "对公司复用已取报价和最新财报，缺字段时分别调用get_financial_statements的quarter/indicators、income、cashflow；"
            "只补需要的报表，同一失败请求不重复；报价工具已内置腾讯备用来源，仍失败时披露缺口，不把失败当看空。"
            "工作表中的unchecked是不曾查询，unavailable是查询未得到，observed_analysis_pending仅代表数据已取得，不能冒充分析完成。"
            "一、盈利质量：在同一报告期比较归母利润、扣非利润、经营现金流及资本开支；区分合并与归母口径。"
            "从公司/交易所原始报告和附注核对投资收益、公允价值损益、减值、处置、税率、少数股东及股本变动。"
            "不要只读摘要指标；扣非利润不等于正常化盈利，投资损益即使被公司列为经常性，也要单独分析。"
            "对每家公司的特殊因素对称调整；调整为税后归母口径，写原始数据、公式、来源与调整理由；税率或归属不明就不强行计算。"
            "二、估值与价格：查历史或同行同口径估值，比较业务和周期差异；资源股拆分金铜等价格、产量、成本、资本开支和新增项目。"
            "正常化盈利必须说明来源与周期假设；半年EPS乘二不是正常化。需要TTM时补齐上年全年、本年累计、上年同期，且核对会计口径和股本，不能重复叠加累计季度。"
            "有依据时列悲观/基准/乐观情景，明确每个盈利和估值倍数的来源或分析假设；不要拿技能文档的示例当市场数据。"
            "用financial_rigor的calc核算：情景每股价值=相应年度EPS×有依据的PE；上涨/下跌空间=情景价值÷未复权现价−1；"
            "安全边际=1−现价÷基准估值。它们分母不同，不要混称。亏损企业不采用PE；使用其他适合的方法。"
            "计算正确不代表假设可靠；没有合理假设就说明无法量化，仍给出已有证据支持的相对排序，不编目标价或胜率。"
            "三、回答：保留‘盈利质量’和‘估值与价格’两节，可简短。第一段说明是初筛排序、条件性偏好，还是已论证当前买入；"
            "比较时对每个候选写当前判断、关键依据与改变条件；仅有低PE/高增长只能支持初筛，不能升级成当前买入建议。"
            "证据不足时说明具体缺口和结论边界，不能把未查询变成看空；有证据时明确作出判断，不用平衡措辞代替回答。"
            "当前入场结论采用：可考虑买入、公司值得关注但价格偏贵、暂不买入（等待）、回避、研究未完成。"
            "缺报价、盈利或估值依据时必须区分‘研究未完成’和市场上的等待；初筛偏好可以单独保留。"
            "回答‘什么时候/什么指标可以买’时列简短状态表：指标、已有读数/日期、判断基准、当前是否满足、缺口或下一步。"
            "已有同比、利润或现金流数值应保留并声明observed，不要以‘同比增长’替代可核对数字来绕过校验。"
            "入场条件要说明什么数据与什么基准比较、为什么选择基准；有充分依据才给价格区间及计算。"
            "未取得指标就写未核实，不能给虚构阈值，也不能把通用清单当作已完成的判断。"
            "技术走势只作辅助，不能机械要求所有投资都先反弹确认；不承诺更高胜率。"
        )
    return (
        "[Equity entry/comparison research] Separate business quality, relative ranking and value at today's price. "
        "Use one stated horizon (default 6-12 months). Treat funds by NAV, fees, tracking and premium, not issuer accounts. "
        "Reuse quotes and financials; retrieve missing quarterly indicators/income/cashflow only as needed. "
        "Unchecked means not queried; unavailable means attempted without usable evidence. Observed inputs still require analysis. "
        "The quote tool has a Tencent fallback. Do not repeat failed reads or turn an outage into a bearish verdict. "
        "Include Earnings quality and Valuation and price sections, however brief. Reconcile same-period parent/core profit, "
        "operating cash flow and capex. Check original filing notes for investments, fair-value changes, impairments, "
        "disposals, tax, minority interests and share changes. Core profit is not normalized earnings. Adjust peers "
        "symmetrically using after-tax parent-attributable effects, sourced inputs and formulas; unknown tax/scope is a gap. "
        "For cyclicals analyze commodity prices, volume, costs, capex and projects. Use comparable historical/peer valuation "
        "and explicit sustainable-earnings assumptions; never normalize by simply doubling interim EPS. TTM needs "
        "prior annual + current cumulative - prior comparable cumulative, with compatible accounts and shares. "
        "When supported, state bear/base/bull earnings and multiple assumptions and verify arithmetic with financial_rigor calc. "
        "Scenario value = annual EPS * justified PE; price return = value / raw price - 1; margin of safety = 1 - raw price / base value. "
        "Do not use PE for loss-makers, skill examples as data, or calculator output as evidence assumptions are true. "
        "If inputs are unavailable, give a bounded relative ranking without invented targets or win rates. "
        "Identify whether the answer is screening, a conditional preference, or a supported current-entry view. "
        "Missing research is not bearish evidence; low PE and high past growth alone support screening, not a buy verdict. "
        "Use favorable / expensive / wait / avoid / research incomplete for current entry. Missing essential price, earnings "
        "or valuation basis means research incomplete; preserve any bounded screening preference separately. "
        "For entry-timing questions show metric, observed reading/date, justified benchmark, current satisfaction and next step. "
        "Retain observed financial figures rather than deleting numbers to bypass validation. Do not invent thresholds. "
        "Price trends are optional supporting evidence, not a mandatory rebound rule for every investment."
    )


def equity_research_input_issues(
    records: Sequence[Any], symbols: set[str], attempts: Sequence[Any],
) -> list[dict[str, Any]]:
    """Request unattempted essential reads before concluding a comparison.

    An unavailable provider is a disclosed gap, not a reason to query forever.
    Optional filing/forecast work is guided, never granted a fake checked flag.
    """
    issues = []
    for symbol in sorted(symbols):
        if not is_mainland_company(symbol):
            continue
        coverage = decision_coverage(records, symbol, attempts)
        required = (
            ("raw_quote", "decision_raw_valuation_not_checked",
             f'get_a_share_valuation(code="{symbol}")'),
            ("profitability", "decision_profitability_not_checked",
             f'get_financial_statements(code="{symbol}", statement="indicators", period="quarter")'),
            ("operating_cash_flow", "decision_cash_flow_not_checked",
             f'get_financial_statements(code="{symbol}", statement="cashflow", period="quarter")'),
        )
        for slot, code, instruction in required:
            if coverage[slot]["status"] != "unchecked":
                continue
            issues.append({
                "code": code, "value": None, "role": None, "span": None,
                "symbol": symbol, "reason": code,
                "message": (
                    f"Before an equity entry/comparison conclusion, check {slot} for {symbol} with {instruction}. "
                    "Reuse an existing result if it covers the field. A failed read is a gap; do not repeat it."
                ),
            })
        if (coverage["profitability"]["status"] == "observed"
                and not any(isinstance(attempt, dict) and attempt.get("symbol") == symbol
                            and attempt.get("statement") == "income" for attempt in attempts)
                and not any(record.symbol == symbol and record.tool == "get_financial_statements"
                            and record.statement == "income" and _observed(record) for record in records)):
            issues.append({
                "code": "decision_income_not_checked", "value": None, "role": None,
                "span": None, "symbol": symbol, "reason": "decision_income_not_checked",
                "message": f'Check get_financial_statements(code="{symbol}", statement="income", period="quarter") '
                "for profit composition before an entry verdict; a summary indicator is not earnings-quality analysis. "
                "Reuse existing observations; do not repeat a failed income query.",
            })
    return issues


def equity_entry_answer_issues(
    content: str, records: Sequence[Any], symbols: set[str], attempts: Sequence[Any],
) -> list[dict[str, Any]]:
    """Do not let missing essential inputs masquerade as a market verdict."""
    gaps = [symbol for symbol in sorted(symbols) if is_mainland_company(symbol)
            and decision_coverage(records, symbol, attempts)["entry_evidence"]["status"] == "incomplete"]
    if not gaps or re.search(r"研究未完成|暂无法评价|research incomplete|cannot yet assess", content, re.IGNORECASE):
        return []
    return [{
        "code": "decision_research_incomplete_unlabelled", "value": None, "role": None,
        "span": None, "symbol": symbol, "reason": "decision_research_incomplete_unlabelled",
        "message": f"Essential entry inputs for {symbol} are missing. Label its current-entry assessment "
        "研究未完成 / research incomplete and name the gaps; preserve screening preferences separately. "
        "A provider failure is not evidence that the price is expensive or that the stock should be avoided.",
    } for symbol in gaps]
