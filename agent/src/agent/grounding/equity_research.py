"""Evidence-backed worksheets for equity entry and comparison research.

The worksheet is a data inventory, not an investment score. It never marks
earnings as sustainable merely because net profit or cash flow is positive.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.agent.grounding.analysis import multiple_comparisons, valuation_workbench
from src.agent.grounding.decision import decision_coverage
from src.agent.grounding.research_plan import company_research_plan

_RESEARCH_RE = re.compile(
    r"买入|值得买|能买吗|买什么|推荐.{0,6}买|投资价值|估值|市盈率|市净率|盈利质量|正常化盈利|"
    r"胜率|更好.{0,8}标的|别的.{0,8}标的|做多|做空|"
    r"\b(?:buy|valuation|undervalued|overvalued|earnings quality|better investment|long or short|go long|go short|long thesis|short thesis)\b",
    re.IGNORECASE,
)
_META_RE = re.compile(r"提示词|怎么问|如何提问|编写代码|代码实现|代码怎么写|修改代码|修复代码|程序|项目|prompt|bug", re.IGNORECASE)
_ENTRY_RE = re.compile(
    r"买入|值得买|能买吗|买什么|推荐我.{0,6}买(?:什么|哪)|胜率|做多|做空|(?:更好|别的)[^。！？\n]{0,8}标的|"
    r"\b(?:buy|worth buying|better investment|long or short|go long|go short|long thesis|short thesis)\b",
    re.IGNORECASE,
)
_COMPANY_FOLLOWUP_RE = re.compile(
    r"^.{2,60}?(?:怎么样|如何|怎么看|值得关注吗)[？?。\s]*$|^what about .{2,60}[?\s]*$",
    re.IGNORECASE,
)
_NON_COMPANY_FOLLOWUP_RE = re.compile(
    r"回答|报告|结果|输出|提示|方案|方法|页面|天气|身体|不要分析投资|不谈投资|"
    r"公司简介|公司介绍|主营业务|官网|\b(?:answer|report|website)\b", re.IGNORECASE,
)
_FINANCIAL_SECTOR_FIELDS = frozenset({
    "NONPERLOAN", "NET_INTEREST_MARGIN", "EARNED_PREMIUM", "NET_ROI", "NBV_LIFE",
})
_FIELDS = {
    "parent_profit": ("PARENTNETPROFIT", "PARENT_NETPROFIT"),
    "core_parent_profit": ("KCFJCXSYJLR",),
    "revenue": ("TOTALOPERATEREVE", "OPERATE_INCOME"),
    "operating_cash_flow": ("NETCASH_OPERATE", "NETCASH_OPERATE_PK"),
    "capex_cash_paid": ("CONSTRUCT_LONG_ASSET",),
    "basic_eps": ("EPSJB",),
    "book_value_per_share": ("BPS", "PER_NETASSET"),
    "total_shares": ("TOTAL_SHARE",),
    "total_equity": ("TOTAL_EQUITY_PK",),
    "investment_income": ("INVEST_INCOME",),
    "fair_value_change": ("FAIRVALUE_CHANGE_INCOME",),
    "impairment": ("ASSET_IMPAIRMENT_INCOME",),
    "gross_profit": ("MLR",),
    "gross_margin_pct": ("XSMLL",),
    "weighted_roe_pct": ("ROEJQ",),
}


def contextual_equity_research_question(
    question: str, history: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Carry an investment objective into a short company-evaluation follow-up.

    Only user requests supply intent. This does not carry a previous company's
    identity to a new subject, or treat every factual company question as a buy
    question. The ordinary resolver must still identify the new company.
    """
    if (is_equity_entry_research(question) or _META_RE.search(question)
            or _NON_COMPANY_FOLLOWUP_RE.search(question)
            or not _COMPANY_FOLLOWUP_RE.fullmatch(question.strip())):
        return question
    horizon = None
    for message in reversed(list(history or [])[-24:]):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "").strip()
        if re.search(r"不谈投资|不要分析投资|只介绍公司|只看业务", text):
            return question
        if horizon is None and re.fullmatch(r"(?:持有|期限[:：]?)?\s*(?:半年|一年|长期|短线|\d+个月)[。\s]*", text):
            horizon = text[:40]
        if is_equity_entry_research(text):
            return (question + "\n[用户此前明确的投资研究需求] " + text[:240]
                    + (f"；持有期限：{horizon}" if horizon else ""))
    return question


def _statement_attempted(attempts: Sequence[Any], symbol: str, statement: str, period: str | None = None) -> bool:
    return any(isinstance(attempt, dict) and attempt.get("symbol") == symbol
               and attempt.get("tool") == "get_financial_statements"
               and attempt.get("statement") == statement
               and (period is None or attempt.get("period") == period) for attempt in attempts)


def _supplementary_tasks(records: Sequence[Any], symbol: str, attempts: Sequence[Any]) -> list[dict[str, Any]]:
    """Offer each missing statement once; query success never implies analysis."""
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    financials = []
    for record in records:
        if record.symbol != symbol or record.tool != "get_financial_statements" or not _observed(record):
            continue
        try:
            stamp = date.fromisoformat(str(record.timestamp)[:10])
        except ValueError:
            continue
        if stamp <= today:
            financials.append(record)
    profit_dates = [str(record.timestamp)[:10] for record in financials
                    if record.field.rsplit(".", 1)[-1].upper() in _FIELDS["parent_profit"]]
    latest = max(profit_dates) if profit_dates else None
    current = [record for record in financials if str(record.timestamp)[:10] == latest]
    fields = {record.field.rsplit(".", 1)[-1].upper() for record in current}
    tasks = []
    if (latest and not fields.intersection(_FINANCIAL_SECTOR_FIELDS)
            and not fields.intersection(_FIELDS["capex_cash_paid"])
            and not _statement_attempted(attempts, symbol, "cashflow")
            and not any(record.statement == "cashflow" for record in current)):
        tasks.append({
            "code": "decision_capex_not_checked", "statement": "cashflow", "period": "quarter",
            "reason": "Operating cash flow alone does not cover capex. Retrieve the cash-flow statement "
                      "once for same-period cash paid for long-lived assets; retain a gap if unavailable.",
        })
    coverage = decision_coverage(records, symbol, attempts)
    annual_eps = any(record.field.rsplit(".", 1)[-1].upper() == "EPSJB"
                     and getattr(record, "report_period", None) == "annual" and record.value > 0
                     and (today - date.fromisoformat(str(record.timestamp)[:10])).days <= 550 for record in financials)
    if (coverage["valuation_multiple"]["status"] == "unavailable"
            and any(record.field.rsplit(".", 1)[-1].upper() == "EPSJB" and record.value > 0 for record in current)
            and not annual_eps and not _statement_attempted(attempts, symbol, "indicators", "annual")):
        tasks.append({
            "code": "decision_valuation_basis_not_checked", "statement": "indicators", "period": "annual",
            "reason": "Provider multiples are unavailable. Retrieve annual indicators once to examine an "
                      "alternative earnings basis, alongside current BPS and share changes. A pre-IPO or "
                      "stale EPS may be unusable; disclose that instead of annualizing interim EPS.",
        })
    return [{**task, "tool": "get_financial_statements", "arguments": {
        "code": symbol, "statement": task["statement"], "period": task["period"],
    }} for task in tasks]


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


def _financial_consistency(facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reconcile selected same-period facts without claiming source verification.

    EPS uses weighted average shares whereas TOTAL_SHARE is a period-end
    balance. A mismatch requests review; it does not establish a data error.
    Consolidated equity is deliberately not compared with parent-company BPS.
    """
    checks = []
    # Deliberately review flags, not invalidation: unusually high margins or
    # ROE can be real, but must not silently become evidence of durable quality.
    for key, threshold in (("gross_margin_pct", 80), ("weighted_roe_pct", 60)):
        fact = facts.get(key, {})
        if "value" in fact and abs(fact["value"]) > threshold:
            checks.append({"check": "unusual_" + key, "status": "review_needed",
                           "reported_pct": fact["value"], "review_threshold_pct": threshold,
                           "refs": [fact["ref"]],
                           "reason": "Unusual ratio; review original filing, units, period and denominator. "
                                     "This is neither proof of an error nor evidence of sustainable quality."})
    for key, fact in facts.items():
        if fact.get("status") == "conflicting":
            checks.append({"check": "conflicting_" + key, "status": "review_needed",
                           "refs": fact["refs"], "reason": "Same-period inputs disagree; reconcile sources before using this fact."})

    def inputs(*keys: str) -> list[Any] | None:
        selected = [facts.get(key, {}) for key in keys]
        if not all("value" in fact for fact in selected):
            return None
        currencies = {fact.get("currency") for fact in selected}
        if len(currencies) != 1 or None in currencies or "" in currencies:
            return None
        return selected

    gross = inputs("gross_profit", "revenue", "gross_margin_pct")
    if gross and gross[1]["value"] > 0:
        computed = gross[0]["value"] / gross[1]["value"] * 100
        matches = abs(computed - gross[2]["value"]) <= 0.1
        checks.append({"check": "gross_margin", "status": "arithmetic_consistent" if matches else "review_needed",
                       "formula": "gross_profit / revenue * 100", "computed_pct": computed,
                       "reported_pct": gross[2]["value"], "tolerance_percentage_points": 0.1,
                       "refs": [fact["ref"] for fact in gross],
                       "reason": "Arithmetic check only; verify units and revenue scope in the filing if inconsistent."})
    shares = inputs("basic_eps", "total_shares", "parent_profit")
    if shares and shares[1]["value"] > 0 and shares[2]["value"] != 0:
        implied = shares[0]["value"] * shares[1]["value"]
        difference = abs(implied - shares[2]["value"]) / abs(shares[2]["value"])
        checks.append({"check": "eps_share_basis", "status": "review_needed" if difference > 0.05 else "no_material_difference_detected",
                       "formula": "basic_eps * period_end_total_shares", "implied_profit": implied,
                       "relative_difference": difference, "review_threshold": 0.05,
                       "refs": [fact["ref"] for fact in shares],
                       "reason": "EPS uses weighted-average shares; period-end shares may differ after issuance or other changes. "
                                 "Check the share basis before calculating valuation; this is not proof of an error."})
    return checks


def equity_worksheet(
    records: Sequence[Any], symbols: set[str], attempts: Sequence[Any], *,
    company_names: Mapping[str, Sequence[str]] | None = None,
    research_attempts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
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
        quote_ref = coverage["raw_quote"].get("ref") or ""
        quote_call_id = quote_ref.partition("::")[0]
        quote_fields = {
            record.field.removeprefix("data."): _fact(record)
            for record in records if record.symbol == symbol and record.tool == "get_a_share_valuation"
            and record.call_id == quote_call_id and _observed(record)
            and record.field in {"data.last_price", "data.pe_ttm", "data.pb", "data.market_cap_cny"}
        }
        names = (company_names or {}).get(symbol, [])
        sector_fields = {record.field.rsplit(".", 1)[-1].upper() for record in financials}
        sector = ("bank" if sector_fields.intersection({"NONPERLOAN", "NET_INTEREST_MARGIN"}) or any("银行" in name for name in names)
                  else "insurance" if sector_fields.intersection({"EARNED_PREMIUM", "NET_ROI", "NBV_LIFE"}) or any(
                      "保险" in name or "人寿" in name for name in names)
                  else "financial" if sector_fields.intersection(_FINANCIAL_SECTOR_FIELDS) or any("证券" in name for name in names)
                  else "industrial")
        income_attempted = any(
            isinstance(attempt, dict) and attempt.get("symbol") == symbol
            and attempt.get("tool") == "get_financial_statements"
            and attempt.get("statement") == "income" for attempt in attempts
        )
        income_observed = any(record.statement == "income" for record in financials)
        annual_eps = [record for record in financials if getattr(record, "report_period", None) == "annual"
                      and record.field.rsplit(".", 1)[-1].upper() == "EPSJB"]
        latest_annual = max(annual_eps, key=lambda record: record.timestamp) if annual_eps else None
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
            "financial_consistency": {
                "checks": _financial_consistency(facts),
                "boundary": "Same-period arithmetic and share-basis review only; not independent source verification. "
                            "Missing checks mean insufficient comparable inputs, not a clean bill of health.",
            },
            "coverage": coverage, "missing_same_period_inputs": missing,
            "valuation_snapshot": {
                "fields": quote_fields,
                "boundary": "One timestamped raw-quote call only. Provider PE/PB are observed ratios, "
                            "not verified fair value. Do not fill missing fields from another snapshot.",
            },
            "analysis_workbench": valuation_workbench(facts, quote_fields, sector,
                                                      multiple_comparisons(records, symbol, symbols, quote_fields)),
            "valuation_recovery_inputs": {
                "latest_annual_eps": _fact(latest_annual) if latest_annual is not None else None,
                "current_bps": facts.get("book_value_per_share"),
                "current_total_shares": facts.get("total_shares"),
                "boundary": "Candidate calculation inputs only. Confirm share basis, reporting scope and "
                            "quote adjustment before using them. Annual EPS is historical, not normalized or TTM.",
            },
            "next_statement_reads": _supplementary_tasks(records, symbol, attempts),
            "source_research_plan": company_research_plan(
                symbol, names, period, coverage, research_attempts or [], sector=sector,
            ) if research_attempts is not None else None,
            "research_progress": {
                "quote": coverage["raw_quote"]["status"],
                "provider_multiples": coverage["valuation_multiple"]["status"],
                "financial_inputs": "incomplete" if missing else "observed_analysis_pending",
                "income_statement": "observed_analysis_pending" if income_observed else (
                    "unavailable" if income_attempted else "unchecked"
                ),
                "capex": "observed_analysis_pending" if "capex_cash_paid" in facts else (
                    "unavailable" if _statement_attempted(attempts, symbol, "cashflow") else "unchecked"
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
            "ETF同样可用get_a_share_valuation获取未复权报价，勿套公司PE/PB。read_url返回web_financial时，"
            "净值或报价声明observed并引用精确call_id::web_financial.unit_nav/last_price；单位净值不是盘中IOPV，不能混用日期计算实时溢价。"
            "未结构化的网页费用和历史收益须用cited并在数字所在行显示来源；read_url::content不是可校验数值字段。"
            "对公司复用已取报价和最新财报，缺字段时分别调用get_financial_statements的quarter/indicators、income、cashflow；"
            "只补需要的报表，同一失败请求不重复；报价工具已内置腾讯备用来源，仍失败时披露缺口，不把失败当看空。"
            "工作表中的unchecked是不曾查询，unavailable是查询未得到，observed_analysis_pending仅代表数据已取得，不能冒充分析完成。"
            "工作表列出next_statement_reads时，先执行尚未尝试的补查再写最终报告，不能只把可查问题列成未来清单。"
            "基础报价和同期财报齐备后，source_research_plan列出历史/同行估值及盈利驱动的priority_actions。"
            "工具可用且迭代预算允许时按顺序补查，复用已取正文；每个主题仅建议一次定向搜索和一次候选正文读取，必要时按pagination继续读同一文档；失败后说明缺口，不重复相同搜索。"
            "source_candidates_found仅表示取得链接，source_read_analysis_pending表示已读但仍须核对日期、股本、业务和会计口径，不能写成已验证估值或正常化盈利。"
            "临近工具预算结束时，用已有证据完成有边界的报告，不为补查清单再开启循环。"
            "若行情源缺PE/PB，继续核对工作表中的BPS、全年EPS和股本变动，口径一致时用financial_rigor计算参考PB或历史PE，"
            "标明报价日与报期；新上市、拆股、增发前的每股数据尤其要核对分母。无法对齐时说明具体冲突，不能机械换算。"
            "读取公告网页只有释义、摘要或页面附带行情时，不能称为已读完整财报；继续定位交易所/公司原始PDF及财务表或附注，"
            "至少尝试一次替代路径，失败后列缺口，不反复抓同页。若首次来源本已是完整原始报告则直接复用。"
            "read_url返回pagination.next_offset时，可用offset翻到财务表或附注，不能把首段截断当作完整阅读。"
            "估值与价格一节应写已能计算的参考倍数、不可计算项及原因，再通过研报或同行/历史数据核对比较基准；"
            "供应商返回数据仅是证据输入，异常增速、利润率或股本关系应指出并核验，不能默认为可靠。"
            "工作表financial_consistency中review_needed项须说明冲突或核验结果；EPS与期末股数不匹配可能源于加权平均股本，不能直接判定造假或数据错误。"
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
            "analysis_workbench提供同报告期差额计算、估值比较列和可持续盈利桥，使用其公式及原始输入引用，不把派生值冒充工具观测。"
            "估值比较按指标、当前值、基准值、双方日期、TTM/前瞻口径、业务周期差异、结论逐项呈现；缺项写未核实，不用低PE替代比较。"
            "最终明确分别标注公司质量、价格吸引力、方向倾向、当前行动；每项独立说明依据、信心与缺口。缺估值可保留公司质量判断，但价格未评估不等于价格贵。"
            "三、回答：保留‘盈利质量’和‘估值与价格’两节，可简短。第一段说明是初筛排序、条件性偏好，还是已论证当前买入；"
            "若团队已完成研究或刚完成补查，最终回答必须独立完整，整合原结论与新增证据，不能只写‘已补读’或‘结论不变’。"
            "valuation_snapshot保留同次原始报价的价格、PE/PB及精确引用；已取得的倍数须解释其口径和意义，不能因旧工具正文压缩而当作缺失。"
            "紧接结论分别写公司判断、当前价格判断和当前行动，逐项给出支持证据；公司判断可为偏积极/中性/偏谨慎，价格判断可为有吸引力/合理/偏贵/依据不足。"
            "‘研究未完成’只限定尚未论证的部分，仍须回答已有证据支持的公司质量和相对偏好；不得凭数据缺失把公司判断写成偏谨慎。"
            "列出最影响结论的两项缺口，并说明各自什么核验结果会令判断上调或下调，避免只写‘补齐资料后再评估’。"
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
        "Execute unattempted next_statement_reads before finalizing, instead of turning retrievable gaps into a generic checklist. "
        "Once quotes and matching financials are present, source_research_plan supplies concrete valuation-benchmark and earnings-driver "
        "priority_actions. When tools and iteration budget permit, follow those actions, reuse existing pages, and stop after one scoped "
        "search and one candidate read recommendation per topic, paginating that document only if needed. "
        "Found links are leads; read pages still need date, accounting and share-basis review. "
        "Near the tool budget limit, finish a bounded report from existing evidence instead of opening another research loop. "
        "When provider PE/PB is missing, examine BPS, annual EPS and share changes; use financial_rigor for a provisional "
        "historical PE or PB only with compatible share and price bases. Pre-IPO EPS is not automatically comparable. "
        "A filing page containing only a glossary, preview or quote sidebar is not a full filing. Try one alternative "
        "path to the exchange/company original PDF and financial tables or notes, then disclose failure without repeated reads. "
        "Use read_url offset with pagination.next_offset for later tables/notes in a long document; a first excerpt is not full coverage. "
        "Show available valuation calculations and unusable inputs separately; investigate anomalous margins, growth or share relationships. "
        "Address financial_consistency review_needed items with reconciliation or an explicit limitation. EPS uses weighted-average shares, "
        "so an EPS versus period-end-share mismatch is a basis review, not proof of bad data. Arithmetic consistency is not independent verification. "
        "The quote tool has a Tencent fallback. Do not repeat failed reads or turn an outage into a bearish verdict. "
        "Include Earnings quality and Valuation and price sections, however brief. Reconcile same-period parent/core profit, "
        "operating cash flow and capex. Check original filing notes for investments, fair-value changes, impairments, "
        "disposals, tax, minority interests and share changes. Core profit is not normalized earnings. Adjust peers "
        "symmetrically using after-tax parent-attributable effects, sourced inputs and formulas; unknown tax/scope is a gap. "
        "Use analysis_workbench for sourced arithmetic, a dated like-for-like valuation comparison and the sustainable earnings bridge. "
        "Explicitly label Company quality, Price attractiveness, Directional view and Current action, with independent evidence, confidence and gaps. "
        "Missing valuation does not erase supported company findings or prove overvaluation. "
        "For cyclicals analyze commodity prices, volume, costs, capex and projects. Use comparable historical/peer valuation "
        "and explicit sustainable-earnings assumptions; never normalize by simply doubling interim EPS. TTM needs "
        "prior annual + current cumulative - prior comparable cumulative, with compatible accounts and shares. "
        "When supported, state bear/base/bull earnings and multiple assumptions and verify arithmetic with financial_rigor calc. "
        "Scenario value = annual EPS * justified PE; price return = value / raw price - 1; margin of safety = 1 - raw price / base value. "
        "Do not use PE for loss-makers, skill examples as data, or calculator output as evidence assumptions are true. "
        "If inputs are unavailable, give a bounded relative ranking without invented targets or win rates. "
        "Identify whether the answer is screening, a conditional preference, or a supported current-entry view. "
        "After team research or supplementary checks, deliver a self-contained final report integrating the existing conclusion "
        "and new findings, not only an update saying unchanged. Explain observed valuation_snapshot ratios with their exact "
        "source/time and limitations; compacted tool bodies do not mean the persisted observations are missing. "
        "Separate company-quality view, current-price assessment and current action, each with its supporting evidence. "
        "Scope research incomplete to the unproven part; preserve supported business-quality or relative preferences. "
        "Name the two gaps most likely to change the decision and explain what result would move the view up or down. "
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
        for task in _supplementary_tasks(records, symbol, attempts):
            args = task["arguments"]
            issues.append({
                "code": task["code"], "value": None, "role": None, "span": None,
                "symbol": symbol, "reason": task["code"],
                "message": f'Check get_financial_statements(code="{symbol}", statement="{args["statement"]}", '
                           f'period="{args["period"]}") before finalizing. ' + task["reason"],
            })
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
    verified_valuation_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Do not let missing essential inputs masquerade as a market verdict."""
    gaps = []
    for symbol in sorted(symbols):
        if not is_mainland_company(symbol):
            continue
        missing = decision_coverage(records, symbol, attempts)["entry_evidence"]["gaps"]
        if symbol in (verified_valuation_symbols or set()):
            missing = [slot for slot in missing if slot != "valuation_multiple"]
        if missing:
            gaps.append(symbol)
    if not gaps or re.search(r"研究未完成|暂无法评价|research incomplete|cannot yet assess", content, re.IGNORECASE):
        return []
    return [{
        "code": "decision_research_incomplete_unlabelled", "value": None, "role": None,
        "span": None, "symbol": symbol, "reason": "decision_research_incomplete_unlabelled",
        "message": f"Essential entry inputs for {symbol} are missing. Label its current-entry assessment "
        "研究未完成 / research incomplete and name the gaps; preserve screening preferences separately. "
        "A provider failure is not evidence that the price is expensive or that the stock should be avoided.",
    } for symbol in gaps]
