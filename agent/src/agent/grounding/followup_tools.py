"""Offer a small research toolbox for verified holding-management follow-ups."""
from __future__ import annotations

import re

# Deliberately narrower than identity inheritance: "continue" can also mean
# finish a backtest or write a report. Never scope such open-ended requests.
_HOLDING_FOLLOWUP = re.compile(
    r"^\s*(?:我(?:应该|该)?怎么办|(?:那)?我(?:应该|该)?怎么做|"
    r"(?:那你说)?(?:不能|不该|不要)(?:无条件)?躺平[，,]?我(?:应该|该)?怎么做"
    r"(?:[，,]?什么时候卖出(?:降低成本)?)?|"
    r"(?:那)?(?:什么时候|何时)(?:该)?(?:卖出|减仓)(?:降低成本)?)\s*[？?。！!]*\s*$"
)
_TOOLS = frozenset({
    "search_symbol", "get_a_share_valuation", "get_market_data", "get_financial_statements",
    "etf_holdings", "financial_rigor", "web_search", "read_url", "read_file", "read_document",
    "read_run_artifact", "load_skill", "skill_file", "compact",
})


def holding_context_subject(question):
    """Recognize an explicit reference to previously supplied holding context."""
    if len(question) > 4000:
        return None
    match = re.match(
        r"^\s*请?(?:结合|沿用|根据|基于)(?:本会话|本次对话|此前|之前|前面)"
        r"(?:中)?(?:我)?(?:已经|已)?(?:提供|给出)(?:的)?"
        r"(?P<subject>[\u3400-\u9fffA-Za-z0-9.·-]{2,32}?)(?:持仓|仓位|成本)", question,
    )
    if not match or not re.search(r"(?:现在|当前).{0,12}(?:怎么做|怎么办|行动|操作)|当前倾向", question):
        return None
    return match.group("subject")


def holding_followup_tools(question, ledger, definitions):
    """Return a scoped schema list only when original evidence was restored."""
    if (not is_holding_followup(question, ledger)
            or not ledger._imported_research_calls):
        return None
    scoped = [item for item in definitions if item.get("function", {}).get("name") in _TOOLS]
    return scoped if scoped and len(scoped) < len(definitions) else None


def is_holding_followup(question, ledger):
    expanded = re.search(r"回测|投委会|团队|写(?:入|成)|保存.{0,12}文件|生成.{0,12}(?:文件|报告)|"
                         r"下单|执行交易|比较|对比|换(?:成|个|一只)", question)
    return bool(not expanded and ledger.inherited_symbols
                and (_HOLDING_FOLLOWUP.fullmatch(question) or holding_context_subject(question)))


def holding_constraints(history, symbols):
    """Only explicit user statements, never assistant assumptions or questions."""
    known = {}
    for message in reversed(list(history or [])[-24:]):
        if message.get("role") != "user":
            continue
        text = str(message.get("content") or "")
        for clause in reversed(re.split(r"[。；;\n，,]", text)):
            if re.search(r"[？?]|是否|有没有|如果|假如|若|吗", clause):
                continue
            for key, pattern in {
                "leverage": r"没有融资|无融资|没有借款|无借款|有融资|有借款",
                "liquidity": r"不需要用钱|不需要用款|无需用款|没有用款需求|需要用钱|需要用款|必须用钱|必须用款",
            }.items():
                match = re.search(pattern, clause)
                if match and key not in known:
                    known[key] = match.group(0)
        if any(symbol in text.upper() or symbol.split(".")[0] in text for symbol in symbols):
            break
    return known


def holding_delivery_review(content, known=None):
    """Bounded editorial hints, not a semantic verdict or financial gate."""
    text = re.sub(r"```.*?```", "", content, flags=re.S)
    units = [part.strip() for part in re.findall(r"[^。！？!?\n]+[。！？!?]?", text)
             if part.strip() and not re.search(r"[？?]|吗", part)]
    action = r"持有|减仓|卖出|不(?:再)?(?:补仓|加仓|追加)|等待|暂不买|降低仓位"
    condition = r"如果|假如|若|一旦|当(?!前)|前提|没有|无融资|不需要|需要用款|有融资|存在"
    requests = []
    opening = [unit for unit in units if unit in text[:600] and not re.search(condition, unit)]
    if (not any(re.search(action, unit) and re.search(r"当前|现在|建议|倾向|优先|先|暂时", unit) for unit in opening)
            or not re.search(r"因为|由于|依据|理由|考虑到|现金流|盈利|估值|集中|用款|融资|承受", text[:600])):
        requests.append("State the current research leaning and its reason first; distinguish today's "
                        "default from conditional changes. Do not stop at 'assess/review risk' or invent a trade order.")
    branches = [unit for unit in units if re.search(condition, unit) and re.search(action, unit)]
    personal = [unit for unit in branches if re.search(r"用款|用钱|应急|融资|借款|生活|变现", unit)]
    if not personal:
        requests.append("Connect the user's funding constraints to a concrete conditional action. "
                        "When constraints are unknown, explain cash-need/leverage and no-pressure branches now; "
                        "do not merely ask questions or require the user to reply before giving a framework.")
    if not any(re.search(r"盈利|现金流|利润|估值|经营|核心论点|持有理由|必须用款|需要用款|需要用钱", unit)
               and re.search(r"恶化|转负|失效|改善|下修|兑现|持续|上升|下降|转弱|不及预期|必须用款|需要用款|需要用钱", unit)
               for unit in branches):
        requests.append("Give an observable review/exit trigger and the action it would change, "
                        "using existing evidence or an explicitly hypothetical condition. 'Watch the market' is insufficient; "
                        "do not fabricate a price, date, probability or forecast.")
    known = known or {}
    if any((key == "leverage" and re.search(r"有没有.{0,8}(?:融资|借款)|是否.{0,8}(?:融资|借款)", text))
           or (key == "liquidity" and re.search(r"是否.{0,12}(?:用钱|用款)|(?:需要|必须).{0,8}(?:用钱|用款)[吗？?]", text))
           for key in known):
        requests.append("Use these user-stated constraints instead of asking for them again: "
                        + "; ".join(known.values()) + ". They are user inputs, not verified market facts.")
    return requests


def holding_research_allowed(question):
    return bool(holding_context_subject(question)
                and re.search(r"估值|盈利|现金流|财报|研究证据", question)
                and not re.search(r"不要(?:再)?(?:查询|补查|联网)|无需(?:查询|补查|联网)|"
                                  r"仅(?:用|基于|根据|使用).{0,12}(?:已有|现有|历史)|只(?:用|基于).{0,12}(?:已有|现有)", question))


def holding_research_preflight():
    return (
        "\n\n[PRE-ANSWER RESEARCH PLAN] Before composing a complete answer, assess the decision-critical "
        "evidence, not just the available quote. First use the dated inventory and original-source index "
        "to classify relevant materials as usable, stale, source candidates, unavailable or unchecked; "
        "retrieval success alone is not verification. Identify the input most likely to change the leaning "
        "(valuation basis, earnings quality or cash conversion) and what opposing evidence to check. "
        "Retrieve material unresolved inputs BEFORE writing a full report; do not spend a round drafting "
        "a generic complete answer merely to announce those gaps. Reuse exact source paths; do not list "
        "the current run directory to search for historical receipts. For funds, identify the tracked "
        "index and dated holdings, then choose material constituents rather than treating their findings "
        "as full fund coverage. Discover source URLs from actual search results or published page links; "
        "do not guess index detail paths. Batch independent reads and follow dependent links in order. "
        "If evidence is sufficient, retrieval is unavailable, or a gap cannot affect the immediate action, "
        "explain that boundary and answer without ceremonial fetching. This is a planning aid, not a "
        "mandatory fixed query count, proof of completeness or reason to invent data."
    )


def holding_research_gaps(question, content):
    """Flag self-disclosed decision gaps, never infer that evidence is absent."""
    if not holding_research_allowed(question):
        return []
    text = re.sub(r"```.*?```", "", content, flags=re.S)
    # Include immediately following bullets for "仍缺少：" declarations.
    text = re.sub(r"((?:缺少|缺乏|尚缺)[^\n。]{0,20}[：:])\n\s*((?:[-*] [^\n]+\n?){1,6})",
                  lambda m: m.group(0).replace("\n", " "), text)
    units = re.split(r"[。！？!?\n]", text)
    missing = r"缺少|缺乏|缺口|不足|尚未.{0,12}(?:核验|验证|取得)|未.{0,6}(?:核验|取得)|无法核验"
    topics = {"valuation": r"估值|市盈率|市净率", "earnings": r"盈利|利润|财报",
              "cash_flow": r"现金流"}
    return [topic for topic, pattern in topics.items()
            if any(re.search(pattern, unit) and re.search(missing, unit)
                   and not re.search(r"不缺少|并不缺乏|没有缺口|无缺口", unit) for unit in units)]


def holding_research_guidance(gaps=None):
    return (
        "\n\n[HOLDING EVIDENCE REVIEW] Accuracy and decision-relevant evidence take priority over token savings. "
        "Separate account-risk management from the asset's current investment merits. Audit both supporting "
        "and opposing evidence, dates, reporting periods and valuation denominators. Original receipts can be "
        "evidence; a search snippet or a visited page is only a lead until the relevant source content is checked. "
        "Do not assume missing entries in the bounded inventory mean no evidence exists: read original artifacts "
        "before abandoning usable research. Identify which unresolved input could actually reverse the leaning. "
        "When market evidence is requested, prioritize retrieving those material gaps rather than ending with "
        "only generic risk advice. For funds, verify the tracked index and dated holdings before choosing material "
        "constituents; do not substitute one company's valuation for the fund or imply full index coverage. "
        "Use available financial tools or official issuer/index/fund reports, resolve new instruments before "
        "requesting their data, and batch independent reads. Different periods, accounting adjustments and "
        "currencies require reconciliation before comparisons or valuation calculations. Seek disconfirming "
        "evidence, not only sources agreeing with the prior conclusion. Necessary research may use additional "
        "calls; avoid repeated failed queries, irrelevant breadth and unsupported precision. "
        "Respect explicit user restrictions on retrieval; when told to use existing materials only, "
        "do not fetch and disclose the resulting limits. "
        "If retrieval is unavailable or the gap cannot change the immediate risk action, explain why and "
        "give a conditional conclusion with appropriately limited confidence. No invented figures, buy/sell "
        "prices or confidence upgrades. A retrieval attempt does not establish accuracy or completeness."
        + ("\nThe draft itself disclosed these gaps: " + ", ".join(gaps)
           + ". Reassess them using existing evidence first, then retrieve what materially affects the answer. "
             "Tools are available for this research pass; this is not a text-only editorial correction."
           if gaps else "")
    )


def holding_followup_guidance():
    return (
        "\n\n[HOLDING FOLLOW-UP] Address the immediate holding-management question first. "
        "Carry forward the user's stated holdings, cost, horizon, liquidity constraints and previous answers; "
        "do not ask again for facts already supplied. Give a current research leaning, its evidence and what "
        "would change it. Explain conditional branches for near-term cash needs, leverage, or neither, "
        "connecting each branch to an observable review or exit condition. If those facts are unknown, "
        "state the branches now and ask only the remaining material question. Do not stop at 'review risk'. "
        "Distinguish selling to reduce exposure from high-sell/low-buy timing and from broker cost accounting. "
        "Treat user-provided account numbers as their stated inputs, not independently verified market data. "
        "Do not invent sell dates, prices, return probabilities or prescribe account-specific order sizes. "
        "Retain the output principles and dated-evidence requirements. Related research tools are available; "
        "this narrow follow-up does not require unrelated strategy, trading-execution or file-write tools. "
        "The general tool catalog may list additional capabilities; use only the tool schemas offered this turn."
    )
