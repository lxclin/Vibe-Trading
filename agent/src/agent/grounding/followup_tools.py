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


def holding_followup_tools(question, ledger, definitions):
    """Return a scoped schema list only when original evidence was restored."""
    if (not is_holding_followup(question, ledger)
            or not ledger._imported_research_calls):
        return None
    scoped = [item for item in definitions if item.get("function", {}).get("name") in _TOOLS]
    return scoped if scoped and len(scoped) < len(definitions) else None


def is_holding_followup(question, ledger):
    return bool(_HOLDING_FOLLOWUP.fullmatch(question) and ledger.inherited_symbols)


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
