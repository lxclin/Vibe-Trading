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
    if (not _HOLDING_FOLLOWUP.fullmatch(question) or not ledger.inherited_symbols
            or not ledger._imported_research_calls):
        return None
    scoped = [item for item in definitions if item.get("function", {}).get("name") in _TOOLS]
    return scoped if scoped and len(scoped) < len(definitions) else None


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
