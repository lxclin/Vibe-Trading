"""Soft checks for the reasoning in holding advice, not truth verification."""
from __future__ import annotations

import re

from src.agent.grounding.followup_tools import holding_context_subject

_TOPICS = {
    "earnings": (r"盈利|利润", r"profit|income|earning|利润|盈利"),
    "cash_flow": (r"现金流", r"cash|现金流"),
    "valuation": (r"估值|市盈率|市净率", r"valuation|percentile|(?:^|[._])p[eb](?:$|[._])|市盈率|市净率|估值"),
}


def holding_argument_review(question, content, evidence):
    """Request clarification of explicit assertions, contradictions and certainty.

    An exact relevant citation is a traceability signal only. It never proves
    that the source entails the claim, is comparable or is sufficiently recent.
    Existing numeric release checks remain authoritative.
    """
    if not holding_context_subject(question):
        return []
    text = re.sub(r"```.*?```", "", content, flags=re.S)
    units = re.split(r"[。！？!?\n]", text)
    findings = []
    for unit in units:
        if (not re.search(r"当前|目前|已经|正在|因此|表明", unit)
                or re.search(r"如果|假如|一旦|尚未|未核验|不足|缺少|缺口|不能|并非|不是|不代表", unit)):
            continue
        for topic, (words, fields) in _TOPICS.items():
            if not (re.search(words, unit) and re.search(r"改善|恶化|增长|下滑|下修|便宜|低估|高估|昂贵|修复", unit)):
                continue
            linked = any(
                row.status in {"observed", "derived"} and re.search(fields, row.field, re.I)
                and re.search(re.escape(f"{row.call_id}::{row.field}") + r"(?![\w.\[])", unit)
                for row in evidence
            )
            if not linked:
                findings.append("Trace this asserted " + topic + " finding to a relevant original source and "
                                "reporting period, explain how it supports the leaning, and label inference "
                                "separately. If unsupported, qualify or remove the assertion; never invent "
                                "a citation. Claim excerpt: " + unit.strip()[:220])
    factual = "。".join(unit for unit in units if not re.search(r"如果|假如|一旦|若|尚未|未核验|缺少", unit))
    profit_up = re.search(r"(?:盈利|利润)[^。\n]{0,25}(?:改善|增长|修复)", factual)
    cash_down = re.search(r"现金流[^。\n]{0,25}(?:恶化|下滑|转负)", factual)
    if profit_up and cash_down and not re.search(
            r"更重视|优先考虑|主导|抵消|不可比|同口径|口径差异|营运资本|一次性|不能据此|暂不据此", text):
        findings.append("Reconcile improving earnings with weakening cash flow: check comparable periods, "
                        "working capital and one-off items, explain which evidence dominates the decision "
                        "and why. If this is only a hypothetical scenario, say so explicitly. Do not invent a cause.")
    if re.search(r"证据不足|缺少.{0,20}(?:估值|盈利|现金流)|尚未核验", text) and any(
            re.search(r"强烈看多|强烈看空|必然上涨|必然下跌|一定回本|保证收益|确定上涨", unit)
            and not re.search(r"不能|无法|不会|不保证|不应|不代表|并非|不是", unit) for unit in units):
        findings.append("Calibrate confidence to the disclosed evidence gaps; remove guarantees and "
                        "unjustified strong market calls. Distinguish a firm liquidity-risk action from "
                        "an uncertain market-direction judgment and state what evidence would reverse it.")
    return list(dict.fromkeys(findings))[:4]


def holding_argument_guidance():
    return (
        "\n\n[CONCLUSION AND EVIDENCE] For each material market judgment, connect the conclusion to "
        "a relevant original source, date/reporting period and exact call_id::field when available. "
        "Explain the inference, the strongest contrary evidence, unresolved conflicts and what would "
        "reverse the leaning. Do not force a table or manufacture an opposing fact when none is verified. "
        "Profit growth with weaker cash flow, or lower multiples with falling earnings estimates, "
        "requires a comparison of periods, accounting bases and decision significance; never silently "
        "pick the favorable side or invent a reconciliation. Separate facts, hypotheses, user inputs "
        "and conditional triggers. Confidence in a cash-need risk action is not confidence in a price "
        "forecast. A citation proves traceability, not correctness or that a conclusion follows."
    )
