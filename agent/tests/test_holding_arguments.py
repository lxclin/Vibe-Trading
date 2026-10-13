from types import SimpleNamespace

import pytest

from src.agent.grounding.holding_arguments import holding_argument_review
from tests.test_holding_evidence_review import QUESTION


def _row(field="income.net_profit", status="observed"):
    return SimpleNamespace(call_id="financial-call", field=field, status=status)


def test_uncited_qualitative_market_claim_needs_traceable_reasoning():
    hints = holding_argument_review(QUESTION, "目前盈利正在改善，因此继续持有。", [])
    assert len(hints) == 1 and "original source" in hints[0]


@pytest.mark.parametrize("ref", ["invented::income.net_profit", "financial-call::income.net_profit_fake", "financial-call::quote.close"])
def test_unknown_unrelated_and_prefix_references_do_not_count(ref):
    assert holding_argument_review(QUESTION, "当前盈利改善，依据" + ref, [_row()])


def test_exact_relevant_reference_is_a_traceability_signal_only():
    text = "当前盈利改善，依据financial-call::income.net_profit。"
    assert not holding_argument_review(QUESTION, text, [_row()])
    assert holding_argument_review(QUESTION, text, [_row(status="unavailable")])


def test_conflicting_trends_need_explanation_not_just_two_citations():
    text = ("目前利润改善，依据financial-call::income.net_profit。"
            "当前现金流恶化，依据financial-call::cashflow.operating_cash_flow。")
    rows = [_row(), _row("cashflow.operating_cash_flow")]
    hints = holding_argument_review(QUESTION, text, rows)
    assert len(hints) == 1 and "Reconcile" in hints[0]
    assert not holding_argument_review(QUESTION, text + "需要核对营运资本和同口径报期，现金流变化主导复盘。", rows)


def test_hypothetical_triggers_are_not_assertions_of_conflicting_facts():
    assert not holding_argument_review(QUESTION, "如果利润改善，可以继续复盘。如果现金流恶化，则减仓。", [])


def test_strong_forecast_with_disclosed_gap_needs_calibration_but_negation_does_not():
    assert holding_argument_review(QUESTION, "估值尚未核验，证据不足。当前强烈看多，一定回本。", [])
    assert not holding_argument_review(QUESTION, "估值尚未核验，证据不足。不能保证收益，也不是一定回本。", [])


def test_figures_and_simple_risk_questions_do_not_trigger_argument_rewrite():
    assert not holding_argument_review(QUESTION, "```figures\n目前盈利改善\n```", [])
    assert not holding_argument_review("我应该怎么办", "当前盈利改善。", [])
