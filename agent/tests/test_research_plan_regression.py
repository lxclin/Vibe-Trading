"""Source retrieval budgets and sector-specific search terms are not conclusions."""

import pytest

from src.agent.grounding.research_plan import company_research_plan, research_attempt

SYMBOL = "601899.SH"
URL = "https://www.sse.com.cn/disclosure/report.pdf"


def plan(attempts=(), names=("紫金矿业",), sector="industrial", ready=True):
    coverage = {key: {"status": "observed" if ready else "unavailable", "as_of": "2026-10-09"}
                for key in ("raw_quote", "profitability", "operating_cash_flow")}
    coverage["financial_periods_match"] = True
    return company_research_plan(SYMBOL, names, "2026-06-30", coverage, attempts, sector=sector)


def search(success=True):
    return research_attempt("web_search", {"query": f"{SYMBOL} 历史 同行 估值 市盈率"},
                            {"results": [{"url": URL}]}, "search", success, {SYMBOL: ["紫金矿业"]}, [])


def test_failed_search_is_not_recommended_again():
    result = plan([search(False)])
    assert [action["topic"] for action in result["priority_actions"]] == ["earnings_drivers"]
    assert result["topics"][0]["status"] == "unavailable"


def test_failed_read_is_disclosed_without_repeat_search_or_read():
    first = search()
    second = research_attempt("read_url", {"url": URL}, {"status": "error"}, "read", False,
                              {SYMBOL: ["紫金矿业"]}, [first])
    result = plan([first, second])
    assert [action["topic"] for action in result["priority_actions"]] == ["earnings_drivers"]
    assert result["topics"][0]["status"] == "unavailable"


def test_page_read_before_search_is_reused_but_never_treated_as_verified():
    read = research_attempt("read_url", {"url": URL + "#page=2"}, {"status": "ok", "content": "report"},
                            "read", True, {SYMBOL: ["紫金矿业"]}, [])
    result = plan([read, search()])
    assert result["topics"][0]["status"] == "source_read_analysis_pending"
    assert "read" in result["topics"][0]["call_ids"]
    assert all(action["topic"] != "valuation_benchmark" for action in result["priority_actions"])


def test_other_company_search_does_not_consume_budget():
    other = research_attempt("web_search", {"query": "600036 招商银行 历史 同行 估值"},
                             {"results": [{"url": URL}]}, "other", True, {SYMBOL: ["紫金矿业"]}, [])
    assert other is None
    assert len(plan()["priority_actions"]) == 2
    assert not plan(ready=False)["priority_actions"]


@pytest.mark.parametrize("names,sector,focus,term", [
    (["长鑫科技"], "industrial", "semiconductor", "产能利用率"),
    (["紫金矿业"], "industrial", "resources", "储量"),
    (["招商银行"], "bank", "bank", "净息差"),
    (["某制造公司"], "industrial", "industrial", "收入结构"),
])
def test_driver_search_uses_relevant_terms(names, sector, focus, term):
    result = plan(names=names, sector=sector)
    assert result["driver_search_focus"] == focus
    query = next(action["arguments"]["query"] for action in result["priority_actions"]
                 if action["topic"] == "earnings_drivers")
    assert term in query
