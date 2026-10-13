import json
from pathlib import Path

import pytest

from src.agent.grounding.followup_tools import holding_research_gaps
from tests.test_holding_delivery import COMPLETE

FIXTURES = Path(__file__).parent / "fixtures/research_20261012"
QUESTION = (FIXTURES / "holding_context_prompt.txt").read_text()
GAP = "当前估值尚未核验。盈利和现金流缺少可靠财报依据。"


def test_real_acceptance_disclosures_need_evidence_review():
    text = (FIXTURES / "holding_acceptance_answer.txt").read_text()
    assert set(holding_research_gaps(QUESTION, text)) == {"valuation", "earnings", "cash_flow"}


@pytest.mark.parametrize("question", ["我应该怎么办", QUESTION + "不要补查", QUESTION + "仅基于已有材料回答"])
def test_simple_risk_or_explicit_no_retrieval_requests_do_not_trigger(question):
    assert not holding_research_gaps(question, GAP)


def test_conditional_risk_and_figures_are_not_missing_evidence_admissions():
    assert not holding_research_gaps(QUESTION, "如果盈利和现金流恶化，则重新评估持有。估值不缺少可靠证据。")
    assert not holding_research_gaps(QUESTION, "```figures\n" + GAP + "\n```")


@pytest.mark.parametrize("mode", ["research", "unavailable", "no_fetch", "no_tools", "numeric", "bad_research"])
def test_evidence_pass_can_use_tools_but_is_not_an_infinite_or_numeric_recovery(tmp_path, monkeypatch, mode):
    from src.agent.grounding.query_budget import SessionEvidenceStore
    from src.agent.loop import AgentLoop
    from src.agent.tools import BaseTool, ToolRegistry
    from src.agent.trace import TraceWriter
    from tests.test_grounding_release import _Response, _ScriptedLLM, _tool_call
    from tests.test_session_dated_evidence import _quote

    saved = json.loads((FIXTURES / "etf_identity.json").read_text())
    call, result = _quote(saved)
    monkeypatch.setattr("src.agent.loop.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("time.time", lambda: result["ts"] + 7200)
    store = SessionEvidenceStore(tmp_path / "followup/research_evidence.json", now=result["ts"])
    store.record(call["tool"], call["args"], result["result"], call["call_id"], {"513330.SH"})

    class Search(BaseTool):
        name = "web_search"
        description = "Find primary source candidates"
        parameters = {"type": "object", "properties": {"query": {"type": "string"}}}
        calls = 0

        def execute(self, **kwargs):
            self.calls += 1
            return json.dumps({"ok": True, "results": []})

    tool, registry = Search(), ToolRegistry()
    if mode != "no_tools":
        registry.register(tool)
    draft = COMPLETE + GAP + "513330.SH（腾讯，CNY）历史快照价格0.338元，不是当前实时价格。"
    first = draft + ("目标价0.999元。" if mode == "numeric" else "")
    responses = [_Response(content=first)]
    if mode == "research":
        responses.append(_Response(tool_calls=[_tool_call("primary-source", "web_search", query="513330 指数估值 官方报告")]))
    if mode == "bad_research":
        responses.append(_Response(content=draft + "目标价0.999元。"))
    responses.append(_Response(content=draft))
    llm = _ScriptedLLM(responses)
    agent = AgentLoop(registry=registry, llm=llm, max_iterations=8)
    run = tmp_path / "run"
    run.mkdir()
    agent.memory.run_dir = str(run)
    question = QUESTION + ("不要补查" if mode == "no_fetch" else "")
    answer = agent.run(question, session_id="followup", history=[
        {"role": "user", "content": "请分析恒生互联网，代码513330，已持有一年。"},
        {"role": "assistant", "content": "报告", "grounding_identity": saved["identity"]},
    ])
    assert answer["status"] == "success"
    assert "0.338" in answer["content"] and "0.999" not in answer["content"]
    trace = TraceWriter.read(tmp_path / "followup")
    review = [e for e in trace if e["type"] == "holding_evidence_review"]
    should_review = mode in {"research", "unavailable", "bad_research"}
    assert len(review) == int(should_review)
    assert tool.calls == int(mode == "research")
    assert llm.calls == (3 if mode in {"research", "bad_research"} else 2 if should_review or mode == "numeric" else 1)
    if should_review:
        assert any(d["function"]["name"] == "web_search" for d in llm.tools_history[1])
        assert "opposing evidence" in llm.messages_history[1][-1]["content"]
        assert "not a text-only" in llm.messages_history[1][-1]["content"]
        costs = json.loads((run / "research_cost.json").read_text())
        assert costs["events"]["holding_evidence_review"] == 1
    if mode == "numeric":
        assert llm.tools_history[1] is None
    if mode == "bad_research":
        assert llm.tools_history[2] is None
        assert any(e["type"] == "answer_rejected" for e in trace)
