import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.grounding.followup_tools import holding_followup_tools


def _definitions():
    return [{"type": "function", "function": {"name": name, "description": "x" * length}}
            for name, length in [("get_a_share_valuation", 100), ("web_search", 100), ("write_file", 5000)]]


@pytest.mark.parametrize("question", ["我应该怎么办", "什么时候卖出降低成本", "那你说不能躺平，我该怎么做，什么时候卖出降低成本"])
def test_verified_holding_followup_gets_research_tools(question):
    ledger = SimpleNamespace(inherited_symbols={"513330.SH"}, _imported_research_calls={"quote"})
    scoped = holding_followup_tools(question, ledger, _definitions())
    assert {d["function"]["name"] for d in scoped} == {"get_a_share_valuation", "web_search"}


@pytest.mark.parametrize("question", ["继续", "帮我回测策略", "我应该怎么办，写成文件", "我应该怎么办，使用投委会团队", "分析600036.SH"])
def test_open_ended_or_expanded_requests_keep_full_registry(question):
    ledger = SimpleNamespace(inherited_symbols={"513330.SH"}, _imported_research_calls={"quote"})
    assert holding_followup_tools(question, ledger, _definitions()) is None


def test_no_identity_or_no_original_receipts_keeps_full_registry():
    for symbols, calls in [(set(), {"quote"}), ({"513330.SH"}, set())]:
        ledger = SimpleNamespace(inherited_symbols=symbols, _imported_research_calls=calls)
        assert holding_followup_tools("我应该怎么办", ledger, _definitions()) is None


@pytest.mark.parametrize("expand", [False, True])
def test_loop_scopes_schemas_and_never_executes_unoffered_batch(tmp_path, monkeypatch, expand):
    from src.agent.grounding.query_budget import SessionEvidenceStore
    from src.agent.loop import AgentLoop
    from src.agent.tools import BaseTool, ToolRegistry
    from src.agent.trace import TraceWriter
    from tests.test_grounding_release import _Response, _ScriptedLLM, _tool_call
    from tests.test_session_dated_evidence import _quote
    from tests.test_holding_delivery import COMPLETE

    saved = json.loads((Path(__file__).parent / "fixtures/research_20261012/etf_identity.json").read_text())
    call, result = _quote(saved)
    monkeypatch.setattr("src.agent.loop.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("time.time", lambda: result["ts"] + 7200)
    store = SessionEvidenceStore(tmp_path / "followup/research_evidence.json", now=result["ts"])
    store.record(call["tool"], call["args"], result["result"], call["call_id"], {"513330.SH"})

    class Tool(BaseTool):
        description = "x" * 2000
        parameters = {"type": "object", "properties": {}}

        def __init__(self, name):
            self.name, self.calls = name, 0

        def execute(self, **kwargs):
            self.calls += 1
            return result["result"]

    registry = ToolRegistry()
    quote, write = Tool("get_a_share_valuation"), Tool("write_file")
    registry.register(quote)
    registry.register(write)
    responses = [_Response(content=COMPLETE + "513330.SH（腾讯，CNY）历史快照价格0.338元；不是当前实时价格。")]
    if expand:
        responses.insert(0, _Response(tool_calls=[_tool_call("not-offered", "write_file", path="unused.md")]))
    llm = _ScriptedLLM(responses)
    agent = AgentLoop(registry=registry, llm=llm, max_iterations=5)
    run = tmp_path / "run"
    run.mkdir()
    agent.memory.run_dir = str(run)
    answer = agent.run("我应该怎么办", session_id="followup", history=[
        {"role": "user", "content": "分析513330，已持有，一年期限"},
        {"role": "assistant", "content": "已有报告", "grounding_identity": saved["identity"]}])
    assert answer["status"] == "success" and write.calls == quote.calls == 0
    assert [d["function"]["name"] for d in llm.tools_history[0]] == ["get_a_share_valuation"]
    assert "[HOLDING FOLLOW-UP]" in llm.messages_history[0][0]["content"]
    assert "do not ask again" in llm.messages_history[0][0]["content"]
    trace = TraceWriter.read(tmp_path / "followup")
    scope = next(e for e in trace if e["type"] == "research_tool_scope")
    assert scope["offered_schema_chars"] < scope["full_schema_chars"]
    costs = json.loads((run / "research_cost.json").read_text())
    assert costs["events"]["research_tool_scope"] == 1
    if expand:
        assert llm.calls == 2
        assert {d["function"]["name"] for d in llm.tools_history[1]} == {"get_a_share_valuation", "write_file"}
        assert any(e["type"] == "research_tool_scope_expanded" for e in trace)
        assert not any(e["type"] == "tool_call" for e in trace)
    else:
        assert llm.calls == 1
