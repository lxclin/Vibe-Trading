"""Exact receipt reuse, failed-query bounds and original reference identity."""
import json
from types import SimpleNamespace

import pytest

from src.agent.grounding.query_budget import ResearchQueryBudget, query_key
from tests.test_research_handoff_regression import SYMBOL, financial, quote


def test_exact_statement_reuse_retains_original_call_and_changed_inputs_fetch():
    receipt = financial("income", {"PARENT_NETPROFIT": 100})
    budget = ResearchQueryBudget([receipt])
    args = receipt["arguments"]
    assert budget.lookup("get_financial_statements", dict(reversed(list(args.items()))))["call_id"] == "income"
    assert budget.lookup("get_financial_statements", {**args, "offset": 20}) is None
    assert budget.lookup("get_financial_statements", {**args, "code": "600036.SH"}) is None
    assert budget.lookup("get_financial_statements", {**args, "no_cache": True}) is None
    assert budget.lookup("get_financial_statements", {**args, "fresh": True}) is None


def test_quotes_and_bars_are_never_reused_as_current_data():
    original = quote()
    budget = ResearchQueryBudget([original])
    assert budget.lookup(original["tool"], original["arguments"]) is None
    assert query_key("write_file", {"path": "a"}) is None


def test_two_failures_block_repeat_but_not_another_source_and_duplicates_do_not_count():
    first = {"tool": "web_search", "arguments": {"query": "601899 valuation"}, "result": '{"ok":false}',
             "call_id": "a", "success": False}
    budget = ResearchQueryBudget([first, first])
    assert not budget.blocked(first["tool"], first["arguments"])
    budget.record({**first, "call_id": "b"})
    assert budget.blocked(first["tool"], first["arguments"])
    assert not budget.blocked(first["tool"], {"query": "601899 official filing"})


def test_invalid_or_oversized_result_does_not_enter_reuse_cache():
    receipt = financial("income", {"PARENT_NETPROFIT": 100})
    for result in ("plain text", "[]", json.dumps({"big": "x" * 180001})):
        budget = ResearchQueryBudget([{**receipt, "result": result}])
        assert budget.lookup(receipt["tool"], receipt["arguments"]) is None


@pytest.mark.parametrize("scenario,expected_executions", [("repeat", 1), ("upstream", 0), ("failure", 2), ("stall", 1)])
def test_worker_reuses_receipts_or_stops_at_two_failed_fetches(tmp_path, monkeypatch, scenario, expected_executions):
    from src.providers.chat import LLMResponse, ToolCallRequest
    from src.swarm.models import SwarmAgentSpec, SwarmTask
    from src.swarm import worker
    from tests.test_swarm_worker_tool_result_truncation import _ScriptedChatLLM, FINAL_TEXT
    original = financial("income", {"PARENT_NETPROFIT": 100})
    args = original["arguments"]
    executions, events = [], []

    class Registry:
        def get_definitions(self):
            return [{"type": "function", "function": {"name": "get_financial_statements", "parameters": {}}}]

        def get(self, name):
            return None

        def execute(self, name, arguments):
            executions.append(arguments)
            return '{"status":"error","error":"offline"}' if scenario == "failure" else original["result"]

    count = 3 if scenario == "failure" else 4 if scenario == "stall" else 2
    llm = _ScriptedChatLLM([
        *[LLMResponse(tool_calls=[ToolCallRequest(id=f"call-{i}", name="get_financial_statements", arguments=args)]) for i in range(count)],
        LLMResponse(content=FINAL_TEXT),
    ])
    monkeypatch.setattr(worker, "build_swarm_registry", lambda *args, **kwargs: Registry())
    monkeypatch.setattr(worker, "ChatLLM", llm)
    result = worker.run_worker(
        agent_spec=SwarmAgentSpec(id="analyst", role="Analyst", system_prompt="Research", tools=["get_financial_statements"],
                                 research_worksheet=True, max_iterations=6),
        task=SwarmTask(id="task", agent_id="analyst", prompt_template=f"Research {SYMBOL}"),
        upstream_summaries={}, user_vars={}, run_dir=tmp_path,
        research_receipts=[original] if scenario == "upstream" else None,
        event_callback=lambda event: events.append(event),
    )
    assert len(executions) == expected_executions
    tool_messages = [message for message in llm.received_messages[-1] if message.get("role") == "tool"]
    if scenario == "stall":
        assert len(llm.received_messages) == 5
        assert any("Three research rounds" in (message.get("content") or "") for message in llm.received_messages[-1])
    if scenario == "failure":
        assert json.loads(tool_messages[-1]["content"])["error_code"] == "research_retry_limit"
        assert len(result.research_receipts) == 2  # refused call is not an executed receipt
    else:
        reused = json.loads(tool_messages[-1]["content"])["_research_reuse"]
        assert reused["original_call_id"] == ("income" if scenario == "upstream" else "call-0")
        assert len(result.research_receipts) == expected_executions


def test_main_loop_reuses_statement_payload_and_keeps_original_refs(tmp_path):
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry
    from tests.test_agent_loop_dedup_arguments import _PagedTool, _drive
    registry = ToolRegistry()
    tool = _PagedTool()
    tool.repeatable = True
    registry.register(tool)
    events = []
    agent = AgentLoop(registry=registry, llm=SimpleNamespace(), event_callback=lambda *event: events.append(event))
    agent.memory.run_dir = str(tmp_path)
    messages, trace = _drive(agent, tool.name, tmp_path, [{"statement": "income"}] * 2)
    assert len(tool.calls) == 1
    reused = json.loads(messages[-1]["content"])
    assert reused["call"] == 1
    assert reused["_research_reuse"]["original_call_id"] == "call_1"
    assert any(item[0] == "tool_result" and item[1].get("cached") is True for item in events)
    assert any(item["type"] == "tool_result_reused" for item in trace)


def test_session_history_preserves_cache_and_retry_flags():
    from src.session.service import SessionService
    trail = []
    SessionService._record_tool_trail_event(trail, "tool_call", {"tool": "web_search", "call_id": "q", "arguments": {"query": "test"}})
    SessionService._record_tool_trail_event(trail, "tool_result", {"tool": "web_search", "call_id": "q", "status": "error", "cached": False, "retry_exhausted": True})
    assert trail[0]["cached"] is False
    assert trail[0]["retry_exhausted"] is True
