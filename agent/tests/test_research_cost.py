import json
from types import SimpleNamespace

import pytest

from src.agent.research_cost import summarize, write_report
from src.agent.loop import AgentLoop
from src.agent.trace import TraceWriter


def test_costs_include_retries_missing_usage_and_overlapping_tools():
    report = summarize([
        {"type": "llm_call_cost", "iter": 1, "status": "error", "elapsed_ms": 20, "usage": None},
        {"type": "llm_call_cost", "iter": 1, "status": "ok", "elapsed_ms": 30,
         "usage": {"input_tokens": 100, "output_tokens": 12}},
        {"type": "tool_call", "iter": 1, "tool": "read_url", "args": {"url": "x"}},
        {"type": "tool_call", "iter": 1, "tool": "read_url", "args": {"url": "x"}},
        {"type": "tool_result", "iter": 1, "tool": "read_url", "elapsed_ms": 80, "status": "ok"},
        {"type": "tool_result", "iter": 1, "tool": "read_url", "elapsed_ms": 70, "status": "error"},
    ])
    assert report["totals"]["model_calls"] == 2
    assert report["totals"]["model_elapsed_ms"] == 50
    assert report["totals"]["input_tokens"] == 100
    assert report["totals"]["model_errors"] == 1
    assert report["usage_complete"] is False
    assert report["totals"]["tool_elapsed_ms_sum"] == 150
    assert report["repeated_exact_queries"] == 1


def test_historical_usage_does_not_invent_model_duration_or_failed_call_count():
    report = summarize([{"type": "tool_result", "iter": 3, "tool": "web_search",
                         "elapsed_ms": 5, "result": '{"results":[]}'}],
                       usage_rows=[{"iter": 3, "input_tokens": 10, "output_tokens": 2}])
    assert report["totals"]["model_elapsed_ms"] is None
    assert report["usage_complete"] is False
    assert report["totals"]["input_tokens"] == 10
    assert report["totals"]["empty_searches"] == 1
    assert summarize([{"type": "start", "iter": 1}])["totals"]["input_tokens"] is None


def test_report_scopes_out_previous_session_turns_and_truncated_line(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('\n'.join(json.dumps(e) for e in [
        {"type": "llm_call_cost", "iter": 1, "elapsed_ms": 999},
        {"type": "llm_call_cost", "iter": 2, "elapsed_ms": 10},
    ]) + '\n{"type":')
    report = write_report(tmp_path, trace, first_iteration=2)
    assert report["totals"]["model_elapsed_ms"] == 10
    assert json.loads((tmp_path / "research_cost.json").read_text()) == report


def test_failed_model_call_has_a_cost_event(tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    agent = SimpleNamespace(llm=SimpleNamespace(stream_chat=fail))
    trace = TraceWriter(tmp_path)
    with pytest.raises(RuntimeError, match="provider failed"):
        AgentLoop._stream_chat_with_cost(agent, trace, 2, "retry", [])
    trace.close()
    event = list(TraceWriter.read(tmp_path))[0]
    assert event["type"] == "llm_call_cost" and event["status"] == "error"
    assert event["usage"] is None
    assert event["elapsed_ms"] >= 0
