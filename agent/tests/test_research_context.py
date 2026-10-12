import json
from pathlib import Path

from src.agent.research_context import compact_financial_result


def payload():
    return {"ok": True, "source": "eastmoney", "statement": "income", "period": "quarter",
        "paging": {"offset": 2, "next_offset": 4, "complete": False},
        "data": {"600036.SH": {"periods": [
            {"REPORT_DATE": "2026-06-30", "CURRENCY": "CNY", "EPSJB": 2.98,
             "CREDIT_IMPAIRMENT_LOSS": -100, "ZERO": 0, "BOOL": False, "TEXT": "",
             **{f"MISSING_{i}": None for i in range(120)}},
            None,
            {"REPORT_DATE": "2025-12-31", "EPSJB": 5.70},
        ]}}}


def leaves(value, prefix=""):
    if isinstance(value, dict):
        return {p: v for k, item in value.items() for p, v in leaves(item, prefix + "." + k).items()}
    if isinstance(value, list):
        return {p: v for i, item in enumerate(value) for p, v in leaves(item, prefix + f"[{i}]").items()}
    return {prefix: value}


def test_projection_keeps_every_non_null_value_and_original_indexes(tmp_path):
    original = json.dumps(payload(), ensure_ascii=False)
    projected = compact_financial_result(original, run_dir=tmp_path, call_id="original")
    view = json.loads(projected)
    metadata = view.pop("_context_view")
    expected = {p: v for p, v in leaves(payload()).items() if v is not None}
    assert {p: v for p, v in leaves(view).items() if v is not None} == expected
    assert len(view["data"]["600036.SH"]["periods"]) == 3
    assert view["data"]["600036.SH"]["periods"][1] is None
    assert view["paging"] == payload()["paging"]
    assert Path(metadata["full_result_path"]).read_text() == original
    assert metadata["original_call_id"] == "original"
    assert len(projected) < len(original) * 0.9


def test_failed_or_incompatible_results_are_unchanged(tmp_path):
    original = json.dumps({**payload(), "ok": False})
    assert compact_financial_result(original, run_dir=tmp_path, call_id="x") == original
    assert compact_financial_result("not JSON" * 500, run_dir=tmp_path, call_id="x").startswith("not JSON")


def test_snapshot_failure_keeps_original_context(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    original = json.dumps(payload())
    assert compact_financial_result(original, run_dir=blocked, call_id="x") == original


def test_reused_context_retains_original_receipt_identity(tmp_path):
    original = json.dumps({**payload(), "_research_reuse": {"original_call_id": "first"}})
    view = json.loads(compact_financial_result(original, run_dir=tmp_path, call_id="second"))
    assert view["_context_view"]["original_call_id"] == "first"
    assert view["_research_reuse"]["original_call_id"] == "first"


def test_main_loop_projects_only_model_message_not_ledger_or_trace(tmp_path):
    from types import SimpleNamespace
    from src.agent.grounding import GroundingLedger
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry
    from tests.test_agent_loop_dedup_arguments import _PagedTool, _drive

    raw = json.dumps(payload())

    class Tool(_PagedTool):
        def execute(self, **kwargs):
            return raw

    registry = ToolRegistry()
    registry.register(Tool())
    agent = AgentLoop(registry=registry, llm=SimpleNamespace(), event_callback=lambda *_: None)
    agent.memory.run_dir = str(tmp_path)
    agent._grounding = GroundingLedger(run_dir=tmp_path, user_message="600036.SH现在值得买入吗？")
    messages, trace = _drive(agent, "get_financial_statements", tmp_path,
                             [{"code": "600036.SH", "statement": "income", "period": "quarter"}])
    assert "_context_view" in json.loads(messages[0]["content"])
    assert next(e for e in trace if e["type"] == "tool_result")["result"] == raw
    eps = [r for r in agent._grounding._evidence if r.field.endswith(".periods[0].EPSJB")]
    assert len(eps) == 1 and eps[0].value == 2.98 and eps[0].call_id == "call_1"
    restored = agent._research_context_result("get_financial_statements", raw, "replay-call",
                                               original_call_id="call_1")
    assert json.loads(restored)["_context_view"]["original_call_id"] == "call_1"


def test_saved_bank_statements_keep_all_values_with_smaller_context(tmp_path):
    from src.config.limits import truncate_tool_result
    fixture = Path(__file__).parent / "fixtures/research_20261010/bank_provenance.json"
    receipts = json.loads(fixture.read_text())["receipts"]
    before = after = 0
    changed = 0
    for receipt in receipts:
        if receipt["tool"] != "get_financial_statements":
            continue
        raw = receipt["result"]
        text = compact_financial_result(raw, run_dir=tmp_path, call_id=receipt["call_id"])
        before += len(truncate_tool_result(raw))
        after += len(truncate_tool_result(text))
        if text == raw:
            continue
        changed += 1
        view = json.loads(text)
        metadata = view.pop("_context_view")
        assert Path(metadata["full_result_path"]).read_text() == raw
        assert {p: v for p, v in leaves(view).items() if v is not None} == {
            p: v for p, v in leaves(json.loads(raw)).items() if v is not None}
    assert changed >= 10
    assert after < before * 0.8
