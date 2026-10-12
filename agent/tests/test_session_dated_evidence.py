"""Follow-up continuity without trusting prose or caching a fresh quote."""
import json
from pathlib import Path

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.query_budget import SessionEvidenceStore, dated_evidence_context


@pytest.fixture
def saved():
    return json.loads((Path(__file__).parent / "fixtures/research_20261012/etf_identity.json").read_text())


def test_real_hk_alternative_resolution_unlocks_financials_without_changing_fund(tmp_path, saved):
    ledger = GroundingLedger(run_dir=tmp_path, user_message="分析513330.SH主要成分股财报和投资价值")
    calls = {}
    for event in saved["events"]:
        if event["type"] == "tool_call":
            calls[event["call_id"]] = event
        elif event["type"] == "tool_result" and event["tool"] == "search_symbol":
            call = calls[event["call_id"]]
            ledger.ingest_tool_result(tool_name=call["tool"], arguments=call["args"], result=event["result"],
                                      call_id=event["call_id"], success=event["status"] == "ok")
    assert ledger.primary_symbols == {"513330.SH"}
    assert ledger.authorized_symbols >= {"513330.SH", "00700.HK", "09988.HK", "09999.HK", "03690.HK"}
    for symbol in ("00700.HK", "09988.HK", "09999.HK", "03690.HK"):
        auth = ledger.authorize_tool_call("get_financial_statements", {"code": symbol},
            batch_authorized_symbols=ledger.authorized_symbols, call_id="financial-" + symbol)
        assert auth.allowed
    assert not ledger.authorize_tool_call("get_financial_statements", {"code": "00700.TW"},
        batch_authorized_symbols=ledger.authorized_symbols, call_id="wrong-venue").allowed
    # Resolver results still cannot grant permission within the same batch.
    assert not ledger.authorize_tool_call("get_financial_statements", {"code": "00700.HK"},
        batch_authorized_symbols={"513330.SH"}, call_id="same-batch").allowed


@pytest.mark.parametrize("question", ["我应该怎么办", "那你说不能躺平，我该怎么做，什么时候卖出降低成本"])
def test_real_followups_inherit_subject_but_new_subject_does_not(tmp_path, saved, question):
    history = [{"role": "user", "content": "分析513330，已持有，一年期限"},
               {"role": "assistant", "content": "历史报告", "grounding_identity": saved["identity"]}]
    ledger = GroundingLedger(run_dir=tmp_path, user_message=question, history=history)
    assert ledger.inherited_symbols == {"513330.SH"}
    other = GroundingLedger(run_dir=tmp_path / "other", user_message="分析招商银行600036.SH", history=history)
    assert not other.inherited_symbols


def test_legacy_empty_followups_do_not_break_continuity_or_cross_subjects(tmp_path, saved):
    history = [{"role": "user", "content": "分析513330，已持有，一年期限"},
               {"role": "assistant", "grounding_identity": saved["identity"]}]
    for question, status in [("我应该怎么办", "not_required"),
                             ("那你说不能躺平，我该怎么做，什么时候卖出降低成本", "unresolved")]:
        history.extend([{"role": "user", "content": question},
                        {"role": "assistant", "grounding_identity": {"status": status, "records": []}}])
    ledger = GroundingLedger(run_dir=tmp_path, user_message="我应该怎么办", history=history)
    assert ledger.inherited_symbols == {"513330.SH"}
    history.extend([{"role": "user", "content": "现在讨论SpaceX"},
                    {"role": "assistant", "grounding_identity": {"status": "not_required", "records": []}}])
    other = GroundingLedger(run_dir=tmp_path / "other", user_message="我应该怎么办", history=history)
    assert not other.inherited_symbols


def _quote(saved):
    call = next(e for e in saved["events"] if e["type"] == "tool_call" and e["tool"] == "get_a_share_valuation")
    result = next(e for e in saved["events"] if e["type"] == "tool_result" and e["call_id"] == call["call_id"])
    return call, result


def test_dated_quote_survives_two_hour_followup_but_never_satisfies_fresh_lookup(tmp_path, saved):
    call, result = _quote(saved)
    path = tmp_path / "evidence.json"
    store = SessionEvidenceStore(path, now=result["ts"])
    store.record(call["tool"], call["args"], result["result"], call["call_id"], {"513330.SH"})
    later = SessionEvidenceStore(path, now=result["ts"] + 7200)
    assert len(later.items) == 1
    assert later.lookup(call["tool"], call["args"], {"513330.SH"}) is None
    ledger = GroundingLedger(run_dir=tmp_path / "followup", user_message="我应该怎么办", history=[
        {"role": "user", "content": "分析513330，已持有，一年期限"},
        {"role": "assistant", "grounding_identity": saved["identity"]}])
    ledger.ingest_research_handoff(later.items, source_call_id="same_session_dated_evidence")
    context = dated_evidence_context(ledger)
    assert call["call_id"] in context and "0.338" in context and "2026-10-12" in context
    assert "NOT new fetches" in context
    other = GroundingLedger(run_dir=tmp_path / "other", user_message="600036.SH现在能买吗")
    other.ingest_research_handoff(later.items, source_call_id="same_session_dated_evidence")
    assert not other._imported_research_calls
    assert not SessionEvidenceStore(path, now=result["ts"] + 8 * 86400).items


def test_trace_migration_recovers_original_quote_and_not_failed_financials(tmp_path, saved):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(e) for e in saved["events"]))
    _, result = _quote(saved)
    store = SessionEvidenceStore(tmp_path / "evidence.json", now=result["ts"] + 7200)
    store.recover_trace(trace)
    assert any(i["tool"] == "get_a_share_valuation" for i in store.items)
    assert not any(i["tool"] in {"get_financial_statements", "search_symbol"} for i in store.items)
    assert all(i["fetched_at"] < store.now for i in store.items)


def test_long_web_documents_cannot_push_quote_out_of_followup_inventory(tmp_path, saved):
    from dataclasses import replace

    call, result = _quote(saved)
    ledger = GroundingLedger(run_dir=tmp_path, user_message="分析513330.SH")
    ledger.ingest_tool_result(tool_name=call["tool"], arguments=call["args"], result=result["result"],
                              call_id=call["call_id"], success=True)
    quote = next(r for r in ledger._evidence if r.field.endswith("last_price"))
    ledger._evidence[:0] = [replace(quote, tool="read_url", call_id="long-document", field=f"data[{i}]")
                            for i in range(200)]
    ledger._imported_research_calls.update({call["call_id"], "long-document"})
    context = dated_evidence_context(ledger)
    assert call["call_id"] in context
    assert context.count('"call_id": "long-document"') == 16
    assert len(context) < 14000


def test_followup_loop_gets_tool_evidence_without_new_fetch(tmp_path, saved, monkeypatch):
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry
    from tests.test_grounding_release import _Response, _ScriptedLLM

    monkeypatch.setattr("src.agent.loop.SESSIONS_DIR", tmp_path)
    session = tmp_path / "followup"
    session.mkdir()
    (session / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in saved["events"]) + "\n")
    _, result = _quote(saved)
    monkeypatch.setattr("time.time", lambda: result["ts"] + 7200)
    llm = _ScriptedLLM([_Response(content="513330.SH（腾讯，CNY）历史快照价格0.338元；不是当前实时价格。")])
    agent = AgentLoop(registry=ToolRegistry(), llm=llm, max_iterations=3)
    run = tmp_path / "run"
    run.mkdir()
    agent.memory.run_dir = str(run)
    answer = agent.run("我应该怎么办", session_id="followup", history=[
        {"role": "user", "content": "分析513330，已持有，一年期限"},
        {"role": "assistant", "content": "既有报告", "grounding_identity": saved["identity"]}])
    assert answer["status"] == "success" and llm.calls == 1
    assert "[SAME SESSION DATED EVIDENCE]" in llm.messages_history[0][0]["content"]
    assert agent._grounding._imported_research_calls
    costs = json.loads((run / "research_cost.json").read_text())
    assert costs["events"]["session_evidence_restored"] == 1
    assert costs["totals"]["tool_calls"] == 0
