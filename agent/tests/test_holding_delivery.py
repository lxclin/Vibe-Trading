import json
from pathlib import Path

import pytest

from src.agent.grounding.followup_tools import holding_constraints, holding_delivery_review

COMPLETE = (
    "当前倾向先不加仓，依据是持仓集中，盈利与估值仍有缺口。"
    "如果需要用款或存在融资，优先减仓以降低被迫变现风险，不等待回本。"
    "如果没有融资且不需要用钱，可条件性持有，不补仓。"
    "如果核心企业盈利和现金流持续恶化，持有理由失效，转向减仓；"
    "如果现金流持续改善且估值提供补偿，重新评估持有。"
)


def test_complete_framework_needs_no_extra_model_round():
    assert holding_delivery_review(COMPLETE) == []


@pytest.mark.parametrize("text", [
    "当前行动：复核风险。条件分支：持续关注。退出条件：注意市场。",
    "先告诉我有没有融资、是否需要用钱，我才能回答。",
    "我不提供交易指令，请自行决策。",
])
def test_labels_questions_and_refusal_do_not_count_as_action_plan(text):
    assert len(holding_delivery_review(text)) >= 2


def test_known_user_constraints_override_old_statements_and_ignore_questions():
    history = [
        {"role": "user", "content": "513330.SH，没有融资，需要用钱。"},
        {"role": "assistant", "content": "你有融资，不需要用钱"},
        {"role": "user", "content": "现在有融资，但不需要用钱。"},
        {"role": "user", "content": "假如没有融资怎么办？有没有借款？"},
    ]
    assert holding_constraints(history, {"513330.SH"}) == {"leverage": "有融资", "liquidity": "不需要用钱"}
    assert holding_constraints([{"role": "assistant", "content": "没有融资"}], {"513330.SH"}) == {}


def test_known_constraints_are_not_asked_again():
    requests = holding_delivery_review(COMPLETE + "你有没有融资？是否需要用钱？", {"leverage": "没有融资", "liquidity": "不需要用钱"})
    assert len(requests) == 1 and "instead of asking" in requests[0]


def test_no_cash_need_keeps_negation():
    assert holding_constraints([{"role": "user", "content": "513330，不需要用款。"}],
                               {"513330.SH"}) == {"liquidity": "不需要用款"}


def test_figures_declarations_cannot_satisfy_editorial_review():
    assert holding_delivery_review("持续关注。\n```figures\n" + COMPLETE + "\n```")


def test_real_answers_have_specific_delivery_gaps():
    saved = json.loads((Path(__file__).parent / "fixtures/research_20261012/holding_answers.json").read_text())
    assert len(saved) == 2
    for row in saved:
        requests = holding_delivery_review(row["answer"])
        assert requests, row["question"]
        assert all("price" in r or "leaning" in r or "constraints" in r for r in requests)


@pytest.mark.parametrize("mode", ["complete", "weak", "stubborn", "numeric", "tools", "empty", "bad_revision", "syntax"])
def test_loop_reviews_once_text_only_and_keeps_numeric_gate(tmp_path, monkeypatch, mode):
    from src.agent.grounding.query_budget import SessionEvidenceStore
    from src.agent.loop import AgentLoop
    from src.agent.tools import ToolRegistry
    from src.agent.trace import TraceWriter
    from tests.test_grounding_release import _Response, _ScriptedLLM, _tool_call
    from tests.test_session_dated_evidence import _quote

    saved = json.loads((Path(__file__).parent / "fixtures/research_20261012/etf_identity.json").read_text())
    call, result = _quote(saved)
    monkeypatch.setattr("src.agent.loop.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("time.time", lambda: result["ts"] + 7200)
    store = SessionEvidenceStore(tmp_path / "followup/research_evidence.json", now=result["ts"])
    store.record(call["tool"], call["args"], result["result"], call["call_id"], {"513330.SH"})
    quote = "513330.SH（腾讯，CNY）历史快照价格0.338元，不是当前实时价格。"
    weak = "建议复核风险，持续关注。" + quote
    first = COMPLETE + quote if mode == "complete" else weak
    if mode == "numeric":
        first += "目标价0.999元。"
    revised = weak if mode == "stubborn" else COMPLETE + quote
    if mode == "bad_revision":
        revised += "目标价0.999元。"
    second = _Response(content=revised)
    if mode == "tools":
        second = _Response(tool_calls=[_tool_call("blocked", "web_search", query="unnecessary")])
    if mode == "empty":
        second = _Response()
    if mode == "syntax":
        second = _Response(content='<││DSML││tool_calls><││invoke name="web_search">')
    llm = _ScriptedLLM([_Response(content=first), second])
    agent = AgentLoop(registry=ToolRegistry(), llm=llm, max_iterations=5)
    run = tmp_path / "run"
    run.mkdir()
    agent.memory.run_dir = str(run)
    answer = agent.run("我应该怎么办", session_id="followup", history=[
        {"role": "user", "content": "分析513330，已持有，一年期限，没有融资，不需要用钱"},
        {"role": "assistant", "content": "已完成研究", "grounding_identity": saved["identity"]}])
    assert answer["status"] == "success"
    assert llm.calls == (1 if mode == "complete" else 2)
    assert "0.338" in answer["content"] and "0.999" not in answer["content"]
    trace = TraceWriter.read(tmp_path / "followup")
    reviews = [e for e in trace if e["type"] == "research_delivery_review"]
    assert len(reviews) == (0 if mode == "complete" else 1)
    assert not any(e["type"] in {"tool_call", "grounding_recovery"} for e in trace)
    if mode != "complete":
        assert llm.tools_history[1] is None
    if mode == "numeric":
        assert reviews[0]["combined_with_grounding"] is True
        assert any(e["type"] == "answer_rejected" for e in trace)
    if mode in {"tools", "empty", "bad_revision", "syntax"}:
        assert "建议复核风险，持续关注" in answer["content"]
        assert len([e for e in trace if e["type"] == "research_delivery_revision_discarded"]) == 1
    assert "没有融资" in llm.messages_history[0][0]["content"]
