import copy
from pathlib import Path

import pytest

from src.agent.grounding.followup_tools import holding_followup_tools
from src.agent.grounding.ledger import GroundingLedger

PROMPT = (Path(__file__).parent / "fixtures/research_20261012/holding_context_prompt.txt").read_text()
AUDIT = {
    "status": "locked", "authorized_symbols": ["513330.SH"], "primary_symbols": ["513330.SH"],
    "records": [{"status": "locked", "symbol": "513330.SH", "currency": "CNY"}],
}


def _history():
    return [
        {"role": "user", "content": "请分析恒生互联网，基金代码513330，已持有一年。"},
        {"role": "assistant", "content": "报告", "grounding_identity": copy.deepcopy(AUDIT)},
        {"role": "user", "content": "我应该怎么办"},
        {"role": "assistant", "content": "条件性持有", "metadata": {
            "grounding_identity": {"status": "unresolved", "records": []}}},
    ]


def test_real_long_prompt_inherits_audited_subject_through_empty_old_followup(tmp_path):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=PROMPT, history=_history())
    assert ledger.inherited_symbols == {"513330.SH"}
    assert ledger.identity_status == "locked"


@pytest.mark.parametrize("change", ["new_name", "longer_name", "new_subject", "ambiguous", "conflicting", "no_audit", "multi_code"])
def test_context_reference_does_not_authorize_new_or_uncertain_subject(tmp_path, change):
    history, prompt = _history(), PROMPT
    if change == "new_name":
        prompt = prompt.replace("恒生互联网", "山东黄金")
    elif change == "longer_name":
        prompt = prompt.replace("恒生互联网", "恒生互联网科技")
    elif change == "new_subject":
        history += [{"role": "user", "content": "分析山东黄金"}, {"role": "assistant", "content": "尚未完成"}]
    elif change == "ambiguous":
        history[1]["grounding_identity"]["records"][0]["status"] = "ambiguous"
    elif change == "conflicting":
        history[-1]["metadata"]["grounding_identity"] = {
            "status": "conflicting", "records": [{"query": "513330", "status": "conflicting"}]}
    elif change == "no_audit":
        del history[1]["grounding_identity"]
    elif change == "multi_code":
        history[0]["content"] += "还有513220.SH。"
        audit = history[1]["grounding_identity"]
        audit["authorized_symbols"].append("513220.SH")
        audit["primary_symbols"].append("513220.SH")
        audit["records"].append({"status": "locked", "symbol": "513220.SH"})
    ledger = GroundingLedger(run_dir=tmp_path, user_message=prompt, history=history)
    assert not ledger.inherited_symbols


@pytest.mark.parametrize("extra", ["请帮我回测策略", "请使用投委会团队", "请写成文件", "请执行交易下单", "请换成山东黄金", "请比较招商银行"])
def test_expanded_long_prompt_keeps_full_toolbox(tmp_path, extra):
    ledger = GroundingLedger(run_dir=tmp_path, user_message=PROMPT + extra, history=_history())
    ledger._imported_research_calls.add("original")
    definitions = [{"function": {"name": name}} for name in ["web_search", "write_file"]]
    assert holding_followup_tools(PROMPT + extra, ledger, definitions) is None


def test_alias_without_explicit_code_requires_single_locked_primary(tmp_path):
    history = _history()
    history[0]["content"] = "请分析恒生互联网持仓。"
    ledger = GroundingLedger(run_dir=tmp_path, user_message=PROMPT, history=history)
    assert ledger.inherited_symbols == {"513330.SH"}
    history[1]["grounding_identity"]["status"] = "conflicting"
    assert not GroundingLedger(run_dir=tmp_path / "bad", user_message=PROMPT, history=history).inherited_symbols
