import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools import hsi_index_resources_tool as resource_tool
from src.tools.public_index_snapshot import public_index_snapshot

FIXTURES = Path(__file__).parent / "fixtures/research_20261012"
CONTENT = (FIXTURES / "hsi_factsheet.txt").read_text()
URL = resource_tool._download_url("/public-website/webapp/factsheets/en-hk/hsiii.pdf")


@pytest.fixture
def directory(monkeypatch):
    data = {key: json.loads((FIXTURES / f"hsi_{key}.json").read_text()) for key in ["indexes", "factsheets"]}
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        payload = data["factsheets" if "content/gcs/download" in url else "indexes"]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(resource_tool.requests, "get", get)
    return data, calls


def test_exact_index_uses_published_file_not_a_guessed_detail_page(directory):
    _, calls = directory
    result = json.loads(resource_tool.lookup_hsi_resources("HSIII"))
    assert result["unique_match"] and result["total_matches"] == 1
    assert result["candidates"][0]["factsheet_url"] == URL
    assert result["candidates"][0]["index_type"] == "PI"
    assert len(calls) == 2


def test_partial_name_preserves_similar_indexes_and_return_variants(directory):
    result = json.loads(resource_tool.lookup_hsi_resources("Internet"))
    assert not result["unique_match"] and result["total_matches"] > 1
    assert {row["index_type"] for row in result["candidates"]} >= {"PI", "TRI_Grs", "TRI_Net"}


def test_duplicate_conflicting_files_and_external_paths_are_not_selected(directory):
    data, _ = directory
    row = next(r for r in data["factsheets"]["data"] if r["seriesCode"] == "hsiii")
    other = copy.deepcopy(row)
    other["fileEng"] = "/public-website/webapp/factsheets/en-hk/different.pdf"
    data["factsheets"]["data"].append(other)
    result = json.loads(resource_tool.lookup_hsi_resources("HSIII"))
    assert result["candidates"][0]["resource_status"] == "ambiguous"
    assert result["candidates"][0]["factsheet_url"] is None
    row["fileEng"] = "https://example.com/untrusted.pdf"
    data["factsheets"]["data"].remove(other)
    assert json.loads(resource_tool.lookup_hsi_resources("HSIII"))["candidates"][0]["factsheet_url"] is None


def test_directory_failure_returns_no_guessed_link(directory):
    data, _ = directory
    data["indexes"]["code"] = 500
    result = json.loads(resource_tool.lookup_hsi_resources("HSIII"))
    assert result["status"] == "error" and "candidates" not in result


def test_real_monthly_statistics_keep_date_and_do_not_invent_percentile():
    snapshot = public_index_snapshot(URL, CONTENT)
    assert snapshot["index_code"] == "HSIII" and snapshot["as_of"] == "2026-09-30"
    assert snapshot["pe_ratio"] == 30.08 and snapshot["dividend_yield_pct"] == 1.14
    assert "symbol" not in snapshot and "percentile" not in snapshot


@pytest.mark.parametrize("change", ["wrong_host", "wrong_index", "conflicting_date", "future", "layout", "conflicting_values"])
def test_unknown_or_conflicting_factsheets_are_not_parsed(change):
    url, content = URL, CONTENT
    if change == "wrong_host":
        url = url.replace("www.hsi.com.hk", "example.com")
    elif change == "wrong_index":
        url = url.replace("hsiii.pdf", "hstech.pdf")
    elif change == "conflicting_date":
        content += "\nAll data as at 31 Aug 2026\n"
    elif change == "future":
        content = content.replace("30 Sep 2026", "30 Sep 2099")
    elif change == "layout":
        content = content.replace("Dividend Yield (%) PE Ratio (Times)", "PE Ratio (Times) Dividend Yield (%)")
    elif change == "conflicting_values":
        content += "\nINDEX FUNDAMENTALS\nIndex Dividend Yield (%) PE Ratio (Times) Annual Volatility*(%)\nHSIII 1.14 99.99 29.90\n"
    assert public_index_snapshot(url, content) is None


def test_ledger_reparses_original_and_index_pe_cannot_become_etf_quote(tmp_path):
    from src.agent.grounding.ledger import GroundingLedger
    ledger = GroundingLedger(run_dir=tmp_path, user_message="分析513330.SH")
    before = ledger.authorized_symbols
    ledger.ingest_tool_result(tool_name="read_url", arguments={"url": URL}, call_id="official-index", success=True,
                              result=json.dumps({"status": "ok", "url": URL, "content": CONTENT,
                                                 "index_financial": {"pe_ratio": 99.99}}))
    record = next(r for r in ledger._evidence if r.field == "index_financial.pe_ratio")
    assert record.value == 30.08 and record.symbol is None and record.timestamp == "2026-09-30"
    assert ledger.authorized_symbols == before
    good = ("HSIII指数截至2026-09-30市盈率30.08倍，来源恒生指数公司；不是基金本身的市盈率。\n"
            "```figures\n30.08 | observed | HSIII指数PE，2026-09-30 | official-index::index_financial.pe_ratio\n```")
    assert ledger.validate_final_answer(good).valid
    assert not ledger.validate_final_answer(good.replace("30.08", "30.09")).valid
    bad = ("513330.SH（恒生指数公司，CNY）当前价格30.08元。\n"
           "```figures\n30.08 | observed | 513330.SH当前价格 | official-index::index_financial.pe_ratio\n```")
    assert not ledger.validate_final_answer(bad).valid


def test_reader_returns_index_metadata_from_real_text(monkeypatch):
    from src.tools import web_reader_tool
    monkeypatch.setattr(web_reader_tool.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=200, text=CONTENT))
    result = json.loads(web_reader_tool.read_url(URL))
    assert result["index_financial"]["pe_ratio"] == 30.08
    assert result["index_financial"]["as_of"] == "2026-09-30"


def test_scoped_followup_can_discover_read_and_publish_dated_index_pe(tmp_path, monkeypatch, directory):
    from src.agent.grounding.query_budget import SessionEvidenceStore
    from src.agent.loop import AgentLoop
    from src.agent.tools import BaseTool, ToolRegistry
    from src.tools.web_reader_tool import WebReaderTool
    from tests.test_grounding_release import _Response, _ScriptedLLM, _tool_call
    from tests.test_holding_delivery import COMPLETE
    from tests.test_session_dated_evidence import _quote

    data, _ = directory
    def get(url, **kwargs):
        if url.startswith("https://r.jina.ai/"):
            return SimpleNamespace(status_code=200, text=CONTENT)
        payload = data["factsheets" if "content/gcs/download" in url else "indexes"]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(resource_tool.requests, "get", get)
    saved = json.loads((FIXTURES / "etf_identity.json").read_text())
    call, quote = _quote(saved)
    monkeypatch.setattr("src.agent.loop.SESSIONS_DIR", tmp_path)
    monkeypatch.setattr("time.time", lambda: quote["ts"] + 7200)
    store = SessionEvidenceStore(tmp_path / "followup/research_evidence.json", now=quote["ts"])
    store.record(call["tool"], call["args"], quote["result"], call["call_id"], {"513330.SH"})
    class Write(BaseTool):
        name, description, parameters = "write_file", "Unused file writer", {}
        def execute(self, **kwargs):
            pytest.fail("research must not write through an unrelated tool")
    registry = ToolRegistry()
    for tool in [resource_tool.HSIIndexResourcesTool(), WebReaderTool(), Write()]:
        registry.register(tool)
    final = COMPLETE + ("HSIII指数截至2026-09-30市盈率30.08倍，来源恒生指数公司；不是基金本身的市盈率。\n"
                        "```figures\n30.08 | observed | HSIII指数PE，2026-09-30 | official-index::index_financial.pe_ratio\n```")
    llm = _ScriptedLLM([
        _Response(tool_calls=[_tool_call("directory", "hsi_index_resources", query="HSIII")]),
        _Response(tool_calls=[_tool_call("official-index", "read_url", url=URL)]),
        _Response(content=final),
    ])
    agent = AgentLoop(registry=registry, llm=llm, max_iterations=8)
    run = tmp_path / "run"
    run.mkdir()
    agent.memory.run_dir = str(run)
    answer = agent.run("我应该怎么办", session_id="followup", history=[
        {"role": "user", "content": "请分析513330，已持有一年"},
        {"role": "assistant", "content": "已有研究", "grounding_identity": saved["identity"]},
    ])
    assert answer["status"] == "success" and llm.calls == 3
    assert "30.08" in answer["content"] and "2026-09-30" in answer["content"]
    assert {d["function"]["name"] for d in llm.tools_history[0]} == {"hsi_index_resources", "read_url"}
