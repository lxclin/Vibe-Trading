import json
from pathlib import Path
from types import SimpleNamespace

from src.agent.grounding.research_sources import research_source_context


def _receipt(call="original", symbol="513330.SH", tool="read_url"):
    return {"tool": tool, "arguments": {"url": "https://example.com/report"},
            "call_id": call, "scope_symbols": [symbol], "success": True, "fetched_at": 100,
            "result": json.dumps({"status": "ok", "title": "Quarterly report", "url": "https://example.com/report",
                                  "content": "# Original report\n\nCash flow explanation.\n"})}


def test_originals_are_readable_in_run_and_keep_provenance_without_broadening_scope(tmp_path):
    ledger = SimpleNamespace(authorized_symbols={"513330.SH"}, _imported_research_calls={"original", "lead"})
    original = _receipt()
    text = research_source_context(ledger, [original, _receipt("lead", tool="web_search"),
                                          _receipt("other", "600036.SH"), _receipt("not-imported")], tmp_path)
    index = tmp_path / "artifacts/research_sources/index.json"
    rows = json.loads(index.read_text())
    assert {r["call_id"] for r in rows} == {"original", "lead"}
    row = next(r for r in rows if r["call_id"] == "original")
    assert Path(row["raw_receipt_path"]).read_text() == original["result"]
    assert Path(row["content_path"]).read_text() == json.loads(original["result"])["content"]
    assert Path(row["content_path"]).is_relative_to(tmp_path)
    assert row["fetched_at"] == 100
    assert next(r for r in rows if r["call_id"] == "lead")["kind"] == "search_candidates_only"
    assert "not a new independent source" in text
    assert "600036.SH" not in text


def test_failed_export_keeps_normal_context_path(tmp_path, monkeypatch):
    ledger = SimpleNamespace(authorized_symbols={"513330.SH"}, _imported_research_calls={"original"})
    def fail(*args, **kwargs):
        raise OSError("readonly")
    monkeypatch.setattr(Path, "write_text", fail)
    assert research_source_context(ledger, [_receipt()], tmp_path) == ""
