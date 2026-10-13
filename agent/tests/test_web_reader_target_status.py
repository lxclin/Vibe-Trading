import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tools import web_reader_tool
from src.tools.web_reader_status import target_page_failure


@pytest.mark.parametrize("title,text,failed", [
    ("404", "A large navigation menu", True),
    ("404 - Not Found", "", True),
    ("Webpage Moved", "", True),
    ("Normal title", "Warning: Target URL returned error 403: Forbidden\n", True),
    ("Annual Report", "404 customers paid their bills; previous webpage moved", False),
    ("How to fix 404 errors", "Page not found troubleshooting", False),
    ("Factsheets", "https://example.com/errors/404/", False),
])
def test_failure_detection_uses_metadata_not_article_keywords(title, text, failed):
    assert bool(target_page_failure(title, text)) is failed


def test_real_hsi_soft_404_is_error_before_pagination_or_financial_parse(monkeypatch):
    saved = json.loads((Path(__file__).parent / "fixtures/research_20261012/reader_404.json").read_text())
    monkeypatch.setattr(web_reader_tool.requests, "get", lambda *a, **k: SimpleNamespace(status_code=200, text=saved["content"]))
    monkeypatch.setattr(web_reader_tool, "public_financial_snapshot", lambda *a: pytest.fail("error page must not be parsed"))
    result = json.loads(web_reader_tool.read_url(saved["url"], offset=1000, max_chars=20))
    assert result["status"] == "error" and result["target_status"] == 404
    assert "content" not in result and "web_financial" not in result


def test_legacy_error_page_cannot_be_restored_as_successful_evidence(tmp_path):
    from src.agent.grounding.query_budget import SessionEvidenceStore
    from src.agent.grounding.research_handoff import scoped_receipts
    saved = json.loads((Path(__file__).parent / "fixtures/research_20261012/reader_404.json").read_text())
    receipt = {"tool": "read_url", "arguments": {"url": saved["url"]}, "call_id": "old-reader",
               "scope_symbols": ["513330.SH"], "success": True, "result": json.dumps(saved)}
    assert not scoped_receipts([receipt], {"513330.SH"})
    store = SessionEvidenceStore(tmp_path / "evidence.json", now=100)
    store.record(receipt["tool"], receipt["arguments"], receipt["result"], receipt["call_id"], {"513330.SH"})
    assert not store.items
