"""Web reader tool: fetch a URL as Markdown text via the Jina Reader API."""

from __future__ import annotations

import ipaddress
import json
import logging
from urllib.parse import urlsplit

import requests

from src.agent.progress import emit_progress
from src.agent.tools import BaseTool
from src.security.scanner import with_security_warnings
from src.tools.public_financial_snapshot import public_financial_snapshot

logger = logging.getLogger(__name__)

_JINA_PREFIX = "https://r.jina.ai/"
_TIMEOUT = 30
_MAX_LENGTH = 8000
_CACHED_MARKER = "Warning: This is a cached snapshot"


def _url_allowed(url: str) -> tuple[bool, str]:
    """Return whether a URL is safe to forward to the remote reader service."""
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return False, "target URL is not allowed"

    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "target URL is not allowed"
    if not parsed.hostname:
        return False, "target URL is not allowed"
    if parsed.username or parsed.password:
        return False, "target URL is not allowed"

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        return False, "target URL is not allowed"

    ip_host = host.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(ip_host)
    except ValueError:
        return True, ""

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global
    ):
        return False, "target URL is not allowed"
    return True, ""


def read_url(url: str, no_cache: bool = False, offset: int = 0, max_chars: int = _MAX_LENGTH) -> str:
    """Fetch web page content via the Jina Reader API.

    The full URL (including query string) is sent to the third-party Jina
    Reader service (r.jina.ai); never pass credentials/tokens or private
    addresses. Results may be a cached snapshot.

    Args:
        url: Target URL.
        no_cache: When true, ask the reader for a fresh (uncached) fetch.
        offset: Character offset for a later section of a long document.
        max_chars: Bounded page size, from 1 to 8000 characters.

    Returns:
        JSON result with title, content, url; ``cached: true`` is added
        when the reader served a stale snapshot.
    """
    target_url = url.strip()
    if (isinstance(offset, bool) or not isinstance(offset, int) or offset < 0
            or isinstance(max_chars, bool) or not isinstance(max_chars, int)
            or not 1 <= max_chars <= _MAX_LENGTH):
        return json.dumps({"status": "error", "error": "offset must be a nonnegative integer; max_chars must be 1..8000"})
    allowed, error = _url_allowed(target_url)
    if not allowed:
        return json.dumps({"status": "error", "error": error}, ensure_ascii=False)

    try:
        headers = {"Accept": "text/markdown"}
        if no_cache:
            headers["x-no-cache"] = "true"
        emit_progress(
            "fetching",
            message=f"GET {target_url[:60]}{'…' if len(target_url) > 60 else ''}",
        )
        resp = requests.get(
            f"{_JINA_PREFIX}{target_url}",
            headers=headers,
            timeout=_TIMEOUT,
        )
        emit_progress("parsing", message="extracting markdown")
        if resp.status_code != 200:
            logger.warning("read_url upstream HTTP %s: %s", resp.status_code, resp.text[:500])
            return json.dumps({
                "status": "error",
                "error": f"remote reader returned HTTP {resp.status_code}: {resp.text[:500]}",
            }, ensure_ascii=False)

        text = resp.text
        title = ""
        for line in text.split("\n"):
            if line.startswith("Title:"):
                title = line[6:].strip()
                break

        total_length = len(text)
        if offset > total_length:
            return json.dumps({
                "status": "error", "error": "offset exceeds document length",
                "length": total_length,
            }, ensure_ascii=False)
        end = min(offset + max_chars, total_length)
        text = text[offset:end]
        if end < total_length:
            text += f"\n\n... (truncated, total {total_length} chars; next offset {end})"

        result = {
            "status": "ok",
            "title": title,
            "url": target_url,
            "content": text,
            "length": len(resp.text),
            "pagination": {
                "offset": offset, "returned": end - offset,
                "next_offset": end if end < total_length else None,
                "complete": end == total_length,
            },
        }
        if _CACHED_MARKER in resp.text:
            result["cached"] = True
        snapshot = public_financial_snapshot(target_url, text)
        if snapshot:
            result["web_financial"] = snapshot
        result = with_security_warnings(result, fields=("content",))
        return json.dumps(result, ensure_ascii=False)

    except requests.Timeout:
        return json.dumps({"status": "error", "error": f"Request timed out ({_TIMEOUT}s)"}, ensure_ascii=False)
    except Exception as exc:
        logger.warning("read_url request failed: %s", exc)
        return json.dumps(
            {"status": "error", "error": f"remote reader request failed: {exc}"},
            ensure_ascii=False,
        )


class WebReaderTool(BaseTool):
    """Web reader tool."""

    name = "read_url"
    description = (
        "Fetch public web pages or reader-supported documents as Markdown. "
        "Long documents are paged: use pagination.next_offset to read later financial tables or notes. "
        "Recognized Tencent quote and ChinaAMC ETF product pages also return web_financial "
        "structured fields. Cite these as observed with call_id::web_financial.last_price or "
        "call_id::web_financial.unit_nav. Daily unit_nav is not intraday IOPV. "
        "A successful page read does not mean the full filing was read."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL of the web page to read"},
            "offset": {"type": "integer", "minimum": 0, "default": 0,
                       "description": "Character offset. Use pagination.next_offset from the prior result for later sections."},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": _MAX_LENGTH, "default": _MAX_LENGTH,
                          "description": "Maximum characters to return in this page."},
            "no_cache": {
                "type": "boolean",
                "description": (
                    "Request a fresh (uncached) fetch. Use only when the prior "
                    "read explicitly reported cached=true or the task genuinely "
                    "requires newer data. Never set no_cache merely because a "
                    "previous result was compacted or replayed."
                ),
                "default": False,
            },
        },
        "required": ["url"],
    }
    repeatable = True
    # Normal repeated reads remain allowed while the prior payload is visible.
    # If compaction removes an exact successful read, AgentLoop may restore the
    # run-scoped result instead of hitting Jina/the origin again. ``no_cache``
    # remains the explicit freshness escape hatch and bypasses that replay.
    replay_after_compaction = True

    def execute(self, **kwargs) -> str:
        """Fetch web page."""
        return read_url(kwargs["url"], no_cache=bool(kwargs.get("no_cache", False)),
                        offset=kwargs.get("offset", 0), max_chars=kwargs.get("max_chars", _MAX_LENGTH))
