"""Compact financial tool envelopes without dropping any non-null evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config.limits import TOOL_RESULT_LIMIT


def _without_null_fields(value):
    if isinstance(value, dict):
        return {key: _without_null_fields(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        # Never remove list entries: evidence field paths include period indexes.
        return [_without_null_fields(item) for item in value]
    return value


def compact_financial_result(result: str, *, run_dir: Path, call_id: str) -> str:
    """Keep every non-null value, source, period and paging index unchanged.

    Only successful statement envelopes are eligible. A complete snapshot is
    persisted before exposing a projection; I/O failure leaves the old result.
    This is a model-context view, never the grounding ledger or receipt cache.
    """
    if len(result) < 2000:
        return result
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return result
    if (not isinstance(payload, dict) or payload.get("ok") is not True
            or not isinstance(payload.get("data"), dict)
            or not any(isinstance(row, dict) and isinstance(row.get("periods"), list)
                       for row in payload["data"].values())):
        return result
    compact = _without_null_fields(payload)
    reuse = payload.get("_research_reuse")
    original_call_id = reuse.get("original_call_id", call_id) if isinstance(reuse, dict) else call_id
    digest = hashlib.sha256((call_id + "\n" + result).encode()).hexdigest()[:24]
    path = Path(run_dir) / "artifacts" / "research_context" / (digest + ".json")
    compact["_context_view"] = {
        "kind": "non_null_financial_fields",
        "original_call_id": original_call_id,
        "full_result_path": str(path),
        "boundary": "Null object fields omitted; all non-null values and list indexes retained. "
                    "Missing fields may be null, not zero. Use read_file on full_result_path for the complete result. "
                    "Keep original report dates, units and paging; this is not a full filing or new observation.",
    }
    text = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    if len(text) > min(len(result) * 0.9, TOOL_RESULT_LIMIT):
        return result
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result)
        temporary.replace(path)
    except OSError:
        return result
    return text
