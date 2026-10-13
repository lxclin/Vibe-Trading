"""Find actual published Hang Seng Indexes factsheets from public directories."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import requests

from src.agent.tools import BaseTool

BASE = "https://www.hsi.com.hk/api/wsit-hsi-ddoc-ea-public-website-proxy/v1/"
DIRECTORY_RESOURCE = "/public-website/webapp/factsheets/factsheet.json"


def _download_url(resource):
    return BASE + "content/gcs/download?" + urlencode({"resource": resource})


def _rows(url):
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if (not isinstance(payload, dict) or type(payload.get("code")) not in {int, str}
            or str(payload.get("code")) != "0" or not isinstance(payload.get("data"), list)):
        raise ValueError("official directory did not return a successful row list")
    return payload["data"]


def lookup_hsi_resources(query, language="en-hk"):
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 160 or language not in {"en-hk", "zh-hk", "zh-cn"}:
        return json.dumps({"status": "error", "error": "Provide a 1..160 character query and supported language."})
    metadata_url = BASE + "product-data/pub/indexes/metadata/v2?" + urlencode({"data": "allIndexes", "language": language})
    directory_url = _download_url(DIRECTORY_RESOURCE)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            indexes, files = list(pool.map(_rows, [metadata_url, directory_url]))
        key = "fileEng" if language == "en-hk" else "fileChi" if language == "zh-hk" else "fileSchi"
        file_map = {}
        for row in files:
            if not isinstance(row, dict):
                continue
            code, path = row.get("seriesCode"), row.get(key)
            if (isinstance(code, str) and isinstance(path, str)
                    and re.fullmatch(r"/public-website/webapp/factsheets/" + re.escape(language)
                                     + r"/[A-Za-z0-9_-]+\.pdf", path)):
                file_map.setdefault(code.casefold(), set()).add(path)
        needle = query.strip().casefold()
        candidates, exact = [], []
        for row in indexes:
            if not isinstance(row, dict):
                continue
            values = [str(row.get(k) or "") for k in ("indexCode", "indexName", "indexShortName", "indexSeriesCode")]
            if not any(needle in value.casefold() for value in values):
                continue
            paths = file_map.get(str(row.get("indexSeriesCode") or "").casefold(), set())
            item = {"index_code": row.get("indexCode"), "index_name": row.get("indexName"),
                    "series_code": row.get("indexSeriesCode"), "index_type": row.get("indexType"),
                    "factsheet_url": _download_url(next(iter(paths))) if len(paths) == 1 else None,
                    "resource_status": "published" if len(paths) == 1 else "ambiguous" if paths else "unavailable"}
            candidates.append(item)
            if needle in {value.casefold() for value in values}:
                exact.append(item)
        chosen = exact or candidates
        return json.dumps({"status": "ok", "source": "hang_seng_indexes_official_directories",
                           "query": query, "language": language, "metadata_url": metadata_url,
                           "directory_url": directory_url, "unique_match": len(chosen) == 1,
                           "total_matches": len(chosen), "candidates": chosen[:12],
                           "boundary": "Published resources only; no fund-to-index mapping or stock authorization. "
                                       "Do not choose the first ambiguous candidate or substitute a similar index. "
                                       "Read the actual factsheet to verify its identity, as-of date and valuation "
                                       "basis. A current download can contain a historical monthly snapshot; "
                                       "no historical percentile or current valuation is inferred here."}, ensure_ascii=False)
    except (requests.RequestException, ValueError, TypeError) as exc:
        return json.dumps({"status": "error", "source": "hang_seng_indexes_official_directories",
                           "error": str(exc), "boundary": "Directory unavailable; no guessed PDF URL or valuation returned."}, ensure_ascii=False)


class HSIIndexResourcesTool(BaseTool):
    name = "hsi_index_resources"
    description = (
        "Find official Hang Seng Indexes factsheet links using the current published directories, "
        "including JavaScript-only pages. Query the exact index name, series code or numeric index code "
        "(use zh-cn for simplified Chinese names, en-hk for English names/codes) "
        "verified from the fund's own documents; this tool does not identify a fund's benchmark. "
        "Preserve price/total-return variants and ambiguous candidates. Use read_url on the returned "
        "factsheet_url, then verify identity, statistics date and PE methodology before using values. "
        "This makes two public HTTPS directory reads and returns no stock authorization or valuation."
    )
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "description": "Exact verified index name/code, or a term for a shortlist."},
        "language": {"type": "string", "enum": ["en-hk", "zh-hk", "zh-cn"], "default": "en-hk"},
    }, "required": ["query"]}
    repeatable = True

    def execute(self, **kwargs):
        return lookup_hsi_resources(kwargs.get("query"), kwargs.get("language", "en-hk"))
