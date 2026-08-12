"""Tests for the search_symbol tool.

All HTTP is mocked at the client functions the tool imports
(``eastmoney_client.get_json``, ``yahoo_client.search``,
``sec_edgar_client.cik_for``), so no test ever reaches a live endpoint.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from src.tools import symbol_search_tool as ss


def _eastmoney_payload() -> dict:
    """A suggest payload spanning A-share, HK, and US markets."""
    return {
        "QuotationCodeTable": {
            "Data": [
                {
                    "QuoteID": "1.600519",
                    "Code": "600519",
                    "Name": "贵州茅台",
                    "MktNum": "1",
                    "SecurityTypeName": "沪A",
                },
                {
                    "QuoteID": "116.00700",
                    "Code": "00700",
                    "Name": "腾讯控股",
                    "MktNum": "116",
                    "SecurityTypeName": "港股",
                },
                {
                    "QuoteID": "105.AAPL",
                    "Code": "AAPL",
                    "Name": "苹果",
                    "MktNum": "105",
                    "SecurityTypeName": "美股",
                },
                {
                    # Unmappable market (e.g. a fund/board) -> dropped, not fatal.
                    "QuoteID": "90.BK0001",
                    "Code": "BK0001",
                    "Name": "板块",
                    "MktNum": "90",
                    "SecurityTypeName": "板块",
                },
            ]
        }
    }


def _yahoo_quotes() -> list:
    return [
        {
            "symbol": "AAPL",
            "shortname": "Apple Inc.",
            "exchange": "NMS",
            "quoteType": "EQUITY",
        },
        {
            "symbol": "0700.HK",
            "shortname": "TENCENT",
            "exchange": "HKG",
            "quoteType": "EQUITY",
        },
        {
            "symbol": "BTC-USD",
            "shortname": "Bitcoin USD",
            "exchange": "CCC",
            "quoteType": "CRYPTOCURRENCY",
        },
        {"symbol": "", "shortname": "no symbol"},  # dropped
    ]


class TestSymbolSearchSuccess:
    """Happy-path fan-out, normalization, merge, and CIK enrichment."""

    def test_merges_and_normalizes_across_sources(self):
        with patch.object(
            ss.eastmoney_client, "get_json", return_value=_eastmoney_payload()
        ), patch.object(
            ss.yahoo_client, "search", return_value=_yahoo_quotes()
        ), patch.object(
            ss.sec_edgar_client, "cik_for", return_value="0000320193"
        ):
            out = ss.SymbolSearchTool().execute(query="apple", limit=10)

        payload = json.loads(out)
        assert payload["ok"] is True
        assert payload["market"] == "multi"
        assert payload["source"] == "symbol_search"

        data = payload["data"]
        assert data["query"] == "apple"
        assert data["sources"]["eastmoney"] == "ok"
        assert data["sources"]["yahoo"] == "ok"
        assert data["sources"]["sec_edgar"] == "ok"

        by_symbol = {c["symbol"]: c for c in data["candidates"]}

        # A-share secid -> 600519.SH, market cn.
        assert by_symbol["600519.SH"]["market"] == "cn"
        assert by_symbol["600519.SH"]["name"] == "贵州茅台"

        # HK code zero-padded to 5 digits from both Eastmoney and Yahoo, merged.
        assert "00700.HK" in by_symbol
        assert by_symbol["00700.HK"]["market"] == "hk"
        assert "yahoo" in by_symbol["00700.HK"].get("also_from", [])

        # US equity: Eastmoney + Yahoo merge, SEC CIK attached.
        aapl = by_symbol["AAPL.US"]
        assert aapl["market"] == "us"
        assert aapl["cik"] == "0000320193"
        assert "yahoo" in aapl.get("also_from", [])

        # Crypto keeps its native Yahoo symbol and a global market label.
        assert by_symbol["BTC-USD"]["market"] == "global"

        # Unmappable Eastmoney market dropped; empty Yahoo symbol dropped.
        assert "BK0001" not in by_symbol
        assert data["count"] == len(data["candidates"])

    def test_limit_clamped_and_applied(self):
        with patch.object(
            ss.eastmoney_client, "get_json", return_value=_eastmoney_payload()
        ), patch.object(
            ss.yahoo_client, "search", return_value=_yahoo_quotes()
        ), patch.object(
            ss.sec_edgar_client, "cik_for", return_value=None
        ):
            out = ss.SymbolSearchTool().execute(query="x", limit=2)
        payload = json.loads(out)
        assert payload["data"]["count"] == 2

    def test_no_us_candidate_omits_sec_source(self):
        em = {
            "QuotationCodeTable": {
                "Data": [
                    {
                        "QuoteID": "1.600519",
                        "Code": "600519",
                        "Name": "贵州茅台",
                        "MktNum": "1",
                    }
                ]
            }
        }
        with patch.object(
            ss.eastmoney_client, "get_json", return_value=em
        ), patch.object(
            ss.yahoo_client, "search", return_value=[]
        ), patch.object(
            ss.sec_edgar_client, "cik_for"
        ) as mock_cik:
            out = ss.SymbolSearchTool().execute(query="茅台")
        payload = json.loads(out)
        assert "sec_edgar" not in payload["data"]["sources"]
        mock_cik.assert_not_called()

    def test_yahoo_shanghai_suffix_is_normalized_to_project_convention(self):
        """Yahoo's .SS alias must be locked as the project's canonical .SH."""
        yahoo_quotes = [
            {
                "symbol": "601012.SS",
                "shortname": "LONGI GREEN ENERGY TECHNOLOGY C",
                "exchange": "SHH",
                "quoteType": "EQUITY",
            }
        ]
        with patch.object(
            ss.eastmoney_client,
            "get_json",
            side_effect=RuntimeError("temporary upstream failure"),
        ), patch.object(
            ss.yahoo_client, "search", return_value=yahoo_quotes
        ), patch.object(
            ss.sec_edgar_client, "cik_for"
        ) as mock_cik:
            out = ss.SymbolSearchTool().execute(query="601012")

        payload = json.loads(out)
        candidate = payload["data"]["candidates"][0]
        assert candidate["symbol"] == "601012.SH"
        assert candidate["market"] == "cn"
        assert candidate["exchange"] == "SHH"
        mock_cik.assert_not_called()

    def test_mixed_code_and_name_prefers_exact_code_query(self):
        """Real model calls like '513330 恒生互联网ETF' must resolve by code."""
        yahoo_quotes = [
            {
                "symbol": "513330.SS",
                "shortname": "CHINAAMC HANG SENG INTERNET ETF",
                "exchange": "SHH",
                "quoteType": "ETF",
            }
        ]
        with patch.object(
            ss.eastmoney_client, "get_json", return_value={"QuotationCodeTable": {"Data": []}}
        ) as mock_em, patch.object(
            ss.yahoo_client, "search", return_value=yahoo_quotes
        ) as mock_yahoo, patch.object(
            ss.sec_edgar_client, "cik_for"
        ):
            out = ss.SymbolSearchTool().execute(query="513330 恒生互联网ETF")

        payload = json.loads(out)
        assert payload["data"]["query"] == "513330 恒生互联网ETF"
        assert payload["data"]["resolved_query"] == "513330"
        assert payload["data"]["candidates"][0]["symbol"] == "513330.SH"
        assert mock_em.call_args.kwargs["params"]["input"] == "513330"
        mock_yahoo.assert_called_once_with("513330")

    def test_multiple_code_tokens_do_not_guess_one(self):
        with patch.object(
            ss.eastmoney_client, "get_json", return_value={"QuotationCodeTable": {"Data": []}}
        ) as mock_em, patch.object(
            ss.yahoo_client, "search", return_value=[]
        ) as mock_yahoo:
            ss.SymbolSearchTool().execute(query="比较 513330 和 159605")

        assert mock_em.call_args.kwargs["params"]["input"] == "比较 513330 和 159605"
        mock_yahoo.assert_called_once_with("比较 513330 和 159605")

    def test_project_qualified_code_uses_bare_provider_query(self):
        """Canonical project suffixes must not make provider lookup fail."""
        with patch.object(
            ss.eastmoney_client, "get_json", return_value={"QuotationCodeTable": {"Data": []}}
        ) as mock_em, patch.object(
            ss.yahoo_client, "search", return_value=[]
        ) as mock_yahoo:
            ss.SymbolSearchTool().execute(query="513330.SH")

        assert mock_em.call_args.kwargs["params"]["input"] == "513330"
        mock_yahoo.assert_called_once_with("513330")


class TestSymbolSearchErrors:
    """Error envelopes and per-source resilience."""

    def test_missing_query_returns_error_envelope(self):
        out = ss.SymbolSearchTool().execute(query="   ")
        payload = json.loads(out)
        assert payload["ok"] is False
        assert "required" in payload["error"]

    def test_one_source_failure_does_not_abort_others(self):
        with patch.object(
            ss.eastmoney_client,
            "get_json",
            side_effect=RuntimeError("HTTP 429 banned"),
        ), patch.object(
            ss.yahoo_client, "search", return_value=_yahoo_quotes()
        ), patch.object(
            ss.sec_edgar_client, "cik_for", return_value="0000320193"
        ):
            out = ss.SymbolSearchTool().execute(query="apple")

        payload = json.loads(out)
        # Overall call still succeeds with the surviving source's hits.
        assert payload["ok"] is True
        sources = payload["data"]["sources"]
        assert "eastmoney search failed" in sources["eastmoney"]
        assert "429" in sources["eastmoney"]
        assert sources["yahoo"] == "ok"
        symbols = {c["symbol"] for c in payload["data"]["candidates"]}
        assert "AAPL.US" in symbols

    def test_sec_lookup_failure_recorded_not_fatal(self):
        with patch.object(
            ss.eastmoney_client, "get_json", return_value=_eastmoney_payload()
        ), patch.object(
            ss.yahoo_client, "search", return_value=[]
        ), patch.object(
            ss.sec_edgar_client,
            "cik_for",
            side_effect=RuntimeError("tickers fetch failed"),
        ):
            out = ss.SymbolSearchTool().execute(query="apple")
        payload = json.loads(out)
        assert payload["ok"] is True
        assert "sec lookup failed" in payload["data"]["sources"]["sec_edgar"]
        # The US candidate still appears, just without a CIK.
        aapl = next(c for c in payload["data"]["candidates"] if c["symbol"] == "AAPL.US")
        assert "cik" not in aapl
