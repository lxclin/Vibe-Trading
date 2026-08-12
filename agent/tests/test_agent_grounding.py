"""Regression coverage for identity and numeric grounding (#887, #886)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from src.agent.context import ContextBuilder
from src.agent.grounding import GroundingLedger
from src.agent.loop import AgentLoop, _is_tool_success
from src.agent.tools import BaseTool, ToolRegistry
from src.agent.trace import TraceWriter


def _resolver_payload(
    symbol: str = "562500.SS",
    *,
    candidates: list[dict[str, Any]] | None = None,
    query: str = "机器人ETF",
) -> str:
    rows = candidates
    if rows is None:
        rows = [
            {
                "symbol": symbol,
                "name": "机器人ETF",
                "market": "cn",
                "type": "ETF",
                "source": "yahoo",
                "also_from": ["eastmoney"],
            }
        ]
    return json.dumps(
        {
            "ok": True,
            "source": "symbol_search",
            "data": {
                "query": query,
                "count": len(rows),
                "candidates": rows,
                "sources": {"eastmoney": "ok", "yahoo": "ok"},
            },
        },
        ensure_ascii=False,
    )


def _market_payload(symbol: str = "562500.SS") -> str:
    return json.dumps(
        {
            symbol: [
                {
                    "trade_date": "2026-06-23",
                    "open": 1.141,
                    "high": 1.164,
                    "low": 1.121,
                    "close": 1.137,
                    "volume": 123456,
                },
                {
                    "trade_date": "2026-06-24",
                    "open": 1.137,
                    "high": 1.180,
                    "low": 1.110,
                    "close": 1.171,
                    "volume": 234567,
                },
            ],
            "_provenance": {
                symbol: {
                    "source": "yahoo",
                    "requested_source": "auto",
                    "detected_source": "yahoo",
                    "fallback_used": False,
                    "currency_conversion": "none",
                }
            },
        }
    )


class _ResolverTool(BaseTool):
    name = "search_symbol"
    description = "Resolve a company or instrument name."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    repeatable = True

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return self.result


class _MarketTool(BaseTool):
    name = "get_market_data"
    description = "Fetch OHLCV bars."
    parameters = {
        "type": "object",
        "properties": {"codes": {"type": "array", "items": {"type": "string"}}},
        "required": ["codes"],
    }
    repeatable = True

    def __init__(self, result: str) -> None:
        self.result = result
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return self.result


class _PrivateCompanySkillTool(BaseTool):
    name = "load_skill"
    description = "Load a skill."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    repeatable = True

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return json.dumps({"status": "ok", "content": "private company workflow"})


def _tool_call(call_id: str, tool_name: str, **arguments: Any) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, name=tool_name, arguments=arguments)


def _build_direct_agent(
    tmp_path: Path,
    resolver_result: str,
) -> tuple[AgentLoop, _ResolverTool, _MarketTool, _PrivateCompanySkillTool, TraceWriter]:
    resolver = _ResolverTool(resolver_result)
    market = _MarketTool(_market_payload())
    private_skill = _PrivateCompanySkillTool()
    registry = ToolRegistry()
    for tool in (resolver, market, private_skill):
        registry.register(tool)
    agent = AgentLoop(registry=registry, llm=SimpleNamespace(), max_iterations=5)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    agent._grounding = GroundingLedger(
        run_dir=run_dir,
        user_message="分析机器人ETF并给出买入价",
    )
    return agent, resolver, market, private_skill, TraceWriter(run_dir)


def test_ok_false_tool_envelope_is_failure() -> None:
    """Business failures must not be recorded as successful tool calls (#886)."""
    assert _is_tool_success('{"ok": false, "error": "upstream failed"}') is False
    assert _is_tool_success('{"success": false, "message": "denied"}') is False
    assert _is_tool_success('{"status": "failed"}') is False
    assert _is_tool_success('{"ok": true, "data": {}}') is True


def test_resolver_and_consumer_in_same_batch_cannot_race(
    tmp_path: Path,
) -> None:
    """A consumer sees the identity snapshot from before its whole LLM batch."""
    agent, resolver, market, _, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload(),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [
            _tool_call("resolve", "search_symbol", query="机器人ETF"),
            _tool_call(
                "prices-too-early",
                "get_market_data",
                codes=["562500.SH"],
                start_date="2026-06-23",
                end_date="2026-06-24",
            ),
        ],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )

    assert resolver.calls == 1
    assert market.calls == 0
    assert agent._grounding.authorized_symbols == {"562500.SS"}
    blocked = [json.loads(message["content"]) for message in messages]
    assert any(item.get("error_code") == "identity_required" for item in blocked)

    agent._process_tool_calls(
        [
            _tool_call(
                "prices-after-lock",
                "get_market_data",
                codes=["562500.SS"],
                start_date="2026-06-23",
                end_date="2026-06-24",
                source="auto",
            )
        ],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert market.calls == 1
    artifact = json.loads(
        (tmp_path / "run" / "artifacts" / "grounding_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["identity"]["status"] == "locked"
    assert any(
        record["field"] == "close"
        and record["value"] == 1.137
        and record["source"] == "yahoo"
        and record["currency"] == "CNY"
        and record["currency_conversion"] == "none"
        for record in artifact["evidence"]
    )


def test_unresolved_single_instrument_blocks_whole_market_screener(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="513330目前是否值得买入")
    decision = ledger.authorize_tool_call(
        "screen_market",
        {"market": "hk", "sort_by": "change_pct"},
        batch_authorized_symbols=set(),
        batch_identity_status=ledger.identity_status,
        call_id="irrelevant-screen",
    )
    assert decision.allowed is False
    assert decision.error_code == "identity_required"


def test_market_screening_request_still_allows_whole_market_screener(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="筛选港股涨幅最大的股票")
    decision = ledger.authorize_tool_call(
        "screen_market",
        {"market": "hk", "sort_by": "change_pct"},
        batch_authorized_symbols=set(),
        batch_identity_status=ledger.identity_status,
        call_id="requested-screen",
    )
    assert decision.allowed is True


def test_market_sensitive_skill_waits_for_prior_identity_batch(
    tmp_path: Path,
) -> None:
    """Workflow selection cannot race the resolver in the same assistant turn."""
    agent, resolver, _, skill, trace = _build_direct_agent(tmp_path, _resolver_payload())
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [
            _tool_call("resolve", "search_symbol", query="机器人ETF"),
            _tool_call("workflow-too-early", "load_skill", name="valuation-model"),
        ],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )

    assert resolver.calls == 1
    assert skill.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_required"

    agent._process_tool_calls(
        [_tool_call("workflow-after-lock", "load_skill", name="valuation-model")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert skill.calls == 1


def test_bare_us_ticker_is_allowed_only_for_explicit_transport_contract(
    tmp_path: Path,
) -> None:
    """SEC's bare ticker consumes AAPL.US without permitting venue aliases."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 AAPL.US 的价格",
    )

    sec = ledger.authorize_tool_call(
        "get_sec_filings",
        {"ticker": "AAPL"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="sec",
    )
    market_alias = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["AAPL"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="prices",
    )

    assert sec.allowed is True
    assert market_alias.allowed is False
    assert market_alias.error_code == "identity_mismatch"


def test_stale_history_identity_does_not_unlock_new_subject(tmp_path: Path) -> None:
    """A previous turn's AAPL identity cannot authorize a SpaceX price request."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="SpaceX 在什么价格买入比较合适？",
        history=[{"role": "user", "content": "请分析 AAPL.US"}],
    )

    authorization = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["AAPL.US"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="stale-price",
    )

    assert ledger.authorized_symbols == set()
    assert authorization.allowed is False
    assert authorization.error_code == "identity_required"


def test_example_crypto_pair_is_not_treated_as_selected_identity(tmp_path: Path) -> None:
    """An example in a prompt must not seed a token identity."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请确认 SKHY 的交易对和交易所，例如 SKHY-USDT。",
    )

    assert ledger.authorized_symbols == set()
    assert ledger.identity_status == "not_required"


def test_negated_symbol_is_not_treated_as_selected_identity(tmp_path: Path) -> None:
    """A symbol named only to forbid its use must not authorize stock tools."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message=(
            "请确认 SKHY 的真实加密交易对；不要使用 SKHY.US 的股票价格代替。"
        ),
    )

    assert ledger.authorized_symbols == set()


def test_crypto_request_rejects_listed_security_resolution(tmp_path: Path) -> None:
    """A stock resolver result cannot silently answer a crypto request."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 SKHY 这只虚拟货币的交易对和现货价格。",
    )
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": "SKHY"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="crypto-resolution",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "SKHY"},
        result=_resolver_payload(
            candidates=[
                {
                    "symbol": "SKHY.US",
                    "name": "SK hynix Inc.",
                    "market": "us",
                    "type": "equity",
                    "exchange": "NMS",
                    "source": "yahoo",
                }
            ],
            query="SKHY",
        ),
        call_id="crypto-resolution",
        success=True,
    )

    assert ledger.identity_status == "conflicting"
    assert ledger.authorized_symbols == set()

    safe = ledger.validate_final_answer(
        "无法核验唯一可靠的 SKHY 加密交易对；聚合报价 $144–164 彼此冲突，"
        "不能视为可靠行情。请提供你看到它的平台或交易所。"
    )
    unsafe = ledger.validate_final_answer(
        "无法确认唯一交易对，但仍建议买入，买入价 10 USD。"
    )

    assert safe.valid is True
    assert unsafe.valid is False
    assert any(issue["code"] == "identity_not_locked" for issue in unsafe.issues)
    fallback = ledger.safe_fallback()
    assert "SKHY.US" in fallback
    assert "具体平台" in fallback


def test_explicit_crypto_pair_survives_stock_resolver_failure(tmp_path: Path) -> None:
    """A Yahoo outage must not invalidate a pair explicitly supplied by the user."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="我指 Binance 的 SKHYB/USDT，请分析它是否值得买入。",
    )
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": "SKHY"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="stock-resolver",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "SKHY"},
        result=json.dumps(
            {
                "ok": False,
                "error": "Yahoo does not resolve this crypto pair",
            }
        ),
        call_id="stock-resolver",
        success=False,
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"SKHYB/USDT"}
    assert ledger.should_request_user_confirmation is False


def test_unrelated_short_crypto_query_cannot_inherit_explicit_pair(tmp_path: Path) -> None:
    """A failed lookup for a short substring must not shadow the active pair."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="我指 Binance 的 SKHYB/USDT，请分析它是否值得买入。",
    )
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": "HY"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="unrelated-resolver",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "HY"},
        result=json.dumps({"ok": False, "error": "not found"}),
        call_id="unrelated-resolver",
        success=False,
    )

    assert ledger.authorized_symbols == {"SKHYB/USDT"}
    assert ledger.identity_status == "locked"
    assert ledger.should_request_user_confirmation is False


def test_crypto_ambiguity_requests_user_confirmation(tmp_path: Path) -> None:
    """An unresolved crypto name becomes a concise clarification question."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="SKHY 加密货币，现在值得买吗？",
    )
    candidates = [
        {
            "symbol": "SKHYB/USDT",
            "name": "SK Hynix Tokenized bStocks",
            "exchange": "Binance",
            "type": "cryptocurrency",
        },
        {
            "symbol": "SKHY/USDC",
            "name": "SK Hynix Backpack Securities",
            "exchange": "Meteora",
            "type": "cryptocurrency",
        },
    ]
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": "SKHY"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="crypto-resolver",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "SKHY"},
        result=_resolver_payload(candidates=candidates, query="SKHY"),
        call_id="crypto-resolver",
        success=True,
    )

    assert ledger.should_request_user_confirmation is True
    prompt = ledger.clarification_prompt()
    assert "SKHYB/USDT" in prompt
    assert "SKHY/USDC" in prompt
    assert "平台" in prompt


def test_meta_delivery_without_report_is_rejected(tmp_path: Path) -> None:
    """The agent may not claim a hidden report was delivered."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请确认 SKHY 的交易对和交易所。",
    )
    result = ledger.validate_final_answer(
        "报告已交付完毕（上方完整版）。当前没有新的问题需要处理。"
    )
    assert result.valid is False
    assert any(issue["code"] == "meta_delivery_without_report" for issue in result.issues)


def test_elliptical_followup_inherits_nearest_explicit_user_symbol(
    tmp_path: Path,
) -> None:
    """Legacy sessions keep the active subject across short follow-up turns."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="同业横向对比",
        history=[
            {
                "role": "user",
                "content": "分析A股山东黄金（600547.SH）现在是否适合买入",
            },
            {"role": "assistant", "content": "已完成山东黄金分析。"},
            {"role": "user", "content": "值得买入吗"},
            {"role": "assistant", "content": "结论见上。"},
        ],
    )

    authorization = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["600547.SH"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="peer-followup-price",
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"600547.SH"}
    assert ledger.inherited_symbols == {"600547.SH"}
    assert authorization.allowed is True


def test_elliptical_followup_prefers_structured_locked_identity(
    tmp_path: Path,
) -> None:
    """New sessions inherit audited identity metadata instead of assistant prose."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="继续",
        history=[
            {"role": "user", "content": "分析这家公司"},
            {
                "role": "assistant",
                "content": "已完成。",
                "grounding_identity": {
                    "status": "locked",
                    "authorized_symbols": ["600547.SH"],
                    "records": [
                        {
                            "status": "locked",
                            "symbol": "600547.SH",
                            "venue": "shanghai",
                            "instrument_type": "listed_security",
                            "currency": "CNY",
                            "source": ["yahoo"],
                        }
                    ],
                },
            },
        ],
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"600547.SH"}
    assert ledger.inherited_symbols == {"600547.SH"}


def test_trade_management_followup_inherits_active_identity(
    tmp_path: Path,
) -> None:
    """A fill-price follow-up keeps the preceding instrument in context."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="我刚刚142买入了100usdt，后面该做什么",
        history=[
            {"role": "user", "content": "SKHY.US"},
            {
                "role": "assistant",
                "content": "已完成分析。",
                "grounding_identity": {
                    "status": "locked",
                    "authorized_symbols": ["SKHY.US"],
                    "records": [
                        {
                            "status": "locked",
                            "symbol": "SKHY.US",
                            "venue": "NMS",
                            "instrument_type": "listed_security",
                            "currency": "USD",
                        }
                    ],
                },
            },
        ],
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"SKHY.US"}
    assert ledger.inherited_symbols == {"SKHY.US"}


def test_trade_management_followup_recovers_after_failed_identity_retry(
    tmp_path: Path,
) -> None:
    """A transient resolver failure must not poison the active position context."""
    locked = {
        "status": "locked",
        "authorized_symbols": ["SKHY.US"],
        "records": [{"status": "locked", "symbol": "SKHY.US"}],
    }
    failed = {
        "status": "invalidated",
        "authorized_symbols": [],
        "records": [{"query": "SKHY.US", "status": "invalidated"}],
    }
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="我刚刚142买入了100usdt，后面该做什么",
        history=[
            {"role": "user", "content": "SKHY.US"},
            {"role": "assistant", "content": "已完成。", "grounding_identity": locked},
            {"role": "user", "content": "我刚刚142买入了100usdt，后面该做什么"},
            {"role": "assistant", "content": "请确认标的。", "grounding_identity": failed},
        ],
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"SKHY.US"}
    assert ledger.inherited_symbols == {"SKHY.US"}


def test_continue_recovers_primary_code_from_legacy_ambiguous_summary(
    tmp_path: Path,
) -> None:
    """A legacy peer-contaminated summary must keep the preceding code usable."""
    legacy_summary = {
        "status": "ambiguous",
        "authorized_symbols": ["00700.HK", "513330.SH", "BABA.US"],
        "records": [
            {"status": "locked", "symbol": "513330.SH", "currency": "CNY"},
            {"status": "locked", "symbol": "00700.HK", "currency": "HKD"},
            {"status": "locked", "symbol": "BABA.US", "currency": "USD"},
            {
                "status": "ambiguous",
                "query": "Meituan",
                "candidates": [{"symbol": "03690.HK"}, {"symbol": "HMTD.SI"}],
            },
        ],
    }
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="继续",
        history=[
            {"role": "user", "content": "513330目前是否值得买入"},
            {"role": "assistant", "content": "上一轮报告", "grounding_identity": legacy_summary},
            {"role": "user", "content": "继续"},
        ],
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"513330.SH"}
    assert ledger.primary_symbols == {"513330.SH"}
    assert ledger.inherited_symbols == {"513330.SH"}


def test_continue_skips_same_subject_failed_retry_before_legacy_summary(
    tmp_path: Path,
) -> None:
    """An empty failed retry must not hide the older primary lock."""
    old = {
        "status": "ambiguous",
        "authorized_symbols": ["513330.SH", "00700.HK"],
        "records": [
            {"status": "locked", "symbol": "513330.SH"},
            {"status": "locked", "symbol": "00700.HK"},
        ],
    }
    failed = {
        "status": "invalidated",
        "authorized_symbols": [],
        "records": [{"query": "513330 恒生互联网ETF", "status": "invalidated"}],
    }
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="继续",
        history=[
            {"role": "user", "content": "513330目前是否值得买入"},
            {"role": "assistant", "content": "上一轮报告", "grounding_identity": old},
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "请确认标的", "grounding_identity": failed},
            {"role": "user", "content": "继续"},
        ],
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"513330.SH"}


def test_continue_inherits_only_primary_symbol_from_new_summary(tmp_path: Path) -> None:
    """Locked comparison peers must not become primary on the next turn."""
    summary = {
        "status": "locked",
        "authorized_symbols": ["00700.HK", "513330.SH", "BABA.US"],
        "primary_symbols": ["513330.SH"],
        "records": [
            {"status": "locked", "symbol": "513330.SH", "currency": "CNY"},
            {"status": "locked", "symbol": "00700.HK", "currency": "HKD"},
            {"status": "locked", "symbol": "BABA.US", "currency": "USD"},
        ],
    }
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="继续",
        history=[
            {"role": "user", "content": "513330目前是否值得买入"},
            {"role": "assistant", "content": "完整报告", "grounding_identity": summary},
        ],
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"513330.SH"}
    assert ledger.primary_symbols == {"513330.SH"}


def test_repeated_resolver_failure_does_not_erase_locked_primary(tmp_path: Path) -> None:
    """A transient retry failure cannot replace an already verified lock."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="513330是否值得买入")
    locked_payload = _resolver_payload(
        symbol="513330.SH",
        query="513330",
        candidates=[{"symbol": "513330.SH", "type": "ETF", "source": "yahoo"}],
    )
    ledger.authorize_tool_call(
        "search_symbol", {"query": "513330"},
        batch_authorized_symbols=set(), call_id="first",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "513330"},
        result=locked_payload, call_id="first", success=True,
    )
    ledger.authorize_tool_call(
        "search_symbol", {"query": "513330"},
        batch_authorized_symbols=ledger.authorized_symbols, call_id="retry",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "513330"},
        result='{"ok": false, "error": "HTTP 429"}', call_id="retry", success=False,
    )

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"513330.SH"}


def test_repeated_resolver_concrete_change_conflicts_with_locked_primary(
    tmp_path: Path,
) -> None:
    """Preserving a lock must not hide a concrete conflicting re-resolution."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="ABC是否值得买入")
    first = _resolver_payload(
        symbol="ABC.US", query="ABC",
        candidates=[{"symbol": "ABC.US", "source": "yahoo"}],
    )
    changed = _resolver_payload(
        symbol="ABC.HK", query="ABC",
        candidates=[{"symbol": "ABC.HK", "source": "yahoo"}],
    )
    for call_id, payload in (("first", first), ("changed", changed)):
        ledger.authorize_tool_call(
            "search_symbol", {"query": "ABC"},
            batch_authorized_symbols=ledger.authorized_symbols, call_id=call_id,
        )
        ledger.ingest_tool_result(
            tool_name="search_symbol", arguments={"query": "ABC"},
            result=payload, call_id=call_id, success=True,
        )

    assert ledger.identity_status == "conflicting"


def test_nearest_nonlocked_identity_does_not_resurrect_older_subject(
    tmp_path: Path,
) -> None:
    """An unresolved subject change blocks inheritance from an older audited turn."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="继续",
        history=[
            {
                "role": "assistant",
                "content": "AAPL analysis",
                "grounding_identity": {
                    "status": "locked",
                    "authorized_symbols": ["AAPL.US"],
                    "records": [{"status": "locked", "symbol": "AAPL.US"}],
                },
            },
            {"role": "user", "content": "分析 SpaceX"},
            {
                "role": "assistant",
                "content": "Could not resolve SpaceX",
                "grounding_identity": {
                    "status": "not_found",
                    "authorized_symbols": [],
                    "records": [],
                },
            },
        ],
    )

    assert ledger.identity_status == "not_required"
    assert ledger.authorized_symbols == set()
    assert ledger.inherited_symbols == set()


def test_single_clean_not_found_source_is_not_enough_for_private_routing(
    tmp_path: Path,
) -> None:
    """A partial resolver outage cannot turn a public entity into private research."""
    resolver_result = json.dumps(
        {
            "ok": True,
            "source": "symbol_search",
            "data": {
                "query": "Acme",
                "count": 0,
                "candidates": [],
                "sources": {
                    "eastmoney": "ok",
                    "yahoo": "HTTP 429",
                },
            },
        }
    )
    agent, _, _, private_skill, trace = _build_direct_agent(tmp_path, resolver_result)
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="Acme")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("private", "load_skill", name="private-company-research")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert agent._grounding.identity_status == "invalidated"
    assert private_skill.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_required"


def test_explicit_symbol_and_resolver_suffix_alias_become_conflicting(
    tmp_path: Path,
) -> None:
    """A later .SH resolver result cannot silently replace explicit .SS identity."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 562500.SS 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "机器人ETF"},
        result=_resolver_payload("562500.SH"),
        call_id="resolver-alias",
        success=True,
    )

    authorization = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["562500.SS"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        batch_identity_status=ledger.identity_status,
        call_id="prices",
    )

    assert ledger.identity_status == "conflicting"
    assert ledger.should_request_user_confirmation is True
    assert authorization.allowed is False
    assert authorization.error_code == "identity_conflict"


def test_locked_symbol_rejects_silent_exchange_suffix_rewrite(
    tmp_path: Path,
) -> None:
    """The consumer may not silently turn the resolver's .SS into .SH."""
    agent, _, market, _, trace = _build_direct_agent(tmp_path, _resolver_payload())
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="机器人ETF")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("wrong-suffix", "get_market_data", codes=["562500.SH"])],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert market.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_mismatch"


def test_listed_identity_blocks_private_company_workflow(
    tmp_path: Path,
) -> None:
    """Model memory cannot relabel a strongly resolved listing as private."""
    candidates = [
        {
            "symbol": "SPCX.US",
            "name": "SpaceX",
            "market": "us",
            "type": "equity",
            "exchange": "NMS",
            "source": "eastmoney",
            "also_from": ["yahoo"],
            "cik": "0001181412",
        }
    ]
    agent, _, _, private_skill, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload("SPCX.US", candidates=candidates, query="SpaceX"),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="SpaceX")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("private", "load_skill", name="private-company-research")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert private_skill.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_conflict"
    validation = agent._grounding.validate_final_answer(
        "SpaceX is a private company and is not publicly traded."
    )
    assert validation.valid is False
    assert any(issue["code"] == "listed_identity_relabelled_private" for issue in validation.issues)


def test_not_found_identity_allows_private_company_workflow(
    tmp_path: Path,
) -> None:
    """A clean multi-source not-found result keeps genuine private research usable."""
    agent, _, _, private_skill, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload(candidates=[], query="Acme Private Labs"),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="Acme Private Labs")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("private", "load_skill", name="private-company-research")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert private_skill.calls == 1
    assert agent._grounding.identity_status == "not_found"


def test_ambiguous_resolution_keeps_consumers_blocked(tmp_path: Path) -> None:
    """Multiple weak candidates must become first-class ambiguous state."""
    candidates = [
        {"symbol": "ABC.US", "name": "ABC Holdings", "source": "yahoo"},
        {"symbol": "ABC.HK", "name": "ABC Group", "source": "eastmoney"},
    ]
    agent, _, market, _, trace = _build_direct_agent(
        tmp_path,
        _resolver_payload(candidates=candidates, query="ABC"),
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    agent._process_tool_calls(
        [_tool_call("resolve", "search_symbol", query="ABC")],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        1,
    )
    agent._process_tool_calls(
        [_tool_call("prices", "get_market_data", codes=["ABC.US"])],
        ContextBuilder,
        messages,
        trace,
        react_trace,
        2,
    )
    trace.close()

    assert agent._grounding.identity_status == "ambiguous"
    assert market.calls == 0
    assert json.loads(messages[-1]["content"])["error_code"] == "identity_conflict"


def test_auxiliary_ambiguous_peer_does_not_block_primary_instrument(tmp_path: Path) -> None:
    """Peer searches must not turn a locked user target into a false refusal."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="513330目前是否值得买入",
    )
    primary = _resolver_payload(
        symbol="513330.SH",
        query="513330",
        candidates=[
            {
                "symbol": "513330.SH",
                "name": "China Asset Management ETF",
                "market": "cn",
                "type": "ETF",
                "source": "yahoo",
            }
        ],
    )
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": "513330"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="primary",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "513330"},
        result=primary,
        call_id="primary",
        success=True,
    )
    assert ledger.identity_status == "locked"
    assert ledger.primary_symbols == {"513330.SH"}

    peer_payload = _resolver_payload(
        query="Meituan",
        candidates=[
            {"symbol": "03690.HK", "name": "MEITUAN-W", "source": "yahoo"},
            {"symbol": "HMTD.SI", "name": "Meituan SDR", "source": "yahoo"},
            {"symbol": "9MDA.F", "name": "Meituan R", "source": "yahoo"},
        ],
    )
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": "Meituan"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="peer",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": "Meituan"},
        result=peer_payload,
        call_id="peer",
        success=True,
    )

    assert ledger.identity_status == "locked"
    assert ledger.should_request_user_confirmation is False
    assert any(
        record.query == "Meituan" and record.status == "ambiguous"
        for record in ledger._identities.values()
    )


def test_513330_full_conversation_matrix_survives_peers_and_retries(
    tmp_path: Path,
) -> None:
    """Regression for the observed first-run -> continue -> retry failure chain."""
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first = GroundingLedger(run_dir=first_dir, user_message="513330目前是否值得买入")
    primary = _resolver_payload(
        symbol="513330.SH",
        query="513330",
        candidates=[{"symbol": "513330.SH", "type": "ETF", "source": "yahoo"}],
    )
    first.authorize_tool_call(
        "search_symbol", {"query": "513330"},
        batch_authorized_symbols=set(), call_id="primary",
    )
    first.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "513330"},
        result=primary, call_id="primary", success=True,
    )
    for call_id, query, candidates in (
        ("peer-lock", "Tencent", [{"symbol": "00700.HK", "source": "yahoo"}]),
        (
            "peer-ambiguous",
            "Meituan",
            [{"symbol": "03690.HK"}, {"symbol": "HMTD.SI"}],
        ),
    ):
        first.authorize_tool_call(
            "search_symbol", {"query": query},
            batch_authorized_symbols=first.authorized_symbols, call_id=call_id,
        )
        first.ingest_tool_result(
            tool_name="search_symbol", arguments={"query": query},
            result=_resolver_payload(query=query, candidates=candidates),
            call_id=call_id, success=True,
        )
    first.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["513330.SH"], "source": "auto"},
        result=json.dumps(
            {
                "513330.SH": [{"date": "2026-08-12", "close": 0.396}],
                "_provenance": {
                    "513330.SH": {"source": "tencent", "currency_conversion": "none"}
                },
            }
        ),
        call_id="prices",
        success=True,
    )
    assert first.identity_status == "locked"
    assert first.validate_final_answer(
        "513330.SH 当前价格为 0.396 CNY（source: tencent）。"
    ).valid

    first_summary = first.identity_summary()
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = GroundingLedger(
        run_dir=second_dir,
        user_message="继续",
        history=[
            {"role": "user", "content": "513330目前是否值得买入"},
            {"role": "assistant", "content": "完整报告", "grounding_identity": first_summary},
        ],
    )
    assert second.authorized_symbols == {"513330.SH"}
    second.authorize_tool_call(
        "search_symbol", {"query": "513330.SH"},
        batch_authorized_symbols=second.authorized_symbols, call_id="failed-retry",
    )
    second.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "513330.SH"},
        result='{"ok": false, "error": "temporary outage"}',
        call_id="failed-retry", success=False,
    )
    assert second.identity_status == "locked"

    third_dir = tmp_path / "third"
    third_dir.mkdir()
    third = GroundingLedger(
        run_dir=third_dir,
        user_message="继续",
        history=[
            {"role": "user", "content": "513330目前是否值得买入"},
            {"role": "assistant", "content": "完整报告", "grounding_identity": first_summary},
            {"role": "user", "content": "继续"},
            {"role": "assistant", "content": "继续报告", "grounding_identity": second.identity_summary()},
        ],
    )
    assert third.identity_status == "locked"
    assert third.authorized_symbols == {"513330.SH"}


def test_china_etf_code_band_overrides_provider_equity_mislabel(tmp_path: Path) -> None:
    """Yahoo sometimes labels Chinese ETFs as EQUITY; exchange code bands are definitive."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="513330是否值得买入")
    payload = _resolver_payload(
        symbol="513330.SH",
        query="513330",
        candidates=[{"symbol": "513330.SH", "type": "equity", "source": "yahoo"}],
    )
    ledger.authorize_tool_call(
        "search_symbol", {"query": "513330"},
        batch_authorized_symbols=set(), call_id="resolver",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "513330"},
        result=payload, call_id="resolver", success=True,
    )

    assert ledger.identity_summary()["records"][0]["instrument_type"] == "fund"


def test_bare_primary_code_must_resolve_before_peer_lookup(tmp_path: Path) -> None:
    """A constituent lookup must never become the primary ETF identity."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="513330目前是否值得买入")

    peer = ledger.authorize_tool_call(
        "search_symbol", {"query": "腾讯控股 Tencent"},
        batch_authorized_symbols=set(), call_id="peer",
    )
    assert not peer.allowed
    assert peer.error_code == "primary_identity_required"

    primary = ledger.authorize_tool_call(
        "search_symbol", {"query": "513330 恒生互联网ETF"},
        batch_authorized_symbols=set(), call_id="primary",
    )
    assert primary.allowed


def test_name_resolution_prefers_exact_company_name_over_unrelated_cik(tmp_path: Path) -> None:
    """A SEC-enriched fuzzy hit must not beat an exact provider company name."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="腾讯控股目前值得买吗")
    payload = _resolver_payload(
        query="腾讯控股 Tencent",
        candidates=[
            {"symbol": "00700.HK", "name": "TENCENT", "type": "equity"},
            {"symbol": "TCEHY.US", "name": "Tencent Holding Ltd.", "type": "equity"},
            {
                "symbol": "TME.US",
                "name": "Tencent Music Entertainment Group",
                "type": "equity",
                "cik": "0001744676",
            },
        ],
    )
    ledger.authorize_tool_call(
        "search_symbol", {"query": "腾讯控股 Tencent"},
        batch_authorized_symbols=set(), call_id="resolver",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "腾讯控股 Tencent"},
        result=payload, call_id="resolver", success=True,
    )

    assert ledger.primary_symbols == {"00700.HK"}


def test_bare_hk_code_never_locks_same_number_taiwan_product(tmp_path: Path) -> None:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="00700目前值得买吗")
    payload = _resolver_payload(
        query="00700",
        candidates=[{"symbol": "00700.TW", "name": "Unrelated Taiwan product"}],
    )
    ledger.authorize_tool_call(
        "search_symbol", {"query": "00700"},
        batch_authorized_symbols=set(), call_id="resolver",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol", arguments={"query": "00700"},
        result=payload, call_id="resolver", success=True,
    )

    assert ledger.primary_symbols == set()
    assert ledger.identity_status == "ambiguous"


def test_final_numeric_gate_rejects_known_trace_contradiction(tmp_path: Path) -> None:
    """Known 1.11-1.18 evidence cannot become 0.88-0.91 in the answer."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 562500.SS 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={
            "codes": ["562500.SS"],
            "source": "auto",
        },
        result=_market_payload(),
        call_id="prices",
        success=True,
    )

    bad = ledger.validate_final_answer(
        """| 日期 | 开盘 | 最高 | 最低 | 收盘 |
|---|---:|---:|---:|---:|
| 2026-06-23 | 0.895 | 0.907 | 0.892 | 0.903 |

建议重仓买入价为 0.881。"""
    )
    good = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）在 2026-06-23 的已观测开盘价为 1.141，收盘价为 1.137。"
    )

    assert bad.valid is False
    assert any(issue["code"] == "numeric_claim_conflict" for issue in bad.issues)
    assert good.valid is True


def test_numeric_gate_validates_derived_formula_and_provenance(tmp_path: Path) -> None:
    """A derived entry level must calculate correctly from observed evidence."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 562500.SS 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["562500.SS"], "source": "auto"},
        result=_market_payload(),
        call_id="prices",
        success=True,
    )

    bad_math = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）的推导买入价：(1.141 + 1.137) / 2 = 0.881。"
    )
    no_observed_input = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）的推导买入价：(0.88 + 0.90) / 2 = 0.89。"
    )
    good = ledger.validate_final_answer(
        "562500.SS（Yahoo，CNY）的推导买入价：(1.141 + 1.137) / 2 = 1.139。"
    )
    missing_provenance = ledger.validate_final_answer(
        "2026-06-23 的已观测收盘价为 1.137。"
    )

    assert bad_math.valid is False
    assert no_observed_input.valid is False
    assert good.valid is True
    assert missing_provenance.valid is False
    assert {
        issue["code"] for issue in missing_provenance.issues
    } >= {
        "canonical_symbol_not_surfaced",
        "data_source_not_surfaced",
        "currency_not_surfaced",
    }


def test_numeric_gate_ignores_month_day_dates_in_price_prose(tmp_path: Path) -> None:
    """A month/day such as 8/6 must not be compared with OHLC evidence."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 SKHY.US 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["SKHY.US"], "source": "auto"},
        result=_market_payload("SKHY.US"),
        call_id="prices",
        success=True,
    )

    draft = (
        "SKHY.US（Yahoo，USD）截至 8/6 的已观测 OHLC 范围为 "
        "1.110–1.180 USD；8/6 今晚开盘前不应把日期当作价格。"
    )

    assert ledger.validate_final_answer(draft).valid is True


def test_numeric_gate_allows_explicitly_unverified_aggregate_quotes(tmp_path: Path) -> None:
    """Conflicting aggregate quotes may be disclosed as caveats, not facts."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="请分析 SKHY.US 并给出买入价",
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["SKHY.US"], "source": "auto"},
        result=_market_payload("SKHY.US"),
        call_id="prices",
        success=True,
    )

    draft = (
        "SKHY.US（Yahoo，USD）的已观测收盘价为 1.137 USD。\n"
        "- 3. 聚合报价 $144–164 彼此冲突，不可作为可靠行情。"
    )

    assert ledger.validate_final_answer(draft).valid is True


def test_numeric_gate_allows_dated_prior_close_when_live_quote_is_missing(
    tmp_path: Path,
) -> None:
    """A dated prior close must not erase a useful follow-up report."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="我刚刚142买入了100usdt的SKHYB/USDT，后面该做什么",
    )

    draft = (
        "实时接口不可用；可参考 ADR 8/5 收盘 $151.03，但这不是当前代币报价。"
        "请以 Binance SKHYB/USDT 页面为准。"
    )

    assert ledger.validate_final_answer(draft).valid is True


def test_safe_fallback_names_locked_symbol_when_live_evidence_is_missing(
    tmp_path: Path,
) -> None:
    """Missing quotes after identity lock must not regress to identity refusal."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="我刚刚142买入了100usdt的SKHYB/USDT，后面该做什么",
    )

    fallback = ledger.safe_fallback()

    assert "SKHYB/USDT" in fallback
    assert "已确认" in fallback
    assert "无法确认唯一" not in fallback


def test_safe_fallback_reports_primary_not_comparison_peer(tmp_path: Path) -> None:
    """A rejected primary report must not dump every peer's OHLC range."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="513330.SH是否值得买入")
    payload = json.dumps(
        {
            "513330.SH": [
                {"date": "2026-08-12", "open": 0.40, "high": 0.41, "low": 0.39, "close": 0.396}
            ],
            "00700.HK": [
                {"date": "2026-08-12", "open": 600.0, "high": 610.0, "low": 590.0, "close": 605.0}
            ],
            "_provenance": {
                "513330.SH": {"source": "tencent", "currency_conversion": "none"},
                "00700.HK": {"source": "yahoo", "currency_conversion": "none"},
            },
        }
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["513330.SH", "00700.HK"], "source": "auto"},
        result=payload,
        call_id="prices",
        success=True,
    )

    fallback = ledger.safe_fallback()
    assert "513330.SH" in fallback
    assert "00700.HK" not in fallback


def test_unlabelled_price_claim_defaults_to_single_primary_not_peer(tmp_path: Path) -> None:
    """Peer evidence must not make a primary price sentence ambiguous."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="513330.SH是否值得买入")
    payload = json.dumps(
        {
            "513330.SH": [{"date": "2026-08-12", "close": 0.396}],
            "00700.HK": [{"date": "2026-08-12", "close": 605.0}],
            "_provenance": {
                "513330.SH": {"source": "tencent", "currency_conversion": "none"},
                "00700.HK": {"source": "yahoo", "currency_conversion": "none"},
            },
        }
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["513330.SH", "00700.HK"], "source": "auto"},
        result=payload,
        call_id="prices",
        success=True,
    )

    result = ledger.validate_final_answer(
        "513330.SH 当前价格为 0.396 CNY（source: tencent）。"
    )
    assert result.valid is True


def test_safe_fallback_inherits_language_from_conversation_history(tmp_path: Path) -> None:
    """A terse symbol follow-up should not switch a Chinese session to English."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="SKHY.US",
        history=[{"role": "user", "content": "SKHY 加密货币，现在值得买吗？"}],
    )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["SKHY.US"], "source": "auto"},
        result=_market_payload("SKHY.US"),
        call_id="prices",
        success=True,
    )

    fallback = ledger.safe_fallback()

    assert "当前已核验到" in fallback
    assert "rejected the previous draft" not in fallback


class _Response:
    def __init__(
        self,
        *,
        content: str = "",
        tool_calls: list[SimpleNamespace] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None
        self.has_tool_calls = bool(self.tool_calls)


class _CorrectingLLM:
    model_name = "grounding-test"

    def __init__(self) -> None:
        self.responses = [
            _Response(
                tool_calls=[
                    _tool_call("resolve", "search_symbol", query="机器人ETF"),
                    _tool_call(
                        "too-early",
                        "get_market_data",
                        codes=["562500.SH"],
                        start_date="2026-06-23",
                        end_date="2026-06-24",
                    ),
                ]
            ),
            _Response(
                tool_calls=[
                    _tool_call(
                        "prices",
                        "get_market_data",
                        codes=["562500.SS"],
                        start_date="2026-06-23",
                        end_date="2026-06-24",
                        source="auto",
                    )
                ]
            ),
            _Response(content="建议买入价为 0.881。"),
            _Response(
                content=(
                    "562500.SS（Yahoo，CNY）在 2026-06-23 的已观测收盘价为 1.137。"
                )
            ),
        ]

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
        on_reasoning_chunk: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> _Response:
        response = self.responses.pop(0)
        if response.content and on_text_chunk:
            on_text_chunk(response.content)
        return response

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Response:
        return _Response()


def test_agent_loop_rejects_then_corrects_ungrounded_final_answer(
    tmp_path: Path,
) -> None:
    """Rejected numeric drafts never become the returned or streamed answer."""
    resolver = _ResolverTool(_resolver_payload())
    market = _MarketTool(_market_payload())
    registry = ToolRegistry()
    registry.register(resolver)
    registry.register(market)
    events: list[tuple[str, dict[str, Any]]] = []
    agent = AgentLoop(
        registry=registry,
        llm=_CorrectingLLM(),
        max_iterations=4,
        event_callback=lambda event, data: events.append((event, data)),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)

    result = agent.run("请分析机器人ETF并给出买入价")

    assert result["status"] == "success"
    assert "1.137" in result["content"]
    assert "0.881" not in result["content"]
    assert market.calls == 1
    streamed = "".join(
        data.get("delta", "") for event, data in events if event == "text_delta"
    )
    assert "0.881" not in streamed
    assert "1.137" in streamed
    completed_thinking = "".join(
        data.get("content", "")
        for event, data in events
        if event == "thinking_done"
    )
    assert "0.881" not in completed_thinking
    artifact = json.loads(
        (run_dir / "artifacts" / "grounding_evidence.json").read_text(encoding="utf-8")
    )
    assert artifact["validations"][0]["valid"] is False
    assert artifact["validations"][-1]["valid"] is True


def test_agent_loop_stops_at_crypto_candidate_confirmation(tmp_path: Path) -> None:
    """Ambiguous crypto names ask the user instead of burning the tool budget."""
    resolver = _ResolverTool(
        _resolver_payload(
            query="SKHY",
            candidates=[
                {
                    "symbol": "SKHYB/USDT",
                    "name": "SK Hynix Tokenized bStocks",
                    "exchange": "Binance",
                    "type": "cryptocurrency",
                },
                {
                    "symbol": "SKHY/USDC",
                    "name": "SK Hynix Backpack Securities",
                    "exchange": "Meteora",
                    "type": "cryptocurrency",
                },
            ],
        )
    )
    registry = ToolRegistry()
    registry.register(resolver)

    class _OneResolverTurn:
        model_name = "grounding-test"

        def __init__(self) -> None:
            self.calls = 0

        def stream_chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Response:
            self.calls += 1
            return _Response(
                tool_calls=[_tool_call("resolve", "search_symbol", query="SKHY")]
            )

    llm = _OneResolverTurn()
    agent = AgentLoop(registry=registry, llm=llm, max_iterations=20)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)

    result = agent.run("SKHY 加密货币现在值得买入吗？")

    assert result["status"] == "success"
    assert result["iterations"] == 1
    assert llm.calls == 1
    assert "SKHYB/USDT" in result["content"]
    assert "SKHY/USDC" in result["content"]
    assert "平台" in result["content"]


_SHORTLIST_QUERY = "A股低价高增长股票"
_SHORTLIST_PAYLOAD = _resolver_payload(
    candidates=[
        {"symbol": "000543.SZ", "name": "皖能电力", "market": "cn", "source": "eastmoney"},
        {"symbol": "000727.SZ", "name": "冠捷科技", "market": "cn", "source": "eastmoney"},
    ],
    query=_SHORTLIST_QUERY,
)
_NARROWED_PAYLOAD = _resolver_payload(
    candidates=[
        {"symbol": "000543.SZ", "name": "皖能电力", "market": "cn", "source": "eastmoney"},
    ],
    query="000543.SZ",
)
_SCREENED_MARKET_PAYLOAD = json.dumps(
    {
        "000543.SZ": [
            {
                "trade_date": "2026-08-01",
                "open": 7.9,
                "high": 8.5,
                "low": 7.9,
                "close": 8.2,
                "volume": 100000,
            }
        ],
        "_provenance": {
            "000543.SZ": {
                "source": "tencent",
                "requested_source": "auto",
                "detected_source": "tencent",
                "fallback_used": False,
                "currency_conversion": "none",
            }
        },
    }
)


def _screened_ledger(tmp_path: Path) -> GroundingLedger:
    """Return a ledger that screened broadly, then locked one candidate."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="推荐A股低价高增长股票并给出买入价",
    )
    for call_id, query, payload in (
        ("shortlist", _SHORTLIST_QUERY, _SHORTLIST_PAYLOAD),
        ("narrow", "000543.SZ", _NARROWED_PAYLOAD),
    ):
        ledger.authorize_tool_call(
            "search_symbol",
            {"query": query},
            batch_authorized_symbols=ledger.authorized_symbols,
            call_id=call_id,
        )
        ledger.ingest_tool_result(
            tool_name="search_symbol",
            arguments={"query": query},
            result=payload,
            call_id=call_id,
            success=True,
        )
    ledger.ingest_tool_result(
        tool_name="get_market_data",
        arguments={"codes": ["000543.SZ"]},
        result=_SCREENED_MARKET_PAYLOAD,
        call_id="prices",
        success=True,
    )
    return ledger


def test_screening_shortlist_does_not_block_workflow_selection(tmp_path: Path) -> None:
    """A many-candidate screening result is an answer, not a stalled resolution (#955)."""
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="推荐A股低价高增长股票并给出买入价",
    )
    ledger.authorize_tool_call(
        "search_symbol",
        {"query": _SHORTLIST_QUERY},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="shortlist",
    )
    ledger.ingest_tool_result(
        tool_name="search_symbol",
        arguments={"query": _SHORTLIST_QUERY},
        result=_SHORTLIST_PAYLOAD,
        call_id="shortlist",
        success=True,
    )

    assert ledger.identity_status == "ambiguous"
    skill = ledger.authorize_tool_call(
        "load_skill",
        {"name": "stock-selection"},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="skill",
        batch_identity_status=ledger.identity_status,
    )

    assert skill.allowed is True


def test_narrowed_lock_retires_the_screening_shortlist(tmp_path: Path) -> None:
    """Locking a shortlisted candidate must unblock the run's final answer (#955)."""
    ledger = _screened_ledger(tmp_path)

    assert ledger.identity_status == "locked"
    assert ledger.authorized_symbols == {"000543.SZ"}
    prices = ledger.authorize_tool_call(
        "get_market_data",
        {"codes": ["000543.SZ"]},
        batch_authorized_symbols=ledger.authorized_symbols,
        call_id="prices",
        batch_identity_status=ledger.identity_status,
    )

    assert prices.allowed is True


def test_price_validation_ignores_symbol_date_and_quantity_digits(tmp_path: Path) -> None:
    """Ticker, calendar, holding-period, and position-cost digits are not prices (#955)."""
    ledger = _screened_ledger(tmp_path)

    for draft in (
        "000543.SZ 截至 8 月 3 日收盘价 8.20 CNY（source: tencent）",
        "000543.SZ 8-4收盘价 8.20 CNY（source: tencent）",
        "## 2️⃣ 现货价格\n000543.SZ 收盘价 8.20 CNY（source: tencent）",
        "3. **聚合报价彼此冲突**\n000543.SZ 收盘价 8.20 CNY（source: tencent）",
        "000543.SZ 买入价 8.20 CNY（100 股成本 820 CNY；source: tencent）",
        "000543.SZ 收盘价 8.20 CNY，套保合约价值 219 亿元（source: tencent）",
        "000543.SZ 建议持有 1–4 周，买入价 8.20 CNY（source: tencent）",
    ):
        result = ledger.validate_final_answer(draft)
        assert result.valid is True, (draft, result.issues)


def test_price_validation_still_rejects_a_quote_outside_observed_range(
    tmp_path: Path,
) -> None:
    """Masking non-price digits must not weaken the contradiction check (#955)."""
    ledger = _screened_ledger(tmp_path)

    result = ledger.validate_final_answer("000543.SZ 收盘价 42.00 CNY（source: tencent）")

    assert result.valid is False
    assert [issue["code"] for issue in result.issues] == ["numeric_claim_conflict"]


def test_screening_run_reaches_a_final_answer_through_the_agent_loop(
    tmp_path: Path,
) -> None:
    """End-to-end: screen, load a workflow skill, narrow, quote, and answer (#955)."""
    agent, resolver, market, skill, trace = _build_direct_agent(tmp_path, _SHORTLIST_PAYLOAD)
    market.result = _SCREENED_MARKET_PAYLOAD
    agent._grounding = GroundingLedger(
        run_dir=Path(agent.memory.run_dir),
        user_message="推荐A股低价高增长股票并给出买入价",
    )
    messages: list[dict[str, Any]] = []
    react_trace: list[dict[str, Any]] = []

    def batch(*calls: SimpleNamespace, iteration: int) -> None:
        agent._process_tool_calls(
            list(calls), ContextBuilder, messages, trace, react_trace, iteration
        )

    batch(_tool_call("shortlist", "search_symbol", query=_SHORTLIST_QUERY), iteration=1)
    batch(_tool_call("workflow", "load_skill", name="stock-selection"), iteration=2)
    assert skill.calls == 1

    resolver.result = _NARROWED_PAYLOAD
    batch(_tool_call("narrow", "search_symbol", query="000543.SZ"), iteration=3)
    batch(_tool_call("prices", "get_market_data", codes=["000543.SZ"]), iteration=4)
    trace.close()

    assert market.calls == 1
    validation = agent._grounding.validate_final_answer(
        "000543.SZ 买入价 8.20 CNY（100 股成本 820 CNY；source: tencent）"
    )
    assert validation.valid is True, validation.issues
