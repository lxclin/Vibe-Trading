"""The :class:`GroundingLedger` facade the agent loop drives."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agent.resolution_context import ResolutionContext

from src.agent.grounding.identity import (
    IdentityRecord,
    _IdentityMixin,
    _normalize_symbol,
    _query_key,
    _RESOLVER_TOOL,
    _scan_symbols,
    _utc_now,
)
from src.agent.grounding.identity_scope import (
    IdentityScopeMixin,
    _BARE_LISTED_CODE_RE,
    _REFERENTIAL_FOLLOWUP_RE,
    _TRADE_MANAGEMENT_FOLLOWUP_RE,
)
from src.agent.grounding.evidence import EvidenceRecord, _EvidenceMixin, _json_object
from src.agent.grounding.decision import decision_coverage, decision_issues, is_single_buy_question
from src.agent.grounding.figures import Figure, parse_figures_block, scan_figures, strip_figures_block
from src.agent.grounding.policies import ValidationResult, _PolicyMixin
from src.agent.grounding.registry import GROUNDING_CHECKS
from src.agent.grounding.release import (
    MAX_GROUNDING_RECOVERY_ROUNDS,
    MAX_PRICE_EVIDENCE_ATTEMPTS,
    MAX_SYMBOL_RESOLUTION_ATTEMPTS,
    _ReleaseMixin,
)

GROUNDING_ARTIFACT = "grounding_evidence.json"


# Tools whose successful completion can ground backtest/analysis metric claims
# (#1336). Success alone is not authority: only results that actually carried
# metric output (parsed from the result or its run-dir artifacts) raise
# ``analysis_claim_unavailable``. A deduplicated ("skipped") call is never a
# completion.
_ANALYSIS_TOOLS = frozenset(
    {"backtest", "factor_analysis", "run_shadow_backtest", "quantlib_call"}
)


def _passed_figures(
    figures: Sequence[Figure], issues: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    """The measured figures no issue flags, as written, in document order.

    The correction prompt needs them for its keep list. A spelling that fails
    anywhere in the draft is endorsed nowhere, so the list only carries values
    the gate can vouch for outright. Bare integers stay off: only the
    price-posing ones are checked, so a plain "bare" pass proves nothing.
    """
    flagged_spans = {
        (int(span[0]), int(span[1]))
        for issue in issues
        if isinstance((span := issue.get("span")), (list, tuple)) and len(span) == 2
    }
    flagged_values = {
        str(issue.get("value")) for issue in issues if issue.get("value") is not None
    }
    passed: list[str] = []
    seen: set[str] = set()
    for figure in figures:
        if figure.shape != "measured":
            continue
        if (figure.start, figure.end) in flagged_spans:
            continue
        if figure.text in flagged_values or figure.text in seen:
            continue
        seen.add(figure.text)
        passed.append(figure.text)
    return tuple(passed)


_ACTIONABLE_MARKET_RE = re.compile(
    r"(?:\bbuy\b|\bsell\b|\bentry\b|\btarget price\b|\bcurrent price\b|"
    r"\blatest price\b|\bprice of\b|\btrade\b|"
    r"\bvaluation of\b|\bwhat (?:is|are) .{1,80} worth\b|"
    r"\bis .{1,80} (?:listed|publicly traded)\b|"
    r"买入|卖出|入场|目标价|现价|最新价|股价|交易价格|估值|值多少钱|"
    r"值得买|能买吗|能否买|适合买|是否值得投资|"
    r".{1,40}(?:是否|有没有|已经|已)(?:在.{0,20})?上市)",
    re.IGNORECASE,
)
_CRYPTO_CONFIRMATION_RE = re.compile(
    r"(?:虚拟货币|加密货币|数字货币|代币|交易对|token(?:s)?|crypto(?:currency)?|"
    r"trading\s+pair)",
    re.IGNORECASE,
)
_SCREENING_REQUEST_RE = re.compile(
    r"(?:推荐|筛选|选股|股票池|候选|低价|高增长|高股息|top\s*\d+|screen|shortlist|"
    r"find\s+(?:stocks?|funds?))",
    re.IGNORECASE,
)
_META_DELIVERY_RE = re.compile(
    r"(?:报告|结果|结论).{0,16}(?:已|已经)(?:交付|完成|给出)|"
    r"上方(?:完整(?:版|报告)|报告正文)|不再重复(?:内容)?",
    re.IGNORECASE,
)
_IDENTITY_ABSTENTION_RE = re.compile(
    r"无法(?:安全)?确认|不能确认|无法核验|请(?:提供|确认|告诉)|"
    r"需要.*(?:平台|交易所|代码|交易对)|存在多个候选|"
    r"unable to (?:verify|confirm|resolve)|please (?:provide|confirm)|multiple candidates",
    re.IGNORECASE,
)
_UNSAFE_MARKET_CONCLUSION_RE = re.compile(
    r"(?:建议|应当|应该|可以|适合|值得|推荐).{0,12}(?:买入|卖出|建仓|加仓)|"
    r"(?:买入价|卖出价|入场价|目标价)\s*[:：]?\s*[-+]?\d|"
    r"(?:但|仍|依然|预计|预测|判断|认为|结论是).{0,12}(?:看涨|看跌|上涨|下跌)|"
    r"(?:现价|价格为|报价为|收盘价为)\s*[$¥￥]?\s*\d|"
    r"(?:recommend|should|worth).{0,20}\b(?:buy|sell|enter)\b",
    re.IGNORECASE,
)


class GroundingLedger(
    IdentityScopeMixin,
    _IdentityMixin,
    _EvidenceMixin,
    _PolicyMixin,
    _ReleaseMixin,
):
    """Run-scoped identity state machine and evidence ledger."""

    @staticmethod
    def _is_safe_identity_abstention(content: str) -> bool:
        return bool(_IDENTITY_ABSTENTION_RE.search(content)) and not bool(
            _UNSAFE_MARKET_CONCLUSION_RE.search(content)
        )

    def __init__(
        self,
        *,
        run_dir: Path,
        user_message: str,
        history: Sequence[Mapping[str, Any]] | None = None,
        contextual_identity_constraints: bool = True,
    ) -> None:
        """Create a ledger and seed only authoritative prior identities.

        Args:
            run_dir: Active run directory.
            user_message: Current user request.
            history: Optional prior message history. It remains available to
                the model. Only explicit constraints whose clause names the
                current resolver subject may carry forward; stale global
                instructions cannot authorize a new subject.
            contextual_identity_constraints: Whether explicit market words in
                the original conversation may narrow resolver candidates.
        """
        self.run_dir = Path(run_dir)
        self.user_message = user_message
        self.resolution_context = ResolutionContext.from_messages(
            user_message,
            history,
            enabled=contextual_identity_constraints,
        )
        self._identities: dict[str, IdentityRecord] = {}
        self._inherited_symbols: set[str] = set()
        self._primary_queries: set[str] = set()
        self._primary_symbols: set[str] = set()
        self._screening_request = bool(_SCREENING_REQUEST_RE.search(user_message))
        self._prefer_chinese = bool(re.search(r"[\u3400-\u9fff]", user_message)) or any(
            bool(re.search(r"[\u3400-\u9fff]", str(item.get("content") or "")))
            for item in (history or []) if isinstance(item, Mapping)
        )
        self._evidence: list[EvidenceRecord] = []
        self._tool_failures: list[dict[str, Any]] = []
        self._decision_tool_attempts: list[dict[str, Any]] = []
        self._analysis_completed: list[dict[str, Any]] = []
        self._analysis_metrics: list[dict[str, Any]] = []
        self._validations: list[dict[str, Any]] = []
        # The one document that actually shipped through the release path, if
        # any. Not a draft, so it stays out of ``validation_count``.
        self._released: dict[str, Any] | None = None
        self._recovery_rounds = 0
        self._symbol_resolution_attempts = 0
        self._price_evidence_attempts = 0
        self._ingested_csvs: set[str] = set()
        # Per-bar tables completed backtests wrote: resolved path -> (sha256,
        # backtest scope), so a later read of one can be recognised as engine
        # output. Files the model wrote itself never qualify.
        self._engine_tables: dict[str, tuple[str, str]] = {}
        self._model_written: set[str] = set()
        # Backtest call -> its run directory, and each directory's latest call.
        self._backtest_scopes: dict[str, str] = {}
        self._scope_latest: dict[str, str] = {}
        self._identity_required = bool(_ACTIONABLE_MARKET_RE.search(user_message))
        self._decision_required = is_single_buy_question(user_message)
        self._buffer_output = self._identity_required
        # Every instrument this run is entitled to write about: the ones the
        # user named, plus the ones a succeeding tool call passed in or returned.
        self._session_symbols: set[str] = _scan_symbols(user_message)
        # Bare tickers a succeeding call passed in, e.g. "AAPL" for the nine
        # tools whose contract is a bare US ticker. "AAPL.US" in the answer then
        # names an instrument the run really handled.
        self._session_symbol_roots: set[str] = set()

        self._seed_symbols(user_message, source="user_message")
        self._primary_symbols.update(self.authorized_symbols)
        if self._identity_required and not self._screening_request and not self._primary_symbols:
            self._primary_queries.update(
                key for code in _BARE_LISTED_CODE_RE.findall(user_message)
                if (key := _query_key(code))
            )
        if not self._identities and (
            _REFERENTIAL_FOLLOWUP_RE.fullmatch(user_message or "")
            or _TRADE_MANAGEMENT_FOLLOWUP_RE.fullmatch(user_message or "")
        ):
            self._seed_followup_history(history or [])
        elif self._identity_required and not self._screening_request and not self._identities:
            self._seed_named_followup_history(history or [])
        self.persist()

    @property
    def authorized_symbols(self) -> set[str]:
        """Return exact symbols locked before the next tool batch."""
        return {
            record.symbol
            for record in self._identities.values()
            if record.status == "locked" and record.symbol
        }

    @property
    def inherited_symbols(self) -> set[str]:
        """Return symbols inherited from prior turns, if any."""
        return set(self._inherited_symbols)



    def clarification_prompt(self) -> str:
        """Describe competing instruments and ask for a venue or full symbol."""
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in self._identities.values():
            if record.status not in {"ambiguous", "conflicting"}:
                continue
            for candidate in record.candidates:
                symbol = _normalize_symbol(candidate.get("symbol"))
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    candidates.append(candidate)

        if _CRYPTO_CONFIRMATION_RE.search(self.user_message):
            candidates.sort(
                key=lambda item: (
                    str(item.get("type") or "").casefold()
                    not in {"crypto", "cryptocurrency"},
                    _normalize_symbol(item.get("symbol")),
                )
            )
        is_chinese = bool(re.search(r"[\u3400-\u9fff]", self.user_message))
        labels: list[str] = []
        for candidate in candidates[:8]:
            symbol = _normalize_symbol(candidate.get("symbol"))
            name = str(candidate.get("name") or "").strip()
            venue = str(
                candidate.get("exchange") or candidate.get("market") or ""
            ).strip()
            kind = str(candidate.get("type") or "").strip()
            details = " / ".join(item for item in (name, venue, kind) if item)
            labels.append(f"{symbol}（{details}）" if details else symbol)

        source_failed = any(
            record.status == "invalidated"
            and not record.candidates
            and (
                self._is_primary_record(record)
                or (not self._primary_queries and not self._primary_symbols)
            )
            for record in self._identities.values()
        )
        if is_chinese:
            if labels:
                return (
                    "我找到多个可能的标的，但还不能确认你要交易哪一个："
                    f"{'；'.join(labels)}。\n\n"
                    "请告诉我具体平台、交易所或完整交易对，再继续查询行情。"
                )
            if source_failed:
                return "标的搜索数据源暂时不可用，无法核验证券代码。请稍后重试，或直接提供证券代码。"
            return "我还不能确认唯一的交易标的。请提供具体平台、交易所或完整交易对。"
        if labels:
            return (
                "I found multiple possible instruments but cannot tell which one you mean: "
                f"{'; '.join(labels)}. Please provide the venue, exchange, or full trading pair."
            )
        if source_failed:
            return (
                "The symbol-search data source is unavailable, so I cannot verify "
                "the ticker. Please retry later or provide the ticker directly."
            )
        return (
            "I cannot confirm one unique instrument yet. Please provide the venue, "
            "exchange, or full trading pair."
        )

    @property
    def should_buffer_output(self) -> bool:
        """Return whether unverified model prose must be hidden from live sinks."""
        return self._buffer_output or bool(self._evidence)

    @property
    def validation_count(self) -> int:
        """Return the number of final drafts checked so far."""
        return len(self._validations)

    @property
    def figures_removed(self) -> int:
        """Return how many figures the discounted release cut, or 0."""
        return int((self._released or {}).get("figures_removed", 0))

    @staticmethod
    def streamable_length(text: str) -> int:
        """Return how much of a streaming answer may be shown live.

        Held back: everything from the figures block's fence on (the model's
        declaration to the gate), a last line that could still become that fence,
        everything from the first measurement-shaped number (the gate has not
        checked it, and a rejected draft must not have shown it), and a number
        still being written ("0." before "666").

        Args:
            text: The answer streamed so far.

        Returns:
            The length of the prefix that is safe to emit.
        """
        span = parse_figures_block(text).span
        limit = len(text)
        if span is not None:
            limit = span[0]
        else:
            line_start = text.rfind("\n") + 1
            if text[line_start:].lstrip()[:1] in ("`", "~"):
                limit = line_start
        prefix = text[:limit]
        measured = GroundingLedger.measurement_start(prefix)
        if measured is not None:
            return measured
        trimmed = prefix.rstrip(" \t")
        cursor = len(trimmed)
        while cursor and (trimmed[cursor - 1].isdigit() or trimmed[cursor - 1] in ".,"):
            cursor -= 1
        if any(char.isdigit() for char in trimmed[cursor:]):
            return cursor
        return limit

    @staticmethod
    def measurement_start(text: str) -> int | None:
        """Where the first measurement-shaped number in ``text`` starts, or None."""
        for figure in scan_figures(text, parse_figures_block(text)):
            if figure.shape == "measured":
                return figure.start
        return None

    def identity_summary(self) -> dict[str, Any]:
        """Return compact identity state for traces and tool errors."""
        return {
            "status": self.identity_status,
            "authorized_symbols": sorted(self.authorized_symbols),
            "primary_symbols": sorted(self.primary_symbols),
            "inherited_symbols": sorted(self.inherited_symbols),
            "records": [asdict(record) for record in self._identities.values()],
            "recovery": self.recovery_summary(),
        }

    def recovery_summary(self) -> dict[str, Any]:
        """Return bounded-recovery budget state for traces and the artifact."""
        return {
            "rounds": self._recovery_rounds,
            "max_rounds": MAX_GROUNDING_RECOVERY_ROUNDS,
            "symbol_resolution_attempts": self._symbol_resolution_attempts,
            "max_symbol_resolution_attempts": MAX_SYMBOL_RESOLUTION_ATTEMPTS,
            "price_evidence_attempts": self._price_evidence_attempts,
            "max_price_evidence_attempts": MAX_PRICE_EVIDENCE_ATTEMPTS,
        }

    def ingest_tool_result(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        result: str,
        call_id: str,
        success: bool,
    ) -> None:
        """Consume the full untruncated tool result and persist its evidence.

        Args:
            tool_name: Executed tool name.
            arguments: Exact normalized tool arguments.
            result: Full raw result, before model-context truncation.
            call_id: Provider tool-call identity.
            success: Result-envelope success classification.
        """
        payload = _json_object(result)
        if self._decision_required and tool_name in {
            "get_a_share_valuation", "get_financial_statements"
        }:
            self._decision_tool_attempts.append({
                "call_id": call_id,
                "tool": tool_name,
                "symbol": _normalize_symbol(arguments.get("code")),
                "statement": (
                    str(arguments.get("statement") or "indicators")
                    if tool_name == "get_financial_statements" else None
                ),
                "period": (
                    str(arguments.get("period") or "annual")
                    if tool_name == "get_financial_statements" else None
                ),
                "success": success,
            })
        if not success:
            self._record_tool_failure(tool_name, call_id, result)
            if tool_name == _RESOLVER_TOOL:
                self._finish_failed_resolution(arguments, call_id)
            elif tool_name == "get_market_data" and payload is not None and payload.get("_unresolved"):
                # A fetch where no symbol returned data is a failed call, but
                # each symbol is still recorded as unavailable, as it was
                # when this envelope counted as a success.
                self._ingest_market_data(arguments, payload, call_id)
            self.persist()
            return

        self._track_session_symbols(arguments, result)
        self._note_model_write(tool_name, arguments, payload)
        if tool_name in _ANALYSIS_TOOLS:
            self._ingest_analysis_result(tool_name, arguments, payload, call_id)
        if tool_name == _RESOLVER_TOOL:
            self._ingest_resolution(arguments, payload, call_id)
        elif tool_name == "get_market_data":
            self._ingest_market_data(arguments, payload, call_id)
        elif tool_name == "read_file" and payload is not None:
            self._ingest_engine_table(payload, call_id)
        elif tool_name == "read_run_artifact" and payload is not None:
            self._ingest_engine_table(payload, call_id, tool_name=tool_name)
        elif payload is not None:
            self._ingest_generic_numeric(tool_name, arguments, payload, call_id)
        self.persist()

    def validate_final_answer(self, content: str) -> ValidationResult:
        """Validate identity assertions and numeric price claims.

        Args:
            content: Candidate assistant answer.

        Returns:
            A deterministic validation result. A record containing the answer
            hash, structured issues, and the names of the declared checks that
            fired is appended to the artifact.
        """
        return self._validate(content, record=True)

    def revalidate(self, content: str) -> ValidationResult:
        """Validate text WITHOUT counting it as a rejected draft.

        For the recheck that follows a deterministic repair or redaction. No
        model round produced that text, so recording it would count a
        rejected draft nobody wrote and add an artifact attempt no draft
        stands behind.

        Args:
            content: The repaired or redacted answer.

        Returns:
            A deterministic validation result.
        """
        return self._validate(content, record=False)

    def _validate(self, content: str, *, record: bool) -> ValidationResult:
        """Run the gate, optionally without recording the attempt.

        ``validation_count`` is the run's checked-DRAFT count: the degraded-run
        reason prints it and the ``grounding_status`` round follows it (the
        revision cap itself is counted in ``loop.py``). The release path's own
        rechecks are not drafts, so they pass ``record=False``.

        Args:
            content: Candidate assistant answer.
            record: Whether to append the attempt to the persisted ledger.

        Returns:
            A deterministic validation result.
        """
        self._ingest_run_dir_ohlc_csvs()
        block = parse_figures_block(content)
        figures = scan_figures(content, block)
        issues: list[dict[str, Any]] = []
        if (
            len(content) <= 1_200
            and _META_DELIVERY_RE.search(content)
            and not re.search(
                r"(?:^|\n)\s{0,3}#{1,4}\s+|(?:核心结论|交易对|风险|建议|findings|conclusion)",
                content,
                re.IGNORECASE,
            )
        ):
            issues.append({
                "code": "meta_delivery_without_report",
                "value": None,
                "role": None,
                "span": None,
                "symbol": None,
                "message": "A report was claimed but its body is absent from this answer.",
            })
        issues.extend(self._validate_identity(content))
        if not (
            self.identity_status in {"unresolved", "ambiguous", "conflicting", "invalidated"}
            and self._is_safe_identity_abstention(content)
        ):
            issues.extend(self._validate_unsourced_symbols(content, figures, block))
            issues.extend(self._validate_figures(content, block, figures))
            if self._decision_required and self.identity_status == "locked":
                issues.extend(decision_issues(
                    content, self._evidence, self.primary_symbols, self._tool_failures,
                    self._decision_tool_attempts,
                ))
        issues = self._dedupe_issues(issues)
        result = ValidationResult(
            valid=not issues,
            issues=issues,
            released_text=strip_figures_block(content, block),
            passed_figures=_passed_figures(figures, issues),
        )
        if not record:
            return result
        self._validations.append(
            {
                "attempt": len(self._validations) + 1,
                "checked_at": _utc_now(),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "valid": result.valid,
                "issues": issues,
                # Names let a run card say which declared checks fired without
                # re-running the gate; inline rules carry codes only until migrated.
                "fired_checks": GROUNDING_CHECKS.names_for_codes(
                    {str(issue.get("code") or "") for issue in issues}
                ),
                "figures_block": block.raw,
            }
        )
        self.persist()
        return result

    def persist(self) -> None:
        """Atomically persist the current structured ledger."""
        artifact_dir = self.run_dir / "artifacts"
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / GROUNDING_ARTIFACT
            temp = path.with_suffix(path.suffix + ".tmp")
            primary = next(iter(self.primary_symbols)) if len(self.primary_symbols) == 1 else None
            payload = {
                "schema_version": 1,
                "updated_at": _utc_now(),
                "identity": self.identity_summary(),
                "session_symbols": sorted(self._session_symbols),
                "session_symbol_roots": sorted(self._session_symbol_roots),
                "evidence": [asdict(record) for record in self._evidence],
                "tool_failures": list(self._tool_failures),
                "decision_tool_attempts": list(self._decision_tool_attempts),
                "research_coverage": (
                    decision_coverage(
                        self._evidence, primary,
                        self._decision_tool_attempts,
                    )
                    if self._decision_required and primary is not None
                    and primary.endswith((".SH", ".SZ", ".BJ"))
                    and not primary.startswith(("5", "1"))
                    else None
                ),
                "analysis_completed": list(self._analysis_completed),
                "analysis_evidence": list(self._analysis_metrics),
                "validations": list(self._validations),
                "released": dict(self._released) if self._released else None,
            }
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temp.replace(path)
        except OSError:
            # Grounding decisions remain in memory; a read-only/broken artifact
            # directory must not crash the agent's error path.
            return
