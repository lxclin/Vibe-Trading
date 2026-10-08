"""Primary-instrument and follow-up identity scope retained across the grounding split.

These methods come from the pre-merge grounding state machine. Peer lookups may
be audited, but only the user's active subject controls run-level identity.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from src.agent.grounding import identity as identity_module
from src.agent.grounding.identity import (
    IdentityRecord, _CANONICAL_SYMBOL_RE, _infer_currency,
    _infer_instrument_type, _infer_venue, _query_key,
)


def _normalize_symbol(value: Any) -> str:
    """Use the canonicalizer owned by the identity gate in every scope check."""
    return identity_module._normalize_symbol(value)

_BARE_NUMERIC_CODE_RE = re.compile(r"(?<![A-Za-z0-9_])\d{3,6}(?![A-Za-z0-9_])")

_BARE_LISTED_CODE_RE = re.compile(r"(?<![A-Za-z0-9_])\d{5,6}(?![A-Za-z0-9_])")

_REFERENTIAL_FOLLOWUP_RE = re.compile(
    r"^\s*(?:请\s*)?(?:"
    r"继续|接着|继续分析|接着分析|进一步分析|再看看|再分析一下|"
    r"同业(?:横向)?对比|同行(?:横向)?对比|横向对比|"
    r"(?:按|按照)?(?:短线|中线|长线)(?:给我|给出|制定)?(?:操作)?(?:方案|策略)?|"
    r"(?:后面|之后|接下来|下一步)(?:该)?(?:怎么|如何)(?:操作|处理|做)|"
    r"(?:成本|持仓成本|成本价)\s*[-+]?\d+(?:[.,]\d+)?\s*[,，、]?\s*"
    r"(?:后面|之后|接下来|下一步)(?:该)?(?:怎么|如何)(?:操作|处理|做)|"
    r"(?:给我|制定|做)?(?:止盈止损|仓位|操作|交易)(?:方案|策略)|"
    r"值得买入吗|值得买吗|能买吗|现在能买吗|"
    r"该股.{0,24}|这只(?:股票|基金)?.{0,24}|它.{0,24}|"
    r"上述(?:标的|股票|基金)?.{0,24}|前面(?:的|提到的)?.{0,24}|"
    r"(?:风险|估值|目标价|基本面|技术面)(?:呢|如何|怎么样)?"
    r")\s*[?？。！!]*\s*$|"
    r"^\s*(?:continue|go on|peer comparison|compare peers|what about it|"
    r"is it worth buying)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_TRADE_MANAGEMENT_FOLLOWUP_RE = re.compile(
    r"^\s*我\s*(?:刚刚|刚才|已经)?\s*"
    r"(?:[-+]?\d+(?:[.,]\d+)?\s*)?"
    r"(?:买入|买了|卖出|卖了|建仓|加仓|减仓|持有)\s*"
    r"(?:了\s*)?[-+]?\d+(?:[.,]\d+)?\s*"
    r"(?:USDT|USDC|USD|BTC|ETH|人民币|元|美元|港币|股|份)?"
    r"[^。！？\n]{0,24}(?:后面|之后|接下来|下一步|怎么办|该做什么|如何)"
    r"[^。！？\n]*\s*[?？。！!]*\s*$",
    re.IGNORECASE,
)

_CRYPTO_REQUEST_RE = re.compile(
    r"(?:虚拟货币|加密货币|数字货币|代币|交易对|现货价格|项目用途|"
    r"解锁计划|持仓集中度|token(?:s)?|crypto(?:currency)?|spot\s+price|"
    r"trading\s+pair|token\s+utility|unlock\s+schedule)",
    re.IGNORECASE,
)

_SCREENING_REQUEST_RE = re.compile(
    r"(?:推荐|筛选|选股|股票池|候选|低价|高增长|高股息|"
    r"(?:其他|别的|更好|替代)[^。！？\n]{0,10}(?:标的|股票|ETF|基金)|"
    r"top\s*\d+|screen|shortlist|find\s+(?:stocks?|funds?))",
    re.IGNORECASE,
)

_COMPARISON_FOLLOWUP_RE = re.compile(
    r"^(?=.{0,80}(?:这只|这个|该)(?:股票|基金|ETF|标的))"
    r"(?=.{0,140}(?:其他|别的|更好|替代|比较|对比)).+",
    re.IGNORECASE | re.DOTALL,
)


class IdentityScopeMixin:
    """Track the requested instrument separately from comparison symbols."""

    @property
    def primary_symbols(self) -> set[str]:
        """Return the user-requested instrument(s), excluding peer lookups."""
        return set(self._primary_symbols)

    def _is_primary_record(self, record: IdentityRecord) -> bool:
        """Whether an identity record belongs to the user's requested subject."""
        if self._screening_request:
            return True
        if record.status in {"ambiguous", "conflicting"} and self._primary_symbols:
            candidate_symbols = {
                _normalize_symbol(candidate.get("symbol"))
                for candidate in record.candidates
                if isinstance(candidate, Mapping)
            }
            if candidate_symbols & self._primary_symbols:
                return True
        if record.source_tool_call_id in {"user_message", "session_history"}:
            return bool(record.symbol and record.symbol in self._primary_symbols)
        if self._query_matches_primary_hint(record.query):
            return True
        return bool(record.symbol and record.symbol in self._primary_symbols)

    def _query_matches_primary_hint(self, query: str) -> bool:
        """Return whether a resolver query explicitly contains a primary hint."""
        key = _query_key(query)
        if key in self._primary_queries:
            return True
        tokens = {_query_key(code) for code in _BARE_LISTED_CODE_RE.findall(query)}
        return bool(tokens & self._primary_queries)

    def _has_locked_primary(self) -> bool:
        """Return whether at least one requested subject has a verified lock."""
        return any(
            record.status == "locked" and self._is_primary_record(record)
            for record in self._identities.values()
        )

    @property
    def identity_status(self) -> str:
        """Return the aggregate first-class identity state.

        Only records belonging to the user's primary subject participate in
        the aggregate. A conflicting primary outranks a successful lock; a
        failed peer or benchmark lookup cannot retract the verified subject.
        Per-symbol consumers still require their own exact locked identity.
        """
        records = [
            record
            for record in self._identities.values()
            if not self._is_shadowed_resolution(record)
            and (
                self._is_primary_record(record)
                or (not self._primary_queries and not self._primary_symbols)
            )
        ]
        if not records:
            return "unresolved" if self._identity_required else "not_required"
        statuses = {record.status for record in records}
        if "conflicting" in statuses:
            return "conflicting"
        if "locked" in statuses:
            return "locked"
        for blocking in ("ambiguous", "invalidated", "unresolved"):
            if blocking in statuses:
                return blocking
        if "not_found" in statuses:
            return "not_found"
        return "unresolved"

    def _is_shadowed_resolution(self, record: IdentityRecord) -> bool:
        """Ignore resolver failures that cannot override an explicit identity.

        The resolver is currently backed primarily by listed-security sources
        such as Yahoo.  A user may already have supplied an exact crypto pair
        (or another canonical symbol), in which case a later resolver outage
        must not invalidate that explicit identity.  The resolver result is
        still persisted for audit, but it is non-blocking until it resolves a
        genuinely different instrument.
        """
        if record.source_tool_call_id in {"user_message", "session_history"}:
            return False
        if record.status not in {"unresolved", "invalidated", "conflicting", "ambiguous"}:
            return False
        explicit = [
            item
            for item in self._identities.values()
            if item.source_tool_call_id in {"user_message", "session_history"}
            and item.status == "locked"
            and item.symbol
        ]
        if not explicit:
            return False
        # Transport failures and in-flight resolver records contain no
        # competing instrument.  Once the user has supplied an exact identity,
        # they must not poison that identity merely because a secondary lookup
        # (often a model-generated alias) failed.
        if record.status in {"unresolved", "invalidated"}:
            return True
        # A resolver that returned concrete alternatives is evidence, not an
        # outage. Shadow it only when every candidate canonicalizes to an
        # already explicit identity; a different venue must remain a conflict.
        candidate_symbols = {
            _normalize_symbol(candidate.get("symbol"))
            for candidate in record.candidates
            if isinstance(candidate, Mapping) and candidate.get("symbol")
        }
        explicit_symbols = {item.symbol for item in explicit if item.symbol}
        if candidate_symbols:
            return candidate_symbols.issubset(explicit_symbols)
        query_compact = re.sub(r"[^a-z0-9]", "", record.query.casefold())
        for item in explicit:
            symbol_compact = re.sub(r"[^a-z0-9]", "", item.symbol.casefold())
            if query_compact and query_compact == symbol_compact:
                return True
            symbol_core = re.split(r"[./-]", item.symbol.casefold(), maxsplit=1)[0]
            symbol_core = re.sub(r"[^a-z0-9]", "", symbol_core)
            # A resolver may ask for the bare ticker (SKHY/AAPL) after the
            # user supplied a canonical pair/suffix.  The explicit symbol is
            # authoritative; a failed lookup must not replace it.
            if query_compact and query_compact == symbol_core:
                return True
            # A resolver may shorten an explicitly supplied crypto base by a
            # venue suffix (SKHY -> SKHYB/USDT).  Only allow a reasonably long
            # prefix; accepting arbitrary substrings such as "HY" would let a
            # failed lookup for an unrelated token inherit the wrong identity.
            if (
                item.instrument_type == "crypto"
                and len(query_compact) >= 4
                and symbol_core.startswith(query_compact)
            ):
                return True
        return False

    @property
    def should_request_user_confirmation(self) -> bool:
        """Whether the run should stop and ask the user to choose a candidate."""
        if not self._identity_required or self.identity_status == "locked":
            return False
        records = [
            record
            for record in self._identities.values()
            if record.status in {"ambiguous", "conflicting", "invalidated", "unresolved"}
            and not self._is_shadowed_resolution(record)
            and (
                self._is_primary_record(record)
                or (not self._primary_queries and not self._primary_symbols)
            )
        ]
        if not records:
            return False
        if _CRYPTO_REQUEST_RE.search(self.user_message):
            return True
        if _SCREENING_REQUEST_RE.search(self.user_message):
            return False
        return any(
            record.status in {"ambiguous", "conflicting", "invalidated", "unresolved"}
            for record in records
        )

    def _seed_followup_history(
        self,
        history: Sequence[Mapping[str, Any]],
    ) -> None:
        """Inherit the nearest trusted identity for an elliptical follow-up.

        New session replies carry a structured ``grounding_identity`` summary.
        That is preferred over prose parsing.  For sessions created before the
        summary existed, walk backward through user turns only: skip other
        elliptical follow-ups, accept the first explicit canonical symbol, and
        stop at the first subject-setting message without a symbol.  The stop
        prevents an old AAPL symbol from authorizing a later SpaceX discussion.
        """
        reversed_history = list(reversed(history))
        for position, message in enumerate(reversed_history):
            if str(message.get("role") or "").casefold() != "assistant":
                continue
            summary = message.get("grounding_identity")
            if not isinstance(summary, Mapping):
                metadata = message.get("metadata")
                summary = (
                    metadata.get("grounding_identity")
                    if isinstance(metadata, Mapping)
                    else None
                )
            if not isinstance(summary, Mapping):
                continue
            # The nearest audited turn is authoritative even when it resolved
            # to not-found/ambiguous.  The narrowly-scoped recovery below is
            # the only exception: it applies when the failed query is the
            # same symbol as an older lock and the user is managing a fill.
            if summary.get("status") != "locked":
                # A retry of the same alternatives question may follow a
                # failed peer lookup. Skip only that exact failed question;
                # never walk through a genuine intervening subject change.
                if summary.get("status") == "invalidated" and _COMPARISON_FOLLOWUP_RE.match(self.user_message):
                    preceding_user = next((
                        prior for prior in reversed_history[position + 1:]
                        if str(prior.get("role") or "").casefold() == "user"
                    ), None)
                    failed = summary.get("records") or []
                    if (
                        preceding_user and preceding_user.get("content") == self.user_message
                        and failed and all(
                            isinstance(raw, Mapping) and raw.get("status") == "invalidated"
                            and not raw.get("candidates") for raw in failed
                        )
                    ):
                        continue
                # Older runs did not persist a primary-symbol scope.  If such
                # a run ended with a mixed/ambiguous summary, recover only the
                # single locked record matching the immediately preceding
                # user's explicit code (for example, 513330 -> 513330.SH).
                # This preserves stale-subject protection for natural-language
                # subject changes while making a safe "继续" useful after the
                # old peer-lookup bug.
                if _REFERENTIAL_FOLLOWUP_RE.fullmatch(self.user_message or ""):
                    subject_hints: set[str] = set()
                    for prior in reversed_history[position + 1 :]:
                        if str(prior.get("role") or "").casefold() != "user":
                            continue
                        content = str(prior.get("content") or "")
                        subject_hints.update(
                            _normalize_symbol(match.group(0))
                            for match in _CANONICAL_SYMBOL_RE.finditer(content)
                        )
                        subject_hints.update(_BARE_NUMERIC_CODE_RE.findall(content))
                        if subject_hints:
                            break
                        if not _REFERENTIAL_FOLLOWUP_RE.fullmatch(content):
                            break
                    locked_records = [
                        raw
                        for raw in (summary.get("records") or [])
                        if isinstance(raw, Mapping) and raw.get("status") == "locked"
                    ]
                    matching: list[Mapping[str, Any]] = []
                    for raw in locked_records:
                        symbol = _normalize_symbol(raw.get("symbol"))
                        if not symbol:
                            continue
                        base = symbol.split(".", 1)[0]
                        if any(
                            hint == symbol
                            or hint == base
                            or symbol.startswith(f"{hint}.")
                            for hint in subject_hints
                        ):
                            matching.append(raw)
                    if len(matching) == 1:
                        symbol = _normalize_symbol(matching[0].get("symbol"))
                        self._inherit_identity_record(symbol, matching[0])
                        self._primary_symbols.add(symbol)
                        return
                    # A subsequent failed retry may have replaced the useful
                    # legacy summary with an empty ``invalidated`` record. If
                    # that failed query still names the preceding subject,
                    # keep walking backward to the older audited summary.
                    if not locked_records and subject_hints:
                        failed_queries = {
                            _query_key(raw.get("query"))
                            for raw in (summary.get("records") or [])
                            if isinstance(raw, Mapping) and raw.get("query")
                        }
                        if any(
                            any(hint.casefold() in query for hint in subject_hints)
                            for query in failed_queries
                        ):
                            continue
                # A failed retry can leave an ``invalidated`` resolver record
                # for the very symbol that was already locked one turn earlier
                # (for example, a transient Yahoo outage on ``SKHY.US``).  For
                # a narrow fill-management follow-up, recover that exact prior
                # lock instead of making the user repeat the identity.  A
                # generic “continue” still fails closed, preserving the stale
                # subject protection below.
                if _TRADE_MANAGEMENT_FOLLOWUP_RE.fullmatch(self.user_message or ""):
                    failed_symbols = {
                        _normalize_symbol(raw.get("query") or raw.get("symbol"))
                        for raw in (summary.get("records") or [])
                        if isinstance(raw, Mapping)
                        and _normalize_symbol(raw.get("query") or raw.get("symbol"))
                    }
                    prior_locked_symbols: set[str] = set()
                    for older in reversed_history[position + 1 :]:
                        if str(older.get("role") or "").casefold() != "assistant":
                            continue
                        older_summary = older.get("grounding_identity")
                        if not isinstance(older_summary, Mapping):
                            older_metadata = older.get("metadata")
                            older_summary = (
                                older_metadata.get("grounding_identity")
                                if isinstance(older_metadata, Mapping)
                                else None
                            )
                        if not isinstance(older_summary, Mapping):
                            continue
                        if older_summary.get("status") == "locked":
                            prior_locked_symbols.update(
                                _normalize_symbol(symbol)
                                for symbol in older_summary.get("authorized_symbols", [])
                                if _normalize_symbol(symbol)
                            )
                            for raw in older_summary.get("records") or []:
                                if isinstance(raw, Mapping) and raw.get("status") == "locked":
                                    symbol = _normalize_symbol(raw.get("symbol"))
                                    if symbol:
                                        prior_locked_symbols.add(symbol)
                    if failed_symbols & prior_locked_symbols:
                        continue
                return
            authorized = {
                _normalize_symbol(symbol)
                for symbol in summary.get("authorized_symbols", [])
                if _normalize_symbol(symbol)
            }
            records = summary.get("records")
            if not isinstance(records, list):
                return
            raw_primary = summary.get("primary_symbols")
            if isinstance(raw_primary, list):
                inherited_scope = {
                    _normalize_symbol(symbol)
                    for symbol in raw_primary
                    if _normalize_symbol(symbol) in authorized
                }
            elif len(authorized) == 1:
                inherited_scope = set(authorized)
            else:
                # Legacy locked summaries did not distinguish the requested
                # instrument from locked peers. Recover a unique code match
                # from the preceding user turn; otherwise fail closed instead
                # of treating every peer as a primary subject.
                subject_hints: set[str] = set()
                for prior in reversed_history[position + 1 :]:
                    if str(prior.get("role") or "").casefold() != "user":
                        continue
                    content = str(prior.get("content") or "")
                    subject_hints.update(
                        _normalize_symbol(match.group(0))
                        for match in _CANONICAL_SYMBOL_RE.finditer(content)
                    )
                    subject_hints.update(_BARE_NUMERIC_CODE_RE.findall(content))
                    if subject_hints:
                        break
                    if not _REFERENTIAL_FOLLOWUP_RE.fullmatch(content):
                        break
                matched = {
                    symbol
                    for symbol in authorized
                    if any(
                        hint == symbol
                        or hint == symbol.split(".", 1)[0]
                        or symbol.startswith(f"{hint}.")
                        for hint in subject_hints
                    )
                }
                inherited_scope = matched if len(matched) == 1 else set()
            if not inherited_scope:
                return
            if _COMPARISON_FOLLOWUP_RE.match(self.user_message) and len(inherited_scope) != 1:
                return
            for raw_record in records:
                if not isinstance(raw_record, Mapping):
                    continue
                symbol = _normalize_symbol(raw_record.get("symbol"))
                if raw_record.get("status") != "locked" or symbol not in inherited_scope:
                    continue
                self._inherit_identity_record(symbol, raw_record)
            return

        for message in reversed(history):
            if str(message.get("role") or "").casefold() != "user":
                continue
            content = str(message.get("content") or "")
            symbols = [
                _normalize_symbol(match.group(0))
                for match in _CANONICAL_SYMBOL_RE.finditer(content)
            ]
            if symbols:
                for symbol in symbols:
                    self._inherit_identity_record(symbol, {})
                return
            if not _REFERENTIAL_FOLLOWUP_RE.fullmatch(content):
                return

    def _seed_named_followup_history(
        self,
        history: Sequence[Mapping[str, Any]],
    ) -> None:
        """Reuse a named subject only when prior audited prose ties it to one code.

        A previous turn may have locked multiple instruments. The current name
        must appear immediately beside exactly one of their canonical symbols
        in that turn's answer; a new company name cannot inherit a stale code.
        """
        for message in reversed(history):
            if str(message.get("role") or "").casefold() != "assistant":
                continue
            summary = message.get("grounding_identity")
            if not isinstance(summary, Mapping):
                metadata = message.get("metadata")
                summary = (
                    metadata.get("grounding_identity")
                    if isinstance(metadata, Mapping)
                    else None
                )
            if not isinstance(summary, Mapping):
                continue
            records = summary.get("records")
            if summary.get("status") != "locked":
                # A failed lookup for this same named subject may sit between
                # the current turn and the last successful audited answer.
                # A genuine ambiguous/conflicting result still stops reuse.
                if summary.get("status") == "invalidated" and isinstance(records, list):
                    failed = [
                        raw for raw in records
                        if isinstance(raw, Mapping)
                        and raw.get("status") == "invalidated"
                        and not raw.get("candidates")
                    ]
                    if failed and any(
                        str(raw.get("query") or "").strip() in self.user_message
                        for raw in failed
                        if str(raw.get("query") or "").strip()
                    ):
                        continue
                return
            if not isinstance(records, list):
                return
            authorized = {
                _normalize_symbol(symbol)
                for symbol in summary.get("authorized_symbols", [])
                if _normalize_symbol(symbol)
            }
            answer = str(message.get("content") or "")
            matches = [
                (symbol, raw)
                for raw in records
                if isinstance(raw, Mapping) and raw.get("status") == "locked"
                if (symbol := _normalize_symbol(raw.get("symbol"))) in authorized
                and self._answer_names_symbol(answer, symbol)
            ]
            if len(matches) == 1:
                symbol, record = matches[0]
                self._inherit_identity_record(symbol, record)
            return

    def _answer_names_symbol(self, answer: str, symbol: str) -> bool:
        """Match only a name printed directly next to a verified symbol."""
        edge = r"[\s*`_【】（）()\[\]{}:：,，;；]*"
        name = r"[\u3400-\u9fffA-Za-z][\u3400-\u9fffA-Za-z0-9·-]{2,23}"
        for match in _CANONICAL_SYMBOL_RE.finditer(answer):
            if _normalize_symbol(match.group(0)) != symbol:
                continue
            left = answer[max(0, match.start() - 48):match.start()]
            right = answer[match.end():match.end() + 48]
            before = re.search(rf"({name}){edge}$", left)
            after = re.match(rf"^{edge}({name})", right)
            if any(
                item and self._name_mentioned_in_request(item.group(1))
                for item in (before, after)
            ):
                return True
        return False

    def _name_mentioned_in_request(self, name: str) -> bool:
        """Reject a name that is only a prefix of a newly named company."""
        if not name:
            return False
        for match in re.finditer(re.escape(name), self.user_message, re.IGNORECASE):
            following = self.user_message[match.end():]
            if not following or not re.match(r"[\u3400-\u9fffA-Za-z0-9]", following):
                return True
            if re.match(
                r"(?:目前|当前|现在|最近|未来|的|是否|能否|机构|财报|估值|"
                r"股价|股票|和|与|跟|相比|值得|还有|怎么|如何)",
                following,
            ):
                return True
        return False

    def _inherit_identity_record(
        self,
        symbol: str,
        record: Mapping[str, Any],
    ) -> None:
        """Copy one previously locked identity into this run's audit ledger."""
        key = f"history:{symbol}"
        prior_sources = record.get("source")
        sources = (
            [str(item) for item in prior_sources if str(item).strip()]
            if isinstance(prior_sources, list)
            else []
        )
        if "session_history" not in sources:
            sources.append("session_history")
        self._identities[key] = IdentityRecord(
            query=symbol,
            status="locked",
            symbol=symbol,
            venue=str(record.get("venue") or "").strip() or _infer_venue(symbol),
            instrument_type=(
                str(record.get("instrument_type") or "").strip()
                or _infer_instrument_type(symbol)
            ),
            currency=(
                str(record.get("currency") or "").strip() or _infer_currency(symbol)
            ),
            source_tool_call_id="session_history",
            source=sources,
        )
        self._inherited_symbols.add(symbol)
        self._primary_symbols.add(symbol)
        self._identity_required = True
        self._buffer_output = True


    @staticmethod
    def _venue_compatible_candidates(
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply deterministic venue families to bare Chinese listed codes.

        Five-digit codes use the project's Hong Kong convention; six-digit
        codes use mainland exchanges.  This prevents a sparse Yahoo response
        for ``00700`` from silently locking an unrelated ``00700.TW`` product.
        Explicitly qualified queries are unaffected.
        """
        compact = str(query or "").strip()
        if _CANONICAL_SYMBOL_RE.search(compact):
            return candidates
        codes = set(_BARE_LISTED_CODE_RE.findall(compact))
        if len(codes) != 1:
            return candidates
        code = next(iter(codes))
        allowed = (".HK",) if len(code) == 5 else (".SH", ".SS", ".SZ", ".BJ")
        return [
            candidate
            for candidate in candidates
            if (
                (symbol := _normalize_symbol(candidate.get("symbol"))).endswith(allowed)
                and symbol.split(".", 1)[0].isdigit()
                and int(symbol.split(".", 1)[0]) == int(code)
            )
        ]

    def _choose_candidate(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Choose only a unique or strongly corroborated resolver candidate.

        Candidates are collapsed onto their canonical symbol first. Two rows
        that differ only by a provider's suffix convention describe one listing,
        and counting them as rival candidates is what left every Shanghai query
        with two "exact" matches and therefore no choice at all.
        """
        by_symbol: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            by_symbol.setdefault(_normalize_symbol(candidate.get("symbol")), candidate)
        candidates = list(by_symbol.values())
        if len(candidates) == 1:
            return candidates[0]
        # A bare crypto name is not enough to select one venue-specific pair.
        # For example, SKHY may resolve to both SKHYB/USDT and SKHY/USDC; the
        # model must ask the user instead of silently preferring the candidate
        # whose base happens to match the query exactly.
        if _CRYPTO_REQUEST_RE.search(self.user_message):
            crypto_candidates = [
                candidate
                for candidate in candidates
                if _infer_instrument_type(
                    _normalize_symbol(candidate.get("symbol")), candidate.get("type")
                ) == "crypto"
            ]
            if len(crypto_candidates) > 1:
                return None
        normalized_query = re.sub(r"[^a-z0-9\u3400-\u9fff]", "", query.casefold())
        exact: list[dict[str, Any]] = []
        strong: list[dict[str, Any]] = []
        for candidate in candidates:
            symbol = _normalize_symbol(candidate.get("symbol"))
            base = symbol.split(".", 1)[0].split("-", 1)[0].split("/", 1)[0]
            name = str(candidate.get("name") or "")
            comparable = {
                re.sub(r"[^a-z0-9\u3400-\u9fff]", "", base.casefold()),
                re.sub(r"[^a-z0-9\u3400-\u9fff]", "", name.casefold()),
                re.sub(r"[^a-z0-9\u3400-\u9fff]", "", symbol.casefold()),
            }
            normalized_name = re.sub(r"[^a-z0-9\u3400-\u9fff]", "", name.casefold())
            if normalized_query and (
                normalized_query in comparable
                or (
                    len(normalized_name) >= 4
                    and normalized_name in normalized_query
                )
            ):
                exact.append(candidate)
            if candidate.get("also_from"):
                strong.append(candidate)
        if len(exact) == 1:
            return exact[0]
        if len(strong) == 1:
            return strong[0]
        return None
