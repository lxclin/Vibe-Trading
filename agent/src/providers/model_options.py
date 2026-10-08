"""Model-specific request options shared by settings and provider adapters."""

from __future__ import annotations


def is_gpt_61_sol(model: str) -> bool:
    return model.strip().split("/")[-1] == "gpt-6.1-sol"


def normalize_reasoning_effort(model: str, effort: str | None) -> str:
    """Migrate legacy disabled-reasoning settings when selecting Sol 6.1.

    Sol 6.1 requires reasoning and supports low/medium/high/xhigh/max.
    Empty retains the provider default; other models keep their own options.
    """
    value = (effort or "").strip().lower()
    if is_gpt_61_sol(model) and value in {"none", "minimal"}:
        return "medium"
    return value
