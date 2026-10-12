"""Offline research cost accounting; missing provider usage is never estimated."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(events, *, usage_rows=()):
    rounds = {}
    tools = {}
    queries = Counter()
    edits = Counter()
    times = []
    for event in events:
        iteration = event.get("iter", 0)
        row = rounds.setdefault(iteration, {"iteration": iteration, "model_calls": 0,
            "model_errors": 0, "model_elapsed_ms": 0, "usage_reports": 0,
            "input_tokens": 0, "output_tokens": 0, "tool_calls": 0,
            "max_reported_input_tokens": 0,
            "tool_elapsed_ms_sum": 0, "empty_searches": 0})
        if isinstance(event.get("ts"), (int, float)):
            times.append(event["ts"])
        kind = event.get("type")
        if kind == "llm_call_cost":
            row["model_calls"] += 1
            row["model_errors"] += event.get("status") == "error"
            row["model_elapsed_ms"] += event.get("elapsed_ms", 0)
            usage = event.get("usage")
            if isinstance(usage, dict):
                row["usage_reports"] += 1
                row["input_tokens"] += usage.get("input_tokens", 0)
                row["output_tokens"] += usage.get("output_tokens", 0)
                row["max_reported_input_tokens"] = max(row["max_reported_input_tokens"], usage.get("input_tokens", 0))
        elif kind == "tool_call":
            row["tool_calls"] += 1
            args = event.get("args", event.get("arguments", {}))
            args = {k: v for k, v in args.items() if k != "run_dir"}
            queries[(event.get("tool"), json.dumps(args, sort_keys=True))] += 1
        elif kind == "tool_result":
            name = event.get("tool", "unknown")
            tool = tools.setdefault(name, {"tool": name, "calls": 0, "errors": 0,
                                          "elapsed_ms_sum": 0, "elapsed_ms_max": 0})
            duration = event.get("elapsed_ms", 0)
            tool["calls"] += 1
            tool["errors"] += event.get("status") == "error"
            tool["elapsed_ms_sum"] += duration
            tool["elapsed_ms_max"] = max(tool["elapsed_ms_max"], duration)
            row["tool_elapsed_ms_sum"] += duration
            if name == "web_search":
                try:
                    payload = json.loads(event.get("result", ""))
                    row["empty_searches"] += isinstance(payload, dict) and payload.get("results") == []
                except (TypeError, ValueError):
                    pass
        if kind in {"answer_rejected", "answer_repaired", "research_delivery_review",
                    "tool_result_reused", "tool_result_cached", "tool_result_replayed"}:
            edits[kind] += 1
    # Historical traces lack model call events. Artifact usage covers only
    # successful reported calls; it cannot reconstruct failed calls or timing.
    if not any(r["model_calls"] for r in rounds.values()):
        for usage in usage_rows:
            row = rounds.get(usage.get("iter"))
            if row is not None:
                row["usage_reports"] += 1
                row["input_tokens"] += usage.get("input_tokens", 0)
                row["output_tokens"] += usage.get("output_tokens", 0)
                row["max_reported_input_tokens"] = max(row["max_reported_input_tokens"], usage.get("input_tokens", 0))
    rows = sorted(rounds.values(), key=lambda r: r["iteration"])
    totals = {k: sum(r[k] for r in rows) for k in rows[0] if k != "iteration"} if rows else {}
    totals["max_reported_input_tokens"] = max((r["max_reported_input_tokens"] for r in rows), default=0)
    for row in [*rows, totals]:
        if not row.get("usage_reports"):
            row["input_tokens"] = row["output_tokens"] = None
            row["max_reported_input_tokens"] = None
        if not row.get("model_calls"):
            row["model_elapsed_ms"] = None
            row["model_calls"] = row["model_errors"] = None
    return {"scope": "main agent; nested swarm and compaction model calls excluded",
            "boundary": "Tool duration sums overlap during parallel execution; not wall time. "
                        "Tokens are reported amounts only; absent usage is null and partial coverage is explicit.",
            "wall_elapsed_ms": round((max(times) - min(times)) * 1000) if times else None,
            "usage_complete": bool(totals.get("model_calls")) and totals.get("model_calls") == totals.get("usage_reports"),
            "totals": totals, "events": dict(edits),
            "repeated_exact_queries": sum(n - 1 for n in queries.values()),
            "tools_by_elapsed": sorted(tools.values(), key=lambda t: t["elapsed_ms_sum"], reverse=True),
            "per_iteration": rows}


def read_events(path, first_iteration=0):
    events = []
    with Path(path).open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
                if event.get("iter", 0) >= first_iteration:
                    events.append(event)
            except (ValueError, TypeError):
                continue  # A truncated final line must not hide earlier costs.
    return events


def write_report(run_dir, trace_path, *, first_iteration=0):
    report = summarize(read_events(trace_path, first_iteration))
    path = Path(run_dir) / "research_cost.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--usage", type=Path, help="Optional historical llm_usage.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    usage = json.loads(args.usage.read_text()).get("per_iteration", []) if args.usage else []
    report = summarize(read_events(args.trace), usage_rows=usage)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
