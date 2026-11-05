#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible chat completions endpoint under varying concurrency.

For each concurrency level N, N identical streaming requests are fired at once.
Per request we measure TTFT and decode speed; per level we also measure the
wall-clock time and how many requests were really generating at the same time.

How to read the results:
  * Server truly parallel : per-request tok/s stays ~flat, TTFT stays low,
                            aggregate tok/s grows ~linearly, peak overlap == N.
  * Server serializing    : per-request tok/s stays flat, but TTFT grows ~linearly,
                            aggregate tok/s stays flat, peak overlap == 1.
  * Server saturated      : per-request tok/s drops as N grows, aggregate flattens.

Usage:
    uv run llm-bench --benchmark [--url URL] [--model MODEL] [--levels 1 2 4 8]

Defaults come from the environment / .env (see llm_bench.settings).
"""
import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from . import settings


@dataclass
class RequestResult:
    success: bool
    error: Optional[str] = None
    # All timestamps are seconds on the perf_counter clock, shared by every request.
    start: float = 0.0
    first_token: float = 0.0
    end: float = 0.0
    tokens: int = 0

    @property
    def ttft_ms(self) -> float:
        return (self.first_token - self.start) * 1000

    @property
    def decode_tps(self) -> float:
        """Tokens/sec while generating (after the first token arrived)."""
        window = self.end - self.first_token
        return (self.tokens - 1) / window if self.tokens > 1 and window > 0 else 0.0


async def single_request(client: httpx.AsyncClient, url: str, payload: dict) -> RequestResult:
    start = time.perf_counter()
    first_token = None
    last_token = None
    chunks = 0
    usage_tokens = None

    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return RequestResult(False, error=f"HTTP {resp.status_code}")

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if event.get("usage"):
                    usage_tokens = event["usage"].get("completion_tokens", usage_tokens)

                choices = event.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    last_token = time.perf_counter()
                    first_token = first_token or last_token
                    chunks += 1
    except httpx.HTTPError as e:
        return RequestResult(False, error=f"{type(e).__name__}: {e}")

    if first_token is None:
        return RequestResult(False, error="no tokens received")

    # Prefer the server-reported count; fall back to streamed chunks (~1 token each).
    tokens = usage_tokens if usage_tokens is not None else chunks
    return RequestResult(True, start=start, first_token=first_token, end=last_token, tokens=tokens)


def peak_overlap(results: list[RequestResult]) -> int:
    """Max number of requests that were generating at the same instant.

    A request is "generating" between its first and last token.
    """
    events = []
    for r in results:
        events.append((r.first_token, +1))
        events.append((r.end, -1))
    events.sort(key=lambda e: (e[0], e[1]))  # at equal times, process the -1 first
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


async def run_level(url: str, concurrency: int, payload: dict, timeout: float) -> dict:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    # Read timeout applies between chunks; connect timeout catches dead endpoints fast.
    t = httpx.Timeout(timeout, connect=min(timeout, 10.0))
    async with httpx.AsyncClient(timeout=t, limits=limits) as client:
        wall_start = time.perf_counter()
        results = await asyncio.gather(*(single_request(client, url, payload) for _ in range(concurrency)))
        wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.success]
    row = {
        "concurrency": concurrency,
        "success_rate": 100.0 * len(ok) / concurrency,
        "errors": sorted({r.error for r in results if not r.success and r.error}),
        "wall_s": wall,
        "avg_ttft": None,
        "max_ttft": None,
        "avg_tps": None,
        "agg_tps": None,
        "overlap": None,
    }
    if ok:
        row.update(
            avg_ttft=statistics.mean(r.ttft_ms for r in ok),
            max_ttft=max(r.ttft_ms for r in ok),
            avg_tps=statistics.mean(r.decode_tps for r in ok),
            # Total tokens produced by everyone / time from first send to last token.
            agg_tps=sum(r.tokens for r in ok) / (max(r.end for r in ok) - min(r.start for r in ok)),
            overlap=peak_overlap(ok),
        )
    return row


def fmt(value: Optional[float], spec: str) -> str:
    return format(value, spec) if value is not None else "N/A"


def print_table(rows: list[dict]) -> None:
    header = ["N", "Avg TTFT ms", "Max TTFT ms", "Per-req tok/s", "Aggregate tok/s", "Peak overlap", "Wall s", "Success %"]
    print("\n| " + " | ".join(header) + " |")
    print("|" + "|".join("-" * (len(h) + 1) + ":" for h in header) + "|")
    for r in rows:
        cells = [
            str(r["concurrency"]),
            fmt(r["avg_ttft"], ".1f"),
            fmt(r["max_ttft"], ".1f"),
            fmt(r["avg_tps"], ".2f"),
            fmt(r["agg_tps"], ".2f"),
            fmt(r["overlap"], "d"),
            fmt(r["wall_s"], ".2f"),
            fmt(r["success_rate"], ".1f"),
        ]
        print("| " + " | ".join(c.rjust(len(h)) for c, h in zip(cells, header)) + " |")


def verdict(rows: list[dict]) -> str:
    """One-line interpretation based on the highest concurrency level that worked."""
    base, top = rows[0], rows[-1]
    if top["concurrency"] <= 1 or base["agg_tps"] is None or top["agg_tps"] is None:
        return "Need at least two successful levels to judge parallelism."
    scaling = top["agg_tps"] / base["agg_tps"]
    n = top["concurrency"]
    if scaling < 1.3 and top["overlap"] <= 1:
        return f"Server is SERIALIZING requests: {n}x load gave only {scaling:.2f}x throughput, no overlap."
    if scaling < 0.6 * n:
        return f"Server is parallel but SATURATING: {n}x load gave {scaling:.2f}x throughput."
    return f"Server scales well: {n}x load gave {scaling:.2f}x throughput."


async def run(url: str, model: str, levels: list[int], prompt: str, max_tokens: int,
              timeout: float, thinking: bool) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }

    rows = []
    for level in sorted(set(levels)):
        print(f"Running concurrency={level} ...", flush=True)
        row = await run_level(url, level, payload, timeout)
        for err in row["errors"]:
            print(f"  ! {err}")
        rows.append(row)

    print_table(rows)
    print("\n" + verdict(rows))


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=settings.URL, help="chat completions endpoint")
    p.add_argument("--model", default=settings.MODEL, help="model name sent in the request")
    p.add_argument("--levels", type=int, nargs="+", default=settings.BENCH_LEVELS, help="concurrency levels")
    p.add_argument("--prompt", default=settings.BENCH_PROMPT)
    p.add_argument("--max-tokens", type=int, default=settings.BENCH_MAX_TOKENS)
    p.add_argument("--timeout", type=float, default=settings.BENCH_TIMEOUT, help="per-request timeout in seconds")
    p.add_argument("--thinking", action="store_true", default=settings.BENCH_THINKING,
                   help="leave model thinking on (default: off)")
    a = p.parse_args(argv)
    asyncio.run(run(a.url, a.model, a.levels, a.prompt, a.max_tokens, a.timeout, a.thinking))
