#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Long-context burst client for the DeepSeek-V4.1-Flash PD port.

The port's earlier performance numbers came from ``vllm bench serve`` with a
2048-token random prompt, which cannot say anything about the deployed workload
(a ~131k-token prompt that mostly hits the prefiller's cache and then decodes
for thousands of tokens).  This client measures that workload directly and
defines every quantity it reports:

  * one deterministic, fixed token-id prompt (hashed and saved), sent as
    ``prompt: list[int]`` so the token count is exact and no chat template or
    tokenizer is involved on the client side;
  * streaming requests, so the first and last *content* deltas are timestamps
    rather than "the HTTP response came back";
  * burst semantics: every level runs ``--warmup-bursts`` excluded bursts plus
    ``--bursts`` measured ones, and each burst submits exactly C requests at
    once -- no refill, so the measured concurrency is the requested one;
  * per-request rates ``(completion_tokens - 1) / (t_last - t_first)``, which is
    the definition the reference deployment used, plus TTFT, ITL, e2el and the
    aggregate ``total output tokens / burst wall time``;
  * percentiles from the *pooled* per-request samples of the measured bursts,
    never an average of per-burst percentiles.

Failures are data: a request that errors, times out or ends without a finish
reason is recorded as a failure with its status and is excluded from the rate
statistics, so a benchmark cannot report a fast number by dropping slow work.

Nothing here talks to the engines' internals.  Optional ``--metrics-url``
sampling records what the engines themselves saw (running/waiting requests,
KV usage, prefix-cache hits) so the client-side and server-side views can be
compared.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp

# One engine's counters, e.g. vllm:num_requests_running{engine="0",...} 3.0
METRIC_RE = re.compile(
    r"^(?P<name>vllm:[a-z_]+)\{(?P<labels>[^}]*)\}\s+(?P<value>[0-9.eE+-]+)$", re.M
)
LABEL_RE = re.compile(r'(?P<key>[a-z_]+)="(?P<value>[^"]*)"')

# Server-side series worth sampling during a burst.  They are recorded as
# {name: {label-value-or-"": value}} so both per-engine vLLM counters and the
# router's unlabelled counters land in the same structure.
INTERESTING = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    # Speculative decoding: a token rate can rise while the *step* cost falls or
    # the other way round, so acceptance has to be recorded next to the rate.
    # ``drafts`` is also the iteration count, which turns a token rate into a
    # real per-verify-step time: a step time obtained by multiplying a TPOT by an
    # acceptance length measured in a *different* run is not a measurement.
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)


@dataclass
class RequestRecord:
    """Everything measured for one request, in seconds since burst start."""

    index: int
    burst: int
    warmup: bool
    prompt_tokens: int
    completion_tokens: int | None
    finish_reason: str | None
    ok: bool
    error: str | None
    t_send: float
    t_first_content: float | None
    t_last_content: float | None
    t_end: float | None
    ttft_s: float | None
    generation_s: float | None
    rate_tok_s: float | None
    tpot_ms: float | None
    e2el_s: float | None
    itl_ms: list[float] = field(default_factory=list)


@dataclass
class MetricsSample:
    t: float
    values: dict[str, dict[str, float]]


class MetricsPoller:
    """Poll one or more /metrics endpoints while the bursts run."""

    def __init__(self, urls: list[str], interval: float = 0.5) -> None:
        self.urls = urls
        self.interval = interval
        self.samples: list[MetricsSample] = []
        self._stop = asyncio.Event()

    async def _read(self, session: aiohttp.ClientSession, url: str) -> dict:
        out: dict[str, dict[str, float]] = {}
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                text = await r.text()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return out
        for match in METRIC_RE.finditer(text):
            name = match.group("name")
            if name not in INTERESTING:
                # Counter series are exposed with a `_total` suffix while the
                # declared name (and the docs) omit it; without this the
                # counters silently read as zero, which is worse than missing.
                name = name.removesuffix("_total")
                if name not in INTERESTING:
                    continue
            labels = {
                m.group("key"): m.group("value")
                for m in LABEL_RE.finditer(match.group("labels"))
            }
            # Per-engine counters keep their engine id; unlabelled ones use "".
            key = labels.get("engine", "")
            if name == "vllm:external_prefix_cache_hits" and not key:
                key = ""
            out.setdefault(name, {})[key] = float(match.group("value"))
        return out

    async def sample_once(self, session: aiohttp.ClientSession) -> None:
        t = time.time()
        merged: dict[str, dict[str, float]] = {}
        for url in self.urls:
            for name, values in (await self._read(session, url)).items():
                merged.setdefault(name, {}).update(values)
        if merged:
            self.samples.append(MetricsSample(t=t, values=merged))

    async def run(self) -> None:
        async with aiohttp.ClientSession() as session:
            while not self._stop.is_set():
                await self.sample_once(session)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval)

    def stop(self) -> None:
        self._stop.set()

    def peak(self, name: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for sample in self.samples:
            for key, value in sample.values.get(name, {}).items():
                if value > out.get(key, float("-inf")):
                    out[key] = value
        return out

    def delta(self, name: str) -> dict[str, float]:
        """Last minus first, for counters (hits, tokens)."""
        if len(self.samples) < 2:
            return {}
        first, last = self.samples[0].values, self.samples[-1].values
        keys = set(first.get(name, {})) | set(last.get(name, {}))
        return {
            key: last.get(name, {}).get(key, 0.0) - first.get(name, {}).get(key, 0.0)
            for key in keys
        }


def make_prompt(length: int, seed: int) -> list[int]:
    """Deterministic token ids.  Avoids the special ids (0/1/2/128799)."""
    rng = random.Random(seed)
    return [rng.randrange(3, 128000) for _ in range(length)]


def prompt_sha256(prompt: list[int]) -> str:
    payload = json.dumps(prompt, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


async def stream_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    ignore_eos: bool,
    index: int,
    burst: int,
    warmup: bool,
    request_timeout: float,
) -> RequestRecord:
    """Send one streaming completion and timestamp its content deltas.

    ``t_first_content``/``t_last_content`` are the first and last SSE events that
    actually carry text.  A final usage-only event (``stream_options``) gives the
    real completion-token count including speculative tokens, which a chunk count
    would miss.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    record = RequestRecord(
        index=index,
        burst=burst,
        warmup=warmup,
        prompt_tokens=len(prompt),
        completion_tokens=None,
        finish_reason=None,
        ok=False,
        error=None,
        t_send=time.time(),
        t_first_content=None,
        t_last_content=None,
        t_end=None,
        ttft_s=None,
        generation_s=None,
        rate_tok_s=None,
        tpot_ms=None,
        e2el_s=None,
    )
    chunk_times: list[float] = []
    try:
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with session.post(url, json=body, timeout=timeout) as resp:
            if resp.status != 200:
                record.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                record.t_end = time.time()
                return record
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except ValueError:
                    continue
                usage = event.get("usage")
                if usage:
                    record.completion_tokens = usage.get("completion_tokens")
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        record.finish_reason = choice["finish_reason"]
                    text = choice.get("text") or ""
                    if text:
                        now = time.time()
                        if record.t_first_content is None:
                            record.t_first_content = now
                        record.t_last_content = now
                        chunk_times.append(now)
        record.t_end = time.time()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.t_end = time.time()
        return record

    if record.t_first_content is None or record.t_last_content is None:
        record.error = "no content delta"
        return record
    record.ttft_s = record.t_first_content - record.t_send
    record.e2el_s = record.t_end - record.t_send
    record.generation_s = record.t_last_content - record.t_first_content
    tokens = record.completion_tokens
    if tokens is None:
        tokens = len(chunk_times)
    if tokens and tokens > 1 and record.generation_s > 0:
        record.rate_tok_s = (tokens - 1) / record.generation_s
        record.tpot_ms = 1000.0 * record.generation_s / (tokens - 1)
    record.itl_ms = [1000.0 * (b - a) for a, b in zip(chunk_times, chunk_times[1:])]
    # A stream that never reported a finish reason did not complete normally,
    # however much text it produced.
    record.ok = record.finish_reason is not None
    if not record.ok:
        record.error = "stream ended without a finish_reason"
    return record


async def run_burst(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompts: list[list[int]],
    max_tokens: int,
    ignore_eos: bool,
    concurrency: int,
    burst: int,
    warmup: bool,
    request_timeout: float,
) -> tuple[list[RequestRecord], float, float]:
    """Submit exactly ``concurrency`` requests at once; return records and wall."""
    gate = asyncio.Event()

    async def one(i: int) -> RequestRecord:
        await gate.wait()
        return await stream_one(
            session,
            url,
            model,
            prompts[i % len(prompts)],
            max_tokens,
            ignore_eos,
            index=i,
            burst=burst,
            warmup=warmup,
            request_timeout=request_timeout,
        )

    tasks = [asyncio.create_task(one(i)) for i in range(concurrency)]
    await asyncio.sleep(0)  # let every task reach the gate
    started = time.time()
    gate.set()
    records = await asyncio.gather(*tasks)
    wall = time.time() - started
    return list(records), started, wall


def summarize(records: list[RequestRecord], wall_total: float) -> dict:
    """Aggregate only the measured (non-warmup) records."""
    measured = [r for r in records if not r.warmup]
    ok = [r for r in measured if r.ok]
    failed = [r for r in measured if not r.ok]
    rates = [r.rate_tok_s for r in ok if r.rate_tok_s]
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    e2els = [r.e2el_s for r in ok if r.e2el_s is not None]
    itls = [x for r in ok for x in r.itl_ms]
    out_tokens = sum(r.completion_tokens or 0 for r in ok)
    return {
        "requests": len(measured),
        "completed": len(ok),
        "failed": len(failed),
        "failed_detail": [
            {"index": r.index, "burst": r.burst, "error": r.error} for r in failed[:5]
        ],
        "output_tokens": out_tokens,
        "burst_wall_s": wall_total,
        "aggregate_output_tok_s": out_tokens / wall_total if wall_total else None,
        "request_rate_tok_s": {
            "p50": percentile(rates, 0.5),
            "p95": percentile(rates, 0.95),
            "min": min(rates) if rates else None,
            "max": max(rates) if rates else None,
            "mean": statistics.fmean(rates) if rates else None,
        },
        "ttft_s": {"p50": percentile(ttfts, 0.5), "p95": percentile(ttfts, 0.95)},
        "tpot_ms": {"p50": percentile(tpots, 0.5), "p95": percentile(tpots, 0.95)},
        "itl_ms": {"p50": percentile(itls, 0.5), "p95": percentile(itls, 0.95)},
        "e2el_s": {"p50": percentile(e2els, 0.5), "p95": percentile(e2els, 0.95)},
        "completion_tokens": {
            "min": min((r.completion_tokens or 0) for r in ok) if ok else None,
            "max": max((r.completion_tokens or 0) for r in ok) if ok else None,
        },
        "eos_finished": sum(1 for r in ok if r.finish_reason == "stop"),
        "length_finished": sum(1 for r in ok if r.finish_reason == "length"),
    }


def print_summary(concurrency: int, summary: dict) -> None:
    rr = summary["request_rate_tok_s"]

    def fmt(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    print(
        f"c{concurrency}: completed={summary['completed']}/{summary['requests']} "
        f"failed={summary['failed']} out_tokens={summary['output_tokens']} "
        f"agg={fmt(summary['aggregate_output_tok_s'], 1)} tok/s"
    )
    print(
        f"    per-request tok/s p50={fmt(rr['p50'], 1)} min={fmt(rr['min'], 1)} "
        f"p95={fmt(rr['p95'], 1)}"
    )
    print(
        f"    ttft_s p50={fmt(summary['ttft_s']['p50'])} "
        f"p95={fmt(summary['ttft_s']['p95'])} | "
        f"tpot_ms p50={fmt(summary['tpot_ms']['p50'], 2)} "
        f"p95={fmt(summary['tpot_ms']['p95'], 2)} | "
        f"e2el_s p50={fmt(summary['e2el_s']['p50'], 2)}"
    )
    print(
        f"    finish: stop={summary['eos_finished']} "
        f"length={summary['length_finished']} "
        f"tokens/req min={summary['completion_tokens']['min']} "
        f"max={summary['completion_tokens']['max']}"
    )
    for detail in summary["failed_detail"]:
        print(
            f"    FAILED idx={detail['index']} burst={detail['burst']}: "
            f"{detail['error']}"
        )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8192")
    ap.add_argument("--endpoint", default="/v1/completions")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompt-tokens", type=int, default=131072)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--bursts", type=int, default=2, help="measured bursts per level")
    ap.add_argument("--warmup-bursts", type=int, default=1)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument(
        "--unique-prompts",
        type=int,
        default=0,
        help="0 = every request shares one prompt (prefiller cache warm); "
        "N = N distinct prompts cycled through the burst (cold KV)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--request-timeout", type=float, default=3600.0)
    ap.add_argument(
        "--metrics-url",
        action="append",
        default=[],
        help="repeatable; /metrics endpoint to sample during the bursts",
    )
    ap.add_argument("--out", type=Path, default=None, help="JSON result file")
    args = ap.parse_args()

    n_prompts = max(1, args.unique_prompts)
    prompts = [make_prompt(args.prompt_tokens, args.seed + i) for i in range(n_prompts)]
    url = args.base_url.rstrip("/") + args.endpoint
    result: dict = {
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "unique_prompts": args.unique_prompts,
        "seed": args.seed,
        "bursts": args.bursts,
        "warmup_bursts": args.warmup_bursts,
        "input_ids_sha256": [prompt_sha256(p) for p in prompts],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "levels": {},
    }
    print(
        f"prompt={args.prompt_tokens} tokens max_tokens={args.max_tokens} "
        f"ignore_eos={args.ignore_eos} prompts={n_prompts} "
        f"sha256={result['input_ids_sha256'][0][:16]}..."
    )

    poller = MetricsPoller(args.metrics_url) if args.metrics_url else None
    poller_task = asyncio.create_task(poller.run()) if poller else None

    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        for concurrency in args.concurrency:
            records: list[RequestRecord] = []
            measured_wall = 0.0
            for burst in range(args.warmup_bursts + args.bursts):
                warmup = burst < args.warmup_bursts
                burst_records, _, wall = await run_burst(
                    session,
                    url,
                    args.model,
                    prompts,
                    args.max_tokens,
                    args.ignore_eos,
                    concurrency,
                    burst=burst,
                    warmup=warmup,
                    request_timeout=args.request_timeout,
                )
                tag = "warmup" if warmup else "measured"
                print(
                    f"  c{concurrency} burst {burst} ({tag}): {wall:.1f}s "
                    f"ok={sum(1 for r in burst_records if r.ok)}/{len(burst_records)}"
                )
                records.extend(burst_records)
                if not warmup:
                    measured_wall += wall
            summary = summarize(records, measured_wall)
            print_summary(concurrency, summary)
            result["levels"][str(concurrency)] = {
                "summary": summary,
                "requests": [asdict(r) for r in records],
            }

    if poller and poller_task:
        poller.stop()
        await poller_task
        # One final sample so a run shorter than the polling interval still has
        # a "last" value to difference the counters against.
        async with aiohttp.ClientSession() as session:
            await poller.sample_once(session)
        result["engine_metrics"] = {
            "urls": args.metrics_url,
            "samples": len(poller.samples),
            "peak_running": poller.peak("vllm:num_requests_running"),
            "peak_waiting": poller.peak("vllm:num_requests_waiting"),
            "peak_kv_cache_usage_perc": poller.peak("vllm:kv_cache_usage_perc"),
            "prompt_tokens_delta": poller.delta("vllm:prompt_tokens"),
            "generation_tokens_delta": poller.delta("vllm:generation_tokens"),
            "prefix_cache_hits_delta": poller.delta("vllm:prefix_cache_hits"),
            "prefix_cache_queries_delta": poller.delta("vllm:prefix_cache_queries"),
            "external_prefix_cache_hits_delta": poller.delta(
                "vllm:external_prefix_cache_hits"
            ),
            "spec_decode_drafts_delta": poller.delta("vllm:spec_decode_num_drafts"),
            "spec_decode_draft_tokens_delta": poller.delta(
                "vllm:spec_decode_num_draft_tokens"
            ),
            "spec_decode_accepted_tokens_delta": poller.delta(
                "vllm:spec_decode_num_accepted_tokens"
            ),
        }
        # Acceptance length per drafting round; 1.0 means every draft was
        # rejected, so a "fast" token rate with acceptance 1.0 is just eager
        # decoding wearing a speculative config.
        drafts = sum(result["engine_metrics"]["spec_decode_drafts_delta"].values())
        accepted = sum(
            result["engine_metrics"]["spec_decode_accepted_tokens_delta"].values()
        )
        result["engine_metrics"]["acceptance_length"] = (
            1.0 + accepted / drafts if drafts else None
        )
        print(
            f"spec decode: drafts={drafts} accepted={accepted} "
            f"acceptance_length={result['engine_metrics']['acceptance_length']}"
        )
        print(
            "engine peak running="
            f"{poller.peak('vllm:num_requests_running')} "
            f"waiting={poller.peak('vllm:num_requests_waiting')} "
            f"kv%={poller.peak('vllm:kv_cache_usage_perc')}"
        )

    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
        print(f"saved {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
