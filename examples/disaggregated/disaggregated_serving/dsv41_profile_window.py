#!/usr/bin/env python3
"""Drive one torch-profiler window on a live vLLM instance.

Collecting a trace is only useful if it is tied to a known batch shape, so this
script does the whole sequence in one place and refuses to guess:

  1. optionally warm the prefix (the deployed workload is a cached 131k prompt),
  2. POST /start_profile,
  3. run a burst of exactly `--concurrency` requests with a short output so the
     trace covers decode steps and not mostly prefill,
  4. POST /stop_profile,
  5. report where the trace files landed.

The profiled burst is deliberately *not* a performance measurement: profiling
slows every kernel down.  Run the same shape without the profiler for numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv41_long_context_bench import make_prompt, run_burst  # noqa: E402


async def post(session: aiohttp.ClientSession, url: str) -> tuple[int, str]:
    # Stopping flushes one trace per rank, which takes minutes on a loaded
    # instance (8 ranks x several hundred MB); a short timeout makes a
    # successful stop look like a failure.
    async with session.post(url, timeout=aiohttp.ClientTimeout(total=900)) as r:
        return r.status, (await r.text())[:200]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8200")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--prompt-tokens", type=int, default=131072)
    ap.add_argument("--concurrency", type=int, default=80)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--warm-concurrency", type=int, default=1)
    ap.add_argument("--warm-output-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/v1/completions"
    prompt = make_prompt(args.prompt_tokens, args.seed)
    result: dict = {
        "base_url": args.base_url,
        "prompt_tokens": args.prompt_tokens,
        "concurrency": args.concurrency,
        "output_len": args.output_len,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(
            f"warmup: {args.warm_concurrency} request(s), {args.warm_output_len} tokens"
        )
        warm, _, wall = await run_burst(
            session,
            url,
            args.model,
            [prompt],
            args.warm_output_len,
            True,
            args.warm_concurrency,
            burst=0,
            warmup=True,
            request_timeout=3600,
        )
        warm_ok = sum(1 for r in warm if r.ok)
        result["warmup_ok"] = warm_ok
        print(f"  warmup wall {wall:.1f}s ok={warm_ok}/{len(warm)}")
        result["warmup_wall_s"] = wall

        status, body = await post(session, f"{args.base_url}/start_profile")
        print(f"start_profile -> {status} {body}")
        if status != 200:
            print("the instance was not started with PREFILL_PROFILER_CONFIG")
            return 1

        print(
            f"profiled burst: {args.concurrency} x (prompt={args.prompt_tokens} "
            f"+ out={args.output_len})"
        )
        records, _, wall = await run_burst(
            session,
            url,
            args.model,
            [prompt],
            args.output_len,
            True,
            args.concurrency,
            burst=1,
            warmup=False,
            request_timeout=3600,
        )
        ok = sum(1 for r in records if r.ok)
        print(f"  profiled wall {wall:.1f}s ok={ok}/{len(records)}")

        status, body = await post(session, f"{args.base_url}/stop_profile")
        print(f"stop_profile -> {status} {body}")
        result.update(
            {
                "profiled_wall_s": wall,
                "profiled_ok": ok,
                "profiled_requests": len(records),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
