#!/usr/bin/env python3
"""Summarize a torch-profiler chrome trace by GPU kernel self time.

A step time can only be attributed to a kernel if the kernel times are summed
from the trace, not guessed from a model of the computation.  This reads one
rank's trace and prints:

  * the wall span the trace covers and how much of it is GPU-active,
  * the top kernels by summed duration (self time -- each event is one launch),
  * the same grouped by a coarse family (attention / moex / gemm / ...) so the
    answer is "which subsystem", not "which of 400 kernel names".

Usage: dsv41_trace_summary.py <trace.json.gz> [--top 25] [--family]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict

# Coarse families, matched in order against the kernel name.  The list is
# deliberately small and readable: the point is to rank subsystems, and a
# mis-sorted kernel is visible in the per-name table that follows.
FAMILIES: list[tuple[str, re.Pattern[str]]] = [
    (
        "indexer_topk",
        # The sparse indexer owns the candidate scoring *and* the top-k
        # selection.  The ported SM90 kernels are named
        # `..._index_logits_kernel`, which a regex looking only for "indexer"
        # misses -- and that is exactly the kernel that dominates here.
        re.compile(
            r"topk|indexer|index_logits|select|radix|mqa_logits|deepselect", re.I
        ),
    ),
    ("mla_attention", re.compile(r"flashmla|mla|fmha|attention|attn", re.I)),
    (
        "moe",
        re.compile(r"moe|expert|group_gemm|grouped|deepgemm|cutlass.*gemm.*fp4", re.I),
    ),
    ("fp8_gemm", re.compile(r"fp8|block32|static", re.I)),
    ("engram", re.compile(r"engram", re.I)),
    ("mhc", re.compile(r"mhc|split_h|sinkhorn", re.I)),
    (
        "communication",
        # `all_reduce` (multimem) and `allreduce` (flashinfer mnnvl) are the
        # same operation; matching only one spelling files half the collective
        # time under "other".
        re.compile(
            r"nccl|all_?reduce|allgather|all_gather|reduce_scatter"
            r"|all_to_all|p2p|multimem",
            re.I,
        ),
    ),
    ("norm_act", re.compile(r"norm|silu|gelu|rope|quant", re.I)),
    ("sampling", re.compile(r"sample|argmax|top_p|top_k|penalt", re.I)),
]


def family_of(name: str) -> str:
    for family, pattern in FAMILIES:
        if pattern.search(name):
            return family
    return "other"


def load(path: str) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--cat", default="kernel", help="trace category to summarize")
    args = ap.parse_args()

    data = load(args.trace)
    events = data.get("traceEvents", [])
    kernels = [e for e in events if e.get("cat") == args.cat and "dur" in e]

    by_name: dict[str, list[float]] = defaultdict(list)
    for event in kernels:
        by_name[event["name"]].append(event["dur"])

    total_us = sum(sum(v) for v in by_name.values())
    if not kernels:
        print(f"no '{args.cat}' events in {args.trace}")
        return 1
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e["dur"] for e in kernels)
    span_us = end - start

    print(f"trace: {args.trace}")
    print(f"kernel events: {len(kernels):,}  distinct: {len(by_name):,}")
    print(f"GPU busy (sum of kernel dur): {total_us / 1e6:.2f} s")
    print(f"span covered:                 {span_us / 1e6:.2f} s")
    print(f"GPU utilisation (sum/span):   {100.0 * total_us / span_us:.1f}%")

    print(f"\ntop {args.top} kernels by self time:")
    print(f"{'ms total':>10} {'%':>6} {'calls':>8} {'ms/call':>9}  name")
    for name, durations in sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[
        : args.top
    ]:
        tot = sum(durations) / 1000.0
        print(
            f"{tot:10.2f} {100.0 * sum(durations) / total_us:6.2f} "
            f"{len(durations):8d} {tot / len(durations):9.3f}  {name[:70]}"
        )

    families: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for name, durations in by_name.items():
        families[family_of(name)] += sum(durations)
        counts[family_of(name)] += len(durations)
    print("\nby family:")
    print(f"{'ms total':>10} {'%':>6} {'calls':>10}  family")
    for family, dur in sorted(families.items(), key=lambda kv: -kv[1]):
        print(
            f"{dur / 1000.0:10.2f} {100.0 * dur / total_us:6.2f} "
            f"{counts[family]:10d}  {family}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
