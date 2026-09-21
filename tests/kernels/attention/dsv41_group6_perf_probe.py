# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level performance probe for the SM90 group-6 K-reuse path.

The serving-level A/B (`VLLM_SM90_FP4_GROUP6=0` vs `1` on the decode instance)
is too noisy to resolve a kernel-level win: on two nodes with a shared NVMe,
run-to-run throughput varied 15-25 % at 64 prompts and concurrency 8.  This
probe measures the effect where it is defined -- one launch of the grouped
kernel versus the same work as six per-row launches -- with warmup, repeats and
an assertion that the timed grouped launch actually took the *shared* branch
(the per-CTA branch codes, so "it was faster" cannot come from a fallback that
merely reloads K per row).

    python3 tests/kernels/attention/dsv41_group6_perf_probe.py --repeats 50

Requires the same environment as the kernel tests: family(90) CUDA with
``VLLM_SM90_FP4_INDEXER=1`` and ``VLLM_SM90_FP4_GROUP6=1``.
"""

import argparse
import functools
import json
import os
import statistics
import sys
import time

import torch

# Runnable as a plain script: the repo root must be importable for the
# `tests.kernels.attention` helpers it reuses.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import vllm.envs as envs
import vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer as sm90_mod
from tests.kernels.attention.test_sm90_fp4_indexer import (
    PAGE_SIZE,
    _packed_cache,
    _rand_q,
    _valid_q_scale,
)
from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
    sm90_fp4_paged_index_logits,
)
from vllm.platforms import current_platform


def _available() -> bool:
    """family(90) CUDA with the SM90 FP4 indexer enabled."""
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and bool(envs.VLLM_SM90_FP4_INDEXER)
    )


def _time_launch(fn, repeats: int) -> float:
    """Median kernel time in microseconds over ``repeats`` launches."""
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def _launch(kind, q, qs, cache, w, lens, bt, row_ids, width):
    """One launch of the SM90 FP4 indexer kernel, per-row or grouped."""
    if kind == "grouped":
        return sm90_fp4_paged_index_logits(
            q,
            qs,
            cache,
            w,
            lens,
            bt,
            page_size=PAGE_SIZE,
            width=width,
            row_indices=row_ids,
            query_group_size=6,
        )
    return sm90_fp4_paged_index_logits(
        q, qs, cache, w, lens, bt, page_size=PAGE_SIZE, width=width
    )


def _case(rows: int, width: int, device: str):
    """Dense case wide enough for the live capture shapes.

    ``width`` is ``num_blocks * PAGE_SIZE`` columns, every row fully visible,
    one request per six rows (the DSpark static-verify shape).
    """
    assert width % PAGE_SIZE == 0
    torch.manual_seed(17)
    heads = 32
    num_blocks = width // PAGE_SIZE
    cache = _packed_cache(num_blocks, PAGE_SIZE, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        .reshape(1, -1)
        .repeat(rows, 1)
    )
    context_lens = torch.full((rows,), width, device=device, dtype=torch.int32)
    row_indices = torch.arange(rows, device=device, dtype=torch.int32) // 6
    return q_values, q_scale, cache, weights, context_lens, block_table, row_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not _available():
        print(
            "SKIP: needs family(90) CUDA with VLLM_SM90_FP4_INDEXER=1",
            file=sys.stderr,
        )
        return 2

    device = "cuda"
    envs_ok = bool(envs.VLLM_SM90_FP4_GROUP6)
    print(
        f"VLLM_SM90_FP4_GROUP6={int(envs_ok)} "
        f"VLLM_SM90_FP4_GROUP6_STATS={int(envs.VLLM_SM90_FP4_GROUP6_STATS)}"
    )
    results = []
    # The live DSpark capture shapes (group-6 is admitted when every request
    # owns exactly six flattened verify rows).
    for requests, width in ((15, 16384), (15, 32768), (30, 16384), (30, 32768)):
        rows = requests * 6
        q, qs, cache, w, lens, bt, row_ids = _case(rows, width, device)

        per_row = functools.partial(
            _launch, "per_row", q, qs, cache, w, lens, bt, row_ids, width
        )
        grouped = functools.partial(
            _launch, "grouped", q, qs, cache, w, lens, bt, row_ids, width
        )

        # Warmup both specializations (they are separate Triton binaries).
        for _ in range(args.warmup):
            per_row()
            grouped()
        torch.cuda.synchronize()

        sm90_mod.reset_group6_stats()
        grouped()
        torch.cuda.synchronize()
        stats = sm90_mod.sm90_fp4_group6_stats()

        t_row = _time_launch(per_row, args.repeats)
        t_grp = _time_launch(grouped, args.repeats)
        entry = {
            "requests": requests,
            "rows": rows,
            "width": width,
            "per_row_us": t_row,
            "grouped_us": t_grp,
            "speedup": t_row / t_grp if t_grp else float("nan"),
            "shared_ctas": stats["shared"],
            "fallback_ctas": stats["fallback"],
            "invalid_tile_ctas": stats["invalid_tile"],
        }
        results.append(entry)
        print(
            f"requests={requests:<3d} rows={rows:<4d} width={width:<6d} "
            f"per_row={t_row:8.1f}us grouped={t_grp:8.1f}us "
            f"speedup={entry['speedup']:5.2f}x shared={stats['shared']} "
            f"fallback={stats['fallback']} skipped={stats['invalid_tile']}"
        )
        if stats["shared"] == 0:
            print("  WARNING: no shared CTA -- the timing is not a reuse win")
        if stats["fallback"] != 0:
            print("  WARNING: fallback CTAs present -- reuse was partial")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"results": results, "ts": time.time()}, fh, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
