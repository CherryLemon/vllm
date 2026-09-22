# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measurement harness for the DeepSeek-V4.1 SM90 FP8 prefill-indexer question.

This script answers the three experiments (E1-E3) of

    /public-nvme/yjwu/dsv41-h100-port/reports/design_invalid_tile_and_fp8_prefill.md
    section "Gap 2 - prefill is not yet the source H100 FP8 prefill scoring path"

It measures; it does **not** implement or enable an FP8 kernel.  No production
file is changed: ``sm90_fp4_indexer.py`` is imported read-only.

===============================================================================
  THIS SCRIPT NEEDS A FREE, IDLE GPU
===============================================================================
It launches real Triton kernels and real CUDA work on the device given by
``--device``.  **Do not run it while another served vLLM instance owns the
device** - it will not be polite about sharing.  It also must not be run inside
a CUDA-graph-capturing process; it is a standalone benchmark.  The reference
checkout used by the live server is never touched: run it from a private
worktree.

===============================================================================
What is measured
===============================================================================
E1 - timing split of the *current* compact prefill path
    ``SparseMQAIndexer._forward_sm90`` prefill branch is:
        (a) ``_gather_prefill_chunk_k``  (``cp_gather_indexer_k_quant_cache``)
        (b) ``sm90_sparse_mqa_logits_prefill_chunk``  (Triton logits)
        (c) ``SparseMQAIndexer._prefill_candidate_topk``
    Each is timed separately and reported for (T, rows) shapes spanning the
    workload: ``--lengths`` are the *compressed* workspace length ``T`` (the
    gathered workspace row count = sum of full compressed contexts) and
    ``--rows`` are the chunk's new query tokens.  A dense-mode probe
    (``candidate_blocks=None``, output width = T) is also timed because the
    design's proposed FP8 kernel replaces the *dense* ``SparseAttnIndexer``
    path; see the note below.  CUDA events, warm-up, median + p10/p90.

E2 - clamp/amax census
    Decode the packed MXFP4 K workspace to fp32 exactly (nibble x UE8M0) and
    census: amax, ``|v| > 448`` (the E4M3 clamp), non-zero ``|v| < 2**-9``
    (the E4M3 subnormal flush), ``|v| < 2**-6`` (subnormal region), the UE8M0
    exponent histogram and the per-32-block decoded amax.  Also done for the
    packed Q.  A deliberate wide-scale workspace (``--wide-factor``) shows
    where the clamp starts to bind.

E3 - numeric + top-k selection equivalence
    Pure-torch *reference-only* FP8 scoring (no Triton):
        decode fp4 -> fp32 -> clamp +/-448 -> E4M3, emulate
        ``tl.dot(q_e4m3, k_e4m3.T, out_dtype=fp32)`` in fp32 and fp64, then
        apply the same bf16 post-dot chain as the current kernel
    compared against the current bf16 path (the real Triton kernel when a GPU
    is present, a torch emulation on ``--cpu-selftest``).  Reports max/mean abs
    logit error, the fraction of entries that are bit-identical after the bf16
    chain, and the top-``index_topk`` selection overlap.

Capturing a real workspace for E2
---------------------------------
The synthetic workspace reproduces the MXFP4 mechanism, not this model's
weights.  To answer E2 for the real model, dump the gathered workspace from a
live step (CPU tensors) and pass it with ``--workspace-pt``::

    # temporary instrumentation only - never commit it
    # in SparseMQAIndexer._forward_sm90, after _gather_prefill_chunk_k(...)
    torch.save({"k_values": k_quant[: chunk.local_total_seq_lens].cpu(),
                "k_scales": k_scale[: chunk.local_total_seq_lens].cpu()},
               f"/tmp/kws_T{chunk.local_total_seq_lens}.pt")

Repeat for 8K/32K/128K/600K chunks.  The harness then decodes exactly those
bytes and the clamp question is answered on real data.

===============================================================================
Shape/model constants (read from the tree at commit db9113b741)
===============================================================================
* workspace K: ``(T, 64)`` uint8 E2M1 nibbles + ``(T, 4)`` uint8 UE8M0,
  built by ``cp_gather_indexer_k_quant_cache`` through
  ``_gather_prefill_chunk_k`` / ``_gather_workspace_shapes``.
* Q: ``(rows, H, 64)`` uint8 packed E4-like E2M1 (vLLM's own fp4 Q), scales
  ``(rows, H)`` int32 = 4 UE8M0 bytes/head.
* The chunker budget is ``min_split_seq_len`` in
  ``vllm/v1/attention/backends/mla/sparse_indexer.py`` (SM90 path: fp32 logits
  so ``min_split_seq_len = num_sparse_cols = candidate_topk_blocks *
  candidate_block_size``), and the per-step logits budget is
  ``VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`` (default 512 MiB):
  ``rows * max(T, min_split_seq_len) * 4 <= max_logits_bytes``.
  ``--lengths``/``--rows`` are checked against that budget and the planned
  chunk split is reported; infeasible combinations are still measured (they
  are valid kernel shapes) but flagged.

===============================================================================
Scope / honesty notes
===============================================================================
* The production SM90 prefill (``SparseMQAIndexer._forward_sm90``) is a
  *candidate* (compact) path, width ``candidate_topk_blocks *
  candidate_block_size``.  The design report's E1 text wraps the *dense*
  ``SparseAttnIndexer`` block, and its proposed FP8 diff targets dense.  This
  harness measures **both**: ``compact`` (the path named in the task) and
  ``dense`` (the path the FP8 kernel would replace).
* E3 scores the compact and/or dense path; the fp4->E4M3 contract is the same.
* The synthetic workspace is drawn i.i.d. Gaussian and quantized through a
  faithful torch re-implementation of the production MXFP4 quantizer
  (``_indexer_k_norm_rope_quant_store_kernel`` / ``_quantize_mxfp4_pair``).
  That reproduces the UE8M0 *mechanism* but **not** real model weights.  The
  only way to settle E2 for this model is ``--workspace-pt`` with a workspace
  captured from a real forward pass.

Usage
-----
    # plan only, no torch, no CUDA:
    python3 tests/kernels/attention/dsv41_fp8_prefill_probe.py --dry-run

    # pure-CPU self-test of the torch reference maths (tiny shapes):
    python3 tests/kernels/attention/dsv41_fp8_prefill_probe.py --cpu-selftest

    # the real measurement, on an idle H100:
    python3 tests/kernels/attention/dsv41_fp8_prefill_probe.py \
        --lengths 8192 32768 131072 600000 --rows 16 64 512 \
        --seeds 0 1 2 --out /tmp/fp8.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants that mirror the production kernels / model family.  Defaults come
# from the design report and the SM90 kernel comments; override on the CLI.
# ---------------------------------------------------------------------------
HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
MXFP4_BLOCK_SIZE = 32
SCALE_BYTES = 4
FP8_E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0**-9  # below this an fp32 value flushes to fp8 zero
E4M3_MIN_NORMAL = 2.0**-6  # [2**-9, 2**-6) is the E4M3 subnormal region
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _default_lengths():
    return [8192, 32768, 131072, 600000]


def _default_rows():
    return [16, 64, 512]


# ===========================================================================
# CLI
# ===========================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "E1-E3 measurement harness for the DeepSeek-V4.1 SM90 FP8 prefill "
            "indexer question. Needs a free GPU (except --dry-run / "
            "--cpu-selftest)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=_default_lengths(),
        help="compressed workspace lengths T (gathered rows).",
    )
    p.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=_default_rows(),
        help="new query-token rows per chunk.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="RNG seeds; E2/E3 are repeated per seed.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="write the JSON summary here (also printed to stdout).",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="vLLM checkout root; defaults to three levels above this file.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="parse args, print shapes/plan and a JSON skeleton; no torch.",
    )
    p.add_argument(
        "--cpu-selftest",
        action="store_true",
        help="run the pure-torch E2/E3 maths on CPU with tiny shapes.",
    )
    p.add_argument("--skip-e1", action="store_true")
    p.add_argument("--skip-e2", action="store_true")
    p.add_argument("--skip-e3", action="store_true")
    p.add_argument(
        "--no-dense", action="store_true", help="skip the dense-mode E1 probe."
    )
    # geometry / model constants
    p.add_argument("--heads", type=int, default=32, help="index_n_heads.")
    p.add_argument("--head-dim", type=int, default=HEAD_DIM)
    p.add_argument(
        "--page-size",
        type=int,
        default=64,
        help="indexer cache tokens per page (attention block size).",
    )
    p.add_argument("--candidate-topk-blocks", type=int, default=2048)
    p.add_argument("--candidate-block-size", type=int, default=8)
    p.add_argument("--index-topk", type=int, default=512)
    p.add_argument(
        "--max-logits-mb",
        type=int,
        default=512,
        help="VLLM_SPARSE_INDEXER_MAX_LOGITS_MB.",
    )
    p.add_argument(
        "--max-prefill-buffer-size",
        type=int,
        default=1 << 20,
        help="N constraint of the chunker (max_total_seq_len).",
    )
    p.add_argument(
        "--compress-ratio",
        type=int,
        default=1,
        help="indexer group ratio, used ONLY to report the raw-token "
        "equivalent of T. --lengths are the compressed workspace "
        "length T itself; set this to the target layer's ratio "
        "(1 or 2 for V4.1) if you want the raw-equivalent field.",
    )
    # benchmark knobs
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)
    # synthetic data
    p.add_argument(
        "--k-std",
        type=float,
        default=1.0,
        help="std of the synthetic pre-quant K vectors.",
    )
    p.add_argument("--q-std", type=float, default=1.0)
    p.add_argument(
        "--wide-factor",
        type=float,
        default=512.0,
        help="amplify every 8th K row / Q head-block by this before "
        "the deliberate-wide-scale E2/E3 runs.",
    )
    p.add_argument("--wide-stride", type=int, default=8)
    p.add_argument(
        "--workspace-pt",
        type=str,
        default=None,
        help="optional real workspace dump (.pt). Keys: k_values, "
        "k_scales and optionally q_values, q_scale, weights. "
        "A dict keyed by 'T' is accepted.",
    )
    # E3 sizing
    p.add_argument("--e3-max-t", type=int, default=65536)
    p.add_argument("--e3-max-rows", type=int, default=64)
    p.add_argument("--e3-col-chunk", type=int, default=2048)
    p.add_argument("--skip-dense-e3", action="store_true", default=False)
    return p


# ===========================================================================
# Pure-python chunker plan (mirrors _split_indexer_prefill_chunks for one req)
# ===========================================================================


def min_split_seq_len(args) -> int:
    """SM90 ``DeepseekV41SparseIndexerMetadataBuilder.min_split_seq_len``.

    ``num_sparse_cols = num_sparse_blocks * sparse_block_kv`` with
    ``num_sparse_blocks = candidate_topk_blocks * (cbs // sparse_block_kv)``,
    and ``sparse_block_kv`` is the largest of (16, 8) dividing ``cbs``
    (``pick_sparse_block_kv``).  Hence ``num_sparse_cols`` simplifies to
    ``candidate_topk_blocks * cbs``.  SM90 logits are fp32, so the item size is
    4 and ``min_split_seq_len = (num_sparse_cols * 4 + 3) // 4``.
    """
    num_sparse_cols = args.candidate_topk_blocks * args.candidate_block_size
    return (num_sparse_cols * 4 + 3) // 4


def chunk_plan(args, T: int, rows: int) -> dict:
    """How the prefill chunker would split a single request of this shape."""
    ms = min_split_seq_len(args)
    max_logits_bytes = args.max_logits_mb * 1024 * 1024
    max_logits_elems = max_logits_bytes // 4
    n_budget = max(T, ms)  # _prefill_split_seq_lens clamps seq_lens to ms
    rows_per_chunk = max(1, max_logits_elems // max(1, n_budget))
    n_chunks = (rows + rows_per_chunk - 1) // rows_per_chunk
    n_ok = args.max_prefill_buffer_size >= T
    return {
        "min_split_seq_len": ms,
        "n_budget": n_budget,
        "rows_per_chunk": rows_per_chunk,
        "num_chunks": n_chunks,
        "n_constraint_ok": bool(n_ok),
        "logits_constraint_ok": bool(rows <= rows_per_chunk),
    }


# ===========================================================================
# Pure-torch reference maths (no Triton, no repo import)
# ===========================================================================


def _require_torch():
    import torch

    return torch


def e2m1_decode_codes(torch, codes):
    """Bit-exact E2M1 -> fp32 (magnitude table, sign from bit 3)."""
    mag = (codes & 0x7).to(torch.int64)
    sign = (codes & 0x8) != 0
    table = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=codes.device)
    vals = table[mag]
    return torch.where(sign, -vals, vals)


def e2m1_encode_rne(torch, x):
    """Round-to-nearest-even E2M1 code for ``x`` in [-6, 6] (matches the kernel).

    Mirrors ``_fp32_to_e2m1_code_rne``: count crossed thresholds, then push an
    odd index sitting exactly on a halfway point back to the even one.
    """
    ax = torch.clamp(x.abs(), max=6.0)
    thresholds = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    boundary = torch.zeros_like(ax, dtype=torch.bool)
    for t in thresholds:
        idx = idx + (ax >= t).to(torch.uint8)
        boundary = boundary | (ax == t)
    idx = torch.where(boundary & ((idx & 1) == 1), idx - 1, idx)
    sign = ((x < 0) & (idx != 0)).to(torch.uint8)
    return (idx & 0x0F) | (sign << 3)


def mxfp4_quantize(torch, x):
    """Faithful torch port of the production MXFP4 quantizer.

    ``x`` is ``[..., 128]`` bf16/fp32.  Returns ``(values [..., 64] uint8,
    scales [..., 4] uint8)`` with byte ``i`` holding element ``2i`` (low) /
    ``2i+1`` (high) and scale byte ``i // 16`` feeding both, exactly the cache
    layout the SM90 kernels decode.
    """
    assert x.shape[-1] == HEAD_DIM, x.shape
    lead = x.shape[:-1]
    x = x.float().reshape(*lead, HEAD_DIM // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    amax = torch.maximum(even.abs().amax(dim=-1), odd.abs().amax(dim=-1))
    amax = torch.clamp(amax, min=6.0 * (2.0**-126))
    log2_ratio = torch.ceil(torch.log2(amax * (1.0 / 6.0)))
    log2_ratio = torch.clamp(log2_ratio, -127.0, 127.0)
    scale = torch.exp2(log2_ratio)
    inv = 1.0 / scale
    code_even = e2m1_encode_rne(torch, even * inv[..., None])
    code_odd = e2m1_encode_rne(torch, odd * inv[..., None])
    packed = (code_even & 0x0F) | ((code_odd & 0x0F) << 4)
    values = packed.reshape(*lead, HALF_D)
    scales = (log2_ratio + 127.0).to(torch.uint8)
    return values, scales


def decode_workspace(torch, k_values, k_scales):
    """``(T, 64)`` packed + ``(T, 4)`` UE8M0 -> ``(T, 128)`` fp32."""
    low = e2m1_decode_codes(torch, k_values & 0x0F)
    high = e2m1_decode_codes(torch, (k_values >> 4) & 0x0F)
    i = torch.arange(HALF_D, device=k_values.device)
    scale = torch.exp2(k_scales[..., i // 16].float() - 127.0)
    out = torch.empty(
        (*k_values.shape[:-1], HEAD_DIM), dtype=torch.float32, device=k_values.device
    )
    out[..., 0::2] = low * scale
    out[..., 1::2] = high * scale
    return out


def decode_q(torch, q_values, q_scale_bytes):
    """``(rows, H, 64)`` packed + ``(rows, H, 4)`` UE8M0 -> ``(rows, H, 128)``."""
    low = e2m1_decode_codes(torch, q_values & 0x0F)
    high = e2m1_decode_codes(torch, (q_values >> 4) & 0x0F)
    i = torch.arange(HALF_D, device=q_values.device)
    scale = torch.exp2(q_scale_bytes[..., i // 16].float() - 127.0)
    out = torch.empty(
        (*q_values.shape[:-1], HEAD_DIM), dtype=torch.float32, device=q_values.device
    )
    out[..., 0::2] = low * scale
    out[..., 1::2] = high * scale
    return out


def to_e4m3(torch, x):
    """Clamp to +/-448 then round to E4M3 (RN-even)."""
    return torch.clamp(x, -FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


def e4m3_roundtrip(torch, x):
    return to_e4m3(torch, x).float()


# ===========================================================================
# Synthetic workspaces / compact inputs
# ===========================================================================


@dataclass
class SyntheticBundle:
    k_values: object
    k_scales: object
    T: int
    tag: str


def synth_k_workspace(torch, T, *, seed, k_std, wide, wide_factor, wide_stride, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(T, HEAD_DIM, generator=gen, dtype=torch.float32) * k_std
    if wide:
        xb = x.view(T, HEAD_DIM // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
        rows = torch.arange(T) % max(1, wide_stride) == 0
        xb[rows] = xb[rows] * wide_factor
        x = xb.reshape(T, HEAD_DIM)
    x = x.to(torch.bfloat16).to(device)
    vals, scales = mxfp4_quantize(torch, x)
    return SyntheticBundle(
        k_values=vals.contiguous(),
        k_scales=scales.contiguous(),
        T=T,
        tag="wide" if wide else "realistic",
    )


def synth_q(torch, rows, heads, *, seed, q_std, wide, wide_factor, wide_stride, device):
    gen = torch.Generator(device="cpu").manual_seed(seed + 100000)
    x = torch.randn(rows, heads, HEAD_DIM, generator=gen, dtype=torch.float32) * q_std
    if wide:
        xb = x.view(rows, heads, HEAD_DIM // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
        hb = torch.arange(heads) % max(1, wide_stride) == 0
        xb[:, hb] = xb[:, hb] * wide_factor
        x = xb.reshape(rows, heads, HEAD_DIM)
    x = x.to(torch.bfloat16).to(device)
    vals, scales = mxfp4_quantize(torch, x)
    q_scale_bytes = scales.reshape(rows, heads, SCALE_BYTES).contiguous()
    q_scale_i32 = q_scale_bytes.view(torch.int32).reshape(rows, heads).contiguous()
    return vals.contiguous(), q_scale_i32, q_scale_bytes


def make_candidate_blocks(torch, rows, T, args, device):
    """``[rows, K]`` int32 request-local candidate block ids.

    Blocks consumed in ascending order from 0, padded with -1 once the gathered
    workspace runs out (which is what the publisher emits for a short context).
    """
    K = args.candidate_topk_blocks
    cbs = args.candidate_block_size
    n_valid = min(K, max(1, T // cbs))
    base = torch.arange(K, dtype=torch.int32)
    base = torch.where(base < n_valid, base, torch.full_like(base, -1))
    return base.reshape(1, -1).repeat(rows, 1).to(device)


def make_row_bounds(torch, rows, T, device):
    """Per-row ``(ks, ke)`` in the gathered workspace: a cold-prefill ramp."""
    r = torch.arange(rows, dtype=torch.int64)
    ke = torch.div((r + 1) * T + rows - 1, rows, rounding_mode="floor")
    ke = torch.clamp(ke, 1, T).to(torch.int32)
    ks = torch.zeros(rows, dtype=torch.int32)
    return ks.to(device), ke.to(device)


# ===========================================================================
# E1: timing
# ===========================================================================


def _event_bench(torch, fn, warmup, repeat):
    for _ in range(max(0, warmup)):
        fn()
    torch.accelerator.synchronize()
    times = []
    for _ in range(max(1, repeat)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.accelerator.synchronize()
        times.append(float(start.elapsed_time(end)))
    times.sort()
    n = len(times)
    return {
        "n": n,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "min_ms": times[0],
        "p10_ms": times[max(0, int(math.floor(0.10 * (n - 1))))],
        "p90_ms": times[min(n - 1, int(math.ceil(0.90 * (n - 1))))],
        "std_ms": statistics.pstdev(times) if n > 1 else 0.0,
        "all_ms": times,
    }


def _empty_logits(torch, rows, width, device):
    return torch.empty((rows, width), dtype=torch.float32, device=device)


def _compact_topk(
    torch, logits, candidate_blocks, cbs, row_ks, row_ke, k, out, chunk_rows=64
):
    """Faithful replica of ``SparseMQAIndexer._prefill_candidate_topk``."""
    rows, width = logits.shape
    out.fill_(-1)
    if rows == 0 or width == 0:
        return
    k = min(k, width, out.shape[1])
    cols = torch.arange(width, device=logits.device)
    block_col = cols // cbs
    within = (cols % cbs).to(torch.int64)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        cand = candidate_blocks[start:end, block_col]
        logical = cand.to(torch.int64) * cbs + within
        valid = (cand >= 0) & (
            (row_ks[start:end].to(torch.int64)[:, None] + logical)
            < row_ke[start:end].to(torch.int64)[:, None]
        )
        masked = logits[start:end].masked_fill(~valid, float("-inf"))
        values, indices = torch.topk(masked, k, dim=-1)
        selected = torch.gather(logical, 1, indices)
        selected = torch.where(
            values > float("-inf"), selected, torch.full_like(selected, -1)
        ).to(torch.int32)
        out[start:end, :k] = selected


def run_e1_gpu(torch, repo, args, results):
    device = args.device
    cbs = args.candidate_block_size
    width = args.candidate_topk_blocks * cbs
    max_logits_elems = (args.max_logits_mb * 1024 * 1024) // 4

    q_cache = {}
    k_bundle_cache = {}
    for seed in args.seeds:
        for rows in args.rows:
            if (seed, rows) not in q_cache:
                q_values, q_scale, _ = synth_q(
                    torch,
                    rows,
                    args.heads,
                    seed=seed,
                    q_std=args.q_std,
                    wide=False,
                    wide_factor=args.wide_factor,
                    wide_stride=args.wide_stride,
                    device=device,
                )
                weights = torch.randn(
                    rows,
                    args.heads,
                    dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(seed + 7),
                ).to(device)
                q_cache[(seed, rows)] = (q_values, q_scale, weights)
        for T in args.lengths:
            if not args.no_dense and T not in k_bundle_cache:
                k_bundle_cache[T] = synth_k_workspace(
                    torch,
                    T,
                    seed=seed,
                    k_std=args.k_std,
                    wide=False,
                    wide_factor=1.0,
                    wide_stride=args.wide_stride,
                    device=device,
                )

    for seed in args.seeds:
        for T in args.lengths:
            for rows in args.rows:
                q_values, q_scale, weights = q_cache[(seed, rows)]
                entry = {
                    "seed": seed,
                    "T": T,
                    "rows": rows,
                    "requested_rows": rows,
                    "raw_tokens_equivalent": T * args.compress_ratio,
                    "chunker": chunk_plan(args, T, rows),
                    "compact_width": width,
                }
                try:
                    entry.update(
                        _e1_one_compact(
                            torch,
                            repo,
                            args,
                            device,
                            T,
                            rows,
                            q_values,
                            q_scale,
                            weights,
                            width,
                            cbs,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                if not args.no_dense:
                    try:
                        entry.update(
                            _e1_one_dense(
                                torch,
                                repo,
                                args,
                                device,
                                T,
                                rows,
                                q_values,
                                q_scale,
                                weights,
                                max_logits_elems,
                                k_bundle_cache[T],
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        entry["dense_error"] = f"{type(exc).__name__}: {exc}"
                results["e1"].append(entry)


def _e1_one_compact(
    torch, repo, args, device, T, rows, q_values, q_scale, weights, width, cbs
):
    # --- K gather workspace (exactly _gather_workspace_shapes, MXFP4) ---
    (vshape, vdtype), (sshape, sdtype) = repo.gather_workspace_shapes(
        T, args.head_dim, torch.float8_e4m3fn, use_fp4_cache=True
    )
    k_quant_full = torch.empty(vshape, dtype=vdtype, device=device)
    k_scale_full = torch.empty(sshape, dtype=sdtype, device=device)
    # Fill the paged cache and gather it.  The gather is measured through the
    # production helper; the fill happens once, outside any timing region.
    page_size = args.page_size
    num_pages = (T + page_size - 1) // page_size
    cache = _make_packed_cache(torch, num_pages, page_size, seed=T, device=device)
    block_table = torch.arange(num_pages, dtype=torch.int32).reshape(1, -1).to(device)
    cu_seq = torch.tensor([0, T], dtype=torch.int32, device=device)
    chunk = repo.PrefillChunk(
        max_local_total_seq_lens=T,
        skip_kv_gather=False,
        local_total_seq_lens=T,
        block_table=block_table,
        local_cu_seq_lens=cu_seq,
    )

    gather_fn = lambda: repo.gather_prefill_chunk_k(
        cache, k_quant_full, k_scale_full, chunk
    )
    gather_ms = _event_bench(torch, gather_fn, args.warmup, args.repeat)
    k_quant, k_scale = gather_fn()

    ks, ke = make_row_bounds(torch, rows, T, device)
    candidate_blocks = make_candidate_blocks(torch, rows, T, args, device)

    def logits_fn():
        return repo.prefill_chunk(
            q_values,
            q_scale,
            k_quant,
            k_scale,
            weights,
            ks,
            ke,
            candidate_blocks,
            cbs,
        )

    logits_ms = _event_bench(torch, logits_fn, args.warmup, args.repeat)
    out = torch.full(
        (rows, min(args.index_topk, width)), -1, dtype=torch.int32, device=device
    )

    def topk_fn():
        repo.prefill_topk(logits_fn(), candidate_blocks, cbs, ks, ke, out)

    topk_ms = _event_bench(torch, topk_fn, args.warmup, args.repeat)

    def full_fn():
        kq, ksc = gather_fn()
        lg = repo.prefill_chunk(
            q_values, q_scale, kq, ksc, weights, ks, ke, candidate_blocks, cbs
        )
        repo.prefill_topk(lg, candidate_blocks, cbs, ks, ke, out)

    total = _event_bench(torch, full_fn, args.warmup, args.repeat)
    combined = {
        k: total[k] - gather_ms[k]
        if k != "all_ms"
        else [t - g for t, g in zip(total["all_ms"], gather_ms["all_ms"])]
        for k in total
    }
    return {
        "compact": {
            "gather_ms": gather_ms,
            "logits_ms": logits_ms,
            "topk_ms": topk_ms,
            "logits_plus_topk_ms": combined,
            "total_ms": total,
            "logits_fraction_of_total": (
                logits_ms["median_ms"] / total["median_ms"]
                if total["median_ms"]
                else None
            ),
            "derived": {
                "t_over_rows": T / rows if rows else None,
                "gather_over_logits": (
                    gather_ms["median_ms"] / logits_ms["median_ms"]
                    if logits_ms["median_ms"]
                    else None
                ),
                # Extra O(T) cost an FP8 pre-decode would add, as a fraction of
                # the per-row logits it would replace: the pre-decode touches
                # T*HALF_D packed bytes once; the current kernel decodes that
                # same work `rows` times.  Below 1.0 the pre-decode can pay off.
                "predecode_over_logits_lower_bound": (
                    (T / (rows * width)) if rows and width else None
                ),
            },
        }
    }


def _e1_one_dense(
    torch,
    repo,
    args,
    device,
    T,
    rows,
    q_values,
    q_scale,
    weights,
    max_logits_elems,
    bundle,
):
    # The dense kernel's output is [rows, T] fp32.  The chunker would split the
    # requested rows down to the budget, so measure the production-sized chunk
    # and record both the requested and the measured row count.
    rows_measured = min(rows, max(1, max_logits_elems // max(1, T)))
    q = q_values[:rows_measured]
    qs = q_scale[:rows_measured]
    w = weights[:rows_measured]
    ks, ke = make_row_bounds(torch, rows_measured, T, device)
    k_values = bundle.k_values[:T].contiguous()
    k_scales = bundle.k_scales[:T].contiguous()

    def fn():
        return repo.workspace_logits(q, qs, w, k_values, k_scales, ks, ke, None, 0)

    dense_ms = _event_bench(torch, fn, args.warmup, args.repeat)
    return {
        "dense": {
            "rows_measured": rows_measured,
            "width": T,
            "logits_ms": dense_ms,
        }
    }


def _make_packed_cache(torch, num_pages, page_size, seed, device):
    """A paged MXFP4 indexer cache in the documented segregated layout.

    Page flat bytes are ``[page_size*64 payload][page_size*4 UE8M0]``; the 3D
    shape is nominal (writers/readers index payload at ``off*64`` and scales at
    ``page_size*64 + off*4`` from the page base).  Values are drawn with a
    realistic UE8M0 exponent range (near unity), matching the quantizer output
    for RMSNorm'd keys.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed + 4242)
    payload = torch.randint(
        0, 256, (num_pages, page_size * HALF_D), generator=gen, dtype=torch.uint8
    )
    exp = torch.randint(
        123,
        127,
        (num_pages, page_size * SCALE_BYTES),
        generator=gen,
        dtype=torch.uint8,
    )
    flat = torch.cat([payload, exp], dim=1)
    return (
        flat.reshape(num_pages, page_size, HALF_D + SCALE_BYTES).contiguous().to(device)
    )


# ===========================================================================
# E2: clamp / amax census
# ===========================================================================


def _census_tensor(torch, values_fp32_2d, scales_u8_2d, block=MXFP4_BLOCK_SIZE):
    """Census over a chunk of ``[N, D]`` decoded fp32 and its UE8M0 bytes."""
    absv = values_fp32_2d.abs()
    nonzero = absv > 0
    n = absv.numel()
    n_nonzero = int(nonzero.sum().item())
    amax = float(absv.max().item()) if n else 0.0
    n_gt = int((absv > FP8_E4M3_MAX).sum().item())
    n_flush = int((nonzero & (absv < E4M3_MIN_SUBNORMAL)).sum().item())
    n_subn = int((nonzero & (absv < E4M3_MIN_NORMAL)).sum().item())
    # per-block decoded amax
    nb = values_fp32_2d.shape[-1] // block
    ablocks = absv.reshape(-1, nb, block).amax(dim=-1)
    exp_hist = torch.bincount(scales_u8_2d.reshape(-1).to(torch.int64), minlength=256)
    return {
        "n_elements": n,
        "n_nonzero": n_nonzero,
        "amax": amax,
        "n_gt_fp8_max": n_gt,
        "frac_gt_fp8_max": n_gt / n if n else 0.0,
        "n_nonzero_lt_min_subnormal": n_flush,
        "frac_nonzero_lt_min_subnormal": n_flush / n_nonzero if n_nonzero else 0.0,
        "n_nonzero_lt_min_normal": n_subn,
        "frac_nonzero_lt_min_normal": n_nonzero and n_subn / n_nonzero or 0.0,
        "block_amax_max": float(ablocks.max().item()) if ablocks.numel() else 0.0,
        "block_amax_mean": float(ablocks.mean().item()) if ablocks.numel() else 0.0,
        "exp_hist": {int(i): int(c) for i, c in enumerate(exp_hist.tolist()) if c},
    }


def _census_chunked(
    torch, values, scales, decode_fn, block=MXFP4_BLOCK_SIZE, row_chunk=65536
):
    totals = None
    for start in range(0, values.shape[0], row_chunk):
        end = min(start + row_chunk, values.shape[0])
        dec = decode_fn(values[start:end], scales[start:end])
        dec2 = dec.reshape(-1, dec.shape[-1])
        sc2 = scales[start:end].reshape(-1, scales.shape[-1])
        part = _census_tensor(torch, dec2, sc2, block=block)
        if totals is None:
            totals = part
        else:
            _merge_census(totals, part)
    return (
        totals
        if totals is not None
        else _census_tensor(
            torch,
            torch.zeros(0, HEAD_DIM),
            torch.zeros(0, SCALE_BYTES, dtype=torch.uint8),
        )
    )


def _merge_census(a, b):
    a_n = a["n_elements"]
    b_n = b["n_elements"]
    n = a_n + b_n
    for key in (
        "n_elements",
        "n_nonzero",
        "n_gt_fp8_max",
        "n_nonzero_lt_min_subnormal",
        "n_nonzero_lt_min_normal",
    ):
        a[key] += b[key]
    a["amax"] = max(a["amax"], b["amax"])
    a["frac_gt_fp8_max"] = a["n_gt_fp8_max"] / n if n else 0.0
    a["frac_nonzero_lt_min_subnormal"] = (
        a["n_nonzero_lt_min_subnormal"] / a["n_nonzero"] if a["n_nonzero"] else 0.0
    )
    a["frac_nonzero_lt_min_normal"] = (
        a["n_nonzero_lt_min_normal"] / a["n_nonzero"] if a["n_nonzero"] else 0.0
    )
    a["block_amax_max"] = max(a["block_amax_max"], b["block_amax_max"])
    a["block_amax_mean"] = statistics.fmean(
        [a["block_amax_mean"], b["block_amax_mean"]]
    )
    for k, v in b["exp_hist"].items():
        a["exp_hist"][k] = a["exp_hist"].get(k, 0) + v


def run_e2(torch, args, results, bundle, q_bundle):
    k_values, k_scales = bundle.k_values, bundle.k_scales
    census_k = _census_chunked(
        torch,
        k_values,
        k_scales,
        lambda v, s: decode_workspace(torch, v, s),
    )
    census_q = None
    if q_bundle is not None:
        qv, _, qsb = q_bundle
        # census Q on its 32-element blocks; scales are per head
        census_q = _census_chunked(
            torch,
            qv.reshape(-1, HALF_D),
            qsb.reshape(-1, SCALE_BYTES),
            lambda v, s: decode_workspace(torch, v, s).reshape(-1, HEAD_DIM),
        )
    return {"k": census_k, "q": census_q}


# ===========================================================================
# E3: numeric + selection equivalence
# ===========================================================================


def _score_chain(torch, acc, weights):
    """``acc [rows, H, width]`` fp32 -> logits [rows, width] fp32 (bf16 chain)."""
    s = acc.to(torch.bfloat16).to(torch.float32)
    s = torch.clamp(s, min=0.0)
    s = (s * weights.to(torch.float32)[:, :, None]).to(torch.bfloat16).to(torch.float32)
    return s.sum(dim=1).to(torch.bfloat16).to(torch.float32)


def reference_score(
    torch,
    q_values,
    q_scale_bytes,
    k_values,
    k_scales,
    weights,
    candidate_blocks,
    cbs,
    row_ks,
    row_ke,
    *,
    mode,
    accum,
    col_chunk=2048,
):
    """Reference-only FP8 (or bf16) compact scoring, pure torch.

    ``mode`` is ``"fp8"`` (decode -> clamp -> E4M3) or ``"bf16"`` (decode ->
    bf16, i.e. the current kernel operands).  ``accum`` is ``"fp32"`` or
    ``"fp64"`` for the emulated ``tl.dot``.
    """
    rows, heads = q_values.shape[:2]
    T = k_values.shape[0]
    width = candidate_blocks.shape[1] * cbs
    dtype = torch.float64 if accum == "fp64" else torch.float32
    q = decode_q(torch, q_values, q_scale_bytes)
    if mode == "fp8":
        q = e4m3_roundtrip(torch, q)
    elif mode == "bf16":
        q = q.to(torch.bfloat16).float()
    else:
        raise ValueError(mode)

    cols = torch.arange(width, device=k_values.device)
    block_col = cols // cbs
    within = (cols % cbs).to(torch.int64)

    logits = torch.empty((rows, width), dtype=torch.float32, device=k_values.device)
    for c0 in range(0, width, col_chunk):
        c1 = min(c0 + col_chunk, width)
        cand = candidate_blocks[:, block_col[c0:c1]]
        logical = cand.to(torch.int64) * cbs + within[c0:c1]
        krow = row_ks[:, None].to(torch.int64) + logical
        valid = (cand >= 0) & (krow < row_ke[:, None].to(torch.int64))
        krow_c = krow.clamp(0, T - 1)
        kv = k_values[krow_c.reshape(-1)].reshape(rows, c1 - c0, HALF_D)
        ks = k_scales[krow_c.reshape(-1)].reshape(rows, c1 - c0, SCALE_BYTES)
        k = decode_workspace(torch, kv, ks)
        k = e4m3_roundtrip(torch, k) if mode == "fp8" else k.to(torch.bfloat16).float()
        acc = torch.einsum("rhd,rcd->rhc", q.to(dtype), k.to(dtype))
        chunk_logits = _score_chain(torch, acc.to(torch.float32), weights)
        chunk_logits = torch.where(valid, chunk_logits, float("-inf"))
        logits[:, c0:c1] = chunk_logits
    return logits


def _error_stats(torch, a, b):
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.any()):
        return {"n_finite": 0}
    d = (a[finite].float() - b[finite].float()).abs()
    exact = (
        (a[finite].to(torch.bfloat16) == b[finite].to(torch.bfloat16)).float().mean()
    )
    ref_absmax = float(b[finite].float().abs().max().item())
    max_abs = float(d.max().item())
    return {
        "n_finite": int(finite.sum().item()),
        "max_abs_err": max_abs,
        "mean_abs_err": float(d.mean().item()),
        "max_rel_err": max_abs / ref_absmax if ref_absmax else 0.0,
        "ref_absmax": ref_absmax,
        "frac_bf16_identical": float(exact.item()),
        "n_exact": int((d == 0).sum().item()),
    }


def _topk_overlap(
    torch, logits_a, logits_b, candidate_blocks, cbs, row_ks, row_ke, k, topk_impl
):
    width = logits_a.shape[1]
    k = min(k, width)
    a = torch.full(
        (logits_a.shape[0], k), -1, dtype=torch.int32, device=logits_a.device
    )
    b = torch.full_like(a, -1)
    topk_impl(logits_a, candidate_blocks, cbs, row_ks, row_ke, a)
    topk_impl(logits_b, candidate_blocks, cbs, row_ks, row_ke, b)
    match = 0
    total = 0
    for r in range(a.shape[0]):
        sa = set(int(x) for x in a[r].tolist() if x >= 0)
        sb = set(int(x) for x in b[r].tolist() if x >= 0)
        match += len(sa & sb)
        total += max(len(sa), len(sb)) or 0
    return {
        "k": k,
        "matched": match,
        "total": total,
        "overlap_frac": (match / total) if total else 1.0,
    }


def run_e3(torch, repo, args, bundle, q_bundle, topk_impl):
    device = args.device
    cbs = args.candidate_block_size
    width = args.candidate_topk_blocks * cbs
    out_rows = []
    for seed in args.seeds:
        for T in args.lengths:
            if args.e3_max_t < T:
                continue
            for rows in args.rows:
                if rows > args.e3_max_rows:
                    continue
                k_values = bundle.k_values[:T]
                k_scales = bundle.k_scales[:T]
                if q_bundle is None:
                    q_values, q_scale_i32, q_scale_bytes = synth_q(
                        torch,
                        rows,
                        args.heads,
                        seed=seed,
                        q_std=args.q_std,
                        wide=False,
                        wide_factor=args.wide_factor,
                        wide_stride=args.wide_stride,
                        device=device,
                    )
                else:
                    q_values, q_scale_bytes = q_bundle[0][:rows], q_bundle[2][:rows]
                    q_scale_i32 = (
                        q_scale_bytes.view(torch.int32)
                        .reshape(rows, args.heads)
                        .contiguous()
                    )
                weights = torch.randn(
                    rows,
                    args.heads,
                    dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(seed + 7),
                ).to(device)
                ks, ke = make_row_bounds(torch, rows, T, device)
                cand = make_candidate_blocks(torch, rows, T, args, device)
                entry = {
                    "seed": seed,
                    "T": T,
                    "rows": rows,
                    "width": width,
                    "kernel_available": repo is not None,
                }
                try:
                    if repo is not None:
                        bf16_logits = repo.prefill_chunk(
                            q_values,
                            q_scale_i32,
                            k_values,
                            k_scales,
                            weights,
                            ks,
                            ke,
                            cand,
                            cbs,
                        )
                        bf16_src = "triton_kernel"
                    else:
                        bf16_logits = reference_score(
                            torch,
                            q_values,
                            q_scale_bytes,
                            k_values,
                            k_scales,
                            weights,
                            cand,
                            cbs,
                            ks,
                            ke,
                            mode="bf16",
                            accum="fp32",
                            col_chunk=args.e3_col_chunk,
                        )
                        bf16_src = "torch_reference"
                    fp8_fp32 = reference_score(
                        torch,
                        q_values,
                        q_scale_bytes,
                        k_values,
                        k_scales,
                        weights,
                        cand,
                        cbs,
                        ks,
                        ke,
                        mode="fp8",
                        accum="fp32",
                        col_chunk=args.e3_col_chunk,
                    )
                    fp8_fp64 = reference_score(
                        torch,
                        q_values,
                        q_scale_bytes,
                        k_values,
                        k_scales,
                        weights,
                        cand,
                        cbs,
                        ks,
                        ke,
                        mode="fp8",
                        accum="fp64",
                        col_chunk=args.e3_col_chunk,
                    )
                    entry["bf16_source"] = bf16_src
                    entry["fp8_fp32_vs_bf16"] = _error_stats(
                        torch, fp8_fp32, bf16_logits
                    )
                    entry["fp8_fp64_vs_bf16"] = _error_stats(
                        torch, fp8_fp64, bf16_logits
                    )
                    entry["fp8_fp32_vs_fp64"] = _error_stats(torch, fp8_fp32, fp8_fp64)
                    entry["topk_overlap_fp8_fp32"] = _topk_overlap(
                        torch,
                        bf16_logits,
                        fp8_fp32,
                        cand,
                        cbs,
                        ks,
                        ke,
                        args.index_topk,
                        topk_impl,
                    )
                    entry["topk_overlap_fp8_fp64"] = _topk_overlap(
                        torch,
                        bf16_logits,
                        fp8_fp64,
                        cand,
                        cbs,
                        ks,
                        ke,
                        args.index_topk,
                        topk_impl,
                    )
                except Exception as exc:  # noqa: BLE001
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                out_rows.append(entry)
    return out_rows


# ===========================================================================
# Repo loading (lazy; only for the GPU path)
# ===========================================================================


@dataclass
class Repo:
    gather_workspace_shapes: object
    gather_prefill_chunk_k: object
    prefill_chunk: object
    workspace_logits: object
    prefill_topk: object
    PrefillChunk: object


def load_repo(args):
    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from types import SimpleNamespace

    from vllm import _custom_ops as _ops  # noqa: F401
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_workspace_index_logits,
    )
    from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
        sm90_sparse_mqa_logits_prefill_chunk,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import (
        _gather_workspace_shapes,
    )
    from vllm.model_executor.layers.sparse_mqa_indexer import (
        SparseMQAIndexer,
        _gather_prefill_chunk_k,
    )

    def make_chunk(
        max_local_total_seq_lens,
        skip_kv_gather,
        local_total_seq_lens,
        block_table,
        local_cu_seq_lens,
    ):
        return SimpleNamespace(
            max_local_total_seq_lens=max_local_total_seq_lens,
            skip_kv_gather=skip_kv_gather,
            local_total_seq_lens=local_total_seq_lens,
            block_table=block_table,
            local_cu_seq_lens=local_cu_seq_lens,
        )

    return Repo(
        gather_workspace_shapes=_gather_workspace_shapes,
        gather_prefill_chunk_k=_gather_prefill_chunk_k,
        prefill_chunk=sm90_sparse_mqa_logits_prefill_chunk,
        workspace_logits=sm90_fp4_workspace_index_logits,
        prefill_topk=SparseMQAIndexer._prefill_candidate_topk,
        PrefillChunk=make_chunk,
    )


def load_workspace_pt(torch, path, args, device):
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "k_values" not in data:
        keys = sorted(data.keys())
        raise SystemExit(
            f"--workspace-pt {path}: expected keys k_values/k_scales, got {keys}"
        )
    k_values = data["k_values"].to(device)
    k_scales = data["k_scales"].to(device)
    T = k_values.shape[0]
    q_bundle = None
    if "q_values" in data and "q_scale" in data:
        qv = data["q_values"].to(device)
        qs = data["q_scale"].to(device)
        q_bytes = qs.view(torch.uint8).reshape(qv.shape[0], qv.shape[1], SCALE_BYTES)
        q_bundle = (qv, qs, q_bytes.contiguous())
    return SyntheticBundle(
        k_values=k_values, k_scales=k_scales, T=T, tag="dump"
    ), q_bundle


# ===========================================================================
# JSON / reporting
# ===========================================================================


def _shape_str(a):
    if isinstance(a, dict):
        return f"{a.get('median_ms', float('nan')):.3f}"
    return "n/a"


def print_human_summary(results):
    print("\n" + "=" * 100)
    print("DSV4.1 SM90 FP8 PREFILL PROBE - HUMAN SUMMARY")
    print("=" * 100)
    cfg = results["config"]
    print(
        f"device={results.get('device')} heads={cfg['heads']} "
        f"cbs={cfg['candidate_block_size']} "
        f"candidate_topk_blocks={cfg['candidate_topk_blocks']} "
        f"index_topk={cfg['index_topk']} "
        f"min_split_seq_len={cfg['min_split_seq_len']} "
        f"max_logits_mb={cfg['max_logits_mb']}"
    )
    if results.get("dry_run"):
        print("DRY RUN: shapes planned, no torch/CUDA touched.")
        print(
            f"{'T':>8} {'rows':>6} {'raw~tok':>10} {'rows/chunk':>11} {'chunks':>7} "
            f"{'compact_w':>10} {'dense_ok':>9}"
        )
        for row in results["plan"]:
            print(
                f"{row['T']:>8} {row['rows']:>6} {row['raw_tokens_equivalent']:>10} "
                f"{row['chunker']['rows_per_chunk']:>11} "
                f"{row['chunker']['num_chunks']:>7} "
                f"{row['compact_width']:>10} "
                f"{str(row['chunker']['logits_constraint_ok']):>9}"
            )
        return

    e1 = results.get("e1", [])
    if e1:
        print("\nE1 - timing split (median ms; gather / logits / topk / total)")
        print(
            f"{'seed':>4} {'T':>8} {'rows':>6} {'gather':>9} {'logits':>9} "
            f"{'topk':>9} {'total':>9} {'logit%':>7} {'dense_ms':>9} {'dense_r':>7}"
        )
        for row in e1:
            c = row.get("compact", {})
            d = row.get("dense", {})
            lg = c.get("logits_ms", {})
            tot = c.get("total_ms", {})
            print(
                f"{row['seed']:>4} {row['T']:>8} {row['rows']:>6} "
                f"{_shape_str(c.get('gather_ms')):>9} {_shape_str(lg):>9} "
                f"{_shape_str(c.get('topk_ms')):>9} {_shape_str(tot):>9} "
                f"{(c.get('logits_fraction_of_total') or 0) * 100:>6.1f}% "
                f"{_shape_str(d.get('logits_ms')):>9} {d.get('rows_measured', '-'):>7}"
            )

    for tag in ("realistic", "wide"):
        e2 = results.get("e2", {}).get(tag)
        if not e2:
            continue
        print(f"\nE2 - clamp/amax census ({tag})")
        for name in ("k", "q"):
            c = e2.get(name)
            if not c:
                continue
            print(
                f"  {name}: amax={c['amax']:.4g}  >448={c['n_gt_fp8_max']} "
                f"({c['frac_gt_fp8_max'] * 100:.4f}%)  "
                f"flush<2^-9={c['n_nonzero_lt_min_subnormal']} "
                f"({c['frac_nonzero_lt_min_subnormal'] * 100:.4f}% of nonzeros)  "
                f"block_amax(max/mean)={c['block_amax_max']:.4g}/{c['block_amax_mean']:.4g}"
            )

    for tag in ("realistic", "wide"):
        e3 = results.get("e3", {}).get(tag, [])
        if not e3:
            continue
        print(f"\nE3 - numeric + top-k equivalence ({tag})")
        print(
            f"{'T':>8} {'rows':>6} {'bf16_src':>15} {'maxerr':>10} {'meanerr':>10} "
            f"{'bf16match':>10} {'topk_ovl':>9}"
        )
        for row in e3:
            st = row.get("fp8_fp32_vs_bf16", {})
            ov = row.get("topk_overlap_fp8_fp32", {})
            print(
                f"{row['T']:>8} {row['rows']:>6} {row.get('bf16_source', '-'):>15} "
                f"{st.get('max_abs_err', float('nan')):>10.4g} "
                f"{st.get('mean_abs_err', float('nan')):>10.4g} "
                f"{st.get('frac_bf16_identical', float('nan')):>10.4f} "
                f"{ov.get('overlap_frac', float('nan')):>9.4f}"
            )
    print("\nsee JSON for full stats (p10/p90, exponent histograms, fp64 reference).")
    print("=" * 100)


def emit(results, out):
    payload = json.dumps(results, indent=2, sort_keys=True, default=_json_default)
    if out:
        with open(out, "w") as fh:
            fh.write(payload)
            fh.write("\n")
        print(f"\n[wrote JSON summary to {out}]")
    print("\n===== JSON SUMMARY =====")
    print(payload)


def _json_default(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# ===========================================================================
# Main
# ===========================================================================


def _config_dict(args):
    return {
        "lengths": args.lengths,
        "rows": args.rows,
        "seeds": args.seeds,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "page_size": args.page_size,
        "candidate_topk_blocks": args.candidate_topk_blocks,
        "candidate_block_size": args.candidate_block_size,
        "index_topk": args.index_topk,
        "max_logits_mb": args.max_logits_mb,
        "max_prefill_buffer_size": args.max_prefill_buffer_size,
        "compress_ratio": args.compress_ratio,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "min_split_seq_len": min_split_seq_len(args),
        "compact_width": args.candidate_topk_blocks * args.candidate_block_size,
        "workspace_pt": args.workspace_pt,
        "k_std": args.k_std,
        "q_std": args.q_std,
        "wide_factor": args.wide_factor,
        "wide_stride": args.wide_stride,
    }


def _base_results(args):
    return {
        "probe": "dsv41_fp8_prefill_probe",
        "version": 1,
        "git_commit": _git_commit(args),
        "config": _config_dict(args),
        "unverified": [],
    }


def _git_commit(args):
    try:
        import subprocess

        root = args.repo_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        return subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _plan_rows(args):
    rows = []
    for T in args.lengths:
        for r in args.rows:
            rows.append(
                {
                    "T": T,
                    "rows": r,
                    "raw_tokens_equivalent": T * args.compress_ratio,
                    "compact_width": args.candidate_topk_blocks
                    * args.candidate_block_size,
                    "chunker": chunk_plan(args, T, r),
                }
            )
    return rows


def do_dry_run(args):
    results = _base_results(args)
    results.update(
        {
            "dry_run": True,
            "cpu_selftest": False,
            "device": "none",
            "plan": _plan_rows(args),
            "e1": [],
            "e2": {},
            "e3": {},
            "unverified": [
                "GPU measurements are unrun: this is --dry-run.",
            ],
        }
    )
    print_human_summary(results)
    emit(results, args.out)
    return 0


def do_cpu_selftest(args):
    """Tiny pure-torch sanity run of the E2/E3 maths on CPU."""
    torch = _require_torch()
    device = "cpu"
    args = argparse.Namespace(**vars(args))
    args.device = device
    # tiny shapes so the whole thing runs in a second
    tiny_T = min(args.lengths) if args.lengths else 2048
    tiny_T = min(tiny_T, 4096)
    tiny_rows = min(args.rows) if args.rows else 4
    tiny_rows = min(tiny_rows, 8)
    results = _base_results(args)
    results.update(
        {"dry_run": False, "cpu_selftest": True, "device": "cpu", "e1": [], "e3": {}}
    )
    bundle = synth_k_workspace(
        torch,
        tiny_T,
        seed=0,
        k_std=args.k_std,
        wide=False,
        wide_factor=1.0,
        wide_stride=args.wide_stride,
        device=device,
    )
    wide = synth_k_workspace(
        torch,
        tiny_T,
        seed=0,
        k_std=args.k_std,
        wide=True,
        wide_factor=args.wide_factor,
        wide_stride=args.wide_stride,
        device=device,
    )
    q = synth_q(
        torch,
        tiny_rows,
        args.heads,
        seed=0,
        q_std=args.q_std,
        wide=False,
        wide_factor=1.0,
        wide_stride=args.wide_stride,
        device=device,
    )
    q_wide = synth_q(
        torch,
        tiny_rows,
        args.heads,
        seed=0,
        q_std=args.q_std,
        wide=True,
        wide_factor=args.wide_factor,
        wide_stride=args.wide_stride,
        device=device,
    )
    results["e2"] = {
        "realistic": run_e2(torch, args, results, bundle, q),
        "wide": run_e2(torch, args, results, wide, q_wide),
    }
    cbs = args.candidate_block_size

    def run_mode(bundle_, qb, tag):
        out = []
        for T in (tiny_T,):
            for rows in (tiny_rows,):
                k_values = bundle_.k_values[:T]
                k_scales = bundle_.k_scales[:T]
                qv, q_i32, q_bytes = qb
                weights = torch.randn(
                    rows,
                    args.heads,
                    dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(7),
                )
                ks, ke = make_row_bounds(torch, rows, T, device)
                cand = make_candidate_blocks(torch, rows, T, args, device)
                bf16_logits = reference_score(
                    torch,
                    qv,
                    q_bytes,
                    k_values,
                    k_scales,
                    weights,
                    cand,
                    cbs,
                    ks,
                    ke,
                    mode="bf16",
                    accum="fp32",
                    col_chunk=args.e3_col_chunk,
                )
                fp8_fp32 = reference_score(
                    torch,
                    qv,
                    q_bytes,
                    k_values,
                    k_scales,
                    weights,
                    cand,
                    cbs,
                    ks,
                    ke,
                    mode="fp8",
                    accum="fp32",
                    col_chunk=args.e3_col_chunk,
                )
                fp8_fp64 = reference_score(
                    torch,
                    qv,
                    q_bytes,
                    k_values,
                    k_scales,
                    weights,
                    cand,
                    cbs,
                    ks,
                    ke,
                    mode="fp8",
                    accum="fp64",
                    col_chunk=args.e3_col_chunk,
                )
                out.append(
                    {
                        "seed": 0,
                        "T": T,
                        "rows": rows,
                        "width": cand.shape[1] * cbs,
                        "bf16_source": "torch_reference",
                        "fp8_fp32_vs_bf16": _error_stats(torch, fp8_fp32, bf16_logits),
                        "fp8_fp64_vs_bf16": _error_stats(torch, fp8_fp64, bf16_logits),
                        "fp8_fp32_vs_fp64": _error_stats(torch, fp8_fp32, fp8_fp64),
                        "topk_overlap_fp8_fp32": _topk_overlap(
                            torch,
                            bf16_logits,
                            fp8_fp32,
                            cand,
                            cbs,
                            ks,
                            ke,
                            args.index_topk,
                            lambda lg, cb, c, rk, rke, o: _compact_topk(
                                torch, lg, cb, c, rk, rke, args.index_topk, o
                            ),
                        ),
                    }
                )
        return out

    results["e3"] = {
        "realistic": run_mode(bundle, q, "realistic"),
        "wide": run_mode(wide, q_wide, "wide"),
    }
    results["unverified"] = [
        "CPU self-test only: no GPU timing, no Triton kernel, synthetic data.",
    ]
    print_human_summary(results)
    emit(results, args.out)
    return 0


def do_gpu_run(args):
    torch = _require_torch()
    if not torch.cuda.is_available():
        print(
            "ERROR: no CUDA device available; use --dry-run or --cpu-selftest.",
            file=sys.stderr,
        )
        return 2
    name = torch.cuda.get_device_name(args.device)
    cap = torch.cuda.get_device_capability(args.device)
    repo = load_repo(args)
    results = _base_results(args)
    results.update(
        {
            "dry_run": False,
            "cpu_selftest": False,
            "device": args.device,
            "device_name": name,
            "compute_capability": list(cap),
            "e1": [],
            "e2": {},
            "e3": {},
        }
    )
    if cap[0] != 9:
        print(
            f"WARNING: {name} is compute capability {cap}, not SM90/Hopper; "
            "the SM90 fp4 kernel is not validated here.",
            file=sys.stderr,
        )

    # ---- E2/E3 workspaces -------------------------------------------------
    if args.workspace_pt:
        bundle, q_bundle = load_workspace_pt(
            torch, args.workspace_pt, args, args.device
        )
        results["workspace_source"] = args.workspace_pt
        wide_bundle = bundle
        q_wide = q_bundle
    else:
        T_max = max(args.lengths) if args.lengths else max(args.rows)
        seed = args.seeds[0] if args.seeds else 0
        bundle = synth_k_workspace(
            torch,
            T_max,
            seed=seed,
            k_std=args.k_std,
            wide=False,
            wide_factor=1.0,
            wide_stride=args.wide_stride,
            device=args.device,
        )
        wide_bundle = synth_k_workspace(
            torch,
            T_max,
            seed=seed,
            k_std=args.k_std,
            wide=True,
            wide_factor=args.wide_factor,
            wide_stride=args.wide_stride,
            device=args.device,
        )
        q_bundle = synth_q(
            torch,
            min(max(args.rows), args.e3_max_rows),
            args.heads,
            seed=seed,
            q_std=args.q_std,
            wide=False,
            wide_factor=1.0,
            wide_stride=args.wide_stride,
            device=args.device,
        )
        q_wide = synth_q(
            torch,
            min(max(args.rows), args.e3_max_rows),
            args.heads,
            seed=seed,
            q_std=args.q_std,
            wide=True,
            wide_factor=args.wide_factor,
            wide_stride=args.wide_stride,
            device=args.device,
        )
        results["workspace_source"] = "synthetic"

    if not args.skip_e1:
        run_e1_gpu(torch, repo, args, results)

    e2 = {}
    if not args.skip_e2:
        bundles = [("realistic", bundle, q_bundle)]
        if args.workspace_pt is None:
            bundles.append(("wide", wide_bundle, q_wide))
        for tag, b, q in bundles:
            try:
                e2[tag] = run_e2(torch, args, results, b, q)
            except Exception as exc:  # noqa: BLE001
                e2[tag] = {"error": f"{type(exc).__name__}: {exc}"}
    results["e2"] = e2

    e3 = {}
    if not args.skip_e3:
        topk_impl = lambda lg, cb, c, rk, rke, o: repo.prefill_topk(  # noqa: E731
            lg, cb, c, rk, rke, o
        )
        e3["realistic"] = run_e3(torch, repo, args, bundle, q_bundle, topk_impl)
        if args.workspace_pt is None:
            e3["wide"] = run_e3(torch, repo, args, wide_bundle, q_wide, topk_impl)
    results["e3"] = e3
    results["unverified"] = [
        "E2/E3 realistic distribution is synthetic unless --workspace-pt is set; "
        "the clamp margin for the real model's weights is NOT measured here.",
        "Single-request chunks only; multi-request packed batches are not exercised.",
        "Per-row visibility is a synthetic cold-prefill ramp, not real metadata.",
        "No end-to-end model/serving comparison; no downstream token agreement.",
        "The FP8 kernel itself is not implemented (measurement-only round).",
    ]
    print_human_summary(results)
    emit(results, args.out)
    return 0


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        return do_dry_run(args)
    if args.cpu_selftest:
        return do_cpu_selftest(args)
    return do_gpu_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
