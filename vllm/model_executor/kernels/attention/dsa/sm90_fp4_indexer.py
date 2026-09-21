# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 (Hopper) indexer-logits kernels for the MXFP4 indexer K cache.

DeepGEMM's ``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits`` FP4 variants
are SM100-only, so on family(90) the MXFP4 indexer cache has no scoring kernel.
These Triton kernels port the numerics of SGLang's
``kernels/ops/attention/dsv4/sm90_fp4_indexer.py`` decode kernel exactly, on top
of vLLM's own packed MXFP4 Q and the byte-identical MXFP4 K cache layout.

Cache layout (vLLM writer ``indexer_k_norm_rope_store(..., use_fp4_cache=True)``):
a page of ``page_size`` tokens is *segregated* -- ``page_size * 64`` payload
bytes followed by ``page_size * 4`` UE8M0 scale bytes -- at offset
``page * page_stride``.  Byte ``i`` of a payload row holds head element ``2i``
in its low nibble and ``2i + 1`` in its high nibble; ``page_size`` payload
bytes back, byte ``i // 16`` is the UE8M0 exponent for both nibbles.  A nibble
decodes to E2M1 ``e`` and contributes ``e * 2 ** (exp - 127)``.

RATIO is always 1 in vLLM: the physical slot of logical (compressed) position
``L`` of decode row ``r`` is exactly
``block_table[r, L // page_size] * page_size + L % page_size``.  There is no
``req_to_token`` pool indirection (SGLang's ``req_to_token[req, L*RATIO]//RATIO``
halving is a property of SGLang's pool layout, not of vLLM's).

The whole module is dead code unless :func:`has_sm90_fp4_indexer` is true, so
importing it never changes any existing (family(100) / fp8) behaviour.
"""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# E4M3 max, kept for parity with the SGLang module's public constant.
FP8_E4M3_MAX = 448.0

INDEX_HEAD_DIM = 128
# 64 payload bytes + 4 UE8M0 bytes per token: the MXFP4 page row width.
PAYLOAD_BYTES = 64
SCALE_BYTES = 4
# Triton @jit bodies may only close over constexpr globals, so the kernels use
# these aliases (the plain ints above stay usable from host Python code).
_PAYLOAD_BYTES = tl.constexpr(PAYLOAD_BYTES)
_SCALE_BYTES = tl.constexpr(SCALE_BYTES)

# One tile of columns per program; the SGLang decode kernel uses 64.
_BLOCK_L = 64
# tl.dot needs H >= 16; H is 32 (V4.1 index_n_heads) or 64.
_NUM_WARPS = 4

# Rows grouped per CTA by the group-6 K-reuse kernel (DSpark static target
# verification: 1 draft bonus + 5 drafted tokens = 6 query rows per request).
# ``_GROUP6_PGROUP`` is only the power-of-two lane count for the in-kernel
# ``tl.arange``; the ``offset < GROUP`` mask restricts the group to six rows.
_GROUP6 = 6
_GROUP6_PGROUP = 8

# Debug-only branch observation for the grouped kernel
# (``VLLM_SM90_FP4_GROUP6_STATS=1``).  It proves *which branch ran*, which a
# pure output-equality test cannot: a fallback group is numerically identical
# to a shared one but re-reads the K tile per row.
#
# One byte per (group, tile) CTA, written by that CTA alone -- no atomics, so
# the write is race-free and costs one store.  A slot still holding
# ``_GROUP6_STAT_UNSET`` after the launch means that CTA did not run.
_GROUP6_STAT_SHARED = 0
_GROUP6_STAT_FALLBACK = 1
_GROUP6_STAT_INVALID_TILE = 2
_GROUP6_STAT_UNSET = 255
# Triton @jit bodies may only close over constexpr globals; see `_PAYLOAD_BYTES`.
_TL_STAT_SHARED = tl.constexpr(_GROUP6_STAT_SHARED)
_TL_STAT_FALLBACK = tl.constexpr(_GROUP6_STAT_FALLBACK)
_TL_STAT_INVALID_TILE = tl.constexpr(_GROUP6_STAT_INVALID_TILE)
# One slab per device, grown on demand and only while the flag is on, plus the
# number of slots the *last* launch used (slots beyond it hold stale codes).
_group6_stats_buffers: dict[torch.device, torch.Tensor] = {}
_group6_stats_last_ctas: dict[torch.device, int] = {}


def _group6_stats_slab(device: torch.device, num_ctas: int) -> torch.Tensor | None:
    """Per-device ``uint8`` branch-code slab of ``num_ctas`` slots, or ``None``.

    Refilled with the unset sentinel before every launch, so a reader only sees
    the current launch's codes even though the buffer is reused.  This is the
    same mechanism that, before the writer->reader test's page-table width was
    corrected, made that test's out-of-row page-table gather fatal: an extra
    small allocation moved the garbage page ids into unmapped memory.  See
    ``reports/review_fixes_round4.md`` for the 2x2 that pinned that down.
    """
    if not envs.VLLM_SM90_FP4_GROUP6_STATS:
        return None
    buf = _group6_stats_buffers.get(device)
    if buf is None or buf.numel() < num_ctas:
        buf = torch.empty(max(num_ctas, 1024), dtype=torch.uint8, device=device)
        _group6_stats_buffers[device] = buf
    buf[:num_ctas].fill_(_GROUP6_STAT_UNSET)
    _group6_stats_last_ctas[device] = num_ctas
    return buf[:num_ctas]


def reset_group6_stats(device: torch.device | None = None) -> None:
    """Mark every group-6 observation slot unset (debug helper)."""
    for dev, buf in _group6_stats_buffers.items():
        if device is None or dev == device:
            buf.fill_(_GROUP6_STAT_UNSET)
    for dev in list(_group6_stats_last_ctas):
        if device is None or dev == device:
            _group6_stats_last_ctas[dev] = 0


def sm90_fp4_group6_stats(device: torch.device | None = None) -> dict[str, int]:
    """Read the group-6 branch codes of the **last** grouped launch (debug).

    Returns ``{"shared": n, "fallback": n, "invalid_tile": n}`` plus
    ``"enabled"``, counted from the per-CTA slots of the most recent launch.
    Only meaningful with ``VLLM_SM90_FP4_GROUP6_STATS=1``; with the flag off the
    kernel writes nothing.  Never called from the model path.
    """
    totals = {
        "enabled": int(bool(envs.VLLM_SM90_FP4_GROUP6_STATS)),
        "shared": 0,
        "fallback": 0,
        "invalid_tile": 0,
    }
    for dev, buf in _group6_stats_buffers.items():
        if device is not None and dev != device:
            continue
        num_ctas = _group6_stats_last_ctas.get(dev, 0)
        if num_ctas == 0:
            continue
        counts = torch.bincount(buf[:num_ctas].detach().cpu(), minlength=3)
        totals["shared"] += int(counts[_GROUP6_STAT_SHARED])
        totals["fallback"] += int(counts[_GROUP6_STAT_FALLBACK])
        totals["invalid_tile"] += int(counts[_GROUP6_STAT_INVALID_TILE])
    return totals


def has_sm90_fp4_indexer() -> bool:
    """True on family(90) CUDA when ``VLLM_SM90_FP4_INDEXER`` is enabled.

    This is the single predicate that gates the new kernels; family(100) and
    every non-CUDA platform return False regardless of the env flag.
    """
    return (
        bool(envs.VLLM_SM90_FP4_INDEXER)
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
    )


@triton.jit
def _e2m1_decode(code):
    """Bit-exact E2M1 -> FP32 via integer ops (SGLang's decode).

    E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6; their exact FP32 bit
    patterns avoid per-element exponentiation.  ``mag == 1`` (0.5) needs the
    exponent-126 encoding, every larger magnitude maps to ``0x3F000000 +
    (mag << 22)``.  The final exact add canonicalizes FP4 ``-0`` to ``+0``,
    matching legacy decoding.
    """
    u = code.to(tl.uint32)
    mag = u & 7
    bits = tl.where(
        mag == 0, 0, tl.where(mag == 1, 0x3F000000, 0x3F000000 + (mag << 22))
    )
    v = (bits | ((u & 8) << 28)).to(tl.float32, bitcast=True)
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, 0f00000000;",
        constraints="=f,f",
        args=[v],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_q_packed(
    q_ptr,
    qs_ptr,
    row,
    offs_h,
    offs_i,
    stride_qr,
    stride_qh,
    stride_sr,
    stride_sh,
    HALF_D: tl.constexpr,
):
    """Decode one row's packed MXFP4 Q to bf16 even/odd halves.

    ``q_ptr`` is uint8 ``[rows, H, HALF_D]`` (2 nibbles/byte) and ``qs_ptr`` is
    uint8 ``[rows, H, HALF_D // 16]`` (4 UE8M0 bytes per head).  Byte ``i``
    holds head elements ``2i`` (low) / ``2i + 1`` (high), both sharing scale
    byte ``i // 16``.
    """
    qpay = tl.load(
        q_ptr + row * stride_qr + offs_h[:, None] * stride_qh + offs_i[None, :]
    )
    qexps = tl.load(
        qs_ptr
        + row * stride_sr
        + offs_h[:, None] * stride_sh
        + (offs_i // 16)[None, :]
    )
    qscale = tl.exp2(qexps.to(tl.float32) - 127.0)
    q_even = (_e2m1_decode(qpay & 0x0F) * qscale).to(tl.bfloat16)
    q_odd = (_e2m1_decode((qpay >> 4) & 0x0F) * qscale).to(tl.bfloat16)
    return q_even, q_odd


@triton.jit
def _score_heads(q_even, q_odd, k_low, k_high, w_ptr, row, offs_h, stride_wr):
    """The SGLang bf16 rounding chain, exact.

    ``acc = dot(q_even, k_low^T) + dot(q_odd, k_high^T)`` in fp32, then
    ``s = bf16(acc)`` -> ``relu`` -> ``s = bf16(s * w)`` -> ``logit = bf16(sum_h s)``.
    """
    acc = tl.dot(q_even, tl.trans(k_low))
    acc += tl.dot(q_odd, tl.trans(k_high))
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + row * stride_wr + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    return tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)


# ---------------------------------------------------------------------------
# Decode: paged MXFP4 K cache
# ---------------------------------------------------------------------------


@triton.jit
def _sm90_fp4_paged_index_logits_kernel(
    q_ptr,
    qs_ptr,
    w_ptr,
    cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    candidate_blocks_ptr,
    out_ptr,
    candidate_scores_ptr,
    candidate_lens_ptr,
    width,
    page_size,
    page_stride,
    stride_qr,
    stride_qh,
    stride_sr,
    stride_sh,
    stride_wr,
    stride_bt,
    stride_cb,
    stride_out,
    stride_cs,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
    SKIP_INVALID: tl.constexpr,
):
    row = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)

    n_vis = tl.load(context_lens_ptr + row)
    if SKIP_INVALID:
        # A fully invisible tile used to run the masked K load, the Q load and
        # both ``tl.dot``s only to overwrite every result with ``-inf``.  Exit
        # before that work, writing exactly what the body would have written.
        #
        # The two modes need *different* "no visible column" predicates.
        # Dense: ``offs_l`` is the logical position, so the tile start bound is
        # sound.  Compact: ``offs_l`` is a candidate-matrix column, so
        # visibility must be re-derived from the mapped ``logical`` position.
        # Never compare a compact column against ``n_vis`` here; see the body's
        # comment for the coordinate-space bug that caused.
        if USE_CANDIDATES:
            block_col = offs_l // CANDIDATE_BLOCK_SIZE
            within = offs_l % CANDIDATE_BLOCK_SIZE
            block = tl.load(
                candidate_blocks_ptr + row * stride_cb + block_col,
                mask=offs_l < width,
                other=-1,
            )
            logical = block.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
            visible = (offs_l < width) & (block >= 0) & (logical < n_vis)
            tile_invisible = tl.sum(visible.to(tl.int32), axis=0) == 0
        else:
            tile_invisible = lb * BLOCK_L >= tl.minimum(n_vis, width)
        if tile_invisible:
            tl.store(
                out_ptr + row * stride_out + offs_l,
                -float("inf"),
                mask=offs_l < width,
            )
            if WRITE_CANDIDATES:
                # Reproduce this kernel's own all-``-inf`` tail, including the
                # forced +inf on the newest visible *logical* block.
                blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
                block_ids = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
                num_blocks = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                last_block = (n_vis - 1) // CANDIDATE_BLOCK_SIZE
                block_scores = tl.full((blocks_per_tile,), float("-inf"), tl.float32)
                block_scores = tl.where(
                    (n_vis > 0) & (block_ids == last_block) & (block_ids < num_blocks),
                    float("inf"),
                    block_scores,
                )
                tl.store(
                    candidate_scores_ptr + row * stride_cs + block_ids,
                    block_scores,
                    mask=block_ids < num_blocks,
                )
                tl.store(
                    candidate_lens_ptr + row,
                    (n_vis + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
                    mask=lb == 0,
                )
            return
    if USE_CANDIDATES:
        # Compact mode: ``offs_l`` indexes the candidate *matrix* column
        # (candidate_blocks.shape[1] * CANDIDATE_BLOCK_SIZE of them), not the
        # logical compressed context.  It therefore can only be bounded by the
        # stored row width; visibility is a property of the mapped position
        # ``logical``.  Comparing the compact column against ``n_vis`` (as the
        # dense branch legitimately does) silently dropped real candidates:
        # the production publisher pins each row's newest -- possibly partial --
        # block first and does not sort the remaining ids, so valid compact
        # columns routinely extend past ``n_vis`` while invalid ones sit below
        # it.  -1 padded candidate blocks stay invalid and are never
        # dereferenced.
        block_col = offs_l // CANDIDATE_BLOCK_SIZE
        within = offs_l % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            candidate_blocks_ptr + row * stride_cb + block_col,
            mask=offs_l < width,
            other=-1,
        )
        logical = block.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
        valid = (offs_l < width) & (block >= 0) & (logical < n_vis)
    else:
        # Dense mode: ``offs_l`` *is* the logical position, so both bounds
        # apply to the same coordinate.
        logical = offs_l.to(tl.int64)
        valid = offs_l < tl.minimum(n_vis, width)

    # RATIO == 1: slot = block_table[row, L // page_size] * page_size + L % page_size.
    page_idx = tl.load(
        block_table_ptr + row * stride_bt + logical // page_size,
        mask=valid,
        other=0,
    ).to(tl.int64)
    slot = page_idx * page_size + (logical % page_size)
    page = slot // page_size
    off = slot % page_size
    row_base = page * page_stride

    pay = tl.load(
        cache_ptr
        + row_base[:, None]
        + off[:, None] * _PAYLOAD_BYTES
        + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    # Element j uses scale block j // 32 -> byte i // 16.
    exps = tl.load(
        cache_ptr
        + row_base[:, None]
        + page_size * _PAYLOAD_BYTES
        + off[:, None] * _SCALE_BYTES
        + (offs_i // 16)[None, :],
        mask=valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    k_low = (_e2m1_decode(pay & 0x0F) * scale).to(tl.bfloat16)
    k_high = (_e2m1_decode((pay >> 4) & 0x0F) * scale).to(tl.bfloat16)

    q_even, q_odd = _load_q_packed(
        q_ptr,
        qs_ptr,
        row,
        offs_h,
        offs_i,
        stride_qr,
        stride_qh,
        stride_sr,
        stride_sh,
        HALF_D,
    )
    logit = _score_heads(q_even, q_odd, k_low, k_high, w_ptr, row, offs_h, stride_wr)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + row * stride_out + offs_l, logit, mask=offs_l < width)

    if WRITE_CANDIDATES:
        blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
        block_scores = tl.reshape(logit, (blocks_per_tile, CANDIDATE_BLOCK_SIZE))
        block_scores = tl.max(block_scores, axis=1)
        block_ids = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
        # Force +inf on the newest visible block; NaN from a corrupt K row must
        # propagate (tl.max propagates NaN, tl.maximum below preserves it).
        num_blocks = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        last_block = (n_vis - 1) // CANDIDATE_BLOCK_SIZE
        block_scores = tl.where(
            (n_vis > 0) & (block_ids == last_block) & (block_ids < num_blocks),
            float("inf"),
            block_scores,
        )
        tl.store(
            candidate_scores_ptr + row * stride_cs + block_ids,
            block_scores,
            mask=block_ids < num_blocks,
        )
        tl.store(
            candidate_lens_ptr + row,
            (n_vis + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
            mask=lb == 0,
        )


# ---------------------------------------------------------------------------
# Decode: group-6 K reuse (DSpark static target verification)
# ---------------------------------------------------------------------------
#
# DSpark drafts ``dspark_block_size`` (5) tokens, so the target verifies
# 1 + 5 = 6 query rows per request.  The per-row kernel above decodes the same
# MXFP4 K tile six times, once per row.  These helpers share one decode across
# the six same-request rows of a group: the K tile is loaded once at the group
# *maximum* visible length and each row applies its own visibility mask.  Each
# query keeps its own packed Q load and its own per-head MMA/reduction layout,
# so the bf16 rounding chain is byte-identical to six independent per-row
# calls.  Request identity (and, in compact mode, per-tile candidate-row
# equality) is checked on device; any failure falls back to per-row K reloads.
#
# The reference (SGLang ``_fp4_index_logits_grouped_kernel``) groups only the
# dense/logical candidate-source pass.  The compact variant here is a labelled
# extension: its shared tile is exact only when all six rows map the tile's
# compact columns to the same logical positions, hence the tile-local candidate
# id equality guard below.


@triton.jit
def _sm90_fp4_group_load_keys(
    cache_ptr,
    block_table_ptr,
    candidate_blocks_ptr,
    leader_row,
    n_vis,
    width,
    page_size,
    page_stride,
    stride_bt,
    stride_cb,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    BLOCK_L: tl.constexpr,
    HALF_D: tl.constexpr,
):
    """Decode one K tile, addressed by logical position or compact column.

    ``leader_row`` supplies the page table and (compact mode) the candidate
    ids; ``n_vis`` is the tile's visibility bound (the group maximum for the
    shared tile, the row's own length on the fallback path).  Returns the two
    bf16 K halves plus the per-column ``logical`` position and ``col_valid``
    (bounds/``-1`` only) so the caller can re-mask per row with a smaller
    ``n_vis`` without re-deriving the coordinate space.
    """
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_i = tl.arange(0, HALF_D)
    if USE_CANDIDATES:
        # Compact mode: ``offs_l`` indexes the candidate *matrix* column, so it
        # is bounded by the stored row width; visibility is a property of the
        # mapped logical position (see the per-row kernel's contract).
        block_col = offs_l // CANDIDATE_BLOCK_SIZE
        within = offs_l % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            candidate_blocks_ptr + leader_row * stride_cb + block_col,
            mask=offs_l < width,
            other=-1,
        )
        logical = block.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
        col_valid = (offs_l < width) & (block >= 0)
    else:
        # Dense mode: ``offs_l`` *is* the logical position.
        logical = offs_l.to(tl.int64)
        col_valid = offs_l < width
    valid = col_valid & (logical < n_vis)
    # RATIO == 1: slot = block_table[row, L // page_size] * page_size + L % page_size.
    page_idx = tl.load(
        block_table_ptr + leader_row * stride_bt + logical // page_size,
        mask=valid,
        other=0,
    ).to(tl.int64)
    slot = page_idx * page_size + (logical % page_size)
    page = slot // page_size
    off = slot % page_size
    row_base = page * page_stride
    pay = tl.load(
        cache_ptr
        + row_base[:, None]
        + off[:, None] * _PAYLOAD_BYTES
        + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    exps = tl.load(
        cache_ptr
        + row_base[:, None]
        + page_size * _PAYLOAD_BYTES
        + off[:, None] * _SCALE_BYTES
        + (offs_i // 16)[None, :],
        mask=valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    k_low = (_e2m1_decode(pay & 0x0F) * scale).to(tl.bfloat16)
    k_high = (_e2m1_decode((pay >> 4) & 0x0F) * scale).to(tl.bfloat16)
    return k_low, k_high, logical, col_valid


@triton.jit
def _sm90_fp4_group_score(
    q_ptr,
    qs_ptr,
    w_ptr,
    out_ptr,
    k_low,
    k_high,
    logical,
    col_valid,
    row,
    n_vis,
    offs_l,
    width,
    stride_qr,
    stride_qh,
    stride_sr,
    stride_sh,
    stride_wr,
    stride_out,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
):
    """Score one row against the (possibly shared) K tile.

    The masking predicate is the per-row kernel's, verbatim:
    ``col_valid & (logical < n_vis)``.  In dense mode ``logical == offs_l`` and
    ``col_valid == offs_l < width``, so this is ``offs_l < min(n_vis, width)``;
    in compact mode ``logical`` is the mapped candidate position.
    """
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)
    q_even, q_odd = _load_q_packed(
        q_ptr,
        qs_ptr,
        row,
        offs_h,
        offs_i,
        stride_qr,
        stride_qh,
        stride_sr,
        stride_sh,
        HALF_D,
    )
    logit = _score_heads(q_even, q_odd, k_low, k_high, w_ptr, row, offs_h, stride_wr)
    valid = col_valid & (logical < n_vis)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + row * stride_out + offs_l, logit, mask=offs_l < width)


@triton.jit
def _sm90_fp4_group_invalid(
    out_ptr, row, offs_l, width, stride_out, BLOCK_L: tl.constexpr
):
    """Write the fully invisible dense tile as ``-inf`` for one row."""
    tl.store(
        out_ptr + row * stride_out + offs_l,
        tl.full([BLOCK_L], float("-inf"), tl.float32),
        mask=offs_l < width,
    )


@triton.jit
def _sm90_fp4_grouped_paged_index_logits_kernel(
    q_ptr,
    qs_ptr,
    w_ptr,
    cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    row_indices_ptr,
    candidate_blocks_ptr,
    out_ptr,
    n_rows,
    width,
    page_size,
    page_stride,
    stride_qr,
    stride_qh,
    stride_sr,
    stride_sh,
    stride_wr,
    stride_bt,
    stride_cb,
    stride_out,
    stats_ptr,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    GROUP: tl.constexpr,
    PGROUP: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    SKIP_INVALID: tl.constexpr,
    COLLECT_STATS: tl.constexpr,
):
    first = tl.program_id(0) * GROUP
    lb = tl.program_id(1)
    stat_slot = stats_ptr + tl.program_id(0) * tl.num_programs(1) + lb
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs = tl.arange(0, PGROUP)
    rows = first + offs
    gmask = (offs < GROUP) & (rows < n_rows)
    lens = tl.load(context_lens_ptr + rows, mask=gmask, other=0)
    max_n_vis = tl.max(lens, 0)

    # Dense-only early exit: when the whole tile is invisible to every row the
    # shared load would be all-masked anyway.  Compact columns are unordered
    # and may legitimately extend past ``n_vis``, so there is no equivalent
    # O(1) bound there (same reasoning as the per-row kernel's compact mask).
    # Keep the constexpr gate separate from the runtime bound: ``USE_CANDIDATES``
    # is compile-time and must not force the runtime comparison into a Tensor.
    if not USE_CANDIDATES:  # noqa: SIM102
        if lb * BLOCK_L >= tl.minimum(max_n_vis, width):
            if COLLECT_STATS:
                tl.store(stat_slot, _TL_STAT_INVALID_TILE)
            for j in tl.range(0, GROUP, loop_unroll_factor=1):
                b = first + j
                if b < n_rows:
                    _sm90_fp4_group_invalid(
                        out_ptr, b, offs_l, width, stride_out, BLOCK_L
                    )
            return

    # Compact early exit, opt-in (``VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES``)
    # because the "no row sees this tile" test costs one 64-wide candidate
    # gather per row instead of the dense O(1) bound.  It mirrors the per-row
    # compact predicate exactly: a column is visible for a row when its mapped
    # logical position is inside that row's context length.  Without this the
    # grouped compact path had no all-invisible skip at all, so the two opt-in
    # optimisations could not be assumed to compose.
    if USE_CANDIDATES and SKIP_INVALID:  # noqa: SIM102
        col_block = offs_l // CANDIDATE_BLOCK_SIZE
        within = offs_l % CANDIDATE_BLOCK_SIZE
        n_blocks_row = width // CANDIDATE_BLOCK_SIZE
        any_visible = tl.zeros([BLOCK_L], dtype=tl.int32)
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < n_rows:
                n_vis_b = tl.load(context_lens_ptr + b)
                ids = tl.load(
                    candidate_blocks_ptr + b * stride_cb + col_block,
                    mask=(offs_l < width) & (col_block < n_blocks_row),
                    other=-1,
                )
                logical = ids.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
                any_visible = any_visible | (
                    (offs_l < width) & (ids >= 0) & (logical < n_vis_b)
                ).to(tl.int32)
        if tl.sum(any_visible, 0) == 0:
            if COLLECT_STATS:
                tl.store(stat_slot, _TL_STAT_INVALID_TILE)
            for j in tl.range(0, GROUP, loop_unroll_factor=1):
                b = first + j
                if b < n_rows:
                    _sm90_fp4_group_invalid(
                        out_ptr, b, offs_l, width, stride_out, BLOCK_L
                    )
            return

    # Request identity is checked on device on every replay: the six rows of a
    # group must all belong to the leader's request, else the shared tile is
    # wrong and the group falls back to per-row K reloads.
    req0 = tl.load(row_indices_ptr + first)
    # ``PGROUP`` (8) reduction lanes cover only ``GROUP`` (6) real rows, so the
    # padding lanes must be masked out of the comparison: their ``other`` value
    # (0) is not a real request id, and comparing it made ``share`` False for
    # *every* full group whose request id was non-zero, silently turning the K
    # reuse into a per-row reload on every CTA (including partial last groups,
    # which never had a chance to share).
    reqs = tl.load(row_indices_ptr + rows, mask=gmask, other=0)
    share = tl.sum(((reqs != req0) & gmask).to(tl.int32), 0) == 0

    if USE_CANDIDATES:
        # Compact extension: the shared tile is exact only if every row maps
        # this tile's compact columns to the same logical positions, i.e. their
        # candidate block ids agree tile-locally.  Dense mode shares
        # unconditionally because all six rows use the same page table.
        n_tile_blocks: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
        base_col = (lb * BLOCK_L) // CANDIDATE_BLOCK_SIZE
        tids = tl.arange(0, n_tile_blocks)
        cmask = (base_col + tids) < (width // CANDIDATE_BLOCK_SIZE)
        lead = tl.load(
            candidate_blocks_ptr + first * stride_cb + base_col + tids,
            mask=cmask,
            other=-1,
        )
        for j in tl.range(1, GROUP, loop_unroll_factor=1):
            bj = first + j
            if bj < n_rows:
                other_ids = tl.load(
                    candidate_blocks_ptr + bj * stride_cb + base_col + tids,
                    mask=cmask,
                    other=-1,
                )
                share = share & (tl.sum((other_ids != lead).to(tl.int32), 0) == 0)

    if share:
        if COLLECT_STATS:
            tl.store(stat_slot, _TL_STAT_SHARED)
        k_low, k_high, logical, col_valid = _sm90_fp4_group_load_keys(
            cache_ptr,
            block_table_ptr,
            candidate_blocks_ptr,
            first,
            max_n_vis,
            width,
            page_size,
            page_stride,
            stride_bt,
            stride_cb,
            USE_CANDIDATES,
            CANDIDATE_BLOCK_SIZE,
            BLOCK_L,
            HALF_D,
        )
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < n_rows:
                n_vis = tl.load(context_lens_ptr + b)
                _sm90_fp4_group_score(
                    q_ptr,
                    qs_ptr,
                    w_ptr,
                    out_ptr,
                    k_low,
                    k_high,
                    logical,
                    col_valid,
                    b,
                    n_vis,
                    offs_l,
                    width,
                    stride_qr,
                    stride_qh,
                    stride_sr,
                    stride_sh,
                    stride_wr,
                    stride_out,
                    H,
                    HALF_D,
                )
    else:
        # Mixed-request / partial / differing-candidate group: reload per row,
        # i.e. the existing one-row-per-CTA behaviour, never wrong.
        if COLLECT_STATS:
            tl.store(stat_slot, _TL_STAT_FALLBACK)
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < n_rows:
                n_vis = tl.load(context_lens_ptr + b)
                k_low, k_high, logical, col_valid = _sm90_fp4_group_load_keys(
                    cache_ptr,
                    block_table_ptr,
                    candidate_blocks_ptr,
                    b,
                    n_vis,
                    width,
                    page_size,
                    page_stride,
                    stride_bt,
                    stride_cb,
                    USE_CANDIDATES,
                    CANDIDATE_BLOCK_SIZE,
                    BLOCK_L,
                    HALF_D,
                )
                _sm90_fp4_group_score(
                    q_ptr,
                    qs_ptr,
                    w_ptr,
                    out_ptr,
                    k_low,
                    k_high,
                    logical,
                    col_valid,
                    b,
                    n_vis,
                    offs_l,
                    width,
                    stride_qr,
                    stride_qh,
                    stride_sr,
                    stride_sh,
                    stride_wr,
                    stride_out,
                    H,
                    HALF_D,
                )


# ---------------------------------------------------------------------------
# Prefill: packed (gathered) MXFP4 K workspace
# ---------------------------------------------------------------------------


@triton.jit
def _sm90_fp4_workspace_index_logits_kernel(
    q_ptr,
    qs_ptr,
    w_ptr,
    k_values_ptr,
    k_scales_ptr,
    cu_ks_ptr,
    cu_ke_ptr,
    candidate_blocks_ptr,
    out_ptr,
    candidate_scores_ptr,
    width,
    stride_kv,
    stride_ks,
    stride_qr,
    stride_qh,
    stride_sr,
    stride_sh,
    stride_wr,
    stride_cb,
    stride_out,
    stride_cs,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
    SKIP_INVALID: tl.constexpr,
):
    row = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)

    ks = tl.load(cu_ks_ptr + row)
    ke = tl.load(cu_ke_ptr + row)
    if SKIP_INVALID:
        # Same early exit as the paged kernel.  Dense visibility here is the
        # absolute workspace interval ``[ks, ke)``; compact visibility is again
        # a property of the mapped workspace row ``ks + logical`` and never of
        # the candidate-matrix column ``offs_l``.
        if USE_CANDIDATES:
            block_col = offs_l // CANDIDATE_BLOCK_SIZE
            within = offs_l % CANDIDATE_BLOCK_SIZE
            block = tl.load(
                candidate_blocks_ptr + row * stride_cb + block_col,
                mask=offs_l < width,
                other=-1,
            )
            logical = block.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
            k_row = ks.to(tl.int64) + logical
            visible = (offs_l < width) & (block >= 0) & (k_row < ke)
            tile_invisible = tl.sum(visible.to(tl.int32), axis=0) == 0
        else:
            # ``ks >= ke`` is an empty interval: a row whose workspace scope has
            # no positions at all (possible for a clamped/degenerate decode
            # row) scores nothing, yet neither half-open bound test above fires
            # for a tile that merely brackets the empty interval.
            tile_invisible = (
                (ks >= ke)
                | (lb * BLOCK_L >= ke)
                | ((lb + 1) * BLOCK_L <= ks)
            )
        if tile_invisible:
            tl.store(
                out_ptr + row * stride_out + offs_l,
                -float("inf"),
                mask=offs_l < width,
            )
            if WRITE_CANDIDATES:
                # This kernel's tail has no forced +inf newest block (the
                # candidate publisher applies that), so an all-``-inf`` tile
                # stores plain ``-inf`` block scores.
                blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
                block_ids = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
                num_blocks = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
                tl.store(
                    candidate_scores_ptr + row * stride_cs + block_ids,
                    tl.full((blocks_per_tile,), float("-inf"), tl.float32),
                    mask=block_ids < num_blocks,
                )
            return
    if USE_CANDIDATES:
        block_col = offs_l // CANDIDATE_BLOCK_SIZE
        within = offs_l % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            candidate_blocks_ptr + row * stride_cb + block_col,
            mask=offs_l < width,
            other=-1,
        )
        logical = block.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
        # cu_seqlen_ks is the row's start in the gathered workspace, so the
        # request-local compressed position L lives at workspace row ks + L.
        k_row = ks.to(tl.int64) + logical
        valid = (offs_l < width) & (block >= 0) & (k_row < ke)
    else:
        k_row = offs_l.to(tl.int64)
        valid = (offs_l < width) & (offs_l >= ks) & (offs_l < ke)

    pay = tl.load(
        k_values_ptr + k_row[:, None] * stride_kv + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    exps = tl.load(
        k_scales_ptr + k_row[:, None] * stride_ks + (offs_i // 16)[None, :],
        mask=valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    k_low = (_e2m1_decode(pay & 0x0F) * scale).to(tl.bfloat16)
    k_high = (_e2m1_decode((pay >> 4) & 0x0F) * scale).to(tl.bfloat16)

    q_even, q_odd = _load_q_packed(
        q_ptr,
        qs_ptr,
        row,
        offs_h,
        offs_i,
        stride_qr,
        stride_qh,
        stride_sr,
        stride_sh,
        HALF_D,
    )
    logit = _score_heads(q_even, q_odd, k_low, k_high, w_ptr, row, offs_h, stride_wr)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + row * stride_out + offs_l, logit, mask=offs_l < width)

    if WRITE_CANDIDATES:
        blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
        block_scores = tl.reshape(logit, (blocks_per_tile, CANDIDATE_BLOCK_SIZE))
        block_scores = tl.max(block_scores, axis=1)
        block_ids = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
        # Fully visible candidate blocks get their max; the forced +inf newest
        # rule is applied by the candidate publisher, not here.
        num_blocks = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        tl.store(
            candidate_scores_ptr + row * stride_cs + block_ids,
            block_scores,
            mask=block_ids < num_blocks,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _check_q(q_values: torch.Tensor, q_scale: torch.Tensor, weights: torch.Tensor):
    rows, heads, half_d = q_values.shape
    assert q_values.dtype == torch.uint8 and half_d == INDEX_HEAD_DIM // 2, (
        q_values.shape,
        q_values.dtype,
    )
    assert q_scale.shape == (rows, heads) and q_scale.dtype == torch.int32
    assert weights.shape == (rows, heads)
    return rows, heads


def _q_scale_bytes(q_scale: torch.Tensor, rows: int, heads: int) -> torch.Tensor:
    """int32 ``[rows, H]`` UE8M0 scales -> the 4 raw bytes the kernel indexes."""
    if q_scale.stride(-1) != 1 or not q_scale.is_contiguous():
        q_scale = q_scale.contiguous()
    return q_scale.view(torch.uint8).reshape(rows, heads, SCALE_BYTES)


def sm90_fp4_paged_index_logits(
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    page_size: int = 0,
    *,
    width: int | None = None,
    ratio: int = 1,
    write_candidates: bool = True,
    row_indices: torch.Tensor | None = None,
    query_group_size: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Score decode rows against the paged MXFP4 indexer K cache on SM90.

    Args:
        q_values: ``[rows, H, 64]`` uint8 packed MXFP4 Q (2 nibbles/byte).
        q_scale: ``[rows, H]`` int32; 4 UE8M0 bytes per head.
        kv_cache: 3D uint8 ``[num_blocks, page_size, 68]`` indexer K cache.
        weights: ``[rows, H]`` bf16/fp32 per-head weights (softmax/head scale
            folded in; the per-block Q scale is applied in-kernel).
        context_lens: ``[rows]`` int32 visible (compressed) positions.
        block_table: ``[rows, P]`` int32 page table, ``stride(-1) == 1``.
        candidate_blocks: ``[rows, K]`` int32 request-local candidate block
            ids (-1 padded).  When given, columns are candidate-major and the
            dense ``width`` is ignored.
        candidate_block_size: visible positions per candidate block.
        page_size: indexer cache tokens per page (attention block size).
        width: Dense output width; defaults to ``block_table.shape[1] *
            page_size`` (the cache's full compressed capacity).
        ratio: Must be 1.  vLLM's indexer cache is compressed-position
            indexed, so the SGLang ``RATIO`` halving has no vLLM equivalent.
        write_candidates: In candidate mode, also reduce each candidate
            block's token logits to a per-block score.  Consumers that already
            own their candidate ids (the compact decode path) do not read
            ``candidate_scores``/``candidate_lens``; pass ``False`` to skip the
            ``[rows, K]`` fp32 allocation and the in-kernel block reduction.
        row_indices: ``[rows]`` int32 row -> request map.  Required (and only
            used) when the group-6 K-reuse path is admitted; the grouped kernel
            checks on device that a group's six rows share one request and
            falls back to per-row K reloads when they do not.
        query_group_size: Semantic hint from the caller, never inferred from
            ``rows``: 6 on an admitted static DSpark target-verify step, 1
            otherwise.  Any value other than 6 keeps the per-row grid.

    Returns:
        Dense/masked mode: ``[rows, width]`` fp32 logits, ``-inf`` outside
        ``context_lens``.  Candidate mode with ``write_candidates``:
        ``(logits [rows, K*CBS], candidate_scores [rows, K])``.  Candidate
        mode without it: a lone ``[rows, K*CBS]`` logits tensor.

    """
    if ratio != 1:
        raise NotImplementedError(
            "vLLM's indexer cache is compressed-position indexed (RATIO == 1); "
            f"ratio={ratio} is not supported."
        )
    rows, heads = _check_q(q_values, q_scale, weights)
    assert kv_cache.dtype == torch.uint8 and kv_cache.dim() == 3
    assert not page_size or kv_cache.shape[1] == page_size, (
        f"kv_cache page dim {kv_cache.shape[1]} must equal page_size "
        f"{page_size}; shape={tuple(kv_cache.shape)} "
        f"stride={tuple(kv_cache.stride())}"
    )
    page_size = page_size or kv_cache.shape[1]
    assert context_lens.shape == (rows,) and context_lens.dtype == torch.int32
    assert block_table.shape[0] == rows and block_table.stride(-1) == 1

    use_candidates = candidate_blocks is not None
    if use_candidates:
        assert candidate_block_size > 0
        assert candidate_blocks.shape[0] == rows
        assert page_size % candidate_block_size == 0
        # The score tile reduces a whole number of candidate blocks.
        assert _BLOCK_L % candidate_block_size == 0
        width = candidate_blocks.shape[1] * candidate_block_size
        num_blocks = candidate_blocks.shape[1]
    else:
        width = width if width is not None else block_table.shape[1] * page_size
        # Dense mode never publishes candidate scores, so no block count is
        # needed; deriving one here would divide by the (zero) block size.
        num_blocks = 0
    write_candidates = use_candidates and write_candidates

    # Group-6 K reuse.  Admission is explicit and conservative: the caller's
    # semantic hint (never inferred from ``rows``), a full group (``rows % 6``),
    # the opt-in env flag, and no in-kernel candidate-score output (the grouped
    # kernel does not reduce block scores).  Request identity and, in compact
    # mode, per-tile candidate-row equality are re-checked on device.  Any
    # non-admitted step takes the existing one-row-per-CTA grid byte-for-byte.
    use_grouped = (
        query_group_size == _GROUP6
        and row_indices is not None
        and rows % _GROUP6 == 0
        and not write_candidates
        and bool(envs.VLLM_SM90_FP4_GROUP6)
    )

    if rows == 0 or width == 0:
        logits = q_values.new_empty((rows, width), dtype=torch.float32)
        if write_candidates:
            return logits, weights.new_empty((rows, num_blocks), dtype=torch.float32)
        return logits

    q_values = q_values.contiguous()
    # q_scale is int32; view as the 4 raw UE8M0 bytes the kernel indexes.
    q_scale_bytes = _q_scale_bytes(q_scale, rows, heads)
    weights = weights.contiguous()
    context_lens = context_lens.contiguous()
    if block_table.stride(-1) != 1:
        block_table = block_table.contiguous()
    cache = kv_cache
    page_stride = int(cache.stride(0))

    logits = torch.empty((rows, width), dtype=torch.float32, device=q_values.device)
    if write_candidates:
        candidate_scores = torch.empty(
            (rows, num_blocks), dtype=torch.float32, device=q_values.device
        )
        candidate_lens = torch.empty(rows, dtype=torch.int32, device=q_values.device)
    else:
        candidate_scores = logits
        candidate_lens = context_lens
    if use_candidates:
        cand = candidate_blocks.contiguous()
        stride_cb = cand.stride(0)
    else:
        cand = context_lens  # unused when USE_CANDIDATES is False
        stride_cb = 0

    if use_grouped:
        assert row_indices is not None
        # The builder publishes ``decode_indices`` as int32 contiguous; the
        # re-normalization is defensive (e.g. a test slicing a wider buffer).
        if row_indices.dtype != torch.int32:
            row_indices = row_indices.to(torch.int32)
        if row_indices.stride(-1) != 1 or not row_indices.is_contiguous():
            row_indices = row_indices.contiguous()
        assert row_indices.shape[0] >= rows, (
            f"row_indices has {row_indices.shape[0]} entries for {rows} rows"
        )
        grid = (triton.cdiv(rows, _GROUP6), triton.cdiv(width, _BLOCK_L))
        # One line per process: this is the only place the group-6 K-reuse
        # launch shape is observable from a serving log (the per-CTA branch
        # codes need VLLM_SM90_FP4_GROUP6_STATS and an in-process reader).
        logger.info_once(
            "SM90 FP4 indexer: grouped (group-6) kernel launched "
            "(rows=%d width=%d groups=%d tiles=%d).",
            rows,
            width,
            grid[0],
            grid[1],
        )
        stats = _group6_stats_slab(logits.device, grid[0] * grid[1])
        _sm90_fp4_grouped_paged_index_logits_kernel[grid](
            q_values,
            q_scale_bytes,
            weights,
            cache,
            block_table,
            context_lens,
            row_indices,
            cand,
            logits,
            rows,
            width,
            page_size,
            page_stride,
            q_values.stride(0),
            q_values.stride(1),
            q_scale_bytes.stride(0),
            q_scale_bytes.stride(1),
            weights.stride(0),
            block_table.stride(0),
            stride_cb,
            logits.stride(0),
            logits if stats is None else stats,
            H=heads,
            HALF_D=INDEX_HEAD_DIM // 2,
            BLOCK_L=_BLOCK_L,
            GROUP=_GROUP6,
            PGROUP=_GROUP6_PGROUP,
            USE_CANDIDATES=use_candidates,
            CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
            SKIP_INVALID=envs.VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES,
            COLLECT_STATS=stats is not None,
            num_warps=_NUM_WARPS,
        )
        # ``use_grouped`` requires ``not write_candidates``, so the grouped
        # kernel never needs to also return block scores.
        return logits

    grid = (rows, triton.cdiv(width, _BLOCK_L))
    _sm90_fp4_paged_index_logits_kernel[grid](
        q_values,
        q_scale_bytes,
        weights,
        cache,
        block_table,
        context_lens,
        cand,
        logits,
        candidate_scores,
        candidate_lens,
        width,
        page_size,
        page_stride,
        q_values.stride(0),
        q_values.stride(1),
        q_scale_bytes.stride(0),
        q_scale_bytes.stride(1),
        weights.stride(0),
        block_table.stride(0),
        stride_cb,
        logits.stride(0),
        candidate_scores.stride(0),
        H=heads,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=_BLOCK_L,
        USE_CANDIDATES=use_candidates,
        CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
        WRITE_CANDIDATES=write_candidates,
        SKIP_INVALID=envs.VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES,
        num_warps=_NUM_WARPS,
    )
    if write_candidates:
        return logits, candidate_scores
    return logits


def sm90_fp4_workspace_index_logits(
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    k_values: torch.Tensor,
    k_scales: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    *,
    ratio: int = 1,
    write_candidates: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Score prefill rows against the packed MXFP4 K gather workspace on SM90.

    ``k_values``/``k_scales`` are exactly the workspace built for the DeepGEMM
    path (``cp_gather_indexer_k_quant_cache``), so the prefill path needs no
    new gather.  ``cu_seqlen_ks`` is the row's start in that workspace, so a
    request-local compressed position ``L`` lives at workspace row
    ``cu_seqlen_ks[row] + L``.

    Args mirror the paged variant; dense output width is ``k_values.shape[0]``.
    ``ratio`` must be 1 (see the paged variant).
    """
    if ratio != 1:
        raise NotImplementedError(
            "vLLM's indexer cache is compressed-position indexed (RATIO == 1); "
            f"ratio={ratio} is not supported."
        )
    rows, heads = _check_q(q_values, q_scale, weights)
    assert k_values.dtype == torch.uint8 and k_values.shape[1] == INDEX_HEAD_DIM // 2
    assert k_scales.dtype == torch.uint8 and k_scales.shape[1] == SCALE_BYTES
    assert k_values.shape[0] == k_scales.shape[0]
    assert cu_seqlen_ks.shape == cu_seqlen_ke.shape == (rows,)
    total = k_values.shape[0]

    use_candidates = candidate_blocks is not None
    if use_candidates:
        assert candidate_block_size > 0
        assert candidate_blocks.shape[0] == rows
        assert _BLOCK_L % candidate_block_size == 0
        width = candidate_blocks.shape[1] * candidate_block_size
        num_blocks = candidate_blocks.shape[1]
    else:
        width = total
        num_blocks = 0
    write_candidates = use_candidates and write_candidates

    if rows == 0 or width == 0:
        logits = q_values.new_empty((rows, width), dtype=torch.float32)
        if write_candidates:
            return logits, weights.new_empty((rows, num_blocks), dtype=torch.float32)
        return logits

    q_values = q_values.contiguous()
    q_scale_bytes = _q_scale_bytes(q_scale, rows, heads)
    weights = weights.contiguous()
    k_values = k_values.contiguous()
    k_scales = k_scales.contiguous()
    cu_seqlen_ks = cu_seqlen_ks.contiguous()
    cu_seqlen_ke = cu_seqlen_ke.contiguous()

    logits = torch.empty((rows, width), dtype=torch.float32, device=q_values.device)
    if write_candidates:
        candidate_scores = torch.empty(
            (rows, num_blocks), dtype=torch.float32, device=q_values.device
        )
    else:
        candidate_scores = logits
    if use_candidates:
        cand = candidate_blocks.contiguous()
        stride_cb = cand.stride(0)
    else:
        cand = cu_seqlen_ks
        stride_cb = 0

    grid = (rows, triton.cdiv(width, _BLOCK_L))
    _sm90_fp4_workspace_index_logits_kernel[grid](
        q_values,
        q_scale_bytes,
        weights,
        k_values,
        k_scales,
        cu_seqlen_ks,
        cu_seqlen_ke,
        cand,
        logits,
        candidate_scores,
        width,
        k_values.stride(0),
        k_scales.stride(0),
        q_values.stride(0),
        q_values.stride(1),
        q_scale_bytes.stride(0),
        q_scale_bytes.stride(1),
        weights.stride(0),
        stride_cb,
        logits.stride(0),
        candidate_scores.stride(0),
        H=heads,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=_BLOCK_L,
        USE_CANDIDATES=use_candidates,
        CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
        WRITE_CANDIDATES=write_candidates,
        SKIP_INVALID=envs.VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES,
        num_warps=_NUM_WARPS,
    )
    if write_candidates:
        return logits, candidate_scores
    return logits