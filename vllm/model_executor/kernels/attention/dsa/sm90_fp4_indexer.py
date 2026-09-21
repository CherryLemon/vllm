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
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

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
):
    row = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)

    n_vis = tl.load(context_lens_ptr + row)
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
):
    row = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(0, HALF_D)

    ks = tl.load(cu_ks_ptr + row)
    ke = tl.load(cu_ke_ptr + row)
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
        num_warps=_NUM_WARPS,
    )
    if write_candidates:
        return logits, candidate_scores
    return logits