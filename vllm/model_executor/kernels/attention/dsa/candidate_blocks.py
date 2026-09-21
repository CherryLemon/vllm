# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit(do_not_specialize=["width", "nblocks"])
def _block_scores_kernel(
    logits,
    starts,
    ends,
    scores,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blocks = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    offsets = tl.arange(0, triton.next_power_of_2(BLOCK_SIZE))
    cols = start + blocks[:, None] * BLOCK_SIZE + offsets[None, :]
    values = tl.load(
        logits + row * stride_row + cols * stride_col,
        (blocks[:, None] < nblocks)
        & (offsets[None, :] < BLOCK_SIZE)
        & (cols < end)
        & (cols < width),
        other=-float("inf"),
    )
    reduced = tl.reduce(values, 1, _max_with_nan)
    reduced = tl.where(
        (end > start) & (blocks == (end - start - 1) // BLOCK_SIZE),
        float("inf"),
        reduced,
    )
    tl.store(scores + row * nblocks + blocks, reduced, blocks < nblocks)


@triton.jit(do_not_specialize=["k"])
def _store_candidates_kernel(
    values,
    indices,
    output,
    out_stride_row,
    out_stride_col,
    k,
    OUT_K: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    value = tl.load(values + row * k + cols, cols < k, other=-float("inf"))
    index = tl.load(indices + row * k + cols, cols < k, other=-1)
    # NaN scores can occur during warmup; only -inf denotes padding.
    tl.store(
        output + row * out_stride_row + cols * out_stride_col,
        tl.where(value != -float("inf"), index, -1),
        cols < OUT_K,
    )


@triton.jit(do_not_specialize=["width", "nblocks"])
def _candidate_flags_kernel(
    candidates,
    starts,
    flags,
    stride_row,
    stride_col,
    stride_start,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, 1024)
    for tile in range(tl.cdiv(nblocks + 1, 1024)):
        slots = tile * 1024 + offsets
        tl.store(flags + row * (nblocks + 1) + slots, 0, slots <= nblocks)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    cols = tl.arange(0, triton.next_power_of_2(K))
    block = tl.load(
        candidates + row * stride_row + cols * stride_col, cols < K, other=-1
    ).to(tl.int64)
    # Preserve the packed-column clamp for candidates beyond the logits width.
    block = tl.where(start + block * BLOCK_SIZE >= width, nblocks, block)
    tl.debug_barrier()
    tl.store(flags + row * (nblocks + 1) + block, 1, (cols < K) & (block >= 0))


@triton.jit(do_not_specialize=["width", "nblocks"])
def _mask_candidates_kernel(
    logits,
    starts,
    ends,
    flags,
    stride_row,
    stride_col,
    stride_start,
    stride_end,
    width,
    nblocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    ROW_REPEAT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    start = tl.load(starts + row // ROW_REPEAT * stride_start) if HAS_STARTS else 0
    end = tl.load(ends + row // ROW_REPEAT * stride_end)
    valid = (cols >= start) & (cols < end) & (cols < width)
    block = (cols - start) // BLOCK_SIZE
    keep = tl.load(flags + row * (nblocks + 1) + block, valid, other=0)
    edge = tl.load(flags + row * (nblocks + 1) + nblocks)
    keep = (keep != 0) | ((cols == width - 1) & (edge != 0))
    tl.store(
        logits + row * stride_row + cols * stride_col,
        -float("inf"),
        (cols < width) & ~(valid & keep),
    )


def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    scores = logits.new_empty((rows, nblocks))
    _block_scores_kernel[(rows, triton.cdiv(nblocks, 128))](
        logits,
        row_ks,
        row_ke,
        scores,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        128,
    )
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )


def select_candidate_block_ids(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
) -> torch.Tensor:
    """Block-level top-k over packed logits, as sorted ascending int32 ids.

    Exact torch port of the SM90 candidate selector: pad the row to a whole
    number of blocks with ``-inf``, ``amax`` per block (NaN-propagating),
    force ``+inf`` on the newest visible block, then ``torch.topk`` -- the same
    padded amax / forced block / topk call as :func:`select_candidate_blocks`,
    so the tie-breaking matches.  Blocks whose score is not finite (padding or
    NaN) become the out-of-range sentinel ``num_blocks`` instead of allocating
    a token-sized boolean mask.  The ids are sorted ascending, so the valid
    prefix is contiguous and the Remap kernel can consume them directly.

    Args:
        logits: ``[rows, width]`` fp32 packed logits; entries outside a row's
            causal window must already be ``-inf``.
        row_ks: ``[rows // row_repeat]`` int32 row starts, or None for zero.
        row_ke: ``[rows // row_repeat]`` int32 row ends.
        topk_blocks: Number of blocks to publish per row (the output width).
        block_size: Positions per candidate block.
        out: ``[rows, topk_blocks]`` int32 output buffer.
        row_repeat: Decode rows sharing one bound (spec-decode queries).

    Returns:
        ``out``.

    """
    rows, width = logits.shape
    if rows == 0:
        return out
    nblocks = triton.cdiv(width, block_size) if width else 0
    if nblocks == 0:
        out.fill_(0)
        return out

    if row_ks is None:
        start = torch.zeros(rows, dtype=torch.int64, device=logits.device)
    else:
        start = row_ks.repeat_interleave(row_repeat)[:rows].to(torch.int64)
    end = row_ke.repeat_interleave(row_repeat)[:rows].to(torch.int64)

    # Padded amax exactly as SGLang's select_candidate_block_ids.
    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    # The newest visible block is always published, even when its score is
    # negative: (end - start - 1) // block_size is the last visible block.
    last = (end - start - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(nblocks, device=logits.device) == last[:, None], torch.inf
    )
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    ids = torch.where(
        top.values > -torch.inf,
        top.indices,
        torch.full_like(top.indices, nblocks),
    ).to(torch.int32)
    ids = ids.sort(dim=-1).values
    out.fill_(nblocks)
    out[:, : ids.shape[1]] = ids
    return out


@triton.jit
def _finalize_candidate_topk_sm90_kernel(
    selected_ptr,
    scores_ptr,
    score_lens_ptr,
    block_table_ptr,
    candidate_blocks_ptr,
    page_indices_ptr,
    raw_indices_ptr,
    stride_sel,
    stride_score,
    stride_bt,
    stride_cb,
    stride_out,
    stride_raw,
    block_size,
    TOPK: tl.constexpr,
    SOURCE_WIDTH: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    HAS_RAW: tl.constexpr,
):
    """Map selected candidate-token columns to vLLM compressed positions.

    ``selected`` are TopK columns into the candidate-token logits (one row per
    query).  Candidate mode resolves column ``c`` to request-local position
    ``candidate_blocks[row, c // CBS] * CBS + c % CBS``; a selected column whose
    logit is ``-inf`` (padding, NaN or beyond the context) is dropped.  The
    physical slot of logical position ``L`` is
    ``block_table[row, L // block_size] * block_size + L % block_size``
    (RATIO == 1: no req_to_token indirection).  ``raw_indices`` receives the
    request-local positions; ``page_indices`` the physical slots.  Both are -1
    padded, sorted ascending because ``selected`` is sorted before the map.
    """
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, TOPK)
    selected = tl.load(selected_ptr + row * stride_sel + offs).to(tl.int64)
    length = tl.load(score_lens_ptr + row).to(tl.int64)
    valid = (selected >= 0) & (selected < SOURCE_WIDTH)
    score = tl.load(
        scores_ptr + row * stride_score + tl.maximum(selected, 0),
        mask=valid,
        other=-float("inf"),
    )
    valid = valid & (score > -float("inf"))
    if USE_CANDIDATES:
        block_col = selected // CANDIDATE_BLOCK_SIZE
        within = selected % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            candidate_blocks_ptr + row * stride_cb + tl.maximum(block_col, 0),
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (block >= 0)
        logical = block * CANDIDATE_BLOCK_SIZE + within
    else:
        logical = selected
    valid = valid & (logical < length)
    safe_logical = tl.where(valid, logical, 0)
    page = tl.load(
        block_table_ptr + row * stride_bt + safe_logical // block_size,
        mask=valid,
        other=0,
    ).to(tl.int64)
    slot = page * block_size + (safe_logical % block_size)
    tl.store(page_indices_ptr + row * stride_out + offs, tl.where(valid, slot, -1))
    if HAS_RAW:
        tl.store(
            raw_indices_ptr + row * stride_raw + offs, tl.where(valid, logical, -1)
        )


def finalize_candidate_topk_sm90(
    selected: torch.Tensor,
    scores: torch.Tensor,
    score_lens: torch.Tensor,
    block_table: torch.Tensor,
    page_indices: torch.Tensor,
    *,
    block_size: int,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 1,
    raw_indices: torch.Tensor | None = None,
) -> None:
    """Finalize a compact candidate TopK into vLLM indexer slots.

    Args:
        selected: ``[rows, k]`` int32 TopK columns over the candidate-token
            logits, ``-1`` padded.  ``k`` must equal ``page_indices.shape[1]``
            (the kernel's TOPK is a constexpr) and be a power of two.
        scores: ``[rows, SOURCE_WIDTH]`` fp32 candidate-token logits; used to
            reject selections whose score is not finite.
        score_lens: ``[rows]`` int32 visible (compressed) length per row.
        block_table: ``[rows, P]`` int32 page table (``stride(-1) == 1``).
        page_indices: ``[rows, k]`` int32 output of physical cache slots.
        block_size: Indexer cache tokens per page.
        candidate_blocks: ``[rows, K]`` int32 request-local candidate blocks
            (-1 padded).  None means ``selected`` is already a local position.
        candidate_block_size: Positions per candidate block.
        raw_indices: Optional ``[rows, k]`` int32 output of request-local
            compressed positions.  Must not alias ``selected``.

    """
    assert selected.dtype == torch.int32 and selected.stride(-1) == 1
    assert page_indices.dtype == torch.int32 and page_indices.stride(-1) == 1
    assert scores.dtype == torch.float32
    k = page_indices.shape[1]
    # The report flags this invariant as a risk: the Triton kernel's TOPK is a
    # constexpr and the store covers exactly page_indices.shape[1] columns.
    assert selected.shape[1] == k, (
        "finalize_candidate_topk_sm90 requires k == page_indices.shape[1], got "
        f"{selected.shape[1]} != {k}"
    )
    assert triton.next_power_of_2(k) == k, f"k must be a power of two, got {k}"
    rows = selected.shape[0]
    if rows == 0:
        return
    use_candidates = candidate_blocks is not None
    assert score_lens.dtype == torch.int32 and score_lens.shape == (rows,)
    assert block_table.shape[0] == rows and block_table.stride(-1) == 1
    if raw_indices is not None:
        assert raw_indices.dtype == torch.int32 and raw_indices.stride(-1) == 1
        assert raw_indices.shape[0] == rows and raw_indices.shape[1] == k
    _finalize_candidate_topk_sm90_kernel[(rows,)](
        selected,
        scores,
        score_lens,
        block_table,
        candidate_blocks if use_candidates else selected,
        page_indices,
        raw_indices if raw_indices is not None else page_indices,
        selected.stride(0),
        scores.stride(0),
        block_table.stride(0),
        candidate_blocks.stride(0) if use_candidates else 0,
        page_indices.stride(0),
        raw_indices.stride(0) if raw_indices is not None else 0,
        block_size,
        TOPK=k,
        SOURCE_WIDTH=scores.shape[1],
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        USE_CANDIDATES=use_candidates,
        HAS_RAW=raw_indices is not None,
        num_warps=8,
    )


def apply_candidate_mask(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    candidate_blocks: torch.Tensor,
    block_size: int,
    row_repeat: int = 1,
) -> None:
    """Mask packed logits outside causal bounds and request-local candidates."""
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows or not width:
        return
    nblocks = triton.cdiv(width, block_size)
    flags = torch.empty((rows, nblocks + 1), device=logits.device, dtype=torch.uint8)
    start_stride = row_ks.stride(0) if row_ks is not None else 0
    _candidate_flags_kernel[(rows,)](
        candidate_blocks,
        row_ks,
        flags,
        *candidate_blocks.stride(),
        start_stride,
        width,
        nblocks,
        block_size,
        candidate_blocks.shape[1],
        row_ks is not None,
        row_repeat,
    )
    _mask_candidates_kernel[(rows, triton.cdiv(width, 1024))](
        logits,
        row_ks,
        row_ke,
        flags,
        *logits.stride(),
        start_stride,
        row_ke.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        1024,
    )
