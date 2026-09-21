# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 (Hopper) MXFP4 sparse-indexer kernel tests.

The Triton kernel tests only run on family(90) with ``VLLM_SM90_FP4_INDEXER=1``
and skip cleanly everywhere else.  The candidate-selector and remap-invariant
tests are pure torch and run on any platform.
"""

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    finalize_candidate_topk_sm90,
    select_candidate_block_ids,
)
from vllm.platforms import current_platform

HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
PAYLOAD_BYTES = 64
SCALE_BYTES = 4
PAGE_SIZE = 64
# Valid, near-unity UE8M0 exponent range used by the generated caches/scales.
_SCALE_EXP_LO = 123
_SCALE_EXP_HI = 125

# E2M1 magnitudes, indexed by the 3 magnitude bits; the sign bit is separate.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Reference E2M1 decode, integer-exact like the kernel."""
    mag = (codes & 0x7).to(torch.int64)
    sign = (codes & 0x8).to(torch.int64)
    values = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)[mag]
    return torch.where(sign != 0, -values, values).to(torch.float32)


def _packed_cache(num_blocks: int, page_size: int, device, dtype=torch.uint8):
    """Build a random cache in the documented segregated MXFP4 layout."""
    payload = torch.randint(
        0, 256, (num_blocks, page_size * PAYLOAD_BYTES), device=device, dtype=dtype
    )
    # Valid UE8M0 exponents kept close to 127.  Widening this range (e.g. up to
    # 135) drives the logits to ~1e5; tl.dot's tensor-core accumulation then
    # legitimately differs from the torch einsum reference by a few fp32 ulps,
    # which flips the final bf16 rounding by one ulp and exceeds the design's
    # atol=2e-2.  Near unity the two agree bit-exactly, so the comparison stays
    # exact rather than accidentally loose.
    scales = torch.randint(
        _SCALE_EXP_LO,
        _SCALE_EXP_HI + 1,
        (num_blocks, page_size * SCALE_BYTES),
        device=device,
        dtype=dtype,
    )
    flat = torch.cat([payload, scales], dim=1)
    return flat.reshape(num_blocks, page_size, PAYLOAD_BYTES + SCALE_BYTES)


def _valid_q_scale(rows: int, heads: int, device) -> torch.Tensor:
    """``[rows, heads]`` int32 whose 4 UE8M0 bytes are valid intra-range exponents.

    ``randint(-2**31, 2**31 - 1)`` also produces byte 255, and
    ``exp2(255 - 127) == inf``; a zero nibble then yields ``0 * inf = NaN`` and
    the comparison is meaningless.  Real packed MXFP4 Q scales never do this.
    """
    b = torch.randint(
        _SCALE_EXP_LO,
        _SCALE_EXP_HI + 1,
        (rows, heads, SCALE_BYTES),
        device=device,
        dtype=torch.uint8,
    )
    return b.contiguous().view(torch.int32).reshape(rows, heads)


def _reference_dequant_k(
    cache: torch.Tensor, slots: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Dequantize one slot's 128 K elements to fp32, documented layout.

    Payload byte ``i`` of a slot holds element ``2i`` (low nibble) and
    ``2i + 1`` (high nibble); scale byte ``i // 16`` feeds both.
    """
    blocks = torch.div(slots, page_size, rounding_mode="floor")
    offs = slots % page_size
    # The page is a flat byte array: [page_size * 64 payload][page_size * 4
    # scales].  Indexing the 3D tensor by [block, off, byte] would instead
    # assume an interleaved 68-byte token stride, which is NOT the writer's
    # layout, so flatten the page explicitly.
    page = cache.reshape(cache.shape[0], -1)
    i = torch.arange(HALF_D, device=cache.device)
    pay = page[blocks[:, None], offs[:, None] * PAYLOAD_BYTES + i[None, :]]
    exps = page[
        blocks[:, None],
        page_size * PAYLOAD_BYTES + offs[:, None] * SCALE_BYTES + i[None, :] // 16,
    ]
    scale = torch.exp2(exps.to(torch.float32) - 127.0)
    low = _decode_e2m1(pay & 0x0F) * scale
    high = _decode_e2m1((pay >> 4) & 0x0F) * scale
    out = torch.empty(
        (slots.shape[0], HEAD_DIM), dtype=torch.float32, device=cache.device
    )
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


def _slot_of(block_table: torch.Tensor, row: int, logical: int) -> int:
    """RATIO == 1 physical slot formula (no req_to_token)."""
    page = int(block_table[row, logical // PAGE_SIZE])
    return page * PAGE_SIZE + logical % PAGE_SIZE


def _reference_logits(
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    n_vis: int,
) -> torch.Tensor:
    """Pure-torch fp32 reference with the SGLang bf16 rounding chain."""
    rows, heads, _ = q_values.shape
    qs = q_scale.view(torch.uint8).reshape(rows, heads, SCALE_BYTES)
    i = torch.arange(HALF_D, device=q_values.device)
    qscale = torch.exp2(qs[..., i // 16].to(torch.float32) - 127.0)
    q_even = (_decode_e2m1(q_values & 0x0F) * qscale).to(torch.bfloat16).float()
    q_odd = (_decode_e2m1((q_values >> 4) & 0x0F) * qscale).to(torch.bfloat16).float()

    k = _reference_dequant_k(cache, slots.reshape(-1), PAGE_SIZE)
    k = k.reshape(rows, -1, HALF_D, 2)
    k_low = k[..., 0].to(torch.bfloat16).float()
    k_high = k[..., 1].to(torch.bfloat16).float()
    acc = torch.einsum("rhd,rwd->rwh", q_even, k_low)
    acc += torch.einsum("rhd,rwd->rwh", q_odd, k_high)
    s = acc.to(torch.bfloat16).to(torch.float32)
    s = torch.clamp(s, min=0.0)
    s = (s * weights.to(torch.float32)[:, None, :]).to(torch.bfloat16).to(torch.float32)
    logits = s.sum(dim=-1).to(torch.bfloat16).to(torch.float32)
    if n_vis < logits.shape[1]:
        logits[:, n_vis:] = float("-inf")
    return logits


def _sm90_available() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and envs.VLLM_SM90_FP4_INDEXER
    )


requires_sm90 = pytest.mark.skipif(
    not _sm90_available(),
    reason="SM90 FP4 indexer requires family(90) and VLLM_SM90_FP4_INDEXER=1",
)


@pytest.fixture(scope="module", autouse=True)
def _workspace_manager():
    """The native indexer top-k backends take their scratch from vLLM's
    workspace manager, which the model runner normally initialises.  Any test
    that drives ``get_indexer_topk`` needs it, and without it the failure is an
    assertion deep inside the top-k rather than a test failure."""
    if not _sm90_available():
        yield
        return
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device("cuda"))
    yield


def _rand_q(rows: int, heads: int, device) -> torch.Tensor:
    return torch.randint(
        0, 256, (rows, heads, HALF_D), device=device, dtype=torch.uint8
    )


# ---------------------------------------------------------------------------
# GPU kernel tests
# ---------------------------------------------------------------------------


@requires_sm90
def test_paged_logits_match_torch_reference():
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )

    torch.manual_seed(0)
    device = "cuda"
    rows, heads, page_size = 4, 32, PAGE_SIZE
    width = 4 * page_size
    num_blocks = 8
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    ).repeat(rows, 1)
    context_lens = torch.full((rows,), width, device=device, dtype=torch.int32)

    logits = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=width,
    )

    slots = torch.tensor(
        [[_slot_of(block_table, r, pos) for pos in range(width)] for r in range(rows)],
        device=device,
    )
    ref = _reference_logits(
        q_values, q_scale, cache, weights, slots.reshape(rows, width), width
    )
    # tl.dot's fp32 accumulation order differs from einsum; compare with the
    # fp32 tolerance the design specifies, and separately assert the bf16
    # rounding points are present (a kernel that skips them fails this).
    torch.testing.assert_close(logits, ref, rtol=0, atol=2e-2)
    assert torch.equal(logits, logits.to(torch.bfloat16).to(torch.float32))


@requires_sm90
def test_paged_logits_context_mask_and_prefill_workspace_agree():
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
        sm90_fp4_workspace_index_logits,
    )

    torch.manual_seed(1)
    device = "cuda"
    rows, heads, page_size = 4, 32, PAGE_SIZE
    num_blocks = 4
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    ).repeat(rows, 1)
    n_vis = 2 * page_size
    context_lens = torch.full((rows,), n_vis, device=device, dtype=torch.int32)

    paged = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
    )
    assert paged.shape == (rows, num_blocks * page_size)
    assert (paged[:, n_vis:] == float("-inf")).all()

    # Gather the same rows into the prefill workspace and score them.
    slots = torch.arange(n_vis, device=device, dtype=torch.int32)
    flat = cache.reshape(num_blocks, -1)
    k_values = flat[:, : page_size * PAYLOAD_BYTES].reshape(-1, PAYLOAD_BYTES)[slots]
    k_scales = flat[:, page_size * PAYLOAD_BYTES :].reshape(-1, SCALE_BYTES)[slots]
    workspace = sm90_fp4_workspace_index_logits(
        q_values,
        q_scale,
        weights,
        k_values,
        k_scales,
        torch.zeros(rows, dtype=torch.int32, device=device),
        torch.full((rows,), n_vis, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(workspace, paged[:, :n_vis], rtol=0, atol=0)


@requires_sm90
def test_candidate_scores_forced_newest_and_lens():
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )

    torch.manual_seed(2)
    device = "cuda"
    rows, heads, page_size = 2, 32, PAGE_SIZE
    num_blocks = 8
    cbs = 8
    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    ).repeat(rows, 1)
    k_cand = 4
    candidate_blocks = torch.randint(
        0, num_blocks, (rows, k_cand), device=device, dtype=torch.int32
    )
    candidate_blocks[1, -1] = -1  # -1 padded candidate must be skipped
    # Row 0 sees 3 blocks; row 1 sees 1.5 blocks.
    context_lens = torch.tensor(
        [3 * cbs, cbs + cbs // 2], device=device, dtype=torch.int32
    )

    logits, scores = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
    )
    assert logits.shape == (rows, k_cand * cbs)
    assert scores.shape == (rows, k_cand)
    # Forced +inf on the newest visible candidate block.
    for r in range(rows):
        last = (int(context_lens[r]) - 1) // cbs
        assert scores[r, last] == float("inf")
    # -1 padded candidate columns are -inf.
    blocked = candidate_blocks[1, -1] < 0
    if blocked:
        assert (logits[1, (k_cand - 1) * cbs :] == float("-inf")).all()


@requires_sm90
def test_compact_logits_cover_every_visible_token_with_unordered_candidates():
    """Compact candidate columns are *not* logical positions.

    Regression for the SM90 paged kernel's compact mask.  The production
    candidate publisher pins each row's newest -- possibly partial -- block
    first and does not sort the remaining ids, so valid compact columns
    routinely sit past ``n_vis`` while invalid ones sit below it.  Bounding the
    compact column by ``n_vis`` (correct in dense mode, where the column *is*
    the logical position) silently dropped every visible token whose compact
    column happened to land beyond ``n_vis``.

    Oracle: the same query scored in dense mode over the full cache is exactly
    what the compact path must reproduce, position by position.
    """
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )

    torch.manual_seed(7)
    device = "cuda"
    rows, heads, page_size = 2, 32, PAGE_SIZE
    num_blocks, cbs = 16, 8
    n_vis = 100
    # Newest (partial) block first, then the older full blocks; the trailing
    # -1 entries are padding and must stay -inf.
    candidates = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1]
    k_cand = len(candidates)
    width = k_cand * cbs

    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    ).repeat(rows, 1)
    context_lens = torch.full((rows,), n_vis, device=device, dtype=torch.int32)
    candidate_blocks = torch.tensor(candidates, device=device, dtype=torch.int32)
    candidate_blocks = candidate_blocks.reshape(1, -1).repeat(rows, 1)

    compact = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    # The compact consumer already owns its candidate ids: no block scores.
    assert isinstance(compact, torch.Tensor)
    assert compact.shape == (rows, width)

    dense = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=num_blocks * page_size,
    )

    # Every visible position must appear exactly once in the compact row: the
    # union of the mapped logical positions must be {0, ..., n_vis - 1}.
    cols = torch.arange(width, device=device)
    block_col = cols // cbs
    logic = (
        candidate_blocks[:, block_col].to(torch.int64) * cbs
        + (cols % cbs).to(torch.int64)
    )
    for r in range(rows):
        finite = compact[r] != float("-inf")
        mapped = logic[r][finite]
        assert mapped.numel() == n_vis, (
            f"row {r}: compact row exposes {mapped.numel()} finite columns, "
            f"expected {n_vis} visible tokens"
        )
        assert torch.equal(torch.sort(mapped).values, torch.arange(n_vis, device=device))

    # And each finite compact column must carry exactly the dense logits of the
    # position it maps to.
    for r in range(rows):
        finite = compact[r] != float("-inf")
        torch.testing.assert_close(
            compact[r][finite],
            dense[r][logic[r][finite]],
            rtol=0,
            atol=0,
        )

    # -1 padded candidate columns stay -inf even far inside the row width.
    assert (compact[:, 13 * cbs :] == float("-inf")).all()


@requires_sm90
def test_compact_decode_dspark_block5_rows_are_independent():
    """DSpark block5 verification shape: 6 query rows per request.

    DSpark drafts ``dspark_block_size`` (5) tokens, so the target verifies
    1 + 5 = 6 rows per request.  The SM90 FP4 indexer consumes one row per
    query (the metadata builder is forced to flatten for it), and each row
    carries its own acceptance-dependent visible length.  A row's finite
    columns must depend only on that row's own context, so a shorter accepted
    prefix in one row must not leak into another, and an all-padding row must
    come back entirely ``-inf``.
    """
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )

    torch.manual_seed(11)
    device = "cuda"
    page_size = PAGE_SIZE
    heads = 32
    num_blocks, cbs = 16, 8
    # One request's 6 verification rows: 5 draft positions + the bonus token,
    # with acceptance shrinking the visible length down the group.  The last
    # row is a pure padding row (idle rank / unused slot).
    n_vis_list = [100, 92, 77, 60, 33, 0]
    rows = len(n_vis_list)
    candidates = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1]
    width = len(candidates) * cbs

    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    ).repeat(rows, 1)
    context_lens = torch.tensor(n_vis_list, device=device, dtype=torch.int32)
    candidate_blocks = torch.tensor(candidates, device=device, dtype=torch.int32)
    candidate_blocks = candidate_blocks.reshape(1, -1).repeat(rows, 1)

    compact = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    dense = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=num_blocks * page_size,
    )

    cols = torch.arange(width, device=device)
    logic = (
        candidate_blocks[:, cols // cbs].to(torch.int64) * cbs
        + (cols % cbs).to(torch.int64)
    )
    for r, n_vis in enumerate(n_vis_list):
        finite = compact[r] != float("-inf")
        mapped = logic[r][finite]
        assert mapped.numel() == n_vis, (
            f"row {r} (n_vis={n_vis}) exposes {mapped.numel()} finite columns"
        )
        if n_vis == 0:
            assert not finite.any()
            continue
        assert torch.equal(
            torch.sort(mapped).values, torch.arange(n_vis, device=device)
        )
        torch.testing.assert_close(
            compact[r][finite], dense[r][logic[r][finite]], rtol=0, atol=0
        )


@requires_sm90
def test_compact_decode_empty_and_single_row_shapes():
    """Padding-only batches must not fault or read out of bounds.

    ``rows == 0`` is the idle-rank / no-decode-row case and ``topk_tokens``
    may exceed the compact width on a first step whose candidates are all
    padding.
    """
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )

    device = "cuda"
    page_size, heads, num_blocks, cbs = PAGE_SIZE, 32, 16, 8
    cache = _packed_cache(num_blocks, page_size, device)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    )
    q_values = _rand_q(1, heads, device)
    q_scale = _valid_q_scale(1, heads, device)
    weights = torch.randn(1, heads, device=device, dtype=torch.bfloat16)
    context_lens = torch.zeros(1, device=device, dtype=torch.int32)
    # All-padding candidates, with no visible token at all.
    candidate_blocks = torch.full((1, 16), -1, device=device, dtype=torch.int32)

    logits = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    assert logits.shape == (1, 16 * cbs)
    assert (logits == float("-inf")).all()

    empty = sm90_fp4_paged_index_logits(
        q_values[:0],
        q_scale[:0],
        cache,
        weights[:0],
        context_lens[:0],
        block_table[:0],
        candidate_blocks[:0],
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    assert empty.shape == (0, 16 * cbs)


def test_compact_valid_predicate_matches_dense_gather_oracle():
    """CPU form of the compact-mask contract (no CUDA needed).

    Encodes the kernel's predicate on the same shape the review used as a
    counterexample, so the coordinate-space rule is pinned down even where the
    Triton kernel cannot run.  ``test_compact_logits_cover_every_visible_token_
    with_unordered_candidates`` is the end-to-end guard; this one documents and
    enforces the contract in CPU-only CI.
    """
    cbs, n_vis = 8, 514
    candidates = [64] + list(range(64)) + [-1] * 8
    width = len(candidates) * cbs
    cols = torch.arange(width)
    block = torch.tensor(candidates)[cols // cbs]
    logical = block.to(torch.int64) * cbs + (cols % cbs).to(torch.int64)

    fixed = (cols < width) & (block >= 0) & (logical < n_vis)
    buggy = (cols < min(n_vis, width)) & (block >= 0) & (logical < n_vis)

    # Correct: exactly the n_vis visible logical positions survive.
    assert int(fixed.sum()) == n_vis
    assert torch.equal(torch.sort(logical[fixed]).values, torch.arange(n_vis))

    # The compact column index is not a logical position: the old predicate
    # dropped real candidates whose mapped position is still inside n_vis.
    assert int(buggy.sum()) < n_vis
    dropped = logical[fixed & ~buggy]
    assert dropped.numel() > 0
    assert bool((dropped < n_vis).all())


def test_compact_topk_uses_compact_length_not_logical_context():
    """The decode top-k must be driven by the compact row width.

    ``logits`` is indexed by compact candidate-matrix column while
    ``row_ke`` is a logical compressed-KV length; the top-k kernels plan their
    read range from ``seq_lens``/``max_seq_len``.  Feeding the logical length
    both reads past the compact row (out of bounds once the context exceeds the
    compact matrix) and hides valid candidates that sit past it.
    """
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer

    width, topk_tokens = 8, 4
    logits = torch.zeros(1, width, dtype=torch.float32)
    logits[0, width - 1] = 10.0
    selected = torch.empty(1, topk_tokens, dtype=torch.int32)
    topk = get_indexer_topk("torch")

    compact_lens = SparseMQAIndexer._compact_decode_lengths(
        1, width, torch.device("cpu")
    )
    assert compact_lens.dtype == torch.int32
    assert int(compact_lens.max()) == width
    topk(logits, compact_lens, 1, selected, topk_tokens, width)
    assert selected[0, 0].item() == width - 1

    logical_lens = torch.full((1, 1), 4, dtype=torch.int32)
    topk(logits, logical_lens, 1, selected, topk_tokens, 4)
    assert selected[0, 0].item() != width - 1, (
        "a logical context length must not be usable as the compact row length"
    )


# ---------------------------------------------------------------------------
# Pure-torch tests (run everywhere)
# ---------------------------------------------------------------------------


def test_select_candidate_block_ids_matches_topk_ties():
    logits = torch.tensor(
        [[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.float32
    )
    row_ke = torch.tensor([12], dtype=torch.int32)
    out = torch.empty(1, 2, dtype=torch.int32)
    select_candidate_block_ids(logits, None, row_ke, 2, 4, out)
    # Block maxes [4, 8, 12]; the newest block (2) is forced +inf and must be
    # selected; the sorted ids are ascending.
    assert out.tolist() == [[1, 2]]


def test_select_candidate_block_ids_order_independent_and_ties():
    # Two blocks with identical finite max: the torch.topk tie choice must be
    # reproduced, and permuting the input must not change the result set.
    logits = torch.tensor([[5.0, 5, 1, 1, 5, 5, 2, 2, 0, 0, 0, 0]], dtype=torch.float32)
    row_ke = torch.tensor([12], dtype=torch.int32)
    ref = torch.empty(1, 2, dtype=torch.int32)
    select_candidate_block_ids(logits, None, row_ke, 2, 4, ref)
    permuted = logits.clone()
    permuted[0, :4] = logits[0, 4:8]
    permuted[0, 4:8] = logits[0, :4]
    got = torch.empty(1, 2, dtype=torch.int32)
    select_candidate_block_ids(permuted, None, row_ke, 2, 4, got)
    # Forced block 2 is always present; blocks 0/1 tie, so per-article topk may
    # differ, but the forced newest block must survive in both.
    assert 2 in ref[0].tolist() and 2 in got[0].tolist()
    assert ref[0].tolist() == sorted(ref[0].tolist())
    assert got[0].tolist() == sorted(got[0].tolist())


def test_select_candidate_block_ids_nan_and_out_of_range():
    logits = torch.full((2, 8), float("nan"), dtype=torch.float32)
    row_ke = torch.tensor([8, 8], dtype=torch.int32)
    out = torch.empty(2, 4, dtype=torch.int32)
    select_candidate_block_ids(logits, None, row_ke, 4, 4, out)
    # num_blocks = 2 sentinel for NaN blocks; the forced newest block survives.
    assert out[0].tolist() == [1, 2, 2, 2]
    # An empty row publishes only the sentinel.
    logits_empty = torch.full((1, 8), float("-inf"), dtype=torch.float32)
    out_empty = torch.empty(1, 4, dtype=torch.int32)
    select_candidate_block_ids(
        logits_empty, None, torch.tensor([0], dtype=torch.int32), 4, 4, out_empty
    )
    assert out_empty[0].tolist() == [2, 2, 2, 2]


def test_finalize_candidate_topk_sm90_k_invariant():
    rows, width, k = 2, 16, 8
    selected = torch.full((rows, k), -1, dtype=torch.int32)
    scores = torch.zeros((rows, width), dtype=torch.float32)
    score_lens = torch.zeros(rows, dtype=torch.int32)
    block_table = torch.zeros((rows, 2), dtype=torch.int32)
    # The report flags this invariant explicitly: the kernel's TOPK is a
    # constexpr equal to page_indices.shape[1].
    with pytest.raises(AssertionError, match="k == page_indices.shape\\[1\\]"):
        finalize_candidate_topk_sm90(
            selected,
            scores,
            score_lens,
            block_table,
            torch.empty((rows, 4), dtype=torch.int32),
            block_size=16,
        )

# ---------------------------------------------------------------------------
# Full production chain: publisher -> compact logits -> native TopK -> remap
# ---------------------------------------------------------------------------


@requires_sm90
@pytest.mark.parametrize(
    "k_cand,n_vis,topk_tokens",
    [
        # Logical context is far wider than the compact matrix (the case the
        # old length-domain bug read out of bounds on).
        (16, 1000, 512),
        # Compact width just exceeds index_topk, so the top-k actually fills.
        (128, 1024, 512),
        # Everything is padding.
        (16, 0, 512),
    ],
)
def test_compact_decode_chain_matches_dense_candidate_oracle(
    k_cand: int, n_vis: int, topk_tokens: int
):
    """Drive the real publisher -> logits -> native TopK -> remap chain.

    The length-domain fix is only *proven* if the native TopK runs on the
    compact row and the remap agrees with a dense candidate-gather oracle.  A
    unit test of the length contract (``_compact_decode_lengths``) cannot catch
    a regression in the call site, and the compact/dense logits tests never run
    the top-k at all -- which is exactly where the out-of-bounds read was.

    Oracle: score every position densely, gather the candidate positions'
    values, take the top-k among the visible ones, and compare the resulting
    set of request-local logical positions.
    """
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer

    torch.manual_seed(1234 + k_cand)
    device = "cuda"
    page_size = PAGE_SIZE
    heads = 32
    num_blocks, cbs, rows = 32, 8, 3
    width = k_cand * cbs

    cache = _packed_cache(num_blocks, page_size, device)
    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    ).repeat(rows, 1)
    context_lens = torch.full((rows,), n_vis, device=device, dtype=torch.int32)

    # Production-shaped candidate order: the newest (partial) block first, then
    # the older blocks ascending, then -1 padding.  Not sorted by id.
    n_cand_blocks = max(1, (n_vis + cbs - 1) // cbs) if n_vis else 0
    newest = min(n_cand_blocks - 1, num_blocks * page_size // cbs - 1) if n_vis else -1
    cand_ids = []
    if newest >= 0:
        cand_ids.append(newest)
    cand_ids += [b for b in range(k_cand - 1) if b != newest and b < num_blocks]
    cand_ids = (cand_ids + [-1] * k_cand)[:k_cand]
    candidate_blocks = torch.tensor(cand_ids, device=device, dtype=torch.int32)
    candidate_blocks = candidate_blocks.reshape(1, -1).repeat(rows, 1)

    # --- production chain -------------------------------------------------
    logits = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=page_size,
        write_candidates=False,
    )
    assert logits.shape == (rows, width)

    compact_lens = SparseMQAIndexer._compact_decode_lengths(rows, width, logits.device)
    selected = torch.empty((rows, topk_tokens), dtype=torch.int32, device=device)
    get_indexer_topk("auto")(
        logits, compact_lens, 1, selected, topk_tokens, width
    )
    # The native top-k must only ever emit compact columns.
    assert int(selected.max()) < width, "top-k returned a column outside the row"
    page_scratch = torch.empty_like(selected)
    raw = torch.full((rows, topk_tokens), -1, dtype=torch.int32, device=device)
    finalize_candidate_topk_sm90(
        selected,
        logits,
        context_lens,
        block_table,
        page_scratch,
        block_size=page_size,
        candidate_blocks=candidate_blocks,
        candidate_block_size=cbs,
        raw_indices=raw,
    )

    # --- dense candidate-gather oracle ------------------------------------
    dense = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=page_size,
        width=num_blocks * page_size,
    )
    cols = torch.arange(width, device=device)
    block_col = cols // cbs
    logical = (
        candidate_blocks[:, block_col].to(torch.int64) * cbs
        + (cols % cbs).to(torch.int64)
    )
    valid = (candidate_blocks[:, block_col] >= 0) & (logical < n_vis)
    gathered = dense.gather(1, logical.clamp(min=0))
    gathered = torch.where(valid, gathered, float("-inf"))
    k = min(topk_tokens, width)
    top_vals, top_cols = gathered.topk(k, dim=-1)
    oracle = torch.where(
        top_vals > float("-inf"),
        logical.gather(1, top_cols),
        torch.full_like(top_cols, -1, dtype=torch.int64),
    )

    for r in range(rows):
        got = sorted(int(x) for x in raw[r].tolist() if x >= 0)
        want = sorted(int(x) for x in oracle[r].tolist() if x >= 0)
        assert got == want, (
            f"row {r} (k_cand={k_cand}, n_vis={n_vis}): the production chain and "
            f"the dense candidate oracle disagree\n  got  ({len(got)}): {got[:12]}\n"
            f"  want ({len(want)}): {want[:12]}"
        )
        # Every remapped position must be a real, visible, candidate position.
        for pos in got:
            assert pos < n_vis, f"remapped position {pos} is outside n_vis={n_vis}"
            assert (pos % cbs) < cbs and (pos // cbs) in cand_ids, pos
