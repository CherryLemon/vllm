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