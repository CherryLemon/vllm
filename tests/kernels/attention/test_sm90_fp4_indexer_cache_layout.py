# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Writer -> reader cache-layout test for the SM90 MXFP4 indexer.

Every other kernel test in this directory *hand-builds* the paged cache in the
layout the readers assume (``_packed_cache`` in ``test_sm90_fp4_indexer.py``).
That proves the readers agree with the test's idea of the layout, not with the
real producer: a writer/reader byte-layout mismatch would sail straight through
them.  This module drives the production writer,
``indexer_k_norm_rope_store(..., use_fp4_cache=True)``, and then reads the cache
back with the SM90 reader, pinning the two sides against each other.

Covered:

* **Non-identity page mapping.** Tokens are written into slots taken from a
  scrambled ``block_table``, and the reader gets that same table, so a
  page/offset mix-up anywhere in the chain shows up.
* **Padded page stride.** The cache is an ``as_strided`` view over a larger
  buffer, so ``stride(0) > page_size * 68``; both sides must use the tensor
  stride rather than assuming a packed block array.
* **Compressed layers** (``compress_ratio`` 1 and 2), where only group-boundary
  tokens produce a key.

The writer's contract, read off ``_indexer_k_norm_rope_quant_store_kernel``
(``vllm/models/deepseek_v41/common/ops/indexer_k_store.py``):

* a page is one contiguous blob -- ``[page_size * 64 payload][page_size * 4
  ue8m0]`` -- addressed as ``block_base + off*64 + i`` and
  ``block_base + page_size*64 + off*4 + s``;
* payload byte ``i`` holds element ``2i`` in the low nibble and ``2i+1`` in the
  high nibble (``_fp32x2_to_fp4x2(x_lo=even, x_hi=odd)``);
* scale byte ``s`` covers elements ``[32s, 32s + 32)``;
* the value is ``rms_norm(k_pre)`` with GPT-J RoPE on the trailing 64 dims at
  the group's first-token position, bf16-round-tripped twice.

Note the payload for token ``off`` starts at byte ``off*64``, *not*
``off*68``: payloads and scales are segregated, so a ``[page, off, 68]``-shaped
view is a convenience, not the writer's addressing.
"""

import pytest
import torch

from vllm.platforms import current_platform

HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
PAYLOAD_BYTES = HALF_D  # 64
SCALE_BYTES = HEAD_DIM // 32  # 4
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_DIM - ROPE_HEAD_DIM
HALF_ROPE = ROPE_HEAD_DIM // 2
PAGE_SIZE = 64

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _sm90() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and torch.cuda.is_available()
    )


requires_sm90 = pytest.mark.skipif(
    not _sm90(), reason="requires an SM90 (Hopper) CUDA device"
)


# ---------------------------------------------------------------------------
# Reference implementation of the writer's maths (torch)
# ---------------------------------------------------------------------------


def _rms_norm(k_pre: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = k_pre.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight.to(torch.float32)


def _gptj_rope(
    k: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor, ratio: int
) -> torch.Tensor:
    """RoPE on the trailing ``ROPE_HEAD_DIM`` dims; cache is cos then sin."""
    pairs = k.view(k.shape[0], HEAD_DIM // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    nope_pairs = NOPE_HEAD_DIM // 2
    pair_idx = torch.arange(HEAD_DIM // 2, device=k.device)
    is_rope = pair_idx >= nope_pairs
    cs_idx = (pair_idx - nope_pairs).clamp(min=0)
    compressed = (positions // ratio) * ratio
    cs = cos_sin_cache[compressed]
    # The kernel gathers with a pair-wide index and masks the non-RoPE lanes,
    # so build the same padded pair-wide source and mask it the same way.
    num_pairs = HEAD_DIM // 2
    cos_src = torch.zeros(cs.shape[0], num_pairs, device=cs.device, dtype=cs.dtype)
    sin_src = torch.zeros_like(cos_src)
    cos_src[:, :HALF_ROPE] = cs[:, :HALF_ROPE]
    sin_src[:, :HALF_ROPE] = cs[:, HALF_ROPE:]
    cos_v = torch.where(is_rope, cos_src[:, cs_idx], torch.ones_like(cos_src))
    sin_v = torch.where(is_rope, sin_src[:, cs_idx], torch.zeros_like(sin_src))
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    # The kernel round-trips both through bf16 before quantising.
    new_even = new_even.to(torch.bfloat16).to(torch.float32)
    new_odd = new_odd.to(torch.bfloat16).to(torch.float32)
    return torch.stack([new_even, new_odd], dim=-1).reshape(k.shape)


def _fp32_to_e2m1_code_rne(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even E2M1 code, matching ``_fp32x2_to_fp4x2``."""
    sign = (x < 0).to(torch.int64)
    a = x.abs().clamp(max=6.0)
    table = torch.tensor(_E2M1, dtype=torch.float32, device=x.device)
    idx = torch.searchsorted(table, a.contiguous(), right=False)
    idx_lo = (idx - 1).clamp(min=0)
    idx_hi = idx.clamp(max=len(_E2M1) - 1)
    lo, hi = table[idx_lo], table[idx_hi]
    d_lo, d_hi = (a - lo).abs(), (hi - a).abs()
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (idx_hi % 2 == 0) & (idx_lo % 2 != 0))
    code = torch.where(take_hi, idx_hi, idx_lo)
    return (code | (sign << 3)).to(torch.int64)


def _mxfp4_quantize_reference(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-32 MXFP4 exactly as the writer packs it."""
    rows = k.shape[0]
    n_blocks = HEAD_DIM // 32
    x = k.view(rows, n_blocks, 32)
    even, odd = x[..., 0::2], x[..., 1::2]
    amax = torch.maximum(even.abs().amax(dim=-1), odd.abs().amax(dim=-1))
    amax = torch.maximum(amax, torch.full_like(amax, 6.0 * (2.0**-126)))
    log2_ratio = torch.ceil(torch.log2(amax * (1.0 / 6.0))).clamp(-127.0, 127.0)
    inv_scale = torch.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(torch.uint8)
    lo = _fp32_to_e2m1_code_rne(even * inv_scale[:, :, None])
    hi = _fp32_to_e2m1_code_rne(odd * inv_scale[:, :, None])
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return packed.to(torch.uint8).reshape(rows, PAYLOAD_BYTES), ue8m0


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    mag = (codes & 0x7).to(torch.int64)
    sign = (codes & 0x8).to(torch.int64)
    values = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)[mag]
    return torch.where(sign != 0, -values, values).to(torch.float32)


def _decode_packed(packed: torch.Tensor, ue8m0: torch.Tensor) -> torch.Tensor:
    """Decode the reference packing into ``[rows, HEAD_DIM]`` fp32."""
    rows = packed.shape[0]
    scale = torch.exp2(ue8m0.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    out = torch.empty((rows, HEAD_DIM), dtype=torch.float32, device=packed.device)
    out[:, 0::2] = _decode_e2m1(packed & 0x0F) * scale[:, 0::2]
    out[:, 1::2] = _decode_e2m1((packed >> 4) & 0x0F) * scale[:, 1::2]
    return out


def _reader_layout_dequant(
    cache: torch.Tensor, slots: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Decode slots using the READER's flat-byte addressing.

    Reproduces exactly what ``_sm90_fp4_paged_index_logits_kernel`` does:
    ``row_base = page * stride(0)``, payload at ``row_base + off*64 + i``,
    scales at ``row_base + page_size*64 + off*4 + i//16``.  The flat byte offset
    is then re-expressed in the ``[page, off, byte]`` view so the tensor strides
    are honoured instead of assumed.
    """
    blocks = torch.div(slots, page_size, rounding_mode="floor")
    offs = slots % page_size
    i = torch.arange(HALF_D, device=cache.device)

    def at(flat_within_page: torch.Tensor) -> torch.Tensor:
        # flat_within_page -> (page row, byte) inside the 3D view
        row = flat_within_page // cache.stride(1)
        col = flat_within_page % cache.stride(1)
        return cache[blocks[:, None], row, col]

    pay = at(offs[:, None] * PAYLOAD_BYTES + i[None, :])
    exps = at(
        page_size * PAYLOAD_BYTES + offs[:, None] * SCALE_BYTES + (i[None, :] // 16)
    )
    scale = torch.exp2(exps.to(torch.float32) - 127.0)
    out = torch.empty(
        (slots.numel(), HEAD_DIM), dtype=torch.float32, device=cache.device
    )
    out[:, 0::2] = _decode_e2m1(pay & 0x0F) * scale
    out[:, 1::2] = _decode_e2m1((pay >> 4) & 0x0F) * scale
    return out


def _make_cache(num_blocks: int, page_pad: int, device) -> torch.Tensor:
    """``[num_blocks, PAGE_SIZE, 68]`` uint8 with a padded block stride."""
    flat = torch.zeros(
        num_blocks * (PAGE_SIZE * 68 + page_pad), dtype=torch.uint8, device=device
    )
    return torch.as_strided(
        flat,
        size=(num_blocks, PAGE_SIZE, 68),
        stride=(PAGE_SIZE * 68 + page_pad, 68, 1),
    )


def _cos_sin_cache(max_pos: int, device) -> torch.Tensor:
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    denom = 10000.0 ** (torch.arange(0, HALF_ROPE, device=device) / HALF_ROPE)
    ang = pos[:, None] / denom[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1)


def _reference_logits_from_k(
    q_values: torch.Tensor,
    q_scale: torch.Tensor,
    k_matrix: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """The SGLang bf16 rounding chain, with K supplied instead of read back."""
    rows, heads, _ = q_values.shape
    qs = q_scale.view(torch.uint8).reshape(rows, heads, SCALE_BYTES)
    i = torch.arange(HALF_D, device=q_values.device)
    qscale = torch.exp2(qs[..., i // 16].to(torch.float32) - 127.0)
    q_even = (_decode_e2m1(q_values & 0x0F) * qscale).to(torch.bfloat16).float()
    q_odd = (_decode_e2m1((q_values >> 4) & 0x0F) * qscale).to(torch.bfloat16).float()
    k = k_matrix.reshape(rows, -1, HALF_D, 2)
    k_low = k[..., 0].to(torch.bfloat16).float()
    k_high = k[..., 1].to(torch.bfloat16).float()
    acc = torch.einsum("rhd,rwd->rwh", q_even, k_low)
    acc += torch.einsum("rhd,rwd->rwh", q_odd, k_high)
    s = acc.to(torch.bfloat16).float().clamp(min=0.0)
    s = (s * weights.to(torch.float32)[:, None, :]).to(torch.bfloat16).float()
    return s.sum(dim=-1).to(torch.bfloat16).float()


# ---------------------------------------------------------------------------
# Test 1: the writer's bytes, decoded with the reader's addressing
# ---------------------------------------------------------------------------


@requires_sm90
@pytest.mark.parametrize("page_pad", [0, 128])
@pytest.mark.parametrize("compress_ratio", [1, 2])
def test_writer_bytes_match_the_readers_layout(
    compress_ratio: int, page_pad: int, monkeypatch
):
    """Decode the production writer's bytes with the reader's addressing.

    This is the check the hand-built-cache tests cannot make: it fails if
    ``indexer_k_norm_rope_store`` and ``sm90_fp4_paged_index_logits`` disagree
    about where a token's payload or its block scales live, or about the
    nibble order within a payload byte.
    """
    from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store

    device = "cuda"
    torch.manual_seed(0)
    num_blocks = 8
    n_groups = 4
    num_tokens = n_groups * compress_ratio
    cache = _make_cache(num_blocks, page_pad, device)

    # A scrambled page order and non-contiguous offsets inside each page.
    perm = [5, 1, 6, 2, 7, 0, 4, 3]
    slots = [perm[g] * PAGE_SIZE + (g * 3) % PAGE_SIZE for g in range(n_groups)]

    slot_mapping = torch.zeros(num_tokens, device=device, dtype=torch.int64)
    for g in range(n_groups):
        # Only the group-boundary token emits a key.
        slot_mapping[(g + 1) * compress_ratio - 1] = slots[g]

    k_pre = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos_sin = _cos_sin_cache(1024, device)
    eps = 1e-20

    indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        weight,
        eps,
        cache,
        slot_mapping,
        compress_ratio,
        use_fp4_cache=True,
    )
    torch.accelerator.synchronize()

    # The intended K, in torch, following the writer's maths exactly.
    k = _gptj_rope(_rms_norm(k_pre, weight, eps), positions, cos_sin, compress_ratio)
    boundary = torch.tensor(
        [(g + 1) * compress_ratio - 1 for g in range(n_groups)], device=device
    )
    k_expected = k[boundary]

    got = _reader_layout_dequant(
        cache,
        torch.tensor(slots, device=device, dtype=torch.int64),
        PAGE_SIZE,
    )
    assert got.shape == (n_groups, HEAD_DIM)

    # (1) Bit-exact against the reference round-trip: this is the layout pin.
    packed, ue8m0 = _mxfp4_quantize_reference(k_expected)
    torch.testing.assert_close(got, _decode_packed(packed, ue8m0), rtol=0, atol=0)

    # (2) And the decoded values are a faithful MXFP4 encoding of the intended
    # K, which catches a reference that is self-consistently wrong.
    torch.testing.assert_close(got, k_expected.to(torch.float32), rtol=0.2, atol=0.7)

    # (3) Slots that were never written must stay zero.
    unwritten = [p * PAGE_SIZE + o for p in range(num_blocks) for o in range(2)]
    unwritten = [s for s in unwritten if s not in slots]
    if unwritten:
        zeros = _reader_layout_dequant(
            cache,
            torch.tensor(unwritten, device=device, dtype=torch.int64),
            PAGE_SIZE,
        )
        assert (zeros == 0).all(), "the writer touched a slot it was not given"


# ---------------------------------------------------------------------------
# Test 2: the SM90 reader consumes a writer-produced cache
# ---------------------------------------------------------------------------


@requires_sm90
def test_reader_consumes_writer_produced_cache(monkeypatch):
    """End to end through the real writer and the real reader.

    The oracle is built from the *intended* K (the writer's input passed through
    the reference quantiser), never from the cache, so the two sides are
    genuinely independent.
    """
    from tests.kernels.attention.test_sm90_fp4_indexer import (
        _rand_q,
        _valid_q_scale,
    )
    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )
    from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store

    device = "cuda"
    torch.manual_seed(7)
    rows, heads = 2, 32
    num_blocks, compress_ratio = 8, 2
    tokens_per_row = 2 * compress_ratio
    total_tokens = rows * tokens_per_row

    cache = _make_cache(num_blocks, 0, device)
    # Each request gets its own pages: sharing a block table would make the rows
    # overwrite each other's KV, which is a test bug, not a layout bug.
    perm = torch.tensor([4, 0, 6, 2, 7, 1, 5, 3], device=device, dtype=torch.int64)
    block_table = torch.stack(
        [perm[r * compress_ratio : (r + 1) * compress_ratio] for r in range(rows)]
    )
    assert block_table.shape == (rows, compress_ratio)
    assert block_table.unique().numel() == rows * compress_ratio

    slot_mapping = torch.zeros(total_tokens, device=device, dtype=torch.int64)
    positions = torch.arange(total_tokens, device=device, dtype=torch.int64)
    for r in range(rows):
        for g in range(compress_ratio):
            token = r * tokens_per_row + (g + 1) * compress_ratio - 1
            slot_mapping[token] = int(block_table[r, g].item()) * PAGE_SIZE + g

    k_pre = torch.randn(total_tokens, HEAD_DIM, device=device, dtype=torch.bfloat16)
    weight = torch.randn(HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos_sin = _cos_sin_cache(1024, device)
    indexer_k_norm_rope_store(
        k_pre,
        positions,
        cos_sin,
        weight,
        1e-20,
        cache,
        slot_mapping,
        compress_ratio,
        use_fp4_cache=True,
    )
    torch.accelerator.synchronize()

    q_values = _rand_q(rows, heads, device)
    q_scale = _valid_q_scale(rows, heads, device)
    weights = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    # Each request's *addressable* context is its own page-table row, not the
    # whole physical pool.  ``block_table`` here is (rows, compress_ratio) =
    # (2, 2), so a logical position >= 2 * PAGE_SIZE has no page-table entry:
    # the kernel would gather ``block_table_ptr + row * stride + L // PAGE_SIZE``
    # past the end of its row -- and past the 4-element tensor itself -- and then
    # translate that garbage page id into a K-cache address.  That is exactly the
    # illegal-access source this test used to feed the reader, so the visible
    # width must be the row capacity.
    width = block_table.shape[1] * PAGE_SIZE
    context_lens = torch.full((rows,), width, device=device, dtype=torch.int32)
    assert int(context_lens.max().item()) <= block_table.shape[1] * PAGE_SIZE, (
        "every visible logical position must have a page-table entry for its "
        "request; wider visibility would read past the row"
    )

    logits = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        page_size=PAGE_SIZE,
        width=width,
    )
    assert logits.shape == (rows, width)

    # Oracle from the intended K, placed at the columns the reader must read.
    k_intended = _gptj_rope(
        _rms_norm(k_pre, weight, 1e-20), positions, cos_sin, compress_ratio
    )
    # The reader sees the *quantised* key, so the oracle must be built from the
    # reference round-trip, not from the raw intended K.
    k_intended = _decode_packed(*_mxfp4_quantize_reference(k_intended))
    k_rows = torch.zeros(rows, width, HEAD_DIM, device=device)
    written = torch.zeros(rows, width, dtype=torch.bool, device=device)
    for r in range(rows):
        for g in range(compress_ratio):
            token = r * tokens_per_row + (g + 1) * compress_ratio - 1
            # `slot = block_table[r, c // 64] * 64 + c % 64`, and the writer put
            # this token at page `block_table[r, g]`, offset `g`.  Invert the
            # mapping to get the *logical column* the reader will use -- the
            # slot itself is not the column.
            assert int(block_table[r, g].item()) == perm[r * compress_ratio + g].item()
            # The reader addresses column `c` through block `c // 64`; the token
            # sits at block index g, offset g, so its logical column is 65g.
            col = g * PAGE_SIZE + g
            k_rows[r, col] = k_intended[token]
            written[r, col] = True
    ref = _reference_logits_from_k(q_values, q_scale, k_rows, weights)
    ref = torch.where(written, ref, float("-inf"))

    # Compare the columns the writer actually filled.  Columns it did not fill
    # are read from zeroed cache bytes and legitimately produce a finite
    # constant, so there is nothing to assert about them beyond their being
    # different from the written ones.
    torch.testing.assert_close(logits[written], ref[written], rtol=0, atol=2e-2)
    if written.sum() > 1:
        assert logits[written].unique().numel() > 1, (
            "every written column produced the same logit: the reader is not "
            "following the block table"
        )
    assert torch.equal(logits, logits.to(torch.bfloat16).to(torch.float32))
