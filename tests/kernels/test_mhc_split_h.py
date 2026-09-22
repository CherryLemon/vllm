# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the SM90 split-H mHC post TileLang kernel (WP C).

The split-H variant grids over ``(tokens, ceildiv(hidden, h_blk))`` instead of
looping over hidden tiles inside one CTA, which is what makes the SM90 (H100)
post path usable for up to 240 tokens. It is selected only through the existing
``mhc_post_tilelang`` dispatcher; these tests pin both the numerics and the
dispatch/rejection contract.
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.kernels.mhc import tilelang as mhc_tilelang
from vllm.platforms import current_platform

try:  # TileLang may be absent on CPU-only test hosts.
    from vllm.model_executor.kernels.mhc import tilelang_kernels
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        _MHC_POST_SPLIT_H_TILELANG_KERNEL,
        MHC_SPLIT_H_BLOCK,
        MHC_SPLIT_H_THREADS,
    )

    HAS_TILELANG = True
except Exception:  # pragma: no cover - depends on the test host
    tilelang_kernels = None  # type: ignore[assignment]
    HAS_TILELANG = False

DEVICE = current_platform.device_type
SM90 = current_platform.is_cuda() and current_platform.is_device_capability_family(90)

SPLIT_H_HIDDEN = 5120
SPLIT_H_HC = 4


def mhc_post_split_h_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch fp32 reference with the kernel's serial channel order.

    ``out[c, h] = post[c] * x[h] + sum_i comb[i, c] * residual[i, h]`` accumulated
    in fp32 and cast to bf16 once, matching ``mhc_post_split_h_tilelang``.
    """
    acc = post.unsqueeze(-1) * x.float().unsqueeze(1)
    for i in range(SPLIT_H_HC):
        acc = acc + comb[:, i, :].unsqueeze(-1) * residual[:, i, :].unsqueeze(1)
    return acc.to(torch.bfloat16)


def _split_h_tensors(num_tokens: int, hidden: int, *, device=DEVICE):
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    residual = torch.randn(
        num_tokens, SPLIT_H_HC, hidden, dtype=torch.bfloat16, device=device
    )
    post = torch.randn(num_tokens, SPLIT_H_HC, dtype=torch.float32, device=device)
    comb = torch.randn(
        num_tokens, SPLIT_H_HC, SPLIT_H_HC, dtype=torch.float32, device=device
    )
    return x, residual, post, comb


# ---------------------------------------------------------------------------
# Dispatcher contract (runs on CPU by stubbing the kernels)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cuda,family,expected",
    [(True, 90, True), (True, 100, False), (True, 120, False), (False, 90, False)],
)
def test_guard_checks_platform(monkeypatch, cuda, family, expected):
    """Only Hopper CUDA devices select the split-H post kernel."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda fam: fam == family
    )
    x, residual, post, comb = _split_h_tensors(2, SPLIT_H_HIDDEN, device="cpu")
    assert mhc_tilelang._mhc_post_split_h_supported(x, residual, post, comb) is expected


@pytest.mark.parametrize("num_tokens", [0, 1, 64, 65, 240, 241])
def test_guard_token_bound(monkeypatch, num_tokens):
    """Outer admission is 1..240, including SGLang's 64/65 boundary."""
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    x, residual, post, comb = _split_h_tensors(num_tokens, SPLIT_H_HIDDEN, device="cpu")
    expected = 1 <= num_tokens <= 240
    assert mhc_tilelang._mhc_post_split_h_supported(x, residual, post, comb) is expected


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
def test_guard_hidden(monkeypatch, hidden):
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    x, residual, post, comb = _split_h_tensors(2, hidden, device="cpu")
    assert mhc_tilelang._mhc_post_split_h_supported(x, residual, post, comb) is (
        hidden == SPLIT_H_HIDDEN
    )


def test_guard_rejects_wrong_dtypes(monkeypatch):
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    n = 2
    x, residual, post, comb = _split_h_tensors(n, SPLIT_H_HIDDEN, device="cpu")

    assert mhc_tilelang._mhc_post_split_h_supported(x, residual, post, comb)

    fp16_x = x.to(torch.float16)
    assert not mhc_tilelang._mhc_post_split_h_supported(fp16_x, residual, post, comb)

    bad_post = post.to(torch.bfloat16)
    assert not mhc_tilelang._mhc_post_split_h_supported(x, residual, bad_post, comb)


def test_guard_rejects_bad_shapes(monkeypatch):
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    n = 2
    x, residual, post, comb = _split_h_tensors(n, SPLIT_H_HIDDEN, device="cpu")

    # post.numel() != N * hc
    assert not mhc_tilelang._mhc_post_split_h_supported(x, residual, post[:, :1], comb)
    # comb must be exactly (N, 4, 4)
    assert not mhc_tilelang._mhc_post_split_h_supported(x, residual, post, comb[:, :3])
    # residual must be exactly (N, 4, hidden)
    assert not mhc_tilelang._mhc_post_split_h_supported(x, residual[:, :3], post, comb)
    # 1-D input must be rejected rather than raising
    assert not mhc_tilelang._mhc_post_split_h_supported(x[:, 0], residual, post, comb)
    # both public post shapes (N, hc) and (N, hc, 1) are accepted
    assert mhc_tilelang._mhc_post_split_h_supported(
        x, residual, post.unsqueeze(-1), comb
    )


def test_guard_rejects_non_contiguous(monkeypatch):
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    n = 2
    x, residual, post, comb = _split_h_tensors(n, SPLIT_H_HIDDEN, device="cpu")

    wide = torch.randn(n, SPLIT_H_HIDDEN * 2, dtype=torch.bfloat16)
    assert not mhc_tilelang._mhc_post_split_h_supported(
        wide[:, :SPLIT_H_HIDDEN], residual, post, comb
    )
    residual_wide = torch.randn(n, SPLIT_H_HC, SPLIT_H_HIDDEN * 2, dtype=torch.bfloat16)
    assert not mhc_tilelang._mhc_post_split_h_supported(
        x, residual_wide[:, :, :SPLIT_H_HIDDEN], post, comb
    )
    assert not mhc_tilelang._mhc_post_split_h_supported(
        x, residual, post.t().contiguous().t(), comb
    )
    assert not mhc_tilelang._mhc_post_split_h_supported(
        x, residual, post, comb.transpose(1, 2)
    )


class _RecordingKernel:
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls

    def __call__(self, *args, **kwargs):
        self._calls.append(self._name)
        return None


@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang kernel module unavailable")
@pytest.mark.parametrize(
    "num_tokens,hidden,dtype,contiguous,expect_split",
    [
        (1, SPLIT_H_HIDDEN, torch.bfloat16, True, True),
        (240, SPLIT_H_HIDDEN, torch.bfloat16, True, True),
        (241, SPLIT_H_HIDDEN, torch.bfloat16, True, False),
        (2, 4096, torch.bfloat16, True, False),
        (2, SPLIT_H_HIDDEN, torch.float16, True, False),
        (2, SPLIT_H_HIDDEN, torch.bfloat16, False, False),
    ],
)
def test_dispatcher_selection(
    monkeypatch, num_tokens, hidden, dtype, contiguous, expect_split
):
    """The wrapper must pick split-H only for the claimed inputs."""
    calls: list[str] = []
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    monkeypatch.setattr(
        tilelang_kernels,
        "_MHC_POST_TILELANG_KERNEL",
        _RecordingKernel("post", calls),
    )
    monkeypatch.setattr(
        tilelang_kernels,
        "_MHC_POST_SPLIT_H_TILELANG_KERNEL",
        _RecordingKernel("split", calls),
    )

    x, residual, post, comb = _split_h_tensors(num_tokens, hidden, device="cpu")
    x = x.to(dtype)
    if not contiguous:
        wide = torch.randn(num_tokens, hidden * 2, dtype=dtype, device="cpu")
        x = wide[:, :hidden]

    mhc_tilelang.mhc_post_tilelang(x, residual, post, comb)
    assert calls == (["split"] if expect_split else ["post"])


@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang kernel module unavailable")
def test_split_h_compile_key(monkeypatch):
    """Warmup keys reduce h_blk by gcd and cover hidden=5120."""
    keys = _MHC_POST_SPLIT_H_TILELANG_KERNEL.get_warmup_keys(
        hidden_size=SPLIT_H_HIDDEN, hc_mult=SPLIT_H_HC
    )
    assert keys == [
        _MHC_POST_SPLIT_H_TILELANG_KERNEL.CompileKey(
            hidden_size=SPLIT_H_HIDDEN, hc_mult=SPLIT_H_HC, h_blk=1024
        )
    ]
    assert MHC_SPLIT_H_BLOCK == 1024
    assert MHC_SPLIT_H_THREADS == 128


# ---------------------------------------------------------------------------
# Kernel numerics (SM90 + TileLang required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SM90 or not HAS_TILELANG, reason="SM90 (H100) CUDA + TileLang required"
)
@pytest.mark.parametrize("num_tokens", [1, 2, 64, 65, 128, 240])
def test_mhc_post_split_h_matches_reference(num_tokens):
    torch.manual_seed(0)
    x, residual, post, comb = _split_h_tensors(num_tokens, SPLIT_H_HIDDEN)
    ref = mhc_post_split_h_ref(x, residual, post, comb)

    out = torch.empty_like(residual)
    _MHC_POST_SPLIT_H_TILELANG_KERNEL(
        comb,
        residual,
        post,
        x,
        out,
        SPLIT_H_HC,
        SPLIT_H_HIDDEN,
        MHC_SPLIT_H_BLOCK,
    )

    assert out.shape == residual.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not SM90 or not HAS_TILELANG, reason="SM90 (H100) CUDA + TileLang required"
)
@pytest.mark.parametrize("hidden", [1024, 2048, 4096, 5120])
def test_mhc_post_split_h_hidden_tiling(hidden):
    """Every hidden element is written exactly once; h_blk reduces by gcd."""
    torch.manual_seed(0)
    x, residual, post, comb = _split_h_tensors(3, hidden)
    ref = mhc_post_split_h_ref(x, residual, post, comb)

    out = torch.full_like(residual, float("nan"))
    _MHC_POST_SPLIT_H_TILELANG_KERNEL(
        comb, residual, post, x, out, SPLIT_H_HC, hidden, MHC_SPLIT_H_BLOCK
    )

    assert torch.isfinite(out).all(), "unwritten tiles left poison in the output"
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.skipif(
    not SM90 or not HAS_TILELANG, reason="SM90 (H100) CUDA + TileLang required"
)
def test_mhc_post_dispatcher_end_to_end():
    """The public wrapper returns the same values as the reference."""
    torch.manual_seed(0)
    x, residual, post, comb = _split_h_tensors(17, SPLIT_H_HIDDEN)
    ref = mhc_post_split_h_ref(x, residual, post, comb)

    out = mhc_tilelang.mhc_post_tilelang(x, residual, post.unsqueeze(-1), comb)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)
