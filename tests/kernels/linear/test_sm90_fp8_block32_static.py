# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the SM90 native MXFP8 block-32 static GEMM (WP B).

Reference: dequantise the E4M3 blocks with their per-32 ue8m0 scales and matmul
in float64, i.e. the mathematically exact value.

Tolerance design (why not ``assert_close(out, ref, atol=2e-2, rtol=2e-2)``)
-------------------------------------------------------------------------
Hopper's FP8 tensor core (the ``tl.dot`` the kernel is built from) does *not*
accumulate with IEEE fp32 precision: its internal accumulation loses roughly
100x more than a plain fp32 matmul.  Measured on this box, for the exact same
operands, ``tl.dot`` over K=32 already shows ~1e-5 relative error, and the full
GEMM shows ~1e-4..1e-3 relative to ``sum(|a*b|)``.  That error is a property of
the hardware MMA, not of this port: ``sm90_static_gemm`` is **bit-identical**
to the SGLang reference kernel it was ported from (same Triton source), for
every ``SWAP_AB``/``SPLIT_K`` config.

Consequently a tolerance relative to the output is meaningless on data with
cancellation: with ``randn`` operands the outputs have ``absmax ~ 3000`` but
individual elements land arbitrarily close to zero, where the *absolute* MMA
error (~0.2) dominates and the relative error is unbounded even for a perfect
kernel.  The meaningful, tight, non-vacuous statement is

    |out - exact| <= rtol * sum(|a*b|)          (rtol ~ 5e-4, measured 1.2e-4)

i.e. relative to the magnitude actually accumulated, which is the natural error
scale of the MMA. A kernel that indexed a scale, K block or tile incorrectly would
exceed this by orders of magnitude.

The bf16 output is checked separately and much more tightly: because only the
final store dtype differs, ``out_bf16`` must equal ``out_fp32`` rounded to
bf16 (at most one bf16 ulp on elements sitting on a rounding boundary).
"""

import types

import pytest
import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)

# The module is import-safe on any platform; only the kernel launches need SM90.
from vllm.model_executor.kernels.linear.mxfp8.sm90_static import (
    _MXFP8_BLOCK,
    _SM90_GENERIC_CONFIG,
    _SM90_STATIC_CONFIGS,
    Sm90StaticMxfp8BmmLinearKernel,
    Sm90StaticMxfp8LinearKernel,
    _reduce_block_fp8_split_k,
    select_sm90_static_config,
    sm90_static_gemm,
)
from vllm.platforms import current_platform

requires_sm90 = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
    ),
    reason="requires an SM90 (Hopper) GPU",
)

# Error of the FP8 tensor-core accumulation relative to sum(|a*b|).
# Measured worst case over the covered shapes/seeds: 1.2e-4 (fp32 out).
MMA_RTOL = 5e-4
# Same, for an output that has additionally been rounded to bf16.
# Measured worst case on the dynamically-quantised activation path: 7.8e-4.
MMA_RTOL_BF16 = 3e-3


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def _e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Host-side ue8m0 uint8 -> fp32 (same bit trick as the kernel)."""
    return (scale.to(torch.int32) << 23).view(torch.float32)


def _dequant64(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantise per-32 blocks in float64."""
    vf = values.to(torch.float64)
    sf = scale.to(torch.float64).repeat_interleave(_MXFP8_BLOCK, dim=-1)
    return vf * sf[..., : vf.shape[-1]]


def _exact_operands(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact dequantised A and B (B is [N, K], i.e. the weight layout)."""
    return _dequant64(A, As), _dequant64(B, _e8m0_to_fp32(Bs))


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Magnitude of one bf16 ulp for every element of ``x``."""
    exp = torch.floor(torch.log2(x.abs().clamp_min(torch.finfo(torch.float64).tiny)))
    return torch.exp2(exp - 7)  # 8 significant bf16 bits -> 2**(exp-7)


def _rand_fp8(shape: tuple[int, ...], dtype=torch.float8_e4m3fn) -> torch.Tensor:
    return (torch.randn(shape, device="cuda") * 0.5).to(dtype)


def _rand_act_scale(shape: tuple[int, ...]) -> torch.Tensor:
    # Exact powers of two so the scale multiply is exact in both paths.
    return torch.exp2(torch.randint(-6, 3, shape, device="cuda").float())


def _rand_weight_scale_uint8(shape: tuple[int, ...]) -> torch.Tensor:
    # ue8m0 bytes 120..133 -> 2**(-7) .. 2**6
    return torch.randint(120, 134, shape, device="cuda", dtype=torch.uint8)


def _assert_fp32_matches_exact(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    rtol: float = MMA_RTOL,
) -> None:
    """|out - exact| <= rtol * sum(|a*b|), elementwise."""
    Ad, Bd = _exact_operands(A, B, As, Bs)
    ref = Ad @ Bd.t()
    bound = Ad.abs() @ Bd.abs().t()
    err = (out.double() - ref).abs()
    bad = err > rtol * bound
    assert not bool(bad.any()), (
        f"{int(bad.sum())}/{err.numel()} elements exceed {rtol:g} x sum|a*b|; "
        f"max normalized error {(err / bound).max().item():.3e}"
    )


def _assert_bf16_matches_exact(
    out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    rtol: float = MMA_RTOL_BF16,
) -> None:
    """A bf16 output against the exact value: MMA error + bf16 output rounding."""
    Ad, Bd = _exact_operands(A, B, As, Bs)
    ref = Ad @ Bd.t()
    bound = Ad.abs() @ Bd.abs().t()
    err = (out.double() - ref).abs()
    tol = rtol * bound + 0.5 * _bf16_ulp(ref)
    bad = err > tol
    assert not bool(bad.any()), (
        f"{int(bad.sum())}/{err.numel()} bf16 elements out of tolerance; "
        f"max normalized error {(err / bound).max().item():.3e}"
    )


def _assert_bf16_is_rounded_fp32(
    out_bf16: torch.Tensor, out_fp32: torch.Tensor
) -> None:
    """bf16 output == fp32 output rounded to bf16 (<=1 ulp on boundary elements)."""
    expected = out_fp32.to(torch.bfloat16)
    neq = out_bf16 != expected
    n = int(neq.sum())
    if n == 0:
        return
    ulp = _bf16_ulp(out_fp32).to(torch.float32)
    diff = (out_bf16.float() - expected.float()).abs()
    assert bool((diff <= ulp).all()), "bf16 output is not the fp32 result rounded"
    # A double rounding (fp32 then bf16) can only differ on boundary elements.
    assert n <= max(1, out_bf16.numel() // 1000), (
        f"{n}/{out_bf16.numel()} elements differ from the fp32 rounding"
    )


# ---------------------------------------------------------------------------
# Selector (platform independent)
# ---------------------------------------------------------------------------
def test_selector_nearest_m_and_pairs():
    for (N, K), table in _SM90_STATIC_CONFIGS.items():
        for m, expected in table.items():
            assert select_sm90_static_config(N, K, m) == {
                **expected,
                "SWAP_AB": expected.get("SWAP_AB", False),
                "SPLIT_K": expected.get("SPLIT_K", 1),
            }
        # Every adjacent x/x+1 pair must round-trip to its own config.
        for x in table:
            if x + 1 in table:
                assert select_sm90_static_config(N, K, x) == {
                    **table[x],
                    "SWAP_AB": table[x].get("SWAP_AB", False),
                    "SPLIT_K": table[x].get("SPLIT_K", 1),
                }
                assert select_sm90_static_config(N, K, x + 1) == {
                    **table[x + 1],
                    "SWAP_AB": table[x + 1].get("SWAP_AB", False),
                    "SPLIT_K": table[x + 1].get("SPLIT_K", 1),
                }


def test_selector_unit_and_uncovered_fallback():
    assert select_sm90_static_config(5120, 5120, 1) == dict(_SM90_GENERIC_CONFIG)
    # Real DSV4.1 shape with no tuned entry -> generic, all-block-32 config.
    cfg = select_sm90_static_config(2304, 5120, 17)
    assert cfg == dict(_SM90_GENERIC_CONFIG)
    assert cfg["BLOCK_SIZE_K"] == _MXFP8_BLOCK


def test_covered_shapes_are_the_dsv41_ones():
    # These (N, K) appear in the DeepSeek-V4.1-Flash checkpoint / at TP8.
    covered = [(1280, 5120), (512, 5120), (4096, 1280), (25600, 6144), (5120, 15360)]
    for shape in covered:
        assert shape in _SM90_STATIC_CONFIGS


# ---------------------------------------------------------------------------
# Kernel correctness
# ---------------------------------------------------------------------------
@requires_sm90
@pytest.mark.parametrize(
    "N,K",
    [
        (1280, 5120),
        (512, 5120),
        (4096, 1280),
        (25600, 6144),
        (5120, 15360),
        (2304, 5120),  # uncovered -> generic config
    ],
)
@pytest.mark.parametrize("M", [1, 2, 6, 7, 24, 96, 97, 385])
def test_matches_torch_reference(N: int, K: int, M: int):
    A = _rand_fp8((M, K))
    B = _rand_fp8((N, K))
    As = _rand_act_scale((M, K // _MXFP8_BLOCK))
    Bs = _rand_weight_scale_uint8((N, K // _MXFP8_BLOCK))

    cfg = select_sm90_static_config(N, K, M)
    out32 = sm90_static_gemm(A, B, As, Bs, cfg, out_dtype=torch.float32)
    out16 = sm90_static_gemm(A, B, As, Bs, cfg, out_dtype=torch.bfloat16)

    _assert_fp32_matches_exact(out32, A, B, As, Bs)
    _assert_bf16_is_rounded_fp32(out16, out32)


@requires_sm90
@pytest.mark.parametrize(
    "N,K",
    [
        (1280, 5120),
        (512, 5120),
        (4096, 1280),
        (25600, 6144),
        (5120, 15360),
        (2304, 5120),
    ],
)
@pytest.mark.parametrize("M", [1, 7, 97, 385])
def test_matches_torch_reference_nonnegative(N: int, K: int, M: int):
    """All-nonnegative operands: the output is the whole accumulated magnitude,
    so a plain tight *relative-to-output* tolerance is meaningful and tight.

    This is the assertion the signed-data test cannot make (there, cancellation
    makes the per-element relative error unbounded even for an exact kernel).
    """
    A = (torch.randn((M, K), device="cuda").abs() * 0.5).to(torch.float8_e4m3fn)
    B = (torch.randn((N, K), device="cuda").abs() * 0.5).to(torch.float8_e4m3fn)
    As = _rand_act_scale((M, K // _MXFP8_BLOCK))
    Bs = _rand_weight_scale_uint8((N, K // _MXFP8_BLOCK))

    cfg = select_sm90_static_config(N, K, M)
    out32 = sm90_static_gemm(A, B, As, Bs, cfg, out_dtype=torch.float32)
    out16 = sm90_static_gemm(A, B, As, Bs, cfg, out_dtype=torch.bfloat16)

    Ad, Bd = _exact_operands(A, B, As, Bs)
    ref = Ad @ Bd.t()
    # measured worst relative error on this regime: 8e-5.
    torch.testing.assert_close(out32.double(), ref, atol=0.0, rtol=1e-3)
    _assert_bf16_is_rounded_fp32(out16, out32)


@requires_sm90
@pytest.mark.parametrize(
    "N,K,M",
    [
        (1280, 5120, 1),  # SWAP_AB, SPLIT_K=8
        (1280, 5120, 7),  # SWAP_AB, SPLIT_K=8
        (1280, 5120, 96),  # SWAP_AB, SPLIT_K=8
        (2304, 5120, 97),  # generic, SWAP_AB=False, SPLIT_K=1
        (512, 48, 33),  # K masking (K % 32 != 0)
        (5120, 15360, 97),  # SWAP_AB=False, SPLIT_K=1, large K
    ],
)
def test_single_k_block_indexing_is_bit_exact(N: int, K: int, M: int):
    """Exactly one K block is non-zero and all operand values are powers of two.

    Every product is then exact and the 32-term MMA sum is representable in
    fp32, so the whole GEMM output is exact regardless of accumulation order.
    Any wrong K-block offset, scale index, SWAP_AB transpose or split-K
    placement changes the result, so this pins the indexing exactly.
    """
    choices = torch.tensor(
        [1.0, 0.5, 0.25, 0.125, 0.0625], device="cuda", dtype=torch.float32
    )

    def pow2(shape: tuple[int, ...]) -> torch.Tensor:
        return choices[torch.randint(0, len(choices), shape, device="cuda")].to(
            torch.float8_e4m3fn
        )

    cfg = select_sm90_static_config(N, K, M)
    n_blocks = (K + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK
    split = cfg["SPLIT_K"]
    tiles_per_split = -(-n_blocks // split)
    # First/last block, and the first block of each split.
    blocks = sorted(
        {0, n_blocks - 1}
        | {min(i * tiles_per_split, n_blocks - 1) for i in range(split)}
    )
    groups = (K + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK
    for kb in blocks:
        A = torch.zeros((M, K), device="cuda").to(torch.float8_e4m3fn)
        B = torch.zeros((N, K), device="cuda").to(torch.float8_e4m3fn)
        width = min(_MXFP8_BLOCK, K - kb * _MXFP8_BLOCK)
        A[:, kb * _MXFP8_BLOCK : kb * _MXFP8_BLOCK + width] = pow2((M, width))
        B[:, kb * _MXFP8_BLOCK : kb * _MXFP8_BLOCK + width] = pow2((N, width))
        As = _rand_act_scale((M, groups))
        Bs = _rand_weight_scale_uint8((N, groups))

        out = sm90_static_gemm(A, B, As, Bs, cfg, out_dtype=torch.float32)
        Ad, Bd = _exact_operands(A, B, As, Bs)
        ref = Ad @ Bd.t()
        assert torch.equal(out, ref.float()), (
            f"non-exact result for (N={N}, K={K}, M={M}) at K block {kb}"
        )


@requires_sm90
@pytest.mark.parametrize("split_k", [1, 2, 4, 8])
def test_split_k_equivalence(split_k: int):
    N, K, M = 1280, 5120, 96
    A = _rand_fp8((M, K))
    B = _rand_fp8((N, K))
    As = _rand_act_scale((M, K // _MXFP8_BLOCK))
    Bs = _rand_weight_scale_uint8((N, K // _MXFP8_BLOCK))

    base = dict(_SM90_GENERIC_CONFIG, SPLIT_K=1)
    split = dict(_SM90_GENERIC_CONFIG, SPLIT_K=split_k)
    ref = sm90_static_gemm(A, B, As, Bs, base, out_dtype=torch.float32)
    out = sm90_static_gemm(A, B, As, Bs, split, out_dtype=torch.float32)
    # Both orders are held to the same exact-value bound.
    _assert_fp32_matches_exact(ref, A, B, As, Bs)
    _assert_fp32_matches_exact(out, A, B, As, Bs)


@requires_sm90
def test_split_k_reduce_kernel_matches_sum():
    S, E = 4, 1280 * 96
    partials = torch.randn(S, E, device="cuda", dtype=torch.float32)
    out = torch.empty(E, device="cuda", dtype=torch.float32)
    _reduce_block_fp8_split_k[((E + 255) // 256,)](partials, out, E, S, 256)
    torch.testing.assert_close(out, partials.sum(dim=0), atol=1e-4, rtol=1e-4)


@requires_sm90
def test_swap_ab_and_tuned_config_equivalence():
    N, K, M = 1280, 5120, 96
    A = _rand_fp8((M, K))
    B = _rand_fp8((N, K))
    As = _rand_act_scale((M, K // _MXFP8_BLOCK))
    Bs = _rand_weight_scale_uint8((N, K // _MXFP8_BLOCK))

    plain = sm90_static_gemm(
        A, B, As, Bs, dict(_SM90_GENERIC_CONFIG), out_dtype=torch.float32
    )
    tuned_cfg = select_sm90_static_config(N, K, M)
    assert tuned_cfg.get("SWAP_AB") is True and tuned_cfg["SPLIT_K"] == 8
    tuned = sm90_static_gemm(A, B, As, Bs, tuned_cfg, out_dtype=torch.float32)
    _assert_fp32_matches_exact(plain, A, B, As, Bs)
    _assert_fp32_matches_exact(tuned, A, B, As, Bs)


@requires_sm90
def test_k_masking_path():
    N, K, M = 512, 48, 33
    A = _rand_fp8((M, K))
    B = _rand_fp8((N, K))
    groups = (K + _MXFP8_BLOCK - 1) // _MXFP8_BLOCK
    As = _rand_act_scale((M, groups))
    Bs = _rand_weight_scale_uint8((N, groups))
    cfg = dict(_SM90_GENERIC_CONFIG)
    out32 = sm90_static_gemm(A, B, As, Bs, cfg, out_dtype=torch.float32)
    out16 = sm90_static_gemm(A, B, As, Bs, cfg, out_dtype=torch.bfloat16)
    _assert_fp32_matches_exact(out32, A, B, As, Bs)
    _assert_bf16_is_rounded_fp32(out16, out32)


# ---------------------------------------------------------------------------
# Linear / BMM class wiring
# ---------------------------------------------------------------------------


def _param(t: torch.Tensor) -> Parameter:
    # uint8 scales cannot require grad; Parameter() would default to True.
    return Parameter(t, requires_grad=False)


@requires_sm90
def test_linear_kernel_apply_weights(monkeypatch):
    N, K, M = 1280, 5120, 97
    layer = types.SimpleNamespace(
        weight=_param(_rand_fp8((N, K))),
        weight_scale=_param(_rand_weight_scale_uint8((N, K // _MXFP8_BLOCK))),
    )
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
    kernel = Sm90StaticMxfp8LinearKernel(Mxfp8LinearLayerConfig())
    kernel.process_weights_after_loading(layer)
    assert layer.weight.shape == (N, K)
    out = kernel.apply_weights(layer, x)

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    x_q, x_s = per_token_group_quant_fp8(x, _MXFP8_BLOCK, use_ue8m0=True)
    x_s = x_s.float()
    cfg = select_sm90_static_config(N, K, M)
    # Wiring: same quantisation + same GEMM, only the store dtype differs.
    out32 = sm90_static_gemm(
        x_q, layer.weight, x_s, layer.weight_scale, cfg, out_dtype=torch.float32
    )
    _assert_bf16_is_rounded_fp32(out, out32)
    # Numerics against the exact value.
    _assert_bf16_matches_exact(out, x_q, layer.weight, x_s, layer.weight_scale)


@requires_sm90
@pytest.mark.parametrize("group", [1, 2])
def test_bmm_kernel_apply_weights(monkeypatch, group: int):
    N, K, T = 1024, 4096, 33
    total_n = group * N
    layer = types.SimpleNamespace(
        weight=_param(_rand_fp8((total_n, K))),
        weight_scale=_param(_rand_weight_scale_uint8((total_n, K // _MXFP8_BLOCK))),
    )
    x = (torch.randn(T, group, K, device="cuda") * 0.5).to(torch.bfloat16)
    kernel = Sm90StaticMxfp8BmmLinearKernel(
        Mxfp8LinearLayerConfig(bmm_batch_size=group)
    )
    kernel.process_weights_after_loading(layer)
    assert layer.weight.shape == (group, N, K)
    out = kernel.apply_weights(layer, x)
    assert out.shape == (T, group, N)

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    x_q, x_s = per_token_group_quant_fp8(
        x.reshape(T * group, K), _MXFP8_BLOCK, use_ue8m0=True
    )
    x_q = x_q.view(T, group, K)
    x_s = x_s.float().view(T, group, K // _MXFP8_BLOCK)
    cfg = select_sm90_static_config(N, K, T)
    for g in range(group):
        out32_g = sm90_static_gemm(
            x_q[:, g, :],
            layer.weight[g],
            x_s[:, g, :],
            layer.weight_scale[g],
            cfg,
            out_dtype=torch.float32,
        )
        _assert_bf16_is_rounded_fp32(out[:, g, :], out32_g)
        _assert_bf16_matches_exact(
            out[:, g, :],
            x_q[:, g, :],
            layer.weight[g],
            x_s[:, g, :],
            layer.weight_scale[g],
        )


# ---------------------------------------------------------------------------
# Hardware selection and linear/BMM admission.
# ---------------------------------------------------------------------------
def test_cuda_priority_list_prefers_sm90_static_by_default():
    from vllm.model_executor.kernels.linear import (
        MarlinMxfp8LinearKernel,
        _cuda_mxfp8_kernels,
    )

    cuda = _cuda_mxfp8_kernels()
    assert cuda.index(Sm90StaticMxfp8LinearKernel) < cuda.index(MarlinMxfp8LinearKernel)


def test_backend_map_filters_to_the_sm90_static_kernels():
    from vllm.model_executor.kernels.linear import (
        _LINEAR_BACKEND_KERNEL_MAP,
        DeepGemmMxfp8BmmLinearKernel,
        EmulationMxfp8LinearKernel,
        _filter_kernels_by_backend,
    )

    assert (
        Sm90StaticMxfp8LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["mxfp8_sm90_static"]
    )
    possible = [DeepGemmMxfp8BmmLinearKernel, EmulationMxfp8LinearKernel]
    candidates = [*possible, Sm90StaticMxfp8BmmLinearKernel]
    filtered = _filter_kernels_by_backend("mxfp8_sm90_static", candidates)
    assert filtered == [Sm90StaticMxfp8BmmLinearKernel]


@pytest.mark.parametrize(
    "cuda,family,expected",
    [(True, 90, True), (True, 100, False), (True, 120, False), (False, 90, False)],
)
def test_platform_controls_is_supported(monkeypatch, cuda, family, expected):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda fam: fam == family
    )
    assert Sm90StaticMxfp8LinearKernel.is_supported()[0] is expected
    assert Sm90StaticMxfp8BmmLinearKernel.is_supported()[0] is expected
    assert (
        Sm90StaticMxfp8LinearKernel.can_implement(
            Mxfp8LinearLayerConfig(bmm_batch_size=2)
        )[0]
        is False
    )
    assert (
        Sm90StaticMxfp8BmmLinearKernel.can_implement(
            Mxfp8LinearLayerConfig(bmm_batch_size=2)
        )[0]
        is True
    )


@requires_sm90
def test_init_selects_sm90_static_by_default():
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel

    assert type(init_mxfp8_linear_kernel()) is Sm90StaticMxfp8LinearKernel


def _triton_variant_count(jit_fn) -> int:
    """Number of compiled Triton variants cached for a @triton.jit function."""
    caches = getattr(jit_fn, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton build does not expose device_caches")
    assert caches is not None
    count = 0
    for entry in caches.values():
        count += len(entry[0])
    return count


@requires_sm90
def test_no_per_m_compilation_including_splitk_reduction():
    """Distinct token counts must not trigger new compilations.

    ``M`` is a runtime argument on the GEMM and the SplitK reduction because
    vLLM feeds this kernel an open-ended set of token counts (every eager
    prefill tail, every mixed-batch size).  A constexpr ``M`` compiled a fresh
    binary per shape, and the first request of each new bucket paid a JIT spike
    that CUDA-graph capture cannot absorb.

    The reduction is checked separately and with a ``SPLIT_K > 1`` config: its
    ``ELEMENTS == M * N`` was a constexpr too, so fixing only the main GEMM
    still left the ``SPLIT_K > 1`` entries (including the tuned 1280x5120 one)
    recompiling per M.  Testing a ``SPLIT_K == 1`` config would miss it.

    Asserting on the *compiled-variant count* rather than wall time keeps this
    independent of machine noise.
    """
    from vllm.model_executor.kernels.linear.mxfp8.sm90_static import (
        _reduce_block_fp8_split_k,
        _w8a8_block_fp8_matmul_hopper_static,
    )

    N, K = 1280, 5120  # the tuned entry that uses SPLIT_K == 8
    cfg = select_sm90_static_config(N, K, 64)
    assert int(cfg["SPLIT_K"]) > 1, cfg

    # Crosses BLOCK_SIZE_M and no-masking/tail boundaries on purpose.
    Ms = [1, 7, 33, 64, 65, 100, 129, 256, 333, 512]

    def run(m: int) -> None:
        a = _rand_fp8((m, K))
        b = _rand_fp8((N, K))
        out = sm90_static_gemm(
            a,
            b,
            _rand_act_scale((m, K // _MXFP8_BLOCK)),
            _rand_weight_scale_uint8((N, K // _MXFP8_BLOCK)),
            cfg,
            out_dtype=torch.bfloat16,
        )
        assert out.shape == (m, N)

    run(Ms[0])
    torch.accelerator.synchronize()
    main0 = _triton_variant_count(_w8a8_block_fp8_matmul_hopper_static)
    red0 = _triton_variant_count(_reduce_block_fp8_split_k)

    for m in Ms[1:]:
        run(m)
        torch.accelerator.synchronize()

    main1 = _triton_variant_count(_w8a8_block_fp8_matmul_hopper_static)
    red1 = _triton_variant_count(_reduce_block_fp8_split_k)
    assert main1 == main0, (
        f"the main GEMM compiled {main1 - main0} extra variant(s) across "
        f"{len(Ms)} distinct M values; M must stay a runtime argument"
    )
    assert red1 == red0, (
        f"the SplitK reduction compiled {red1 - red0} extra variant(s) across "
        f"{len(Ms)} distinct M values; ELEMENTS must stay a runtime argument"
    )
