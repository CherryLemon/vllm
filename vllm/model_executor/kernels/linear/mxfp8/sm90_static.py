# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native SM90 (Hopper) MXFP8 block-32 static GEMM.

This is the vLLM port of SGLang's shape-specialised Hopper block-FP8 GEMM

    python/sglang/kernels/ops/quantization/fp8_hopper_static.py

(spec: ``reports/port_design_ABC.md`` WP B, ``reports/sglang_delta_spec.md`` §B).
It runs W8A8: E4M3 weights with per-32 E8M0 (ue8m0) scales and a dynamically
quantised E4M3 activation, so DeepSeek-V4.1's SM90 linears no longer dequantise
to BF16 (Marlin) or run a plain BF16 GEMM (emulation).

Deviations from the SGLang source, both deliberate and numerically neutral:

* **Weight scale decode.** vLLM stores the weight scale as **uint8** ue8m0
  bytes (``MXFP8_SCALE_DTYPE``), not as a float tensor. The kernel decodes the
  byte in-register with the same bit trick used by
  ``fp8_utils._upcast_e8m0_to_fp32`` (``bits = exp << 23`` re-interpreted as
  fp32), so the scale value and the multiply order are bit-identical to the
  SGLang kernel loading a pre-upcast fp32 scale. This avoids materialising a
  4x-larger fp32 scale parameter and, crucially, keeps ``layer.weight_scale``
  in its registered uint8 dtype, which ``KMxfp8Static.process`` re-asserts on
  weight reload. (Design report B.4 option 2.)
* **``group_n=1``.** vLLM's loaded weight scale is ``[N, K // 32]``: the
  checkpoint's ``(32, 32)`` block scales are row-replicated by
  ``KMxfp8Static.get_scale_weight_loader`` (``repeat_interleave(block_rows)``),
  so row ``n`` carries the scale of block ``n // 32``. Passing ``group_n=1``
  makes the verbatim kernel's ``offs_bn // group_n`` index the per-row scale
  directly. The value is identical to indexing the un-replicated
  ``[N // 32, K // 32]`` checkpoint scale with ``group_n=32``.

Everything else -- split-K partials shaped ``(SPLIT_K, M, N)`` plus the reduce
kernel, the ``group_k`` scale stepping, the ``SWAP_AB`` accumulation order and
the per-K32 ``tl.dot`` -- is a faithful port.

Selection is **opt-in and default-off**; see ``Sm90StaticMxfp8LinearKernel``.
"""

import os

import torch
import triton
import triton.language as tl
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# Block size of the MXFP8 scale grid along K (and, on disk, along N).
_MXFP8_BLOCK = 32

# Explicit opt-in (VLLM_SM90_FP8_BLOCK32_STATIC, registered in vllm/envs.py).
# Default off: existing Marlin/Emulation selection on SM90 is untouched unless
# this is set. The raw environment is consulted as a fallback so the module also
# works when imported before the env registry is populated.
_SM90_STATIC_ENV = "VLLM_SM90_FP8_BLOCK32_STATIC"


def _sm90_static_enabled() -> bool:
    from vllm import envs

    value = getattr(envs, _SM90_STATIC_ENV, None)
    if value is not None:
        return bool(value)
    return os.environ.get(_SM90_STATIC_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ---------------------------------------------------------------------------
# Tuned (N, K) -> {M: config} table.
#
# Ported from the eleven H100 JSONs shipped by SGLang:
#   configs/N=<N>,K=<K>,device_name=NVIDIA_H100_80GB_HBM3,
#           dtype=fp8_w8a8,block_shape=[32, 32].json
# The M grid is irregular and every kernel-switch risk boundary has paired
# ``x``/``x + 1`` keys so that a nearest-M lookup cannot snap past a switch.
# DeepSeek-V4.1-Flash (hidden 5120) actually hits:
#   (1280, 5120)  wq_a
#   (512,  5120)  wkv
#   (4096, 1280)  wq_b @ TP8 (32768/8) and indexer.wq_b
#   (25600, 6144) engram.wkv
#   (5120, 15360) MTP main_proj
# The remaining table entries ((1536/1792/576, 5120), (16384, 1280),
# (5120, 288), (5120, 4096)) cover the same family at other TP/sizes.
# ---------------------------------------------------------------------------
_SM90_STATIC_CONFIGS: dict[tuple[int, int], dict[int, dict]] = {
    (1280, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (1536, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (16384, 1280): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        44: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        45: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        64: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        96: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        384: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        385: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        768: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        769: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (1792, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        64: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        80: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        81: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        96: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        384: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        385: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        768: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        769: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (25600, 6144): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        64: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        96: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (4096, 1280): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        32: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        33: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (512, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (5120, 15360): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (5120, 288): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (5120, 4096): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        44: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        45: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        64: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        97: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        128: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        320: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        321: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        384: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        385: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        768: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 1,
        },
        769: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
    (576, 5120): {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        6: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        24: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        64: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        96: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        128: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        192: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        256: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        384: {
            "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 8,
        },
        385: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        1536: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 4,
        },
        1537: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        3072: {
            "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": True, "SPLIT_K": 2,
        },
        3073: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
        4096: {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 32,
            "num_warps": 4, "num_stages": 3,
            "SWAP_AB": False, "SPLIT_K": 1,
        },
    },
}

# For (N, K) without a tuned entry. SWAP_AB/SplitK are off: correctness first,
# and any N is handled by the store mask.
_SM90_GENERIC_CONFIG: dict = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 3,
    "SWAP_AB": False,
    "SPLIT_K": 1,
}


def select_sm90_static_config(N: int, K: int, M: int) -> dict:
    """Pick the tuned config for (N, K, M), else the generic fallback.

    Mirrors SGLang's ``configs[min(configs, key=|M - key|)]`` lookup.
    """
    table = _SM90_STATIC_CONFIGS.get((N, K))
    if table is None:
        return dict(_SM90_GENERIC_CONFIG)
    chosen = table[min(table.keys(), key=lambda key: abs(key - M))]
    config = dict(chosen)
    config.setdefault("SWAP_AB", False)
    config.setdefault("SPLIT_K", 1)
    return config


# ---------------------------------------------------------------------------
# Verbatim port of SGLang fp8_hopper_static.py, with the uint8 ue8m0 decode of
# ``Bs`` added in-register (see module docstring).
# ---------------------------------------------------------------------------
@triton.jit
def _w8a8_block_fp8_matmul_hopper_static(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    split = tl.program_id(axis=1)
    tiles_per_split = tl.cdiv(tl.cdiv(K, BLOCK_SIZE_K), SPLIT_K)
    first_tile = split * tiles_per_split
    C += split * M * N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    a_ptrs += first_tile * BLOCK_SIZE_K * stride_ak
    b_ptrs += first_tile * BLOCK_SIZE_K * stride_bk
    As_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_As_k
    Bs_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_Bs_k

    # Small-M Hopper configs transpose the MMA so the weight tile occupies M.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(
        first_tile,
        tl.minimum(first_tile + tiles_per_split, tl.cdiv(K, BLOCK_SIZE_K)),
        loop_unroll_factor=1,
    ):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        # vLLM keeps the weight scale as uint8 ue8m0 bytes; decode to the exact
        # fp32 value of 2**(exp-127) with the same bit trick as
        # fp8_utils._upcast_e8m0_to_fp32.
        b_s_raw = tl.load(Bs_ptrs)
        b_s = (b_s_raw.to(tl.int32) << 23).to(tl.float32, bitcast=True)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        if SWAP_AB:
            accumulator += (
                tl.dot(tl.trans(b), tl.trans(a)) * b_s[:, None] * a_s[None, :]
            )
        else:
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if SWAP_AB:
        accumulator = tl.trans(accumulator)

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _reduce_block_fp8_split_k(
    Parts, Out, ELEMENTS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        Parts + splits[:, None] * ELEMENTS + offsets[None, :],
        offsets[None, :] < ELEMENTS,
        0.0,
    )
    tl.store(Out + offsets, tl.sum(values, axis=0), offsets < ELEMENTS)


def _contiguous_2d(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def sm90_static_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    config: dict,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Launch the static block-32 GEMM with an explicit config.

    A:  [M, K] E4M3, contiguous, ``K % 32 == 0``.
    B:  [N, K] E4M3 (the vLLM weight layout; ``B.T`` is the [K, N] operand).
    As: [M, K // 32] fp32 per-row activation scale.
    Bs: [N, K // 32] uint8 ue8m0 per-row weight scale.
    """
    assert A.dtype == MXFP8_VALUE_DTYPE and B.dtype == MXFP8_VALUE_DTYPE
    assert A.stride(-1) == 1, "A groups must be contiguous"
    assert As.dtype == torch.float32
    assert Bs.dtype == MXFP8_SCALE_DTYPE, (
        f"SM90 static kernel expects {MXFP8_SCALE_DTYPE} weight_scale, "
        f"got {Bs.dtype}"
    )
    M, K = A.shape
    N = B.shape[0]
    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    split_k = int(config.get("SPLIT_K", 1))
    swap_ab = bool(config.get("SWAP_AB", False))
    assert split_k & (split_k - 1) == 0, "SPLIT_K must be a power of two"

    out = torch.empty((M, N), device=A.device, dtype=out_dtype)
    partials = (
        torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)
        if split_k > 1
        else out
    )
    needs_masking = bool(K % block_k != 0)
    grid = (
        triton.cdiv(M, block_m) * triton.cdiv(N, block_n),
        split_k,
    )
    _w8a8_block_fp8_matmul_hopper_static[grid](
        A,
        B,
        partials,
        As,
        Bs,
        M,
        N,
        K,
        1,  # group_n: vLLM scale is per-output-row (see module docstring)
        _MXFP8_BLOCK,  # group_k
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        partials.stride(-2),
        partials.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        needs_masking=needs_masking,
        SWAP_AB=swap_ab,
        SPLIT_K=split_k,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    if split_k > 1:
        _reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](
            partials, out, M * N, split_k, 256
        )
    return out


def _quantize_activation_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-32 E4M3 activation quantisation with fp32 ue8m0 scales."""
    x_q, x_s = per_token_group_quant_fp8(
        x, _MXFP8_BLOCK, use_ue8m0=True, column_major_scales=False
    )
    return x_q, x_s.to(torch.float32)


class Sm90StaticMxfp8LinearKernel(Mxfp8LinearKernel):
    """Native W8A8 MXFP8 (block 32x32) GEMM on SM90 via the static Triton kernel.

    Selection is opt-in and default-off. ``is_supported`` is true only on
    family(90) CUDA with ``VLLM_SM90_FP8_BLOCK32_STATIC=1``, so ``auto`` keeps
    selecting Marlin on SM90 and the SM100 entries on family(100). The kernel
    class is also registered under the (intended) ``mxfp8_sm90_static``
    ``--linear-backend`` key in ``kernels/linear/__init__.py``.
    """

    supports_pre_processed_weights = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "SM90 static MXFP8 requires CUDA"
        if not current_platform.is_device_capability_family(90):
            return False, "SM90 static MXFP8 requires SM90 (Hopper)"
        if not _sm90_static_enabled():
            return False, (
                f"set {_SM90_STATIC_ENV}=1 to enable the native SM90 MXFP8 "
                "static GEMM"
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None:
            return False, "SM90 static GEMM is not a batched (BMM) kernel"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        scale = layer.weight_scale.data
        assert weight.ndim == 2 and weight.dtype == MXFP8_VALUE_DTYPE, (
            f"unexpected MXFP8 weight {tuple(weight.shape)} {weight.dtype}"
        )
        N, K = weight.shape
        assert K % _MXFP8_BLOCK == 0, f"K={K} must be divisible by {_MXFP8_BLOCK}"
        assert scale.ndim == 2 and scale.dtype == MXFP8_SCALE_DTYPE
        assert scale.shape == (N, K // _MXFP8_BLOCK), (
            f"weight_scale {tuple(scale.shape)} != {(N, K // _MXFP8_BLOCK)}"
        )
        # Pure rewrite (contiguity only) -> pre-processed weights stay valid.
        layer.weight = Parameter(_contiguous_2d(weight), requires_grad=False)
        layer.weight_scale = Parameter(_contiguous_2d(scale), requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        N, K = weight.shape
        x_2d = x.reshape(-1, K)
        x_q, x_s = _quantize_activation_fp8(x_2d)
        config = select_sm90_static_config(N, K, x_2d.shape[0])
        out = sm90_static_gemm(
            x_q, weight, x_s, weight_scale, config, out_dtype=x.dtype
        )
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], N)


class Sm90StaticMxfp8BmmLinearKernel(Mxfp8LinearKernel):
    """Batched (grouped) MXFP8 static GEMM on SM90 (DeepSeek-V4.1 ``wo_a``).

    The 2D weight ``[G * N, K]`` (and its ``[G * N, K // 32]`` scale) is viewed
    as ``[G, N, K]`` at load; ``apply_weights`` runs the 2D static kernel once
    per group. For DeepSeek-V4.1 ``G = n_groups // tp_size`` is 1 at TP8.
    """

    supports_pre_processed_weights = False

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "SM90 static MXFP8 BMM requires CUDA"
        if not current_platform.is_device_capability_family(90):
            return False, "SM90 static MXFP8 BMM requires SM90 (Hopper)"
        if not _sm90_static_enabled():
            return False, (
                f"set {_SM90_STATIC_ENV}=1 to enable the native SM90 MXFP8 "
                "static BMM"
            )
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is None or c.bmm_batch_size <= 0:
            return False, "SM90 static BMM requires a positive batch size"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        scale = layer.weight_scale.data
        group = self.config.bmm_batch_size
        assert group is not None and group > 0
        if weight.ndim == 3:
            return  # already grouped (weight reload)
        assert weight.ndim == 2 and weight.dtype == MXFP8_VALUE_DTYPE
        assert scale.ndim == 2 and scale.dtype == MXFP8_SCALE_DTYPE
        total_n, K = weight.shape
        assert total_n % group == 0, f"N={total_n} not divisible by G={group}"
        assert K % _MXFP8_BLOCK == 0
        N = total_n // group
        assert scale.shape == (total_n, K // _MXFP8_BLOCK)
        layer.weight = Parameter(
            _contiguous_2d(weight).view(group, N, K), requires_grad=False
        )
        layer.weight_scale = Parameter(
            _contiguous_2d(scale).view(group, N, K // _MXFP8_BLOCK),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        group, N, K = weight.shape
        assert x.shape[-2:] == (group, K), (
            f"wo_a BMM expects x [..., {group}, {K}], got {tuple(x.shape)}"
        )
        tokens = x.numel() // (group * K)
        x_q, x_s = _quantize_activation_fp8(x.reshape(tokens * group, K))
        x_q = x_q.view(tokens, group, K)
        x_s = x_s.view(tokens, group, K // _MXFP8_BLOCK)
        config = select_sm90_static_config(N, K, tokens)
        out = torch.empty((tokens, group, N), device=x.device, dtype=x.dtype)
        for g in range(group):
            out[:, g, :] = sm90_static_gemm(
                x_q[:, g, :],
                weight[g],
                x_s[:, g, :],
                weight_scale[g],
                config,
                out_dtype=x.dtype,
            )
        if bias is not None:
            out = out + bias.view(out.shape[1:])
        return out.view(*x.shape[:-2], group, N)


__all__ = [
    "Sm90StaticMxfp8LinearKernel",
    "Sm90StaticMxfp8BmmLinearKernel",
    "select_sm90_static_config",
    "sm90_static_gemm",
]