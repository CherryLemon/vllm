# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scope regression tests for the SM90 static MXFP8 BMM (``wo_a``) gap.

These are **CPU-only** tests: they pin the registry policy, the selection
outcome and the group-count arithmetic. They deliberately do not launch a
kernel -- the SM90 kernel numerics live in
``tests/kernels/linear/test_sm90_fp8_block32_static.py`` under
``@requires_sm90``.

Background (see ``kernels/linear/__init__.py::_mxfp8_bmm_candidate_kernels``):
``Sm90StaticMxfp8BmmLinearKernel`` is not registered for ``is_bmm`` layers on
SM90. Registering it is not a one-line change: on SM90 ``wo_a`` is consumed by
``deep_gemm_fp8_o_proj``, never through ``apply_weights``, and keeping the
weight fp8 flips that function's dtype-keyed ``use_fp8`` gate to DeepGEMM's
``(1, 128, 128)`` fp32 einsum against a ``[N, K // 32]`` uint8 MXFP8 scale.

The tests below fail loudly if that class is added to the candidate list
without also wiring the rest of the o-projection contract.
"""

import pytest

from vllm.model_executor.kernels.linear.mxfp8.deep_gemm import (
    DeepGemmMxfp8BmmLinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.sm90_static import (
    Sm90StaticMxfp8BmmLinearKernel,
    Sm90StaticMxfp8LinearKernel,
)
from vllm.platforms import current_platform

# DeepSeek-V4.1 ``config.o_groups`` (the HF config default; the deployed JSON is
# authoritative). ``wo_a.bmm_batch_size = n_local_groups = o_groups // tp_size``.
DSV41_O_GROUPS = 8


def _simulate_sm90(monkeypatch) -> None:
    """Make the platform predicates claim a family(90) CUDA device.

    Same pattern as ``tests/models/deepseek_v41/test_sm90_fp4_indexer_gating.py``.
    """
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family: family == 90,
    )


def _refuse_deep_gemm_bmm(monkeypatch) -> None:
    """Force DeepGEMM's MXFP8 BMM refusal (SM100-only) without probing DeepGEMM.

    ``is_deep_gemm_supported()`` is not CPU-safe, and on a real SM90 box the
    answer is "not supported" anyway.
    """
    monkeypatch.setattr(
        DeepGemmMxfp8BmmLinearKernel,
        "is_supported",
        classmethod(
            lambda cls, compute_capability=None: (
                False,
                "DeepGEMM MXFP8 BMM requires Blackwell.",
            )
        ),
    )


def test_bmm_candidate_list_excludes_the_sm90_static_bmm():
    from vllm.model_executor.kernels.linear import _mxfp8_bmm_candidate_kernels

    possible = _mxfp8_bmm_candidate_kernels()
    assert possible == [DeepGemmMxfp8BmmLinearKernel, EmulationMxfp8LinearKernel]
    assert Sm90StaticMxfp8BmmLinearKernel not in possible
    # The last entry is the fallback when DeepGEMM refuses, which is exactly the
    # SM90 case; assert Emulation is last for that reason.
    assert possible[-1] is EmulationMxfp8LinearKernel


def test_sm90_static_bmm_exclusion_is_policy_not_inability():
    # The class is fully capable of a bmm (so it *could* be selected) and it is
    # the plain static kernel that refuses bmm. The absence from the candidate
    # list is therefore a deliberate contract decision, not a can_implement()
    # accident that a future change might "fix" by making it selectable.
    assert (
        Sm90StaticMxfp8BmmLinearKernel.can_implement(
            Mxfp8LinearLayerConfig(bmm_batch_size=1)
        )[0]
        is True
    )
    assert (
        Sm90StaticMxfp8LinearKernel.can_implement(
            Mxfp8LinearLayerConfig(bmm_batch_size=1)
        )[0]
        is False
    )


def test_sm90_bmm_selection_is_emulation(monkeypatch):
    """The wo_a BMM slot requires emulation until o-projection supports MXFP8."""
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel

    _simulate_sm90(monkeypatch)
    _refuse_deep_gemm_bmm(monkeypatch)
    assert Sm90StaticMxfp8LinearKernel.is_supported()[0] is True
    kernel = init_mxfp8_linear_kernel(bmm_batch_size=1)
    assert type(kernel) is EmulationMxfp8LinearKernel
    assert kernel.supports_pre_processed_weights is True


@pytest.mark.parametrize(
    "tp_size,expected_groups",
    [(8, 1), (4, 2), (2, 4), (1, 8)],
)
def test_wo_a_group_count_by_tp(tp_size: int, expected_groups: int):
    # Mirrors attention.py: n_local_groups = o_groups // tp_size, and
    # wo_a.bmm_batch_size = n_local_groups.
    assert DSV41_O_GROUPS // tp_size == expected_groups


def test_bmm_degenerates_to_a_single_2d_gemm_at_tp8():
    """At TP8 ``G == 1``, so the grouped kernel's batch loop runs once.

    ``Sm90StaticMxfp8BmmLinearKernel.apply_weights`` does
    ``for g in range(group)`` with ``group = bmm_batch_size = n_local_groups``.
    With ``G == 1`` that is one 2D GEMM: the BMM machinery itself buys nothing.
    The grouped form only becomes a real batch at TP <= 4 (G in {2, 4, 8}),
    which is where a BMM-specific kernel would start to matter -- and even
    there ``(N, K) = (1024, 4096)`` has no tuned config.
    """
    g_tp8 = DSV41_O_GROUPS // 8
    assert g_tp8 == 1

    groups = {tp: DSV41_O_GROUPS // tp for tp in (8, 4, 2, 1)}
    assert groups[8] == 1
    assert all(groups[tp] > 1 for tp in (4, 2, 1))
