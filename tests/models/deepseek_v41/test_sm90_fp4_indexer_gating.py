# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating tests for the opt-in SM90 MXFP4 sparse indexer.

These are CPU tests: they exercise the predicates and the config guard, not the
kernels.  Kernel behaviour is covered (and skipped without a GPU) by
``tests/kernels/attention/test_sm90_fp4_indexer.py``.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer as sm90
import vllm.v1.attention.backends.mla.indexer as indexer_mod
from vllm.v1.attention.backends.mla.indexer import dsa_indexer_uses_fp4

FLAG = "VLLM_SM90_FP4_INDEXER"


@pytest.fixture
def flag(monkeypatch):
    """Set/clear the kill switch and re-read it through vllm.envs."""
    def _set(value: str | None):
        if value is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, value)
        return bool(envs.VLLM_SM90_FP4_INDEXER)

    return _set


def _config(kv_dtype: str = "mxfp4"):
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: kv_dtype
        )
    )


def test_flag_defaults_off(flag):
    assert flag(None) is False
    assert flag("0") is False


def test_has_sm90_fp4_indexer_predicate(flag, monkeypatch):
    # Flag off: False even on a family(90) CUDA device.
    flag(None)
    monkeypatch.setattr(sm90.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sm90.current_platform, "is_device_capability_family", lambda fam: fam == 90
    )
    assert sm90.has_sm90_fp4_indexer() is False

    # Flag on + family(90) CUDA: True.
    flag("1")
    assert sm90.has_sm90_fp4_indexer() is True

    # family(100) stays on the existing DeepGEMM/DeepSelect path.
    monkeypatch.setattr(
        sm90.current_platform, "is_device_capability_family", lambda fam: fam == 100
    )
    assert sm90.has_sm90_fp4_indexer() is False

    # Non-CUDA platforms never take the branch.
    monkeypatch.setattr(
        sm90.current_platform, "is_device_capability_family", lambda fam: fam == 90
    )
    monkeypatch.setattr(sm90.current_platform, "is_cuda", lambda: False)
    assert sm90.has_sm90_fp4_indexer() is False


def test_dsa_indexer_uses_fp4_gating(monkeypatch):
    def _platform(cuda=True, family=None):
        monkeypatch.setattr(indexer_mod.current_platform, "is_cuda", lambda: cuda)
        monkeypatch.setattr(
            indexer_mod.current_platform,
            "is_device_capability_family",
            lambda fam: family is not None and fam == family,
        )

    # mxfp4 on SM100 is unchanged.
    _platform(family=100)
    assert dsa_indexer_uses_fp4(_config()) is True

    # mxfp4 on family(90) needs the flag.
    _platform(family=90)
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: False)
    with pytest.raises(ValueError, match="VLLM_SM90_FP4_INDEXER=1"):
        dsa_indexer_uses_fp4(_config())
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: True)
    assert dsa_indexer_uses_fp4(_config()) is True

    # sm_120 / older architectures are still rejected.
    _platform(family=120)
    with pytest.raises(ValueError, match="not supported"):
        dsa_indexer_uses_fp4(_config())

    # fp8 is accepted everywhere.
    _platform(family=90)
    assert dsa_indexer_uses_fp4(_config("fp8")) is False

    # Unknown dtype is rejected.
    with pytest.raises(ValueError, match="not supported"):
        dsa_indexer_uses_fp4(_config("nf4"))


def _dummy_paged_args():
    rows, heads, half = 1, 32, 64
    return dict(
        q_values=torch.zeros((rows, heads, half), dtype=torch.uint8),
        q_scale=torch.zeros((rows, heads), dtype=torch.int32),
        kv_cache=torch.zeros((2, 64, 68), dtype=torch.uint8),
        weights=torch.zeros((rows, heads), dtype=torch.bfloat16),
        context_lens=torch.zeros(rows, dtype=torch.int32),
        block_table=torch.zeros((rows, 1), dtype=torch.int32),
    )


def test_prefill_candidate_topk_bounds_are_per_row():
    """Regression: ``row_ks`` must be unsqueezed before it is added to the
    ``[rows, width]`` logical positions.  Without the unsqueeze the add
    broadcasts ``row_ks`` along the token axis and raises (or silently mixes
    rows) whenever ``width != rows`` -- which is always true in practice
    (2048 candidate blocks x 8 tokens here).
    """
    from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer

    rows, blocks, cbs = 4, 2, 8
    width = blocks * cbs
    logits = torch.zeros((rows, width), dtype=torch.float32)
    # Logical position 10 is out of every row's context; make it the argmax so
    # a broken bounds check would select it.
    logits[:, 10] = 100.0
    logits[:, 2] = 1.0
    candidates = torch.arange(blocks, dtype=torch.int32).repeat(rows, 1)
    row_ks = torch.tensor([10, 100, 5, 7], dtype=torch.int32)
    # row_ke = row_ks + 8: only local positions 0..7 are visible.
    row_ke = row_ks + 8
    out = torch.full((rows, 5), -1, dtype=torch.int32)

    SparseMQAIndexer._prefill_candidate_topk(
        logits, candidates, cbs, row_ks, row_ke, out
    )

    # The highest visible position (2, value 1.0) wins, and no row may select
    # the invisible position 10.
    assert (out[:, 0] == 2).all(), out
    assert (out[:, :5] < 8).all(), out
    assert (out[:, :5] >= 0).all(), out


def test_sm90_kernels_reject_ratio_two():
    # RATIO is hard-coded to 1: vLLM's cache is compressed-position indexed,
    # so ratio=2 must fail loudly instead of silently misreading slots.
    with pytest.raises(NotImplementedError, match="RATIO == 1"):
        sm90.sm90_fp4_paged_index_logits(**_dummy_paged_args(), page_size=64, ratio=2)
    with pytest.raises(NotImplementedError, match="RATIO == 1"):
        sm90.sm90_fp4_workspace_index_logits(
            q_values=torch.zeros((1, 32, 64), dtype=torch.uint8),
            q_scale=torch.zeros((1, 32), dtype=torch.int32),
            weights=torch.zeros((1, 32), dtype=torch.bfloat16),
            k_values=torch.zeros((4, 64), dtype=torch.uint8),
            k_scales=torch.zeros((4, 4), dtype=torch.uint8),
            cu_seqlen_ks=torch.zeros(1, dtype=torch.int32),
            cu_seqlen_ke=torch.zeros(1, dtype=torch.int32),
            ratio=2,
        )

# ---------------------------------------------------------------------------
# DSpark speculative-decode capability (DeepSeek-V4.1's MTP-equivalent)
# ---------------------------------------------------------------------------


def _spec_config(kv_dtype: str, num_speculative_tokens: int):
    """Minimal stand-in for the pieces ``_use_flattening`` touches."""
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: kv_dtype
        ),
        speculative_config=None,
        num_speculative_tokens=num_speculative_tokens,
    )


@pytest.mark.parametrize("next_n", [2, 4, 6, 11])
def test_sm90_fp4_forces_flattening_for_spec_decode(monkeypatch, next_n):
    """The SM90 FP4 indexer declares one query per row, so decode must flatten.

    Regression: the builder used to decide native multi-query purely from
    DeepGEMM's ``native_next_n_supported`` table.  That table describes a
    *different* kernel; on SM90 it reports 2 and 4 as native, so a DSpark/MTP
    step (``num_speculative_tokens`` 1 or 3 -> next_n 2 or 4) could select
    native rows and then fail in the SM90 FP4 caller, which only accepts
    ``next_n == 1``.
    """
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: True)
    # The DeepGEMM table SM90 would otherwise consult: 2 and 4 are "native".
    monkeypatch.setattr(
        indexer_mod, "_supports_native_decode", lambda n: n in (1, 2, 4)
    )

    cfg = _spec_config("mxfp4", next_n - 1)
    assert indexer_mod._sm90_fp4_indexer_active(cfg) is True
    assert indexer_mod._use_flattening(cfg) is True


def test_sm90_fp4_capability_is_fp4_only(monkeypatch):
    """An fp8 indexer cache on SM90 keeps the pre-existing decode layout."""
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: True)
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: False)
    monkeypatch.setattr(
        indexer_mod, "_supports_native_decode", lambda n: n in (1, 2, 4)
    )

    fp8_cfg = _spec_config("fp8", 1)  # next_n == 2
    assert indexer_mod._sm90_fp4_indexer_active(fp8_cfg) is False
    # No SM90 FP4 kernel involved -> the DeepGEMM table still governs.
    assert indexer_mod._use_flattening(fp8_cfg) is False

    # mxfp4 with the opt-in off is still not the SM90 FP4 path.
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: True)
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: False)
    assert indexer_mod._sm90_fp4_indexer_active(_spec_config("mxfp4", 1)) is False


def test_dspark_block5_next_n_matches_block_size(monkeypatch):
    """DSpark block5 verifies 1 + dspark_block_size rows and always flattens."""
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: True)
    monkeypatch.setattr(
        indexer_mod, "_supports_native_decode", lambda n: n in (1, 2, 4)
    )
    # dspark_block_size == 5 -> num_speculative_tokens == 5 -> 6 verify rows.
    cfg = _spec_config("mxfp4", 5)
    next_n = 1 + cfg.num_speculative_tokens
    assert next_n == 6
    assert indexer_mod._use_flattening(cfg) is True
