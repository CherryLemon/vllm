# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gating tests for the SM90 MXFP4 sparse indexer.

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


def _config(kv_dtype: str = "mxfp4"):
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: kv_dtype
        )
    )


@pytest.mark.parametrize(
    "cuda,family,expected",
    [(True, 90, True), (True, 100, False), (True, 120, False), (False, 90, False)],
)
def test_has_sm90_fp4_indexer_predicate(monkeypatch, cuda, family, expected):
    monkeypatch.setattr(sm90.current_platform, "is_cuda", lambda: cuda)
    monkeypatch.setattr(
        sm90.current_platform, "is_device_capability_family", lambda fam: fam == family
    )
    assert sm90.has_sm90_fp4_indexer() is expected


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

    # mxfp4 on Hopper uses the native Triton indexer.
    _platform(family=90)
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
    # The subject here is the flattening contract, not which platform/kv-dtype
    # combination is legal (``test_dsa_indexer_uses_fp4_gating`` covers that).
    # The real ``dsa_indexer_uses_fp4`` inspects the platform and would raise on
    # a CPU-only box, so pin it; otherwise this test only passes where a real
    # family(90) device happens to be present.
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: True)
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

    # mxfp4 on a different architecture is not the SM90 FP4 path.
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: True)
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: False)
    assert indexer_mod._sm90_fp4_indexer_active(_spec_config("mxfp4", 1)) is False


def test_dspark_block5_next_n_matches_block_size(monkeypatch):
    """DSpark block5 verifies 1 + dspark_block_size rows and always flattens."""
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: True)
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: True)
    monkeypatch.setattr(
        indexer_mod, "_supports_native_decode", lambda n: n in (1, 2, 4)
    )
    # dspark_block_size == 5 -> num_speculative_tokens == 5 -> 6 verify rows.
    cfg = _spec_config("mxfp4", 5)
    next_n = 1 + cfg.num_speculative_tokens
    assert next_n == 6
    assert indexer_mod._use_flattening(cfg) is True


def test_dspark_group6_capability_gating(monkeypatch):
    """Group-6 needs the SM90 FP4 path, DSpark, and the block5 (next_n == 6) shape.

    The predicate only decides whether the builder publishes the flattened
    row->request map; the kernel still re-checks request identity on device.
    Turning it on never un-flattens the SM90 FP4 decode path.
    """
    monkeypatch.setattr(indexer_mod, "has_sm90_fp4_indexer", lambda: True)
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: True)

    def _cfg(num_speculative_tokens=5, method="dspark"):
        return SimpleNamespace(
            attention_config=SimpleNamespace(
                resolve_indexer_kv_dtype=lambda default: "mxfp4"
            ),
            speculative_config=SimpleNamespace(
                use_dspark=lambda: method == "dspark",
                num_speculative_tokens=num_speculative_tokens,
            ),
            num_speculative_tokens=num_speculative_tokens,
        )

    assert indexer_mod._sm90_dspark_group6_active(_cfg()) is True
    # Block size other than 5 -> next_n != 6 -> no group.
    assert indexer_mod._sm90_dspark_group6_active(_cfg(4)) is False
    # Not DSpark.
    assert indexer_mod._sm90_dspark_group6_active(_cfg(method="mtp")) is False
    # No speculative config at all.
    no_spec = _cfg()
    no_spec.speculative_config = None
    assert indexer_mod._sm90_dspark_group6_active(no_spec) is False
    # Group-6 requires the SM90 FP4 path; it never flips flattening off.
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: False)
    assert indexer_mod._sm90_dspark_group6_active(_cfg()) is False
    monkeypatch.setattr(indexer_mod, "dsa_indexer_uses_fp4", lambda cfg: True)
    assert indexer_mod._use_flattening(_cfg()) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_reserve_workspaces_claims_overlap_in_lifetime(monkeypatch):
    """The logits and top-k scratch reservations must be live together.

    vLLM's memory profiling reads ``allocated_bytes.all.peak``, so two
    anonymous sequential ``torch.empty`` calls only ever register the larger
    one.  The real step holds the fp32 compact logits *and* the prefill top-k
    scratch at the same time, so the reservation has to as well -- otherwise
    the transient-headroom estimate is short by the smaller claim.
    """
    import torch.nn as nn

    import vllm.model_executor.layers.sparse_mqa_indexer as mod

    # Stub the K-gather workspace: it goes through vLLM's workspace manager and
    # is not what this test is about.
    monkeypatch.setattr(mod, "_prefill_k_workspaces", lambda *a, **k: (None, None))

    obj = object.__new__(mod.SparseMQAIndexer)
    nn.Module.__init__(obj)
    obj.max_total_seq_len = 1024
    obj.head_dim = 128
    obj.use_sm90 = True
    obj.candidate_blocks = torch.zeros((1, 16), dtype=torch.int32)
    obj.candidate_block_size = 8

    device = torch.device("cuda")
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    base = torch.accelerator.memory_allocated()

    obj._reserve_workspaces(device)

    torch.accelerator.synchronize()
    peak = torch.accelerator.max_memory_allocated() - base

    logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    scratch_bytes = (
        mod._PREFILL_TOPK_ROW_CHUNK * 16 * 8 * mod._PREFILL_TOPK_BYTES_PER_ELEM
    )
    assert peak >= logits_bytes + scratch_bytes, (
        f"peak {peak} < logits {logits_bytes} + scratch {scratch_bytes}: the two "
        "reservations do not overlap in lifetime"
    )
