# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the SM90 group-6 row -> request mapping.

These drive the **real** ``DeepseekV32IndexerMetadataBuilder.build()`` (the
dense/source builder that ships with DeepSeek-V4.1) with an SM90 + MXFP4 +
DSpark(block5) + group-6 configuration.  A kernel-level unit test that hands
``row_indices`` to ``sm90_fp4_paged_index_logits`` directly cannot catch the
bug these cover: the group-6 map used to be published through the *same*
``decode_indices`` variable that the builder also forwards to DeepGEMM's
``get_paged_mqa_logits_metadata(indices=...)``, whose non-empty-``indices``
branch asserts ``arch_major == 10`` -- i.e. it crashes on Hopper.  Keeping the
two maps in distinct fields is the contract enforced here.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.v1.attention.backends.mla.indexer as indexer_mod
from tests.v1.attention.utils import create_vllm_config
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadataBuilder,
    get_row_request_ids,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.block_table import get_block_table_width

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _sm90_compact_group6_config(max_model_len: int = 1024):
    """The same config plus the sparse-logits (compact consumer) opt-in."""
    cfg = _sm90_group6_config(max_model_len)
    cfg.attention_config.indexer_sparse_logits = True
    cfg.model_config.hf_config.update(
        {
            "candidate_block_size": 8,
            "candidate_topk_blocks": 2048,
            "index_topk": 512,
        }
    )
    return cfg


def _sm90_group6_config(max_model_len: int = 1024):
    """A VllmConfig that satisfies the real group-6 admission predicates.

    ``create_vllm_config``'s default model is a Hub id; when it is not available
    offline (the usual case on an air-gapped H100 box) fall back to the local
    DeepSeek-V4.1-Flash checkpoint the harness serves, and skip if neither is
    present.  Only the config is read, never the weights.
    """
    try:
        cfg = create_vllm_config(max_model_len=max_model_len)
    except Exception:  # noqa: BLE001 - offline Hub id
        local = os.environ.get(
            "DSV41_TEST_MODEL_PATH", "/public-nvme/models/DeepSeek-V4.1-Flash"
        )
        if not os.path.isdir(local):
            pytest.skip(f"no local model config for the builder ({local} missing)")
        cfg = create_vllm_config(model_name=local, max_model_len=max_model_len)
    cfg.attention_config.indexer_kv_dtype = "mxfp4"
    cfg.speculative_config = SimpleNamespace(
        use_dspark=lambda: True,
        num_speculative_tokens=5,
        enable_adaptive_verification=False,
    )
    return cfg


def _kv_cache_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )


def _block6_common(num_requests: int = 2, decode_len: int = 6):
    """``num_requests`` x six flattened verify rows, request-major."""
    device = torch.device("cuda")
    query_lens = [decode_len] * num_requests
    num_tokens = sum(query_lens)
    query_start_loc = torch.zeros(num_requests + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    seq_lens = torch.full(
        (num_requests,), 512, dtype=torch.int32, device=device
    )
    block_table_tensor = torch.arange(
        num_requests * 16, dtype=torch.int32, device=device
    ).reshape(num_requests, 16)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=num_requests,
        num_actual_tokens=num_tokens,
        max_query_len=decode_len,
        max_seq_len=512,
        block_table_tensor=block_table_tensor,
        slot_mapping=torch.zeros(num_tokens, dtype=torch.int64, device=device),
        causal=True,
    )
    return common


def _dense_group6_case(num_requests: int = 2, decode_len: int = 6):
    """Two requests x six verify rows, dense (non-varlen) SM90 layout."""
    device = torch.device("cuda")
    kv_cache_spec = _kv_cache_spec()
    vllm_config = _sm90_group6_config()
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(
        max_num_blocks, kv_cache_spec.block_size
    )
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )
    return builder, _block6_common(num_requests, decode_len)


def test_group6_builder_keeps_deepgemm_indices_empty(monkeypatch):
    """The dense/source builder must publish the map, but never to DeepGEMM."""
    monkeypatch.setattr(envs, "VLLM_SM90_FP4_INDEXER", True)
    monkeypatch.setattr(envs, "VLLM_SM90_FP4_GROUP6", True)

    builder, common = _dense_group6_case()
    assert builder.sm90_group6 is True, "fixture must admit the group-6 path"
    assert builder.supports_varlen is False, "the SM90 dense builder is non-varlen"

    calls: list[dict] = []

    def _recorder(*args, **kwargs):
        # Never call the real builder: its non-empty-``indices`` branch is the
        # very thing under test (it asserts arch_major == 10 there).
        calls.append({"args": args, "kwargs": kwargs})
        return torch.zeros((builder.num_sms, 2), dtype=torch.int32, device="cuda")

    monkeypatch.setattr(indexer_mod, "get_paged_mqa_logits_metadata", _recorder)
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    decode = md.decode
    assert decode is not None
    # The launch-shape hint is derived from the request map, not from the
    # DeepGEMM field.
    assert decode.spec_group_size == 6
    assert decode.row_request_ids is not None
    expected = torch.tensor(
        [0] * 6 + [1] * 6, dtype=torch.int32, device=decode.row_request_ids.device
    )
    torch.testing.assert_close(decode.row_request_ids[:12], expected)
    # The DeepGEMM-only field stays empty: that is what keeps Hopper off
    # DeepGEMM's SM100-only `indices` branch.
    assert decode.indices is None
    if has_deep_gemm():
        assert len(calls) == 1, "the DeepGEMM schedule builder must be exercised"
        assert calls[0]["kwargs"].get("indices") is None, calls
    # And the grouped kernel's view of the metadata is the new field.
    assert torch.equal(get_row_request_ids(decode), decode.row_request_ids)


def test_row_request_ids_take_precedence_in_the_consumer_helper():
    """``get_row_request_ids`` prefers the group-6 field over the DeepGEMM one."""
    device = torch.device("cuda")
    varlen_ids = torch.tensor([3, 3, 4, 4], dtype=torch.int32, device=device)
    group6_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    base = dict(
        block_table=torch.zeros((4, 1), dtype=torch.int32, device=device),
        seq_lens=torch.zeros((4, 1), dtype=torch.int32, device=device),
        decode_lens=torch.ones(4, dtype=torch.int32, device=device),
        requires_padding=False,
        schedule_metadata=torch.zeros((1, 2), dtype=torch.int32, device=device),
    )
    both = DeepSeekV32IndexerDecodeMetadata(
        **base, indices=varlen_ids, row_request_ids=group6_ids
    )
    assert torch.equal(get_row_request_ids(both), group6_ids)
    only_varlen = DeepSeekV32IndexerDecodeMetadata(**base, indices=varlen_ids)
    assert torch.equal(get_row_request_ids(only_varlen), varlen_ids)
    neither = DeepSeekV32IndexerDecodeMetadata(**base)
    assert get_row_request_ids(neither) is None


def test_group6_builder_leaves_varlen_map_on_the_deepgemm_field(monkeypatch):
    """Negative control: the varlen builder still fills ``indices``.

    Proves the two fields are genuinely independent -- if the group-6 fix had
    simply dropped ``decode_indices`` everywhere, the SM100 varlen path would
    silently lose its DeepGEMM schedule input.
    """
    monkeypatch.setattr(envs, "VLLM_SM90_FP4_INDEXER", True)
    monkeypatch.setattr(envs, "VLLM_SM90_FP4_GROUP6", True)
    monkeypatch.setattr(
        indexer_mod, "_supports_varlen_paged_mqa_logits", lambda: True
    )
    monkeypatch.setattr(indexer_mod, "_use_flattening", lambda cfg: True)

    builder, common = _dense_group6_case()
    assert builder.supports_varlen is True
    assert builder.sm90_group6 is True

    calls: list[dict] = []

    def _recorder(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return torch.zeros((builder.num_sms, 2), dtype=torch.int32, device="cuda")

    monkeypatch.setattr(indexer_mod, "get_paged_mqa_logits_metadata", _recorder)
    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    decode = md.decode
    assert decode is not None
    assert decode.indices is not None
    # The varlen builder publishes no separate group-6 field.
    assert decode.row_request_ids is None
    if has_deep_gemm():
        assert len(calls) == 1
        assert calls[0]["kwargs"].get("indices") is not None, calls

def test_compact_builder_admits_group6_on_a_block5_step(monkeypatch):
    """The compact (varlen) builder must also get ``spec_group_size == 6``.

    Regression: splitting the group-6 map out of the DeepGEMM field made the
    admission check look at the *dense* field only.  The SM90 compact builder
    sets ``supports_varlen = True``, so its row -> request map is published as
    ``indices`` and ``row_request_ids`` stays None -- which silently kept
    ``spec_group_size == 1`` for the consumer path this port exists for.  The
    admission check now uses the effective map, and this test pins it on the
    real ``DeepseekV41SparseIndexerMetadataBuilder``.
    """
    from vllm.v1.attention.backends.mla.sparse_indexer import (
        DeepseekV41SparseIndexerMetadataBuilder,
    )

    monkeypatch.setattr(envs, "VLLM_SM90_FP4_INDEXER", True)
    monkeypatch.setattr(envs, "VLLM_SM90_FP4_GROUP6", True)

    device = torch.device("cuda")
    kv_cache_spec = _kv_cache_spec()
    vllm_config = _sm90_compact_group6_config()
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(
        max_num_blocks, kv_cache_spec.block_size
    )
    builder = DeepseekV41SparseIndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )
    assert builder.supports_varlen is True, "the SM90 compact builder is varlen"
    assert builder.sm90_group6 is True

    calls: list[dict] = []

    def _recorder(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return torch.zeros((builder.num_sms, 2), dtype=torch.int32, device="cuda")

    monkeypatch.setattr(indexer_mod, "get_paged_mqa_logits_metadata", _recorder)
    md = builder.build(common_prefix_len=0, common_attn_metadata=_block6_common())

    decode = md.decode
    assert decode is not None
    # The group-6 launch shape is admitted...
    assert decode.spec_group_size == 6, (
        "the compact consumer did not admit the group-6 launch shape: "
        f"spec_group_size={decode.spec_group_size}"
    )
    # ... through the varlen map, while the dense field stays empty...
    assert decode.row_request_ids is None
    assert decode.indices is not None
    # ... and the grouped kernel's accessor resolves to that map.
    assert torch.equal(get_row_request_ids(decode), decode.indices)
    # DeepGEMM's SM100-only varlen branch must still never be reached from
    # Hopper: the varlen DeepGEMM condition is false here, so no call at all.
    assert not calls, calls
