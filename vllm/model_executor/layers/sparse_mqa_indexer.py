# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse Attention Indexer that scores only the candidate blocks.

DeepSeek V4.1 two-level selection: the candidate-source indexer publishes
the top candidate blocks and later indexers pick their top-k inside them.
`SparseAttnIndexer` does that by computing dense logits over the whole
context and masking; this layer instead calls DeepGEMM's sparse MQA-logits
kernels on the candidate blocks only, so the work is O(candidate blocks)
instead of O(context). It requires the `DeepseekV41SparseIndexerBackend`
metadata (see ``AttentionConfig.indexer_sparse_logits``).
"""

import torch
from torch import nn

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    finalize_candidate_topk_sm90,
)
from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
    has_sm90_fp4_indexer,
)
from vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits import (
    has_deep_select,
    sm90_sparse_mqa_logits_paged_decode,
    sm90_sparse_mqa_logits_prefill_chunk,
    sparse_mqa_logits_paged_decode,
    sparse_mqa_logits_prefill_chunk,
)
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.model_executor.layers.sparse_attn_indexer import (
    _gather_workspace_shapes,
    kv_cache_as_quant_view,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm_sparse_mqa
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerPrefillChunkMetadata,
)
from vllm.v1.attention.backends.mla.sparse_indexer import (
    DeepseekV41SparseIndexerMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager


def _prefill_k_workspaces(
    total_seq_lens: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The packed MXFP4 K-gather workspace ``(values, scales)``, shared with
    the dense indexer layers of the same model."""
    values_spec, scales_spec = _gather_workspace_shapes(
        total_seq_lens, head_dim, current_platform.fp8_dtype(), use_fp4_cache=True
    )
    k_quant, k_scale = current_workspace_manager().get_simultaneous(
        values_spec, scales_spec
    )
    return k_quant, k_scale


def _gather_prefill_chunk_k(
    kv_cache: torch.Tensor,
    k_quant_full: torch.Tensor,
    k_scale_full: torch.Tensor,
    chunk: DeepseekV32IndexerPrefillChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one prefill chunk's paged K into the packed workspace."""
    assert chunk.local_cu_seq_lens is not None
    k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
    k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
    if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            k_quant,
            k_scale,
            chunk.block_table,
            chunk.local_cu_seq_lens,
        )
    return k_quant, k_scale


class SparseMQAIndexer(nn.Module):
    """Candidate-consuming indexer on DeepGEMM's sparse MQA-logits kernels.

    Only valid for indexer layers that read candidate blocks with the MXFP4
    indexer cache on SM100. The K cache is written by the model before this
    runs; ``forward`` takes the same arguments as `SparseAttnIndexer` so the
    attention layer can call either.
    """

    weights_dtype = torch.bfloat16
    """Per-head weights dtype the sparse kernels take. The fused Q RoPE-quant
    kernel writes it directly so no cast runs per step."""

    def __init__(
        self,
        k_cache,
        topk_tokens: int,
        head_dim: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        candidate_blocks: torch.Tensor,
        candidate_block_size: int,
    ):
        super().__init__()
        self.use_sm90 = has_sm90_fp4_indexer()
        if self.use_sm90:
            # family(90) MXFP4 path: Triton logits kernels + vLLM's existing
            # fp32 decode top-k. DeepGEMM's sparse MQA logits and DeepSelect
            # are SM100-only and are deliberately not required here.
            if not current_platform.is_cuda():
                raise ValueError(
                    "SparseMQAIndexer SM90 path requires a CUDA platform."
                )
        elif not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
            and has_deep_gemm_sparse_mqa()
        ):
            raise ValueError(
                "SparseMQAIndexer requires an SM100-class GPU and DeepGEMM >= 2.8, "
                "or family(90) with VLLM_SM90_FP4_INDEXER=1."
            )
        if not self.use_sm90 and not has_deep_select():
            raise ValueError(
                "SparseMQAIndexer requires the DeepSelect top-k extension "
                "(vllm._deepselect_C)."
            )
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        if self.use_sm90:
            vllm_config = get_current_vllm_config()
            self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
            # NOTE: do NOT cache the page size from ``k_cache.cache_config``
            # here.  ``CacheConfig.block_size`` is still the unresolved default
            # (16) while the model is being constructed and is only bumped to
            # the kernel block size the backend requires (64) during worker
            # init.  The compact path therefore reads the page size off the
            # bound cache tensor at call time, exactly like the K-cache writer
            # (``indexer_k_norm_rope_store``) and the dense reader
            # (``sparse_attn_indexer``) already do.
            self.max_model_len = (
                vllm_config.model_config.max_model_len
                // getattr(k_cache, "compress_ratio", 1)
            )

    def _reserve_workspaces(self, device: torch.device) -> None:
        """Profiling run: claim the K-gather workspace and the peak sparse
        logits allocation so the memory estimate covers them."""
        _prefill_k_workspaces(self.max_total_seq_len, self.head_dim)
        max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        torch.empty(max_logits_bytes, dtype=torch.uint8, device=device)

    def _forward_sm90(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor,
        q_scale: torch.Tensor,
        weights: torch.Tensor,
        metadata: DeepseekV41SparseIndexerMetadata,
        topk_indices_buffer: torch.Tensor,
    ) -> None:
        """family(90) compact path: SM90 Triton logits + existing top-k.

        The candidate-token logits are produced straight out of the MXFP4
        cache/workspace by the SM90 kernels; the decode top-k reuses vLLM's
        ``get_indexer_topk`` and the selected columns are mapped back to
        request-local compressed positions (the DeepGEMM path's
        ``sparse_topk_remap`` convention) by ``finalize_candidate_topk_sm90``.
        """
        topk_tokens = self.topk_tokens
        cbs = self.candidate_block_size
        kv_cache = self.k_cache.kv_cache
        # The indexer page holds ``kv_cache.shape[1]`` compressed positions
        # (kernel block size / compress ratio).  Derive it from the bound
        # tensor: ``k_cache.cache_config.block_size`` is unresolved at
        # construction time, and during CUDA-graph profiling the layer is
        # driven against the minimal profiling cache before the real one is
        # bound.  This is the same page size the writer and the dense reader
        # use, so the packed-page layout the SM90 kernels assume is unchanged.
        assert kv_cache.dim() == 3, (
            "indexer K cache must be the 3D [num_blocks, page, row_bytes] "
            f"uint8 view, got shape={tuple(kv_cache.shape)}"
        )
        page_size = kv_cache.shape[1]
        topk_indices_buffer[: hidden_states.shape[0]] = -1

        if metadata.num_prefills > 0:
            prefill = metadata.prefill
            assert prefill is not None
            k_quant_full, k_scale_full = _prefill_k_workspaces(
                self.max_total_seq_len, self.head_dim
            )
            for chunk in prefill.chunks:
                k_quant, k_scale = _gather_prefill_chunk_k(
                    kv_cache, k_quant_full, k_scale_full, chunk
                )
                if chunk.local_total_seq_lens == 0:
                    continue  # buffer already holds -1
                start, end = chunk.token_start, chunk.token_end
                logits = sm90_sparse_mqa_logits_prefill_chunk(
                    q_values[start:end],
                    q_scale[start:end],
                    k_quant,
                    k_scale,
                    weights[start:end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    self.candidate_blocks[start:end],
                    cbs,
                )
                self._prefill_candidate_topk(
                    logits,
                    self.candidate_blocks[start:end],
                    cbs,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices_buffer[start:end, :topk_tokens],
                )

        if metadata.num_decodes > 0:
            decode = metadata.sparse_decode
            assert decode is not None and decode.block_table is not None
            assert decode.row_ke is not None
            num_rows = decode.row_ke.shape[0]
            if num_rows > 0:
                cand = self.candidate_blocks[:num_rows]
                logits = sm90_sparse_mqa_logits_paged_decode(
                    q_values[:num_rows],
                    q_scale[:num_rows],
                    kv_cache,
                    weights[:num_rows],
                    decode.row_ke,
                    decode.block_table,
                    cand,
                    cbs,
                    page_size,
                )
                # One query per row (varlen flattening), so the row end is the
                # query's own visible compressed length.
                selected = torch.empty(
                    (num_rows, topk_tokens),
                    dtype=torch.int32,
                    device=logits.device,
                )
                get_indexer_topk(self.topk_backend)(
                    logits,
                    decode.row_ke.view(-1, 1),
                    1,
                    selected,
                    topk_tokens,
                    self.max_model_len,
                )
                page_scratch = torch.empty_like(selected)
                finalize_candidate_topk_sm90(
                    selected,
                    logits,
                    decode.row_ke,
                    decode.block_table,
                    page_scratch,
                    block_size=page_size,
                    candidate_blocks=cand,
                    candidate_block_size=cbs,
                    raw_indices=topk_indices_buffer[:num_rows, :topk_tokens],
                )

    @staticmethod
    def _prefill_candidate_topk(
        logits: torch.Tensor,
        candidate_blocks: torch.Tensor,
        candidate_block_size: int,
        row_ks: torch.Tensor,
        row_ke: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Torch top-k over compact prefill candidate logits -> local positions.

        Prefill candidate columns are scattered (a candidate block may reach
        past the row's context), so the contiguous-bound prefill top-k kernel
        cannot consume them.  Invalid candidates are masked to ``-inf`` here;
        the surviving columns are mapped to the request-local compressed
        position ``block * CBS + within``.  This mirrors SGLang's candidate
        prefill remap and keeps only ``O(rows * K * CBS)`` work.
        """
        rows, width = logits.shape
        if rows == 0 or width == 0:
            out.fill_(-1)
            return
        cols = torch.arange(width, device=logits.device)
        block = candidate_blocks[:, cols // candidate_block_size]
        within = cols % candidate_block_size
        logical = block.to(torch.int64) * candidate_block_size + within
        valid = (block >= 0) & (
            (row_ks.to(torch.int64)[:, None] + logical)
            < row_ke.to(torch.int64)[:, None]
        )
        masked = logits.masked_fill(~valid, float("-inf"))
        k = min(out.shape[1], width)
        values, indices = torch.topk(masked, k, dim=-1)
        selected = torch.gather(logical, 1, indices)
        selected = torch.where(
            values > float("-inf"), selected, torch.full_like(selected, -1)
        ).to(torch.int32)
        out.fill_(-1)
        out[:, :k] = selected

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_quant: tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            self._reserve_workspaces(hidden_states.device)
            return self.topk_indices_buffer

        metadata = attn_metadata[self.k_cache.prefix]
        assert isinstance(metadata, DeepseekV41SparseIndexerMetadata), (
            "SparseMQAIndexer needs DeepseekV41SparseIndexerBackend metadata"
        )
        assert k is None, "the model writes the indexer K cache"
        q_values, q_scale = q_quant
        topk_tokens = self.topk_tokens
        topk_indices_buffer = self.topk_indices_buffer
        if self.use_sm90:
            self._forward_sm90(
                hidden_states,
                q_values,
                q_scale,
                weights,
                metadata,
                topk_indices_buffer,
            )
            return topk_indices_buffer
        kv_cache = self.k_cache.kv_cache
        topk_indices_buffer[: hidden_states.shape[0]] = -1
        sparse_block_kv = metadata.sparse_block_kv

        if metadata.num_prefills > 0:
            prefill = metadata.prefill
            sparse_chunks = metadata.sparse_prefill
            assert prefill is not None and sparse_chunks is not None
            k_quant_full, k_scale_full = _prefill_k_workspaces(
                self.max_total_seq_len, self.head_dim
            )
            for chunk, rows in zip(prefill.chunks, sparse_chunks):
                k_quant, k_scale = _gather_prefill_chunk_k(
                    kv_cache, k_quant_full, k_scale_full, chunk
                )
                if chunk.local_total_seq_lens == 0:
                    continue  # buffer already holds -1
                start, end = chunk.token_start, chunk.token_end
                rows.kernel_metadata = sparse_mqa_logits_prefill_chunk(
                    q_values[start:end].view(torch.int8),
                    q_scale[start:end],
                    k_quant.view(torch.int8),
                    k_scale.view(torch.int32).squeeze(-1),
                    weights[start:end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    self.candidate_blocks[start:end],
                    self.candidate_block_size,
                    sparse_block_kv,
                    topk_tokens,
                    topk_indices_buffer[start:end, :topk_tokens],
                    sparse_indices=rows.sparse_indices,
                    end=rows.end,
                    col_indices=rows.col_indices,
                    kernel_metadata=rows.kernel_metadata,
                )

        if metadata.num_decodes > 0:
            decode_rows = metadata.sparse_decode
            assert decode_rows is not None and decode_rows.block_table is not None
            assert decode_rows.row_indices is not None
            num_rows = decode_rows.row_ke.shape[0]
            if num_rows > 0:
                decode_rows.kernel_metadata = sparse_mqa_logits_paged_decode(
                    q_values[:num_rows].view(torch.int8).unsqueeze(1),
                    q_scale[:num_rows].unsqueeze(1),
                    kv_cache_as_quant_view(kv_cache, self.head_dim, use_fp4_cache=True),
                    weights[:num_rows],
                    decode_rows.row_ke,
                    decode_rows.block_table,
                    decode_rows.row_indices,
                    self.candidate_blocks[:num_rows],
                    self.candidate_block_size,
                    sparse_block_kv,
                    topk_tokens,
                    topk_indices_buffer[:num_rows, :topk_tokens],
                    row_ks=decode_rows.row_ks,
                    sparse_indices=decode_rows.sparse_indices,
                    end=decode_rows.end,
                    col_indices=decode_rows.col_indices,
                    kernel_metadata=decode_rows.kernel_metadata,
                )
        return topk_indices_buffer
