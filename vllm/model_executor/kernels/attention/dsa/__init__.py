# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels for the DeepSeek sparse-attention (DSA) indexer.

``has_sm90_fp4_indexer`` is the opt-in family(90) MXFP4 indexer predicate; the
SM100 DeepSelect predicate stays in ``sparse_mqa_logits`` to avoid importing
DeepGEMM-heavy modules just to read the package attribute.
"""

from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
    has_sm90_fp4_indexer,
)

__all__ = ["has_sm90_fp4_indexer"]