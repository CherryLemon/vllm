# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute Sanitizer driver for the SM90 compact indexer decode chain.

Run under memcheck to close out the compact Indexer out-of-bounds review item:

    compute-sanitizer --tool memcheck --error-exitcode 1 \
        python3 tests/kernels/attention/dsv41_compact_memcheck.py

The scenario is the one that used to over-read: the *logical* compressed
context (``TORCH_...``/``--n-vis``, default 131072) is far wider than the
*compact* candidate matrix (``--k-cand`` x ``--candidate-block-size``), and the
decode top-k used to be handed the logical length, so it planned
``n_vis`` element reads on a row holding only ``width``.  This drives the whole
production chain -- candidate publisher shape, SM90 compact logits, the native
``auto`` top-k, and ``finalize_candidate_topk_sm90`` -- so a regression in the
call site (not just in the kernel) is caught.

The K-cache is also given a deliberately short block table so an over-read of
the page table, not just of the logits row, would be visible too.
"""

import argparse

import torch

PAGE_SIZE = 64
HEADS = 32
HEAD_DIM = 128
HALF_D = HEAD_DIM // 2
PAYLOAD_BYTES = 64
SCALE_BYTES = 4


def _packed_cache(num_blocks: int, device) -> torch.Tensor:
    payload = torch.randint(
        0, 256, (num_blocks, PAGE_SIZE * PAYLOAD_BYTES), device=device, dtype=torch.uint8
    )
    scales = torch.randint(
        123, 126, (num_blocks, PAGE_SIZE * SCALE_BYTES), device=device, dtype=torch.uint8
    )
    return torch.cat([payload, scales], dim=1).reshape(
        num_blocks, PAGE_SIZE, PAYLOAD_BYTES + SCALE_BYTES
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--n-vis", type=int, default=131072)
    ap.add_argument("--k-cand", type=int, default=16)
    ap.add_argument("--candidate-block-size", type=int, default=8)
    ap.add_argument("--topk-tokens", type=int, default=512)
    args = ap.parse_args()

    from vllm.model_executor.kernels.attention.dsa.sm90_fp4_indexer import (
        sm90_fp4_paged_index_logits,
    )
    from vllm.model_executor.layers.indexer_topk import get_indexer_topk
    from vllm.model_executor.layers.sparse_mqa_indexer import SparseMQAIndexer
    from vllm.v1.worker.workspace import init_workspace_manager

    device = "cuda"
    torch.cuda.init()
    torch.cuda.set_device(0)
    # The native top-k backends hand out a scratch buffer through vLLM's
    # workspace manager, which the runner normally initialises.
    init_workspace_manager(torch.device("cuda"))
    rows, n_vis, cbs = args.rows, args.n_vis, args.candidate_block_size
    width = args.k_cand * cbs
    pages_needed = max(1, (n_vis + PAGE_SIZE - 1) // PAGE_SIZE)
    num_blocks = pages_needed

    torch.manual_seed(0)
    cache = _packed_cache(num_blocks, device)
    q_values = torch.randint(
        0, 256, (rows, HEADS, HALF_D), device=device, dtype=torch.uint8
    )
    q_scale = torch.randint(
        123, 126, (rows, HEADS, SCALE_BYTES), device=device, dtype=torch.uint8
    ).contiguous().view(torch.int32).reshape(rows, HEADS)
    weights = torch.randn(rows, HEADS, device=device, dtype=torch.bfloat16)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        1, -1
    ).repeat(rows, 1)
    context_lens = torch.full((rows,), n_vis, device=device, dtype=torch.int32)

    # Unordered candidates with the newest partial block first, then padding.
    newest = min((n_vis - 1) // cbs, num_blocks * PAGE_SIZE // cbs - 1)
    cand = [newest] + [b for b in range(args.k_cand - 1) if b != newest]
    cand = (cand + [-1] * args.k_cand)[: args.k_cand]
    candidate_blocks = torch.tensor(cand, device=device, dtype=torch.int32)
    candidate_blocks = candidate_blocks.reshape(1, -1).repeat(rows, 1)

    logits = sm90_fp4_paged_index_logits(
        q_values,
        q_scale,
        cache,
        weights,
        context_lens,
        block_table,
        candidate_blocks,
        cbs,
        page_size=PAGE_SIZE,
        write_candidates=False,
    )
    compact_lens = SparseMQAIndexer._compact_decode_lengths(rows, width, logits.device)
    selected = torch.empty((rows, args.topk_tokens), dtype=torch.int32, device=device)
    get_indexer_topk("auto")(
        logits, compact_lens, 1, selected, args.topk_tokens, width
    )
    page_scratch = torch.empty_like(selected)
    raw = torch.full((rows, args.topk_tokens), -1, dtype=torch.int32, device=device)
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        finalize_candidate_topk_sm90,
    )

    finalize_candidate_topk_sm90(
        selected,
        logits,
        context_lens,
        block_table,
        page_scratch,
        block_size=PAGE_SIZE,
        candidate_blocks=candidate_blocks,
        candidate_block_size=cbs,
        raw_indices=raw,
    )
    torch.cuda.synchronize()
    finite = int((logits != float("-inf")).sum().item())
    print(
        f"rows={rows} n_vis={n_vis} compact_width={width} pages={num_blocks} "
        f"finite_logits={finite} selected_max={int(selected.max().item())} "
        f"raw>=0: {int((raw >= 0).sum().item())}"
    )


if __name__ == "__main__":
    main()