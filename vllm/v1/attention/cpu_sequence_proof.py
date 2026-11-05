# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Immutable CPU input-step facts for bounded sparse-indexer metadata."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class CPUSequenceLengthsProof:
    """Immutable packet facts in the current reordered request order."""

    issuer: str
    generation: int
    request_ids: tuple[str, ...]
    query_starts: tuple[int, ...]
    computed: tuple[int, ...]
    prompt_lengths: tuple[int, ...]
    previous_positions: tuple[int, ...]
    previous_drafts: tuple[int, ...]


@dataclass(frozen=True)
class PrefillChunkSequenceBounds:
    """Per-request KV extents for a current paged prefill query slice."""

    step_proof: CPUSequenceLengthsProof
    request_start: int
    request_stop: int
    chunk_start: int
    chunk_end: int
    seq_lens: tuple[int, ...]


_current_cpu_seq_lens_proof: ContextVar[CPUSequenceLengthsProof | None] = ContextVar(
    "current_cpu_seq_lens_proof", default=None
)


def set_current_cpu_seq_lens_proof(proof: CPUSequenceLengthsProof | None) -> None:
    _current_cpu_seq_lens_proof.set(proof)


def get_current_cpu_seq_lens_proof() -> CPUSequenceLengthsProof | None:
    return _current_cpu_seq_lens_proof.get()


def validated_cpu_seq_lens_proof(
    stager,
    request_ids,
    previous_req_map,
    query_starts,
    upper,
    prefilling,
    prompt_lengths,
    num_reqs,
    num_reqs_padded,
    num_tokens,
    num_tokens_padded,
    for_capture,
    ubatch_slices,
):
    """Accept only immutable host facts from the current successful packet."""
    try:
        if (
            for_capture
            or ubatch_slices is not None
            or stager is None
            or not stager.in_step
            or stager.plan is None
            or num_reqs_padded != num_reqs
            or num_tokens_padded != num_tokens
            or any(
                getattr(x.device, "type", None) != "cpu"
                for x in (query_starts, upper, prefilling, prompt_lengths)
            )
            or previous_req_map is None
        ):
            return None
        plan = stager.plan
        proof = getattr(plan, "cpu_seq_lens_proof", None)
        if (
            not isinstance(proof, CPUSequenceLengthsProof)
            or proof.issuer != stager._cpu_proof_issuer
            or proof.generation != stager._cpu_proof_generation
            or plan.num_reqs != num_reqs
            or plan.total_tokens != num_tokens
            or tuple(request_ids) != plan.req_ids
            or tuple(request_ids) != proof.request_ids
            or tuple(int(previous_req_map.get(r, -1)) for r in request_ids)
            != proof.previous_positions
            or len(proof.request_ids) != num_reqs
            or len(set(proof.request_ids)) != num_reqs
            or len(proof.query_starts) != num_reqs + 1
            or any(
                len(x) != num_reqs
                for x in (
                    proof.computed,
                    proof.prompt_lengths,
                    proof.previous_positions,
                    proof.previous_drafts,
                )
            )
        ):
            return None
        starts = tuple(int(x) for x in query_starts[: num_reqs + 1])
        if starts != proof.query_starts or starts[0] != 0 or starts[-1] != num_tokens:
            return None
        for i in range(num_reqs):
            q = starts[i + 1] - starts[i]
            computed = proof.computed[i]
            prompt = proof.prompt_lengths[i]
            if (
                q < 0
                or computed < 0
                or prompt < 0
                or int(upper[i]) != computed + q
                or int(prompt_lengths[i]) != prompt
                or bool(prefilling[i]) != (computed < prompt)
            ):
                return None
        return proof
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None


def validated_prefill_chunk_bound(
    common,
    chunk,
    req_start,
    req_stop,
    num_decodes,
    decode_threshold,
    compress_ratio,
    dcp_world,
    pcp_world,
):
    """Bind a paged prefill chunk to immutable facts from this input step.

    Per-request lengths bound score columns; their sum bounds flattened
    gather offsets. They intentionally have different capacity contracts.
    """
    proof = common.cpu_seq_lens_proof
    if (
        not isinstance(proof, CPUSequenceLengthsProof)
        or compress_ratio != 1
        or dcp_world != 1
        or pcp_world != 1
        or not num_decodes <= req_start < req_stop <= common.num_reqs
        or chunk.num_reqs != req_stop - req_start
        or common.num_reqs != len(proof.request_ids)
        or any(
            x.device.type != "cpu"
            for x in (
                common.query_start_loc_cpu,
                common.seq_lens_cpu_upper_bound,
                common.is_prefilling,
            )
        )
        or tuple(int(x) for x in common.query_start_loc_cpu) != proof.query_starts
    ):
        return None
    starts = proof.query_starts[req_start : req_stop + 1]
    if not (starts[0] <= chunk.token_start < chunk.token_end <= starts[-1]):
        return None
    bounds = []
    for i in range(req_start, req_stop):
        prev, draft = proof.previous_positions[i], proof.previous_drafts[i]
        query_len = proof.query_starts[i + 1] - proof.query_starts[i]
        computed = proof.computed[i]
        bound = computed + query_len
        if (
            query_len <= decode_threshold
            or computed < 0
            or not computed < proof.prompt_lengths[i]
            or not bool(common.is_prefilling[i])
            or not ((prev == -1 and draft == -1) or (prev >= 0 and draft == 0))
            or int(common.seq_lens_cpu_upper_bound[i]) != bound
        ):
            return None
        bounds.append(bound)
    total = sum(bounds)
    if (
        chunk.total_seq_lens != total
        or chunk.local_total_seq_lens != total
        or chunk.num_reqs != chunk.block_table.shape[0]
    ):
        return None
    return PrefillChunkSequenceBounds(
        proof, req_start, req_stop, chunk.token_start, chunk.token_end, tuple(bounds)
    )
