# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DSpark + NixlConnector PD acceptance test.

DeepSeek-V4.1 has no classic MTP draft: its checkpoints ship DSpark stages
under ``mtp.*`` (``dspark_block_size=5``, ``num_nextn_predict_layers=3``), so
speculative decoding on this model is ``method="dspark"`` and one draft round
proposes 5 tokens that the target verifies with 6 query rows.

What this file asserts, and -- just as importantly -- what it refuses to accept
as evidence:

1. **The speculative path really ran this round.**  Spec-decode counters are
   cumulative, so they are sampled before and after and the *delta* is checked.
   A server that drafted for some earlier workload must not be able to carry
   the assertion.

2. **KV transfer really happened this round.**  A working PD test has to show
   the decode side actually received remote KV.  Checking that the proxy
   returned HTTP 200 does not do that: a decode instance that silently
   recomputes the prefill locally answers correctly too.  So this test samples
   the NIXL transfer counters across the round, requires a positive delta, and
   requires the failure counters to stay flat.  It also runs a negative
   control: a ``do_remote_prefill`` request pointed at an unreachable producer
   must *not* come back with a confident answer, because that is only possible
   if the remote prefill was skipped.  See ``test_kv_transfer_is_actually_used``.

3. **The output is right.**  Compared against a standalone (non-PD) reference.
   The comparison is on the *first* token, which measurably is deterministic
   here (10/10 identical repeats), and which is read straight off the
   transferred prefill context -- so a lossy transfer changes it.  Full-text
   equality is deliberately NOT asserted: with DSpark the multi-token
   verification changes the GEMM reduction order with the batch, and at
   fp8/fp4 precision that flips argmaxes at near-ties.  Measured on this stack,
   the same prompt at ``temperature=0`` produced 6 distinct continuations over
   10 sequential requests.  Byte-equality would therefore be a flaky gate that
   proves nothing; the agreement rate is reported instead.

Environment variables (set by ``dsv41_dspark_pd_sm90.sh``):
    TEST_MODEL              - served model name (default: deepseek-v4.1-flash)
    SERVER_HOST / PROXY_PORT - the proxy
    DECODE_HOST / DECODE_PORT - the decode instance (for /metrics)
    NUM_PROMPTS             - prompts to send (default: 8)
    MAX_TOKENS              - output tokens (default: 64)
    DSV41_DSPARK_REFERENCE  - standalone reference JSON
    DSV41_STRICT_REFERENCE  - 1 turns the reference checks into hard gates
"""

import json
import os
import time
from urllib.request import urlopen

import openai
import pytest

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
PROXY_PORT = os.environ.get("PROXY_PORT", "8192")
DECODE_HOST = os.environ.get("DECODE_HOST", SERVER_HOST)
DECODE_PORT = os.environ.get("DECODE_PORT", "8300")
MODEL_NAME = os.environ.get("TEST_MODEL", "deepseek-v4.1-flash")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
REFERENCE = os.environ.get("DSV41_DSPARK_REFERENCE", "")
STRICT = os.environ.get("DSV41_STRICT_REFERENCE", "0") == "1"

PROXY_BASE_URL = f"http://{SERVER_HOST}:{PROXY_PORT}/v1"
DECODE_METRICS_URL = f"http://{DECODE_HOST}:{DECODE_PORT}/metrics"

# Plain, non-chat-templated prompts: the completions API must not re-add a BOS
# (the PD path compares raw continuations), and short prompts keep the run
# cheap while still exercising a real prefill->decode handoff.
PROMPTS = [
    "The capital of France is",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n\nfib(10) =",
    "2 + 2 =",
    "Q: What is the largest planet in the solar system?\nA:",
    "Translate 'good morning' into German:",
    "The chemical symbol for gold is",
    "List the first five prime numbers:",
    "Once upon a time, in a small village by the sea,",
]

# --- metric series we sample -------------------------------------------------
SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
# NIXL transfer evidence. `_count`/`_sum` are the histogram components, which
# are what the connector actually observes (the raw byte/descriptor histograms
# are not even exported unless a transfer was recorded).
NIXL_XFER_COUNT = "vllm:nixl_xfer_time_seconds_count"
NIXL_POST_COUNT = "vllm:nixl_post_time_seconds_count"
NIXL_FAILED_XFER = "vllm:nixl_num_failed_transfers_total"
NIXL_FAILED_NOTIFY = "vllm:nixl_num_failed_notifications_total"
NIXL_EXPIRED = "vllm:nixl_num_kv_expired_reqs_total"

ALL_METRICS = (
    SPEC_DRAFTS,
    SPEC_DRAFT_TOKENS,
    SPEC_ACCEPTED,
    NIXL_XFER_COUNT,
    NIXL_POST_COUNT,
    NIXL_FAILED_XFER,
    NIXL_FAILED_NOTIFY,
    NIXL_EXPIRED,
)


def scrape(url: str, names: tuple[str, ...]) -> dict[str, float]:
    """Sum every series of each named metric.

    Series carry per-engine (and per-position) labels, so taking the first
    match would read an arbitrary subset -- on a data-parallel deployment it
    could even read a rank that did no work.
    """
    body = urlopen(url, timeout=60).read().decode()
    totals = {name: 0.0 for name in names}
    for line in body.split("\n"):
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition("{")
        if not rest:
            name, _, value = line.rpartition(" ")
        if name not in totals:
            continue
        try:
            totals[name] += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return totals


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in before}


def _client() -> openai.OpenAI:
    return openai.OpenAI(api_key="EMPTY", base_url=PROXY_BASE_URL)


def _complete(
    prompt: str, max_tokens: int = MAX_TOKENS, extra_body: dict | None = None
) -> str:
    body = {"add_special_tokens": False}
    if extra_body:
        body.update(extra_body)
    resp = _client().completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra_body=body,
    )
    return resp.choices[0].text


# ---------------------------------------------------------------------------
# Session fixtures: one measured round
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompts() -> list[str]:
    return PROMPTS[:NUM_PROMPTS]


@pytest.fixture(scope="module")
def round_metrics(prompts):
    """Metrics sampled around the round, plus that round's outputs.

    Everything the assertions need is captured here so no test can quietly
    re-use another test's traffic.
    """
    before = scrape(DECODE_METRICS_URL, ALL_METRICS)
    outputs = [_complete(p) for p in prompts]
    first_tokens = [_complete(p, max_tokens=1) for p in prompts]
    after = scrape(DECODE_METRICS_URL, ALL_METRICS)
    return {
        "before": before,
        "after": after,
        "delta": delta(before, after),
        "outputs": outputs,
        "first_tokens": first_tokens,
    }


# ---------------------------------------------------------------------------
# 1. speculative decoding ran this round
# ---------------------------------------------------------------------------


def test_dspark_spec_decode_ran_this_round(round_metrics):
    d = round_metrics["delta"]
    assert d[SPEC_DRAFTS] > 0, (
        "no DSpark drafts were produced in this round "
        f"(delta {SPEC_DRAFTS}=0). Cumulative counters would still look "
        "healthy, which is why this checks the delta."
    )
    assert d[SPEC_ACCEPTED] > 0, "DSpark drafted but accepted nothing"
    mean_acceptance = 1 + d[SPEC_ACCEPTED] / d[SPEC_DRAFTS]
    per_position = d[SPEC_ACCEPTED] / d[SPEC_DRAFT_TOKENS] if d[SPEC_DRAFT_TOKENS] else 0
    print(
        f"\nDSpark this round: drafts={d[SPEC_DRAFTS]:.0f} "
        f"draft_tokens={d[SPEC_DRAFT_TOKENS]:.0f} accepted={d[SPEC_ACCEPTED]:.0f} "
        f"mean_acceptance_length={mean_acceptance:.3f} "
        f"token_acceptance={per_position:.3f}"
    )
    assert mean_acceptance > 1.0, (
        "mean acceptance length is 1.0: every draft token was rejected, so the "
        "speculative path is not contributing"
    )


# ---------------------------------------------------------------------------
# 2. KV transfer really happened this round
# ---------------------------------------------------------------------------


def test_kv_transfer_is_actually_used(round_metrics):
    """Positive + negative evidence that decode consumed remote KV.

    Positive: the decode instance must report at least one completed transfer
    in this round.  A zero delta means the requests were served by a local
    recompute -- which produces perfectly correct text, so no output check can
    catch it, and the whole PD configuration is untested.

    Negative control: a ``do_remote_prefill`` request whose producer address is
    unreachable must not answer confidently.  An immediate, correct answer
    there is only possible by skipping the remote prefill.
    """
    d = round_metrics["delta"]
    for name in (NIXL_FAILED_XFER, NIXL_FAILED_NOTIFY, NIXL_EXPIRED):
        assert d[name] == 0, f"{name} moved by {d[name]} during the round"

    transfer_evidence = d[NIXL_XFER_COUNT] + d[NIXL_POST_COUNT]
    assert transfer_evidence > 0, (
        "no KV transfer was recorded during this round "
        f"({NIXL_XFER_COUNT} delta={d[NIXL_XFER_COUNT]}, "
        f"{NIXL_POST_COUNT} delta={d[NIXL_POST_COUNT]}). The decode instance "
        "answered from a local prefill, so this run does not exercise "
        "disaggregation at all."
    )

    # Negative control, on a prompt the decode instance cannot have cached.
    novel = f"[pd-negative-control-{time.time_ns()}] The chemical symbol for gold is"
    bogus = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_engine_id": "deadbeef-0000-0000-0000-000000000000",
        "remote_request_id": "cmpl-negative-control",
        "remote_host": "10.255.255.1",  # TEST-NET-3 style black hole
        "remote_port": 5999,
        "remote_block_ids": [0],
        "remote_num_tokens": 12,
        "tp_size": 1,
        "dcp_size": 1,
        "pp_size": 1,
        "transfer_mode": "pull",
    }
    started = time.time()
    try:
        text = _complete(novel, max_tokens=4, extra_body={"kv_transfer_params": bogus})
    except Exception as exc:  # noqa: BLE001 - any failure is the expected outcome
        print(f"negative control failed as expected: {type(exc).__name__}")
        return
    elapsed = time.time() - started
    pytest.fail(
        "the decode instance answered a do_remote_prefill request whose "
        f"producer (10.255.255.1:5999) is unreachable, in {elapsed:.1f}s: "
        f"{text[:40]!r}. That is only possible by recomputing the prefill "
        "locally, so remote prefill is not actually in effect."
    )


# ---------------------------------------------------------------------------
# 3. output correctness against a standalone reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference() -> dict:
    if not REFERENCE:
        if STRICT:
            pytest.fail(
                "DSV41_STRICT_REFERENCE=1 but DSV41_DSPARK_REFERENCE is unset"
            )
        pytest.skip("DSV41_DSPARK_REFERENCE not set")
    with open(REFERENCE) as fh:
        ref = json.load(fh)
    assert isinstance(ref, dict) and ref, f"{REFERENCE} is not a non-empty JSON object"
    return ref


def _reference_texts(reference: dict) -> dict[str, str]:
    """The reference's full continuation per prompt."""
    return {
        prompt: (entry["text"] if isinstance(entry, dict) else entry)
        for prompt, entry in reference.items()
    }


def _reference_first_tokens(reference: dict) -> dict[str, str]:
    """The reference's first token per prompt, when recorded.

    Falls back to the first character of the continuation, which is what a
    ``max_tokens=1`` request would have returned for these prompts.
    """
    out = {}
    for prompt, entry in reference.items():
        if isinstance(entry, dict) and "first_token" in entry:
            out[prompt] = entry["first_token"]
        else:
            text = entry["text"] if isinstance(entry, dict) else entry
            out[prompt] = text[:1]
    return out


def test_reference_covers_every_prompt(prompts, reference):
    """Otherwise the comparison below can compare nothing and still pass."""
    ref = _reference_texts(reference)
    missing = [p for p in prompts if p not in ref]
    assert not missing, (
        f"{len(missing)}/{len(prompts)} prompts have no reference entry, so the "
        f"correctness comparison would silently skip them: {missing[:2]}"
    )
    assert len(prompts) > 0


def test_dspark_pd_first_token_matches_reference(prompts, reference, round_metrics):
    """The first token is read straight off the transferred prefill context.

    It is also the one quantity that is measurably stable here: 10 sequential
    repeats at ``temperature=0`` returned the identical first token for every
    prompt, while the full continuations diverged (6 distinct over 10 for one
    prompt).  So this is a real, non-flaky transport-content check.
    """
    ref = _reference_first_tokens(reference)
    compared = 0
    mismatches = []
    for prompt, got in zip(prompts, round_metrics["first_tokens"]):
        if prompt not in ref:
            continue
        compared += 1
        want = ref[prompt]
        if got != want:
            mismatches.append((prompt, want, got))
    assert compared == len(prompts), (
        f"compared {compared} of {len(prompts)} prompts; a partial comparison "
        "must not be reported as a pass"
    )
    assert not mismatches, (
        f"{len(mismatches)} prompts differ on the first token:\n"
        + "\n".join(f"  {p[:50]!r}\n    ref={w!r}\n    pd ={g!r}" for p, w, g in mismatches)
    )


def test_dspark_pd_full_text_agreement_is_reported(prompts, reference, round_metrics):
    """Report -- do not gate on -- full-continuation agreement.

    ``temperature=0`` is not bitwise reproducible on this stack (DSpark's
    multi-token verification changes the GEMM reduction order with the batch,
    and fp8/fp4 rounding then flips argmaxes at near-ties), so equality here is
    neither necessary nor sufficient for a correct transfer.
    """
    ref = _reference_texts(reference)
    agree = sum(
        1
        for p, got in zip(prompts, round_metrics["outputs"])
        if p in ref and got == ref[p]
    )
    prefix_ok = sum(
        1
        for p, got in zip(prompts, round_metrics["outputs"])
        if p in ref and got[:32] == ref[p][:32]
    )
    print(
        f"\nfull-text agreement with standalone: {agree}/{len(prompts)}; "
        f"first-32-char agreement: {prefix_ok}/{len(prompts)}"
    )
    assert prefix_ok >= 1, "not a single continuation shares a 32-char prefix"


def test_dspark_pd_outputs_are_non_degenerate(round_metrics):
    assert any(t.strip() for t in round_metrics["outputs"]), "every completion was empty"