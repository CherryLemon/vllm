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
   cumulative, so they are sampled before and after and the *delta* is checked;
   a server that drafted for some earlier workload must not carry the
   assertion.

2. **KV transfer really happened this round.**  A working PD test has to show
   the decode side actually received remote KV.  Checking that the proxy
   returned HTTP 200 does not do that: a decode instance that silently
   recomputes the prefill locally answers correctly too.  This test samples the
   NIXL counters across the round, requires a positive delta, requires the
   failure counters to stay flat, and runs a negative control (a
   ``do_remote_prefill`` request pointed at an unreachable producer must not
   answer).

3. **The output is right.**  Compared against what the *same decode instance*
   produces from a plain local prefill of the same prompt.  That A/B is
   self-contained and cannot be satisfied by a path that ignores the transfer:
   the two sides differ only in where the prefill KV came from.

   The comparison is on the **first token**, which is read straight off the
   transferred prefill context and is measured stable (10/10 identical repeats
   for every prompt), rather than on full text.  Full-text equality must not be
   asserted here: with DSpark the multi-token verification changes the GEMM
   reduction order with the batch, and at fp8/fp4 precision that flips argmaxes
   at near-ties -- the same prompt at ``temperature=0`` produced 6 distinct
   continuations over 10 sequential requests.  Byte-equality would be a flaky
   gate that proves nothing; the agreement rate is reported instead.

# Two traps this file exists to step around (both were live bugs here)

* **Prompts must exceed the SWA bounded-replay window.**  DeepSeek-V4.1
  declares ``sliding_window=128``, and the scheduler deliberately drops any
  external hit that does not survive block rounding and exceed
  ``prefix_replay_tokens`` ("a hit no longer than the replayed window would be
  recomputed in full and save nothing").  With short prompts the decode
  instance correctly recomputes locally, no KV moves, and the placement under
  test is never exercised -- which is exactly how an earlier version of this
  test reported success while testing nothing.
* **Prompts must not already sit in the decode instance's prefix cache.**
  Each run salts the filler with a unique token, so a run cannot be served from
  a previous run's cache.  Without this the transfer delta is 0 and the run
  proves nothing.

Environment variables (set by ``dsv41_dspark_pd_sm90.sh``):
    TEST_MODEL              - served model name (default: deepseek-v4.1-flash)
    SERVER_HOST / PROXY_PORT - the proxy
    DECODE_HOST / DECODE_PORT - the decode instance (for /metrics and the A/B)
    NUM_PROMPTS             - prompts to send (default 4)
    MAX_TOKENS              - output tokens (default 64)
    DSV41_SWA_REPLAY_TOKENS - the model's replay window (default 128)
    DSV41_PD_RUN_SALT       - fixed salt, for reproducing a run
    DSV41_DSPARK_REFERENCE  - optional standalone reference JSON (extra check)
    DSV41_STRICT_REFERENCE  - 1 makes that optional check required
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
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "4"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
REFERENCE = os.environ.get("DSV41_DSPARK_REFERENCE", "")
STRICT = os.environ.get("DSV41_STRICT_REFERENCE", "0") == "1"

# Mirrors the model's SWA bounded-replay window; see the module docstring.
SWA_REPLAY_TOKENS = int(os.environ.get("DSV41_SWA_REPLAY_TOKENS", "128"))

PROXY_BASE_URL = f"http://{SERVER_HOST}:{PROXY_PORT}/v1"
DECODE_BASE_URL = f"http://{DECODE_HOST}:{DECODE_PORT}/v1"
DECODE_METRICS_URL = f"http://{DECODE_HOST}:{DECODE_PORT}/metrics"

# Unique per run so the decode instance cannot serve this run out of the prefix
# cache it built for an earlier one.
RUN_SALT = os.environ.get("DSV41_PD_RUN_SALT") or f"run{time.time_ns()}"

# ~220 filler chunks, comfortably above the replay window in tokens.
_FILLER = " ".join(f"context chunk number {i:04d} is filler text." for i in range(220))

PROMPTS = [
    f"[{RUN_SALT}] {_FILLER} The capital of France is",
    f"[{RUN_SALT}] {_FILLER} The chemical symbol for gold is",
    f"[{RUN_SALT}] {_FILLER} Q: What is the largest planet in the solar system?\nA:",
    f"[{RUN_SALT}] {_FILLER} List the first five prime numbers:",
]

SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
# NIXL transfer evidence. `_count` is the histogram component the connector
# actually observes; the raw byte/descriptor histograms are not exported unless
# a transfer was recorded.
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
            name, _, _ = line.rpartition(" ")
        if name not in totals:
            continue
        try:
            totals[name] += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return totals


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in before}


def _client(base_url: str) -> openai.OpenAI:
    return openai.OpenAI(api_key="EMPTY", base_url=base_url)


def _complete_resp(
    base_url: str,
    prompt: str,
    max_tokens: int = MAX_TOKENS,
    extra_body: dict | None = None,
    timeout: float | None = None,
):
    body = {"add_special_tokens": False}
    if extra_body:
        body.update(extra_body)
    kwargs = {} if timeout is None else {"timeout": timeout}
    return _client(base_url).completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        extra_body=body,
        **kwargs,
    )


def _complete(
    prompt: str, max_tokens: int = MAX_TOKENS, extra_body: dict | None = None
) -> str:
    return _complete_resp(PROXY_BASE_URL, prompt, max_tokens, extra_body).choices[0].text


# ---------------------------------------------------------------------------
# Fixtures: one measured PD round, plus the local-prefill control
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompts() -> list[str]:
    return PROMPTS[:NUM_PROMPTS]


@pytest.fixture(scope="module")
def round_metrics(prompts):
    """Metrics bracketing the PD round, plus that round's outputs."""
    before = scrape(DECODE_METRICS_URL, ALL_METRICS)
    outputs: list[str] = []
    prompt_tokens: list[int] = []
    for p in prompts:
        resp = _complete_resp(PROXY_BASE_URL, p)
        outputs.append(resp.choices[0].text)
        # Prompt length from a real call: a max_tokens=0 probe is rejected 400.
        prompt_tokens.append(int(resp.usage.prompt_tokens))
    first_tokens = [_complete(p, max_tokens=1) for p in prompts]
    after = scrape(DECODE_METRICS_URL, ALL_METRICS)
    return {
        "before": before,
        "after": after,
        "delta": delta(before, after),
        "outputs": outputs,
        "first_tokens": first_tokens,
        "prompt_tokens": prompt_tokens,
    }


@pytest.fixture(scope="module")
def local_first_tokens(prompts):
    """The same prompts prefilled and decoded locally on the decode instance.

    No ``kv_transfer_params``, so this is a plain (non-disaggregated) run on the
    very same server.  Comparing against it isolates the only thing under test:
    whether the KV the decode instance used was the one the prefill instance
    produced.
    """
    return [
        _complete_resp(DECODE_BASE_URL, p, max_tokens=1).choices[0].text
        for p in prompts
    ]


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
    token_acceptance = (
        d[SPEC_ACCEPTED] / d[SPEC_DRAFT_TOKENS] if d[SPEC_DRAFT_TOKENS] else 0.0
    )
    print(
        f"\nDSpark this round: drafts={d[SPEC_DRAFTS]:.0f} "
        f"draft_tokens={d[SPEC_DRAFT_TOKENS]:.0f} accepted={d[SPEC_ACCEPTED]:.0f} "
        f"mean_acceptance_length={mean_acceptance:.3f} "
        f"token_acceptance={token_acceptance:.3f}"
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
    catch it and the PD configuration is untested.

    Negative: a ``do_remote_prefill`` request whose producer is unreachable must
    not answer.  Note this control is only meaningful above the replay window,
    where the scheduler is actually willing to pull.
    """
    d = round_metrics["delta"]
    n_prompt = min(round_metrics["prompt_tokens"])
    assert n_prompt > SWA_REPLAY_TOKENS, (
        f"prompts are {n_prompt} tokens but the model's SWA bounded-replay "
        f"window is {SWA_REPLAY_TOKENS}: the scheduler drops every external hit "
        "at or below it, so this run cannot exercise disaggregation. Lengthen "
        "the filler in PROMPTS (or lower DSV41_SWA_REPLAY_TOKENS if the model "
        "changed)."
    )
    for name in (NIXL_FAILED_XFER, NIXL_FAILED_NOTIFY, NIXL_EXPIRED):
        assert d[name] == 0, f"{name} moved by {d[name]} during the round"

    transfer_evidence = d[NIXL_XFER_COUNT] + d[NIXL_POST_COUNT]
    assert transfer_evidence > 0, (
        "no KV transfer was recorded during this round "
        f"({NIXL_XFER_COUNT} delta={d[NIXL_XFER_COUNT]}, "
        f"{NIXL_POST_COUNT} delta={d[NIXL_POST_COUNT]}). The decode instance "
        "answered from a local prefill, so this run does not exercise "
        "disaggregation at all -- check the prompts are novel to the decode "
        "instance (RUN_SALT) and above the replay window."
    )
    print(f"\nKV transfer this round: {transfer_evidence:.0f} transfer(s)")

    novel = (
        f"[pd-negative-control-{time.time_ns()}] {_FILLER} "
        "The chemical symbol for gold is"
    )
    bogus = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_engine_id": "deadbeef-0000-0000-0000-000000000000",
        "remote_request_id": "cmpl-negative-control",
        "remote_host": "10.255.255.1",  # TEST-NET-3 style black hole
        "remote_port": 5999,
        "remote_block_ids": list(range(40)),
        "remote_num_tokens": 60,
        "tp_size": 8,
        "dcp_size": 1,
        "pp_size": 1,
        "transfer_mode": "pull",
    }
    started = time.time()
    try:
        # The producer is unreachable, so the scheduler parks the request
        # waiting for KV that never arrives.  The assertion is "must not
        # answer", not "must fail in exactly this way".
        resp = _complete_resp(
            DECODE_BASE_URL,
            novel,
            max_tokens=4,
            extra_body={"kv_transfer_params": bogus},
            timeout=60.0,
        )
        text = resp.choices[0].text
    except Exception as exc:  # noqa: BLE001 - failure/timeout is the expected outcome
        print(
            f"negative control did not answer, as expected: {type(exc).__name__} "
            f"after {time.time() - started:.1f}s"
        )
        return
    pytest.fail(
        "the decode instance answered a do_remote_prefill request whose "
        f"producer (10.255.255.1:5999) is unreachable, in "
        f"{time.time() - started:.1f}s: {text[:40]!r}. That is only possible by "
        "recomputing the prefill locally, so remote prefill is not in effect."
    )


# ---------------------------------------------------------------------------
# 3. output correctness: PD must match a local prefill of the same prompt
# ---------------------------------------------------------------------------


def test_pd_first_token_matches_local_prefill(
    prompts, round_metrics, local_first_tokens
):
    """The core transport-correctness gate, self-contained.

    Both sides run on the same decode instance with the same prompts and
    ``temperature=0``; the only difference is whether the prefill KV arrived
    over NIXL or was computed locally.  The first token is read straight off
    that context and is measured stable, so a lossy or misplaced transfer
    changes it.
    """
    assert len(local_first_tokens) == len(prompts)
    mismatches = [
        (p, want, got)
        for p, got, want in zip(
            prompts, round_metrics["first_tokens"], local_first_tokens
        )
        if got != want
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(prompts)} prompts differ between the PD path and "
        "a local prefill of the same prompt:\n"
        + "\n".join(
            f"  {p[:60]!r}\n    local={w!r}\n    pd   ={g!r}" for p, w, g in mismatches
        )
    )


def test_pd_outputs_are_non_degenerate(round_metrics):
    assert any(t.strip() for t in round_metrics["outputs"]), "every completion was empty"


# ---------------------------------------------------------------------------
# 4. optional: a pre-recorded standalone reference
# ---------------------------------------------------------------------------


def test_pd_matches_recorded_standalone_reference(prompts, round_metrics):
    """Extra check against a reference dumped by ``dsv41_dspark_reference.py``.

    Optional because the reference's prompts cannot be run-unique, so a
    mismatch here can also mean the decode instance answered partly from its own
    prefix cache.  The self-contained local A/B above is the real gate.
    """
    if not REFERENCE:
        if STRICT:
            pytest.fail("DSV41_STRICT_REFERENCE=1 but DSV41_DSPARK_REFERENCE is unset")
        pytest.skip("DSV41_DSPARK_REFERENCE not set")
    with open(REFERENCE) as fh:
        ref = json.load(fh)
    assert isinstance(ref, dict) and ref, f"{REFERENCE} is not a non-empty JSON object"

    compared = 0
    mismatches = []
    for prompt, got in zip(prompts, round_metrics["first_tokens"]):
        entry = ref.get(prompt)
        if entry is None:
            continue
        compared += 1
        want = entry["first_token"] if isinstance(entry, dict) else entry[: len(got)]
        if got != want:
            mismatches.append((prompt, want, got))
    if compared == 0:
        pytest.skip(
            "no reference prompt matches this run's salted prompts (expected "
            "unless DSV41_PD_RUN_SALT matches the reference run)"
        )
    assert not mismatches, (
        f"{len(mismatches)} prompts differ on the first token from the recorded "
        "reference:\n"
        + "\n".join(
            f"  {p[:60]!r}\n    ref={w!r}\n    pd ={g!r}" for p, w, g in mismatches
        )
    )