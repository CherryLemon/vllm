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

3. **The output is right, against an independent control.**  The first token
   from the PD request is compared with the first token of a plain local
   prefill on the same decode instance.  Both sides are taken **cold**: the
   decode instance's prefix cache (including the connector-managed external
   cache) is reset and the reset is verified before *each* request, and the
   local request is additionally required to record zero prefix-cache hits.
   Without that, the "control" could be served from the very cache entry the
   PD request created -- a comparison that is guaranteed to succeed and proves
   nothing.  The transfer counters are also sampled around the *individual* PD
   request whose first token is compared, so "the compared request really
   pulled remote KV" is established rather than assumed.

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
  a previous run's cache.  That is necessary but *not sufficient* within one
  run: the salted prompts are still shared by the PD leg and the local control
  leg, so the local control would hit the entry the PD leg just created.  Both
  legs therefore reset the decode instance's prefix cache first (and the local
  leg's zero-hit requirement is asserted), which is what makes the control
  independent.  See the module docstring point 3.

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
    DSV41_COLD_RESET        - 0 disables the per-request prefix-cache reset
                              (default 1).  Only for debugging a server without
                              VLLM_SERVER_DEV_MODE=1; disabling it weakens the
                              local control back to the old shared-cache case.
    DSV41_SKIP_NEGATIVE_CONTROL - 1 skips the unreachable-producer control
                              (default 0).  It parks a request the engine
                              cannot abort, which blocks prefix-cache resets
                              until the decode instance is restarted; the A/B
                              must run before it (test order) for that reason.
    DSV41_KV_CONNECTOR      - connector under test (default NixlConnector).
                              Only the *evidence* differs between connectors,
                              so the same acceptance criteria run against
                              MooncakeConnector: NIXL exports per-transfer
                              Prometheus counters, Mooncake does not (its
                              stats are periodic log lines), and both are
                              bracketed by the connector-agnostic
                              ``vllm:external_prefix_cache_hits`` counter,
                              which is what actually proves remote KV was
                              adopted.
    DSV41_DECODE_LOG        - path to the decode instance's stdout/stderr, for
                              the evidence Mooncake only emits as log lines
                              (transfer stats, bootstrap failures).  Unset
                              means those checks report "no evidence" instead
                              of passing silently.
    DSV41_ABORT_TIMEOUT_PROBE - 1 enables the final probe that asks whether a
                              request parked on an unreachable producer is ever
                              released (default 0).  Opt-in because it waits.
    DSV41_ABORT_TIMEOUT_PROBE_WAIT - seconds to wait for that release
                              (default 90)
"""

import contextlib
import json
import os
import re
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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
# The per-request prefix-cache reset is what makes the local control
# independent; it is on by default and only turned off to debug a server that
# was started without VLLM_SERVER_DEV_MODE=1.
COLD_RESET = os.environ.get("DSV41_COLD_RESET", "1") == "1"
# The negative control parks a request on an unreachable producer, which the
# engine cannot abort (see the test's docstring).  Skip it when re-running only
# the A/B against an instance that already has one parked.
SKIP_NEGATIVE_CONTROL = os.environ.get("DSV41_SKIP_NEGATIVE_CONTROL", "0") == "1"

# Connector under test.  The acceptance criteria are the same for every
# cross-instance connector; what differs is where the transfer evidence lives.
KV_CONNECTOR = os.environ.get("DSV41_KV_CONNECTOR", "NixlConnector")
IS_NIXL = KV_CONNECTOR.endswith("NixlConnector")
# Mooncake records its transfer stats in KVConnectorStats, which has no
# Prometheus exporter (the connector carries a TODO for one), so the counters
# only reach the log as periodic "KV Transfer metrics: ..." lines.
DECODE_LOG = os.environ.get("DSV41_DECODE_LOG", "")
ABORT_TIMEOUT_PROBE = os.environ.get("DSV41_ABORT_TIMEOUT_PROBE", "0") == "1"
ABORT_TIMEOUT_PROBE_WAIT = float(
    os.environ.get("DSV41_ABORT_TIMEOUT_PROBE_WAIT", "90")
)

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
# Local prefix-cache evidence: the control leg must show a *zero* hit delta.
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
# Connector-agnostic evidence that remote KV was *adopted*: the scheduler
# records the number of tokens the connector claimed to have matched
# externally.  It exists for every cross-instance connector, unlike the
# per-connector transfer counters below.
EXTERNAL_QUERIES = "vllm:external_prefix_cache_queries"
EXTERNAL_HITS = "vllm:external_prefix_cache_hits"

ALL_METRICS = (
    SPEC_DRAFTS,
    SPEC_DRAFT_TOKENS,
    SPEC_ACCEPTED,
    NIXL_XFER_COUNT,
    NIXL_POST_COUNT,
    NIXL_FAILED_XFER,
    NIXL_FAILED_NOTIFY,
    NIXL_EXPIRED,
    EXTERNAL_QUERIES,
    EXTERNAL_HITS,
)

TRANSFER_METRICS = (
    NIXL_XFER_COUNT,
    NIXL_POST_COUNT,
    NIXL_FAILED_XFER,
    NIXL_FAILED_NOTIFY,
    PREFIX_QUERIES,
    PREFIX_HITS,
    EXTERNAL_QUERIES,
    EXTERNAL_HITS,
)

# Mooncake's periodic stats line, e.g.
#   KV Transfer metrics: Num successful transfers=4, Avg xfer time (ms)=1.2, ...
_KV_METRICS_RE = re.compile(r"KV Transfer metrics: (.*)")


def _decode_log_text() -> str:
    """The decode instance's log, or "" when no log path was provided."""
    if not DECODE_LOG:
        return ""
    try:
        with open(DECODE_LOG, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def mooncake_log_stats() -> dict[str, float]:
    """Sum the Mooncake counters that only exist in the log.

    Returned keys are connector vocabulary, not Prometheus names: the stats
    module renders "Num successful transfers", "Num failed recvs", etc.
    """
    totals = {
        "successful": 0.0,
        "failed_transfers": 0.0,
        "failed_recvs": 0.0,
        "expired": 0.0,
    }
    for line in _KV_METRICS_RE.findall(_decode_log_text()):
        for item in line.split(","):
            key, _, value = item.partition("=")
            key = key.strip().lower()
            try:
                number = float(value)
            except ValueError:
                continue
            if key.startswith("num successful"):
                totals["successful"] += number
            elif key.startswith("num failed transfers"):
                totals["failed_transfers"] += number
            elif key.startswith("num failed recvs"):
                totals["failed_recvs"] += number
            elif key.startswith("num kv expired"):
                totals["expired"] += number
    return totals


def mooncake_pull_failure_seen(addr: str) -> bool:
    """True when the decode log shows a failed remote-KV pull from *addr*.

    That is connector-level evidence the request reached the pull path: the
    consumer queried the producer's bootstrap server and gave up.  Used
    because a *bootstrap* failure (unlike a mid-transfer failure) has no
    counter of its own.
    """
    text = _decode_log_text()
    if not text or addr not in text:
        return False
    return (
        f"Failed to connect to bootstrap server http://{addr}" in text
        or "not found from bootstrap server" in text
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


def _client(base_url: str, max_retries: int | None = None) -> openai.OpenAI:
    """OpenAI client; ``max_retries=0`` makes one attempt for one timeout.

    The SDK retries timeouts twice by default, so a 60 s timeout can take ~180 s
    and leave one parked request per attempt on the server.  The negative
    control sets 0 so its single request maps to a single engine-side request.
    """
    kwargs = {} if max_retries is None else {"max_retries": max_retries}
    return openai.OpenAI(api_key="EMPTY", base_url=base_url, **kwargs)


def _complete_resp(
    base_url: str,
    prompt: str,
    max_tokens: int = MAX_TOKENS,
    extra_body: dict | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
):
    body = {"add_special_tokens": False}
    if extra_body:
        body.update(extra_body)
    kwargs = {} if timeout is None else {"timeout": timeout}
    return _client(base_url, max_retries=max_retries).completions.create(
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
    result = _complete_resp(PROXY_BASE_URL, prompt, max_tokens, extra_body)
    return result.choices[0].text


def reset_decode_prefix_cache() -> bool:
    """Drop every cached prefix on the decode instance.

    ``reset_external=true`` also asks the connector to drop its own cache; that
    is a no-op for connectors which do not implement ``reset_cache()``
    (NixlConnector returns ``None``, which the scheduler treats as success), and
    the *local* prefix cache -- the one that could serve the local control from
    the PD request's entry -- is cleared either way.

    The endpoint is behind the dev routers (``VLLM_SERVER_DEV_MODE=1``) and
    reports ``success=false`` while blocks are still held; retry briefly rather
    than accepting a partial reset, because a surviving entry is exactly the
    confound this call exists to remove.
    """
    url = (
        f"http://{DECODE_HOST}:{DECODE_PORT}/reset_prefix_cache"
        "?reset_running_requests=true&reset_external=true"
    )
    last_error = ""
    for attempt in range(10):
        try:
            body = json.loads(
                urlopen(Request(url, method="POST"), timeout=60).read().decode()
            )
        except HTTPError as exc:
            if exc.code in (404, 405):
                pytest.fail(
                    f"{url} returned HTTP {exc.code}: the dev endpoints are "
                    "disabled, so the local control cannot be made independent "
                    "from the PD request's cache entry. Start the decode "
                    "instance with VLLM_SERVER_DEV_MODE=1."
                )
            if exc.code >= 500:
                # The scheduler raises while blocks are still held (e.g. a
                # request parked on a remote transfer); retry, then fail loudly
                # with the server's own message.
                with contextlib.suppress(Exception):
                    # Best-effort diagnostics: the server's own message.
                    last_error = f"HTTP {exc.code} {exc.read().decode()[:240]}"
                if not last_error:
                    last_error = f"HTTP {exc.code}"
                time.sleep(1 + attempt)
                continue
            raise
        if body.get("success"):
            return True
        last_error = f"success=false ({body})"
        time.sleep(1 + attempt)
    print(f"prefix-cache reset did not succeed: {last_error}")
    return False


def connector_reports_pending_remote_kv() -> bool:
    """Ask the engine whether a request is still waiting for remote KV.

    That is *connector-level* evidence that the negative control reached the
    pull path: the scheduler refuses a prefix-cache reset for exactly that
    reason.  Used instead of a "it took longer than N seconds" threshold, which
    a plain queueing delay would also satisfy.
    """
    url = (
        f"http://{DECODE_HOST}:{DECODE_PORT}/reset_prefix_cache"
        "?reset_running_requests=true&reset_external=true"
    )
    try:
        body = json.loads(
            urlopen(Request(url, method="POST"), timeout=60).read().decode()
        )
    except HTTPError as exc:
        if exc.code < 500:
            return False
        with contextlib.suppress(Exception):
            return "waiting for remote KV transfer" in exc.read().decode()
        return False
    except Exception:  # noqa: BLE001 - an unreachable engine is not evidence
        return False
    # A successful reset means nothing is parked any more.
    return not body.get("success", False)


def decode_is_healthy() -> bool:
    try:
        urlopen(f"http://{DECODE_HOST}:{DECODE_PORT}/health", timeout=30).read()
        return True
    except Exception:  # noqa: BLE001 - "still serving" is the assertion
        return False


# ---------------------------------------------------------------------------
# Fixtures: one measured PD round, plus the local-prefill control
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompts() -> list[str]:
    return PROMPTS[:NUM_PROMPTS]


@pytest.fixture(scope="module")
def round_metrics(prompts):
    """Metrics bracketing the PD round, plus that round's outputs.

    Only full-output requests run here.  The single-token tokens used for the
    A/B live in ``first_token_ab``, because they need a cache reset immediately
    around each request and must not be mixed into this cumulative delta.
    """
    before = scrape(DECODE_METRICS_URL, ALL_METRICS)
    outputs: list[str] = []
    prompt_tokens: list[int] = []
    for p in prompts:
        resp = _complete_resp(PROXY_BASE_URL, p)
        outputs.append(resp.choices[0].text)
        # Prompt length from a real call: a max_tokens=0 probe is rejected 400.
        prompt_tokens.append(int(resp.usage.prompt_tokens))
    after = scrape(DECODE_METRICS_URL, ALL_METRICS)
    return {
        "before": before,
        "after": after,
        "delta": delta(before, after),
        "outputs": outputs,
        "prompt_tokens": prompt_tokens,
    }


@pytest.fixture(scope="module")
def first_token_ab(prompts):
    """Cold PD vs cold local first tokens, one independently reset pair each.

    The order is deliberate: reset -> local -> reset -> PD, so neither request
    can see the other's prefix cache entry.  The reset itself is verified (the
    endpoint reports success), the local leg must record **zero** prefix-cache
    hits, and the transfer counters are sampled around the single PD request
    whose first token is actually compared.
    """
    pairs = []
    for prompt in prompts:
        local_hits = None
        local_first = None
        if COLD_RESET:
            assert reset_decode_prefix_cache(), (
                "the decode instance refused to reset its prefix cache "
                "(reset_external=true), so the local control cannot be trusted "
                "to be independent of the PD request. The usual cause is a "
                "request still parked waiting for remote KV -- vLLM cannot "
                "reset the cache in that state ('not supported yet'). Restart "
                "the decode instance, or run this file with "
                "DSV41_SKIP_NEGATIVE_CONTROL=1 so no such request is created."
            )
        hits_before = scrape(DECODE_METRICS_URL, (PREFIX_QUERIES, PREFIX_HITS))
        local_first = _complete_resp(DECODE_BASE_URL, prompt, max_tokens=1).choices[
            0
        ].text
        hits_after = scrape(DECODE_METRICS_URL, (PREFIX_QUERIES, PREFIX_HITS))
        local_hits = hits_after[PREFIX_HITS] - hits_before[PREFIX_HITS]

        if COLD_RESET:
            assert reset_decode_prefix_cache(), (
                "the decode instance refused the second (pre-PD) prefix-cache "
                "reset, so the PD request may be served from the local cache "
                "entry the control just created"
            )
        before = scrape(DECODE_METRICS_URL, TRANSFER_METRICS)
        pd_first = _complete_resp(PROXY_BASE_URL, prompt, max_tokens=1).choices[0].text
        after = scrape(DECODE_METRICS_URL, TRANSFER_METRICS)
        pair_delta = delta(before, after)
        pairs.append(
            {
                "prompt": prompt,
                "pd": pd_first,
                "local": local_first,
                "local_prefix_hits": local_hits,
                "pd_transfer": (
                    pair_delta[NIXL_XFER_COUNT] + pair_delta[NIXL_POST_COUNT]
                    if IS_NIXL
                    else 0.0
                ),
                "pd_external_tokens": pair_delta[EXTERNAL_HITS],
                "pd_failed": (
                    pair_delta[NIXL_FAILED_XFER] + pair_delta[NIXL_FAILED_NOTIFY]
                    if IS_NIXL
                    else 0.0
                ),
            }
        )
    return pairs


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
    """Positive evidence that decode consumed remote KV.

    The decode instance must report at least one completed transfer in this
    round.  A zero delta means the requests were served by a local recompute --
    which produces perfectly correct text, so no output check can catch it and
    the PD configuration is untested.

    The negative control (an unreachable producer) lives in its own test at the
    end of the file: it deliberately leaves a request parked on a remote
    transfer, which would make a later prefix-cache reset fail.
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
    if IS_NIXL:
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
    else:
        # Mooncake's counters are log-only; assert on the *whole-run* totals,
        # which is coarser than the per-request NIXL counters but still a
        # failure gate rather than no gate at all.
        stats = mooncake_log_stats()
        assert stats["failed_transfers"] == 0 and stats["failed_recvs"] == 0, (
            "Mooncake recorded failed transfers during this round: "
            f"{stats}. A retry that recovered does not belong in the clean-path "
            "gate; investigate the link before accepting the run."
        )
        assert stats["expired"] == 0, (
            f"Mooncake expired {stats['expired']:.0f} producer-side transfer(s) "
            "waiting for the decode side; those blocks were freed without being "
            "read."
        )
        transfer_evidence = stats["successful"]

    # Connector-agnostic: the scheduler only records these for tokens the
    # connector actually supplied, so a zero here means the decode instance
    # recomputed locally and disaggregation was not exercised.
    external = d[EXTERNAL_HITS]
    assert external > 0, (
        f"{EXTERNAL_HITS} delta is 0: the decode instance adopted no remotely "
        f"computed tokens this round (queries delta {d[EXTERNAL_QUERIES]:.0f}). "
        "Text alone cannot catch this -- a local prefill produces the same "
        "answer -- so the connector either never matched the prompt or the "
        "external hit was dropped (SWA replay window, block rounding)."
    )
    print(
        f"\nKV transfer this round: {transfer_evidence:.0f} transfer(s), "
        f"{external:.0f} externally matched token(s) adopted by the decoder"
    )


# ---------------------------------------------------------------------------
# 3. output correctness: PD must match a local prefill of the same prompt
# ---------------------------------------------------------------------------


def test_pd_first_token_matches_local_prefill(first_token_ab):
    """The core transport-correctness gate, on a genuinely independent A/B.

    Both sides run on the same decode instance with the same prompt and
    ``temperature=0``; the only difference is whether the prefill KV arrived
    over NIXL or was computed locally.  Cache state is not left to chance: each
    leg is preceded by a verified prefix-cache reset, the local leg must record
    no prefix-cache hit, and the PD leg must record a real transfer -- so the
    comparison cannot be satisfied by both legs reading one shared cache entry.
    """
    assert first_token_ab, "no prompt pairs were measured"
    if COLD_RESET:
        polluted = [
            p for p in first_token_ab if p["local_prefix_hits"] > 0
        ]
        assert not polluted, (
            "the local control leg hit the decode instance's prefix cache for "
            f"{len(polluted)}/{len(first_token_ab)} prompts despite a reset "
            "(hits: "
            + ", ".join(f"{p['local_prefix_hits']:.0f}" for p in polluted)
            + "). The comparison would be against the PD request's own cache "
            "entry. Check that /reset_prefix_cache?reset_external=true really "
            "clears the connector cache."
        )
    failed_transfers = [p for p in first_token_ab if p["pd_failed"] > 0]
    assert not failed_transfers, (
        f"{len(failed_transfers)}/{len(first_token_ab)} compared PD requests "
        f"recorded a *failed* {KV_CONNECTOR} transfer while still answering. "
        "This gate is for the clean path: a retry that recovered the transfer "
        "must not be accepted here (test fault tolerance separately).\n"
        + "\n".join(
            f"  {p['prompt'][:50]!r} failures={p['pd_failed']:.0f}"
            for p in failed_transfers
        )
    )
    # Per-request transfer counters are NIXL-only.  For a connector without
    # them the equivalent per-request gate is the external-token count below,
    # which the scheduler keeps for every connector.
    if IS_NIXL:
        no_transfer = [p for p in first_token_ab if p["pd_transfer"] <= 0]
        assert not no_transfer, (
            f"{len(no_transfer)}/{len(first_token_ab)} compared PD requests "
            "recorded no KV transfer of their own, so their first token cannot "
            "be evidence about disaggregation (it came from a local prefill or "
            "a cache hit):\n"
            + "\n".join(
                f"  {p['prompt'][:50]!r} transfer_delta={p['pd_transfer']:.0f}"
                for p in no_transfer
            )
        )
    no_external = [p for p in first_token_ab if p["pd_external_tokens"] <= 0]
    assert not no_external, (
        f"{len(no_external)}/{len(first_token_ab)} compared PD requests adopted "
        f"no externally computed tokens ({EXTERNAL_HITS} delta 0), so their first "
        "token cannot be evidence about disaggregation (it came from a local "
        "prefill or from the local prefix cache):\n"
        + "\n".join(
            f"  {p['prompt'][:50]!r} "
            f"external_tokens={p['pd_external_tokens']:.0f}"
            for p in no_external
        )
    )

    mismatches = [
        (p["prompt"], p["local"], p["pd"])
        for p in first_token_ab
        if p["pd"] != p["local"]
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(first_token_ab)} prompts differ between the PD "
        "path and a local prefill of the same prompt:\n"
        + "\n".join(
            f"  {p!r}\n    local={w!r}\n    pd   ={g!r}" for p, w, g in mismatches
        )
    )
    print(
        f"\nfirst-token A/B this round: {len(first_token_ab) - len(mismatches)}"
        f"/{len(first_token_ab)} identical; per compared request: "
        + ", ".join(
            (
                f"{p['pd_transfer']:.0f} transfer(s)"
                if IS_NIXL
                else f"{p['pd_external_tokens']:.0f} external tokens"
            )
            for p in first_token_ab
        )
    )


def test_pd_outputs_are_non_degenerate(round_metrics):
    assert any(t.strip() for t in round_metrics["outputs"]), (
        "every completion was empty"
    )


# ---------------------------------------------------------------------------
# 4. optional: a pre-recorded standalone reference
# ---------------------------------------------------------------------------


def test_pd_matches_recorded_standalone_reference(first_token_ab):
    """Extra check against a reference dumped by ``dsv41_dspark_reference.py``.

    Optional because the reference's prompts cannot be run-unique, so a
    mismatch here can also mean the decode instance answered partly from its own
    prefix cache.  The self-contained local A/B above is the real gate.

    Strict mode (``DSV41_STRICT_REFERENCE=1``) must not be satisfiable by
    comparing *fewer* prompts than were sent: the reference is checked for
    model/salt identity, every prompt is required to have an entry, and the
    comparison count is asserted -- "0 compared, skipped" and "1 of 4 compared,
    passed" were both accepted before.
    """
    if not REFERENCE:
        if STRICT:
            pytest.fail("DSV41_STRICT_REFERENCE=1 but DSV41_DSPARK_REFERENCE is unset")
        pytest.skip("DSV41_DSPARK_REFERENCE not set")
    with open(REFERENCE) as fh:
        ref = json.load(fh)
    assert isinstance(ref, dict) and ref, f"{REFERENCE} is not a non-empty JSON object"

    prompts = [p["prompt"] for p in first_token_ab]
    got_by_prompt = {p["prompt"]: p["pd"] for p in first_token_ab}
    if STRICT:
        meta = ref.get("__meta__", {})
        if isinstance(meta, dict):
            for key, have in (("model", MODEL_NAME), ("run_salt", RUN_SALT)):
                want = meta.get(key)
                if want is not None and want != have:
                    pytest.fail(
                        f"reference {key}={want!r} does not match this run "
                        f"({have!r}): its tokens are not comparable"
                    )

    compared = 0
    missing: list[str] = []
    mismatches = []
    for prompt in prompts:
        entry = ref.get(prompt)
        if entry is None:
            missing.append(prompt)
            continue
        compared += 1
        got = got_by_prompt[prompt]
        want = entry["first_token"] if isinstance(entry, dict) else entry[: len(got)]
        if got != want:
            mismatches.append((prompt, want, got))

    if STRICT:
        assert compared > 0, (
            f"strict reference comparison matched 0 of {len(prompts)} prompts "
            f"against {REFERENCE}"
        )
        assert not missing, (
            f"strict reference comparison covered only {compared}/{len(prompts)} "
            f"prompts; {len(missing)} have no reference entry:\n"
            + "\n".join(f"  {p[:60]!r}" for p in missing[:3])
        )
    elif compared == 0:
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
    print(f"\nreference comparison: {compared - len(mismatches)}/{compared} identical")


# ---------------------------------------------------------------------------
# 2b. negative control: an unreachable producer must not answer (LAST)
# ---------------------------------------------------------------------------
# Deliberately the final test: the request it parks is still waiting for
# remote KV when the client gives up, so the decode instance holds its
# blocks for a while and a prefix-cache reset would be refused (HTTP 500).
# Running it after the cold A/B keeps the two independent.


def test_negative_control_unreachable_producer(round_metrics):
    """A do_remote_prefill request to an unreachable producer must not answer.

    And the non-answer must be *connector-level*: a rejected request (HTTP
    400) or an unreachable decode service proves nothing, so both fail here.
    A non-answer is accepted only with evidence -- a moved transfer failure
    counter (per connector: NIXL's Prometheus series or Mooncake's log lines),
    a logged bootstrap failure, or a request parked for at least 30 s -- and
    the decode instance must still be healthy and still schedule afterwards.

    Known vLLM limitation: the parked request cannot be aborted while it waits
    for a remote transfer, so it keeps the decode instance's blocks referenced
    and *a later prefix-cache reset will be refused* until the engine restarts.
    Set ``DSV41_SKIP_NEGATIVE_CONTROL=1`` to skip it when re-running only the
    cold A/B against an instance that had one parked.
    """
    if SKIP_NEGATIVE_CONTROL:
        pytest.skip("DSV41_SKIP_NEGATIVE_CONTROL=1")
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
    if not IS_NIXL:
        # MooncakeConnector silently falls back to a *local* prefill when these
        # three keys are missing ("This request will not utilize KVTransfer"),
        # so a NIXL-shaped dict would be answered locally and the control would
        # pass for the wrong reason.
        bogus |= {
            "transfer_id": "cmpl-negative-control",
            "remote_bootstrap_addr": "http://10.255.255.1:8998",
        }
    # Sample the failure counters across the negative control only, so a
    # "transfer failed" claim cannot be inherited from an unrelated step.
    before = scrape(DECODE_METRICS_URL, (NIXL_FAILED_XFER, NIXL_FAILED_NOTIFY))
    before_log = mooncake_log_stats() if not IS_NIXL else {}
    started = time.time()
    answered = None
    failure: Exception | None = None
    try:
        resp = _complete_resp(
            DECODE_BASE_URL,
            novel,
            max_tokens=4,
            extra_body={"kv_transfer_params": bogus},
            timeout=60.0,
            # One attempt = one engine-side request.  With the SDK's default
            # two retries a "60 s" control actually made three requests and
            # parked three; that also made the measured 181.5 s unreadable.
            max_retries=0,
        )
        answered = resp.choices[0].text
    except openai.BadRequestError as exc:
        # A rejected request never reached the connector: it proves nothing
        # about remote prefill, and accepting it was the old bug.
        pytest.fail(
            "the negative control was rejected by request validation before it "
            f"could reach the connector (HTTP {exc.status_code}). Build the "
            "bogus params from a metadata block the server already accepted, "
            "changing only the producer address."
        )
    except openai.APITimeoutError as exc:
        # The expected outcome: the request is parked waiting for KV that never
        # arrives.  This must be caught before APIConnectionError, because
        # ``APITimeoutError`` subclasses it in the OpenAI SDK.
        failure = exc
    except openai.APIConnectionError as exc:
        pytest.fail(
            "the negative control could not reach the *decode* service at all "
            f"({type(exc).__name__}), so no request was ever made"
        )
    except Exception as exc:  # noqa: BLE001 - a 5xx from the connector is fine
        failure = exc
    elapsed = time.time() - started

    if answered is not None:
        pytest.fail(
            "the decode instance answered a do_remote_prefill request whose "
            f"producer (10.255.255.1:5999) is unreachable, in {elapsed:.1f}s: "
            f"{answered[:40]!r}. That is only possible by recomputing the "
            "prefill locally, so remote prefill is not in effect."
        )

    after = scrape(DECODE_METRICS_URL, (NIXL_FAILED_XFER, NIXL_FAILED_NOTIFY))
    failed = delta(before, after)
    if IS_NIXL:
        failure_evidence = failed[NIXL_FAILED_XFER] + failed[NIXL_FAILED_NOTIFY]
        connector_failure = ""
    else:
        after_log = mooncake_log_stats()
        log_delta = {k: after_log[k] - before_log.get(k, 0.0) for k in after_log}
        failure_evidence = log_delta["failed_recvs"] + log_delta["failed_transfers"]
        # A bootstrap connection failure is reported as a failed recv (the
        # connector hands the request back to the scheduler) and logged, but a
        # refused *connection* may only show up in the log, so accept either.
        bootstrap_failed = mooncake_pull_failure_seen("10.255.255.1")
        failure_evidence += 1.0 if bootstrap_failed else 0.0
        connector_failure = (
            f", bootstrap pull log evidence: {bootstrap_failed}"
            f", failed recvs +{log_delta['failed_recvs']:.0f}"
            f", failed transfers +{log_delta['failed_transfers']:.0f}"
        )
    still_pulling = connector_reports_pending_remote_kv()
    print(
        f"negative control: no answer after {elapsed:.1f}s "
        f"({type(failure).__name__ if failure else 'no exception'}), "
        f"transfer failures +{failure_evidence:.0f}, "
        f"engine still reports a pending remote KV wait: {still_pulling}"
        f"{connector_failure}"
    )
    assert failure_evidence > 0 or still_pulling, (
        "the negative control did not answer, but there is no connector-level "
        "evidence that it reached the pull path: the failure counters stayed "
        "flat, the log shows no failed bootstrap pull, and the engine does not "
        "report a request waiting for remote KV. A plain queueing delay (or a "
        'silently dropped request) satisfies "it did not answer" without '
        "proving anything about remote prefill.\n"
        f"  exception: {type(failure).__name__ if failure else 'none'}, "
        f"elapsed {elapsed:.1f}s, failures +{failure_evidence:.0f}, "
        f"decode log provided: {bool(DECODE_LOG)}"
    )
    assert decode_is_healthy(), (
        "the decode instance stopped serving after the negative control; the "
        "unreachable producer must not wedge the engine"
    )
    # The engine must also still *schedule*: a stuck waiting queue would leave
    # /health green.  This probe is a plain local request.
    probe = _complete_resp(
        DECODE_BASE_URL,
        f"[pd-negative-control-probe-{time.time_ns()}] hello",
        max_tokens=1,
    )
    assert probe.choices, "the decode instance scheduled nothing after the control"


# ---------------------------------------------------------------------------
# 2c. is a parked remote-KV request ever released? (opt-in, LAST)
# ---------------------------------------------------------------------------


def test_parked_remote_kv_request_is_released():
    """Does the connector release the request the negative control parked?

    This is the round-4 open item, made measurable.  On NixlConnector the
    request stayed parked until the engine restarted: the blocks stayed
    referenced, so ``/reset_prefix_cache`` kept answering 500 ("requests
    waiting for remote KV transfer, which is not supported yet") and the only
    recovery documented in this file was a restart.  The question is whether
    the connector's own failure path hands the request back to the scheduler.

    Opt-in (``DSV41_ABORT_TIMEOUT_PROBE=1``) because it waits, and because it
    is only meaningful after the negative control created the parked request.
    """
    if not ABORT_TIMEOUT_PROBE:
        pytest.skip("DSV41_ABORT_TIMEOUT_PROBE is not set")
    if SKIP_NEGATIVE_CONTROL:
        pytest.skip("no parked request: DSV41_SKIP_NEGATIVE_CONTROL=1")

    deadline = time.time() + ABORT_TIMEOUT_PROBE_WAIT
    released = False
    while time.time() < deadline:
        if reset_decode_prefix_cache():
            released = True
            break
        time.sleep(5)
    elapsed = ABORT_TIMEOUT_PROBE_WAIT - max(deadline - time.time(), 0.0)
    print(
        f"\nparked-request release probe ({KV_CONNECTOR}): "
        f"reset_prefix_cache succeeded={released} after <= {elapsed:.1f}s "
        f"(waited up to {ABORT_TIMEOUT_PROBE_WAIT:.0f}s)"
    )
    if IS_NIXL:
        # Known, reported limitation: NIXL leaves the request parked.  Recorded
        # rather than asserted, so an upstream fix does not turn a real
        # improvement into a test failure.
        if released:
            print(
                "NOTE: NixlConnector released the parked request -- the known "
                "limitation documented in this file no longer holds, update it."
            )
    else:
        assert released, (
            f"the parked request was still not released after "
            f"{ABORT_TIMEOUT_PROBE_WAIT:.0f}s with {KV_CONNECTOR}; the decode "
            "instance needs a restart before its prefix cache can be reset, so "
            "abort/reset is connector-independent."
        )


