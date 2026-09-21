# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DSpark + NixlConnector PD acceptance test.

DeepSeek-V4.1 has no classic MTP draft: its checkpoints ship DSpark stages
under ``mtp.*`` (``dspark_block_size=5``, ``num_nextn_predict_layers=3``), so
speculative decoding on this model is ``method="dspark"`` and one draft round
proposes 5 tokens that the target verifies with 6 query rows.

This test drives the PD proxy (prefill instance + decode instance) with
``temperature=0`` completions and checks three things that the standalone
accuracy tests cannot check:

  1. the speculative path really engaged (drafts were produced and the mean
     acceptance length is above 1.0, i.e. at least one draft token is accepted
     per round);
  2. the PD path is *transport-transparent*: the same prompt must produce the
     same text when replayed, so a KV-transfer bug that silently drops or
     misroutes the transferred blocks shows up as non-determinism rather than
     as plausible-looking text;
  3. when ``DSV41_DSPARK_REFERENCE`` points at a JSON file of
     ``{prompt_id: text}`` produced by a standalone (non-PD) run, the PD output
     must match it exactly.

Environment variables (set by ``dsv41_dspark_pd_sm90.sh``):
    TEST_MODEL              - served model name (default: deepseek-v4.1-flash)
    SERVER_HOST             - proxy host (default: 127.0.0.1)
    PROXY_PORT              - proxy port (default: 8192)
    DECODE_PORT             - decode vLLM port, for /metrics
    NUM_PROMPTS             - prompts to send (default: 8)
    MAX_TOKENS              - output tokens (default: 64)
    DSV41_DSPARK_REFERENCE  - optional standalone reference JSON
"""

import json
import os
from urllib.request import urlopen

import openai
import pytest

SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
PROXY_PORT = os.environ.get("PROXY_PORT", "8192")
DECODE_PORT = os.environ.get("DECODE_PORT", "8200")
MODEL_NAME = os.environ.get("TEST_MODEL", "deepseek-v4.1-flash")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
REFERENCE = os.environ.get("DSV41_DSPARK_REFERENCE", "")

PROXY_BASE_URL = f"http://{SERVER_HOST}:{PROXY_PORT}/v1"

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


def _metric(name: str) -> float:
    """Read one counter from the decode server's Prometheus endpoint."""
    body = urlopen(f"http://{SERVER_HOST}:{DECODE_PORT}/metrics").read().decode()
    for line in body.split("\n"):
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[-1])
    raise ValueError(f"metric {name} not found on decode /metrics")


def _run(prompts: list[str]) -> list[str]:
    client = openai.OpenAI(api_key="EMPTY", base_url=PROXY_BASE_URL)
    out = []
    for prompt in prompts:
        resp = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            extra_body={"add_special_tokens": False},
        )
        out.append(resp.choices[0].text)
    return out


@pytest.fixture(scope="module")
def prompts() -> list[str]:
    return PROMPTS[:NUM_PROMPTS]


@pytest.fixture(scope="module")
def first_pass(prompts) -> list[str]:
    return _run(prompts)


def test_dspark_spec_decode_engages(first_pass):
    """A real draft run happened and some draft tokens were accepted."""
    n_drafts = _metric("vllm:spec_decode_num_drafts_total")
    n_accepted = _metric("vllm:spec_decode_num_accepted_tokens_total")
    assert n_drafts > 0, "no DSpark drafts were produced"
    mean_acceptance = 1 + n_accepted / n_drafts
    print(f"\nDSpark mean acceptance length = {mean_acceptance:.3f} "
          f"(drafts={n_drafts:.0f}, accepted={n_accepted:.0f})")
    assert mean_acceptance > 1.0, (
        "mean acceptance length is exactly 1.0: every draft token was rejected, "
        "so the speculative path is not actually contributing"
    )


def test_dspark_pd_outputs_are_deterministic(prompts, first_pass):
    """Replaying the same prompts through PD must reproduce byte-identical text.

    A KV-transfer defect that loses or misplaces blocks changes the decode
    context, which shows up here even when the resulting text still looks
    fluent.
    """
    second = _run(prompts)
    for prompt, a, b in zip(prompts, first_pass, second):
        assert a == b, (
            f"non-deterministic PD output for {prompt!r}:\n  first={a!r}\n"
            f"  second={b!r}"
        )


def test_dspark_pd_outputs_are_non_degenerate(first_pass):
    assert any(text.strip() for text in first_pass), "every completion was empty"


def test_dspark_pd_matches_standalone_reference(first_pass):
    if not REFERENCE:
        pytest.skip("DSV41_DSPARK_REFERENCE not set")
    with open(REFERENCE) as fh:
        ref = json.load(fh)
    mismatches = []
    for prompt, got in zip(PROMPTS[:NUM_PROMPTS], first_pass):
        want = ref.get(prompt)
        if want is None:
            continue
        if want != got:
            mismatches.append((prompt, want, got))
    assert not mismatches, (
        f"{len(mismatches)} prompts differ from the standalone reference:\n"
        + "\n".join(f"  {p!r}\n    ref={w!r}\n    pd ={g!r}" for p, w, g in mismatches)
    )