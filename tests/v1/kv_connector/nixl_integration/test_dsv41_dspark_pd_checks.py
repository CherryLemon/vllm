# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only control-flow tests for the DSpark PD acceptance gates.

No server, no GPU: the strict-reference gate is a pure function of the
reference file and the measured prompt pairs, and its failure modes were
previously silent.  Both of these used to pass:

  * a non-empty reference whose prompt keys all miss this run -> ``skip``;
  * two prompts measured, one present in the reference -> ``passed`` after
    comparing a single prompt.

These call the acceptance test functions directly with a fabricated
``first_token_ab`` fixture (the shape the real fixture returns) and a temporary
reference file.
"""

import json

import openai
import pytest
import test_dsv41_dspark_pd as pd_test

# The acceptance module reports a broken gate either with ``pytest.fail`` (a
# config error) or with a plain assertion; both are failures, neither is a skip.
GATE_FAILURE = (AssertionError, pytest.fail.Exception)


def _pair(prompt: str, pd: str, local: str | None = None, transfer: float = 1.0):
    """One measured A/B pair, as ``first_token_ab`` builds it.

    ``pd_external_tokens`` is the connector-agnostic evidence that the request
    adopted remote KV; it is what the gate falls back to for a connector
    without per-transfer Prometheus counters.
    """
    return {
        "prompt": prompt,
        "pd": pd,
        "local": local if local is not None else pd,
        "local_prefix_hits": 0.0,
        "pd_transfer": transfer,
        "pd_external_tokens": 200.0 if transfer else 0.0,
        "pd_failed": 0.0,
    }


def _write_reference(tmp_path, payload: dict) -> str:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload))
    return str(path)


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setattr(pd_test, "STRICT", True)
    monkeypatch.setattr(pd_test, "RUN_SALT", "run-fixed")
    monkeypatch.setattr(pd_test, "MODEL_NAME", "deepseek-v4.1-flash")


def test_strict_fails_without_reference(monkeypatch):
    monkeypatch.setattr(pd_test, "STRICT", True)
    monkeypatch.setattr(pd_test, "REFERENCE", "")
    with pytest.raises(pytest.fail.Exception, match="DSV41_DSPARK_REFERENCE is unset"):
        pd_test.test_pd_matches_recorded_standalone_reference([_pair("p", "a")])


def test_strict_fails_when_no_prompt_matches(strict, tmp_path, monkeypatch):
    """A reference that matches nothing must fail, never skip."""
    monkeypatch.setattr(
        pd_test,
        "REFERENCE",
        _write_reference(tmp_path, {"some other prompt": {"first_token": "a"}}),
    )
    with pytest.raises(GATE_FAILURE, match="matched 0 of 2 prompts"):
        pd_test.test_pd_matches_recorded_standalone_reference(
            [_pair("p0", "a"), _pair("p1", "b")]
        )


def test_strict_fails_on_partial_coverage(strict, tmp_path, monkeypatch):
    """Two prompts measured, one in the reference -> must fail, not pass."""
    monkeypatch.setattr(
        pd_test,
        "REFERENCE",
        _write_reference(tmp_path, {"p0": {"first_token": "a"}}),
    )
    with pytest.raises(GATE_FAILURE, match="covered only 1/2 prompts"):
        pd_test.test_pd_matches_recorded_standalone_reference(
            [_pair("p0", "a"), _pair("p1", "b")]
        )


def test_strict_passes_on_full_agreement(strict, tmp_path, monkeypatch):
    monkeypatch.setattr(
        pd_test,
        "REFERENCE",
        _write_reference(
            tmp_path,
            {
                "p0": {"first_token": "a"},
                "p1": {"first_token": "b"},
                "__meta__": {"model": "deepseek-v4.1-flash", "run_salt": "run-fixed"},
            },
        ),
    )
    pd_test.test_pd_matches_recorded_standalone_reference(
        [_pair("p0", "a"), _pair("p1", "b")]
    )


def test_strict_fails_on_reference_from_another_run(strict, tmp_path, monkeypatch):
    monkeypatch.setattr(
        pd_test,
        "REFERENCE",
        _write_reference(
            tmp_path,
            {
                "p0": {"first_token": "a"},
                "__meta__": {"model": "deepseek-v4.1-flash", "run_salt": "other-run"},
            },
        ),
    )
    with pytest.raises(pytest.fail.Exception, match="run_salt"):
        pd_test.test_pd_matches_recorded_standalone_reference([_pair("p0", "a")])


def test_strict_reports_mismatch(strict, tmp_path, monkeypatch):
    monkeypatch.setattr(
        pd_test,
        "REFERENCE",
        _write_reference(tmp_path, {"p0": {"first_token": "a"}}),
    )
    with pytest.raises(AssertionError, match="differ on the first token"):
        pd_test.test_pd_matches_recorded_standalone_reference([_pair("p0", "z")])


def test_non_strict_skips_when_nothing_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(pd_test, "STRICT", False)
    monkeypatch.setattr(
        pd_test,
        "REFERENCE",
        _write_reference(tmp_path, {"other": {"first_token": "a"}}),
    )
    with pytest.raises(pytest.skip.Exception):
        pd_test.test_pd_matches_recorded_standalone_reference([_pair("p0", "a")])


def test_local_control_must_not_hit_the_prefix_cache(monkeypatch):
    """A local leg served from cache is not an independent control."""
    pair = _pair("p0", "a", local="a")
    pair["local_prefix_hits"] = 4.0
    with pytest.raises(AssertionError, match="hit the decode instance's prefix cache"):
        pd_test.test_pd_first_token_matches_local_prefill([pair])


def test_compared_pd_request_must_show_its_own_transfer(monkeypatch):
    """A zero-delta compared request proves nothing about disaggregation."""
    pair = _pair("p0", "a", local="a", transfer=0.0)
    # Isolate the NIXL counter gate: the external-token gate has its own test.
    pair["pd_external_tokens"] = 200.0
    with pytest.raises(AssertionError, match="no KV transfer of their own"):
        pd_test.test_pd_first_token_matches_local_prefill([pair])


def test_compared_pd_request_must_adopt_remote_tokens(monkeypatch):
    """The connector-agnostic gate: no external tokens adopted.

    This is the one that works for a connector without per-transfer
    Prometheus counters (Mooncake), so it must not be satisfiable by the
    NIXL-only counters alone.
    """
    pair = _pair("p0", "a", local="a", transfer=16.0)
    pair["pd_external_tokens"] = 0.0
    with pytest.raises(AssertionError, match="adopted no externally computed"):
        pd_test.test_pd_first_token_matches_local_prefill([pair])


def test_first_token_ab_passes_when_cold_and_transferred(monkeypatch):
    pd_test.test_pd_first_token_matches_local_prefill(
        [_pair("p0", "a", local="a", transfer=3.0)]
    )


def test_first_token_ab_reports_mismatch(monkeypatch):
    with pytest.raises(AssertionError, match="differ between the PD path"):
        pd_test.test_pd_first_token_matches_local_prefill(
            [_pair("p0", "a", local="z", transfer=3.0)]
        )

def test_api_timeout_is_caught_before_connection_error():
    """``APITimeoutError`` subclasses ``APIConnectionError`` in the SDK.

    The negative control must treat a timeout as "parked waiting for KV" (the
    expected outcome) and a connection error as "the decode service is down"
    (a hard failure).  Catching the base class first inverted those in the
    first live run, so the ordering is pinned here.
    """
    assert issubclass(openai.APITimeoutError, openai.APIConnectionError)
    assert issubclass(openai.BadRequestError, openai.APIStatusError)


def test_failed_transfer_on_a_compared_request_is_rejected():
    """A recovered (retried) transfer must not pass the clean-path gate."""
    pair = _pair("p0", "a", local="a", transfer=16.0)
    pair["pd_failed"] = 240.0
    with pytest.raises(
        AssertionError, match="recorded a \\*failed\\* NixlConnector transfer"
    ):
        pd_test.test_pd_first_token_matches_local_prefill([pair])


def test_mooncake_log_stats_sums_the_periodic_metrics_lines(tmp_path, monkeypatch):
    """Mooncake's only failure counters live in its periodic log lines."""
    log = tmp_path / "decode.log"
    log.write_text(
        "INFO some other line\n"
        "INFO KV Transfer metrics: Num successful transfers=4, "
        "Avg xfer time (ms)=1.2, Num failed transfers=0, Num failed recvs=0, "
        "Num KV expired reqs=0\n"
        "INFO KV Transfer metrics: Num successful transfers=2, "
        "Avg xfer time (ms)=3.4, Num failed transfers=1, Num failed recvs=2, "
        "Num KV expired reqs=3\n"
    )
    monkeypatch.setattr(pd_test, "DECODE_LOG", str(log))
    stats = pd_test.mooncake_log_stats()
    assert stats == {
        "successful": 6.0,
        "failed_transfers": 1.0,
        "failed_recvs": 2.0,
        "expired": 3.0,
    }


def test_mooncake_log_stats_is_empty_without_a_log_path(monkeypatch):
    """Unset path must report "no evidence", never a silent pass."""
    monkeypatch.setattr(pd_test, "DECODE_LOG", "")
    assert pd_test.mooncake_log_stats() == {
        "successful": 0.0,
        "failed_transfers": 0.0,
        "failed_recvs": 0.0,
        "expired": 0.0,
    }
    assert not pd_test.mooncake_pull_failure_seen("10.255.255.1")


def test_mooncake_pull_failure_needs_the_address_and_the_message(tmp_path, monkeypatch):
    """A failed pull is proven by the bootstrap error, not by the IP alone."""
    log = tmp_path / "decode.log"
    log.write_text(
        "INFO Failed to connect to bootstrap server "
        "http://10.255.255.1:8998: refused\n"
    )
    monkeypatch.setattr(pd_test, "DECODE_LOG", str(log))
    assert pd_test.mooncake_pull_failure_seen("10.255.255.1")
    assert not pd_test.mooncake_pull_failure_seen("10.9.9.9")

    log.write_text(
        "INFO pulling kv_caches for ['r1'] failed: remote engine_id deadbeef not "
        "found from bootstrap server http://10.255.255.1:8998\n"
    )
    assert pd_test.mooncake_pull_failure_seen("10.255.255.1")
