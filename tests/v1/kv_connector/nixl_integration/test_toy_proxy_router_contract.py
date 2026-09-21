# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The toy proxy's connector-specific router bookkeeping.

MooncakeConnector expects the *router* to supply two things no vLLM instance
produces, and this was silently broken before these checks existed: the proxy
sent only ``do_remote_decode``/``do_remote_prefill``, so the Mooncake prefiller
logged "Missing transfer_id in kv_transfer_params from router!", skipped the
send, and the decoder recomputed the prefill locally.  The request still
succeeded end to end, so the only observable symptom was that no KV moved --
which is why the contract is pinned here rather than left to the acceptance
test (see ``test_dsv41_dspark_pd.py``, which gates on the external
prefix-cache counters for exactly this reason).

Reference for the Mooncake side:
``examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py``.

CPU-only: no server, no GPU, no Mooncake.
"""

from types import SimpleNamespace

import pytest
import toy_proxy_server as toy


@pytest.fixture
def nixl(monkeypatch):
    monkeypatch.setattr(
        toy,
        "global_args",
        SimpleNamespace(kv_connector="NixlConnector"),
        raising=False,
    )


@pytest.fixture
def mooncake(monkeypatch):
    monkeypatch.setattr(
        toy,
        "global_args",
        SimpleNamespace(kv_connector="MooncakeConnector"),
        raising=False,
    )


def _prefill_client(engine_ids: dict[int, str] | None = None) -> dict:
    return {
        "bootstrap_addr": "http://10.8.2.13:8998",
        "dp_engine_id": dict(engine_ids or {0: "prefill-engine-id"}),
        "dp_next": 0,
    }


def test_mooncake_mode_follows_the_connector_name(nixl, monkeypatch):
    assert toy.mooncake_mode() is False
    for name in ("MooncakeConnector", "MooncakeStoreConnector"):
        monkeypatch.setattr(
            toy, "global_args", SimpleNamespace(kv_connector=name), raising=False
        )
        assert toy.mooncake_mode() is True


def test_nixl_prefiller_params_are_unchanged(nixl):
    """The verified NIXL runs must not see a different request body."""
    assert toy.prefiller_params("r1") == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


def test_mooncake_prefiller_gets_the_router_transfer_id(mooncake):
    """Without this the prefiller skips the send entirely."""
    params = toy.prefiller_params("r1")
    assert params == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": "xfer-r1",
    }


def test_mooncake_decoder_params_name_the_prefiller(mooncake):
    """Both legs must carry the same id, and the decoder must be told where.

    ``remote_engine_id`` has to be the id the prefiller's bootstrap server
    reports (the decoder resolves it in that table), not the host or the URL.
    """
    client = _prefill_client()
    decoder = toy.decoder_params("r1", client)
    assert decoder == {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": "http://10.8.2.13:8998",
        "remote_engine_id": "prefill-engine-id",
        "transfer_id": "xfer-r1",
    }
    assert decoder["transfer_id"] == toy.prefiller_params("r1")["transfer_id"]


def test_mooncake_decoder_round_robins_over_dp_ranks(mooncake):
    client = _prefill_client({0: "dp0-engine", 1: "dp1-engine"})
    seen = [toy.decoder_params(f"r{i}", client)["remote_engine_id"] for i in range(4)]
    assert seen == ["dp0-engine", "dp1-engine", "dp0-engine", "dp1-engine"]


def test_transfer_ids_are_unique_per_request(mooncake):
    """A shared id would let two requests collide on one transfer."""
    assert (
        toy.prefiller_params("r1")["transfer_id"]
        != toy.prefiller_params("r2")["transfer_id"]
    )
