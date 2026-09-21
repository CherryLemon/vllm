# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DSV41 PD router (`examples/.../dsv41_pd_router.py`).

The router decides *what the two legs are told*, and round 6 showed that
getting this wrong is invisible: a Mooncake pair whose router does not mint
``transfer_id`` logs one warning per request, skips the send, and the decoder
recomputes the prefill locally -- correct text, zero KV moved.  These tests pin
every one of those decisions against fake instances that record the requests
they receive, so they need no GPU, no model and no Mooncake:

* both connector dialects (what the prefiller and the decoder must see),
* the refusal to fall back to a local prefill when the prefiller's answer is
  incomplete,
* prefix-affinity routing and its control (round-robin),
* real concurrency (the toy proxy was synchronous per request, which is why
  every concurrency number taken through it measured the proxy),
* streaming passthrough, health aggregation and input validation.

Only ``aiohttp`` is required, which the runtime image already has.
"""

import asyncio
import itertools
import json
from typing import Any

import aiohttp

from examples.disaggregated.disaggregated_serving import dsv41_pd_router as router_mod

# (instance, body) in the order the instances received their calls; the router
# starts every leg of every request as its own task, so a per-instance list is
# not enough to reason about request order.
ORDER: list[tuple["FakeInstance", dict[str, Any]]] = []
_SEQ = itertools.count()


class FakeInstance:
    """A fake vLLM instance: records what the router sent, answers like vLLM."""

    def __init__(
        self,
        engine_id: str = "engine-0",
        prefill_params: dict[str, Any] | None | str = "nixl",
        status: int = 200,
        delay: float = 0.0,
        stream: bool = False,
    ) -> None:
        self.engine_id = engine_id
        # "nixl" = answer with a populated kv_transfer_params (what
        # NixlConnector does); None = answer without one and with no
        # kv_transfer_params at all (what MooncakeConnector does, because for
        # Mooncake the router owns those fields).
        self.prefill_params = prefill_params
        self.status = status
        self.delay = delay
        self.stream = stream
        self.calls: list[dict[str, Any]] = []
        self.stream_calls = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.url = ""

    async def _record(self, request: aiohttp.web.Request) -> dict[str, Any]:
        body = await request.json()
        self.calls.append(body)
        ORDER.append((self, body))
        next(_SEQ)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.delay:
            await asyncio.sleep(self.delay)
        return body

    def params_for_answer(self) -> dict[str, Any] | None:
        if self.prefill_params == "nixl":
            return {
                "do_remote_decode": False,
                "do_remote_prefill": True,
                "remote_engine_id": self.engine_id,
                "remote_block_ids": [0, 1, 2],
                "remote_host": "10.0.0.1",
                "remote_port": 5999,
            }
        return self.prefill_params  # type: ignore[return-value]

    def build_app(self) -> aiohttp.web.Application:
        app = aiohttp.web.Application()

        async def health(_: aiohttp.web.Request) -> aiohttp.web.Response:
            return aiohttp.web.json_response({"ok": True})

        async def query(_: aiohttp.web.Request) -> aiohttp.web.Response:
            # The Mooncake bootstrap table the decoder resolves engine ids in.
            return aiohttp.web.json_response(
                {
                    "0": {
                        "engine_id": self.engine_id,
                        "worker_addr": {"0": {"0": "tcp://h:1"}},
                    }
                }
            )

        async def completions(request: aiohttp.web.Request) -> aiohttp.web.Response:
            body = await self._record(request)
            if self.status != 200:
                self.in_flight -= 1
                return aiohttp.web.json_response(
                    {"error": {"message": "boom"}}, status=self.status
                )
            if self.stream and body.get("stream"):
                self.stream_calls += 1
                response = aiohttp.web.StreamResponse(
                    status=200, headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                for token in ("alpha", "beta"):
                    await response.write(
                        f'data: {{"choices":[{{"text":"{token}"}}]}}\n\n'.encode()
                    )
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                self.in_flight -= 1
                return response
            payload: dict[str, Any] = {"choices": [{"text": "ok"}]}
            params = self.params_for_answer()
            if params is not None:
                payload["kv_transfer_params"] = params
            self.in_flight -= 1
            return aiohttp.web.json_response(payload)

        app.router.add_get("/health", health)
        app.router.add_get("/query", query)
        app.router.add_post("/v1/completions", completions)
        return app


class Deployment:
    """Fake prefiller/decoder instances plus the router under test."""

    def __init__(self) -> None:
        self.runners: list[aiohttp.web.AppRunner] = []
        self.router: router_mod.PDRouter | None = None
        self.url = ""

    async def _serve(
        self, app: aiohttp.web.Application
    ) -> tuple[str, aiohttp.web.TCPSite]:
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        self.runners.append(runner)
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return f"http://127.0.0.1:{port}", site

    async def start(
        self,
        prefills: list[FakeInstance],
        decodes: list[FakeInstance],
        connector: str = "MooncakeConnector",
        routing: str = "prefix-affinity",
    ) -> None:
        for fake in prefills + decodes:
            fake.url, _ = await self._serve(fake.build_app())
        self.router = router_mod.PDRouter(
            prefills=[
                # /query lives on the same fake port as the instance.
                router_mod.PrefillInstance(url=f.url, index=i, bootstrap_addr=f.url)
                for i, f in enumerate(prefills)
            ],
            decodes=[f.url for f in decodes],
            connector=connector,
            routing=routing,
            affinity_entries=64,
            affinity_prefix_chars=512,
            request_timeout_s=30.0,
        )
        self.url, _ = await self._serve(router_mod.build_app(self.router))
        await self.router.start()

    async def close(self) -> None:
        if self.router is not None:
            await self.router.close()
        for runner in self.runners:
            await runner.cleanup()
        ORDER.clear()


async def post(
    url: str, body: dict[str, Any], expect: int = 200
) -> tuple[int, dict[str, Any] | str]:
    async with (
        aiohttp.ClientSession() as session,
        session.post(url, json=body) as resp,
    ):
        text = await resp.text()
        if expect and resp.status != expect:
            raise AssertionError(f"expected HTTP {expect}, got {resp.status}: {text}")
        try:
            return resp.status, json.loads(text)
        except ValueError:
            return resp.status, text


def completion_body(prompt: str, **extra: Any) -> dict[str, Any]:
    body = {"model": "m", "prompt": prompt, "max_tokens": 8}
    body.update(extra)
    return body


def prefill_prompts(fake: FakeInstance) -> list[str]:
    return [call["prompt"] for call in fake.calls if "prompt" in call]


# ---------------------------------------------------------------------------
# 1. connector dialects
# ---------------------------------------------------------------------------


def test_mooncake_dialect_mints_the_transfer_id_and_names_the_prefiller():
    """Both legs must carry one router-minted id, and the decoder the address."""

    async def scenario() -> None:
        prefill = FakeInstance(prefill_params=None)
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            await post(f"{dep.url}/v1/completions", completion_body("hello world"))

            assert len(prefill.calls) == 1 and len(decode.calls) == 1
            p_params = prefill.calls[0]["kv_transfer_params"]
            d_params = decode.calls[0]["kv_transfer_params"]
            assert p_params["do_remote_decode"] is True
            assert p_params["transfer_id"].startswith("xfer-")
            assert d_params["do_remote_prefill"] is True
            assert d_params["do_remote_decode"] is False
            assert d_params["transfer_id"] == p_params["transfer_id"]
            # The decoder resolves remote_engine_id in the bootstrap table, so
            # it has to be the id the prefiller reported, not the host or URL.
            assert d_params["remote_engine_id"] == prefill.engine_id
            assert d_params["remote_bootstrap_addr"] == prefill.url
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_nixl_dialect_forwards_what_the_prefiller_returned():
    """NixlConnector answers with its own params; the router must not invent them."""

    async def scenario() -> None:
        prefill = FakeInstance(prefill_params="nixl", engine_id="p-engine")
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="NixlConnector")
        try:
            await post(f"{dep.url}/v1/completions", completion_body("hello"))
            p_params = prefill.calls[0]["kv_transfer_params"]
            # No transfer_id is minted for NIXL -- that is the Mooncake contract.
            assert "transfer_id" not in p_params
            assert decode.calls[0]["kv_transfer_params"] == prefill.params_for_answer()
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_missing_prefiller_params_is_a_hard_error_not_a_local_prefill():
    """A prefiller that answers without params must not reach the decoder.

    This is the round-6 failure mode: proceeding would answer from a local
    prefill, which the client cannot tell apart from a healthy PD run.
    """

    async def scenario() -> None:
        prefill = FakeInstance(prefill_params=None)  # nothing to forward
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="NixlConnector")
        try:
            status, body = await post(
                f"{dep.url}/v1/completions", completion_body("hello"), expect=0
            )
            assert status == 502, body
            assert "kv_transfer_params" in str(body)
            assert decode.calls == [], "the decoder must not be asked to run"
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_mooncake_router_without_a_bootstrap_address_fails_loudly():
    """A Mooncake deployment started without its bootstrap port must not serve."""

    async def scenario() -> None:
        prefill = FakeInstance(prefill_params=None)
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            assert dep.router is not None
            dep.router.prefills[0].bootstrap_addr = None
            status, body = await post(
                f"{dep.url}/v1/completions", completion_body("x"), expect=0
            )
            # 500: this is the router's own configuration, not a leg failure.
            assert status == 500, body
            assert "bootstrap" in str(body)
            assert prefill.calls == [] and decode.calls == []
        finally:
            await dep.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 2. routing
# ---------------------------------------------------------------------------


def test_prefix_affinity_keeps_the_dp_rank_of_a_repeated_prompt():
    """Affinity has to pin the *rank*: the prefix lives in one engine's cache.

    With Mooncake the router chooses a data-parallel rank per request, so
    routing a repeat to the same instance but another rank would still miss.
    """

    async def scenario() -> None:
        class TwoRank(FakeInstance):
            def build_app(self) -> aiohttp.web.Application:
                app = super().build_app()

                async def query(_: aiohttp.web.Request) -> aiohttp.web.Response:
                    return aiohttp.web.json_response(
                        {
                            "0": {
                                "engine_id": "e0",
                                "worker_addr": {"0": {"0": "tcp://h:0"}},
                            },
                            "1": {
                                "engine_id": "e1",
                                "worker_addr": {"0": {"0": "tcp://h:1"}},
                            },
                        }
                    )

                app.router.add_get("/query", query)
                return app

        prefill = TwoRank(prefill_params=None)
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            prompt = "same prompt twice"
            await post(f"{dep.url}/v1/completions", completion_body(prompt))
            await post(f"{dep.url}/v1/completions", completion_body("different"))
            await post(f"{dep.url}/v1/completions", completion_body(prompt))
            first = prefill.calls[0]["kv_transfer_params"]["transfer_id"]
            repeat = prefill.calls[2]["kv_transfer_params"]["transfer_id"]
            # Same rank => same decode-side engine id for both occurrences of
            # the prompt; the middle request is free to use the other one.
            d_params = [
                c["kv_transfer_params"]["remote_engine_id"] for c in decode.calls
            ]
            assert d_params[0] == d_params[2], d_params
            assert dep.router is not None
            assert dep.router.metrics.affinity_hits == 1
            assert first != repeat  # ids are per request, not per prompt
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_prefix_affinity_sends_a_repeated_prompt_to_the_same_instances():
    async def scenario() -> None:
        prefills = [FakeInstance("p0"), FakeInstance("p1")]
        decodes = [FakeInstance("d0"), FakeInstance("d1")]
        dep = Deployment()
        await dep.start(prefills, decodes, routing="prefix-affinity")
        try:
            repeated = "shared prefix here and more text"
            await post(f"{dep.url}/v1/completions", completion_body(repeated))
            await post(f"{dep.url}/v1/completions", completion_body("unrelated"))
            await post(f"{dep.url}/v1/completions", completion_body(repeated))

            assert dep.router is not None
            assert dep.router.metrics.affinity_hits == 1
            # The two occurrences of the repeated prompt must share an instance;
            # the unrelated one is free to go anywhere.
            owners = [
                i
                for i, fake in enumerate(prefills)
                if repeated in prefill_prompts(fake)
            ]
            assert len(owners) == 1, [prefill_prompts(f) for f in prefills]
            assert len(prefills[owners[0]].calls) == 2
            assert sum(len(f.calls) for f in prefills) == 3
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_round_robin_spreads_requests_over_both_instances():
    async def scenario() -> None:
        prefills = [FakeInstance("p0"), FakeInstance("p1")]
        decodes = [FakeInstance("d0"), FakeInstance("d1")]
        dep = Deployment()
        await dep.start(prefills, decodes, routing="round-robin")
        try:
            for i in range(4):
                await post(f"{dep.url}/v1/completions", completion_body(f"prompt {i}"))
            assert dep.router is not None
            assert dep.router.metrics.affinity_hits == 0
            assert [len(f.calls) for f in prefills] == [2, 2]
            assert [len(f.calls) for f in decodes] == [2, 2]
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_a_failed_prefill_leg_drops_the_affinity_entry():
    """Otherwise every later request keeps retrying the broken instance."""

    async def scenario() -> None:
        bad = FakeInstance("bad", status=500)
        good = FakeInstance("good")
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([bad, good], [decode], routing="prefix-affinity")
        try:
            status, _ = await post(
                f"{dep.url}/v1/completions", completion_body("same prompt"), expect=0
            )
            assert status == 502
            assert decode.calls == []
            # The prompt is free to be routed again, and the healthy instance
            # takes it.
            await post(f"{dep.url}/v1/completions", completion_body("same prompt"))
            assert len(good.calls) == 1
        finally:
            await dep.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 3. concurrency (the property the toy proxy did not have)
# ---------------------------------------------------------------------------


def test_concurrent_requests_really_overlap():
    """Six clients must be in flight at once, not queued behind each other.

    The toy proxy ran one request end to end before starting the next, which
    made every concurrency measurement a measurement of the proxy.
    """

    async def scenario() -> None:
        delay = 0.3
        prefill = FakeInstance(prefill_params=None, delay=delay)
        decode = FakeInstance(delay=delay)
        dep = Deployment()
        await dep.start([prefill], [decode], routing="round-robin")
        try:
            started = asyncio.get_running_loop().time()
            await asyncio.gather(
                *(
                    post(f"{dep.url}/v1/completions", completion_body(f"p{i}"))
                    for i in range(6)
                )
            )
            elapsed = asyncio.get_running_loop().time() - started
            # Serialised would be 6 * (0.3 + 0.3) = 3.6 s; overlapped is ~0.6 s.
            assert elapsed < 1.5, f"requests did not overlap ({elapsed:.2f}s)"
            assert decode.max_in_flight > 1, decode.max_in_flight
            assert prefill.max_in_flight > 1, prefill.max_in_flight
            assert dep.router is not None
            assert dep.router.metrics.max_in_flight > 1
            assert dep.router.metrics.in_flight == 0
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_streaming_is_passed_through_in_order():
    async def scenario() -> None:
        prefill = FakeInstance(prefill_params=None)
        decode = FakeInstance(stream=True)
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{dep.url}/v1/completions",
                    json=completion_body("stream me", stream=True),
                ) as resp,
            ):
                body = await resp.text()
                status = resp.status
            assert status == 200
            assert "alpha" in body and "beta" in body
            assert body.index("alpha") < body.index("beta")
            assert "[DONE]" in body
            assert decode.stream_calls == 1
            # The prefill leg is never streamed and always asks for one token.
            assert prefill.calls[0]["stream"] is False
            assert prefill.calls[0]["max_tokens"] == 1
        finally:
            await dep.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 4. guard rails
# ---------------------------------------------------------------------------


def test_client_supplied_transfer_params_are_rejected():
    """The router owns kv_transfer_params; a caller must not be able to set it."""

    async def scenario() -> None:
        prefill = FakeInstance(prefill_params=None)
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            status, _ = await post(
                f"{dep.url}/v1/completions",
                completion_body("hi", kv_transfer_params={"do_remote_prefill": True}),
                expect=0,
            )
            assert status == 400
            assert prefill.calls == [] and decode.calls == []
        finally:
            await dep.close()

    asyncio.run(scenario())


def test_healthcheck_requires_every_instance():
    async def healthy() -> None:
        prefill = FakeInstance(prefill_params=None)
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"{dep.url}/healthcheck") as resp,
            ):
                body = await resp.json()
                assert resp.status == 200
                assert body["connector"] == "MooncakeConnector"
                assert body["prefill_instances"] == 1
                assert body["status"] == "ok"
        finally:
            await dep.close()

    async def one_instance_down() -> None:
        prefill = FakeInstance(prefill_params=None)
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            assert dep.router is not None
            # Point the router at a decoder that is not listening any more.
            dep.router.decodes = ["http://127.0.0.1:1"]
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"{dep.url}/healthcheck") as resp,
            ):
                assert resp.status == 503
        finally:
            await dep.close()

    asyncio.run(healthy())
    asyncio.run(one_instance_down())


def test_metrics_expose_the_routing_decision():
    async def scenario() -> None:
        prefill = FakeInstance(prefill_params=None)
        decode = FakeInstance()
        dep = Deployment()
        await dep.start([prefill], [decode], connector="MooncakeConnector")
        try:
            await post(f"{dep.url}/v1/completions", completion_body("count me"))
            async with (
                aiohttp.ClientSession() as session,
                session.get(f"{dep.url}/metrics") as resp,
            ):
                text = await resp.text()
            assert "dsv41_router_requests_total 1" in text
            assert "dsv41_router_requests_failed_total 0" in text
            assert "dsv41_router_max_in_flight 1" in text
        finally:
            await dep.close()

    asyncio.run(scenario())
