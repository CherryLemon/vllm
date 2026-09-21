# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSV41 prefill/decode router: connector-aware, concurrent, prefix-affine.

Why this file exists
====================

The DSpark PD harness has always used ``toy_proxy_server.py`` for the two-hop
orchestration (``do_remote_decode`` -> forward ``kv_transfer_params`` ->
``do_remote_prefill``).  That proxy is test scaffolding, and round 6 measured
what it costs:

* **It is synchronous per request.**  One request goes through prefill, waits
  for the prefill to finish, then decodes; a second client queues behind it.
  Every concurrency number taken through it is a measurement of the proxy, not
  of the deployment (which is why the c8/c32 spreads were 15-25 % and why c32
  was unusable).
* **It speaks one dialect.**  ``NixlConnector`` answers with a fully populated
  ``kv_transfer_params``, so forwarding the response is enough.
  ``MooncakeConnector`` instead expects the *router* to mint ``transfer_id``
  for the prefill leg and to name the prefiller (``remote_engine_id`` from its
  bootstrap ``/query``, ``remote_bootstrap_addr``) for the decode leg --
  without them it silently recomputes the prefill **locally** and the run looks
  green while moving no KV.

llm-d's EPP and vLLM's ``vllm-router`` are the production answers, but neither
is installable in this offline lab (llm-d needs a Kubernetes Gateway API
control plane and a Gateway/HTTPRoute, and the images are not on the shared
disk).  This router implements the three things they provide for *this*
deployment, and nothing else:

1. **Connector dialects** (``--kv-connector``): NIXL forwards the prefiller's
   own parameters; Mooncake gets a router-minted ``transfer_id`` plus the
   prefiller's bootstrap address and engine id.
2. **Concurrency**: each request is an independent asyncio task against a
   pooled ``aiohttp`` session, so N concurrent clients really overlap.  The
   router reports ``in_flight``/``max_in_flight`` so that this is observable
   rather than asserted.
3. **Prefix-affinity routing** (``--routing prefix-affinity``): a prompt whose
   leading characters were served before goes back to the *same* prefill
   instance, so its prefix cache hits and the prefill leg becomes nearly free.
   That is the cheap, correct core of KV-aware routing; the event-driven
   version (llm-d/GIE consume ``--kv-events-config`` block hashes) makes the
   same decision with a better-informed key.  ``--routing round-robin`` is the
   control.

Failure modes it refuses
========================

* A client request that already carries ``kv_transfer_params`` is rejected
  (HTTP 400): the router owns that field, and letting a caller set it is how a
  PD deployment ends up doing local prefills that look like successes.
* A decode leg whose connector-specific parameters are missing is a hard error
  (502), never a fallback to local prefill.
* A failed prefill leg does not reach the decoder at all, and its affinity
  entry is dropped so the next attempt can go elsewhere.

Usage
=====

::

    python -m examples.disaggregated.disaggregated_serving.dsv41_pd_router \\
        --prefill http://10.8.2.13:8200 --prefill-bootstrap-port 8998 \\
        --decode http://10.8.2.9:8300 \\
        --kv-connector MooncakeConnector \\
        --routing prefix-affinity \\
        --host 0.0.0.0 --port 8192

Both ``--prefill`` and ``--decode`` may be repeated for multi-instance
deployments; ``--routing round-robin`` spreads requests over them.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import web

logger = logging.getLogger("dsv41_pd_router")

# Connector names are matched on the same substring the harness uses, so
# "MooncakeConnector" and "MooncakeStoreConnector" share one dialect.
MOONCAKE_DIALECT = "Mooncake"

# Endpoints the decoder needs when the router (not the prefiller) owns the
# decode-side parameters.
DECODER_PARAM_KEYS = ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")


class RouterError(Exception):
    """A leg failed in a way the client must see."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


def uses_mooncake_dialect(connector: str) -> bool:
    return MOONCAKE_DIALECT in connector


def transfer_id_for(request_id: str) -> str:
    """The id both legs of one request share (router-minted)."""
    return f"xfer-{request_id}"


def prefiller_params(connector: str, request_id: str) -> dict[str, Any]:
    """``kv_transfer_params`` for the prefill leg."""
    if uses_mooncake_dialect(connector):
        return {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "transfer_id": transfer_id_for(request_id),
        }
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


def decoder_params(
    connector: str, request_id: str, instance: PrefillInstance, dp_rank: int
) -> dict[str, Any]:
    """``kv_transfer_params`` for the decode leg (Mooncake dialect).

    ``remote_engine_id`` must be the id the prefiller's bootstrap server
    reports: the decoder resolves it in exactly that table.
    """
    if not uses_mooncake_dialect(connector):
        raise ValueError("decoder_params is Mooncake-only")
    return {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": instance.bootstrap_addr,
        "remote_engine_id": instance.dp_engine_id[dp_rank],
        "transfer_id": transfer_id_for(request_id),
    }


def affinity_key(prompt: str, prefix_chars: int) -> str:
    """Routing key: a hash of the prompt's leading characters.

    Characters, not tokens, on purpose: the router would otherwise need the
    model's tokenizer and the same block-hash function vLLM uses.  Prefix
    caching only needs the key to be *stable and shared by the prompts that
    share a prefix*, which a character prefix gives us.
    """
    if prefix_chars <= 0:
        prefix_chars = len(prompt)
    return hashlib.blake2b(
        prompt[:prefix_chars].encode("utf-8", "replace"), digest_size=16
    ).hexdigest()


def prompt_of(body: dict[str, Any]) -> str:
    """The routable text of a completion or chat-completion request."""
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):  # token ids or a batch; keep it hashable
        return "\x00".join(str(p) for p in prompt)
    messages = body.get("messages")
    if isinstance(messages, list):
        return "\x00".join(str(m.get("content", "")) for m in messages)
    return ""


@dataclass
class PrefillInstance:
    url: str
    index: int
    bootstrap_addr: str | None = None
    dp_engine_id: dict[int, str] = field(default_factory=dict)
    dp_next: int = 0

    def next_dp_rank(self) -> int:
        if not self.dp_engine_id:
            raise RouterError(
                f"prefiller {self.url} has no engine ids: the Mooncake "
                "bootstrap /query either was not called or failed",
                status=503,
            )
        ranks = sorted(self.dp_engine_id)
        rank = ranks[self.dp_next % len(ranks)]
        self.dp_next += 1
        return rank


class Metrics:
    """Plain-text counters; the router is a data path, not a metrics server."""

    def __init__(self) -> None:
        self.requests = 0
        self.failed = 0
        self.rejected = 0
        self.affinity_hits = 0
        self.prefill_ms_sum = 0.0
        self.total_ms_sum = 0.0
        self.in_flight = 0
        self.max_in_flight = 0

    def render(self) -> str:
        lines = [
            f"dsv41_router_requests_total {self.requests}",
            f"dsv41_router_requests_failed_total {self.failed}",
            f"dsv41_router_requests_rejected_total {self.rejected}",
            f"dsv41_router_affinity_hits_total {self.affinity_hits}",
            f"dsv41_router_prefill_ms_sum {self.prefill_ms_sum:.3f}",
            f"dsv41_router_total_ms_sum {self.total_ms_sum:.3f}",
            f"dsv41_router_in_flight {self.in_flight}",
            f"dsv41_router_max_in_flight {self.max_in_flight}",
        ]
        return "\n".join(lines) + "\n"


class PDRouter:
    def __init__(
        self,
        prefills: list[PrefillInstance],
        decodes: list[str],
        connector: str,
        routing: str,
        affinity_entries: int,
        affinity_prefix_chars: int,
        request_timeout_s: float,
    ) -> None:
        if not prefills:
            raise ValueError("at least one --prefill is required")
        if not decodes:
            raise ValueError("at least one --decode is required")
        self.prefills = prefills
        self.decodes = decodes
        self.connector = connector
        self.routing = routing
        self.affinity_prefix_chars = affinity_prefix_chars
        # prompt key -> (prefill index, decode index, prefiller DP rank).  The
        # DP rank belongs in the key: with Mooncake the router picks one rank
        # per request, and a prefix only hits if the *same* rank sees it again
        # (that is the granularity llm-d/GIE route at too).
        self.affinity: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
        self.affinity_entries = affinity_entries
        self.request_timeout_s = request_timeout_s
        self.metrics = Metrics()
        self.session: aiohttp.ClientSession | None = None
        self._rr_prefill = 0
        self._rr_decode = 0
        self._rr_dp = 0
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=0, ttl_dns_cache=60),
        )

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    async def healthy(self) -> bool:
        """Every instance must answer /health; a partial deployment is not ready."""
        assert self.session is not None
        urls = [f"{i.url}/health" for i in self.prefills]
        urls += [f"{u}/health" for u in self.decodes]
        for url in urls:
            try:
                async with self.session.get(url) as resp:
                    if resp.status != 200:
                        return False
            except aiohttp.ClientError:
                return False
        return True

    # -- routing -----------------------------------------------------------
    def _pick_round_robin(self) -> tuple[int, int]:
        prefill = self._rr_prefill % len(self.prefills)
        decode = self._rr_decode % len(self.decodes)
        self._rr_prefill += 1
        self._rr_decode += 1
        self._rr_dp += 1
        return prefill, decode

    def pick(self, prompt: str, dp_rank: int) -> tuple[int, int, int, bool]:
        """Return ``(prefill_index, decode_index, dp_rank, affinity_hit)``.

        ``dp_rank`` is what a fresh assignment would use; an affinity hit
        overrides it with the rank that served the prefix before.
        """
        if self.routing != "prefix-affinity":
            prefill, decode = self._pick_round_robin()
            return prefill, decode, dp_rank, False
        key = affinity_key(prompt, self.affinity_prefix_chars)
        hit = self.affinity.get(key)
        if hit is not None:
            self.affinity.move_to_end(key)
            self.metrics.affinity_hits += 1
            return hit[0], hit[1], hit[2], True
        prefill, decode = self._pick_round_robin()
        self.affinity[key] = (prefill, decode, dp_rank)
        while len(self.affinity) > self.affinity_entries:
            self.affinity.popitem(last=False)
        return prefill, decode, dp_rank, False

    def _next_dp_seed(self) -> int:
        """A fresh DP rank for a new affinity entry (round-robin per instance)."""
        if not self.prefills:
            return 0
        instance = self.prefills[0]
        if not instance.dp_engine_id:
            return 0
        ranks = sorted(instance.dp_engine_id)
        return ranks[self._rr_dp % len(ranks)]

    def forget(self, prompt: str) -> None:
        if self.routing == "prefix-affinity":
            self.affinity.pop(affinity_key(prompt, self.affinity_prefix_chars), None)

    # -- connector bookkeeping --------------------------------------------
    async def resolve_prefiller(self, instance: PrefillInstance) -> None:
        """Learn the prefiller's per-DP engine ids from its bootstrap server."""
        assert self.session is not None
        url = f"{instance.bootstrap_addr}/query"
        try:
            async with self.session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except (aiohttp.ClientError, ValueError) as exc:
            raise RouterError(f"prefiller bootstrap {url} failed: {exc}") from exc
        instance.dp_engine_id = {
            int(rank): entry["engine_id"] for rank, entry in data.items()
        }
        if not instance.dp_engine_id:
            raise RouterError(f"prefiller {url} reported no data-parallel ranks")
        logger.info(
            "prefiller %s resolved: %d DP rank(s)",
            instance.url,
            len(instance.dp_engine_id),
        )

    # -- request path ------------------------------------------------------
    async def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        try:
            async with self.session.post(url, json=body) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RouterError(
                        f"{url} returned HTTP {resp.status}: {text[:300]}"
                    )
                try:
                    return json.loads(text)
                except ValueError as exc:
                    raise RouterError(f"{url} returned non-JSON: {text[:120]}") from exc
        except aiohttp.ClientError as exc:
            raise RouterError(f"{url} is unreachable: {exc}") from exc

    async def _orchestrate(
        self, endpoint: str, body: dict[str, Any], request_id: str
    ) -> tuple[dict[str, Any], float, bool]:
        """Run the prefill leg and return the decode body plus its cost."""
        prompt = prompt_of(body)
        if uses_mooncake_dialect(self.connector):
            # The bootstrap table is per instance, so resolve every candidate
            # before choosing a rank: an affinity hit needs the rank that holds
            # the prefix, not just the instance.
            for instance in self.prefills:
                if instance.bootstrap_addr is None:
                    raise RouterError(
                        "Mooncake routing needs --prefill-bootstrap-port", status=500
                    )
                if not instance.dp_engine_id:
                    await self.resolve_prefiller(instance)
        seed_rank = self._next_dp_seed()
        prefill_index, decode_index, dp_rank, hit = self.pick(prompt, seed_rank)
        prefill = self.prefills[prefill_index]
        decode_url = f"{self.decodes[decode_index]}{endpoint}"

        # The prefill leg is always one token, never streamed.
        p_body = dict(body)
        p_body["kv_transfer_params"] = prefiller_params(self.connector, request_id)
        p_body["stream"] = False
        p_body["max_tokens"] = 1
        if "max_completion_tokens" in p_body:
            p_body["max_completion_tokens"] = 1
        for key in ("min_tokens", "min_completion_tokens", "stream_options"):
            p_body.pop(key, None)

        started = time.perf_counter()
        try:
            p_json = await self._post_json(f"{prefill.url}{endpoint}", p_body)
        except RouterError:
            # Never let a broken prefiller keep a poisoned affinity entry: the
            # next attempt must be free to pick another instance.
            self.forget(prompt)
            raise
        prefill_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.prefill_ms_sum += prefill_ms

        d_body = dict(body)
        if uses_mooncake_dialect(self.connector):
            d_body["kv_transfer_params"] = decoder_params(
                self.connector, request_id, prefill, dp_rank
            )
        else:
            params = p_json.get("kv_transfer_params") or {}
            missing = [
                k for k in ("remote_engine_id", "remote_block_ids") if k not in params
            ]
            if missing:
                # The prefiller answered without the fields the decoder needs.
                # Proceeding would answer from a local prefill, which is the
                # silent failure this router exists to prevent.
                self.forget(prompt)
                raise RouterError(
                    "prefiller did not return kv_transfer_params "
                    f"(missing {missing}); the decode leg would recompute locally"
                )
            d_body["kv_transfer_params"] = params
        return {"body": d_body, "decode_url": decode_url}, prefill_ms, hit

    async def handle(self, request: web.Request, endpoint: str) -> web.StreamResponse:
        if request.body_exists is False:
            self.metrics.rejected += 1
            raise web.HTTPBadRequest(reason="empty body")
        try:
            body = await request.json()
        except ValueError:
            self.metrics.rejected += 1
            raise web.HTTPBadRequest(reason="body must be JSON") from None
        if not isinstance(body, dict):
            self.metrics.rejected += 1
            raise web.HTTPBadRequest(reason="body must be a JSON object")
        if body.get("kv_transfer_params"):
            self.metrics.rejected += 1
            raise web.HTTPBadRequest(
                reason="kv_transfer_params is owned by the router; send a plain "
                "request and let it do the two-hop orchestration"
            )

        request_id = str(uuid.uuid4())
        self.metrics.requests += 1
        self.metrics.in_flight += 1
        self.metrics.max_in_flight = max(
            self.metrics.max_in_flight, self.metrics.in_flight
        )
        started = time.perf_counter()
        try:
            plan, prefill_ms, hit = await self._orchestrate(endpoint, body, request_id)
            logger.debug(
                "request %s -> prefill %.1f ms (affinity_hit=%s)",
                request_id,
                prefill_ms,
                hit,
            )
            if body.get("stream"):
                return await self._stream(request, plan, request_id, started)
            return await self._unary(plan, request_id, started)
        except RouterError as exc:
            self.metrics.failed += 1
            logger.warning("request %s failed: %s", request_id, exc)
            return web.json_response(
                {"error": {"message": str(exc)}}, status=exc.status
            )
        except asyncio.CancelledError:
            self.metrics.failed += 1
            raise
        finally:
            self.metrics.in_flight -= 1

    async def _unary(
        self, plan: dict[str, Any], request_id: str, started: float
    ) -> web.Response:
        assert self.session is not None
        try:
            async with self.session.post(plan["decode_url"], json=plan["body"]) as resp:
                payload = await resp.read()
                status = resp.status
                # Pass the upstream Content-Type through verbatim: it carries a
                # charset, which web.Response(content_type=...) rejects.
                headers = {}
                if content_type := resp.headers.get("Content-Type"):
                    headers["Content-Type"] = content_type
        except aiohttp.ClientError as exc:
            self.metrics.failed += 1
            raise RouterError(f"decode leg unreachable: {exc}") from exc
        if status != 200:
            self.metrics.failed += 1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.total_ms_sum += elapsed_ms
        logger.debug("request %s done in %.1f ms", request_id, elapsed_ms)
        return web.Response(body=payload, status=status, headers=headers)

    async def _stream(
        self,
        request: web.Request,
        plan: dict[str, Any],
        request_id: str,
        started: float,
    ) -> web.StreamResponse:
        assert self.session is not None
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)
        first_chunk_ms: float | None = None
        try:
            async with self.session.post(
                plan["decode_url"], json=plan["body"]
            ) as upstream:
                if upstream.status != 200:
                    payload = await upstream.read()
                    self.metrics.failed += 1
                    await response.write(
                        b'data: {"error": {"message": "decode leg failed: '
                        + payload[:200].replace(b'"', b"'")
                        + b'"}}\n\ndata: [DONE]\n\n'
                    )
                    await response.write_eof()
                    return response
                async for chunk in upstream.content.iter_any():
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - started) * 1000.0
                    await response.write(chunk)
        except aiohttp.ClientError as exc:
            self.metrics.failed += 1
            logger.warning("request %s stream broke: %s", request_id, exc)
        self.metrics.total_ms_sum += (time.perf_counter() - started) * 1000.0
        logger.debug(
            "request %s streamed, first chunk at %s ms",
            request_id,
            f"{first_chunk_ms:.1f}" if first_chunk_ms else "n/a",
        )
        await response.write_eof()
        return response


def build_app(router: PDRouter) -> web.Application:
    app = web.Application()
    app["router"] = router

    async def healthcheck(_: web.Request) -> web.Response:
        ready = await router.healthy()
        return web.json_response(
            {
                "status": "ok" if ready else "not ready",
                "prefill_instances": len(router.prefills),
                "decode_instances": len(router.decodes),
                "connector": router.connector,
                "routing": router.routing,
            },
            status=200 if ready else 503,
        )

    async def metrics(_: web.Request) -> web.Response:
        return web.Response(text=router.metrics.render(), content_type="text/plain")

    async def completions(request: web.Request) -> web.StreamResponse:
        return await router.handle(request, "/v1/completions")

    async def chat_completions(request: web.Request) -> web.StreamResponse:
        return await router.handle(request, "/v1/chat/completions")

    app.router.add_get("/healthcheck", healthcheck)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat_completions)
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument(
        "--prefill",
        action="append",
        required=True,
        metavar="URL",
        help="prefiller base URL (repeatable), e.g. http://10.8.2.13:8200",
    )
    parser.add_argument(
        "--prefill-bootstrap-port",
        type=int,
        default=8998,
        help="Mooncake bootstrap port on each prefiller host",
    )
    parser.add_argument(
        "--decode",
        action="append",
        required=True,
        metavar="URL",
        help="decoder base URL (repeatable), e.g. http://10.8.2.9:8300",
    )
    parser.add_argument("--kv-connector", default="NixlConnector")
    parser.add_argument(
        "--routing",
        choices=("round-robin", "prefix-affinity"),
        default="prefix-affinity",
    )
    parser.add_argument(
        "--affinity-entries",
        type=int,
        default=4096,
        help="prefix-affinity table size (LRU)",
    )
    parser.add_argument(
        "--affinity-prefix-chars",
        type=int,
        default=512,
        help="characters of the prompt used as the affinity key (0 = whole prompt)",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=600.0,
        help="per-leg HTTP timeout; a PD pair can be slow to prefill",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def prefill_instances_from_args(args: argparse.Namespace) -> list[PrefillInstance]:
    instances = []
    for index, url in enumerate(args.prefill):
        url = url.rstrip("/")
        host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
        instances.append(
            PrefillInstance(
                url=url,
                index=index,
                bootstrap_addr=f"http://{host}:{args.prefill_bootstrap_port}",
            )
        )
    return instances


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    router = PDRouter(
        prefills=prefill_instances_from_args(args),
        decodes=[u.rstrip("/") for u in args.decode],
        connector=args.kv_connector,
        routing=args.routing,
        affinity_entries=args.affinity_entries,
        affinity_prefix_chars=args.affinity_prefix_chars,
        request_timeout_s=args.request_timeout_s,
    )
    app = build_app(router)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await router.start()
    try:
        # Resolve the Mooncake bookkeeping once, before serving: a router that
        # cannot name its prefiller must not accept traffic.
        if uses_mooncake_dialect(router.connector):
            for instance in router.prefills:
                await router.resolve_prefiller(instance)
        await site.start()
        logger.info(
            "dsv41 PD router on %s:%d "
            "(connector=%s routing=%s, %d prefill / %d decode)",
            args.host,
            args.port,
            router.connector,
            router.routing,
            len(router.prefills),
            len(router.decodes),
        )
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await router.close()


def main(argv: list[str] | None = None) -> None:
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()
