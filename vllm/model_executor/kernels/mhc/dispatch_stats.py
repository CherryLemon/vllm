# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in host-side dispatch counters for the SM90 mHC TileLang paths.

Set ``VLLM_MHC_DISPATCH_STATS=1`` to turn these on (default off, so production
pays one bool test per dispatch). Everything counted here is already on the
host: ``num_tokens`` is ``x.shape[0]`` and the selected kernel is a Python
branch, so recording never enqueues a device-to-host copy and is safe while a
CUDA graph is capturing. Note that Python only runs while a graph is being
*captured* (or for eager calls), never during replay, so the counts describe
which shapes the dispatcher saw while capturing, not how many times each
captured graph was replayed.

A summary is emitted through :mod:`vllm.logger` every
``VLLM_MHC_DISPATCH_STATS_INTERVAL`` recorded calls (default 1000; 0 disables
it) and can be read at any time with :func:`snapshot`.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENV_ENABLED = "VLLM_MHC_DISPATCH_STATS"
_ENV_INTERVAL = "VLLM_MHC_DISPATCH_STATS_INTERVAL"


def _env_value(name: str) -> Any | None:
    """Read ``name`` through :mod:`vllm.envs`, falling back to ``os.environ``.

    The fallback mirrors ``has_sm90_mhc_split_h``: it keeps this usable before
    the env entry is registered and lets a test set ``os.environ`` without
    reloading :mod:`vllm.envs`.
    """
    try:
        from vllm import envs

        return getattr(envs, name, None)
    except Exception:  # pragma: no cover - import-order safety only
        return None


def _read_bool(name: str, default: bool) -> bool:
    value = _env_value(name)
    if value is not None:
        return bool(value)
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_int(name: str, default: int) -> int:
    value = _env_value(name)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Resolved once at import time; the call sites test this bool directly.
ENABLED: bool = _read_bool(_ENV_ENABLED, False)
# Recorded calls between periodic summaries; <= 0 disables the periodic log.
LOG_INTERVAL: int = _read_int(_ENV_INTERVAL, 1000)


def num_tokens_bucket(num_tokens: int) -> str:
    """Fixed-width floor(log2) bucket label, e.g. ``"[64,127]"``.

    ``num_tokens <= 0`` has its own ``"0"`` bucket. The split-H guard boundary
    (64 vs 65) falls inside ``[64,127]``; :func:`small_token_split` keeps that
    distinction for the dispatch summary.
    """
    if num_tokens <= 0:
        return "0"
    high_bit = num_tokens.bit_length() - 1
    low = 1 << high_bit
    return f"[{low},{(low << 1) - 1}]"


def small_token_split(num_tokens: int) -> str:
    """``"le64"`` / ``"gt64"`` / ``"0"``: SGLang's small-M dispatch boundary.

    SGLang routes ``x.shape[0] <= 64`` to ``mhc_post_split_h`` (Triton) and
    ``65..240`` to ``mhc_post_split_h_tilelang``; vLLM has no counterpart to the
    former. This label makes that boundary visible in the histogram.
    """
    if num_tokens <= 0:
        return "0"
    return "le64" if num_tokens <= 64 else "gt64"


class _DispatchStats:
    """Thread-safe counters keyed by ``(path, outcome)`` and token bucket."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: Counter[tuple[str, str]] = Counter()
        self._tokens: Counter[tuple[str, str, str]] = Counter()
        self._small: Counter[tuple[str, str, str]] = Counter()
        self._total = 0
        self._logged = 0

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
            self._tokens.clear()
            self._small.clear()
            self._total = 0
            self._logged = 0

    def record(self, path: str, outcome: str, num_tokens: int) -> None:
        due = False
        with self._lock:
            self._calls[(path, outcome)] += 1
            self._tokens[(path, outcome, num_tokens_bucket(num_tokens))] += 1
            self._small[(path, outcome, small_token_split(num_tokens))] += 1
            self._total += 1
            if LOG_INTERVAL > 0 and self._total - self._logged >= LOG_INTERVAL:
                self._logged = self._total
                due = True
        # Log outside the lock; logging is host-only and never syncs the device.
        if due:
            logger.info("%s", format_summary())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            calls = {
                f"{path}/{outcome}": count
                for (path, outcome), count in sorted(self._calls.items())
            }
            tokens: dict[str, dict[str, int]] = {}
            for (path, outcome, bucket), count in sorted(self._tokens.items()):
                tokens.setdefault(f"{path}/{outcome}", {})[bucket] = count
            small: dict[str, dict[str, int]] = {}
            for (path, outcome, split), count in sorted(self._small.items()):
                small.setdefault(f"{path}/{outcome}", {})[split] = count
            return {
                "enabled": ENABLED,
                "total_calls": self._total,
                "calls": calls,
                "num_tokens_buckets": tokens,
                "small_token_split": small,
            }


_STATS = _DispatchStats()


def record(path: str, outcome: str, num_tokens: int) -> None:
    """Record one dispatch decision. No-op unless stats are enabled.

    ``num_tokens`` must already be a Python int; nothing here touches a tensor.
    """
    if not ENABLED:
        return
    _STATS.record(path, outcome, num_tokens)


def snapshot() -> dict[str, Any]:
    """Return the current counters as a JSON-serializable dict."""
    return _STATS.snapshot()


def reset() -> None:
    """Clear all counters (used by tests and harnesses)."""
    _STATS.reset()


def _format_nested(mapping: dict[str, dict[str, int]]) -> str:
    return ", ".join(
        f"{key}{{{', '.join(f'{bucket}={count}' for bucket, count in inner.items())}}}"
        for key, inner in mapping.items()
    )


def format_summary(snapshot_dict: dict[str, Any] | None = None) -> str:
    """One-line human-readable summary of the counters."""
    snap = snapshot() if snapshot_dict is None else snapshot_dict
    calls = ", ".join(f"{key}={count}" for key, count in snap["calls"].items())
    return (
        f"mHC dispatch stats: total={snap['total_calls']} "
        f"calls[{calls or 'none'}] "
        f"buckets[{_format_nested(snap['num_tokens_buckets']) or 'none'}] "
        f"small_token_split[{_format_nested(snap['small_token_split']) or 'none'}]"
    )
