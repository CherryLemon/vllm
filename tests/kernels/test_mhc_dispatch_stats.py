# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the opt-in mHC dispatch counters and the split-H boundary.

These exercise the pure-Python counter logic and the dispatcher wiring without
launching any kernel; the TileLang kernel module only has to be importable.
The counters are host-only by construction (they record ``x.shape[0]`` and a
Python branch), which is what makes them safe under CUDA graph capture.
"""

from __future__ import annotations

import json

import pytest
import torch

from vllm.model_executor.kernels.mhc import dispatch_stats
from vllm.model_executor.kernels.mhc import tilelang as mhc_tilelang

try:  # TileLang may be absent on CPU-only test hosts.
    from vllm.model_executor.kernels.mhc import tilelang_kernels

    HAS_TILELANG = True
except Exception:  # pragma: no cover - depends on the test host
    tilelang_kernels = None  # type: ignore[assignment]
    HAS_TILELANG = False

SPLIT_H_HIDDEN = 5120


@pytest.fixture(autouse=True)
def _clean_stats():
    dispatch_stats.reset()
    yield
    dispatch_stats.reset()


# ---------------------------------------------------------------------------
# Bucket / boundary logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_tokens,expected",
    [
        (-1, "0"),
        (0, "0"),
        (1, "[1,1]"),
        (2, "[2,3]"),
        (3, "[2,3]"),
        (4, "[4,7]"),
        (7, "[4,7]"),
        (8, "[8,15]"),
        (63, "[32,63]"),
        (64, "[64,127]"),
        (65, "[64,127]"),
        (127, "[64,127]"),
        (128, "[128,255]"),
        (240, "[128,255]"),
        (241, "[128,255]"),
        (512, "[512,1023]"),
    ],
)
def test_num_tokens_bucket(num_tokens, expected):
    assert dispatch_stats.num_tokens_bucket(num_tokens) == expected


@pytest.mark.parametrize(
    "num_tokens,expected",
    [
        (0, "0"),
        (1, "le64"),
        (64, "le64"),
        (65, "gt64"),
        (240, "gt64"),
        (241, "gt64"),
    ],
)
def test_small_token_split(num_tokens, expected):
    """The 64/65 boundary is SGLang's small-M kernel switch, not vLLM's guard."""
    assert dispatch_stats.small_token_split(num_tokens) == expected


# ---------------------------------------------------------------------------
# Counter recording
# ---------------------------------------------------------------------------


def test_disabled_by_default_records_nothing(monkeypatch):
    monkeypatch.setattr(dispatch_stats, "ENABLED", False)
    dispatch_stats.record("post", "split_h", 1)
    assert dispatch_stats.snapshot()["total_calls"] == 0
    assert dispatch_stats.snapshot()["calls"] == {}


def test_record_counts_per_path_outcome_and_bucket(monkeypatch):
    monkeypatch.setattr(dispatch_stats, "ENABLED", True)
    dispatch_stats.record("post", "split_h", 1)
    dispatch_stats.record("post", "split_h", 64)
    dispatch_stats.record("post", "split_h", 65)
    dispatch_stats.record("post", "base", 241)
    dispatch_stats.record("fused_post_pre_delayed", "post_gemm", 2)

    snap = dispatch_stats.snapshot()
    assert snap["enabled"] is True
    assert snap["total_calls"] == 5
    assert snap["calls"] == {
        "fused_post_pre_delayed/post_gemm": 1,
        "post/base": 1,
        "post/split_h": 3,
    }
    assert snap["num_tokens_buckets"]["post/split_h"] == {
        "[1,1]": 1,
        "[64,127]": 2,
    }
    assert snap["num_tokens_buckets"]["post/base"] == {"[128,255]": 1}
    assert snap["small_token_split"]["post/split_h"] == {"le64": 2, "gt64": 1}
    assert snap["small_token_split"]["post/base"] == {"gt64": 1}
    json.dumps(snap)  # the snapshot is JSON-serializable


def test_reset_clears_counters(monkeypatch):
    monkeypatch.setattr(dispatch_stats, "ENABLED", True)
    dispatch_stats.record("post", "base", 3)
    dispatch_stats.reset()
    assert dispatch_stats.snapshot()["total_calls"] == 0


def test_periodic_summary_logs_at_interval(monkeypatch):
    monkeypatch.setattr(dispatch_stats, "ENABLED", True)
    monkeypatch.setattr(dispatch_stats, "LOG_INTERVAL", 2)
    logged: list[str] = []

    class _FakeLogger:
        def info(self, msg, *args):
            logged.append(msg % args if args else msg)

    monkeypatch.setattr(dispatch_stats, "logger", _FakeLogger())
    for i in range(5):
        dispatch_stats.record("post", "base", i + 1)

    assert len(logged) == 2  # after call 2 and after call 4
    assert all("mHC dispatch stats" in line for line in logged)
    assert "post/base=2" in logged[0]
    assert "post/base=4" in logged[1]


def test_format_summary_includes_all_sections(monkeypatch):
    monkeypatch.setattr(dispatch_stats, "ENABLED", True)
    dispatch_stats.record("post", "split_h", 65)
    text = dispatch_stats.format_summary(dispatch_stats.snapshot())
    assert "total=1" in text
    assert "post/split_h=1" in text
    assert "[64,127]=1" in text
    assert "gt64=1" in text


# ---------------------------------------------------------------------------
# Dispatcher integration: the guard decision and the counter agree
# ---------------------------------------------------------------------------


class _RecordingKernel:
    def __init__(self, name: str, calls: list[str]) -> None:
        self._name = name
        self._calls = calls

    def __call__(self, *args, **kwargs):
        self._calls.append(self._name)
        return None


def _split_h_tensors(num_tokens: int, hidden: int):
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    residual = torch.randn(num_tokens, 4, hidden, dtype=torch.bfloat16)
    post = torch.randn(num_tokens, 4, dtype=torch.float32)
    comb = torch.randn(num_tokens, 4, 4, dtype=torch.float32)
    return x, residual, post, comb


@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang kernel module unavailable")
def test_post_dispatch_counter_matches_guard(monkeypatch):
    """split-H served vs bypassed, and the M histogram, match the guard."""
    monkeypatch.setattr(dispatch_stats, "ENABLED", True)
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    calls: list[str] = []
    monkeypatch.setattr(
        tilelang_kernels, "_MHC_POST_TILELANG_KERNEL", _RecordingKernel("base", calls)
    )
    monkeypatch.setattr(
        tilelang_kernels,
        "_MHC_POST_SPLIT_H_TILELANG_KERNEL",
        _RecordingKernel("split", calls),
    )

    cases = [
        (1, SPLIT_H_HIDDEN, "split"),
        (64, SPLIT_H_HIDDEN, "split"),
        (65, SPLIT_H_HIDDEN, "split"),
        (240, SPLIT_H_HIDDEN, "split"),
        (241, SPLIT_H_HIDDEN, "base"),  # outer admission upper bound
        (2, 4096, "base"),  # hidden mismatch
    ]
    for num_tokens, hidden, expected in cases:
        x, residual, post, comb = _split_h_tensors(num_tokens, hidden)
        mhc_tilelang.mhc_post_tilelang(x, residual, post, comb)
        assert calls[-1] == expected

    snap = dispatch_stats.snapshot()
    assert snap["calls"] == {"post/base": 2, "post/split_h": 4}
    assert snap["small_token_split"]["post/split_h"] == {"le64": 2, "gt64": 2}
    assert snap["num_tokens_buckets"]["post/split_h"] == {
        "[1,1]": 1,
        "[64,127]": 2,
        "[128,255]": 1,
    }
    assert snap["num_tokens_buckets"]["post/base"] == {
        "[2,3]": 1,
        "[128,255]": 1,
    }


@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang kernel module unavailable")
def test_post_dispatch_not_counted_when_disabled(monkeypatch):
    monkeypatch.setattr(dispatch_stats, "ENABLED", False)
    monkeypatch.setattr(mhc_tilelang, "has_sm90_mhc_split_h", lambda: True)
    calls: list[str] = []
    monkeypatch.setattr(
        tilelang_kernels, "_MHC_POST_TILELANG_KERNEL", _RecordingKernel("base", calls)
    )
    monkeypatch.setattr(
        tilelang_kernels,
        "_MHC_POST_SPLIT_H_TILELANG_KERNEL",
        _RecordingKernel("split", calls),
    )
    x, residual, post, comb = _split_h_tensors(3, SPLIT_H_HIDDEN)
    mhc_tilelang.mhc_post_tilelang(x, residual, post, comb)
    assert calls == ["split"]
    assert dispatch_stats.snapshot()["total_calls"] == 0


@pytest.mark.skipif(not HAS_TILELANG, reason="TileLang kernel module unavailable")
def test_fused_delayed_empty_path_is_counted(monkeypatch):
    """M=0 returns before any kernel but is still visible in the census."""
    monkeypatch.setattr(dispatch_stats, "ENABLED", True)
    residual = torch.empty(0, 4, SPLIT_H_HIDDEN, dtype=torch.bfloat16)
    x = torch.empty(0, SPLIT_H_HIDDEN, dtype=torch.bfloat16)
    post = torch.empty(0, 4, dtype=torch.float32)
    comb = torch.empty(0, 4, 4, dtype=torch.float32)
    fn = torch.empty(24, 4 * SPLIT_H_HIDDEN, dtype=torch.float32)
    hc_scale = torch.empty(3, dtype=torch.float32)
    hc_base = torch.empty(24, dtype=torch.float32)

    mhc_tilelang.mhc_fused_post_pre_delayed_tilelang(
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        0.5,
        2,
    )
    snap = dispatch_stats.snapshot()
    assert snap["calls"] == {"fused_post_pre_delayed/empty": 1}
    assert snap["num_tokens_buckets"]["fused_post_pre_delayed/empty"] == {"0": 1}
