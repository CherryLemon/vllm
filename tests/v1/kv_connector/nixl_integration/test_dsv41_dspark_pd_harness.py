# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-test for the DSpark PD harness' flag resolution.

Runs ``dsv41_dspark_pd_sm90.sh`` in its ``ROLE=print-config`` mode, which
resolves the serve flags and exits without touching a GPU, a model or NIXL.
The bugs this pins down were all silent: they only appeared with a *non-default*
configuration or on failure, which is exactly what a plain "it started" smoke
check does not cover.

  * ``"${VAR:-{...}}"`` closes the parameter expansion at the first ``}``, so an
    override containing braces came back with a stray trailing ``}`` -- only
    the default (no override) stayed legal JSON.
  * ``ENABLE_GRAPHS=0`` documented eager but never passed ``--enforce-eager``,
    so vLLM picked its own capture sizes and captured anyway.
  * decode ``/metrics`` was read from the proxy host, which is wrong as soon as
    the proxy runs on a third machine.

These are local (CPU) checks; no server is started.
"""

import json
import os

import subprocess
from pathlib import Path

import pytest

HARNESS = Path(__file__).with_name("dsv41_dspark_pd_sm90.sh")
US = "\x1f"


def harness_config(**env_overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("ATTENTION_CONFIG", None)
    env.pop("DECODE_HOST", None)
    env.pop("ENABLE_GRAPHS", None)
    env.pop("BUCKETS", None)
    env.update(env_overrides)
    env["ROLE"] = "print-config"
    proc = subprocess.run(
        ["bash", str(HARNESS)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def args_of(cfg: dict[str, str], key: str) -> list[str]:
    raw = cfg.get(key, "")
    return [a for a in raw.split(US) if a != ""]


def test_default_configs_are_valid_json():
    cfg = harness_config()
    for key in ("ATTENTION_CONFIG", "PREFILL_SPEC_CONFIG", "DECODE_SPEC_CONFIG"):
        json.loads(cfg[key])  # raises on a stray brace
    assert json.loads(cfg["PREFILL_SPEC_CONFIG"])["method"] == "dspark"
    assert json.loads(cfg["DECODE_SPEC_CONFIG"])["num_speculative_tokens"] == 5


def test_json_override_is_not_mangled():
    """An override must come back byte-for-byte, braces and all."""
    override = '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":false}'
    cfg = harness_config(ATTENTION_CONFIG=override)
    assert cfg["ATTENTION_CONFIG"] == override
    assert json.loads(cfg["ATTENTION_CONFIG"])["indexer_sparse_logits"] is False

    spec = '{"method":"dspark","num_speculative_tokens":10}'
    cfg = harness_config(DECODE_SPEC_CONFIG=spec)
    assert cfg["DECODE_SPEC_CONFIG"] == spec


def test_graphs_off_means_eager():
    cfg = harness_config(ENABLE_GRAPHS="0")
    graph_args = args_of(cfg, "GRAPH_ARGS")
    assert "--enforce-eager" in graph_args
    assert "--cudagraph-capture-sizes" not in graph_args


def test_graphs_on_uses_the_bucket_list():
    cfg = harness_config(ENABLE_GRAPHS="1", BUCKETS="6 12 24")
    graph_args = args_of(cfg, "GRAPH_ARGS")
    assert "--enforce-eager" not in graph_args
    idx = graph_args.index("--cudagraph-capture-sizes")
    assert graph_args[idx + 1 : idx + 4] == ["6", "12", "24"]
    # The cap must follow the bucket list, not a hard-coded value.
    cap = graph_args[graph_args.index("--max-cudagraph-capture-size") + 1]
    assert cap == "24"


def test_serve_flags_are_individual_arguments():
    """JSON values and paths must survive as single argv entries."""
    cfg = harness_config()
    args = args_of(cfg, "COMMON_ARGS")
    assert int(cfg["COMMON_ARGS_COUNT"]) == len(args)
    assert args[args.index("--attention-config") + 1] == cfg["ATTENTION_CONFIG"]
    assert args[args.index("--model") + 1] == "/public-nvme/models/DeepSeek-V4.1-Flash"


def test_decode_metrics_host_follows_decode_hosts():
    """Decode /metrics is not on the proxy when they are different machines."""
    cfg = harness_config(PREFILL_HOSTS="10.8.2.13", DECODE_HOSTS="10.8.2.9")
    assert cfg["DECODE_HOST"] == "10.8.2.9"

    cfg = harness_config(
        PREFILL_HOSTS="10.8.2.13", DECODE_HOSTS="10.8.2.9", DECODE_HOST="10.8.2.77"
    )
    assert cfg["DECODE_HOST"] == "10.8.2.77"


def _code_lines(text: str) -> list[str]:
    """The shell script without comment lines (the bug is quoted in a comment)."""
    return [
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize("role", ["prefill", "decode", "proxy", "test"])
def test_test_role_is_not_a_single_quoted_word(role):
    """`run_test` must not quote the interpreter and its flags together.

    Regression: ``"${PYTEST:-python3 -m pytest} ..."`` made bash look for one
    executable whose name contained every argument (exit 127).
    """
    code = _code_lines(HARNESS.read_text())
    assert not any("${PYTEST" in line for line in code)
    # The interpreter is invoked as its own argv[0] followed by -m pytest.
    assert any('"$PYTHON_BIN" -m pytest' in line for line in code)
    assert any(line.strip().startswith(f"{role})") for line in code)


def test_cleanup_is_trapped_not_sequenced():
    """`set -e` would skip a trailing cleanup block after a failing test."""
    code = "\n".join(_code_lines(HARNESS.read_text()))
    assert "trap cleanup EXIT" in code
    # No cleanup call may sit after the role dispatch.
    assert "cleanup\n" not in code