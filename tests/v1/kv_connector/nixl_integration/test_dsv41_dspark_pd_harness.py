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
    # Connector selection must come from the test, not from the caller's shell:
    # a leaked KV_CONNECTOR would silently turn the "default" tests into
    # Mooncake tests.
    for leaked in (
        "KV_CONNECTOR",
        "WITH_NVIDIA_PEERMEM",
        "MOONCAKE_ABORT_REQUEST_TIMEOUT",
        "PREFILL_MOONCAKE_BOOTSTRAP_PORT",
        "DECODE_MOONCAKE_BOOTSTRAP_PORT",
    ):
        env.pop(leaked, None)
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


def test_connector_defaults_to_nixl_and_gets_the_side_channel_env():
    """NIXL is the default; its two env vars must still reach both commands."""
    cfg = harness_config(NIXL_SIDE_CHANNEL_HOST="10.8.2.13")
    assert cfg["KV_CONNECTOR"] == "NixlConnector"
    p_args = args_of(cfg, "PREFILL_CMD")
    d_args = args_of(cfg, "DECODE_CMD")
    assert "VLLM_NIXL_SIDE_CHANNEL_HOST=10.8.2.13" in p_args
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT=5600" in p_args
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT=5610" in d_args
    assert json.loads(p_args[p_args.index("--kv-transfer-config") + 1]) == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
    }
    assert (
        json.loads(d_args[d_args.index("--kv-transfer-config") + 1])["kv_role"]
        == "kv_consumer"
    )
    # Mooncake-only knobs must not leak into a NIXL launch.
    assert not any(a.startswith("VLLM_MOONCAKE") for a in p_args + d_args)


def test_mooncake_connector_swaps_the_json_and_the_env():
    """The whole point of the switch: same harness, different connector."""
    cfg = harness_config(
        KV_CONNECTOR="MooncakeConnector",
        WITH_NVIDIA_PEERMEM="0",
        MOONCAKE_ABORT_REQUEST_TIMEOUT="60",
        NIXL_SIDE_CHANNEL_HOST="10.8.2.13",
    )
    p_args = args_of(cfg, "PREFILL_CMD")
    d_args = args_of(cfg, "DECODE_CMD")
    assert json.loads(p_args[p_args.index("--kv-transfer-config") + 1]) == {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
    }
    assert (
        json.loads(d_args[d_args.index("--kv-transfer-config") + 1])["kv_role"]
        == "kv_consumer"
    )
    # One bootstrap server per instance: a shared port would collide when the
    # two instances run on one host (ROLE=all).
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT=8998" in p_args
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT=8999" in d_args
    assert "WITH_NVIDIA_PEERMEM=0" in p_args and "WITH_NVIDIA_PEERMEM=0" in d_args
    assert "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=60" in p_args
    assert "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=60" in d_args
    assert not any(a.startswith("VLLM_NIXL") for a in p_args + d_args)


def test_optional_connector_knobs_are_omitted_when_unset():
    """Passing an empty value would override the library default with ""."""
    cfg = harness_config(KV_CONNECTOR="MooncakeConnector")
    for key in ("PREFILL_CMD", "DECODE_CMD"):
        args = args_of(cfg, key)
        assert not any(a.startswith("WITH_NVIDIA_PEERMEM") for a in args)
        assert not any(
            a.startswith("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT") for a in args
        )


def test_proxy_is_told_which_connector_the_instances_use():
    """Mooncake's router bookkeeping depends on it; NIXL must stay the default."""
    cfg = harness_config(
        KV_CONNECTOR="MooncakeConnector", PREFILL_MOONCAKE_BOOTSTRAP_PORT="9100"
    )
    args = args_of(cfg, "PROXY_CMD")
    assert args[args.index("--kv-connector") + 1] == "MooncakeConnector"
    assert args[args.index("--prefiller-bootstrap-port") + 1] == "9100"

    args = args_of(harness_config(), "PROXY_CMD")
    assert args[args.index("--kv-connector") + 1] == "NixlConnector"
    # The default bootstrap port is the library default, so a Mooncake run that
    # never set it still tells the proxy the truth.
    assert args[args.index("--prefiller-bootstrap-port") + 1] == "8998"


def test_unknown_connector_fails_instead_of_starting_a_mismatched_pair():
    env = dict(os.environ)
    env["ROLE"] = "print-config"
    env["KV_CONNECTOR"] = "NotAConnector"
    proc = subprocess.run(
        ["bash", str(HARNESS)], env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode != 0, proc.stdout
    assert "unknown KV_CONNECTOR" in proc.stderr


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


def test_services_are_killed_by_process_group_not_wrapper_shell():
    """The recorded PID must be the service itself, in its own process group.

    Regression: ``run_prefill & STARTED_PIDS+=($!)`` recorded a *function*
    running in a subshell; killing that shell left the Python service (and the
    TP workers it spawns) alive after the suite failed.  Now each service is
    started under ``setsid`` by a shell that writes ``$$`` and then ``exec``s
    Python, so the recorded PID is the service *and* its process-group id.
    """
    code = "\n".join(_code_lines(HARNESS.read_text()))
    assert "setsid bash -c" in code
    # The PID file is written by the shell that execs into the service.
    assert 'printf "%s" "$$"' in code
    assert "exec \"$@\"" in code
    # Group kill (negative PID), TERM before KILL, with a grace period.
    assert 'kill -"$sig" -- "-${pid}"' in code
    term_lines = [
        line for line in code.splitlines() if 'kill_service "$pidfile" TERM' in line
    ]
    kill_lines = [
        line for line in code.splitlines() if 'kill_service "$pidfile" KILL' in line
    ]
    assert term_lines and kill_lines
    term_idx = code.index('kill_service "$pidfile" TERM')
    kill_idx = code.index('kill_service "$pidfile" KILL')
    assert term_idx < kill_idx, "SIGKILL must not be the only signal sent"
    assert "sleep 1" in code[term_idx:kill_idx], "no grace period between TERM and KILL"
    # The old single-PID bookkeeping must be gone.
    assert "STARTED_PIDS" not in code


def test_common_args_are_untouched_by_the_process_work():
    """The service argv still carries the JSON flags as single entries."""
    cfg = harness_config()
    args = args_of(cfg, "PREFILL_CMD")
    assert args[args.index("--attention-config") + 1] == cfg["ATTENTION_CONFIG"]
    spec_idx = args.index("--speculative-config") + 1
    assert json.loads(args[spec_idx])["method"] == "dspark"
    kv_idx = args.index("--kv-transfer-config") + 1
    assert json.loads(args[kv_idx])["kv_role"] == "kv_producer"

    cfg = harness_config()
    args = args_of(cfg, "DECODE_CMD")
    assert json.loads(args[args.index("--kv-transfer-config") + 1])["kv_role"] == (
        "kv_consumer"
    )
    assert json.loads(args[args.index("--speculative-config") + 1])[
        "num_speculative_tokens"
    ] == 5


def test_dev_mode_is_on_so_the_test_can_reset_caches():
    """The acceptance test's independent local control needs /reset_prefix_cache."""
    cfg = harness_config()
    assert cfg["VLLM_SERVER_DEV_MODE"] == "1"
    cfg = harness_config(VLLM_SERVER_DEV_MODE="0")
    assert cfg["VLLM_SERVER_DEV_MODE"] == "0"

def test_default_parallelism_is_homogeneous_tp8():
    """The verified baseline: P and D both TP8, no DP, disjoint NIXL ports."""
    cfg = harness_config()
    assert cfg["PREFILL_TP"] == cfg["DECODE_TP"] == "8"
    assert cfg["PREFILL_DP"] == cfg["DECODE_DP"] == "1"
    for key in ("PREFILL_CMD", "DECODE_CMD"):
        args = args_of(cfg, key)
        assert args[args.index("--tensor-parallel-size") + 1] == "8"
        assert "--data-parallel-size" not in args
    # DP ranks bind base_port + dp_rank, so the ranges must not overlap.
    p_base = int(cfg["PREFILL_SIDE_CHANNEL_PORT"])
    d_base = int(cfg["DECODE_SIDE_CHANNEL_PORT"])
    assert d_base >= p_base + int(cfg["PREFILL_DP"])


def test_heterogeneous_pd_splits_tp_and_adds_decode_dp():
    """The documented target: P TP8/EP8, D attention TP2 x DP4/EP8.

    Regression: the harness had a single `TP` for both instances, so the
    heterogeneous topology could not even be expressed.
    """
    cfg = harness_config(DECODE_TP="2", DECODE_DP="4")
    prefill = args_of(cfg, "PREFILL_CMD")
    decode = args_of(cfg, "DECODE_CMD")

    assert cfg["PREFILL_TP"] == "8" and cfg["PREFILL_DP"] == "1"
    assert prefill[prefill.index("--tensor-parallel-size") + 1] == "8"
    assert "--data-parallel-size" not in prefill

    assert cfg["DECODE_TP"] == "2" and cfg["DECODE_DP"] == "4"
    assert decode[decode.index("--tensor-parallel-size") + 1] == "2"
    assert decode[decode.index("--data-parallel-size") + 1] == "4"

    # EP spans TP * DP ranks on each side, so both stay EP8.
    assert "--enable-expert-parallel" in prefill
    assert "--enable-expert-parallel" in decode
    # Roles are unchanged by the parallel split.
    assert json.loads(prefill[prefill.index("--kv-transfer-config") + 1]) == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_producer",
    }
    assert json.loads(decode[decode.index("--kv-transfer-config") + 1]) == {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_consumer",
    }
    # DSpark stays on both sides (a compatibility-hash requirement), and only
    # the draft depth may differ.
    assert json.loads(prefill[prefill.index("--speculative-config") + 1]) == {
        "method": "dspark",
        "num_speculative_tokens": 1,
    }
    assert json.loads(decode[decode.index("--speculative-config") + 1]) == {
        "method": "dspark",
        "num_speculative_tokens": 5,
    }


def test_parallel_sizes_must_fit_the_node():
    """TP x DP is a GPU count, and the harness has eight."""
    # print-config builds both commands, so an over-subscribed decode must make
    # the script fail loudly instead of launching a partial instance.
    proc = subprocess.run(
        ["bash", str(HARNESS)],
        env={
            **os.environ,
            "ROLE": "print-config",
            "DECODE_TP": "4",
            "DECODE_DP": "4",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "FAIL: TP" in combined and "this harness has 8" in combined, combined
