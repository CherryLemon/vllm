#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# DeepSeek-V4.1 DSpark + NixlConnector P/D disaggregation harness (H100/SM90).
#
# V4.1 has no classic-MTP draft: the checkpoint ships DSpark stages under
# ``mtp.*``, so speculative decoding is ``method="dspark"`` (block size 5).
# This harness starts one prefill and one decode vLLM instance with
# NixlConnector and a toy proxy in front, then runs
# ``test_dsv41_dspark_pd.py`` against the proxy.
#
# The two instances need TP8 each (the FP8 checkpoint is ~476 GiB), so they
# normally live on two nodes.  Roles are therefore granular:
#
#   # on the prefill node
#   ROLE=prefill NIXL_SIDE_CHANNEL_HOST=<p_ip> bash dsv41_dspark_pd_sm90.sh
#   # on the decode node
#   ROLE=decode  NIXL_SIDE_CHANNEL_HOST=<d_ip> bash dsv41_dspark_pd_sm90.sh
#   # anywhere that can reach both
#   ROLE=proxy PREFILL_HOSTS=<p_ip> DECODE_HOSTS=<d_ip> bash dsv41_dspark_pd_sm90.sh
#   ROLE=test  PREFILL_HOSTS=<p_ip> DECODE_HOSTS=<d_ip> bash dsv41_dspark_pd_sm90.sh
#
# Default ROLE=all runs everything on one host, which only works when both
# instances fit (e.g. a smaller model); it is kept for parity and for
# single-node smoke checks.
#
# Environment variables:
#   MODEL_PATH          - checkpoint path (default /public-nvme/models/DeepSeek-V4.1-Flash)
#   SERVED_NAME         - served model name (default deepseek-v4.1-flash)
#   TP                  - tensor parallel size for BOTH instances (default 8)
#   PREFILL_TP / DECODE_TP - per-instance tensor parallel override (default TP)
#   PREFILL_DP / DECODE_DP - internal data-parallel size per instance (default 1).
#                         NixlConnector PD supports heterogeneous TP for MLA
#                         (the KV cache is replicated across TP workers, so there
#                         is no head splitting) and data parallel is universally
#                         supported, so e.g. the documented target topology is
#
#                           PREFILL_TP=8 PREFILL_DP=1 \
#                           DECODE_TP=2  DECODE_DP=4    # attention TP2 x DP4, EP8
#
#                         Both instances must use the same attention backend and
#                         KV cache dtype, and the same block size (this model is
#                         hybrid SWA + full attention, which requires HMA and so
#                         forbids heterogeneous block sizes).  DP ranks bind
#                         side-channel ports base_port + dp_rank, so keep the two
#                         port ranges disjoint (5600 for P, 5610 for D).
#   MAX_MODEL_LEN       - (default 32768)
#   GPU_MEMORY_UTILIZATION - (default 0.88)
#   PREFILL_PORT        - (default 8200)
#   DECODE_PORT         - (default 8300)
#   PROXY_PORT          - (default 8192)
#   NIXL_SIDE_CHANNEL_HOST - address the NIXL side channel binds to
#   PREFILL_HOSTS / DECODE_HOSTS - comma-separated host list for the proxy
#   PROXY_HOST          - host the proxy listens on (default 127.0.0.1)
#   DECODE_HOST         - host the *test* reads decode /metrics from
#                         (defaults to the first entry of DECODE_HOSTS).
#                         Set it explicitly when the proxy runs on a third
#                         machine: decode /metrics is not on the proxy.
#   ATTENTION_CONFIG    - (default mxfp4 indexer + sparse logits)
#   PREFILL_SPEC_CONFIG / DECODE_SPEC_CONFIG - speculative configs
#   ENABLE_GRAPHS       - 1 keeps CUDA graphs on (default 0 = --enforce-eager)
#   BUCKETS             - capture sizes used when ENABLE_GRAPHS=1
#   VLLM_SERVER_DEV_MODE - (default 1) exposes /reset_prefix_cache, which the
#                         acceptance test uses to keep the PD request and the
#                         local-prefill control cold (and therefore independent)
#   DSV41_STRICT_REFERENCE - 1 makes the reference comparison a hard gate
set -euo pipefail

ROLE="${ROLE:-all}"

MODEL_PATH="${MODEL_PATH:-/public-nvme/models/DeepSeek-V4.1-Flash}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4.1-flash}"
TP="${TP:-8}"
# Per-instance parallelism: heterogeneous PD needs P and D to differ, and
# decode-side DP is expressed as data_parallel_size (internal "mp" backend, so
# one API server fronts every rank).
PREFILL_TP="${PREFILL_TP:-$TP}"
DECODE_TP="${DECODE_TP:-$TP}"
PREFILL_DP="${PREFILL_DP:-1}"
DECODE_DP="${DECODE_DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Empty = let the attention backend pick the block size it requires.
BLOCK_SIZE="${BLOCK_SIZE:-}"

PREFILL_PORT="${PREFILL_PORT:-8200}"
DECODE_PORT="${DECODE_PORT:-8300}"
PROXY_PORT="${PROXY_PORT:-8192}"
# DP ranks bind base_port + dp_rank, so the two ranges must not overlap; the
# old decode default (5601) left no headroom for decode-side DP.
PREFILL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT:-5600}"
DECODE_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNEL_PORT:-5610}"
NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-127.0.0.1}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"

PREFILL_HOSTS="${PREFILL_HOSTS:-127.0.0.1}"
DECODE_HOSTS="${DECODE_HOSTS:-127.0.0.1}"
DECODE_HOST="${DECODE_HOST:-${DECODE_HOSTS%%,*}}"

# Defaults live in their own variables.  Writing `"${VAR:-{...}}"` directly
# closes the parameter expansion at the first `}`, so an override containing
# braces (`{"a":1}`) came back with a stray trailing `}`.
DEFAULT_ATTENTION_CONFIG='{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
DEFAULT_PREFILL_SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":1}'
DEFAULT_DECODE_SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":5}'

ATTENTION_CONFIG="${ATTENTION_CONFIG:-$DEFAULT_ATTENTION_CONFIG}"
# DSpark block5: the decode instance drafts 5 tokens per round; the prefill
# instance only needs a legal (smaller) speculative config, matching the
# upstream PD+SD harness convention.
PREFILL_SPEC_CONFIG="${PREFILL_SPEC_CONFIG:-$DEFAULT_PREFILL_SPEC_CONFIG}"
DECODE_SPEC_CONFIG="${DECODE_SPEC_CONFIG:-$DEFAULT_DECODE_SPEC_CONFIG}"

ENABLE_GRAPHS="${ENABLE_GRAPHS:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

# Opt-in SM90 indexer/linear/mHC kernels. Default on for this harness: the
# point is to exercise the ported paths.
export VLLM_SM90_FP4_INDEXER="${VLLM_SM90_FP4_INDEXER:-1}"
export VLLM_SM90_FP8_BLOCK32_STATIC="${VLLM_SM90_FP8_BLOCK32_STATIC:-1}"
export VLLM_SM90_MHC_SPLIT_H="${VLLM_SM90_MHC_SPLIT_H:-1}"

# The acceptance test resets the decode instance's prefix cache between the PD
# request and the local control (`POST /reset_prefix_cache?reset_external=true`)
# so the two cannot share a cache entry.  That endpoint lives behind the dev
# routers, which exist only when VLLM_SERVER_DEV_MODE=1.  This is a test
# harness, never a production deployment; the flag is documented in the header.
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Diagnostics go to stderr so stdout stays a pure `KEY=value` stream for the
# harness self-test (which parses one key per line).
log() { echo "[dsv41_dspark_pd] $*" >&2; }

# Process management.  Every service runs in its own session (``setsid``) and
# records its real PID -- the shell that ``setsid`` starts writes ``$$`` and
# then ``exec``s Python, so the recorded PID *is* the service process and also
# its process-group id.  Killing the group takes the TP worker processes with
# it; killing a backgrounded wrapper shell (the old behaviour) left Python and
# its workers alive.
PID_DIR="$(mktemp -d)"
SERVICE_PIDFILES=()

_svc_pidfile() { printf '%s/%s.pid' "$PID_DIR" "$1"; }

start_service() {
  local name="$1"
  shift
  local pidfile
  pidfile="$(_svc_pidfile "$name")"
  log "starting ${name} (pidfile ${pidfile})"
  setsid bash -c 'printf "%s" "$$" > "$1"; shift; exec "$@"' \
    _ "$pidfile" "$@" &
  SERVICE_PIDFILES+=("$pidfile")
}

_service_pid() {
  local pidfile="$1" pid
  [ -s "$pidfile" ] || return 1
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [ -n "${pid:-}" ] || return 1
  printf '%s' "$pid"
}

kill_service() {
  local pidfile="$1" sig="$2" pid
  pid="$(_service_pid "$pidfile")" || return 0
  # Negative PID = the whole process group (the service plus every worker it
  # spawned); fall back to the single process if the group is already gone.
  kill -"$sig" -- "-${pid}" 2>/dev/null || kill -"$sig" "$pid" 2>/dev/null || true
}

cleanup() {
  local pidfile pid still_alive
  # SIGTERM first and give the services a moment for an orderly shutdown
  # (NIXL/zmq teardown); only escalate to SIGKILL for what is still alive.
  for pidfile in "${SERVICE_PIDFILES[@]:-}"; do
    [ -n "$pidfile" ] && kill_service "$pidfile" TERM
  done
  local waited=0
  while (( waited < 15 )); do
    still_alive=0
    for pidfile in "${SERVICE_PIDFILES[@]:-}"; do
      [ -z "$pidfile" ] && continue
      if pid="$(_service_pid "$pidfile")" && kill -0 "$pid" 2>/dev/null; then
        still_alive=1
      fi
    done
    (( still_alive == 0 )) && break
    sleep 1
    waited=$((waited + 1))
  done
  for pidfile in "${SERVICE_PIDFILES[@]:-}"; do
    [ -n "$pidfile" ] && kill_service "$pidfile" KILL
  done
  rm -rf "$PID_DIR"
}
trap cleanup EXIT

# Block until the service has recorded its PID, then until it exits.  ``wait``
# cannot be used: the service is not a direct child of this shell.
wait_service() {
  local name="$1" pidfile pid i
  pidfile="$(_svc_pidfile "$name")"
  for ((i = 0; i < 240; i++)); do
    pid="$(_service_pid "$pidfile")" && break
    sleep 0.5
  done
  [ -n "${pid:-}" ] || { log "FAIL: ${name} never recorded a PID"; return 1; }
  log "${name} running as pid ${pid}"
  while kill -0 "$pid" 2>/dev/null; do sleep 2; done
  log "${name} exited"
}

wait_for_http() {
  local url="$1" name="$2" deadline="${3:-3600}" elapsed=0
  log "waiting for ${name} at ${url} ..."
  while (( elapsed < deadline )); do
    if curl -sf -o /dev/null "$url"; then
      log "${name} ready"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  log "FAIL: ${name} not ready within ${deadline}s"
  return 1
}

# Common serve flags, as an array so JSON values and paths with spaces survive
# shell word splitting.
COMMON_ARGS=()
GRAPH_ARGS=()
common_args() {
  COMMON_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --enable-expert-parallel
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --attention-config "$ATTENTION_CONFIG"
  )
  [ -n "$BLOCK_SIZE" ] && COMMON_ARGS+=(--block-size "$BLOCK_SIZE")

  if [ "$ENABLE_GRAPHS" = "1" ]; then
    # DSpark: target verifies 1 + dspark_block_size rows per request, draft
    # proposes dspark_block_size. Buckets below cover both series.
    BUCKETS="${BUCKETS:-6 12 18 24 30 36 48 60 72 90 120 162 216 288 384 480 576}"
    local max_bucket=0 b
    for b in $BUCKETS; do [ "$b" -gt "$max_bucket" ] && max_bucket=$b; done
    GRAPH_ARGS=(
      --cudagraph-capture-sizes $BUCKETS
      --max-cudagraph-capture-size "$max_bucket"
    )
  else
    # ENABLE_GRAPHS=0 must actually mean eager.  Only omitting the capture-size
    # flags left vLLM free to pick its own defaults and capture anyway.
    GRAPH_ARGS=(--enforce-eager)
  fi
}

# Each service command is built into the ``CMD`` array (one element per argv
# entry) instead of being a function that shells out; ``start_service`` runs it
# under ``setsid`` so the recorded PID is the service itself.  ``env`` carries
# the per-service NIXL channel variables without relying on inline
# ``VAR=value cmd`` parsing.
# `--tensor-parallel-size` / `--data-parallel-size` are per instance: a
# heterogeneous PD pair must not share them.  `--enable-expert-parallel` stays
# common, so EP spans TP * DP ranks on each side (P TP8/DP1 -> EP8,
# D TP2xDP4 -> EP8).
parallel_args() {
  local tp="$1" dp="$2" rank_gpus
  rank_gpus=$((tp * dp))
  if (( rank_gpus > 8 )); then
    log "FAIL: TP ${tp} x DP ${dp} = ${rank_gpus} GPUs, this harness has 8"
    return 1
  fi
  PARALLEL_ARGS=(--tensor-parallel-size "$tp")
  if (( dp > 1 )); then
    PARALLEL_ARGS+=(--data-parallel-size "$dp")
  fi
}

prefill_cmd() {
  common_args
  parallel_args "$PREFILL_TP" "$PREFILL_DP"
  CMD=(
    env
    "VLLM_NIXL_SIDE_CHANNEL_HOST=${NIXL_SIDE_CHANNEL_HOST}"
    "VLLM_NIXL_SIDE_CHANNEL_PORT=${PREFILL_SIDE_CHANNEL_PORT}"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    "${COMMON_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${GRAPH_ARGS[@]}"
    --port "$PREFILL_PORT"
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
    --speculative-config "$PREFILL_SPEC_CONFIG"
  )
  log "prefill instance on port ${PREFILL_PORT} (TP=$PREFILL_TP DP=$PREFILL_DP graphs=$ENABLE_GRAPHS)"
}

decode_cmd() {
  common_args
  parallel_args "$DECODE_TP" "$DECODE_DP"
  CMD=(
    env
    "VLLM_NIXL_SIDE_CHANNEL_HOST=${NIXL_SIDE_CHANNEL_HOST}"
    "VLLM_NIXL_SIDE_CHANNEL_PORT=${DECODE_SIDE_CHANNEL_PORT}"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    "${COMMON_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${GRAPH_ARGS[@]}"
    --port "$DECODE_PORT"
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
    --speculative-config "$DECODE_SPEC_CONFIG"
  )
  log "decode instance on port ${DECODE_PORT} (TP=$DECODE_TP DP=$DECODE_DP graphs=$ENABLE_GRAPHS)"
}

proxy_cmd() {
  local p_hosts=() d_hosts=()
  IFS=',' read -r -a p_hosts <<< "$PREFILL_HOSTS"
  IFS=',' read -r -a d_hosts <<< "$DECODE_HOSTS"
  CMD=(
    "$PYTHON_BIN" "${SCRIPT_DIR}/toy_proxy_server.py"
    --host "$PROXY_HOST"
    --port "$PROXY_PORT"
    --prefiller-hosts "${p_hosts[@]}"
    --prefiller-ports "$PREFILL_PORT"
    --decoder-hosts "${d_hosts[@]}"
    --decoder-ports "$DECODE_PORT"
  )
  log "toy proxy on ${PROXY_HOST}:${PROXY_PORT}"
}

run_test() {
  # Every one of these must be its own shell word.  Quoting the whole command
  # (`"${PYTEST:-python3 -m pytest} ..."`) made bash look for a single
  # executable whose name contained every argument, which exits 127.
  log "running DSpark PD acceptance test (proxy=${SERVER_HOST}:${PROXY_PORT}, decode metrics=${DECODE_HOST}:${DECODE_PORT})"
  SERVER_HOST="$SERVER_HOST" \
  PROXY_HOST="$PROXY_HOST" \
  PROXY_PORT="$PROXY_PORT" \
  DECODE_HOST="$DECODE_HOST" \
  DECODE_PORT="$DECODE_PORT" \
  TEST_MODEL="$SERVED_NAME" \
  DSV41_DSPARK_REFERENCE="${DSV41_DSPARK_REFERENCE:-}" \
  DSV41_STRICT_REFERENCE="${DSV41_STRICT_REFERENCE:-0}" \
  "$PYTHON_BIN" -m pytest -s -x "${SCRIPT_DIR}/test_dsv41_dspark_pd.py"
}

case "$ROLE" in
  prefill) prefill_cmd; start_service prefill "${CMD[@]}"; wait_service prefill ;;
  decode)  decode_cmd;  start_service decode  "${CMD[@]}"; wait_service decode ;;
  proxy)   proxy_cmd;   start_service proxy   "${CMD[@]}"; wait_service proxy ;;
  test)    run_test ;;
  # Machine-readable resolution of the serve flags, for the harness'
  # self-test: it exercises the JSON defaults/overrides and the eager/graph
  # branch without needing a GPU or a model.
  print-config)
    common_args
    printf 'ATTENTION_CONFIG=%s\n' "$ATTENTION_CONFIG"
    printf 'PREFILL_SPEC_CONFIG=%s\n' "$PREFILL_SPEC_CONFIG"
    printf 'DECODE_SPEC_CONFIG=%s\n' "$DECODE_SPEC_CONFIG"
    printf 'DECODE_HOST=%s\n' "$DECODE_HOST"
    printf 'VLLM_SERVER_DEV_MODE=%s\n' "$VLLM_SERVER_DEV_MODE"
    printf 'PREFILL_TP=%s\n' "$PREFILL_TP"
    printf 'DECODE_TP=%s\n' "$DECODE_TP"
    printf 'PREFILL_DP=%s\n' "$PREFILL_DP"
    printf 'DECODE_DP=%s\n' "$DECODE_DP"
    printf 'PREFILL_SIDE_CHANNEL_PORT=%s\n' "$PREFILL_SIDE_CHANNEL_PORT"
    printf 'DECODE_SIDE_CHANNEL_PORT=%s\n' "$DECODE_SIDE_CHANNEL_PORT"
    printf 'COMMON_ARGS_COUNT=%d\n' "${#COMMON_ARGS[@]}"
    printf 'COMMON_ARGS=%s\n' "$(printf '%s\x1f' "${COMMON_ARGS[@]}")"
    printf 'GRAPH_ARGS_COUNT=%d\n' "${#GRAPH_ARGS[@]}"
    printf 'GRAPH_ARGS=%s\n' "$(printf '%s\x1f' "${GRAPH_ARGS[@]}")"
    # Exercise the command builders too: the argv arrays are what the process
    # manager actually launches, and they must survive JSON values with braces.
    prefill_cmd
    printf 'PREFILL_CMD=%s\n' "$(printf '%s\x1f' "${CMD[@]}")"
    decode_cmd
    printf 'DECODE_CMD=%s\n' "$(printf '%s\x1f' "${CMD[@]}")"
    ;;
  all)
    prefill_cmd; start_service prefill "${CMD[@]}"
    decode_cmd;  start_service decode  "${CMD[@]}"
    wait_for_http "http://${SERVER_HOST}:${PREFILL_PORT}/health" "prefill"
    wait_for_http "http://${SERVER_HOST}:${DECODE_PORT}/health" "decode"
    proxy_cmd; start_service proxy "${CMD[@]}"
    wait_for_http "http://${PROXY_HOST}:${PROXY_PORT}/healthcheck" "proxy" 120
    run_test
    ;;
  *) echo "unknown ROLE=$ROLE (prefill|decode|proxy|test|all)" >&2; exit 2 ;;
esac