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
#   MAX_MODEL_LEN       - (default 32768).  This is the *total* context, input
#                         plus output: a 131072-token prompt that may generate
#                         8192 tokens needs at least 139264.
#   GPU_MEMORY_UTILIZATION - (default 0.88)
#   MAX_NUM_SEQS / MAX_NUM_BATCHED_TOKENS - (defaults 16 / 8192) both roles
#   PREFILL_MAX_NUM_SEQS / DECODE_MAX_NUM_SEQS - per-role scheduler limits
#                         (default: MAX_NUM_SEQS).  They are *per engine*: with
#                         decode-side DP the client concurrency has to be
#                         divided by DECODE_DP before comparing.
#   PREFILL_MAX_NUM_BATCHED_TOKENS / DECODE_MAX_NUM_BATCHED_TOKENS - same, for
#                         the token budget
#   PREFILL_PREFIX_CACHING / DECODE_PREFIX_CACHING - 1/0 per role (both 1).
#                         The reference deployment keeps the prefiller's cache
#                         on and the decoder's off.
#   PREFILL_PORT        - (default 8200)
#   DECODE_PORT         - (default 8300)
#   PROXY_PORT          - (default 8192)
#   NIXL_SIDE_CHANNEL_HOST - address the NIXL side channel binds to
#   ROUTER              - toy (default) or dsv41.  `toy` is the test
#                         scaffolding proxy in this directory; `dsv41` is the
#                         port's own router
#                         (examples/disaggregated/disaggregated_serving/
#                         dsv41_pd_router.py), which is connector-aware,
#                         genuinely concurrent (the toy proxy serialises per
#                         request, so concurrency numbers taken through it
#                         measure the proxy) and can route by prompt affinity.
#                         Both are interchangeable for this harness: it only
#                         needs /v1/completions and /healthcheck.
#   ROUTING             - prefix-affinity (default) or round-robin; ROUTER=dsv41
#                         only.
#   ROUTER_TIMEOUT_S    - (default 600) per-leg HTTP timeout of the dsv41
#                         router.  Long-context legs need a stated value rather
#                         than an implicit one.
#   KV_CONNECTOR        - NixlConnector (default) or MooncakeConnector.  Both
#                         implement the same kv_role/do_remote_* protocol and
#                         carry KV over their own side channel, so the toy proxy
#                         is unchanged: it only forwards kv_transfer_params.
#                         MooncakeConnector additionally starts a bootstrap HTTP
#                         server per instance, so the two roles use different
#                         ports (8998 / 8999) to stay safe when co-located.
#   WITH_NVIDIA_PEERMEM - forwarded to the connector when set ("" = library
#                         default).  Set 0 on a host without the nvidia-peermem
#                         module so Mooncake uses its DMA-BUF path instead of
#                         failing to register memory.
#   MOONCAKE_ABORT_REQUEST_TIMEOUT - forwarded when set; how long the producer
#                         keeps blocks for an unsent transfer before freeing
#                         them (connector default 480 s)
#   MOONCAKE_DEVICE_NAME - comma-separated RDMA devices the Mooncake transfer
#                         engine may use (kv_connector_extra_config.device_name).
#                         Needed on a multi-rail host: with the default (empty)
#                         each side's topology discovery picks its own HCA set --
#                         measured here, one side landed on the management bond
#                         and the other on a 100.75.4.x rail -- and every RDMA
#                         write then fails with "transport retry counter
#                         exceeded".  The engine's own MC_FORCE_HCA is *not* an
#                         alternative: it takes an integer, not a device name.
#   PREFILL_HOSTS / DECODE_HOSTS - comma-separated host list for the proxy
#   PROXY_HOST          - host the proxy listens on (default 127.0.0.1)
#   DECODE_HOST         - host the *test* reads decode /metrics from
#                         (defaults to the first entry of DECODE_HOSTS).
#                         Set it explicitly when the proxy runs on a third
#                         machine: decode /metrics is not on the proxy.
#   ATTENTION_CONFIG    - (default mxfp4 indexer + sparse logits)
#   PREFILL_SPEC_CONFIG / DECODE_SPEC_CONFIG - speculative configs; the literal
#                         value `none` (or empty) omits --speculative-config
#                         entirely, which is the only legal way to turn DSpark
#                         off (num_speculative_tokens must be > 0).  Turning it
#                         off changes the KV layout, so a PD ablation must do it
#                         on both sides and re-check the graph shapes.
#   PREFILL_PROFILER_CONFIG / DECODE_PROFILER_CONFIG - JSON --profiler-config
#                         for one role (empty = off).  Only the API server of the
#                         profiled role exposes /start_profile and /stop_profile.
#   VLLM_SM90_FP4_INDEXER / VLLM_SM90_FP8_BLOCK32_STATIC / VLLM_SM90_MHC_SPLIT_H
#                       - (default 1) the ported SM90 kernels; exported into the
#                         service environment, not just the caller's shell.
#   VLLM_SM90_FP4_GROUP6 / VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES
#                       - (default 0 = library default) the two kernels that are
#                         only entered on specific shapes; an A/B has to set
#                         them here so the worker's own environment shows it.
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
# The context limit covers input *and* output, so a 131072-token prompt needs
# at least 131072 + the output budget.  Anything smaller silently truncates the
# request (the API server rejects it, or worse, a shorter prompt is measured).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
# Scheduling limits are per *engine* and per role: with decode-side DP every
# rank has its own scheduler, so a global client concurrency of N needs
# N / DECODE_DP slots on each engine (plus admission headroom, or the router
# just queues).  The bare variables stay as both roles' default.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PREFILL_MAX_NUM_SEQS="${PREFILL_MAX_NUM_SEQS:-$MAX_NUM_SEQS}"
DECODE_MAX_NUM_SEQS="${DECODE_MAX_NUM_SEQS:-$MAX_NUM_SEQS}"
PREFILL_MAX_NUM_BATCHED_TOKENS="${PREFILL_MAX_NUM_BATCHED_TOKENS:-$MAX_NUM_BATCHED_TOKENS}"
DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS:-$MAX_NUM_BATCHED_TOKENS}"
# Prefix caching per role.  The deployment this harness is measured against
# keeps it on for the prefiller (one shared prefix, reused across requests) and
# off for the decoder (every request gets its own KV), so the two roles must be
# independently settable.  1 = on, 0 = off; any other value is a config error.
PREFILL_PREFIX_CACHING="${PREFILL_PREFIX_CACHING:-1}"
DECODE_PREFIX_CACHING="${DECODE_PREFIX_CACHING:-1}"
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
# KV connector under test.  The harness is otherwise connector-agnostic: roles,
# speculative config and the proxy protocol are identical.
KV_CONNECTOR="${KV_CONNECTOR:-NixlConnector}"
# Which proxy/router sits in front of the pair (see the header).
ROUTER="${ROUTER:-toy}"
ROUTING="${ROUTING:-prefix-affinity}"
# Per-leg HTTP timeout of the dsv41 router.  A 131k-token prefill plus a long
# generation can legitimately take minutes, but the timeout has to be a stated
# number: a client-side failure that is really "the leg took longer than the
# default" must not be counted as a slow success.
ROUTER_TIMEOUT_S="${ROUTER_TIMEOUT_S:-600}"
# Mooncake's bootstrap HTTP server binds a port per *instance*; two instances on
# one host (ROLE=all) would collide on the library default 8998.
PREFILL_MOONCAKE_BOOTSTRAP_PORT="${PREFILL_MOONCAKE_BOOTSTRAP_PORT:-8998}"
DECODE_MOONCAKE_BOOTSTRAP_PORT="${DECODE_MOONCAKE_BOOTSTRAP_PORT:-8999}"
# Empty means "do not pass it": a host with nvidia-peermem loaded should keep
# the transfer engine's default (peermem on).
WITH_NVIDIA_PEERMEM="${WITH_NVIDIA_PEERMEM:-}"
MOONCAKE_ABORT_REQUEST_TIMEOUT="${MOONCAKE_ABORT_REQUEST_TIMEOUT:-}"
# RDMA rail(s) the Mooncake transfer engine may use; empty = engine discovery.
# Both instances must name the same rail, or RDMA writes fail across fabrics.
MOONCAKE_DEVICE_NAME="${MOONCAKE_DEVICE_NAME:-}"
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
#
# "none" (or an empty value) means *no* speculative decoding: the flag is
# omitted.  It cannot be expressed by a config value, because
# num_speculative_tokens must be > 0.  The draft config is also part of the KV
# layout (the DSpark layers have their own caches), so a PD ablation must turn
# it off on both sides and re-check graphs and cache compatibility.
PREFILL_SPEC_CONFIG="${PREFILL_SPEC_CONFIG:-$DEFAULT_PREFILL_SPEC_CONFIG}"
DECODE_SPEC_CONFIG="${DECODE_SPEC_CONFIG:-$DEFAULT_DECODE_SPEC_CONFIG}"

# Torch profiler, per role, empty = off.  A profile that is not tied to one
# instance and one concurrency tells nothing: the trace has to come from the
# batch shape under investigation, and it must be collected next to a
# profiler-free run of the same load.
PREFILL_PROFILER_CONFIG="${PREFILL_PROFILER_CONFIG:-}"
DECODE_PROFILER_CONFIG="${DECODE_PROFILER_CONFIG:-}"

ENABLE_GRAPHS="${ENABLE_GRAPHS:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

# Opt-in SM90 indexer/linear/mHC kernels. Default on for this harness: the
# point is to exercise the ported paths.
export VLLM_SM90_FP4_INDEXER="${VLLM_SM90_FP4_INDEXER:-1}"
export VLLM_SM90_FP8_BLOCK32_STATIC="${VLLM_SM90_FP8_BLOCK32_STATIC:-1}"
export VLLM_SM90_MHC_SPLIT_H="${VLLM_SM90_MHC_SPLIT_H:-1}"
# These two keep the library default (off) unless the caller opts in.  They are
# exported *explicitly* rather than left to the ambient shell: a flag that only
# exists in the calling shell never reaches the container, and an A/B that
# cannot prove which value the worker saw is not an A/B.  Both are echoed by
# ROLE=print-config.
export VLLM_SM90_FP4_GROUP6="${VLLM_SM90_FP4_GROUP6:-0}"
export VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES="${VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES:-0}"

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
# shell word splitting.  `common_args <role>`: the scheduling limits and the
# prefix-caching switch are per role, everything else is shared.
COMMON_ARGS=()
GRAPH_ARGS=()
SPEC_ARGS=()
common_args() {
  local role="$1" seqs batched prefix_caching
  case "$role" in
    prefill)
      seqs="$PREFILL_MAX_NUM_SEQS"
      batched="$PREFILL_MAX_NUM_BATCHED_TOKENS"
      prefix_caching="$PREFILL_PREFIX_CACHING"
      ;;
    decode)
      seqs="$DECODE_MAX_NUM_SEQS"
      batched="$DECODE_MAX_NUM_BATCHED_TOKENS"
      prefix_caching="$DECODE_PREFIX_CACHING"
      ;;
    *)
      log "FAIL: common_args needs a role (prefill|decode), got '${role}'"
      return 1
      ;;
  esac
  COMMON_ARGS=(
    --model "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --enable-expert-parallel
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$seqs"
    --max-num-batched-tokens "$batched"
    --attention-config "$ATTENTION_CONFIG"
  )
  [ -n "$BLOCK_SIZE" ] && COMMON_ARGS+=(--block-size "$BLOCK_SIZE")

  # Both directions are passed explicitly so the resolved value is in argv and
  # not implied by a default that upstream may change.
  case "$prefix_caching" in
    1) COMMON_ARGS+=(--enable-prefix-caching) ;;
    0) COMMON_ARGS+=(--no-enable-prefix-caching) ;;
    *)
      log "FAIL: prefix caching must be 1 or 0 for role ${role}, got '${prefix_caching}'"
      return 1
      ;;
  esac

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

# Speculative decoding is opt-*out* here: there is no legal JSON value that
# means "off" (num_speculative_tokens must be > 0), so "none" has to remove the
# flag instead of setting a zero.
spec_config_args() {
  local spec="$1"
  SPEC_ARGS=()
  case "$spec" in
    "" | none | None) return 0 ;;
  esac
  SPEC_ARGS=(--speculative-config "$spec")
}

# Profiling is off unless a config is given; the value is a JSON object
# ({"profiler":"torch","torch_profiler_dir":"/work/profiles"}).
PROFILER_ARGS=()
profiler_args() {
  local config="$1"
  PROFILER_ARGS=()
  [ -n "$config" ] || return 0
  PROFILER_ARGS=(--profiler-config "$config")
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

# The --kv-transfer-config JSON.  Built (not hard-coded) so the connector can be
# swapped without editing the file, and so connector extra config -- the
# Mooncake RDMA rail in particular -- survives as a single argument.
kv_transfer_config() {
  local role="$1" extra="{}"
  if [ -n "$MOONCAKE_DEVICE_NAME" ]; then
    extra="{\"device_name\":\"${MOONCAKE_DEVICE_NAME}\"}"
  fi
  printf '{"kv_connector":"%s","kv_role":"%s","kv_connector_extra_config":%s}' \
    "$KV_CONNECTOR" "$role" "$extra"
}

# Per-connector service environment.  The two connectors disagree on how the
# side channel is addressed: NIXL takes host + port, Mooncake only a bootstrap
# port (the host is resolved from the local address the peers can reach).
# CONNECTOR_ENV is filled by connector_env() and expanded into the `env` argv.
CONNECTOR_ENV=()
connector_env() {
  local role="$1"
  CONNECTOR_ENV=()
  case "$KV_CONNECTOR" in
    NixlConnector)
      CONNECTOR_ENV+=("VLLM_NIXL_SIDE_CHANNEL_HOST=${NIXL_SIDE_CHANNEL_HOST}")
      if [ "$role" = prefill ]; then
        CONNECTOR_ENV+=("VLLM_NIXL_SIDE_CHANNEL_PORT=${PREFILL_SIDE_CHANNEL_PORT}")
      else
        CONNECTOR_ENV+=("VLLM_NIXL_SIDE_CHANNEL_PORT=${DECODE_SIDE_CHANNEL_PORT}")
      fi
      ;;
    MooncakeConnector)
      if [ "$role" = prefill ]; then
        CONNECTOR_ENV+=(
          "VLLM_MOONCAKE_BOOTSTRAP_PORT=${PREFILL_MOONCAKE_BOOTSTRAP_PORT}"
        )
      else
        CONNECTOR_ENV+=(
          "VLLM_MOONCAKE_BOOTSTRAP_PORT=${DECODE_MOONCAKE_BOOTSTRAP_PORT}"
        )
      fi
      # Optional knobs: passing an empty value would override the library
      # default with "", so they are only forwarded when set.
      [ -n "$WITH_NVIDIA_PEERMEM" ] &&
        CONNECTOR_ENV+=("WITH_NVIDIA_PEERMEM=${WITH_NVIDIA_PEERMEM}")
      [ -n "$MOONCAKE_ABORT_REQUEST_TIMEOUT" ] &&
        CONNECTOR_ENV+=(
          "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=${MOONCAKE_ABORT_REQUEST_TIMEOUT}"
        )
      ;;
    *)
      # Caught here rather than at launch time: a typo must not start a server
      # that silently disagrees with the other side about the connector.
      log "FAIL: unknown KV_CONNECTOR=${KV_CONNECTOR} (NixlConnector|MooncakeConnector)"
      return 1
      ;;
  esac
  return 0
}

prefill_cmd() {
  common_args prefill
  parallel_args "$PREFILL_TP" "$PREFILL_DP"
  connector_env prefill
  spec_config_args "$PREFILL_SPEC_CONFIG"
  profiler_args "$PREFILL_PROFILER_CONFIG"
  CMD=(
    env
    "${CONNECTOR_ENV[@]}"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    "${COMMON_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}"
    "${PROFILER_ARGS[@]}"
    --port "$PREFILL_PORT"
    --kv-transfer-config "$(kv_transfer_config kv_producer)"
  )
  log "prefill instance on port ${PREFILL_PORT} (TP=$PREFILL_TP DP=$PREFILL_DP graphs=$ENABLE_GRAPHS connector=$KV_CONNECTOR prefix-caching=$PREFILL_PREFIX_CACHING max-num-seqs=$PREFILL_MAX_NUM_SEQS spec=${PREFILL_SPEC_CONFIG:-none})"
}

decode_cmd() {
  common_args decode
  parallel_args "$DECODE_TP" "$DECODE_DP"
  connector_env decode
  spec_config_args "$DECODE_SPEC_CONFIG"
  profiler_args "$DECODE_PROFILER_CONFIG"
  CMD=(
    env
    "${CONNECTOR_ENV[@]}"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    "${COMMON_ARGS[@]}" "${PARALLEL_ARGS[@]}" "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}"
    "${PROFILER_ARGS[@]}"
    --port "$DECODE_PORT"
    --kv-transfer-config "$(kv_transfer_config kv_consumer)"
  )
  log "decode instance on port ${DECODE_PORT} (TP=$DECODE_TP DP=$DECODE_DP graphs=$ENABLE_GRAPHS connector=$KV_CONNECTOR prefix-caching=$DECODE_PREFIX_CACHING max-num-seqs=$DECODE_MAX_NUM_SEQS spec=${DECODE_SPEC_CONFIG:-none})"
}

proxy_cmd() {
  local p_hosts=() d_hosts=() instance_args=() host
  IFS=',' read -r -a p_hosts <<< "$PREFILL_HOSTS"
  IFS=',' read -r -a d_hosts <<< "$DECODE_HOSTS"
  case "$ROUTER" in
    toy)
      CMD=(
        "$PYTHON_BIN" "${SCRIPT_DIR}/toy_proxy_server.py"
        --host "$PROXY_HOST"
        --port "$PROXY_PORT"
        --prefiller-hosts "${p_hosts[@]}"
        --prefiller-ports "$PREFILL_PORT"
        --decoder-hosts "${d_hosts[@]}"
        --decoder-ports "$DECODE_PORT"
        --kv-connector "$KV_CONNECTOR"
        --prefiller-bootstrap-port "$PREFILL_MOONCAKE_BOOTSTRAP_PORT"
      )
      log "toy proxy on ${PROXY_HOST}:${PROXY_PORT} (connector=$KV_CONNECTOR)"
      ;;
    dsv41)
      # The port's router takes full URLs and one flag per instance, which is
      # what makes multi-instance deployments (and therefore routing) possible.
      for host in "${p_hosts[@]}"; do
        instance_args+=(--prefill "http://${host}:${PREFILL_PORT}")
      done
      for host in "${d_hosts[@]}"; do
        instance_args+=(--decode "http://${host}:${DECODE_PORT}")
      done
      CMD=(
        "$PYTHON_BIN" "${GIT_ROOT}/examples/disaggregated/disaggregated_serving/dsv41_pd_router.py"
        --host "$PROXY_HOST"
        --port "$PROXY_PORT"
        "${instance_args[@]}"
        --kv-connector "$KV_CONNECTOR"
        --prefill-bootstrap-port "$PREFILL_MOONCAKE_BOOTSTRAP_PORT"
        --routing "$ROUTING"
        --request-timeout-s "$ROUTER_TIMEOUT_S"
      )
      log "dsv41 router on ${PROXY_HOST}:${PROXY_PORT} (connector=$KV_CONNECTOR routing=$ROUTING timeout=${ROUTER_TIMEOUT_S}s)"
      ;;
    *)
      # A typo must not silently start the other router: the two have very
      # different concurrency and routing behaviour.
      log "FAIL: unknown ROUTER=${ROUTER} (toy|dsv41)"
      return 1
      ;;
  esac
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
  DSV41_KV_CONNECTOR="$KV_CONNECTOR" \
  DSV41_DECODE_LOG="${DSV41_DECODE_LOG:-}" \
  DSV41_PREFILL_LOG="${DSV41_PREFILL_LOG:-}" \
  DSV41_ABORT_TIMEOUT_PROBE="${DSV41_ABORT_TIMEOUT_PROBE:-0}" \
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
    common_args prefill
    printf 'ATTENTION_CONFIG=%s\n' "$ATTENTION_CONFIG"
    printf 'PREFILL_SPEC_CONFIG=%s\n' "$PREFILL_SPEC_CONFIG"
    printf 'DECODE_SPEC_CONFIG=%s\n' "$DECODE_SPEC_CONFIG"
    printf 'DECODE_HOST=%s\n' "$DECODE_HOST"
    printf 'KV_CONNECTOR=%s\n' "$KV_CONNECTOR"
    printf 'ROUTER=%s\n' "$ROUTER"
    printf 'ROUTING=%s\n' "$ROUTING"
    printf 'ROUTER_TIMEOUT_S=%s\n' "$ROUTER_TIMEOUT_S"
    printf 'PREFILL_MOONCAKE_BOOTSTRAP_PORT=%s\n' "$PREFILL_MOONCAKE_BOOTSTRAP_PORT"
    printf 'DECODE_MOONCAKE_BOOTSTRAP_PORT=%s\n' "$DECODE_MOONCAKE_BOOTSTRAP_PORT"
    printf 'WITH_NVIDIA_PEERMEM=%s\n' "$WITH_NVIDIA_PEERMEM"
    printf 'MOONCAKE_ABORT_REQUEST_TIMEOUT=%s\n' "$MOONCAKE_ABORT_REQUEST_TIMEOUT"
    printf 'MOONCAKE_DEVICE_NAME=%s\n' "$MOONCAKE_DEVICE_NAME"
    printf 'VLLM_SERVER_DEV_MODE=%s\n' "$VLLM_SERVER_DEV_MODE"
    printf 'VLLM_SM90_FP4_INDEXER=%s\n' "$VLLM_SM90_FP4_INDEXER"
    printf 'VLLM_SM90_FP8_BLOCK32_STATIC=%s\n' "$VLLM_SM90_FP8_BLOCK32_STATIC"
    printf 'VLLM_SM90_MHC_SPLIT_H=%s\n' "$VLLM_SM90_MHC_SPLIT_H"
    printf 'VLLM_SM90_FP4_GROUP6=%s\n' "$VLLM_SM90_FP4_GROUP6"
    printf 'VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES=%s\n' \
      "$VLLM_SM90_FP4_INDEXER_SKIP_INVALID_TILES"
    printf 'MAX_MODEL_LEN=%s\n' "$MAX_MODEL_LEN"
    printf 'PREFILL_PREFIX_CACHING=%s\n' "$PREFILL_PREFIX_CACHING"
    printf 'DECODE_PREFIX_CACHING=%s\n' "$DECODE_PREFIX_CACHING"
    printf 'PREFILL_MAX_NUM_SEQS=%s\n' "$PREFILL_MAX_NUM_SEQS"
    printf 'DECODE_MAX_NUM_SEQS=%s\n' "$DECODE_MAX_NUM_SEQS"
    printf 'PREFILL_MAX_NUM_BATCHED_TOKENS=%s\n' "$PREFILL_MAX_NUM_BATCHED_TOKENS"
    printf 'DECODE_MAX_NUM_BATCHED_TOKENS=%s\n' "$DECODE_MAX_NUM_BATCHED_TOKENS"
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
    printf 'PREFILL_SPEC_ARGS_COUNT=%d\n' "${#SPEC_ARGS[@]}"
    printf 'PREFILL_PROFILER_CONFIG=%s\n' "$PREFILL_PROFILER_CONFIG"
    decode_cmd
    printf 'DECODE_CMD=%s\n' "$(printf '%s\x1f' "${CMD[@]}")"
    printf 'DECODE_SPEC_ARGS_COUNT=%d\n' "${#SPEC_ARGS[@]}"
    printf 'DECODE_PROFILER_CONFIG=%s\n' "$DECODE_PROFILER_CONFIG"
    # The proxy's connector flag is what makes the Mooncake router bookkeeping
    # (transfer_id, remote_engine_id, remote_bootstrap_addr) appear at all.
    proxy_cmd
    printf 'PROXY_CMD=%s\n' "$(printf '%s\x1f' "${CMD[@]}")"
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