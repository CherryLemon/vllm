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
#   TP                  - tensor parallel size per instance (default 8)
#   MAX_MODEL_LEN       - (default 32768)
#   GPU_MEMORY_UTILIZATION - (default 0.88)
#   PREFILL_PORT        - (default 8200)
#   DECODE_PORT         - (default 8300)
#   PROXY_PORT          - (default 8192)
#   NIXL_SIDE_CHANNEL_HOST - address the NIXL side channel binds to
#   PREFILL_HOSTS / DECODE_HOSTS - comma-separated host list for the proxy
#   ATTENTION_CONFIG    - (default mxfp4 indexer + sparse logits)
#   PREFILL_SPEC_CONFIG / DECODE_SPEC_CONFIG - speculative configs
#   ENABLE_GRAPHS       - 1 to keep CUDA graphs on (default 0 = eager)
set -euo pipefail

ROLE="${ROLE:-all}"

MODEL_PATH="${MODEL_PATH:-/public-nvme/models/DeepSeek-V4.1-Flash}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4.1-flash}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
# Leave empty to let the backend pick its required attention block size.
BLOCK_SIZE="${BLOCK_SIZE:-}"

PREFILL_PORT="${PREFILL_PORT:-8200}"
DECODE_PORT="${DECODE_PORT:-8300}"
PROXY_PORT="${PROXY_PORT:-8192}"
PREFILL_SIDE_CHANNEL_PORT="${PREFILL_SIDE_CHANNEL_PORT:-5600}"
DECODE_SIDE_CHANNEL_PORT="${DECODE_SIDE_CHANNEL_PORT:-5601}"
NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"

PREFILL_HOSTS="${PREFILL_HOSTS:-127.0.0.1}"
DECODE_HOSTS="${DECODE_HOSTS:-127.0.0.1}"

ATTENTION_CONFIG="${ATTENTION_CONFIG:-{\"indexer_kv_dtype\":\"mxfp4\",\"indexer_sparse_logits\":true}}"
# DSpark block5: the decode instance drafts 5 tokens per round; the prefill
# instance only needs a legal (smaller) speculative config, matching the
# upstream PD+SD harness convention.
PREFILL_SPEC_CONFIG="${PREFILL_SPEC_CONFIG:-{\"method\":\"dspark\",\"num_speculative_tokens\":1}}"
DECODE_SPEC_CONFIG="${DECODE_SPEC_CONFIG:-{\"method\":\"dspark\",\"num_speculative_tokens\":5}}"

ENABLE_GRAPHS="${ENABLE_GRAPHS:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GIT_ROOT="${GIT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)}"

# Opt-in SM90 indexer/linear/mHC kernels. Default on for this harness: the
# point is to exercise the ported paths.
export VLLM_SM90_FP4_INDEXER="${VLLM_SM90_FP4_INDEXER:-1}"
export VLLM_SM90_FP8_BLOCK32_STATIC="${VLLM_SM90_FP8_BLOCK32_STATIC:-1}"
export VLLM_SM90_MHC_SPLIT_H="${VLLM_SM90_MHC_SPLIT_H:-1}"

GRAPHS_ARGS=()
if [[ "$ENABLE_GRAPHS" == "1" ]]; then
  # DSpark: target verifies 1 + dspark_block_size rows per request, draft
  # proposes dspark_block_size. Buckets below cover both series.
  BUCKETS="${BUCKETS:-6 12 18 24 30 36 48 60 72 90 120 162 216 288 384 480 576}"
  GRAPHS_ARGS=(--cudagraph-capture-sizes $BUCKETS --max-cudagraph-capture-size 576)
fi

log() { echo "[dsv41_dspark_pd] $*"; }

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

serve_args() {
  local role="$1"
  echo "--model ${MODEL_PATH} --served-model-name ${SERVED_NAME} \
--tensor-parallel-size ${TP} --enable-expert-parallel \
--max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
--max-num-seqs ${MAX_NUM_SEQS} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
--attention-config ${ATTENTION_CONFIG}"
}

run_prefill() {
  log "starting prefill instance on port ${PREFILL_PORT}"
  VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_SIDE_CHANNEL_HOST" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_SIDE_CHANNEL_PORT" \
  python3 -m vllm.entrypoints.openai.api_server \
    $(serve_args prefill) \
    ${GRAPHS_ARGS[@]+"${GRAPHS_ARGS[@]}"} \
    ${BLOCK_SIZE:+--block-size "$BLOCK_SIZE"} \
    --port "$PREFILL_PORT" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    --speculative-config "$PREFILL_SPEC_CONFIG"
}

run_decode() {
  log "starting decode instance on port ${DECODE_PORT}"
  VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_SIDE_CHANNEL_HOST" \
  VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_SIDE_CHANNEL_PORT" \
  python3 -m vllm.entrypoints.openai.api_server \
    $(serve_args decode) \
    ${GRAPHS_ARGS[@]+"${GRAPHS_ARGS[@]}"} \
    ${BLOCK_SIZE:+--block-size "$BLOCK_SIZE"} \
    --port "$DECODE_PORT" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' \
    --speculative-config "$DECODE_SPEC_CONFIG"
}

run_proxy() {
  local p_hosts=() d_hosts=()
  IFS=',' read -r -a p_hosts <<< "$PREFILL_HOSTS"
  IFS=',' read -r -a d_hosts <<< "$DECODE_HOSTS"
  log "starting toy proxy on port ${PROXY_PORT}"
  python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    --port "$PROXY_PORT" \
    --prefiller-hosts "${p_hosts[@]}" \
    --prefiller-ports "$PREFILL_PORT" \
    --decoder-hosts "${d_hosts[@]}" \
    --decoder-ports "$DECODE_PORT"
}

run_test() {
  local p_host="${PREFILL_HOSTS%%,*}"
  log "running DSpark PD acceptance test through the proxy"
  SERVER_HOST="$SERVER_HOST" \
  PROXY_PORT="$PROXY_PORT" \
  DECODE_PORT="$DECODE_PORT" \
  TEST_MODEL="$SERVED_NAME" \
  DSV41_DSPARK_REFERENCE="${DSV41_DSPARK_REFERENCE:-}" \
  "${PYTEST:-python3 -m pytest} -s -x ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/test_dsv41_dspark_pd.py"
  : "$p_host"
}

case "$ROLE" in
  prefill) run_prefill ;;
  decode)  run_decode ;;
  proxy)   run_proxy ;;
  test)    run_test ;;
  all)
    run_prefill & PREFILL_PID=$!
    run_decode  & DECODE_PID=$!
    wait_for_http "http://${SERVER_HOST}:${PREFILL_PORT}/health" "prefill"
    wait_for_http "http://${SERVER_HOST}:${DECODE_PORT}/health" "decode"
    run_proxy & PROXY_PID=$!
    wait_for_http "http://${SERVER_HOST}:${PROXY_PORT}/healthcheck" "proxy" 120
    run_test
    kill "$PREFILL_PID" "$DECODE_PID" "$PROXY_PID" 2>/dev/null || true
    ;;
  *) echo "unknown ROLE=$ROLE (prefill|decode|proxy|test|all)" >&2; exit 2 ;;
esac