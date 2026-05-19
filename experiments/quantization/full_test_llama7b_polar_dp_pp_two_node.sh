#!/usr/bin/env bash
set -euo pipefail

# Run the same script on both machines, changing NODE_RANK:
#   MASTER_ADDR=10.31.10.210 NODE_RANK=0 bash experiments/quantization/full_test_llama7b_polar_dp_pp_two_node.sh
#   MASTER_ADDR=10.31.10.210 NODE_RANK=1 bash experiments/quantization/full_test_llama7b_polar_dp_pp_two_node.sh
#
# tc is applied on the selected interface before each run. Set APPLY_TC=0 to
# skip traffic shaping, or SUDO= if the script is already running as root.

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-10.31.10.210}"
MASTER_PORT="${MASTER_PORT:-29500}"

IFACE="${IFACE:-${NCCL_SOCKET_IFNAME}}"
APPLY_TC="${APPLY_TC:-1}"
SUDO="${SUDO:-sudo}"
NETEM_DELAY="${NETEM_DELAY:-0ms}"

BATCH_SIZE="${BATCH_SIZE:-16}"
PP_SIZE="${PP_SIZE:-8}"
LR="${LR:-2e-5}"
MICRO_BATCHES="${MICRO_BATCHES:-16}"
MAX_STEPS="${MAX_STEPS:-50}"
SEQ_LENGTH="${SEQ_LENGTH:-256}"
TRAIN_MODE="${TRAIN_MODE:-baseline}"
BASELINE_MODE="${BASELINE_MODE:-manual}"
METHOD="${METHOD:-bitscom}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-experiments/quantization/full_test_logs/llama7b_polar_dp_pp_${RUN_ID}}"
mkdir -p "${LOG_ROOT}"

BANDWIDTHS=(
  100mbit
  200mbit
  300mbit
  400mbit
  500mbit
  600mbit
  700mbit
  800mbit
  900mbit
  1gbit
  2gbit
  3gbit
  5gbit
  10gbit
)

run_tc() {
  if [[ -n "${SUDO}" ]]; then
    "${SUDO}" tc "$@"
  else
    tc "$@"
  fi
}

clear_tc() {
  if [[ "${APPLY_TC}" == "1" ]]; then
    run_tc qdisc del dev "${IFACE}" root 2>/dev/null || true
  fi
}

set_bandwidth() {
  local rate="$1"
  if [[ "${APPLY_TC}" != "1" ]]; then
    echo "[tc] APPLY_TC=0, skip bandwidth limit for ${rate}"
    return
  fi

  clear_tc
  run_tc qdisc add dev "${IFACE}" root handle 1: htb default 10
  run_tc class add dev "${IFACE}" parent 1: classid 1:10 htb rate "${rate}" ceil "${rate}"
  run_tc qdisc add dev "${IFACE}" parent 1:10 handle 10: netem delay "${NETEM_DELAY}"
  echo "[tc] ${IFACE}: rate=${rate}, delay=${NETEM_DELAY}"
}

trap clear_tc EXIT

SUMMARY_CSV="${LOG_ROOT}/summary_node${NODE_RANK}.csv"
if [[ "${NODE_RANK}" == "0" ]]; then
  echo "bandwidth,elapsed_s,tokens_per_s,exit_code,log_file" > "${SUMMARY_CSV}"
fi

echo "[config] node_rank=${NODE_RANK}/${NNODES}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[config] iface=${IFACE}, apply_tc=${APPLY_TC}, log_root=${LOG_ROOT}"
echo "[config] batch_size=${BATCH_SIZE}, pp_size=${PP_SIZE}, micro_batches=${MICRO_BATCHES}, max_steps=${MAX_STEPS}, seq_length=${SEQ_LENGTH}"

for bandwidth in "${BANDWIDTHS[@]}"; do
  bandwidth_label="${bandwidth//[^[:alnum:]]/_}"
  log_file="${LOG_ROOT}/node${NODE_RANK}_${bandwidth_label}.log"

  echo
  echo "========== bandwidth=${bandwidth} =========="
  set_bandwidth "${bandwidth}"

  start_ts="$(date +%s)"
  set +e
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    experiments/quantization/run_llama7b_polar_dp_pp.py \
    --batch-size "${BATCH_SIZE}" \
    --pp-size "${PP_SIZE}" \
    --lr "${LR}" \
    --micro-batches "${MICRO_BATCHES}" \
    --max-steps "${MAX_STEPS}" \
    --seq-length "${SEQ_LENGTH}" \
    --train-mode "${TRAIN_MODE}" \
    --baseline-mode "${BASELINE_MODE}" \
    --method "${METHOD}" \
    2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  set -e
  end_ts="$(date +%s)"
  elapsed_s="$((end_ts - start_ts))"

  if [[ "${NODE_RANK}" == "0" ]]; then
    tokens_per_s="$(
      awk \
        -v steps="${MAX_STEPS}" \
        -v batch="${BATCH_SIZE}" \
        -v seq="${SEQ_LENGTH}" \
        -v nnodes="${NNODES}" \
        -v nproc="${NPROC_PER_NODE}" \
        -v pp="${PP_SIZE}" \
        -v elapsed="${elapsed_s}" \
        'BEGIN {
          dp_world = (nnodes * nproc) / pp;
          if (elapsed > 0) {
            printf "%.6f", (steps * batch * seq * dp_world) / elapsed;
          } else {
            printf "0.000000";
          }
        }'
    )"
    echo "${bandwidth},${elapsed_s},${tokens_per_s},${exit_code},${log_file}" >> "${SUMMARY_CSV}"
    echo "[summary] bandwidth=${bandwidth} elapsed=${elapsed_s}s throughput=${tokens_per_s} tokens/s exit=${exit_code}"
  fi

  if [[ "${exit_code}" != "0" ]]; then
    echo "[warn] torchrun failed for bandwidth=${bandwidth}; continuing to next bandwidth"
  fi

  sleep 5
done

clear_tc
echo "[done] logs: ${LOG_ROOT}"
