#!/usr/bin/env bash
set -euo pipefail

# Demonstration run for thesis/defense logs:
#   - shows PolarParallel is active at each training step,
#   - shows POLAR hook trigger/wait timing,
#   - shows bitscom quantization and low-bit all-reduce path.

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"

export POLAR_STEP_DEBUG="${POLAR_STEP_DEBUG:-1}"
export POLAR_HOOK_DEBUG="${POLAR_HOOK_DEBUG:-1}"
export BITSCOM_COMM_DEBUG="${BITSCOM_COMM_DEBUG:-1}"
export TRACE_EXPLAIN="${TRACE_EXPLAIN:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
TRACE_LOG_DIR="${TRACE_LOG_DIR:-outputs/trace_logs}"
TRACE_RUN_DIR="${TRACE_LOG_DIR}/polar_bitscom_trace_${RUN_ID}/node1"
mkdir -p "${TRACE_RUN_DIR}"
TRACE_README="${TRACE_RUN_DIR}/README_trace.txt"

cat > "${TRACE_README}" <<EOF
POLAR + bitscom Llama7B trace log
==================================

Purpose
-------
This short run is for demonstration/evidence. It should show, step by step:
1. Llama7B is trained with PolarParallel pipeline parallelism.
2. The POLAR backward hook is triggered before the full backward step finishes.
3. The predicted/early gradient buffer is handed to bitscom LowBitGroup.
4. bitscom performs low-bit communication: quantize/pack, collective communication,
   unpack/dequantize, and restore the reduced gradient buffer.

Run configuration
-----------------
node_rank=1
run_id=${RUN_ID}
nproc_per_node=${NPROC_PER_NODE:-8}
nnodes=${NNODES:-2}
master_addr=${MASTER_ADDR:-10.31.10.210}
master_port=${MASTER_PORT:-29500}
batch_size=${BATCH_SIZE:-16}
train_mode=polar
method=bitscom
polar_hook=${POLAR_HOOK:-io}
comm_timing=${COMM_TIMING:-4}
bitwidth=${BITWIDTH:-4}
pp_size=${PP_SIZE:-8}
micro_batches=${MICRO_BATCHES:-16}
max_steps=${MAX_STEPS:-3}
seq_length=${SEQ_LENGTH:-640}

Important markers
-----------------
[trace-evidence] component=PolarParallel
  PolarParallel is active and the rank has joined one pipeline stage.

[polar-step-debug ... schedule.step enter/exit]
  One training step has entered/exited the 1F1B pipeline schedule.

[trace-evidence] component=POLAR action=hook_trigger
  POLAR has triggered early DP communication at the configured microbatch.

[trace-evidence] component=POLAR action=handoff_to_bitscom_lowbit
  POLAR is no longer using dense DP all-reduce here; it passes the flat gradient
  buffer to bitscom LowBitGroup.

[trace-evidence] component=bitscom action=quantize_pack
  bitscom is quantizing and packing floating-point shards into low-bit payloads.

[trace-evidence] component=bitscom action=collective
  bitscom is doing the distributed collective over packed low-bit payloads.

Torchrun log layout
-------------------
This script uses torchrun --log-dir --redirects 3 --tee 3.
Each local rank gets its own stdout/stderr file under this directory, so GPU logs
are not mixed into one file.
EOF

echo "[trace-demo] node_rank=1 run_id=${RUN_ID}"
echo "[trace-demo] expect markers: [PolarParallel], [polar-step-debug], [polar-hook-debug], [bitscom-debug]"
echo "[trace-demo] per-rank logs under ${TRACE_RUN_DIR}"
echo "[trace-demo] human-readable guide: ${TRACE_README}"
echo "[trace-demo] torchrun writes rank-specific stdout/stderr files instead of one mixed log"

set +e
torchrun --log-dir "${TRACE_RUN_DIR}" \
    --redirects 3 \
    --tee 3 \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --nnodes="${NNODES:-2}" \
    --node_rank="${NODE_RANK:-1}" \
    --master_addr="${MASTER_ADDR:-10.31.10.210}" \
    --master_port="${MASTER_PORT:-29500}" \
    run_llama7b_polar_dp_pp.py \
    --batch-size "${BATCH_SIZE:-16}" \
    --train-mode polar \
    --comm-timing "${COMM_TIMING:-4}" \
    --bitwidth "${BITWIDTH:-4}" \
    --method bitscom \
    --polar-hook "${POLAR_HOOK:-io}" \
    --run-label "${RUN_LABEL:-polar_bitscom_trace}" \
    --step-log-dir "${STEP_LOG_DIR:-outputs/step_csv}" \
    --pp-size "${PP_SIZE:-8}" \
    --lr "${LR:-2e-5}" \
    --micro-batches "${MICRO_BATCHES:-16}" \
    --max-steps "${MAX_STEPS:-3}" \
    --seq-length "${SEQ_LENGTH:-640}" \
    --disable-profiler
exit_code="$?"
set -e

echo "[trace-demo] exit_code=${exit_code} log_dir=${TRACE_RUN_DIR}"
exit "${exit_code}"
