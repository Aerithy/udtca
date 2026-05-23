#!/usr/bin/env bash
set -euo pipefail

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --nnodes="${NNODES:-2}" \
    --node_rank="${NODE_RANK:-1}" \
    --master_addr="${MASTER_ADDR:-10.31.10.210}" \
    --master_port="${MASTER_PORT:-29500}" \
    run_llama7b_dp_pp.py \
    --batch-size "${BATCH_SIZE:-16}" \
    --pp-size "${PP_SIZE:-8}" \
    --lr "${LR:-2e-5}" \
    --methods "${METHOD:-none}" \
    --micro-batches "${MICRO_BATCHES:-16}" \
    --max-steps "${MAX_STEPS:-100}" \
    --seq-length "${SEQ_LENGTH:-640}" \
    --out-dir "${OUT_DIR:-experiments/results}"
