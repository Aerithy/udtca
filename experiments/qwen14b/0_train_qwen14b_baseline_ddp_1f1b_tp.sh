#!/usr/bin/env bash
set -euo pipefail

# Node 0 of 2. Traditional baseline: dense DP sync + 1F1B PP + TP.
# The dense DP sync is performed after each stage completes 1F1B backward,
# matching the usual DDP synchronization point without wrapping PipelineStage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MASTER_ADDR="${MASTER_ADDR:-10.31.10.210}"
MASTER_PORT="${MASTER_PORT:-29501}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/run_qwen14b_polar_dp_pp_tp.py" \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --pp-size 8 \
  --tp-size 2 \
  --micro-batches 32 \
  --comm-timing 0 \
  --max-steps 200 \
  --per-device-batch-size 32 \
  --seq-len 256 \
  --lr 2e-4 \
  --dataset-name-or-path HuggingFaceFW/fineweb \
  --text-field text \
  --using-polar false \
  --baseline-mode manual \
  --run-label baseline_ddp_1f1b_tp \
  --polar-hook none \
  --method none
