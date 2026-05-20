#!/usr/bin/env bash
set -euo pipefail

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"
export NCCL_PORT_RANGE="${NCCL_PORT_RANGE:-30000-30100}"

NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-10.31.10.210}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  experiments/quantization/run_resnet50_bert_large_partial_sync.py \
  --models resnet50 bert-large \
  --sync-interval "${SYNC_INTERVAL:-4}" \
  --micro-steps "${MICRO_STEPS:-0}" \
  --optimizer adamw \
  --bitwidth "${BITWIDTH:-4}" \
  --run-name "${RUN_NAME:-partial_bitscom}" \
  --log-dir "${LOG_DIR:-experiments/quantization/outputs}" \
  --out-dir "${OUT_DIR:-experiments/results}" \
  --resnet-steps "${RESNET_STEPS:-60}" \
  --resnet-batch-size "${RESNET_BATCH_SIZE:-16}" \
  --resnet-lr "${RESNET_LR:-2e-4}" \
  --bert-steps "${BERT_STEPS:-120}" \
  --bert-batch-size "${BERT_BATCH_SIZE:-2}" \
  --bert-lr "${BERT_LR:-2e-6}" \
  --bert-data-source "${BERT_DATA_SOURCE:-synthetic}" \
  --bert-synthetic-task "${BERT_SYNTHETIC_TASK:-marker}"
