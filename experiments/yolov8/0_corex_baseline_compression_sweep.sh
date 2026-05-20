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
METHODS="${METHODS:-none quant8 topk powersgd bitscom}"

for method in ${METHODS}; do
  torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    experiments/yolov8/run_yolov8_ddp_baseline.py \
    --task detect \
    --model "${MODEL:-yolov8n.pt}" \
    --data "${DATA:-experiments/yolov8/holes_v3.yaml}" \
    --imgsz "${IMGSZ:-640}" \
    --steps "${STEPS:-100}" \
    --optimizer "${OPTIMIZER:-adamw}" \
    --lr "${LR:-0.001}" \
    --weight-decay "${WEIGHT_DECAY:-0.01}" \
    --grad-clip "${GRAD_CLIP:-5.0}" \
    --run-name "${RUN_NAME_PREFIX:-baseline}_${method}" \
    --method "${method}" \
    --bitwidth "${BITWIDTH:-4}" \
    --topk-ratio "${TOPK_RATIO:-0.01}" \
    --powersgd-rank "${POWERSGD_RANK:-2}" \
    --powersgd-dim "${POWERSGD_DIM:-1024}" \
    --eval-ddp
done
