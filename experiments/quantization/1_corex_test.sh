#!/usr/bin/env bash
set -euo pipefail

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1f0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

run_polar() {
    local hook="$1"
    local label="$2"
    local extra_args=()
    if [[ "${DISABLE_PROFILER:-1}" == "1" ]]; then
        extra_args+=(--disable-profiler)
    fi
    torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
        --nnodes="${NNODES:-2}" \
        --node_rank="${NODE_RANK:-1}" \
        --master_addr="${MASTER_ADDR:-10.31.10.210}" \
        --master_port="${MASTER_PORT:-29500}" \
        run_llama7b_polar_dp_pp.py \
        --batch-size "${BATCH_SIZE:-16}" \
        --train-mode polar \
        --comm-timing "${COMM_TIMING:-4}" \
        --bitwidth "${BITWIDTH:-4}" \
        --method "${METHOD:-bitscom}" \
        --polar-hook "${hook}" \
        --run-label "${label}" \
        --step-log-dir "${STEP_LOG_DIR:-outputs/step_csv}" \
        --pp-size "${PP_SIZE:-8}" \
        --lr "${LR:-2e-5}" \
        --micro-batches "${MICRO_BATCHES:-16}" \
        --max-steps "${MAX_STEPS:-50}" \
        --seq-length "${SEQ_LENGTH:-640}" \
        "${extra_args[@]}"
}

if [[ "${RUN_ABLATIONS:-0}" == "1" ]]; then
    run_polar "io" "polar_full"
    run_polar "scaling_only" "polar_no_error_feedback"
    run_polar "ef_only" "polar_no_gradient_scaling"
    run_polar "none" "polar_no_error_feedback_no_gradient_scaling"
else
    run_polar "${POLAR_HOOK:-io}" "${RUN_LABEL:-polar_${POLAR_HOOK:-io}}"
fi
