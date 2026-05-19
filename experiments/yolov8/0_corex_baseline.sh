export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens1f0
torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=10.31.10.210 \
  --master_port=29500 \
  experiments/yolov8/run_yolov8_ddp_baseline.py \
  --task detect \
  --model yolov8n.pt \
  --data experiments/yolov8/holes_v3.yaml \
  --imgsz 640 \
  --steps 100 \
  --optimizer adamw \
  --lr 0.001 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --run-name baseline \
  --eval-ddp