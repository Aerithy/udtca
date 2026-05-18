torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=10.31.10.210 \
  --master_port=29500 \
  experiments/yolov8/run_yolov8_ddp_partial_sync.py \
  --task detect \
  --model yolov8n.pt \
  --data experiments/yolov8/holes_v3.yaml \
  --imgsz 640 \
  --sync-interval 4 \
  --steps 100