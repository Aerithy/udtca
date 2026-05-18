# udtca

## Experiments

### YOLOv8 DDP partial-gradient sync
Script: `experiments/yolov8/run_yolov8_ddp_partial_sync.py`

Example (single node, 4 GPUs):
```
torchrun --nproc_per_node=4 experiments/yolov8/run_yolov8_ddp_partial_sync.py \
  --model yolov8n-cls.pt \
  --data /path/to/classification_dataset \
  --sync-interval 4 \
  --steps 100
```

Notes:
- `--sync-interval N` synchronizes roughly 1/N of gradient buckets per step and performs an optimizer update every N steps.
- `--bucket-numel` can be tuned to align bucket counts with `--sync-interval`.
- If `--data` is omitted, synthetic classification data is used.
