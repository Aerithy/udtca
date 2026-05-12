# Experiments

This folder contains standalone scripts to run compression experiments for:
- Llama7B (DP+PP)
- GPT-2
- BERT
- ResNet-50

Results are saved under:
- experiments/results/baselines
- experiments/results/bitscom

Each run writes a loss curve CSV/PNG plus a summary CSV with throughput and final loss.
Accuracy is not computed (loss-only comparisons).

## Methods
- none: dense all-reduce (no compression)
- quant8: per-tensor int8 quantization + all-gather
- topk: top-k sparsification + all-gather
- powersgd: low-rank approximation + all-reduce
- bitscom: bitscom low-bit all-reduce (separate category)

## Run: ResNet50 / BERT / GPT-2
Datasets (real):
- ResNet50: CIFAR-10
- BERT: GLUE/SST-2
- GPT-2: WikiText-2 (raw)

Use torchrun with multiple GPUs.

Examples:
```
cd /home/aerith/udtca
torchrun --nproc_per_node=2 experiments/run_multimodel.py \
  --models resnet50 bert gpt2 \
  --methods none quant8 topk powersgd bitscom
```

To run a single model/method:
```
torchrun --nproc_per_node=2 experiments/run_multimodel.py \
  --models gpt2 --methods topk

Disable downloads (use local cache only):
```
torchrun --nproc_per_node=2 experiments/run_multimodel.py --no-download
```

## Run: Llama7B DP+PP
This script uses pipeline parallelism with DP gradient sync.

Example:
```
cd /home/aerith/udtca
torchrun --nproc_per_node=4 experiments/run_llama7b_dp_pp.py \
  --pp-size 2 \
  --methods none bitscom quant8
```

Notes:
- Llama7B defaults to WikiText-103 (switch to wikitext-2 or c4 via flags).
- Downloads are enabled by default; pass --no-download to use cache only.
- Loss curves are stored per method; throughput is tokens/sec.
- For ResNet50 throughput is samples/sec; for BERT/GPT-2 it is tokens/sec.
- bitscom runs are stored under experiments/results/bitscom.

## Run Order
Recommended order for experiments:
1) Run ResNet50/BERT/GPT-2 baselines first (none, quant8, topk, powersgd).
2) Run bitscom for ResNet50/BERT/GPT-2.
3) Run Llama7B DP+PP baselines (none, quant8, topk, powersgd).
4) Run Llama7B DP+PP with bitscom.
