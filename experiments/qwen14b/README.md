# Qwen14B 2DP x 8PP x 2TP Experiments

Two-node Qwen2.5-14B-Instruct experiments with:

- 2 nodes x 16 GPUs = 32 GPUs
- DP = 2 across nodes
- PP = 8
- TP = 2
- 1F1B pipeline schedule
- micro-batches = 8
- per-device batch size = 8
- sequence length = 256
- node-local token cache under `experiments/qwen14b/cache`

Only `LOCAL_RANK=0` on each node contacts Hugging Face to warm the
tokenizer/config cache and materialize the token-block dataset cache. Other
local ranks wait at a distributed barrier and then read local files.

## POLAR + bitscom

This is the innovation path:

- `--using-polar true`
- POLAR hook = `ef_lowmem`
- bitscom DP communication = 4-bit
- run label = `polar_bitscom_1f1b_tp`

```bash
# node 0
bash experiments/qwen14b/0_train_qwen14b_polar_dp_pp_tp.sh

# node 1
bash experiments/qwen14b/1_train_qwen14b_polar_dp_pp_tp.sh
```

## Dense DP Baseline

This baseline uses the same 1F1B + TP topology but disables POLAR and bitscom:

- `--using-polar false`
- dense DP gradient averaging after each stage finishes 1F1B backward
- run label = `baseline_ddp_1f1b_tp`

The synchronization point is the same place DDP would synchronize gradients,
but it is implemented as explicit dense DP all-reduce to avoid wrapping
`PipelineStage` with DDP.

```bash
# node 0
bash experiments/qwen14b/0_train_qwen14b_baseline_ddp_1f1b_tp.sh

# node 1
bash experiments/qwen14b/1_train_qwen14b_baseline_ddp_1f1b_tp.sh
```

Override launch address/port if needed:

```bash
MASTER_ADDR=<node0_ip> MASTER_PORT=11234 bash experiments/qwen14b/0_train_qwen14b_polar_dp_pp_tp.sh
MASTER_ADDR=<node0_ip> MASTER_PORT=11234 bash experiments/qwen14b/1_train_qwen14b_polar_dp_pp_tp.sh
```

For the baseline scripts, use the same override pattern with
`0_train_qwen14b_baseline_ddp_1f1b_tp.sh` and
`1_train_qwen14b_baseline_ddp_1f1b_tp.sh`.
