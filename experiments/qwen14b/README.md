# Qwen14B Polar DP + PP + TP

Two-node experiment for Qwen2.5-14B-Instruct with:

- 2 nodes x 16 GPUs = 32 GPUs
- DP = 2 across nodes
- PP = 8 within each node
- TP = 2 within each pipeline stage
- POLAR hook = `ef_lowmem`
- bitscom DP communication = 4-bit
- micro-batches = 8
- per-device batch size = 8
- sequence length = 256
- dataset cache = node-local token blocks under `experiments/qwen14b/cache`

Only `LOCAL_RANK=0` on each node contacts Hugging Face. It warms the
tokenizer/config cache and materializes the small token-block dataset cache;
all other local ranks wait at a distributed barrier and then read local files.

Run from either the repository root or this directory:

```bash
# node 0
bash experiments/qwen14b/0_train_qwen14b_polar_dp_pp_tp.sh

# node 1
bash experiments/qwen14b/1_train_qwen14b_polar_dp_pp_tp.sh
```

Override the launch address if needed:

```bash
MASTER_ADDR=<node0_ip> MASTER_PORT=11234 bash experiments/qwen14b/0_train_qwen14b_polar_dp_pp_tp.sh
MASTER_ADDR=<node0_ip> MASTER_PORT=11234 bash experiments/qwen14b/1_train_qwen14b_polar_dp_pp_tp.sh
```

If the Hugging Face cache is already present and the run should stay fully
offline, add `--hf-local-files-only` to both launch scripts.

## TODO

- [x] Fix the double label-shift bug: the dataset previously emitted shifted
  labels, while the loss function shifts logits/labels again.
- [x] Complete TP support: the final `lm_head` now participates in tensor
  parallelism while returning replicated logits for the existing loss path.
- [ ] Continue investigating memory imbalance: explain why stage 0 ranks use
  much more memory than other PP stages under the 2TP + 8PP layout. The
  training wrapper now prints per-stage parameter and CUDA memory snapshots.
- [ ] Leave `ef_lowmem` unchanged for this issue. It is the intended innovation
  path and is not considered the active cause of the current abnormal run.
