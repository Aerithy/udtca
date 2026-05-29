# Qwen14B Polar DP + PP + TP

Two-node experiment for Qwen2.5-14B-Instruct with:

- 2 nodes x 16 GPUs = 32 GPUs
- DP = 2 across nodes
- PP = 8 within each node
- TP = 2 within each pipeline stage
- POLAR hook = `ef_lowmem`
- bitscom DP communication = 4-bit

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
