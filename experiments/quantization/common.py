import csv
import math
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist


class CompressionConfig:
    def __init__(
        self,
        mode: str,
        *,
        bitwidth: int = 4,
        topk_ratio: float = 0.01,
        powersgd_rank: int = 2,
        powersgd_dim: int = 1024,
        simulate_quantization: bool = False,
        stochastic_rounding: bool = False,
    ) -> None:
        self.mode = str(mode)
        self.bitwidth = int(bitwidth)
        self.topk_ratio = float(topk_ratio)
        self.powersgd_rank = int(powersgd_rank)
        self.powersgd_dim = int(powersgd_dim)
        self.simulate_quantization = bool(simulate_quantization)
        self.stochastic_rounding = bool(stochastic_rounding)


def _quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_abs = float(x.abs().max().item())
    scale = max_abs / 127.0
    if scale == 0.0:
        q = torch.zeros_like(x, dtype=torch.int8)
        return q, torch.tensor(scale, dtype=torch.float32, device=x.device)
    q = torch.clamp((x / scale).round(), -127, 127).to(torch.int8)
    return q, torch.tensor(scale, dtype=torch.float32, device=x.device)


def _dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def _topk_sparsify(x: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor, int]:
    numel = x.numel()
    k = max(1, int(numel * ratio))
    vals, idx = torch.topk(x.abs(), k)
    sel_vals = x.view(-1)[idx]
    return sel_vals, idx, k


def _gather_list(tensor: torch.Tensor, group) -> List[torch.Tensor]:
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return gathered


def _reduce_quant8(flat: torch.Tensor, group, world_size: int) -> torch.Tensor:
    q, scale = _quantize_int8(flat)
    gathered_q = _gather_list(q, group)
    scale_tensor = scale.view(1)
    gathered_scale = _gather_list(scale_tensor, group)

    out = torch.zeros_like(flat, dtype=torch.float32)
    for q_i, s_i in zip(gathered_q, gathered_scale):
        out.add_(_dequantize_int8(q_i, float(s_i.item())))
    return out


def _reduce_topk(flat: torch.Tensor, group, world_size: int, ratio: float) -> torch.Tensor:
    vals, idx, k = _topk_sparsify(flat, ratio)
    gathered_vals = _gather_list(vals, group)
    gathered_idx = _gather_list(idx, group)

    out = torch.zeros_like(flat, dtype=torch.float32)
    for v_i, i_i in zip(gathered_vals, gathered_idx):
        out.scatter_add_(0, i_i, v_i.to(torch.float32))
    return out


def _reshape_for_powersgd(flat: torch.Tensor, target_dim: int) -> Tuple[torch.Tensor, int]:
    numel = flat.numel()
    dim = min(max(1, target_dim), numel)
    rows = int(math.ceil(numel / float(dim)))
    padded = rows * dim - numel
    if padded:
        flat = torch.cat([flat, flat.new_zeros(padded)], dim=0)
    matrix = flat.view(rows, dim)
    return matrix, numel


def _reduce_powersgd(
    flat: torch.Tensor,
    group,
    world_size: int,
    rank: int,
    dim: int,
) -> torch.Tensor:
    matrix, original_numel = _reshape_for_powersgd(flat, dim)
    rows, cols = matrix.shape
    rank = max(1, min(rank, rows, cols))

    q = torch.randn(cols, rank, device=flat.device, dtype=flat.dtype)
    p = matrix @ q
    dist.all_reduce(p, op=dist.ReduceOp.SUM, group=group)
    p, _ = torch.linalg.qr(p, mode="reduced")

    q = matrix.transpose(0, 1) @ p
    dist.all_reduce(q, op=dist.ReduceOp.SUM, group=group)

    approx = p @ q.transpose(0, 1)
    return approx.reshape(-1)[:original_numel]


def reduce_flat(
    flat: torch.Tensor,
    *,
    group,
    world_size: int,
    cfg: CompressionConfig,
    lowbit_group=None,
) -> torch.Tensor:
    mode = cfg.mode

    if mode == "none":
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        return flat

    if mode == "quant8":
        return _reduce_quant8(flat, group, world_size)

    if mode == "topk":
        return _reduce_topk(flat, group, world_size, cfg.topk_ratio)

    if mode == "powersgd":
        return _reduce_powersgd(flat, group, world_size, cfg.powersgd_rank, cfg.powersgd_dim)

    if mode == "bitscom":
        if lowbit_group is None:
            raise RuntimeError("bitscom LowBitGroup is required for mode=bitscom")
        lowbit_group.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=False)
        return flat

    raise ValueError(f"unknown compression mode: {mode}")


def sync_grads_bucketed(
    *,
    grads: Iterable[torch.Tensor],
    group,
    world_size: int,
    cfg: CompressionConfig,
    bucket_numel: int,
    lowbit_group=None,
) -> None:
    bucket: List[torch.Tensor] = []
    bucket_size = 0

    def _flush() -> None:
        nonlocal bucket, bucket_size
        if not bucket:
            return

        flat = torch.cat([g.contiguous().view(-1) for g in bucket], dim=0)
        reduced = reduce_flat(
            flat,
            group=group,
            world_size=world_size,
            cfg=cfg,
            lowbit_group=lowbit_group,
        )
        reduced.div_(world_size)

        offset = 0
        for g in bucket:
            n = g.numel()
            g.copy_(reduced[offset : offset + n].view_as(g))
            offset += n

        bucket = []
        bucket_size = 0

    for g in grads:
        n = int(g.numel())
        if bucket and bucket_size + n > bucket_numel:
            _flush()
        bucket.append(g)
        bucket_size += n

    _flush()


def measure_throughput_tokens(
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    world_size: int,
    total_time_s: float,
) -> float:
    total_tokens = float(steps * batch_size * seq_len * world_size)
    return total_tokens / max(total_time_s, 1e-9)


def measure_throughput_samples(
    *,
    steps: int,
    batch_size: int,
    world_size: int,
    total_time_s: float,
) -> float:
    total_samples = float(steps * batch_size * world_size)
    return total_samples / max(total_time_s, 1e-9)


def save_loss_curve(
    *,
    out_dir: str,
    run_name: str,
    losses: List[float],
    smooth_window: int,
) -> Tuple[str, str]:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{run_name}_loss.csv")
    png_path = os.path.join(out_dir, f"{run_name}_loss.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for idx, val in enumerate(losses, start=1):
            writer.writerow([idx, val])

    def _moving_avg(vals: List[float], window: int) -> List[float]:
        if window <= 1:
            return list(vals)
        out: List[float] = []
        running = 0.0
        for i, v in enumerate(vals):
            running += v
            if i >= window:
                running -= vals[i - window]
            out.append(running / float(min(i + 1, window)))
        return out

    steps = list(range(1, len(losses) + 1))
    window = max(1, int(smooth_window))

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(steps, losses, color="#1f77b4", alpha=0.35, linewidth=1.2)
    plt.plot(steps, _moving_avg(losses, window), color="#1f77b4", linewidth=2.0)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(run_name)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    return csv_path, png_path


def append_summary_row(
    *,
    csv_path: str,
    row: Dict[str, object],
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def step_timer() -> Tuple[callable, List[float]]:
    times: List[float] = []

    def _record(dt: float) -> None:
        times.append(dt)

    return _record, times


def sync_loss_across_ranks(loss: torch.Tensor, group) -> float:
    loss_scalar = loss.detach().to(torch.float32).view(1)
    gathered = [torch.zeros_like(loss_scalar) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, loss_scalar, group=group)
    return float(torch.stack(gathered).mean().item())


def barrier(group) -> None:
    dist.barrier(group=group)
