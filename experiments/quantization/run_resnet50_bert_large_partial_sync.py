import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import (
    append_summary_row,
    measure_throughput_samples,
    measure_throughput_tokens,
    save_loss_curve,
)
from run_resnet50_bert_large import (
    _build_dataloader,
    _build_model,
    _model_run_shape,
    _model_train_hparams,
    _run_step,
)


def import_bitscom():
    try:
        import bitscom

        return bitscom
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        bitscom_python = repo_root / "bitscom" / "python"
        if bitscom_python.exists():
            sys.path.insert(0, str(bitscom_python))
        import bitscom

        return bitscom


def init_distributed() -> Tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def init_run_dir(log_dir: str, run_name: str, rank: int) -> Optional[Path]:
    if rank != 0:
        return None
    base = Path(log_dir).expanduser()
    run_id = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _method_dir(base: str, method: str) -> str:
    return os.path.join(base, "bitscom" if "bitscom" in method else "baselines")


def build_buckets(
    params: List[torch.nn.Parameter],
    bucket_numel: int,
) -> List[List[torch.nn.Parameter]]:
    buckets: List[List[torch.nn.Parameter]] = []
    current: List[torch.nn.Parameter] = []
    current_size = 0

    for param in params:
        numel = int(param.numel())
        if current and current_size + numel > bucket_numel:
            buckets.append(current)
            current = []
            current_size = 0
        current.append(param)
        current_size += numel

    if current:
        buckets.append(current)
    return buckets


@dataclass
class PendingSync:
    work: Any
    flat: torch.Tensor
    bucket: List[torch.nn.Parameter]


def launch_bucket_sync(
    *,
    bucket: List[torch.nn.Parameter],
    residuals: Dict[torch.nn.Parameter, torch.Tensor],
    group,
    lowbit_group=None,
) -> Optional[PendingSync]:
    flat = torch.cat([residuals[p].view(-1) for p in bucket], dim=0)
    if flat.numel() == 0:
        return None
    if lowbit_group is not None:
        work = lowbit_group.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
    else:
        work = dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group, async_op=True)
    return PendingSync(work=work, flat=flat, bucket=bucket)


def finish_bucket_sync(
    *,
    pending: PendingSync,
    residuals: Dict[torch.nn.Parameter, torch.Tensor],
    synced: Dict[torch.nn.Parameter, torch.Tensor],
    world_size: int,
) -> None:
    pending.work.wait()
    pending.flat.div_(world_size)

    offset = 0
    for param in pending.bucket:
        numel = param.numel()
        synced[param].copy_(pending.flat[offset : offset + numel].view_as(param))
        residuals[param].zero_()
        offset += numel


def _finite_grads(params: List[torch.nn.Parameter]) -> bool:
    for param in params:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


def _make_optimizer(params: List[torch.nn.Parameter], args, hparams):
    if args.optimizer == "adam":
        return torch.optim.Adam(
            params,
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=hparams["lr"],
            momentum=0.9,
            weight_decay=hparams["weight_decay"],
        )
    return torch.optim.AdamW(
        params,
        lr=hparams["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=hparams["weight_decay"],
    )


def _make_scheduler(optimizer, *, steps: int, hparams):
    warmup_steps = max(1, int(steps * hparams["warmup_ratio"]))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_ratio = hparams["min_lr_ratio"]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _mean_loss(loss: torch.Tensor, world_size: int) -> float:
    loss_tensor = loss.detach().to(torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_tensor.div_(world_size)
    return float(loss_tensor.item())


def run_partial_sync(
    *,
    model_name: str,
    args,
    rank: int,
    world_size: int,
    local_rank: int,
    lowbit_group,
    run_dir: Optional[Path],
) -> Dict[str, object]:
    device = torch.device(f"cuda:{local_rank}")
    shape = _model_run_shape(model_name, args)
    hparams = _model_train_hparams(model_name, args)
    dataloader = _build_dataloader(
        model_name=model_name,
        args=args,
        batch_size=shape["batch_size"],
        dataset_size=shape["dataset_size"],
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = _build_model(model_name, args).to(device)
    model.train()
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    total_numel = sum(int(p.numel()) for p in params)
    max_param_numel = max((int(p.numel()) for p in params), default=0)
    bucket_numel = int(args.bucket_numel)
    if bucket_numel <= 0:
        bucket_numel = max(
            max_param_numel,
            math.ceil(total_numel / max(1, args.sync_interval)),
        )
    buckets = build_buckets(params, bucket_numel)

    if rank == 0:
        print(
            f"[setup] model={model_name} params={total_numel / 1e6:.2f}M "
            f"bucket_numel={bucket_numel} buckets={len(buckets)} "
            f"sync_interval={args.sync_interval} bitwidth={args.bitwidth} "
            f"lr={hparams['lr']} warmup={hparams['warmup_ratio']} "
            f"clip={hparams['grad_clip_norm']}",
            flush=True,
        )

    residuals = {p: torch.zeros_like(p, device=device) for p in params}
    synced = {p: torch.zeros_like(p, device=device) for p in params}
    optimizer = _make_optimizer(params, args, hparams)
    scheduler = _make_scheduler(optimizer, steps=shape["steps"], hparams=hparams)

    total_micro_steps = shape["steps"] * args.sync_interval
    if args.micro_steps and args.micro_steps > 0:
        if args.micro_steps % args.sync_interval != 0:
            raise RuntimeError("--micro-steps must be divisible by --sync-interval")
        total_micro_steps = args.micro_steps
    target_updates = total_micro_steps // args.sync_interval

    data_iter = _infinite_loader(dataloader)
    pending_by_bucket: Dict[int, PendingSync] = {}
    synced_in_cycle = [False for _ in buckets]
    loss_history: List[Tuple[int, float, float]] = []
    update_step = 0

    torch.cuda.synchronize(device)
    start_time = time.time()

    for micro_step in range(1, total_micro_steps + 1):
        cycle_step = (micro_step - 1) % args.sync_interval
        cycle_id = (micro_step - 1) // args.sync_interval
        if cycle_step == 0:
            synced_in_cycle = [False for _ in buckets]

        batch = next(data_iter)
        with ddp_model.no_sync():
            loss = _run_step(
                model_name=model_name,
                model=ddp_model,
                batch=batch,
                device=device,
            )
            raw_loss = loss.detach()
            if not args.no_scale_loss:
                loss = loss / float(args.sync_interval)
            loss.backward()

        if not _finite_grads(params):
            if rank == 0:
                print(
                    f"[warn] non-finite gradients at micro_step={micro_step}; "
                    "zeroing this local accumulation before partial sync",
                    flush=True,
                )
            for param in params:
                if param.grad is not None:
                    param.grad.zero_()

        for param in params:
            if param.grad is None:
                continue
            residuals[param].add_(param.grad.detach())
            param.grad = None

        for bucket_idx, bucket in enumerate(buckets):
            if bucket_idx % args.sync_interval != cycle_step:
                continue
            pending = pending_by_bucket.pop(bucket_idx, None)
            if pending is not None:
                finish_bucket_sync(
                    pending=pending,
                    residuals=residuals,
                    synced=synced,
                    world_size=world_size,
                )
            pending = launch_bucket_sync(
                bucket=bucket,
                residuals=residuals,
                group=dist.group.WORLD,
                lowbit_group=lowbit_group,
            )
            if pending is not None:
                pending_by_bucket[bucket_idx] = pending
            synced_in_cycle[bucket_idx] = True

        if cycle_step == args.sync_interval - 1:
            for pending in list(pending_by_bucket.values()):
                finish_bucket_sync(
                    pending=pending,
                    residuals=residuals,
                    synced=synced,
                    world_size=world_size,
                )
            pending_by_bucket.clear()

            missing = sum(1 for was_synced in synced_in_cycle if not was_synced)
            if rank == 0 and missing:
                print(
                    f"[warn] cycle {cycle_id}: {missing} buckets were not synced",
                    flush=True,
                )

            for param in params:
                param.grad = synced[param]

            if hparams["grad_clip_norm"] > 0:
                if args.grad_clip_steps <= 0 or update_step < args.grad_clip_steps:
                    torch.nn.utils.clip_grad_norm_(params, hparams["grad_clip_norm"])

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            for param in params:
                synced[param].zero_()

            update_step += 1
            mean_loss = _mean_loss(raw_loss, world_size)
            if rank == 0:
                elapsed = time.time() - start_time
                loss_history.append((update_step, mean_loss, elapsed))
                if (
                    update_step == 1
                    or update_step == target_updates
                    or update_step % max(1, target_updates // 10) == 0
                ):
                    print(
                        f"[update {update_step}/{target_updates}] "
                        f"loss={mean_loss:.4f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    dist.barrier()
    total_time = time.time() - start_time
    losses = [row[1] for row in loss_history]
    if model_name == "resnet50":
        throughput = measure_throughput_samples(
            steps=target_updates,
            batch_size=shape["batch_size"],
            world_size=world_size,
            total_time_s=total_time,
        )
        throughput_name = "throughput_samples_per_s"
    else:
        throughput = measure_throughput_tokens(
            steps=target_updates,
            batch_size=shape["batch_size"],
            seq_len=shape["seq_len"],
            world_size=world_size,
            total_time_s=total_time,
        )
        throughput_name = "throughput_tokens_per_s"

    if rank == 0 and run_dir is not None:
        loss_path = run_dir / f"{model_name}_loss_curve.csv"
        with loss_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "loss", "elapsed_s"])
            writer.writerows(loss_history)

        summary = {
            "run_name": args.run_name,
            "model": model_name,
            "method": "partial_bitscom",
            "updates": target_updates,
            "sync_interval": args.sync_interval,
            "bitwidth": args.bitwidth,
            "simulate_quantization": args.simulate_quantization,
            "stochastic_rounding": args.stochastic_rounding,
            "micro_steps": total_micro_steps,
            "bucket_numel": bucket_numel,
            "world_size": world_size,
            "batch_size": shape["batch_size"],
            "total_time_s": total_time,
            "avg_step_time_ms": (total_time / max(1, target_updates)) * 1000.0,
            throughput_name: throughput,
            "final_loss": losses[-1] if losses else None,
            "lr": hparams["lr"],
            "warmup_ratio": hparams["warmup_ratio"],
            "grad_clip_norm": hparams["grad_clip_norm"],
            "weight_decay": hparams["weight_decay"],
        }
        with (run_dir / f"{model_name}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return {
        "losses": losses,
        "loss_history": loss_history,
        "avg_step_time_ms": (total_time / max(1, target_updates)) * 1000.0,
        "throughput": throughput,
        "throughput_name": throughput_name,
        "final_loss": losses[-1] if losses else 0.0,
        "world_size": world_size,
        "batch_size": shape["batch_size"],
        "steps": target_updates,
        "bucket_numel": bucket_numel,
        "lr": hparams["lr"],
        "warmup_ratio": hparams["warmup_ratio"],
        "grad_clip_norm": hparams["grad_clip_norm"],
        "weight_decay": hparams["weight_decay"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResNet50/BERT-large DDP partial sync with bitscom"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "bert-large"],
        choices=["resnet50", "bert-large"],
    )
    parser.add_argument("--sync-interval", type=int, default=4)
    parser.add_argument("--micro-steps", type=int, default=0)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--bucket-numel", type=int, default=0)
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--simulate-quantization", action="store_true")
    parser.add_argument("--stochastic-rounding", action="store_true")
    parser.add_argument("--no-scale-loss", action="store_true")
    parser.add_argument("--grad-clip-steps", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="experiments/quantization/outputs")
    parser.add_argument("--run-name", type=str, default="partial_bitscom")
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    parser.add_argument("--smooth-window", type=int, default=4)

    parser.add_argument("--resnet-steps", type=int, default=60)
    parser.add_argument("--resnet-batch-size", type=int, default=16)
    parser.add_argument("--resnet-dataset-size", type=int, default=512)
    parser.add_argument("--resnet-image-size", type=int, default=224)
    parser.add_argument("--resnet-num-classes", type=int, default=10)
    parser.add_argument("--resnet-lr", type=float, default=2e-4)
    parser.add_argument("--resnet-weight-decay", type=float, default=1e-2)
    parser.add_argument("--resnet-warmup-ratio", type=float, default=0.08)
    parser.add_argument("--resnet-grad-clip-norm", type=float, default=1.0)

    parser.add_argument("--bert-steps", type=int, default=120)
    parser.add_argument("--bert-batch-size", type=int, default=2)
    parser.add_argument("--bert-dataset-size", type=int, default=1024)
    parser.add_argument("--bert-seq-len", type=int, default=128)
    parser.add_argument("--bert-num-labels", type=int, default=2)
    parser.add_argument("--bert-lr", type=float, default=2e-6)
    parser.add_argument("--bert-weight-decay", type=float, default=1e-2)
    parser.add_argument("--bert-warmup-ratio", type=float, default=0.15)
    parser.add_argument("--bert-grad-clip-norm", type=float, default=0.5)
    parser.add_argument(
        "--bert-data-source",
        choices=["auto", "glue", "synthetic"],
        default="auto",
    )
    parser.add_argument(
        "--bert-synthetic-task",
        choices=["marker", "random"],
        default="marker",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.sync_interval <= 0:
        raise ValueError("--sync-interval must be positive")

    rank, world_size, local_rank = init_distributed()
    bitscom = import_bitscom()
    bitscom.init(bitwidth=args.bitwidth)
    lowbit_group = bitscom.LowBitGroup(
        bitwidth=args.bitwidth,
        process_group=dist.group.WORLD,
        simulate_quantization=args.simulate_quantization,
        stochastic_rounding=args.stochastic_rounding,
    )

    run_dir = init_run_dir(args.log_dir, args.run_name, rank)

    try:
        for model_name in args.models:
            run = run_partial_sync(
                model_name=model_name,
                args=args,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                lowbit_group=lowbit_group,
                run_dir=run_dir,
            )
            if rank == 0:
                method = "partial_bitscom"
                out_dir = _method_dir(args.out_dir, method)
                run_name = f"{model_name}_{method}"
                save_loss_curve(
                    out_dir=out_dir,
                    run_name=run_name,
                    losses=run["losses"],
                    smooth_window=args.smooth_window,
                )
                summary_path = os.path.join(out_dir, f"summary_{model_name}.csv")
                append_summary_row(
                    csv_path=summary_path,
                    row={
                        "model": model_name,
                        "method": method,
                        "avg_step_time_ms": run["avg_step_time_ms"],
                        run["throughput_name"]: run["throughput"],
                        "final_loss": run["final_loss"],
                        "world_size": run["world_size"],
                        "batch_size": run["batch_size"],
                        "steps": run["steps"],
                        "sync_interval": args.sync_interval,
                        "bitwidth": args.bitwidth,
                        "bucket_numel": run["bucket_numel"],
                        "lr": run["lr"],
                        "warmup_ratio": run["warmup_ratio"],
                        "grad_clip_norm": run["grad_clip_norm"],
                        "weight_decay": run["weight_decay"],
                    },
                )
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
