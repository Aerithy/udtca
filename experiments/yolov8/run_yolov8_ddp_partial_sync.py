import argparse
import math
import os
import time
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ultralytics import YOLO


class SyntheticClassificationDataset(Dataset):
    def __init__(self, *, samples: int, num_classes: int, img_size: int, seed: int) -> None:
        self.samples = int(samples)
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = idx
        image = torch.rand(3, self.img_size, self.img_size, generator=self.generator)
        label = torch.randint(0, self.num_classes, (1,), generator=self.generator).item()
        return image, torch.tensor(label, dtype=torch.long)


def init_distributed() -> Tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def build_classification_loader(
    *,
    data_root: str,
    batch_size: int,
    img_size: int,
    num_classes: int,
    steps: int,
    rank: int,
    world_size: int,
    workers: int,
    seed: int,
) -> Tuple[DataLoader, int]:
    if data_root:
        try:
            from torchvision import datasets, transforms
        except ImportError as exc:  # pragma: no cover - depends on optional torchvision
            raise RuntimeError("torchvision is required for --data training") from exc

        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )
        dataset = datasets.ImageFolder(data_root, transform=transform)
        num_classes = len(dataset.classes)
    else:
        dataset = SyntheticClassificationDataset(
            samples=max(steps * batch_size * world_size, 128),
            num_classes=num_classes,
            img_size=img_size,
            seed=seed + rank,
        )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, num_classes


def infinite_loader(loader: DataLoader) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def build_buckets(params: List[torch.nn.Parameter], bucket_numel: int) -> List[List[torch.nn.Parameter]]:
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


def sync_bucket(
    *,
    bucket: List[torch.nn.Parameter],
    residuals: Dict[torch.nn.Parameter, torch.Tensor],
    synced: Dict[torch.nn.Parameter, torch.Tensor],
    world_size: int,
    group,
) -> None:
    flat = torch.cat([residuals[p].view(-1) for p in bucket], dim=0)
    if flat.numel() == 0:
        return
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    flat.div_(world_size)

    offset = 0
    for param in bucket:
        numel = param.numel()
        synced[param].copy_(flat[offset : offset + numel].view_as(param))
        residuals[param].zero_()
        offset += numel


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 DDP partial-gradient sync training")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt")
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sync-interval", type=int, default=4)
    parser.add_argument("--bucket-numel", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-scale-loss", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP training.")

    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    yolo = YOLO(args.model)
    model = yolo.model
    model.to(device)
    model.train()

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    loader, detected_classes = build_classification_loader(
        data_root=args.data,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        num_classes=args.num_classes,
        steps=args.steps,
        rank=rank,
        world_size=world_size,
        workers=args.workers,
        seed=args.seed,
    )
    if hasattr(model, "nc") and model.nc != detected_classes and rank == 0:
        print(
            f"[warn] dataset classes ({detected_classes}) != model.nc ({model.nc}); "
            "ensure they match to avoid loss errors."
        )

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    total_numel = sum(int(p.numel()) for p in params)
    max_param_numel = max((int(p.numel()) for p in params), default=0)

    bucket_numel = int(args.bucket_numel)
    if bucket_numel <= 0:
        bucket_numel = max(max_param_numel, math.ceil(total_numel / max(1, args.sync_interval)))

    buckets = build_buckets(params, bucket_numel)
    if rank == 0 and buckets and len(buckets) % args.sync_interval != 0:
        print(
            f"[warn] buckets ({len(buckets)}) not divisible by sync_interval "
            f"({args.sync_interval}); some steps will sync uneven bucket counts."
        )

    if rank == 0:
        total_params = total_numel / 1e6
        print(
            f"[setup] total params={total_params:.2f}M "
            f"bucket_numel={bucket_numel} buckets={len(buckets)} "
            f"sync_interval={args.sync_interval}"
        )

    residuals = {p: torch.zeros_like(p, device=device) for p in params}
    synced = {p: torch.zeros_like(p, device=device) for p in params}

    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    data_iter = infinite_loader(loader)
    total_micro_steps = args.steps * args.sync_interval
    update_step = 0

    torch.cuda.synchronize(device)
    start_time = time.time()

    synced_in_cycle = [False for _ in buckets]

    for micro_step in range(1, total_micro_steps + 1):
        cycle_step = (micro_step - 1) % args.sync_interval
        cycle_id = (micro_step - 1) // args.sync_interval

        if cycle_step == 0:
            synced_in_cycle = [False for _ in buckets]

        images, labels = next(data_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with ddp_model.no_sync():
            logits = ddp_model(images)
            loss = criterion(logits, labels)
            raw_loss = loss.detach()
            if not args.no_scale_loss:
                loss = loss / float(args.sync_interval)
            loss.backward()

        for param in params:
            if param.grad is None:
                continue
            residuals[param].add_(param.grad.detach())
            param.grad = None

        for bucket_idx, bucket in enumerate(buckets):
            if bucket_idx % args.sync_interval != cycle_step:
                continue
            sync_bucket(
                bucket=bucket,
                residuals=residuals,
                synced=synced,
                world_size=world_size,
                group=dist.group.WORLD,
            )
            synced_in_cycle[bucket_idx] = True

        if cycle_step == args.sync_interval - 1:
            if rank == 0 and any(not synced for synced in synced_in_cycle):
                print(f"[warn] cycle {cycle_id}: not all buckets synced before update")

            for param in params:
                param.grad = synced[param]

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for param in params:
                synced[param].zero_()

            update_step += 1
            if update_step % max(1, args.steps // 10) == 0 or update_step == 1:
                loss_scalar = raw_loss.to(torch.float32)
                dist.all_reduce(loss_scalar, op=dist.ReduceOp.SUM)
                loss_scalar.div_(world_size)
                if rank == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[update {update_step}/{args.steps}] "
                        f"loss={loss_scalar.item():.4f} elapsed={elapsed:.1f}s"
                    )

    if rank == 0:
        total_time = time.time() - start_time
        print(f"[done] updates={update_step} total_time={total_time:.1f}s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
