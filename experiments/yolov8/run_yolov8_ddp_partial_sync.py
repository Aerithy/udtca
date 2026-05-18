import argparse
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ultralytics import YOLO

try:  # optional dependency for detection datasets
    import yaml
except ImportError:  # pragma: no cover - depends on optional PyYAML
    yaml = None

try:  # optional dependency for image loading
    from PIL import Image
except ImportError:  # pragma: no cover - depends on optional Pillow
    Image = None

try:  # optional dependency for image conversion
    import numpy as np
except ImportError:  # pragma: no cover - depends on optional numpy
    np = None


MIN_SYNTHETIC_SAMPLES = 128


class SyntheticClassificationDataset(Dataset):
    def __init__(self, *, samples: int, num_classes: int, img_size: int, seed: int) -> None:
        self.samples = int(samples)
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.base_seed = int(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.base_seed + int(idx))
        image = torch.rand(3, self.img_size, self.img_size, generator=generator)
        label = torch.randint(0, self.num_classes, (1,), generator=generator).item()
        return image, torch.tensor(label, dtype=torch.long)


class YoloDetectionDataset(Dataset):
    def __init__(self, *, images_dir: Path, labels_dir: Path, img_size: int) -> None:
        if Image is None or np is None:
            raise RuntimeError("Pillow and numpy are required for detection datasets.")
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = int(img_size)
        self.image_paths = self._collect_images(images_dir)
        if not self.image_paths:
            raise RuntimeError(f"No images found in {images_dir}")

    @staticmethod
    def _collect_images(images_dir: Path) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted(
            path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in exts
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        image_path = self.image_paths[int(idx)]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        image_array = np.asarray(image).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            boxes = []
            classes = []
            with label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        raise RuntimeError(f"Invalid label line in {label_path}: {line}")
                    cls, x, y, w, h = parts
                    classes.append(int(float(cls)))
                    boxes.append([float(x), float(y), float(w), float(h)])
            cls_tensor = torch.tensor(classes, dtype=torch.long).view(-1, 1)
            box_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            cls_tensor = torch.zeros((0, 1), dtype=torch.long)
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)

        return image_tensor, (cls_tensor, box_tensor)


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
            samples=max(steps * batch_size * world_size, MIN_SYNTHETIC_SAMPLES),
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


def _load_data_yaml(data_yaml: str) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML datasets.")
    with open(data_yaml, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML format: {data_yaml}")
    return data


def _resolve_yaml_path(data_yaml: str, value: str, root: str) -> Path:
    yaml_dir = Path(data_yaml).resolve().parent
    if root:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = (yaml_dir / root_path).resolve()
    else:
        root_path = yaml_dir

    value_path = Path(value)
    if not value_path.is_absolute():
        value_path = (root_path / value_path).resolve()
    return value_path


def _derive_labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    if images_dir.name in {"train", "val", "test"}:
        return images_dir.parent.parent / "labels" / images_dir.name
    return images_dir.parent / "labels"


def build_detection_loader(
    *,
    data_yaml: str,
    batch_size: int,
    img_size: int,
    rank: int,
    world_size: int,
    workers: int,
) -> Tuple[DataLoader, int]:
    data = _load_data_yaml(data_yaml)
    train_entry = data.get("train")
    if not train_entry:
        raise RuntimeError("YAML must define a 'train' entry for detection datasets.")
    if isinstance(train_entry, (list, tuple)):
        train_entry = train_entry[0]

    root = data.get("path", "")
    images_dir = _resolve_yaml_path(data_yaml, str(train_entry), str(root))
    labels_dir = _derive_labels_dir(images_dir)

    dataset = YoloDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        img_size=img_size,
    )

    names = data.get("names", {})
    if isinstance(names, dict):
        num_classes = len(names)
    elif isinstance(names, (list, tuple)):
        num_classes = len(names)
    else:
        num_classes = 0

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    def collate_fn(batch):
        images = []
        cls_list = []
        box_list = []
        batch_idx_list = []

        for idx, (img, (cls, boxes)) in enumerate(batch):
            images.append(img)
            if boxes.numel():
                cls_list.append(cls)
                box_list.append(boxes)
                batch_idx_list.append(
                    torch.full((boxes.shape[0], 1), idx, dtype=torch.long)
                )

        images = torch.stack(images, dim=0)
        if cls_list:
            cls_tensor = torch.cat(cls_list, dim=0)
            box_tensor = torch.cat(box_list, dim=0)
            batch_idx = torch.cat(batch_idx_list, dim=0)
        else:
            cls_tensor = torch.zeros((0, 1), dtype=torch.long)
            box_tensor = torch.zeros((0, 4), dtype=torch.float32)
            batch_idx = torch.zeros((0, 1), dtype=torch.long)

        return {
            "img": images,
            "cls": cls_tensor,
            "bboxes": box_tensor,
            "batch_idx": batch_idx,
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return loader, num_classes


def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {
            key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def _normalize_hyp(hyp, fallback_args=None):
    if isinstance(hyp, dict):
        hyp = SimpleNamespace(**hyp)
    if hyp is None:
        args = fallback_args
        if isinstance(args, dict):
            hyp_values = {k: args.get(k) for k in ("box", "cls", "dfl")}
        else:
            hyp_values = {k: getattr(args, k, None) for k in ("box", "cls", "dfl")}
        for key in hyp_values:
            if hyp_values[key] is None:
                hyp_values[key] = 1.0
        hyp = SimpleNamespace(**hyp_values)

    for key in ("box", "cls", "dfl"):
        if not hasattr(hyp, key):
            setattr(hyp, key, 1.0)
    return hyp


def ensure_detection_hyp(model) -> None:
    hyp = _normalize_hyp(getattr(model, "hyp", None), fallback_args=getattr(model, "args", None))
    model.hyp = hyp
    criterion = getattr(model, "criterion", None)
    if criterion is None and hasattr(model, "init_criterion"):
        criterion = model.init_criterion()
        model.criterion = criterion
    if criterion is not None:
        criterion.hyp = _normalize_hyp(getattr(criterion, "hyp", None), fallback_args=hyp)


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
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=("auto", "classify", "detect"),
        help="auto detects from --data; use detect for YAML detection datasets",
    )
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

    if not any(p.requires_grad for p in model.parameters()):
        if rank == 0:
            print("[warn] model has no trainable parameters; enabling gradients.")
        model.requires_grad_(True)

    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    task = args.task
    if task == "auto":
        if args.data and args.data.endswith((".yaml", ".yml")):
            task = "detect"
        else:
            task = "classify"

    if task == "detect":
        if not args.data:
            raise RuntimeError("--data must point to a YAML file for detection datasets.")
        if args.model.endswith("-cls.pt") and rank == 0:
            print("[warn] detection task with a classification model; consider using yolov8n.pt.")
        loader, detected_classes = build_detection_loader(
            data_yaml=args.data,
            batch_size=args.batch_size,
            img_size=args.imgsz,
            rank=rank,
            world_size=world_size,
            workers=args.workers,
        )
        ensure_detection_hyp(model)
    else:
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
    if detected_classes and hasattr(model, "nc") and model.nc != detected_classes and rank == 0:
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
    if rank == 0 and len(buckets) % args.sync_interval != 0:
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
    criterion = torch.nn.CrossEntropyLoss() if task == "classify" else None

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

        batch = next(data_iter)
        if task == "detect":
            ensure_detection_hyp(ddp_model.module)
            batch = move_batch_to_device(batch, device)
            with ddp_model.no_sync():
                outputs = ddp_model(batch)
                loss = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                if torch.is_tensor(loss) and loss.ndim != 0:
                    loss = loss.mean()
                raw_loss = loss.detach()
                if not args.no_scale_loss:
                    loss = loss / float(args.sync_interval)
                loss.backward()
        else:
            images, labels = batch
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

        # Sync buckets round-robin: bucket i syncs when cycle_step == i % sync_interval.
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
            missing = sum(1 for was_synced in synced_in_cycle if not was_synced)
            if rank == 0 and missing:
                print(
                    f"[warn] cycle {cycle_id}: {missing} buckets not synced before update; "
                    "consider adjusting bucket_numel or sync_interval."
                )

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
