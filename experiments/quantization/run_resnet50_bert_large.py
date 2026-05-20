import argparse
import math
import os
import time
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

from common import (
    CompressionConfig,
    append_summary_row,
    barrier,
    measure_throughput_samples,
    measure_throughput_tokens,
    save_loss_curve,
    sync_grads_bucketed,
    sync_loss_across_ranks,
)


class BertSST2Dataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[int(idx)]
        encoded = self.tokenizer(
            row["sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


class SyntheticBertDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_size: int,
        seq_len: int,
        vocab_size: int,
        num_labels: int,
        seed: int,
    ):
        self.dataset_size = int(dataset_size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.num_labels = int(num_labels)
        self.seed = int(seed)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(idx))
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            dtype=torch.long,
            generator=generator,
        )
        input_ids[0] = 101
        if self.seq_len > 1:
            input_ids[-1] = 102
        label = torch.randint(
            low=0,
            high=self.num_labels,
            size=(),
            dtype=torch.long,
            generator=generator,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": label,
        }


def _limit_hf_dataset(dataset, dataset_size: int):
    if dataset_size <= 0:
        return dataset
    return dataset.select(range(min(dataset_size, len(dataset))))


def _build_resnet50_model(num_classes: int):
    import torchvision.models as tvm

    return tvm.resnet50(weights=None, num_classes=num_classes)


def _build_bert_large_model(num_labels: int):
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=30522,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=512,
        type_vocab_size=2,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(config)


def _build_resnet50_dataloader(
    *,
    dataset_size: int,
    image_size: int,
    allow_download: bool,
    batch_size: int,
    rank: int,
    world_size: int,
):
    import torchvision
    import torchvision.transforms as T

    root = os.path.join(os.path.dirname(__file__), "..", ".cache", "cifar10")
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=True,
        transform=transform,
        download=allow_download,
    )
    if dataset_size > 0:
        dataset = Subset(dataset, list(range(min(dataset_size, len(dataset)))))

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


def _build_bert_large_dataloader(
    *,
    dataset_size: int,
    seq_len: int,
    allow_download: bool,
    batch_size: int,
    rank: int,
    world_size: int,
    data_source: str,
    num_labels: int,
    seed: int,
):
    if data_source not in {"auto", "glue", "synthetic"}:
        raise ValueError(f"unsupported bert data source: {data_source}")

    dataset = None
    if data_source in {"auto", "glue"}:
        try:
            download_cfg = DownloadConfig(local_files_only=not allow_download)
            raw = load_dataset(
                "glue",
                "sst2",
                split="train",
                download_config=download_cfg,
            )
            raw = _limit_hf_dataset(raw, dataset_size)
            tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased",
                local_files_only=not allow_download,
            )
            dataset = BertSST2Dataset(raw, tokenizer, seq_len)
        except Exception as exc:
            if data_source == "glue":
                raise
            if rank == 0:
                print(
                    "[warn] GLUE/SST-2 is unavailable; falling back to "
                    f"synthetic BERT data. Original error: {exc}",
                    flush=True,
                )

    if dataset is None:
        dataset = SyntheticBertDataset(
            dataset_size=dataset_size,
            seq_len=seq_len,
            vocab_size=30522,
            num_labels=num_labels,
            seed=seed,
        )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _method_dir(base: str, method: str) -> str:
    return os.path.join(base, "bitscom" if method == "bitscom" else "baselines")


def _build_model(model_name: str, args: argparse.Namespace):
    if model_name == "resnet50":
        return _build_resnet50_model(args.resnet_num_classes)
    if model_name == "bert-large":
        return _build_bert_large_model(args.bert_num_labels)
    raise ValueError(f"unsupported model: {model_name}")


def _build_dataloader(
    *,
    model_name: str,
    args: argparse.Namespace,
    batch_size: int,
    dataset_size: int,
    rank: int,
    world_size: int,
):
    allow_download = not args.no_download
    if model_name == "resnet50":
        return _build_resnet50_dataloader(
            dataset_size=dataset_size,
            image_size=args.resnet_image_size,
            allow_download=allow_download,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
        )
    if model_name == "bert-large":
        return _build_bert_large_dataloader(
            dataset_size=dataset_size,
            seq_len=args.bert_seq_len,
            allow_download=allow_download,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            data_source=args.bert_data_source,
            num_labels=args.bert_num_labels,
            seed=args.data_seed,
        )
    raise ValueError(f"unsupported model: {model_name}")


def _model_run_shape(model_name: str, args: argparse.Namespace) -> Dict[str, int]:
    if model_name == "resnet50":
        return {
            "steps": int(args.resnet_steps),
            "batch_size": int(args.resnet_batch_size),
            "dataset_size": int(args.resnet_dataset_size),
            "seq_len": 0,
        }
    if model_name == "bert-large":
        return {
            "steps": int(args.bert_steps),
            "batch_size": int(args.bert_batch_size),
            "dataset_size": int(args.bert_dataset_size),
            "seq_len": int(args.bert_seq_len),
        }
    raise ValueError(f"unsupported model: {model_name}")


def _model_train_hparams(model_name: str, args: argparse.Namespace) -> Dict[str, float]:
    if model_name == "resnet50":
        return {
            "lr": float(args.resnet_lr),
            "warmup_ratio": float(args.resnet_warmup_ratio),
            "min_lr_ratio": float(args.min_lr_ratio),
            "grad_clip_norm": float(args.resnet_grad_clip_norm),
            "weight_decay": float(args.resnet_weight_decay),
        }
    if model_name == "bert-large":
        return {
            "lr": float(args.bert_lr),
            "warmup_ratio": float(args.bert_warmup_ratio),
            "min_lr_ratio": float(args.min_lr_ratio),
            "grad_clip_norm": float(args.bert_grad_clip_norm),
            "weight_decay": float(args.bert_weight_decay),
        }
    raise ValueError(f"unsupported model: {model_name}")


def _run_step(
    *,
    model_name: str,
    model: torch.nn.Module,
    batch,
    device: torch.device,
) -> torch.Tensor:
    if model_name == "resnet50":
        inputs, targets = batch
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        return F.cross_entropy(logits, targets)

    if model_name == "bert-large":
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss

    raise ValueError(f"unsupported model: {model_name}")


def _finite_grads(parameters: Iterable[torch.nn.Parameter]) -> bool:
    for p in parameters:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _run_train(
    *,
    model_name: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    steps: int,
    batch_size: int,
    seq_len: int,
    hparams: Dict[str, float],
    cfg: CompressionConfig,
    args: argparse.Namespace,
    world_size: int,
    group,
    lowbit_group,
    device: torch.device,
) -> Dict[str, object]:
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=hparams["weight_decay"],
    )
    warmup_steps = max(1, int(steps * hparams["warmup_ratio"]))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_ratio = hparams["min_lr_ratio"]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    data_iter = _infinite_loader(dataloader)
    losses: List[float] = []
    step_times: List[float] = []

    torch.cuda.synchronize(device)
    for _step in range(steps):
        t0 = time.perf_counter()
        batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)
        loss = _run_step(
            model_name=model_name,
            model=model,
            batch=batch,
            device=device,
        )
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        sync_grads_bucketed(
            grads=grads,
            group=group,
            world_size=world_size,
            cfg=cfg,
            bucket_numel=args.bucket_numel,
            lowbit_group=lowbit_group,
        )

        if not _finite_grads(model.parameters()):
            optimizer.zero_grad(set_to_none=True)
        else:
            if hparams["grad_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    hparams["grad_clip_norm"],
                )
            optimizer.step()
            scheduler.step()

        mean_loss = sync_loss_across_ranks(loss, group)
        losses.append(mean_loss)

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

    total_time_s = sum(step_times)
    if model_name == "resnet50":
        throughput = measure_throughput_samples(
            steps=steps,
            batch_size=batch_size,
            world_size=world_size,
            total_time_s=total_time_s,
        )
        throughput_name = "throughput_samples_per_s"
    else:
        throughput = measure_throughput_tokens(
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            world_size=world_size,
            total_time_s=total_time_s,
        )
        throughput_name = "throughput_tokens_per_s"

    return {
        "losses": losses,
        "avg_step_time_ms": (total_time_s / max(1, steps)) * 1000.0,
        "throughput": throughput,
        "throughput_name": throughput_name,
        "lr": hparams["lr"],
        "warmup_ratio": hparams["warmup_ratio"],
        "grad_clip_norm": hparams["grad_clip_norm"],
        "weight_decay": hparams["weight_decay"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResNet50 and BERT-large compression experiments"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "bert-large"],
        choices=["resnet50", "bert-large"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "quant8", "topk", "powersgd", "bitscom"],
        choices=["none", "quant8", "topk", "powersgd", "bitscom"],
    )
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--powersgd-rank", type=int, default=2)
    parser.add_argument("--powersgd-dim", type=int, default=1024)
    parser.add_argument("--bucket-numel", type=int, default=4_000_000)

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
    parser.add_argument("--bert-batch-size", type=int, default=1)
    parser.add_argument("--bert-dataset-size", type=int, default=1024)
    parser.add_argument("--bert-seq-len", type=int, default=128)
    parser.add_argument("--bert-num-labels", type=int, default=2)
    parser.add_argument("--bert-lr", type=float, default=5e-6)
    parser.add_argument("--bert-weight-decay", type=float, default=1e-2)
    parser.add_argument("--bert-warmup-ratio", type=float, default=0.15)
    parser.add_argument("--bert-grad-clip-norm", type=float, default=0.5)
    parser.add_argument(
        "--bert-data-source",
        choices=["auto", "glue", "synthetic"],
        default="auto",
        help="Use GLUE/SST-2, synthetic token data, or auto fallback.",
    )

    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if "bitscom" in args.methods:
        import bitscom

        bitscom.init(bitwidth=args.bitwidth)
        lowbit_group = bitscom.LowBitGroup(
            bitwidth=args.bitwidth,
            process_group=dist.group.WORLD,
        )
    else:
        lowbit_group = None

    try:
        for model_name in args.models:
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

            for method in args.methods:
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                model = _build_model(model_name, args)

                cfg = CompressionConfig(
                    method,
                    bitwidth=args.bitwidth,
                    topk_ratio=args.topk_ratio,
                    powersgd_rank=args.powersgd_rank,
                    powersgd_dim=args.powersgd_dim,
                )

                if rank == 0:
                    print(
                        f"[run] model={model_name} method={method} "
                        f"steps={shape['steps']} batch={shape['batch_size']} "
                        f"dataset={shape['dataset_size']} lr={hparams['lr']} "
                        f"warmup={hparams['warmup_ratio']} "
                        f"clip={hparams['grad_clip_norm']}"
                    )

                run = _run_train(
                    model_name=model_name,
                    model=model,
                    dataloader=dataloader,
                    steps=shape["steps"],
                    batch_size=shape["batch_size"],
                    seq_len=shape["seq_len"],
                    hparams=hparams,
                    cfg=cfg,
                    args=args,
                    world_size=world_size,
                    group=dist.group.WORLD,
                    lowbit_group=lowbit_group,
                    device=device,
                )

                if rank == 0:
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
                            "final_loss": run["losses"][-1] if run["losses"] else 0.0,
                            "world_size": world_size,
                            "batch_size": shape["batch_size"],
                            "steps": shape["steps"],
                            "bucket_numel": args.bucket_numel,
                            "lr": run["lr"],
                            "warmup_ratio": run["warmup_ratio"],
                            "grad_clip_norm": run["grad_clip_norm"],
                            "weight_decay": run["weight_decay"],
                        },
                    )

                barrier(dist.group.WORLD)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
