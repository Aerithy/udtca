import argparse
import math
import os
import time
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

from experiments.common import (
    CompressionConfig,
    append_summary_row,
    barrier,
    measure_throughput_samples,
    measure_throughput_tokens,
    save_loss_curve,
    step_timer,
    sync_grads_bucketed,
    sync_loss_across_ranks,
)


def _build_resnet50_model(num_classes: int):
    import torchvision.models as tvm

    return tvm.resnet50(num_classes=num_classes)


def _build_bert_model(num_labels: int, hidden_size: int, num_layers: int, num_heads: int):
    from transformers import BertConfig, BertForSequenceClassification

    cfg = BertConfig(
        vocab_size=30522,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=512,
        num_labels=num_labels,
    )
    return BertForSequenceClassification(cfg)


def _build_gpt2_model(hidden_size: int, num_layers: int, num_heads: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=50257,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=512,
        n_ctx=512,
    )
    return GPT2LMHeadModel(cfg)


class BertSST2Dataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

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
            "labels": torch.tensor(row["label"], dtype=torch.int64),
        }


class GPT2WikitextDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[int(idx)]
        encoded = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


def _limit_dataset(dataset, dataset_size: int):
    if dataset_size <= 0:
        return dataset
    return dataset.select(range(min(dataset_size, len(dataset))))


def _build_resnet_dataloader(
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


def _build_bert_dataloader(
    *,
    dataset_size: int,
    seq_len: int,
    allow_download: bool,
    batch_size: int,
    rank: int,
    world_size: int,
):
    download_cfg = DownloadConfig(local_files_only=not allow_download)
    raw = load_dataset(
        "glue",
        "sst2",
        split="train",
        download_config=download_cfg,
    )
    raw = _limit_dataset(raw, dataset_size)
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        local_files_only=not allow_download,
    )

    dataset = BertSST2Dataset(raw, tokenizer, seq_len)
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


def _build_gpt2_dataloader(
    *,
    dataset_size: int,
    seq_len: int,
    allow_download: bool,
    batch_size: int,
    rank: int,
    world_size: int,
):
    download_cfg = DownloadConfig(local_files_only=not allow_download)
    raw = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
        download_config=download_cfg,
    )
    raw = _limit_dataset(raw, dataset_size)
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        local_files_only=not allow_download,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = GPT2WikitextDataset(raw, tokenizer, seq_len)
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


def _run_train(
    *,
    model_name: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    steps: int,
    batch_size: int,
    grad_clip_norm: float,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    min_lr_ratio: float,
    cfg: CompressionConfig,
    bucket_numel: int,
    world_size: int,
    group,
    lowbit_group,
    device: torch.device,
    seq_len: int,
) -> Dict[str, object]:
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    warmup_steps = max(1, int(steps * warmup_ratio))

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress = (step_idx - warmup_steps) / max(1, steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    record_step, step_times = step_timer()
    losses: List[float] = []
    data_iter = _infinite_loader(dataloader)

    torch.cuda.synchronize(device)

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        batch = next(data_iter)

        optimizer.zero_grad(set_to_none=True)

        if model_name == "resnet50":
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        elif model_name == "bert":
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out.loss
        elif model_name == "gpt2":
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = out.loss
        else:
            raise ValueError(f"unsupported model: {model_name}")

        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        sync_grads_bucketed(
            grads=grads,
            group=group,
            world_size=world_size,
            cfg=cfg,
            bucket_numel=bucket_numel,
            lowbit_group=lowbit_group,
        )

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        scheduler.step()

        mean_loss = sync_loss_across_ranks(loss, group)
        losses.append(mean_loss)

        torch.cuda.synchronize(device)
        record_step(time.perf_counter() - t0)

    total_time = sum(step_times)
    if model_name == "resnet50":
        throughput = measure_throughput_samples(
            steps=steps,
            batch_size=batch_size,
            world_size=world_size,
            total_time_s=total_time,
        )
    else:
        throughput = measure_throughput_tokens(
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            world_size=world_size,
            total_time_s=total_time,
        )

    return {
        "losses": losses,
        "avg_step_time_ms": (total_time / max(1, steps)) * 1000.0,
        "throughput": throughput,
    }


def _method_dir(base: str, method: str) -> str:
    return os.path.join(base, "bitscom" if method == "bitscom" else "baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model compression experiments")
    parser.add_argument("--models", nargs="+", default=["resnet50", "bert", "gpt2"])
    parser.add_argument("--methods", nargs="+", default=["none", "quant8", "topk", "powersgd", "bitscom"])
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dataset-size", type=int, default=512)
    parser.add_argument("--resnet-steps", type=int, default=60)
    parser.add_argument("--resnet-batch-size", type=int, default=16)
    parser.add_argument("--resnet-dataset-size", type=int, default=512)
    parser.add_argument("--nlp-steps", type=int, default=120)
    parser.add_argument("--nlp-batch-size", type=int, default=4)
    parser.add_argument("--nlp-dataset-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--bucket-numel", type=int, default=4_000_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-seed", type=int, default=17)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=4)
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--powersgd-rank", type=int, default=2)
    parser.add_argument("--powersgd-dim", type=int, default=1024)
    parser.add_argument("--resnet-image-size", type=int, default=224)
    parser.add_argument("--resnet-num-classes", type=int, default=20)
    parser.add_argument("--bert-seq-len", type=int, default=64)
    parser.add_argument("--bert-hidden", type=int, default=256)
    parser.add_argument("--bert-layers", type=int, default=4)
    parser.add_argument("--bert-heads", type=int, default=4)
    parser.add_argument("--bert-num-labels", type=int, default=2)
    parser.add_argument("--gpt2-seq-len", type=int, default=64)
    parser.add_argument("--gpt2-hidden", type=int, default=256)
    parser.add_argument("--gpt2-layers", type=int, default=4)
    parser.add_argument("--gpt2-heads", type=int, default=4)
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
        lowbit_group = bitscom.LowBitGroup(bitwidth=args.bitwidth, process_group=dist.group.WORLD)
    else:
        lowbit_group = None

    try:
        for model_name in args.models:
            is_nlp = model_name in {"bert", "gpt2"}
            steps = args.nlp_steps if is_nlp else args.resnet_steps
            batch_size = args.nlp_batch_size if is_nlp else args.resnet_batch_size
            dataset_size = args.nlp_dataset_size if is_nlp else args.resnet_dataset_size
            seq_len = args.bert_seq_len if model_name == "bert" else args.gpt2_seq_len

            allow_download = not args.no_download

            if model_name == "resnet50":
                dataloader = _build_resnet_dataloader(
                    dataset_size=dataset_size,
                    image_size=args.resnet_image_size,
                    allow_download=allow_download,
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                )
            elif model_name == "bert":
                dataloader = _build_bert_dataloader(
                    dataset_size=dataset_size,
                    seq_len=args.bert_seq_len,
                    allow_download=allow_download,
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                )
            elif model_name == "gpt2":
                dataloader = _build_gpt2_dataloader(
                    dataset_size=dataset_size,
                    seq_len=args.gpt2_seq_len,
                    allow_download=allow_download,
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                )
            else:
                raise ValueError(model_name)

            for method in args.methods:
                if model_name == "resnet50":
                    model = _build_resnet50_model(args.resnet_num_classes)
                elif model_name == "bert":
                    model = _build_bert_model(
                        args.bert_num_labels,
                        args.bert_hidden,
                        args.bert_layers,
                        args.bert_heads,
                    )
                elif model_name == "gpt2":
                    model = _build_gpt2_model(args.gpt2_hidden, args.gpt2_layers, args.gpt2_heads)
                else:
                    raise ValueError(model_name)

                cfg = CompressionConfig(
                    method,
                    bitwidth=args.bitwidth,
                    topk_ratio=args.topk_ratio,
                    powersgd_rank=args.powersgd_rank,
                    powersgd_dim=args.powersgd_dim,
                )

                if rank == 0:
                    print(
                        f"[run] model={model_name} method={method} steps={steps} "
                        f"batch={batch_size} dataset={dataset_size}"
                    )

                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)

                run = _run_train(
                    model_name=model_name,
                    model=model,
                    dataloader=dataloader,
                    steps=steps,
                    batch_size=batch_size,
                    grad_clip_norm=args.grad_clip_norm,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    warmup_ratio=args.warmup_ratio,
                    min_lr_ratio=args.min_lr_ratio,
                    cfg=cfg,
                    bucket_numel=args.bucket_numel,
                    world_size=world_size,
                    group=dist.group.WORLD,
                    lowbit_group=lowbit_group,
                    device=device,
                    seq_len=seq_len,
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
                            "throughput": run["throughput"],
                            "final_loss": run["losses"][-1] if run["losses"] else 0.0,
                            "world_size": world_size,
                        },
                    )

                barrier(dist.group.WORLD)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
