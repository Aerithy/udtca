import argparse
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from psgd.models.llama.llama_nn import LlamaConfig, MyLlamaForCausalLM, RMSNorm

# from experiments.common import (
from common import (
    CompressionConfig,
    append_summary_row,
    barrier,
    measure_throughput_tokens,
    save_loss_curve,
    sync_grads_bucketed,
)


class TokenizedDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_length=2048, text_field="text"):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text_field = text_field

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx][self.text_field]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_length + 1,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        if len(tokens) < 2:
            tokens = [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]

        valid_token_count = min(len(tokens), self.seq_length + 1)
        if len(tokens) > self.seq_length + 1:
            tokens = tokens[: self.seq_length + 1]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (
                self.seq_length + 1 - len(tokens)
            )

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        input_positions = torch.arange(self.seq_length, dtype=torch.long)
        attention_mask = (input_positions < min(valid_token_count, self.seq_length)).long()
        label_mask = input_positions < max(0, valid_token_count - 1)
        labels = labels.masked_fill(~label_mask, -100)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def get_dataloader(
    *,
    pp_size: int,
    dataset_name: str,
    dataset_config: str,
    tokenizer_name: str,
    seq_length: int,
    batch_size: int,
    num_workers: int,
    split: str,
    use_auth_token: bool,
    allow_download: bool,
    seed: int,
):
    download_cfg = DownloadConfig(local_files_only=not allow_download)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=False,
            trust_remote_code=False,
            use_auth_token=use_auth_token,
            local_files_only=not allow_download,
        )
    except OSError:
        print("[warn] tokenizer unavailable; using llama-tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            use_fast=False,
            local_files_only=not allow_download,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dataset_name == "c4":
        dataset_config = dataset_config or "en"
        dataset = load_dataset(
            "allenai/c4",
            dataset_config,
            split=split,
            streaming=False,
            download_config=download_cfg,
        )
        text_field = "text"
    else:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            download_config=download_cfg,
        )
        text_field = "text"

    tokenized_dataset = TokenizedDataset(
        dataset,
        tokenizer,
        seq_length=seq_length,
        text_field=text_field,
    )

    sampler = None
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            tokenized_dataset,
            num_replicas=dist.get_world_size() // pp_size,
            rank=dist.get_rank() // pp_size,
            shuffle=True,
            seed=seed,
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    return dataloader, tokenizer


def partition_llama_model(config: LlamaConfig, stage_idx: int, num_stages: int):
    with torch.device("meta"):
        model = MyLlamaForCausalLM(config)
        if dist.is_initialized() and dist.get_rank() == 0:
            total_params = sum(p.numel() for p in model.parameters()) / 1e9
            print(f"[rank 0] model params: {total_params:.2f}B")

    num_layers = config.num_hidden_layers
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    start_layer = stage_idx * layers_per_stage + min(stage_idx, remainder)
    end_layer = start_layer + layers_per_stage + (1 if stage_idx < remainder else 0)

    for i in list(model.model.layers.keys()):
        if not (start_layer <= int(i) < end_layer):
            del model.model.layers[i]

    if len(model.model.layers) == 0:
        import torch.nn as nn

        model.model.layers = nn.ModuleDict({"dummy": nn.Identity()})

    if stage_idx == 0:
        model.lm_head = None
        model.model.final_norm = None
    elif stage_idx == num_stages - 1:
        model.model.embed_tokens = None
    else:
        model.model.embed_tokens = None
        model.model.final_norm = None
        model.lm_head = None

    assigned_layers = [int(i) for i in model.model.layers.keys()]
    print(f"[partition] stage {stage_idx}: layers {assigned_layers}")
    return model


def init_llama_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def broadcast_stage_parameters(model: nn.Module, group) -> None:
    src_rank = dist.distributed_c10d.get_global_rank(group, 0)
    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank, group=group)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=src_rank, group=group)


def _method_dir(base: str, method: str) -> str:
    return os.path.join(base, "bitscom" if method == "bitscom" else "baselines")


def _compression_description(args: argparse.Namespace, method: str) -> str:
    descriptions = {
        "none": "No gradient compression; dense DP gradient all-reduce.",
        "quant8": "8-bit gradient quantization before DP synchronization.",
        "topk": f"Top-k sparsification with topk_ratio={args.topk_ratio}.",
        "powersgd": (
            "PowerSGD low-rank gradient compression "
            f"with rank={args.powersgd_rank}, dim={args.powersgd_dim}."
        ),
        "bitscom": (
            "bitscom LowBitGroup gradient compression "
            f"with bitwidth={args.bitwidth}."
        ),
    }
    return descriptions.get(method, f"Unknown compression method: {method}")


def _write_compression_metadata(
    writer: SummaryWriter,
    *,
    args: argparse.Namespace,
    method: str,
    dp_world: int,
    pp_size: int,
    micro_batches: int,
) -> None:
    selected_methods = ", ".join(args.methods)
    method_details = "\n".join(
        f"- {name}: {_compression_description(args, name)}"
        for name in args.methods
    )
    writer.add_text("compression/selected_methods", selected_methods, 0)
    writer.add_text("compression/selected_methods_detail", method_details, 0)
    writer.add_text(
        "compression/current_method",
        _compression_description(args, method),
        0,
    )
    writer.add_text(
        "run/config",
        "\n".join(
            [
                f"method: {method}",
                f"selected_methods: {selected_methods}",
                f"batch_size: {args.batch_size}",
                f"seq_length: {args.seq_length}",
                f"lr: {args.lr}",
                f"dp_world_size: {dp_world}",
                f"pp_size: {pp_size}",
                f"micro_batches: {micro_batches}",
                f"bucket_numel: {args.bucket_numel}",
                f"bitwidth: {args.bitwidth}",
                f"topk_ratio: {args.topk_ratio}",
                f"powersgd_rank: {args.powersgd_rank}",
                f"powersgd_dim: {args.powersgd_dim}",
            ]
        ),
        0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama7B DP+PP experiments")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--micro-batches", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--methods", nargs="+", default=["none", "quant8", "topk", "powersgd", "bitscom"])
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--powersgd-rank", type=int, default=2)
    parser.add_argument("--powersgd-dim", type=int, default=1024)
    parser.add_argument("--bucket-numel", type=int, default=4_000_000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    parser.add_argument("--smooth-window", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_reproducibility_seed(args.seed)

    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    pp_size = args.pp_size
    if world_size % pp_size != 0:
        raise RuntimeError(f"world_size {world_size} must be divisible by pp_size {pp_size}")

    if args.micro_batches < pp_size:
        args.micro_batches = pp_size
        if dist.get_rank() == 0:
            print(
                f"[warn] micro-batches < pp-size; bump to {args.micro_batches} to satisfy GPipe"
            )

    dp_size = world_size // pp_size
    device_mesh = init_device_mesh("cuda", (dp_size, pp_size), mesh_dim_names=("dp", "pp"))
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if "bitscom" in args.methods:
        import bitscom

        bitscom.init(bitwidth=args.bitwidth)
        lowbit_group = bitscom.LowBitGroup(bitwidth=args.bitwidth, process_group=dp_mesh.get_group())
    else:
        lowbit_group = None

    dataloader, tokenizer = get_dataloader(
        pp_size=pp_size,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=2,
        split="train",
        use_auth_token=args.use_auth_token,
        allow_download=not args.no_download,
        seed=args.seed,
    )

    def loss_fn(output, target):
        return F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            target.reshape(-1),
            ignore_index=-100,
        )

    dp_group = dp_mesh.get_group()
    dp_world = dist.get_world_size(dp_group)
    dp_rank = dp_mesh.get_local_rank()

    for method in args.methods:
        set_reproducibility_seed(args.seed)
        stage_idx = pp_mesh.get_local_rank()
        stage_model = partition_llama_model(
            LlamaConfig(
                vocab_size=32000,
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                rope_theta=10000.0,
                pad_token_id=0,
                tie_word_embeddings=True,
            ),
            stage_idx,
            pp_size,
        )
        stage_model.to_empty(device=device, recurse=True)
        stage_model.apply(init_llama_weights)
        broadcast_stage_parameters(stage_model, dp_group)

        stage = PipelineStage(
            stage_model,
            stage_index=stage_idx,
            num_stages=pp_size,
            device=device,
            group=pp_mesh.get_group(),
        )

        cfg = CompressionConfig(
            method,
            bitwidth=args.bitwidth,
            topk_ratio=args.topk_ratio,
            powersgd_rank=args.powersgd_rank,
            powersgd_dim=args.powersgd_dim,
        )

        optimizer = torch.optim.AdamW(
            stage.submod.parameters(),
            lr=args.lr,
            foreach=False,
        )
        micro_batches = min(int(args.micro_batches), int(args.batch_size))
        if micro_batches < pp_size:
            raise ValueError(
                "micro-batches must be >= pp-size; "
                "increase --batch-size or reduce --pp-size"
            )
        if micro_batches != args.micro_batches and dist.get_rank() == 0:
            print(
                f"[warn] micro-batches {args.micro_batches} > batch-size {args.batch_size}; "
                f"clamping to {micro_batches}"
            )
        schedule = Schedule1F1B(stage, n_microbatches=micro_batches, loss_fn=loss_fn)

        writer = None
        if stage.is_last and dp_rank == 0:
            tb_dir = os.path.join(
                _method_dir(args.out_dir, method),
                "tensorboard",
                f"llama7b_{method}",
            )
            writer = SummaryWriter(log_dir=tb_dir)
            _write_compression_metadata(
                writer,
                args=args,
                method=method,
                dp_world=dp_world,
                pp_size=pp_size,
                micro_batches=micro_batches,
            )

        global_step = 0
        losses: List[float] = []
        step_times: List[float] = []
        elapsed_times: List[float] = []

        steps_per_epoch = max(1, int(args.steps_per_epoch))
        run_t0 = time.perf_counter()
        for epoch in range(int(args.epochs)):
            step_in_epoch = 0
            if stage.is_last:
                pbar = tqdm(
                    dataloader,
                    desc=f"Llama7B {method} [epoch {epoch + 1}/{args.epochs}]",
                )
                epoch_iter = pbar
            else:
                epoch_iter = dataloader
                pbar = None

            for batch in epoch_iter:
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
                if step_in_epoch >= steps_per_epoch:
                    break

                t0 = time.perf_counter()
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device) if stage.is_last else None
                attention_mask = batch["attention_mask"].to(device)

                optimizer.zero_grad(set_to_none=True)

                if stage.is_first:
                    schedule.step(input_ids, attention_mask=attention_mask)
                elif stage.is_last:
                    step_losses: List[torch.Tensor] = []
                    schedule.step(target=labels, losses=step_losses, attention_mask=attention_mask)
                    loss = torch.stack(step_losses).mean()
                    loss_tensor = loss.detach().to(torch.float32)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM, group=dp_group)
                    loss_tensor.div_(dp_world)
                    losses.append(float(loss_tensor.item()))
                    if writer is not None:
                        writer.add_scalar(
                            f"Loss/train/{method}",
                            float(loss_tensor.item()),
                            global_step,
                        )
                    if pbar is not None:
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                else:
                    schedule.step(attention_mask=attention_mask)

                grads = [p.grad for p in stage.submod.parameters() if p.grad is not None]
                sync_grads_bucketed(
                    grads=grads,
                    group=dp_group,
                    world_size=dp_world,
                    cfg=cfg,
                    bucket_numel=args.bucket_numel,
                    lowbit_group=lowbit_group,
                )

                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        stage.submod.parameters(),
                        max_norm=float(args.grad_clip_norm),
                    )

                step_log_idx = global_step
                optimizer.step()
                global_step += 1
                step_in_epoch += 1

                if stage.is_last:
                    torch.cuda.synchronize(device)
                    step_time_s = time.perf_counter() - t0
                    step_times.append(step_time_s)
                    elapsed_times.append(time.perf_counter() - run_t0)
                    if writer is not None:
                        writer.add_scalar(
                            f"Time/step_ms/{method}",
                            step_time_s * 1000.0,
                            step_log_idx,
                        )
                        writer.add_scalar(
                            f"Throughput/tokens_per_s_step/{method}",
                            measure_throughput_tokens(
                                steps=1,
                                batch_size=args.batch_size,
                                seq_len=args.seq_length,
                                world_size=dp_world,
                                total_time_s=step_time_s,
                            ),
                            step_log_idx,
                        )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if stage.is_last and dp_rank == 0:
            total_time = sum(step_times)
            throughput = measure_throughput_tokens(
                steps=len(losses),
                batch_size=args.batch_size,
                seq_len=args.seq_length,
                world_size=dp_world,
                total_time_s=total_time,
            )

            out_dir = _method_dir(args.out_dir, method)
            run_name = f"llama7b_{method}"
            save_loss_curve(
                out_dir=out_dir,
                run_name=run_name,
                losses=losses,
                smooth_window=args.smooth_window,
                step_times_s=step_times,
                elapsed_times_s=elapsed_times,
            )

            summary_path = os.path.join(out_dir, "summary_llama7b.csv")
            append_summary_row(
                csv_path=summary_path,
                row={
                    "model": "llama7b",
                    "method": method,
                    "avg_step_time_ms": (total_time / max(1, len(losses))) * 1000.0,
                    "throughput_tokens_per_s": throughput,
                    "final_loss": losses[-1] if losses else 0.0,
                    "dp_world_size": dp_world,
                    "pp_size": pp_size,
                },
            )
            if writer is not None:
                writer.add_scalar(
                    "Summary/avg_step_time_ms",
                    (total_time / max(1, len(losses))) * 1000.0,
                    0,
                )
                writer.add_scalar("Summary/throughput_tokens_per_s", throughput, 0)
                writer.add_scalar("Summary/final_loss", losses[-1] if losses else 0.0, 0)
                writer.flush()

        if writer is not None:
            writer.close()

        barrier(dp_group)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
