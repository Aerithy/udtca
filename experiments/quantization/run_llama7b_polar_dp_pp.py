import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from psgd.models.llama.llama_nn import LlamaConfig, MyLlamaForCausalLM
from psgd.parallelism.polar.wrapper import PolarParallel

from common import CompressionConfig, sync_grads_bucketed


class TokenizedDataset(Dataset):
    """Tokenize text samples into fixed-length causal LM inputs."""
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

        if len(tokens) > self.seq_length + 1:
            tokens = tokens[: self.seq_length + 1]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (
                self.seq_length + 1 - len(tokens)
            )

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
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
):
    """Build a dataloader with optional distributed sampling for DP+PP."""
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
        )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader, tokenizer


def partition_llama_model(config: LlamaConfig, stage_idx: int, num_stages: int):
    """Partition Llama layers across pipeline stages; add a dummy layer if empty."""
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
        model.model.layers = torch.nn.ModuleDict(
            {"empty_stage_placeholder": torch.nn.Identity()}
        )

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


class CompressedPolarParallel(PolarParallel):
    """PolarParallel variant that syncs DP gradients with compression.

    Relies on PolarParallel's args namespace for logging configuration.
    """
    def __init__(
        self,
        *,
        compression_cfg: CompressionConfig,
        bucket_numel: int,
        lowbit_group,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compression_cfg = compression_cfg
        self.bucket_numel = int(bucket_numel)
        self.lowbit_group = lowbit_group

    def _allreduce_dp_grads_(self):
        dp_group = self.dp_mesh.get_group()
        dp_size = self.dp_mesh.size()
        if dp_size <= 1:
            return

        grads = [p.grad for p in self.stage.submod.parameters() if p.grad is not None]
        sync_grads_bucketed(
            grads=grads,
            group=dp_group,
            world_size=dp_size,
            cfg=self.compression_cfg,
            bucket_numel=self.bucket_numel,
            lowbit_group=self.lowbit_group,
        )


def main():
    parser = argparse.ArgumentParser(description="Llama7B DP+PP (Polar + bitscom)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--seq-length", "--seq_length", dest="seq_length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset-config", "--dataset_config", dest="dataset_config", type=str,
                        default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--pp-size", "--pp_size", dest="pp_size", type=int, default=1)
    parser.add_argument("--micro-batches", "--micro_batches", dest="micro_batches", type=int, default=4)
    parser.add_argument("--comm-timing", type=int, default=-1)
    parser.add_argument(
        "--train-mode",
        choices=["baseline", "polar"],
        default="baseline",
        help="Use baseline DP sync or polar hooks; compression is only available with baseline.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["manual", "ddp"],
        default="manual",
        help="Baseline DP sync strategy when train-mode=baseline.",
    )
    parser.add_argument("--max-steps", type=int, default=200)

    parser.add_argument("--method", type=str, default="bitscom",
                        choices=["none", "quant8", "topk", "powersgd", "bitscom"])
    parser.add_argument("--bitwidth", type=int, default=4)
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--powersgd-rank", type=int, default=2)
    parser.add_argument("--powersgd-dim", type=int, default=1024)
    parser.add_argument("--bucket-numel", type=int, default=4_000_000)
    parser.add_argument("--simulate-quantization", action="store_true")
    parser.add_argument("--stochastic-rounding", action="store_true")

    parser.add_argument("--eval-split", type=str, default="")
    parser.add_argument("--train-val-ratio", type=float, default=0.0)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-max-batches", type=int, default=20)

    args = parser.parse_args()
    args.using_polar = args.train_mode == "polar"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.method != "none" and args.train_mode == "polar":
        raise ValueError(
            "--train-mode=polar does not support compressed DP sync; "
            "use --train-mode=baseline with compression methods"
        )
    if args.method != "none" and args.baseline_mode == "ddp":
        raise ValueError(
            "--baseline-mode=ddp does not support compressed DP sync; "
            "use --baseline-mode=manual with compression methods"
        )

    bitscom_module = None
    if args.method == "bitscom":
        import bitscom as bitscom_module

        bitscom_module.init(bitwidth=args.bitwidth)

    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()

    if world_size % args.pp_size != 0:
        raise RuntimeError(
            f"world_size {world_size} must be divisible by pp_size {args.pp_size}"
        )

    if args.micro_batches < args.pp_size:
        if dist.get_rank() == 0:
            print(
                f"[warn] micro-batches < pp-size; bump to {args.pp_size} to satisfy GPipe"
            )
        args.micro_batches = args.pp_size

    micro_batches = min(args.micro_batches, args.batch_size)
    if micro_batches < args.pp_size:
        raise ValueError(
            "micro-batches must be >= pp-size; "
            "increase --batch-size or reduce --pp-size"
        )
    if micro_batches != args.micro_batches and dist.get_rank() == 0:
        print(
            f"[warn] micro-batches {args.micro_batches} > batch-size {args.batch_size}; "
            f"clamping to {micro_batches}"
        )
    args.micro_batches = micro_batches

    dp_size = world_size // args.pp_size
    device_mesh = init_device_mesh("cuda", (dp_size, args.pp_size), mesh_dim_names=("dp", "pp"))
    dp_mesh = device_mesh["dp"]
    pp_mesh = device_mesh["pp"]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    lowbit_group = None
    if args.method == "bitscom":
        lowbit_group = bitscom_module.LowBitGroup(
            bitwidth=args.bitwidth,
            process_group=dp_mesh.get_group(),
            simulate_quantization=args.simulate_quantization,
            stochastic_rounding=args.stochastic_rounding,
        )

    dataloader, tokenizer = get_dataloader(
        pp_size=args.pp_size,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=2,
        split="train",
        use_auth_token=args.use_auth_token,
        allow_download=not args.no_download,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    eval_dataloader: Optional[DataLoader] = None
    if args.eval_split:
        eval_dataloader, _ = get_dataloader(
            pp_size=args.pp_size,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            tokenizer_name=args.tokenizer,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            num_workers=2,
            split=args.eval_split,
            use_auth_token=args.use_auth_token,
            allow_download=not args.no_download,
        )
    elif args.train_val_ratio and args.train_val_ratio > 0:
        from torch.utils.data import random_split

        n = len(dataloader.dataset)
        n_val = max(1, int(n * float(args.train_val_ratio)))
        n_train = n - n_val
        train_ds, val_ds = random_split(dataloader.dataset, [n_train, n_val])

        dataloader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        eval_dataloader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    def loss_fn(output, target):
        shift_logits = output[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=pad_token_id,
        )

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
        args.pp_size,
    )

    compression_cfg = CompressionConfig(
        args.method,
        bitwidth=args.bitwidth,
        topk_ratio=args.topk_ratio,
        powersgd_rank=args.powersgd_rank,
        powersgd_dim=args.powersgd_dim,
        simulate_quantization=args.simulate_quantization,
        stochastic_rounding=args.stochastic_rounding,
    )

    trainer = CompressedPolarParallel(
        args=args,
        device_mesh=device_mesh,
        micro_batches=args.micro_batches,
        loss_fn=loss_fn,
        stage_model=stage_model,
        dataloader=dataloader,
        comm_timing=args.comm_timing,
        eval_dataloader=eval_dataloader,
        eval_interval=args.eval_interval,
        eval_max_batches=args.eval_max_batches,
        use_local_sgd=False,
        local_sgd_steps=1,
        baseline_mode=args.baseline_mode,
        compression_cfg=compression_cfg,
        bucket_numel=args.bucket_numel,
        lowbit_group=lowbit_group,
    )

    if args.train_mode == "polar":
        trainer.train()
    else:
        trainer._train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
