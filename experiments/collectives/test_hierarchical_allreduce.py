#!/usr/bin/env python3
"""
测试 hierarchical all-reduce 和 pipelined hierarchical all-reduce 的正确性与吞吐量

运行方式:
    # 单节点测试（验证实现逻辑）
    torchrun --nproc_per_node=4 experiments/collectives/test_hierarchical_allreduce.py
    
    # 两机测试（需要预先配置好环境）
    # Node 0:
    # torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    #     --master_addr=<IP> --master_port=29500 \
    #     experiments/collectives/test_hierarchical_allreduce.py
    # Node 1:
    # torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    #     --master_addr=<IP> --master_port=29500 \
    #     experiments/collectives/test_hierarchical_allreduce.py
"""

import os
import time
import json
from datetime import timedelta
import torch
import torch.distributed as dist
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'bitscom', 'python'))
import bitscom


def _env_int(name, default=None):
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise RuntimeError(f"Missing required distributed env var: {name}")
        return default
    return int(value)


def _dist_env_snapshot():
    keys = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "GROUP_RANK",
        "GPUS_PER_NODE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_SOCKET_IFNAME",
    )
    return {key: os.environ.get(key) for key in keys}


def init_distributed():
    """初始化分布式环境"""
    local_rank = _env_int("LOCAL_RANK", 0)
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        print(
            f"[dist] init_process_group start env={_dist_env_snapshot()}",
            flush=True,
        )
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=10),
        )
        print("[dist] init_process_group done", flush=True)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_world_size = _env_int("LOCAL_WORLD_SIZE", torch.cuda.device_count())
    gpus_per_node = _env_int("GPUS_PER_NODE", local_world_size)
    node_id = _env_int("NODE_RANK", _env_int("GROUP_RANK", rank // gpus_per_node))

    if os.environ.get("GPUS_PER_NODE") and gpus_per_node != local_world_size:
        raise RuntimeError(
            "GPUS_PER_NODE must match LOCAL_WORLD_SIZE for this test. "
            f"env={_dist_env_snapshot()}"
        )
    if world_size % gpus_per_node != 0:
        raise RuntimeError(
            f"WORLD_SIZE ({world_size}) must be divisible by GPUS_PER_NODE "
            f"({gpus_per_node}). env={_dist_env_snapshot()}"
        )
    if os.environ.get("GPUS_PER_NODE") is None and world_size != 8:
        raise RuntimeError(
            "Expected WORLD_SIZE=8 for the default 2-node/4-GPU run. "
            "Set GPUS_PER_NODE explicitly to run a different topology. "
            f"env={_dist_env_snapshot()}"
        )
    if os.environ.get("GPUS_PER_NODE") is None and local_world_size != 4:
        raise RuntimeError(
            "Expected LOCAL_WORLD_SIZE=4 for the default 2-node/4-GPU run. "
            "Set GPUS_PER_NODE explicitly to run a different topology. "
            f"env={_dist_env_snapshot()}"
        )

    return rank, world_size, local_rank, node_id, local_world_size, gpus_per_node


def log_barrier(label, rank, node_id, local_rank):
    current_device = torch.cuda.current_device()
    prefix = (
        f"[Rank {rank} Node {node_id} Local {local_rank} "
        f"CUDA {current_device}]"
    )
    print(f"{prefix} before {label}", flush=True)
    dist.barrier(device_ids=[local_rank])
    print(f"{prefix} after {label}", flush=True)


def build_hierarchical_groups(rank, world_size, gpus_per_node=None):
    """构建本地组和节点间组"""
    if gpus_per_node is None:
        gpus_per_node = _env_int("GPUS_PER_NODE", _env_int("LOCAL_WORLD_SIZE", 4))
    num_nodes = world_size // gpus_per_node
    
    node_id = rank // gpus_per_node
    
    # 本地组：同节点内的所有GPU
    # local_ranks = [node_id * gpus_per_node + i for i in range(gpus_per_node)]
    # print(f"[Rank {rank}] Node ID: {node_id}, Local ranks: {local_ranks}")
    # local_group = dist.new_group(local_ranks)
    
    groups = []
    for i in range(num_nodes):
        local_ranks = [i * gpus_per_node + j for j in range(gpus_per_node)]
        groups.append(dist.new_group(local_ranks))
        # if i == node_id:
        #     print(f"[Rank {rank}] Local group ranks: {local_ranks}")
    
    # 节点间组：每个节点的rank 0作为代表
    inter_ranks = [i * gpus_per_node for i in range(num_nodes)]
    print(f"[Rank {rank}] Inter-group ranks: {inter_ranks}", flush=True)
    inter_group = dist.new_group(inter_ranks)
    
    return groups[node_id], inter_group, gpus_per_node, num_nodes


def test_correctness(rank, world_size, local_rank, device, gpus_per_node):
    """测试 hierarchical all-reduce 的数值正确性"""
    # device = torch.device(f"cuda:{rank}")
    # torch.cuda.set_device(device)
    
    local_group, inter_group, gpus_per_node, num_nodes = build_hierarchical_groups(
        rank,
        world_size,
        gpus_per_node,
    )
    
    # 创建测试张量
    tensor = torch.ones(1024, device=device) * (rank + 1)
    
    # 使用 hierarchical all-reduce（非流水线）
    group = bitscom.LowBitGroup(bitwidth=4)
    group.all_reduce(
        tensor,
        local_group=local_group,
        inter_group=inter_group,
        chunk_size=1024,  # 禁用流水线
        local_quantize=False
    )
    
    # 计算期望值
    expected_val = sum(range(1, world_size + 1))
    expected = torch.ones(1024, device=device) * expected_val
    
    max_error = torch.abs(tensor - expected).max().item()
    
    # 测试 pipelined hierarchical all-reduce
    tensor_pipe = torch.ones(4096, device=device) * (rank + 1)
    group.all_reduce(
        tensor_pipe,
        local_group=local_group,
        inter_group=inter_group,
        chunk_size=512,  # 启用流水线
        local_quantize=False
    )
    expected_pipe = torch.ones(4096, device=device) * expected_val
    max_error_pipe = torch.abs(tensor_pipe - expected_pipe).max().item()
    
    if rank == 0:
        print(f"[Correctness] Hierarchical max error: {max_error:.6f}")
        print(f"[Correctness] Pipelined max error: {max_error_pipe:.6f}")
        print(f"[Correctness] Test passed: {max_error < 1.0 and max_error_pipe < 1.0}")
    
    return max_error < 1.0 and max_error_pipe < 1.0


def test_single_node_fallback(rank, world_size, local_rank):
    """测试单节点场景下是否正确fallback到全精度"""
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # 模拟单节点：local_group = WORLD
    local_group = dist.new_group(list(range(world_size)))
    
    tensor = torch.ones(1024, device=device) * (rank + 1)
    
    group = bitscom.LowBitGroup(bitwidth=4)
    group.all_reduce(
        tensor,
        local_group=local_group,
        inter_group=None,  # 单节点场景
        local_quantize=False
    )
    
    expected_val = sum(range(1, world_size + 1))
    expected = torch.ones(1024, device=device) * expected_val
    
    max_error = torch.abs(tensor - expected).max().item()
    
    if rank == 0:
        print(f"[Single Node] Max error: {max_error:.6f}")
        print(f"[Single Node] Test passed: {max_error < 1e-5}")
    
    return max_error < 1e-5


def benchmark_throughput(rank, world_size, local_rank, device, gpus_per_node):
    """吞吐量基准测试"""
    # device = torch.device(f"cuda:{local_rank}")
    # torch.cuda.set_device(device)
    
    local_group, inter_group, gpus_per_node, num_nodes = build_hierarchical_groups(
        rank,
        world_size,
        gpus_per_node,
    )
    
    # 测试不同张量大小
    tensor_sizes = [
        1 << 20,   # 4MB
        1 << 22,   # 16MB
        1 << 24,   # 64MB
        1 << 25,   # 128MB
    ]
    
    warmup = 5
    iterations = 20
    
    results = []
    
    for numel in tensor_sizes:
        if rank == 0:
            print(f"\n[Benchmark] Testing {numel/1e6:.1f}M elements ({numel*4/1e3:.1f}MB)")
        
        # NCCL baseline
        times = []
        for i in range(warmup + iterations):
            t = torch.randn(numel, device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(t)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - start)
        nccl_time = sum(times) / len(times)
        nccl_tp = (numel * 4 / 1e9) / nccl_time
        
        # Hierarchical all-reduce (non-pipelined)
        times = []
        for i in range(warmup + iterations):
            t = torch.randn(numel, device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            start = time.perf_counter()
            group = bitscom.LowBitGroup(bitwidth=4)
            group.all_reduce(t, local_group=local_group, inter_group=inter_group, 
                           chunk_size=numel, local_quantize=False)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - start)
        hier_time = sum(times) / len(times)
        hier_tp = (numel * 4 / 1e9) / hier_time
        
        # Pipelined hierarchical all-reduce
        times = []
        for i in range(warmup + iterations):
            t = torch.randn(numel, device=device, dtype=torch.float32)
            torch.cuda.synchronize()
            start = time.perf_counter()
            group = bitscom.LowBitGroup(bitwidth=4)
            group.all_reduce(t, local_group=local_group, inter_group=inter_group,
                           chunk_size=numel//4, local_quantize=False)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - start)
        pipe_time = sum(times) / len(times)
        pipe_tp = (numel * 4 / 1e9) / pipe_time
        
        if rank == 0:
            print(f"  NCCL:          {nccl_time*1000:.2f} ms, {nccl_tp:.2f} GB/s")
            print(f"  Hierarchical:  {hier_time*1000:.2f} ms, {hier_tp:.2f} GB/s")
            print(f"  Pipelined:     {pipe_time*1000:.2f} ms, {pipe_tp:.2f} GB/s")
        
        results.append({
            'size_mb': numel * 4 / 1e6,
            'nccl_ms': nccl_time * 1000,
            'hier_ms': hier_time * 1000,
            'pipe_ms': pipe_time * 1000,
            'nccl_gbs': nccl_tp,
            'hier_gbs': hier_tp,
            'pipe_gbs': pipe_tp,
        })
    
    return results


def main():
    bitscom.init(bitwidth=4)
    
    rank, world_size, local_rank, node_id, local_world_size, gpus_per_node = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        print(f"[INFO] World size: {world_size}", flush=True)
    print(
        f"[INFO] rank={rank} node={node_id} local_rank={local_rank} "
        f"local_world_size={local_world_size} gpus_per_node={gpus_per_node}",
        flush=True,
    )
    
    log_barrier("first barrier", rank, node_id, local_rank)
    
    # 测试1: 正确性验证
    if rank == 0:
        print("\n=== Test 1: Correctness ===", flush=True)
    correct = test_correctness(rank, world_size, local_rank, device, gpus_per_node)
    log_barrier("correctness barrier", rank, node_id, local_rank)
    
    # 测试2: 单节点fallback
    if rank == 0:
        print("\n=== Test 2: Single Node Fallback ===", flush=True)
    # single_node_ok = test_single_node_fallback(rank, world_size, local_rank)
    log_barrier("single-node fallback barrier", rank, node_id, local_rank)
    
    # 测试3: 吞吐量基准
    if rank == 0:
        print("\n=== Test 3: Throughput Benchmark ===", flush=True)
    results = benchmark_throughput(rank, world_size, local_rank, device, gpus_per_node)
    log_barrier("benchmark barrier", rank, node_id, local_rank)
    
    # 输出结果
    if rank == 0:
        print("\n=== Summary ===")
        print(f"Hierarchical correct: {correct}")
        # print(f"Single node fallback: {single_node_ok}")
        
        with open('hierarchical_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to hierarchical_benchmark_results.json")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
