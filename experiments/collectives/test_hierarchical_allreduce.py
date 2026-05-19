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
import torch
import torch.distributed as dist
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'bitscom', 'python'))
import bitscom


def init_distributed():
    """初始化分布式环境"""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 获取节点信息（通过环境变量或计算）
    node_id = int(os.environ.get("NODE_RANK", rank // 4))  # 假设每节点2卡
    local_rank = rank % 4
    
    return rank, world_size, local_rank, node_id

def build_hierarchical_groups(rank, world_size):
    """构建本地组和节点间组"""
    # 默认每节点2卡
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", 4))
    num_nodes = world_size // gpus_per_node
    
    node_id = rank // gpus_per_node
    local_rank = rank % gpus_per_node
    
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
    print(f"[Rank {rank}] Inter-group ranks: {inter_ranks}")
    inter_group = dist.new_group(inter_ranks)
    
    return groups[node_id], inter_group, gpus_per_node, num_nodes


def test_correctness(rank, world_size, local_rank, device):
    """测试 hierarchical all-reduce 的数值正确性"""
    # device = torch.device(f"cuda:{rank}")
    # torch.cuda.set_device(device)
    
    local_group, inter_group, gpus_per_node, num_nodes = build_hierarchical_groups(rank, world_size)
    
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


def benchmark_throughput(rank, world_size, local_rank, device):
    """吞吐量基准测试"""
    # device = torch.device(f"cuda:{local_rank}")
    # torch.cuda.set_device(device)
    
    local_group, inter_group, gpus_per_node, num_nodes = build_hierarchical_groups(rank, world_size)
    
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
    
    rank, world_size, local_rank, node_id = init_distributed()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"[INFO] World size: {world_size}")
        print(f"[INFO] Running on node {node_id}, local rank {local_rank}")
    
    dist.barrier()
    
    # 测试1: 正确性验证
    if rank == 0:
        print("\n=== Test 1: Correctness ===")
    correct = test_correctness(rank, world_size, local_rank, device)
    dist.barrier()
    
    # 测试2: 单节点fallback
    if rank == 0:
        print("\n=== Test 2: Single Node Fallback ===")
    # single_node_ok = test_single_node_fallback(rank, world_size, local_rank)
    dist.barrier()
    
    # 测试3: 吞吐量基准
    if rank == 0:
        print("\n=== Test 3: Throughput Benchmark ===")
    results = benchmark_throughput(rank, world_size, local_rank, device)
    dist.barrier()
    
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