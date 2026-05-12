# 实验说明

本目录包含用于压缩对比实验的脚本，覆盖：
- Llama7B (DP+PP)
- GPT-2
- BERT
- ResNet-50

结果输出目录：
- experiments/results/baselines
- experiments/results/bitscom

每次运行会生成 loss 曲线 CSV/PNG 以及包含吞吐率和最终 loss 的 summary CSV。
本实验仅比较 loss，不计算准确率。

## 方法说明
- none：全精度 all-reduce（不压缩）
- quant8：按张量 int8 量化 + all-gather
- topk：Top-K 稀疏化 + all-gather
- powersgd：低秩近似 + all-reduce
- bitscom：bitscom 低比特 all-reduce（单独分类）

## 运行：ResNet50 / BERT / GPT-2
真实数据集：
- ResNet50：CIFAR-10
- BERT：GLUE/SST-2
- GPT-2：WikiText-2 (raw)

使用 torchrun 多卡运行。

示例：
```
cd /home/aerith/udtca
torchrun --nproc_per_node=2 experiments/run_multimodel.py \
  --models resnet50 bert gpt2 \
  --methods none quant8 topk powersgd bitscom
```

单模型/单方法：
```
torchrun --nproc_per_node=2 experiments/run_multimodel.py \
  --models gpt2 --methods topk
```

禁用下载（只用本地缓存）：
```
torchrun --nproc_per_node=2 experiments/run_multimodel.py --no-download
```

## 运行：Llama7B DP+PP
该脚本使用流水线并行，DP 进行梯度同步。

示例：
```
cd /home/aerith/udtca
torchrun --nproc_per_node=4 experiments/run_llama7b_dp_pp.py \
  --pp-size 2 \
  --methods none bitscom quant8
```

注意：
- Llama7B 默认使用 WikiText-103（可切换为 wikitext-2 或 c4）。
- 默认允许下载；需要仅使用缓存请加 --no-download。
- loss 曲线按方法分别保存；吞吐率为 tokens/sec。
- ResNet50 吞吐率为 samples/sec；BERT/GPT-2 为 tokens/sec。
- bitscom 相关结果会单独存放在 experiments/results/bitscom。

## 运行顺序
推荐的实验顺序：
1) 先跑 ResNet50/BERT/GPT-2 的基线（none/quant8/topk/powersgd）。
2) 再跑 ResNet50/BERT/GPT-2 的 bitscom。
3) 再跑 Llama7B DP+PP 的基线（none/quant8/topk/powersgd）。
4) 最后跑 Llama7B DP+PP 的 bitscom。
