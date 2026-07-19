#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 69: ZeRO-1/2/3 显存优化

标准数据并行（DDP）里，每个 rank 都存了「完整的」参数 / 梯度 / 优化器状态，
造成巨大冗余。ZeRO（Zero Redundancy Optimizer，DeepSpeed 提出）的思想：
把这些「本可切分」的状态按 rank 切成 N 份，每个 rank 只存 1/N，用到时再通信。

  ZeRO-1：切分【优化器状态】（Adam 的 m/v，通常最占显存）
  ZeRO-2：在 ZeRO-1 基础上，再切分【梯度】
  ZeRO-3：在 ZeRO-2 基础上，再切分【参数】本身（前向/反向时按需 all_gather）

本天以【原理 + 显存占用对比表】为主：用数据结构演示各级如何在 N 个 rank
间切分状态，并按「混合精度 Adam 训练」的经典公式计算每卡显存，打印对比表。

无 GPU 也能跑（纯计算 + 打印），稳定可退出。
"""

import numpy as np  # noqa: F401  (保持与系列脚本一致的依赖风格)

# ------------------------------------------------------------
# 环境探测（三布尔量）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DIST_AVAILABLE = False
CUDA_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import torch.distributed as dist  # noqa: F401
        DIST_AVAILABLE = True
    except ImportError:
        DIST_AVAILABLE = False
    CUDA_AVAILABLE = torch.cuda.is_available()

SEP = "=" * 64


# ============================================================
#  混合精度 Adam 训练的「每参数」显存账本（经典 16 字节模型）
#  设参数量 Ψ，采用 fp16 计算 + fp32 主权重（Adam）：
#    - fp16 参数     : 2Ψ 字节
#    - fp16 梯度     : 2Ψ 字节
#    - fp32 主参数   : 4Ψ 字节  ┐
#    - fp32 动量 m   : 4Ψ 字节  ├─ 优化器状态，共 12Ψ
#    - fp32 方差 v   : 4Ψ 字节  ┘
#  合计 16Ψ 字节。ZeRO 就是把其中「可切分」部分按 N 卡切成 1/N。
# ============================================================
BYTES = {
    "fp16_param": 2,   # 参数（前向计算用）
    "fp16_grad": 2,    # 梯度
    "opt_states": 12,  # fp32 主参数(4) + 动量(4) + 方差(4)
}


def mem_per_gpu(psi, world_size, stage):
    """返回某 ZeRO 级别下，单卡各部分显存（GB）与总和。

    psi        : 参数量（个数）
    world_size : 卡数 N
    stage      : 0=DDP基线, 1=ZeRO-1, 2=ZeRO-2, 3=ZeRO-3
    """
    N = world_size
    # 各部分是否被切分（除以 N），取决于 ZeRO 级别
    param_div = N if stage >= 3 else 1   # 参数：仅 ZeRO-3 切分
    grad_div = N if stage >= 2 else 1    # 梯度：ZeRO-2/3 切分
    opt_div = N if stage >= 1 else 1     # 优化器状态：ZeRO-1/2/3 切分

    p = BYTES["fp16_param"] * psi / param_div
    g = BYTES["fp16_grad"] * psi / grad_div
    o = BYTES["opt_states"] * psi / opt_div
    to_gb = 1.0 / (1024 ** 3)
    return {
        "param": p * to_gb,
        "grad": g * to_gb,
        "opt": o * to_gb,
        "total": (p + g + o) * to_gb,
    }


def explain_zero():
    print(SEP)
    print("ZeRO 核心思想：消除数据并行中的状态冗余")
    print(SEP)
    print("""
  DDP（每卡都存完整状态，冗余 N 份）：
    rank0: [参数 P][梯度 G][优化器 O]   ← 完整
    rank1: [参数 P][梯度 G][优化器 O]   ← 完整（冗余）
    ...
    rankN: [参数 P][梯度 G][优化器 O]   ← 完整（冗余）

  ZeRO 分级切分（每卡只存 1/N，用到再通信）：
    ┌─────────┬──────────┬──────────┬──────────────┐
    │ 级别    │ 参数 P   │ 梯度 G   │ 优化器状态 O │
    ├─────────┼──────────┼──────────┼──────────────┤
    │ DDP     │ 完整     │ 完整     │ 完整         │
    │ ZeRO-1  │ 完整     │ 完整     │ 切分 1/N     │
    │ ZeRO-2  │ 完整     │ 切分 1/N │ 切分 1/N     │
    │ ZeRO-3  │ 切分 1/N │ 切分 1/N │ 切分 1/N     │
    └─────────┴──────────┴──────────┴──────────────┘

  混合精度 Adam 的「每参数 16 字节」账本：
    fp16 参数 2  + fp16 梯度 2  + fp32主参4 + 动量4 + 方差4(=优化器12) = 16 字节
    优化器状态占了 12/16 = 75%，所以先切它（ZeRO-1）收益最大。

  通信代价（记住权衡）：
    ZeRO-1/2：通信量与 DDP 同量级（reduce_scatter + all_gather）；
    ZeRO-3  ：前向/反向都要 all_gather 参数，通信量约 1.5 倍，换取最大省显存。
""")


def demo_partition_datastructure(world_size=4):
    """用数据结构直观演示：一个 12 元素的「优化器状态」如何切给 4 个 rank。"""
    print(SEP)
    print(f"数据结构演示：把状态切分到 {world_size} 个 rank（以 12 元素为例）")
    print(SEP)
    full = list(range(12))
    print(f"  完整状态（DDP 每卡都存这 12 个）: {full}")
    shards = np.array_split(np.array(full), world_size)
    print(f"  ZeRO 切分后，每个 rank 只存 1/{world_size}：")
    for r in range(world_size):
        print(f"    rank {r} 持有: {shards[r].tolist()}")
    print("  → 需要完整状态时，用 all_gather 临时拼回；用完即释放。\n")


def print_table(psi_billion, world_size):
    """打印某个模型规模 + 卡数下，DDP / ZeRO-1/2/3 的单卡显存对比表。"""
    psi = int(psi_billion * 1e9)
    print(f"  模型规模 = {psi_billion}B 参数, world_size = {world_size} 卡")
    print("  " + "-" * 60)
    print(f"  {'级别':<8}{'参数(GB)':>11}{'梯度(GB)':>11}{'优化器(GB)':>13}{'单卡合计(GB)':>14}")
    print("  " + "-" * 60)
    names = {0: "DDP", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
    baseline = None
    for stage in (0, 1, 2, 3):
        m = mem_per_gpu(psi, world_size, stage)
        if stage == 0:
            baseline = m["total"]
        tag = ""
        if stage > 0:
            tag = f"  (↓{baseline / m['total']:.1f}x)"
        print(f"  {names[stage]:<8}{m['param']:>11.2f}{m['grad']:>11.2f}"
              f"{m['opt']:>13.2f}{m['total']:>14.2f}{tag}")
    print("  " + "-" * 60)
    print()


def main():
    explain_zero()
    demo_partition_datastructure(world_size=4)

    print(SEP)
    print("单卡显存对比表（混合精度 Adam，16 字节/参数模型）")
    print(SEP)
    print_table(psi_billion=7.5, world_size=8)
    print_table(psi_billion=7.5, world_size=64)
    print_table(psi_billion=70, world_size=64)

    # 简单校验：ZeRO-3 单卡总显存应约等于 16Ψ/N
    print(SEP)
    print("正确性校验")
    print(SEP)
    psi = int(7.5e9)
    N = 8
    m_ddp = mem_per_gpu(psi, N, 0)
    m_z3 = mem_per_gpu(psi, N, 3)
    expect_z3 = 16 * psi / N / (1024 ** 3)
    ok1 = abs(m_z3["total"] - expect_z3) < 1e-6
    ok2 = abs(m_ddp["total"] - 16 * psi / (1024 ** 3)) < 1e-6
    print(f"  DDP 单卡 = {m_ddp['total']:.2f} GB，期望 16Ψ = {16*psi/(1024**3):.2f} GB "
          f"→ {'✓' if ok2 else '✗'}")
    print(f"  ZeRO-3 单卡 = {m_z3['total']:.2f} GB，期望 16Ψ/N = {expect_z3:.2f} GB "
          f"→ {'✓' if ok1 else '✗'}")
    print(f"  ZeRO-3 相对 DDP 节省 = {m_ddp['total']/m_z3['total']:.1f}x（≈ N={N}）")
    print(f"  [校验] 显存公式自洽 ? {'通过 ✓' if (ok1 and ok2) else '失败 ✗'}\n")

    print(SEP)
    print("DeepSpeed ZeRO 配置骨架（原理对应，供实操参考，不执行）")
    print(SEP)
    print("""
  // ds_config.json
  {
    "train_batch_size": 512,
    "fp16": { "enabled": true },
    "zero_optimization": {
      "stage": 3,                         // 1 / 2 / 3 对应本天三级
      "offload_optimizer": { "device": "cpu" },   // 进一步把优化器状态放 CPU
      "offload_param":     { "device": "cpu" },    // ZeRO-3 可 offload 参数
      "overlap_comm": true,
      "contiguous_gradients": true
    }
  }

  # 启动：
  #   deepspeed --num_gpus=8 train.py --deepspeed --deepspeed_config ds_config.json
  # 代码里：
  #   model, optimizer, _, _ = deepspeed.initialize(
  #       model=model, optimizer=optimizer, config="ds_config.json")
""")

    print(SEP)
    if CUDA_AVAILABLE:
        print("环境：检测到 CUDA。可 `pip install deepspeed` 用上面配置跑真实 ZeRO。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA。本天为显存原理计算，无需 GPU 即可完整学习。")
    else:
        print("环境：未安装 PyTorch。本天为纯显存原理计算，不依赖任何框架。")
    print("说明：ZeRO 是显存优化，本质是「状态切分 + 按需通信」的数据布局问题。")
    print(SEP)


if __name__ == "__main__":
    main()
