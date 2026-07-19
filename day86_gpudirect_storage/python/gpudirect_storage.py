#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 86: GPUDirect Storage (GDS) 原理 — IO 路径分析

在大模型训练/推理里，数据（数据集、权重、checkpoint）要从存储（NVMe SSD /
并行文件系统）搬到 GPU 显存。传统路径要在 CPU 内存里中转一次（bounce
buffer），既多一次拷贝，又占用宝贵的 CPU / PCIe 带宽。GPUDirect Storage
让存储控制器通过 DMA **直达** GPU 显存，绕过 CPU bounce buffer。

本天以「原理为主 + 单机可跑的路径/带宽模拟」呈现：
  1. 用 ASCII 图画出「传统 IO 路径」vs「GPUDirect Storage 路径」；
  2. 统计两条路径的数据拷贝次数与经过的总线；
  3. 用一个简单的带宽/延迟模型，模拟不同数据量下两条路径的耗时与有效带宽，
     量化 GDS 省掉一次拷贝带来的收益。

运行策略（能不回退就不回退）：
  - 有 torch + CUDA：打印 GPU 信息，并做一次真实的「磁盘 -> CPU -> GPU」
    拷贝计时，作为传统路径的真实参考（GDS 本身需 Linux + 特定驱动，这里不强求）；
  - 无 GPU：纯 Python 带宽模型模拟，任何环境都稳定运行、结果可复现。
"""

import os
import tempfile
import time

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量：torch / cuda / 真实计时可用）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# 是否装了 kvikio / cufile（GDS 的 Python 入口），仅探测不强求
GDS_AVAILABLE = False
try:
    import kvikio  # noqa: F401
    GDS_AVAILABLE = True
except Exception:
    GDS_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. IO 路径 ASCII 图
# ============================================================
def draw_io_paths():
    print(SEP)
    print("传统 IO 路径 vs GPUDirect Storage 路径")
    print(SEP)
    print(r"""
  【传统路径】存储 -> CPU 内存(bounce buffer) -> GPU 显存
  ------------------------------------------------------------
    +---------+   PCIe/DMA   +-------------+   PCIe/DMA   +--------+
    |  NVMe   | -----------> |  CPU 内存    | -----------> |  GPU   |
    |  SSD    |   (拷贝1)    | bounce buf  |   (拷贝2)    |  显存  |
    +---------+              +-------------+              +--------+
                                   ^
                                   |  数据要先落到 CPU 内存，再由 CPU
                                   |  发起第二次 DMA 送进 GPU
                                   +-- 多一次拷贝 + 占用 CPU/内存带宽

  【GPUDirect Storage】存储 -> GPU 显存（直达，绕过 CPU bounce buffer）
  ------------------------------------------------------------
    +---------+          PCIe/DMA (直达)            +--------+
    |  NVMe   | ---------------------------------->  |  GPU   |
    |  SSD    |   存储控制器直接 DMA 写入 GPU 显存    |  显存  |
    +---------+                                      +--------+
                                   ^
                                   |  CPU 只负责下发命令(控制面)，
                                   |  数据面完全不经过 CPU 内存
                                   +-- 少一次拷贝, 省 CPU/内存带宽, 降延迟
""")


# ============================================================
#  2. 拷贝次数 / 总线占用对比
# ============================================================
def compare_copies():
    print(SEP)
    print("两条路径的数据拷贝次数与总线占用对比")
    print(SEP)
    rows = [
        ("数据面拷贝次数", "2 次(存储->CPU, CPU->GPU)", "1 次(存储->GPU 直达)"),
        ("CPU 内存带宽占用", "占用(需容纳 bounce buffer)", "几乎不占用"),
        ("CPU 参与", "参与数据搬运", "仅下发命令(控制面)"),
        ("PCIe 往返", "2 段(2 次经过 PCIe)", "1 段(1 次经过 PCIe)"),
        ("小 IO 延迟", "较高(经 CPU 中转)", "较低(直达)"),
        ("需要条件", "无(通用)", "Linux + 支持的 NVMe/文件系统 + 驱动"),
    ]
    print(f"  {'维度':<16}{'传统路径':<28}{'GPUDirect Storage'}")
    print("  " + "-" * 70)
    for name, a, b in rows:
        print(f"  {name:<16}{a:<28}{b}")
    print()
    print("  一句话：GDS 把『存储->CPU->GPU』的两跳压成『存储->GPU』一跳，")
    print("          在大批量数据加载/checkpoint 读写时收益最明显。")


# ============================================================
#  3. 带宽/延迟模型：模拟两条路径耗时
# ============================================================
def model_time_seconds(num_bytes, bw_gbps, latency_us, copies):
    """极简 IO 时间模型：time = copies * (latency + bytes / 带宽)。

    - bw_gbps: 有效带宽 (GB/s)
    - latency_us: 每次传输的固定启动延迟 (微秒)
    - copies: 数据面拷贝次数（传统=2，GDS=1）
    """
    bw_bytes = bw_gbps * 1e9
    per_copy = latency_us * 1e-6 + num_bytes / bw_bytes
    return copies * per_copy


def simulate_bandwidth():
    print(SEP)
    print("演示：不同数据量下两条路径的耗时与有效带宽（带宽模型模拟）")
    print(SEP)

    # 一组典型参数（示意值，不代表具体硬件）
    pcie_bw = 12.0     # 单段 PCIe 有效带宽 GB/s (示意, ~Gen4 x16)
    latency = 50.0     # 每次传输固定延迟 us
    sizes_mb = [16, 128, 1024, 4096]   # 从小 batch 到大权重/checkpoint

    print(f"  参数：单段 PCIe 带宽={pcie_bw} GB/s, 固定延迟={latency} us")
    print(f"  传统路径=2 次拷贝(存储->CPU->GPU), GDS=1 次(存储->GPU 直达)")
    print()
    print(f"  {'数据量':>10}{'传统(ms)':>12}{'GDS(ms)':>12}{'加速比':>10}"
          f"{'GDS有效带宽':>14}")
    print("  " + "-" * 60)
    for mb in sizes_mb:
        nbytes = mb * 1024 * 1024
        t_trad = model_time_seconds(nbytes, pcie_bw, latency, copies=2)
        t_gds = model_time_seconds(nbytes, pcie_bw, latency, copies=1)
        speedup = t_trad / t_gds
        eff_bw = nbytes / t_gds / 1e9
        label = f"{mb} MB" if mb < 1024 else f"{mb // 1024} GB"
        print(f"  {label:>10}{t_trad*1e3:>12.2f}{t_gds*1e3:>12.2f}"
              f"{speedup:>9.2f}x{eff_bw:>12.1f}GB/s")
    print()
    print("  观察：数据量越大，省掉的那次拷贝占比越稳定 → 传统≈2倍耗时；")
    print("        GDS 理论上把端到端带宽提升到接近单段 PCIe 上限。")


# ============================================================
#  4. 可选：真实测一次「磁盘 -> CPU -> GPU」作为传统路径参考
# ============================================================
def measure_real_traditional_path():
    print(SEP)
    print("真实参考：磁盘 -> CPU 内存 -> GPU 显存（传统路径实测）")
    print(SEP)
    mb = 128
    nbytes = mb * 1024 * 1024
    data = np.random.bytes(nbytes)

    tmp = tempfile.NamedTemporaryFile(prefix="gds_demo_", suffix=".bin",
                                      delete=False)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()

        # 步骤1：磁盘 -> CPU 内存（读文件）
        t0 = time.perf_counter()
        with open(tmp.name, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        t1 = time.perf_counter()

        # 步骤2：CPU 内存 -> GPU 显存
        cpu_tensor = torch.from_numpy(arr.copy())
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        gpu_tensor = cpu_tensor.to("cuda")
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        read_ms = (t1 - t0) * 1e3
        h2d_ms = (t3 - t2) * 1e3
        print(f"  数据量={mb} MB")
        print(f"  步骤1 磁盘->CPU 内存 : {read_ms:8.2f} ms")
        print(f"  步骤2 CPU->GPU 显存  : {h2d_ms:8.2f} ms")
        print(f"  合计(传统两跳)       : {read_ms + h2d_ms:8.2f} ms")
        print(f"  GPU 张量形状={tuple(gpu_tensor.shape)}, 设备={gpu_tensor.device}")
        print("  → GDS 目标即省掉『步骤1 落到 CPU 内存』这一跳，直达 GPU。")
        del cpu_tensor, gpu_tensor
        torch.cuda.empty_cache()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ============================================================
#  主流程
# ============================================================
def main():
    draw_io_paths()
    print()
    compare_copies()
    print()
    simulate_bandwidth()

    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        print()
        try:
            measure_real_traditional_path()
        except Exception as e:  # noqa: BLE001
            print(f"  (实测跳过：{e})")

    print()
    print(SEP)
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        name = torch.cuda.get_device_name(0)
        print(f"环境：PyTorch + CUDA 就绪（{name}）。已实测传统路径作为参考。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU。已用带宽模型演示两条路径差异。")
    else:
        print("环境：未安装 PyTorch。已用纯 Python 带宽模型演示原理。")
    if GDS_AVAILABLE:
        print("检测到 kvikio（GDS Python 入口），可在 Linux+NVMe 上做真实 GDS 读写。")
    else:
        print("提示：真实 GDS 需 Linux + 支持的 NVMe/并行文件系统 + cuFile 驱动，")
        print("      Python 侧可用 `pip install kvikio` 体验（本天以原理为主）。")
    print(SEP)


if __name__ == "__main__":
    main()
