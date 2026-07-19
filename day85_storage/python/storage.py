#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 85: 存储 — 并行文件系统 / 小文件问题 / 数据加载瓶颈

GPU 再快，数据喂不上也白搭。本天讲训练的「最后一公里」：数据从磁盘到 GPU
的流水线，以及三大痛点：并行文件系统、小文件问题、数据加载瓶颈。

核心目标（原理为主 + 单机可跑模拟）：
  1. 讲解并行文件系统（Lustre / GPFS）的条带化与元数据瓶颈；
  2. 演示「小文件问题」：大量小文件 vs 打包成一个大文件（tar/WebDataset 思路）；
  3. 模拟数据加载流水线：朴素串行 vs prefetch 多 worker 流水线，对比吞吐；
  4. 讲清 pin_memory / num_workers / prefetch 的作用。

无需 GPU，无需真实数据集；用 time.sleep 模拟 IO/计算耗时，纯 Python 即可跑。
"""

import time
import queue
import threading

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TORCH_CUDA = TORCH_AVAILABLE and torch.cuda.is_available()
MULTI_GPU = TORCH_CUDA and torch.cuda.device_count() > 1

SEP = "=" * 64


# ============================================================
#  1. 并行文件系统原理
# ============================================================
def explain_parallel_fs():
    print(SEP)
    print("并行文件系统（Lustre / GPFS）：为什么单机文件系统不够用")
    print(SEP)
    print(r"""
  训练集群里成百上千 GPU 同时读数据，本地磁盘/单 NFS 扛不住。
  并行文件系统把「一个文件条带化（striping）」打散到多个存储节点：

       客户端(GPU节点) ── 元数据服务(MDS) 查「文件在哪」
              │
              ├──▶ OSS/OST 1  ┐
              ├──▶ OSS/OST 2  ├─ 一个大文件被切成条带并行读/写
              └──▶ OSS/OST 3  ┘

    - Lustre：MDS(元数据) + 多个 OST(对象存储)，带宽随 OST 数近线性扩展。
    - GPFS(IBM Spectrum Scale)：块级并行，元数据分布式。

  两大瓶颈：
    1) 元数据瓶颈：海量小文件 → 频繁问 MDS「文件在哪」，MDS 成为热点。
    2) 带宽 vs IOPS：并行 FS 擅长「大文件高带宽」，怕「小文件高 IOPS」。
  ⇒ 这引出下面的「小文件问题」。
""")


# ============================================================
#  2. 小文件问题：大量小文件 vs 打包成大文件
# ============================================================
def simulate_small_files(num_files=2000, per_file_us=200, seq_read_us_per_file=5):
    """模拟读取数据集的两种方式，返回 (小文件耗时, 打包耗时)。

    模型（简化）：
      - 小文件：每个文件都要一次「打开/元数据/寻道」开销 per_file_us，
                读 num_files 个文件 = num_files * per_file_us。
      - 打包成一个大文件（tar/WebDataset）：只需 1 次打开，之后顺序流式读，
                = per_file_us（一次开销）+ num_files * seq_read_us_per_file。
    单位微秒；用比例关系体现差异，不做真实磁盘 IO。
    """
    small = num_files * per_file_us
    packed = per_file_us + num_files * seq_read_us_per_file
    return small, packed


def demo_small_files():
    print(SEP)
    print("小文件问题：ImageNet 式「一图一文件」 vs 打包（tar / WebDataset）")
    print(SEP)
    num_files = 2000
    small_us, packed_us = simulate_small_files(num_files)
    print(f"  数据集：{num_files} 个样本")
    print(f"  方式 A｜每样本一个小文件：每次都付『打开+元数据+寻道』开销")
    print(f"         预计耗时 ≈ {small_us / 1000:.1f} ms")
    print(f"  方式 B｜打包成 1 个大文件顺序流式读（WebDataset/tar/TFRecord）：")
    print(f"         预计耗时 ≈ {packed_us / 1000:.1f} ms")
    print(f"  加速比 ≈ {small_us / packed_us:.1f}x")
    print("""
  为什么打包更快：
    - 只付一次元数据开销，避免打爆并行 FS 的 MDS；
    - 顺序读能打满带宽，随机小读浪费 IOPS；
    - WebDataset 用 .tar 分片 + 顺序 shuffle，是大规模训练的常用格式。
""")
    return small_us > packed_us


# ============================================================
#  3. 数据加载流水线：朴素串行 vs prefetch 多 worker
# ============================================================
def _load_one(sample_idx, io_us):
    """模拟从磁盘加载 + 解码一个样本（用 sleep 表示 IO/解码耗时）。"""
    time.sleep(io_us / 1e6)
    # 返回一个小张量代表样本
    return np.full(8, sample_idx, dtype=np.float32)


def _train_step(batch, compute_us):
    """模拟 GPU 训练一步（用 sleep 表示计算耗时）。"""
    time.sleep(compute_us / 1e6)
    return float(np.sum(batch))


def naive_pipeline(num_batches, io_us, compute_us):
    """朴素串行：加载完一个 batch 才开始算，算完才加载下一个（无重叠）。"""
    t0 = time.perf_counter()
    total = 0.0
    for i in range(num_batches):
        batch = _load_one(i, io_us)   # 加载（GPU 空等）
        total += _train_step(batch, compute_us)  # 计算（IO 空等）
    return (time.perf_counter() - t0) * 1000, total


def prefetch_pipeline(num_batches, io_us, compute_us, num_workers=4, prefetch=4):
    """流水线：多 worker 后台预取，训练时下一批已在准备（计算与 IO 重叠）。

    对应 PyTorch DataLoader 的 num_workers + prefetch_factor + pin_memory。
    """
    q = queue.Queue(maxsize=prefetch)
    stop = object()
    index_q = queue.Queue()
    for i in range(num_batches):
        index_q.put(i)
    for _ in range(num_workers):
        index_q.put(stop)

    def worker():
        while True:
            idx = index_q.get()
            if idx is stop:
                break
            q.put((idx, _load_one(idx, io_us)))  # 后台加载，放入预取队列

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(num_workers)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()

    total = 0.0
    got = 0
    while got < num_batches:
        _, batch = q.get()  # 训练时批已就绪，几乎不等 IO
        total += _train_step(batch, compute_us)
        got += 1
    for th in threads:
        th.join()
    return (time.perf_counter() - t0) * 1000, total


def demo_pipeline():
    print(SEP)
    print("数据加载流水线：朴素串行 vs prefetch 多 worker（计算/IO 重叠）")
    print(SEP)
    num_batches = 20
    io_us = 8000       # 每批加载 8 ms（IO 慢）
    compute_us = 5000  # 每批计算 5 ms
    print(f"  配置：{num_batches} 批，每批 加载≈{io_us/1000:.0f}ms / 计算≈{compute_us/1000:.0f}ms")
    print()

    t_naive, s1 = naive_pipeline(num_batches, io_us, compute_us)
    t_pf, s2 = prefetch_pipeline(num_batches, io_us, compute_us, num_workers=4)

    print(f"  朴素串行（无重叠）  ：{t_naive:7.1f} ms  ≈ Σ(加载+计算)")
    print(f"  prefetch 4 workers ：{t_pf:7.1f} ms  （加载被计算/多worker藏起来）")
    print(f"  吞吐提升 ≈ {t_naive / t_pf:.1f}x，结果一致：{'✓' if abs(s1 - s2) < 1e-6 else '✗'}")
    print("""
  为什么更快：
    - num_workers：多进程/线程并行加载，把 IO 时间摊薄；
    - prefetch_factor：提前预取若干批，训练取到时数据已就绪；
    - pin_memory：数据放锁页内存，H2D 拷贝可异步且更快（配合 Day10 的流）；
    - 目标：让「数据加载」完全被「GPU 计算」掩盖，GPU 不空等。
""")
    return t_pf < t_naive


def explain_dataloader():
    print(SEP)
    print("PyTorch DataLoader 关键参数速查")
    print(SEP)
    print("""
  参数              作用                                   经验值
  ----------------------------------------------------------------
  num_workers      并行加载的子进程数                     4~8/GPU 起调
  prefetch_factor  每个 worker 预取的批数                 2~4
  pin_memory       锁页内存，加速 H2D 拷贝(配 non_blocking) 训练时 True
  persistent_workers 跨 epoch 不重建 worker                大数据集 True
  shuffle          打乱；大数据用 WebDataset 的分片内 shuffle

  诊断瓶颈：若 GPU 利用率忽高忽低/偏低，多半是数据加载没跟上，
           先加 num_workers、用打包格式、上 pin_memory。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_parallel_fs()
    print()
    ok1 = demo_small_files()
    print()
    ok2 = demo_pipeline()
    print()
    explain_dataloader()

    print()
    print(SEP)
    all_ok = ok1 and ok2
    print("[结果] 存储与数据加载优化演示" + ("全部通过 ✓" if all_ok else "存在异常 ✗"))
    if TORCH_CUDA:
        print("环境：有 GPU。真实训练建议 DataLoader(num_workers>0, pin_memory=True)。")
    elif TORCH_AVAILABLE:
        print("环境：有 torch 无 GPU。流水线用线程模拟，思路同 DataLoader 多进程。")
    else:
        print("环境：无 torch。已用纯 Python 线程模拟数据加载流水线。")
    print(SEP)


if __name__ == "__main__":
    main()
