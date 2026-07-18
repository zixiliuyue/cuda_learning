#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 35: Nsight 深度性能分析 — Python/Numba 版本

使用 Numba 创建多个具有不同性能特征的 Kernel，自动计时并输出性能分析表格：
1. 内存带宽受限 Kernel（vector add）
2. 计算受限 Kernel（每元素大量 FMA）
3. 低占用率 Kernel（大量本地变量）
4. 高占用率 Kernel（轻量寄存器）

对比工具：Nsight Systems / Nsight Compute
    nsys profile python nsight_profiling.py
"""

import sys
import time
import numpy as np

try:
    from numba import cuda, float32
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)

# 常量定义
N = 1024 * 1024 * 8          # 8M 个元素
BLOCK_SIZE = 256
ITERATIONS = 20
COMPUTE_ITERS = 512


# Kernel 1: 内存带宽受限
@cuda.jit
def kernel_memory_bound(a, b, c, n):
    """内存受限：只做一次加法，读2写1，算术强度极低"""
    tid = cuda.grid(1)
    if tid < n:
        c[tid] = a[tid] + b[tid]


# Kernel 2: 计算受限
@cuda.jit
def kernel_compute_bound(a, c, n):
    """计算受限：每元素做大量 FMA，全在寄存器完成"""
    tid = cuda.grid(1)
    if tid < n:
        x = a[tid]
        acc = float32(0.0)
        for _ in range(COMPUTE_ITERS):
            acc = acc * float32(1.000001) + x
        c[tid] = acc


# Kernel 3: 低占用率（大量本地变量占用寄存器）
@cuda.jit
def kernel_low_occupancy(a, c, n):
    """低占用率：使用多个独立累加器，迫使分配较多寄存器"""
    tid = cuda.grid(1)
    if tid < n:
        r0 = a[tid]
        r1 = r0 + float32(1.0)
        r2 = r0 + float32(2.0)
        r3 = r0 + float32(3.0)
        r4 = r0 + float32(4.0)
        r5 = r0 + float32(5.0)
        r6 = r0 + float32(6.0)
        r7 = r0 + float32(7.0)
        for _ in range(64):
            r0 = r0 * float32(1.01) + r1
            r1 = r1 * float32(1.01) + r2
            r2 = r2 * float32(1.01) + r3
            r3 = r3 * float32(1.01) + r4
            r4 = r4 * float32(1.01) + r5
            r5 = r5 * float32(1.01) + r6
            r6 = r6 * float32(1.01) + r7
            r7 = r7 * float32(1.01) + r0
        c[tid] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7


# Kernel 4: 高占用率（单一累加器）
@cuda.jit
def kernel_high_occupancy(a, c, n):
    """高占用率：单一累加器，寄存器占用极低"""
    tid = cuda.grid(1)
    if tid < n:
        acc = a[tid]
        for _ in range(512):
            acc = acc * float32(1.01) + float32(0.5)
        c[tid] = acc


def bench(kernel, args, grid, block, iterations):
    """预热 + 多次计时取平均（毫秒）"""
    kernel[grid, block](*args)
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        kernel[grid, block](*args)
    cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1000.0


def main():
    print()
    print("=" * 60)
    print("     Day 35: Nsight 深度性能分析 — Python/Numba 版本         ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，Numba 无法执行 CUDA Kernel。")
        return 1

    # 打印设备信息
    dev = cuda.get_current_device()
    print(f"[设备] 名称: {dev.name.decode() if isinstance(dev.name, bytes) else dev.name}")
    print(f"[设备] SM 数量: {dev.MULTIPROCESSOR_COUNT}")
    print(f"[主机] 数组大小: {N} 个元素 ({N*4/(1024*1024):.2f} MB)")
    print()

    h_a = np.random.uniform(-1, 1, size=N).astype(np.float32)
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_a)
    d_c = cuda.device_array_like(h_a)

    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    print("[计时] 正在测量各 Kernel（含预热）...")
    t_mem = bench(kernel_memory_bound, (d_a, d_b, d_c, N), grid, BLOCK_SIZE, ITERATIONS)
    t_cmp = bench(kernel_compute_bound, (d_a, d_c, N), grid, BLOCK_SIZE, ITERATIONS)
    t_low = bench(kernel_low_occupancy, (d_a, d_c, N), grid, BLOCK_SIZE, ITERATIONS)
    t_high = bench(kernel_high_occupancy, (d_a, d_c, N), grid, BLOCK_SIZE, ITERATIONS)

    # 计算指标
    bw_mem = (3.0 * N * 4 / 1e9) / (t_mem / 1000.0)          # GB/s
    gflops_cmp = (N * 2.0 * COMPUTE_ITERS / 1e9) / (t_cmp / 1000.0)  # GFLOPS

    # 性能分析表格
    print()
    print("=" * 60)
    print("                     性能分析总结表格                       ")
    print("=" * 60)
    print(f"  {'Kernel':<14} | {'耗时(ms)':>10} | {'指标':>16}")
    print(f"  {'-'*14} | {'-'*10} | {'-'*16}")
    print(f"  {'内存受限':<12} | {t_mem:>10.4f} | {bw_mem:>10.1f} GB/s")
    print(f"  {'计算受限':<12} | {t_cmp:>10.4f} | {gflops_cmp:>8.1f} GFLOPS")
    print(f"  {'低占用率':<12} | {t_low:>10.4f} | {'大量寄存器':>16}")
    print(f"  {'高占用率':<12} | {t_high:>10.4f} | {'轻量寄存器':>16}")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. Roofline: 算术强度决定 Kernel 落在带宽屋顶还是算力屋顶")
    print("  2. memory-bound: 算术强度低，优化方向是减少访存 / 提升合并度")
    print("  3. compute-bound: 算术强度高，优化方向是减少运算 / 用更快指令")
    print("  4. 占用率: 更高的占用率有助于隐藏延迟，但并非越高越好")
    print("  5. 使用 nsys profile / ncu 获取硬件级精确指标")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
