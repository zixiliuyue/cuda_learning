#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 8: 寄存器与占用率优化 — Python/Numba 版本

演示不同 block size 对 GPU Kernel 性能的影响，
帮助理解占用率（Occupancy）的概念。
"""

import sys
import time
import numpy as np

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)

# 常量定义
VECTOR_SIZE = 4 * 1024 * 1024  # 4M 元素
TOLERANCE = 1e-5
ITERATIONS = 5

# 测试不同的 block size
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]


@cuda.jit
def vec_add_kernel(d_a, d_b, d_c, n):
    """
    简单向量加法 Kernel。
    通过改变 block size 来观察占用率对性能的影响。
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < n:
        d_c[tid] = d_a[tid] + d_b[tid]


@cuda.jit
def vec_mul_kernel(d_a, d_b, d_c, n):
    """
    向量乘法 Kernel — 稍微多些寄存器压力。
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < n:
        temp = d_a[tid] * d_b[tid]
        d_c[tid] = temp + temp * 0.5 - d_a[tid] * 0.25


def vec_add_cpu(a, b):
    """CPU 参考实现"""
    return a + b


def main():
    print()
    print("=" * 60)
    print("     Day 8: 寄存器与占用率优化 — Python/Numba 版本           ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")
        print("[提示] 不同 block size 的差异在真实 GPU 上更明显。")

    # 设备信息
    if cuda.is_available():
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else device.name
        cc = device.compute_capability
        sms = getattr(device, 'MULTIPROCESSOR_COUNT', None)
        max_threads = getattr(device, 'MAX_THREADS_PER_BLOCK', None)
        print(f"[设备] GPU: {name}")
        print(f"[设备] 计算能力: {cc}")
        if sms:
            print(f"[设备] 多处理器数: {sms}")
        if max_threads:
            print(f"[设备] 每 Block 最大线程数: {max_threads}")
        print()

    # 数据准备
    print(f"[主机] 向量大小: {VECTOR_SIZE:,} 元素 ({VECTOR_SIZE*4/1024/1024:.1f} MB)")
    h_a = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)
    h_b = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)
    h_c_ref = vec_add_cpu(h_a, h_b)

    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_c = cuda.device_array_like(h_a)

    print()
    print(f"{'Block Size':>12} {'Grid Size':>12} {'时间(ms)':>12} {'占用线程':>14} {'验证':>8}")
    print("-" * 60)

    results = []
    for bs in BLOCK_SIZES:
        # 跳过超过设备限制的 block size
        if cuda.is_available() and max_threads and bs > max_threads:
            print(f"{bs:>12} {'N/A':>12} {'N/A':>12} {'N/A':>14} {'跳过':>8}")
            continue

        grid_size = (VECTOR_SIZE + bs - 1) // bs
        total_threads = min(grid_size * bs, VECTOR_SIZE)

        # 预热
        vec_add_kernel[grid_size, bs](d_a, d_b, d_c, VECTOR_SIZE)
        cuda.synchronize()

        # 计时
        start = time.time()
        for _ in range(ITERATIONS):
            vec_add_kernel[grid_size, bs](d_a, d_b, d_c, VECTOR_SIZE)
            cuda.synchronize()
        elapsed = (time.time() - start) / ITERATIONS * 1000

        # 验证
        h_c = d_c.copy_to_host()
        max_diff = np.max(np.abs(h_c - h_c_ref))
        ok = "通过" if max_diff < TOLERANCE else "失败"

        print(f"{bs:>12} {grid_size:>12} {elapsed:>12.4f} {total_threads:>14,} {ok:>8}")
        results.append((bs, elapsed))

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    if results:
        best = min(results, key=lambda x: x[1])
        worst = max(results, key=lambda x: x[1])
        print(f"  测试的 Block Size: {[r[0] for r in results]}")
        print(f"  最佳 Block Size  : {best[0]} ({best[1]:.4f} ms)")
        print(f"  最差 Block Size  : {worst[0]} ({worst[1]:.4f} ms)")
        if worst[1] > 0:
            print(f"  最佳 vs 最差     : {worst[1]/best[1]:.2f}x 差异")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 占用率 = 活跃 warp 数 / 最大 warp 数")
    print("  2. Block size 太小: 不足的线程无法隐藏延迟")
    print("  3. Block size 太大: 寄存器/共享内存不足导致occupancy降低")
    print("  4. 通常 128~512 是较好的 block size 范围")
    print("  5. 256 是最常用的默认值，但最佳值因 Kernel 而异")
    print("  6. 实际项目中应使用 occupancy calculator 辅助调优")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
