#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 17: 并行前缀和 — Python/Numba 版本

实现 Blelloch Scan（工作高效的前缀和算法），
对比串行前缀和 vs GPU 并行前缀和的性能差异。
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
ARRAY_SIZE = 1024 * 1024  # 1M 元素
BLOCK_SIZE = 256
TOLERANCE = 1e-3


# =============================================================================
# Blelloch Scan — Up-Sweep (Reduce) Phase
# =============================================================================
@cuda.jit
def scan_upsweep_kernel(data, n):
    """
    Up-Sweep (Reduce) 阶段:
    从叶子节点向上构建，将数据变为部分和。
    d: 距离，从 1 开始逐步翻倍。
    """
    d = 1
    while d < n:
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if tid < n:
            # 计算配对
            left = 2 * d * (tid // (2 * d)) + (tid % (2 * d))
            right = left + d
            if right < n:
                data[right] += data[left]
        cuda.syncthreads()
        d *= 2


# =============================================================================
# Blelloch Scan — Down-Sweep Phase
# =============================================================================
@cuda.jit
def scan_downsweep_kernel(data, n):
    """
    Down-Sweep 阶段:
    将部分和分布回正确位置，得到前缀和。
    """
    d = n // 2
    while d > 0:
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if tid < n:
            left = 2 * d * (tid // (2 * d)) + (tid % (2 * d))
            right = left + d
            if right < n:
                temp = data[right]
                data[right] += data[left]
                data[left] = temp
        cuda.syncthreads()
        d //= 2


@cuda.jit
def scan_set_last_kernel(data, n):
    """将最后一个元素设为0（Blelloch 要求）"""
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid == n - 1:
        data[tid] = 0


def scan_sequential(data):
    """串行前缀和（参考实现）"""
    output = np.zeros_like(data)
    output[0] = data[0]
    for i in range(1, len(data)):
        output[i] = output[i - 1] + data[i]
    return output


def blelloch_scan_gpu(data):
    """GPU Blelloch Scan 完整流程"""
    n = len(data)
    # 确保 n 是 2 的幂
    padded_n = 1
    while padded_n < n:
        padded_n *= 2

    padded_data = np.zeros(padded_n, dtype=data.dtype)
    padded_data[:n] = data

    d_data = cuda.to_device(padded_data)
    grid_size = (padded_n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Up-Sweep
    # 注意: 使用循环方式，每次只处理一个 d 值
    d = 1
    while d < padded_n:
        _blelloch_step_kernel[grid_size, BLOCK_SIZE](d_data, padded_n, d, True)
        cuda.synchronize()
        d *= 2

    # 设置最后一个元素为0
    d_data_host = d_data.copy_to_host()
    d_data_host[-1] = 0
    d_data = cuda.to_device(d_data_host)

    # Down-Sweep
    d = padded_n // 2
    while d > 0:
        _blelloch_step_kernel[grid_size, BLOCK_SIZE](d_data, padded_n, d, False)
        cuda.synchronize()
        d //= 2

    result = d_data.copy_to_host()
    return result[:n]


@cuda.jit
def _blelloch_step_kernel(data, n, d, is_upsweep):
    """
    Blelloch Scan 的单步 Kernel。
    is_upsweep: True=Up-Sweep, False=Down-Sweep
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # 计算当前线程处理的对
    block_start = (tid // (2 * d)) * 2 * d
    pos_in_block = tid % (2 * d)

    left = block_start + pos_in_block
    right = left + d

    if right < n:
        if is_upsweep:
            data[right] += data[left]
        else:
            temp = data[right]
            data[right] += data[left]
            data[left] = temp


def main():
    print()
    print("=" * 60)
    print("     Day 17: 并行前缀和 — Python/Numba 版本                  ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 数据准备
    print(f"[主机] 数组大小: {ARRAY_SIZE:,} 元素")
    h_data = np.random.uniform(0, 10, size=ARRAY_SIZE).astype(np.float32)

    # GPU Blelloch Scan
    print("[主机] 运行 GPU Blelloch Scan...")
    # 预热
    _ = blelloch_scan_gpu(h_data[:256])

    start = time.time()
    h_scan_gpu = blelloch_scan_gpu(h_data)
    gpu_time = (time.time() - start) * 1000

    # CPU 串行前缀和
    print("[主机] 运行 CPU 串行前缀和...")
    start = time.time()
    h_scan_cpu = scan_sequential(h_data)
    cpu_time = (time.time() - start) * 1000

    # NumPy cumsum（参考）
    start = time.time()
    h_scan_np = np.cumsum(h_data)
    # Blelloch scan 是 exclusive scan，需要调整对比
    np_time = (time.time() - start) * 1000

    # 验证: Blelloch 是 exclusive scan
    # exclusive scan: output[i] = sum(data[0..i-1]), output[0] = 0
    h_ref_exclusive = np.zeros(ARRAY_SIZE, dtype=np.float32)
    h_ref_exclusive[1:] = np.cumsum(h_data[:-1])

    max_diff = np.max(np.abs(h_scan_gpu - h_ref_exclusive))
    is_ok = max_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU Blelloch Scan : {gpu_time:.4f} ms")
    print(f"  CPU 串行 Scan     : {cpu_time:.4f} ms")
    print(f"  NumPy cumsum      : {np_time:.4f} ms")
    if gpu_time > 0:
        print(f"  GPU vs CPU         : {cpu_time / gpu_time:.2f} x")
    print(f"  结果验证          : {'通过' if is_ok else '失败'} (最大误差: {max_diff:.2e})")
    print(f"  Scan 类型         : Exclusive Scan")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 前缀和: output[i] = sum(input[0..i-1])")
    print("  2. Exclusive vs Inclusive: 是否包含当前元素")
    print("  3. Blelloch (工作高效): O(n log n) 工作, O(log n) 步")
    print("  4. Up-Sweep: 构建部分和树")
    print("  5. Down-Sweep: 将和分配回各位置")
    print("  6. 应用: 流压缩、直方图、排序、树结构运算")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
