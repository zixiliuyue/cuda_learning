#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 16: 并行排序 — Python/Numba 版本

实现 Bitonic Sort（双调排序），一种经典的并行排序算法。
Bitonic Sort 利用比较和交换操作构建双调序列，然后在 O(n log^2 n) 内完成排序。
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
ARRAY_SIZE = 65536  # 必须2的幂（Bitonic Sort 要求）
BLOCK_SIZE = 256
TOLERANCE = 1.0


@cuda.jit
def bitonic_merge_kernel(data, stage, step, ascending):
    """
    Bitonic Merge Kernel: 在给定阶段进行合并。
    stage: 当前合并阶段（log2(n) 向下递减）
    step: 当前步骤内的比较距离
    ascending: True=升序, False=降序
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    n = len(data)

    # 计算比较对的位置
    pair_distance = 1 << stage
    block_distance = 1 << step

    # 两个元素的索引
    j = tid ^ block_distance

    # 只处理第一组（避免重复比较）
    if j > tid:
        # 判断是否属于同一个块
        if (tid & pair_distance) == 0:
            # 升序比较
            if (ascending and data[tid] > data[j]) or (not ascending and data[tid] < data[j]):
                data[tid], data[j] = data[j], data[tid]
        else:
            # 降序比较
            if (ascending and data[tid] < data[j]) or (not ascending and data[tid] > data[j]):
                data[tid], data[j] = data[j], data[tid]


def bitonic_sort_gpu(data):
    """GPU Bitonic Sort 主控制逻辑"""
    n = len(data)
    d_data = cuda.to_device(data.copy())
    grid_size = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Bitonic Sort 外层循环
    for stage in range(int(np.log2(n))):
        for step in range(stage + 1):
            # 所有块都升序排列（最终结果为升序）
            bitonic_merge_kernel[grid_size, BLOCK_SIZE](d_data, stage, step, True)
            cuda.synchronize()

    return d_data.copy_to_host()


def bitonic_sort_cpu(data):
    """CPU Bitonic Sort — 用于验证"""
    arr = data.copy()
    n = len(arr)

    for stage in range(int(np.log2(n))):
        for step in range(stage + 1):
            pair_distance = 1 << stage
            block_distance = 1 << step
            for tid in range(n):
                j = tid ^ block_distance
                if j > tid:
                    if (tid & pair_distance) == 0:
                        if arr[tid] > arr[j]:
                            arr[tid], arr[j] = arr[j], arr[tid]
                    else:
                        if arr[tid] < arr[j]:
                            arr[tid], arr[j] = arr[j], arr[tid]

    return arr


def main():
    print()
    print("=" * 60)
    print("     Day 16: 并行排序 — Python/Numba 版本                   ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 数据准备（必须是2的幂）
    actual_size = ARRAY_SIZE
    print(f"[主机] 数组大小: {actual_size:,} (2^{int(np.log2(actual_size))})")

    h_data = np.random.uniform(0, 1000, size=actual_size).astype(np.float32)

    # GPU Bitonic Sort
    print("[主机] 运行 GPU Bitonic Sort...")

    # 预热
    _ = bitonic_sort_gpu(h_data[:1024])

    start = time.time()
    h_sorted_gpu = bitonic_sort_gpu(h_data)
    gpu_time = (time.time() - start) * 1000

    # CPU Bitonic Sort（用小数据验证正确性）
    small_data = h_data[:1024].copy()
    start = time.time()
    h_sorted_cpu = bitonic_sort_cpu(small_data)
    cpu_time_small = (time.time() - start) * 1000

    # NumPy 排序（参考）
    start = time.time()
    h_sorted_np = np.sort(h_data)
    np_time = (time.time() - start) * 1000

    # 验证 GPU 结果
    gpu_correct = np.array_equal(h_sorted_gpu, h_sorted_np)

    # 验证 CPU 小数据
    small_gpu = bitonic_sort_gpu(small_data)
    small_np = np.sort(small_data)
    cpu_correct = np.array_equal(h_sorted_cpu, small_np) and np.array_equal(small_gpu, small_np)

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU Bitonic Sort  : {gpu_time:.4f} ms")
    print(f"  NumPy sort (CPU)  : {np_time:.4f} ms")
    print(f"  CPU Bitonic (1K)  : {cpu_time_small:.4f} ms")
    if np_time > 0:
        print(f"  GPU vs NumPy       : {np_time / max(gpu_time, 0.01):.2f} x")
    print(f"  GPU 结果验证      : {'通过' if gpu_correct else '失败'}")
    print(f"  CPU 结果验证      : {'通过' if cpu_correct else '失败'}")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. Bitonic Sort: 经典并行排序，复杂度 O(n log^2 n)")
    print("  2. 要求输入大小为 2 的幂")
    print("  3. 分两个阶段: 构建双调序列 + 合并排序")
    print("  4. 每步可并行执行: 比较和交换操作独立")
    print("  5. GPU 优势: 大量可并行的比较操作")
    print("  6. 实际中 Thrust 的 sort 更高效（Radix Sort 基础）")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
