#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2: 向量加法 — Python/Numba 版本
=============================================================

对应 C++ 版本的 vector_add，使用 Numba CUDA 实现。
每个线程处理一个元素的加法运算。
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

# 配置参数
VECTOR_SIZE = 1000000
THREADS_PER_BLOCK = 256
TOLERANCE = 1e-5


@cuda.jit
def vector_add_kernel(d_a, d_b, d_c, n):
    """
    向量加法 Kernel — Numba 版本
    每个线程计算: d_c[tid] = d_a[tid] + d_b[tid]
    """
    # 计算全局线程 ID
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # 边界检查
    if tid < n:
        d_c[tid] = d_a[tid] + d_b[tid]


def vector_add_cpu(a, b):
    """CPU 版本向量加法（用于验证）"""
    return a + b


def main():
    print()
    print("=" * 60)
    print("     Day 2: 向量加法 — Python/Numba 版本                  ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 在主机内存中创建输入数据
    print(f"[主机] 向量大小: {VECTOR_SIZE} 个元素")
    h_a = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)
    h_b = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)

    # 将数据复制到设备
    print("[主机] 正在将数据复制到设备 ...")
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_c = cuda.device_array_like(h_a)

    # 计算 Grid 维度
    grid_size = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    print(f"[主机] Grid: {grid_size} Blocks, Block: {THREADS_PER_BLOCK} 线程")

    # 启动 Kernel
    print("[主机] 启动向量加法 Kernel ...")
    start = time.time()
    vector_add_kernel[grid_size, THREADS_PER_BLOCK](d_a, d_b, d_c, VECTOR_SIZE)
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    # 复制结果回主机
    h_c = d_c.copy_to_host()

    # CPU 参考计算
    start = time.time()
    h_c_ref = vector_add_cpu(h_a, h_b)
    cpu_time = (time.time() - start) * 1000

    # 验证结果
    max_diff = np.max(np.abs(h_c - h_c_ref))
    is_correct = max_diff < TOLERANCE

    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU Kernel 时间 : {gpu_time:.4f} ms")
    print(f"  CPU 计算时间    : {cpu_time:.4f} ms")
    if gpu_time > 0:
        print(f"  GPU 加速比      : {cpu_time / gpu_time:.2f} x")
    print(f"  结果验证        : {'通过' if is_correct else '失败'} (最大误差: {max_diff:.2e})")
    print("=" * 60)
    print()

    return 0 if is_correct else 1


if __name__ == "__main__":
    sys.exit(main())
