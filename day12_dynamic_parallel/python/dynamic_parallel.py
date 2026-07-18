#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 12: 动态并行 — Python/Numba 版本

Numba 不完全支持 CUDA 动态并行（CDP），
本示例用多流 + 递归式 Kernel 启动来模拟动态并行的概念。

动态并行允许 GPU Kernel 在设备端启动子 Kernel，
减少 CPU-GPU 交互，适合递归算法（如树遍历、稀疏矩阵）。
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
ARRAY_SIZE = 65536
BLOCK_SIZE = 256
TOLERANCE = 1e-5
DEPTH = 4  # 模拟的递归深度


# =============================================================================
# Kernel: 第一层处理 — 对输入数组进行分段处理
# =============================================================================
@cuda.jit
def process_segment_kernel(data, result, offset, length):
    """
    处理数据的一个分段: result[i] = data[offset+i] * (i+1) / length
    模拟动态并行中每个子任务处理一部分数据。
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < length:
        idx = offset + tid
        result[idx] = data[idx] * (float(tid + 1) / float(length))


# =============================================================================
# Kernel: 简单的归约汇总
# =============================================================================
@cuda.jit
def sum_kernel(data, result, n):
    """将 data 求和到 result[0]"""
    sdata = cuda.shared.array(BLOCK_SIZE, dtype=np.float32)
    tid = cuda.threadIdx.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + tid

    if gid < n:
        sdata[tid] = data[gid]
    else:
        sdata[tid] = 0.0
    cuda.syncthreads()

    step = BLOCK_SIZE // 2
    while step > 0:
        if tid < step:
            sdata[tid] += sdata[tid + step]
        cuda.syncthreads()
        step //= 2

    if tid == 0:
        cuda.atomic.add(result, 0, sdata[0])


def process_segment_cpu(data, offset, length):
    """CPU 参考实现"""
    n = min(length, len(data) - offset)
    result = np.zeros(n, dtype=np.float32)
    for i in range(n):
        result[i] = data[offset + i] * (float(i + 1) / float(length))
    return result


def main():
    print()
    print("=" * 60)
    print("     Day 12: 动态并行 — Python/Numba 版本                    ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")
    print("[提示] Numba 不完全支持动态并行（CDP），")
    print("[提示] 本示例用多流 + 分段启动来模拟动态并行的概念。")
    print()

    # 数据准备
    print(f"[主机] 数组大小: {ARRAY_SIZE:,} 元素")
    print(f"[主机] 模拟递归深度: {DEPTH}")
    print(f"[主机] 每层分段数: {DEPTH}^depth")

    h_data = np.random.uniform(0.5, 2.0, size=ARRAY_SIZE).astype(np.float32)
    d_data = cuda.to_device(h_data)
    d_result = cuda.device_array_like(h_data)

    # =========================================================================
    # 模拟动态并行: 用多个流并发启动分段 Kernel
    # =========================================================================
    print()
    print("[主机] 方式1: 串行分段处理（模拟无动态并行）...")
    segments = DEPTH * DEPTH
    seg_len = ARRAY_SIZE // segments
    grid_size = (seg_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    start = time.time()
    for s in range(segments):
        offset = s * seg_len
        length = seg_len if s < segments - 1 else ARRAY_SIZE - offset
        gs = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        process_segment_kernel[gs, BLOCK_SIZE](d_data, d_result, offset, length)
        cuda.synchronize()
    serial_time = (time.time() - start) * 1000

    # 复位结果
    d_result2 = cuda.device_array_like(h_data)
    streams = [cuda.stream() for _ in range(segments)]

    print("[主机] 方式2: 多流并发分段处理（模拟动态并行）...")
    start = time.time()
    for s in range(segments):
        offset = s * seg_len
        length = seg_len if s < segments - 1 else ARRAY_SIZE - offset
        gs = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        with streams[s]:
            process_segment_kernel[gs, BLOCK_SIZE, streams[s]](
                d_data, d_result2, offset, length
            )
    for s in streams:
        s.synchronize()
    parallel_time = (time.time() - start) * 1000

    h_result = d_result2.copy_to_host()

    # CPU 参考计算
    start = time.time()
    h_ref = np.zeros(ARRAY_SIZE, dtype=np.float32)
    for s in range(segments):
        offset = s * seg_len
        length = seg_len if s < segments - 1 else ARRAY_SIZE - offset
        cpu_seg = process_segment_cpu(h_data, offset, length)
        h_ref[offset:offset + length] = cpu_seg
    cpu_time = (time.time() - start) * 1000

    # 验证
    max_diff = np.max(np.abs(h_result - h_ref))
    is_ok = max_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  串行分段处理     : {serial_time:.4f} ms")
    print(f"  多流并发处理     : {parallel_time:.4f} ms")
    print(f"  CPU 参考计算     : {cpu_time:.4f} ms")
    if parallel_time > 0:
        print(f"  多流 vs 串行     : {serial_time / parallel_time:.2f}x")
    print(f"  结果验证         : {'通过' if is_ok else '失败'} (最大误差: {max_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 动态并行(CDP): GPU Kernel 在设备端启动子 Kernel")
    print("  2. 优势: 减少 CPU-GPU 往返，适合递归/不规则算法")
    print("  3. Numba 不完全支持 CDP，可用多流模拟类似效果")
    print("  4. 典型应用: 树遍历、图算法、自适应网格")
    print("  5. 缺点: 占用率管理复杂，可能有性能开销")
    print("  6. CUDA C++ 中用 <<<>>> 在 __global__ 函数中启动")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
