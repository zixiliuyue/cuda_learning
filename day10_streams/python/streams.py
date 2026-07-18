#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 10: CUDA 流与异步执行 — Python/Numba 版本

演示 Numba 的 cuda.stream() 功能，实现数据传输与 Kernel 执行的重叠，
以及多个流之间的并发执行。
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
BLOCK_SIZE = 256
TOLERANCE = 1e-5
NUM_STREAMS = 4


@cuda.jit
def vec_add_kernel(d_a, d_b, d_c, n):
    """向量加法 Kernel"""
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < n:
        d_c[tid] = d_a[tid] + d_b[tid] + 1.0  # +1.0 用于区分


@cuda.jit
def vec_scale_kernel(d_c, d_result, n):
    """向量缩放 Kernel"""
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < n:
        d_result[tid] = d_c[tid] * 2.0


def vec_cpu(a, b):
    """CPU 参考计算: (a + b + 1.0) * 2.0"""
    return (a + b + 1.0) * 2.0


def main():
    print()
    print("=" * 60)
    print("     Day 10: CUDA 流与异步执行 — Python/Numba 版本            ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 数据准备
    chunk = VECTOR_SIZE // NUM_STREAMS
    print(f"[主机] 总向量大小: {VECTOR_SIZE:,} 元素")
    print(f"[主机] 流数量     : {NUM_STREAMS}")
    print(f"[主机] 每流处理   : {chunk:,} 元素")
    print()

    h_a = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)
    h_b = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)
    h_result = np.zeros(VECTOR_SIZE, dtype=np.float32)

    # CPU 参考计算
    start = time.time()
    h_ref = vec_cpu(h_a, h_b)
    cpu_time = (time.time() - start) * 1000

    # =========================================================================
    # 方法1: 默认流（串行执行）
    # =========================================================================
    print("[主机] 测试默认流（串行执行）...")
    d_a_all = cuda.to_device(h_a)
    d_b_all = cuda.to_device(h_b)
    d_c_all = cuda.device_array_like(h_a)
    d_r_all = cuda.device_array_like(h_a)

    grid = (VECTOR_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    start = time.time()
    vec_add_kernel[grid, BLOCK_SIZE](d_a_all, d_b_all, d_c_all, VECTOR_SIZE)
    cuda.synchronize()
    vec_scale_kernel[grid, BLOCK_SIZE](d_c_all, d_r_all, VECTOR_SIZE)
    cuda.synchronize()
    serial_time = (time.time() - start) * 1000
    h_serial = d_r_all.copy_to_host()

    # =========================================================================
    # 方法2: 多流并发执行
    # =========================================================================
    print("[主机] 测试多流并发执行...")
    # 创建多个流
    streams = [cuda.stream() for _ in range(NUM_STREAMS)]
    d_r_multi = np.zeros(VECTOR_SIZE, dtype=np.float32)

    grid_chunk = (chunk + BLOCK_SIZE - 1) // BLOCK_SIZE

    start = time.time()
    for i in range(NUM_STREAMS):
        offset = i * chunk
        a_chunk = h_a[offset:offset + chunk]
        b_chunk = h_b[offset:offset + chunk]

        # 在不同流中执行: 传输 -> Kernel -> 传输回
        with streams[i]:
            d_a_chunk = cuda.to_device(a_chunk, stream=streams[i])
            d_b_chunk = cuda.to_device(b_chunk, stream=streams[i])
            d_c_chunk = cuda.device_array_like(a_chunk)
            d_r_chunk = cuda.device_array_like(a_chunk)

            vec_add_kernel[grid_chunk, BLOCK_SIZE, streams[i]](
                d_a_chunk, d_b_chunk, d_c_chunk, chunk
            )
            vec_scale_kernel[grid_chunk, BLOCK_SIZE, streams[i]](
                d_c_chunk, d_r_chunk, chunk
            )

            h_r_chunk = d_r_chunk.copy_to_host(stream=streams[i])
            d_r_multi[offset:offset + chunk] = h_r_chunk

    # 等待所有流完成
    for s in streams:
        s.synchronize()

    multi_time = (time.time() - start) * 1000

    # 验证
    serial_diff = np.max(np.abs(h_serial - h_ref))
    multi_diff = np.max(np.abs(d_r_multi - h_ref))
    serial_ok = serial_diff < TOLERANCE
    multi_ok = multi_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  默认流（串行）   : {serial_time:.4f} ms")
    print(f"  多流（并发）     : {multi_time:.4f} ms")
    print(f"  CPU 参考          : {cpu_time:.4f} ms")
    if multi_time > 0:
        print(f"  多流 vs 默认流   : {serial_time / multi_time:.2f}x")
    print(f"  串行验证          : {'通过' if serial_ok else '失败'} (误差: {serial_diff:.2e})")
    print(f"  多流验证          : {'通过' if multi_ok else '失败'} (误差: {multi_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. CUDA 流: 异步执行引擎，允许操作重叠")
    print("  2. 默认流: 所有操作串行执行")
    print("  3. 多流: 不同流中的操作可以并发")
    print("  4. 数据传输与计算可以重叠，提高利用率")
    print("  5. Numba: cuda.stream() 创建流, with stream: 指定流")
    print("  6. 适合: 多批次数据处理、流水线并行")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
