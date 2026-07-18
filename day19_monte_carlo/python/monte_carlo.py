#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Day 19: 蒙特卡洛模拟 — Python/Numba 版本

使用蒙特卡洛方法估算圆周率 pi。
通过在单位正方形内随机投点，统计落在单位圆内的比例来估算 pi。
GPU 的大规模并行能力非常适合蒙特卡洛模拟。
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
NUM_SAMPLES = 10 * 1024 * 1024  # 10M 样本
BLOCK_SIZE = 256
ITERATIONS = 5
PI_REF = 3.14159265358979323846


@cuda.jit
def monte_carlo_kernel(result, num_samples):
    """
    蒙特卡洛 Kernel: 每个线程生成多个随机样本。
    使用简单的伪随机数生成（xorshift32）。
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    threads_total = cuda.blockDim.x * cuda.gridDim.x

    # 使用简单的 xorshift32 PRNG
    state = tid * 12345 + 67890
    local_count = 0

    samples_per_thread = (num_samples + threads_total - 1) // threads_total

    for i in range(samples_per_thread):
        # xorshift32 伪随机数
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5

        # 生成 [-1, 1] 范围的坐标
        x = (float(state & 0x7FFFFFFF) / float(0x7FFFFFFF)) * 2.0 - 1.0

        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        y = (float(state & 0x7FFFFFFF) / float(0x7FFFFFFF)) * 2.0 - 1.0

        # 判断是否在单位圆内
        if x * x + y * y <= 1.0:
            local_count += 1

    # 原子加到全局计数器
    cuda.atomic.add(result, 0, local_count)


@cuda.jit
def monte_carlo_kernel_shared(result, num_samples):
    """
    蒙特卡洛 Kernel（共享内存归约版）:
    每 Block 内使用共享内存归约，减少原子操作争用。
    """
    sdata = cuda.shared.array(BLOCK_SIZE, dtype=np.int32)
    tid = cuda.threadIdx.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + tid
    threads_total = cuda.blockDim.x * cuda.gridDim.x

    state = gid * 12345 + 67890
    local_count = 0

    samples_per_thread = (num_samples + threads_total - 1) // threads_total

    for i in range(samples_per_thread):
        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        x = (float(state & 0x7FFFFFFF) / float(0x7FFFFFFF)) * 2.0 - 1.0

        state ^= state << 13
        state ^= state >> 17
        state ^= state << 5
        y = (float(state & 0x7FFFFFFF) / float(0x7FFFFFFF)) * 2.0 - 1.0

        if x * x + y * y <= 1.0:
            local_count += 1

    sdata[tid] = local_count
    cuda.syncthreads()

    # Block 内归约
    step = BLOCK_SIZE // 2
    while step > 0:
        if tid < step:
            sdata[tid] += sdata[tid + step]
        cuda.syncthreads()
        step //= 2

    if tid == 0:
        cuda.atomic.add(result, 0, sdata[0])


def monte_carlo_cpu(num_samples):
    """CPU 蒙特卡洛（参考实现）"""
    x = np.random.uniform(-1, 1, size=num_samples)
    y = np.random.uniform(-1, 1, size=num_samples)
    count = np.sum(x * x + y * y <= 1.0)
    return count


def main():
    print()
    print("=" * 60)
    print("     Day 19: 蒙特卡洛模拟 — Python/Numba 版本               ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    print(f"[主机] 采样数量: {NUM_SAMPLES:,} 点")
    print(f"[主机] 理论 pi 值: {PI_REF:.10f}")

    grid_size = (NUM_SAMPLES + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ---- GPU 原子操作版 ----
    d_result = cuda.device_array(1, dtype=np.int32)
    print()
    print("[主机] 运行 GPU 蒙特卡洛（原子操作版）...")

    # 预热
    d_result_host = np.zeros(1, dtype=np.int32)
    d_result = cuda.to_device(d_result_host)
    monte_carlo_kernel[grid_size, BLOCK_SIZE](d_result, NUM_SAMPLES)
    cuda.synchronize()

    start = time.time()
    for _ in range(ITERATIONS):
        d_result_host = np.zeros(1, dtype=np.int32)
        d_result = cuda.to_device(d_result_host)
        monte_carlo_kernel[grid_size, BLOCK_SIZE](d_result, NUM_SAMPLES)
        cuda.synchronize()
    atomic_time = (time.time() - start) / ITERATIONS * 1000
    atomic_count = d_result.copy_to_host()[0]
    pi_atomic = 4.0 * atomic_count / NUM_SAMPLES

    # ---- GPU 共享内存版 ----
    print("[主机] 运行 GPU 蒙特卡洛（共享内存版）...")
    d_result_host = np.zeros(1, dtype=np.int32)
    d_result = cuda.to_device(d_result_host)
    monte_carlo_kernel_shared[grid_size, BLOCK_SIZE](d_result, NUM_SAMPLES)
    cuda.synchronize()

    start = time.time()
    for _ in range(ITERATIONS):
        d_result_host = np.zeros(1, dtype=np.int32)
        d_result = cuda.to_device(d_result_host)
        monte_carlo_kernel_shared[grid_size, BLOCK_SIZE](d_result, NUM_SAMPLES)
        cuda.synchronize()
    shared_time = (time.time() - start) / ITERATIONS * 1000
    shared_count = d_result.copy_to_host()[0]
    pi_shared = 4.0 * shared_count / NUM_SAMPLES

    # ---- CPU 参考计算 ----
    print("[主机] 运行 CPU 蒙特卡洛...")
    start = time.time()
    cpu_count = monte_carlo_cpu(NUM_SAMPLES)
    cpu_time = (time.time() - start) * 1000
    pi_cpu = 4.0 * cpu_count / NUM_SAMPLES

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU 原子版   : {atomic_time:.4f} ms, pi = {pi_atomic:.8f}")
    print(f"  GPU 共享内存 : {shared_time:.4f} ms, pi = {pi_shared:.8f}")
    print(f"  CPU NumPy    : {cpu_time:.4f} ms, pi = {pi_cpu:.8f}")
    print(f"  参考值 pi    : {PI_REF:.8f}")
    print(f"  GPU 原子误差 : {abs(pi_atomic - PI_REF)/PI_REF*100:.4f}%")
    print(f"  GPU 共享误差 : {abs(pi_shared - PI_REF)/PI_REF*100:.4f}%")
    if shared_time > 0:
        print(f"  共享 vs 原子 : {atomic_time / shared_time:.2f}x")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 蒙特卡洛: 通过随机采样估算确定性问题的解")
    print("  2. GPU 优势: 大量独立的随机采样可完全并行")
    print("  3. 共享内存归约减少原子操作争用")
    print("  4. 应用: 金融定价、物理模拟、渲染、优化")
    print("  5. 注意: GPU PRNG 质量影响结果精度")
    print("  6. Curand 库提供 GPU 端高质量随机数生成")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
