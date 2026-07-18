#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 9: 原子操作与并行归约 — Python/Numba 版本

对比交互式归约（所有线程原子加）vs 树形归约（共享内存+Warp shuffle），
演示原子操作和并行归约的性能差异。
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
VECTOR_SIZE = 1024 * 1024  # 1M 元素
BLOCK_SIZE = 256
TOLERANCE = 1e-1
ITERATIONS = 10


# =============================================================================
# 方法1: 原子操作归约（所有线程直接原子加到全局结果）
# =============================================================================
@cuda.jit
def reduction_atomic_kernel(data, global_result, n):
    """
    交互式归约: 每个线程将自己的值通过原子加到全局变量。
    简单但效率低 — 大量线程争用同一个全局内存地址。
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if tid < n:
        cuda.atomic.add(global_result, 0, data[tid])


# =============================================================================
# 方法2: 树形归约（每 Block 内使用共享内存归约）
# =============================================================================
@cuda.jit
def reduction_tree_kernel(data, partial_sums, n):
    """
    树形归约: 每个 Block 内使用共享内存进行树形归约，
    最终结果写入 partial_sums[blockIdx.x]。
    最后在 CPU 上对 partial_sums 求和。
    """
    # 共享内存 — 存放当前 Block 的中间结果
    sdata = cuda.shared.array(BLOCK_SIZE, dtype=np.float32)

    tid = cuda.threadIdx.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + tid

    # 将数据加载到共享内存
    if gid < n:
        sdata[tid] = data[gid]
    else:
        sdata[tid] = 0.0

    cuda.syncthreads()

    # 树形归约: 步长逐步减半
    step = BLOCK_SIZE // 2
    while step > 0:
        if tid < step:
            sdata[tid] += sdata[tid + step]
        cuda.syncthreads()
        step //= 2

    # Block 内第一个线程写出结果
    if tid == 0:
        partial_sums[cuda.blockIdx.x] = sdata[0]


def main():
    print()
    print("=" * 60)
    print("     Day 9: 原子操作与并行归约 — Python/Numba 版本           ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 数据准备
    print(f"[主机] 向量大小: {VECTOR_SIZE:,} 元素")
    h_data = np.random.uniform(0.0, 10.0, size=VECTOR_SIZE).astype(np.float32)
    cpu_sum = float(np.sum(h_data))
    print(f"[主机] CPU 参考结果: {cpu_sum:.6f}")

    d_data = cuda.to_device(h_data)

    # Grid 配置
    grid_size = (VECTOR_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ---- 原子操作归约 ----
    d_global_result = cuda.device_array(1, dtype=np.float32)
    print()
    print("[主机] 测试原子操作归约...")

    # 预热
    d_global_result_host = np.zeros(1, dtype=np.float32)
    d_global_result = cuda.to_device(d_global_result_host)
    reduction_atomic_kernel[grid_size, BLOCK_SIZE](d_data, d_global_result, VECTOR_SIZE)
    cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(ITERATIONS):
        d_global_result_host = np.zeros(1, dtype=np.float32)
        d_global_result = cuda.to_device(d_global_result_host)
        reduction_atomic_kernel[grid_size, BLOCK_SIZE](d_data, d_global_result, VECTOR_SIZE)
        cuda.synchronize()
    atomic_time = (time.time() - start) / ITERATIONS * 1000
    atomic_result = d_global_result.copy_to_host()[0]

    # ---- 树形归约 ----
    partial_sums = cuda.device_array(grid_size, dtype=np.float32)
    print("[主机] 测试树形归约（共享内存）...")

    # 预热
    reduction_tree_kernel[grid_size, BLOCK_SIZE](d_data, partial_sums, VECTOR_SIZE)
    cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(ITERATIONS):
        reduction_tree_kernel[grid_size, BLOCK_SIZE](d_data, partial_sums, VECTOR_SIZE)
        cuda.synchronize()
    tree_time = (time.time() - start) / ITERATIONS * 1000
    h_partial = partial_sums.copy_to_host()
    tree_result = float(np.sum(h_partial))

    # 验证
    atomic_diff = abs(atomic_result - cpu_sum) / max(abs(cpu_sum), 1e-10)
    tree_diff = abs(tree_result - cpu_sum) / max(abs(cpu_sum), 1e-10)
    atomic_ok = atomic_diff < TOLERANCE
    tree_ok = tree_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  原子操作归约: {atomic_time:.4f} ms  结果={atomic_result:.6f} {'通过' if atomic_ok else '失败'}")
    print(f"  树形归约    : {tree_time:.4f} ms  结果={tree_result:.6f} {'通过' if tree_ok else '失败'}")
    print(f"  CPU 参考值  : {cpu_sum:.6f}")
    if atomic_time > 0:
        print(f"  树形 vs 原子: {atomic_time / tree_time:.2f}x 更快")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 原子操作: cuda.atomic.add() 保证线程安全但串行化严重")
    print("  2. 树形归约: 利用共享内存减少全局内存争用")
    print("  3. 树形归约的复杂度: O(log N) vs O(N)")
    print("  4. 共享内存: cuda.shared.array() 声明，每 Block 独立一份")
    print("  5. cuda.syncthreads() 确保 Block 内所有线程同步")
    print("  6. 实际应用中应结合 Warp Shuffle 进一步优化")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
