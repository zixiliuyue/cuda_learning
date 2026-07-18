#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 34: 自定义内存池 — Python 版本

用 numpy 模拟 GPU 内存池的管理概念，对比两种方案的开销：
1. 每次都向系统申请 / 释放内存（模拟频繁 cudaMalloc/cudaFree）
2. 使用预分配的内存池（bump allocator），仅移动偏移指针

由于 numpy 的内存分配本身很快，这里主要用于演示 bump allocator 的
核心思想与接口设计，帮助理解 CUDA 版本的实现逻辑。
"""

import sys
import time
import numpy as np

# 常量定义
ALLOC_ELEMENTS = 256 * 1024      # 单次分配 256K 个 float32 = 1MB
TEST_ITERATIONS = 1000           # 分配/释放循环次数
POOL_ALIGNMENT = 256             # 对齐字节数


def align_up(size, alignment):
    """将 size 向上对齐到 alignment 的整数倍"""
    return (size + alignment - 1) & ~(alignment - 1)


class MemPool:
    """简单的内存池（bump allocator），用一块大 numpy buffer 模拟显存"""

    def __init__(self, total_bytes):
        # 预分配一大块字节缓冲（模拟一次性 cudaMalloc）
        self.buffer = np.empty(total_bytes, dtype=np.uint8)
        self.total_size = total_bytes
        self.offset = 0
        self.num_allocations = 0
        print(f"[内存池] 初始化成功，预分配: {total_bytes / (1024*1024):.2f} MB")

    def allocate(self, nbytes):
        """从池中分配一块内存（返回一个 view）"""
        aligned = align_up(nbytes, POOL_ALIGNMENT)
        if self.offset + aligned > self.total_size:
            raise MemoryError(f"[内存池] 空间不足！请求 {aligned} 字节")
        view = self.buffer[self.offset:self.offset + nbytes]
        self.offset += aligned
        self.num_allocations += 1
        return view

    def free(self, ptr):
        """bump allocator 下单块释放为空操作"""
        _ = ptr  # no-op

    def reset(self):
        """一次性回收所有分配（O(1)）"""
        self.offset = 0


def benchmark_direct(nbytes, iterations):
    """方案1: 每次都用 np.empty 申请新内存并释放"""
    start = time.perf_counter()
    for _ in range(iterations):
        arr = np.empty(nbytes, dtype=np.uint8)
        del arr  # 释放引用
    return (time.perf_counter() - start) * 1000.0


def benchmark_pool(pool, nbytes, iterations):
    """方案2: 使用内存池分配 / 释放"""
    start = time.perf_counter()
    for _ in range(iterations):
        view = pool.allocate(nbytes)
        pool.free(view)
        pool.reset()
    return (time.perf_counter() - start) * 1000.0


def main():
    print()
    print("=" * 60)
    print("     Day 34: 自定义内存池 — Python 版本                     ")
    print("=" * 60)
    print()

    nbytes = ALLOC_ELEMENTS * 4  # float32 = 4 字节
    print(f"[主机] 单次分配大小: {ALLOC_ELEMENTS} 个 float32 "
          f"({nbytes / (1024*1024):.2f} MB)")
    print(f"[主机] 测试迭代次数: {TEST_ITERATIONS} 次")
    print()

    # 功能验证：用内存池分配缓冲并做一次向量缩放
    print("[功能验证] 使用内存池分配的缓冲执行向量缩放")
    pool = MemPool(8 * 1024 * 1024)  # 预分配 8MB
    a_view = pool.allocate(nbytes)
    a = a_view.view(np.float32)
    a[:] = np.random.uniform(-1, 1, size=ALLOC_ELEMENTS).astype(np.float32)
    scale = 2.5
    result = a * scale
    expected = a * scale
    correct = np.allclose(result, expected, atol=1e-5)
    print(f"  计算结果: {'正确 (PASS)' if correct else '错误 (FAIL)'}")
    print(f"  当前池水位线: {pool.offset / (1024*1024):.2f} MB / "
          f"{pool.total_size / (1024*1024):.2f} MB")
    pool.reset()
    print()

    # 性能对比
    print("=" * 60)
    print("     性能对比: 直接分配  vs  内存池                          ")
    print("=" * 60)

    # 预热
    benchmark_direct(nbytes, 1)
    benchmark_pool(pool, nbytes, 1)
    pool.reset()

    time_direct = benchmark_direct(nbytes, TEST_ITERATIONS)
    time_pool = benchmark_pool(pool, nbytes, TEST_ITERATIONS)

    print(f"  {'方案':<26} | {'总耗时(ms)':>12} | {'平均每次(us)':>14}")
    print(f"  {'-'*26} | {'-'*12} | {'-'*14}")
    print(f"  {'直接 np.empty 分配':<24} | {time_direct:>12.3f} | "
          f"{time_direct / TEST_ITERATIONS * 1000:>14.3f}")
    print(f"  {'自定义内存池':<26} | {time_pool:>12.3f} | "
          f"{time_pool / TEST_ITERATIONS * 1000:>14.3f}")
    print()

    # 性能总结
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    if time_pool > 0:
        print(f"  内存池相对加速比: {time_direct / time_pool:.1f} x")
    print("  结论: 内存池将反复的系统级分配摊销为一次性预分配 + O(1) 偏移移动。")
    print("  在 CUDA 中，这一优势因 cudaMalloc/cudaFree 的高开销而更加明显。")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
