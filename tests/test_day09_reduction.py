#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 9 测试：原子操作与并行归约 — 正确性验证
==============================================

测试内容：
  1. 原子操作归约（reduction_atomic_kernel）结果与 CPU numpy 求和一致
  2. 树形归约（reduction_tree_kernel，共享内存）结果与 CPU numpy 求和一致
  3. Warp Shuffle 归约结果与 CPU numpy 求和一致
  4. 三种归约方式的结果互相一致

说明：
  被测源码模块 reduction.py 提供了原子归约与树形归约两个 Kernel；
  Warp Shuffle 归约在本测试文件内使用 numba.cuda 就地实现（对应
  day09_reduction/src/main.cu 中的 reduce_warp_shuffle_kernel），
  以便完整覆盖 Day 9 三种归约实现。

  归约涉及大量浮点累加，存在舍入误差，因此使用相对误差 + 较宽松容差。
"""

import numpy as np
import pytest

# 导入 conftest 中的公共工具和常量
from conftest import (require_cuda, safe_import_day_module,  # noqa: E402
                      REDUCTION_SIZE, REDUCTION_BLOCK_SIZE,
                      REDUCTION_TOLERANCE)

# 尝试安全导入被测模块
_reduction = safe_import_day_module("reduction", "day09_reduction")

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


# 每个 Block 的线程数（须为 32 的倍数，供 Warp 归约使用）
BLOCK_SIZE = REDUCTION_BLOCK_SIZE
WARP_SIZE = 32


# ---------------------------------------------------------------------------
# Warp Shuffle 归约 Kernel（就地实现，对应 main.cu 中的实现）
# ---------------------------------------------------------------------------
if cuda is not None:
    @cuda.jit
    def reduction_warp_shuffle_kernel(data, partial_sums, n):
        """Warp Shuffle 归约：Warp 内使用 shfl_down_sync 交换数据。

        每个 Block 先在各 Warp 内用 shuffle 指令归约，再由第一个 Warp
        归约所有 Warp 的部分和，最终写入 partial_sums[blockIdx.x]。
        """
        tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        lane = cuda.threadIdx.x % WARP_SIZE
        warp_id = cuda.threadIdx.x // WARP_SIZE

        # 每个线程读取一个元素（越界补 0）
        val = data[tid] if tid < n else np.float32(0.0)

        # Warp 内归约
        offset = WARP_SIZE // 2
        while offset > 0:
            val += cuda.shfl_down_sync(0xFFFFFFFF, val, offset)
            offset //= 2

        # 每个 Warp 的结果写入共享内存
        warp_sums = cuda.shared.array(BLOCK_SIZE // WARP_SIZE, dtype=np.float32)
        if lane == 0:
            warp_sums[warp_id] = val
        cuda.syncthreads()

        # 由第一个 Warp 归约所有 Warp 的部分和
        if warp_id == 0:
            val = warp_sums[lane] if lane < (BLOCK_SIZE // WARP_SIZE) else np.float32(0.0)
            offset = WARP_SIZE // 2
            while offset > 0:
                val += cuda.shfl_down_sync(0xFFFFFFFF, val, offset)
                offset //= 2
            if lane == 0:
                partial_sums[cuda.blockIdx.x] = val


class TestDay09Reduction:
    """Day 9: 原子操作与并行归约的测试集合。"""

    def _grid_size(self, n):
        """计算覆盖 n 个元素所需的 Block 数量。"""
        return (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    def _run_atomic(self, d_data, n):
        """辅助函数：执行原子操作归约，返回标量结果。"""
        h_result = np.zeros(1, dtype=np.float32)
        d_result = cuda.to_device(h_result)

        grid = self._grid_size(n)
        _reduction.reduction_atomic_kernel[grid, BLOCK_SIZE](d_data, d_result, n)
        cuda.synchronize()

        return float(d_result.copy_to_host()[0])

    def _run_tree(self, d_data, n):
        """辅助函数：执行树形（共享内存）归约，返回标量结果。

        每个 Block 归约到 partial_sums[blockIdx.x]，最后在 CPU 上求和。
        """
        grid = self._grid_size(n)
        partial_sums = cuda.device_array(grid, dtype=np.float32)

        _reduction.reduction_tree_kernel[grid, BLOCK_SIZE](d_data, partial_sums, n)
        cuda.synchronize()

        return float(np.sum(partial_sums.copy_to_host()))

    def _run_warp_shuffle(self, d_data, n):
        """辅助函数：执行 Warp Shuffle 归约，返回标量结果。"""
        grid = self._grid_size(n)
        partial_sums = cuda.device_array(grid, dtype=np.float32)

        reduction_warp_shuffle_kernel[grid, BLOCK_SIZE](d_data, partial_sums, n)
        cuda.synchronize()

        return float(np.sum(partial_sums.copy_to_host()))

    def test_atomic_reduction_correctness(self, random_reduction_vector):
        """测试原子操作归约的正确性。

        将 GPU 原子归约结果与 CPU numpy 求和对比，验证相对误差在容差内。

        Args:
            random_reduction_vector: conftest 提供的随机正数向量
        """
        if _reduction is None:
            pytest.skip("跳过：Numba 未安装，无法导入 reduction 模块")
        require_cuda()

        data = random_reduction_vector
        d_data = cuda.to_device(data)

        gpu_sum = self._run_atomic(d_data, REDUCTION_SIZE)
        cpu_sum = float(np.sum(data))

        rel_diff = abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-10)
        assert rel_diff < REDUCTION_TOLERANCE, (
            f"原子归约结果与 CPU 不一致，GPU={gpu_sum:.6f}，CPU={cpu_sum:.6f}，"
            f"相对误差: {rel_diff:.2e}"
        )

    def test_tree_reduction_correctness(self, random_reduction_vector):
        """测试树形归约（共享内存）的正确性。

        将 GPU 树形归约结果与 CPU numpy 求和对比，验证相对误差在容差内。

        Args:
            random_reduction_vector: conftest 提供的随机正数向量
        """
        if _reduction is None:
            pytest.skip("跳过：Numba 未安装，无法导入 reduction 模块")
        require_cuda()

        data = random_reduction_vector
        d_data = cuda.to_device(data)

        gpu_sum = self._run_tree(d_data, REDUCTION_SIZE)
        cpu_sum = float(np.sum(data))

        rel_diff = abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-10)
        assert rel_diff < REDUCTION_TOLERANCE, (
            f"树形归约结果与 CPU 不一致，GPU={gpu_sum:.6f}，CPU={cpu_sum:.6f}，"
            f"相对误差: {rel_diff:.2e}"
        )

    def test_warp_shuffle_reduction_correctness(self, random_reduction_vector):
        """测试 Warp Shuffle 归约的正确性。

        将 GPU Warp Shuffle 归约结果与 CPU numpy 求和对比，
        验证相对误差在容差内。

        Args:
            random_reduction_vector: conftest 提供的随机正数向量
        """
        require_cuda()

        data = random_reduction_vector
        d_data = cuda.to_device(data)

        gpu_sum = self._run_warp_shuffle(d_data, REDUCTION_SIZE)
        cpu_sum = float(np.sum(data))

        rel_diff = abs(gpu_sum - cpu_sum) / max(abs(cpu_sum), 1e-10)
        assert rel_diff < REDUCTION_TOLERANCE, (
            f"Warp Shuffle 归约结果与 CPU 不一致，GPU={gpu_sum:.6f}，"
            f"CPU={cpu_sum:.6f}，相对误差: {rel_diff:.2e}"
        )

    def test_three_reductions_consistent(self, random_reduction_vector):
        """测试三种归约方式的结果互相一致。

        使用相同输入分别运行原子归约、树形归约、Warp Shuffle 归约，
        验证三者结果两两之间的相对误差在容差范围内。

        Args:
            random_reduction_vector: conftest 提供的随机正数向量
        """
        if _reduction is None:
            pytest.skip("跳过：Numba 未安装，无法导入 reduction 模块")
        require_cuda()

        data = random_reduction_vector
        d_data = cuda.to_device(data)

        atomic_sum = self._run_atomic(d_data, REDUCTION_SIZE)
        tree_sum = self._run_tree(d_data, REDUCTION_SIZE)
        shuffle_sum = self._run_warp_shuffle(d_data, REDUCTION_SIZE)

        base = max(abs(atomic_sum), 1e-10)
        diff_at = abs(atomic_sum - tree_sum) / base
        diff_as = abs(atomic_sum - shuffle_sum) / base
        diff_ts = abs(tree_sum - shuffle_sum) / base

        assert diff_at < REDUCTION_TOLERANCE, (
            f"原子归约与树形归约不一致，atomic={atomic_sum:.6f}，"
            f"tree={tree_sum:.6f}，相对误差: {diff_at:.2e}"
        )
        assert diff_as < REDUCTION_TOLERANCE, (
            f"原子归约与 Warp Shuffle 归约不一致，atomic={atomic_sum:.6f}，"
            f"shuffle={shuffle_sum:.6f}，相对误差: {diff_as:.2e}"
        )
        assert diff_ts < REDUCTION_TOLERANCE, (
            f"树形归约与 Warp Shuffle 归约不一致，tree={tree_sum:.6f}，"
            f"shuffle={shuffle_sum:.6f}，相对误差: {diff_ts:.2e}"
        )
