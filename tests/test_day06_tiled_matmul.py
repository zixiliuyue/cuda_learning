#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 6 测试：Tiled 矩阵乘法 — 正确性验证
=========================================

测试内容：
  1. 朴素矩阵乘法（naive）结果与 CPU numpy 结果一致
  2. Tiled 矩阵乘法结果与 CPU numpy 结果一致
  3. Naive 和 Tiled 两种 GPU 实现的结果互相一致
  4. 使用 256x256 小规模矩阵，保证运行速度快
"""

import numpy as np
import pytest

# 导入 conftest 中的公共工具和常量
from conftest import (require_cuda, safe_import_day_module,  # noqa: E402
                      MATMUL_SIZE, TILE_WIDTH, MATMUL_TOLERANCE)

# 尝试安全导入被测模块
_matmul = safe_import_day_module("tiled_matmul", "day06_tiled_matmul")

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


class TestDay06TiledMatmul:
    """Day 6: Tiled 矩阵乘法的测试集合。"""

    def _run_naive(self, d_a, d_b, size):
        """辅助函数：在 GPU 上执行朴素矩阵乘法 Kernel。

        Args:
            d_a: 左矩阵（GPU 设备数组）
            d_b: 右矩阵（GPU 设备数组）
            size: 矩阵边长

        Returns:
            numpy.ndarray: 乘积结果（主机内存）
        """
        d_c = cuda.device_array_like(d_a)
        grid = ((size + TILE_WIDTH - 1) // TILE_WIDTH,
                (size + TILE_WIDTH - 1) // TILE_WIDTH)

        _matmul.matmul_naive_kernel[grid, (TILE_WIDTH, TILE_WIDTH)](
            d_a, d_b, d_c, size
        )
        cuda.synchronize()

        return d_c.copy_to_host()

    def _run_tiled(self, d_a, d_b, size):
        """辅助函数：在 GPU 上执行 Tiled 矩阵乘法 Kernel。

        Args:
            d_a: 左矩阵（GPU 设备数组）
            d_b: 右矩阵（GPU 设备数组）
            size: 矩阵边长

        Returns:
            numpy.ndarray: 乘积结果（主机内存）
        """
        d_c = cuda.device_array_like(d_a)
        grid = ((size + TILE_WIDTH - 1) // TILE_WIDTH,
                (size + TILE_WIDTH - 1) // TILE_WIDTH)

        _matmul.matmul_tiled_kernel[grid, (TILE_WIDTH, TILE_WIDTH)](
            d_a, d_b, d_c, size
        )
        cuda.synchronize()

        return d_c.copy_to_host()

    def test_naive_matmul_correctness(self, random_matmul_matrices, cpu_matmul):
        """测试朴素矩阵乘法的正确性。

        将 GPU naive kernel 的结果与 CPU numpy @ 运算的结果对比，
        验证最大误差在容差范围内。

        Args:
            random_matmul_matrices: conftest 提供的 fixture，返回 (a, b) 随机方阵
            cpu_matmul: conftest 提供的 fixture，返回 CPU 矩阵乘法函数
        """
        if _matmul is None:
            pytest.skip("跳过：Numba 未安装，无法导入 tiled_matmul 模块")
        require_cuda()

        a, b = random_matmul_matrices
        cpu_mul = cpu_matmul

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)

        gpu_result = self._run_naive(d_a, d_b, MATMUL_SIZE)
        cpu_result = cpu_mul(a, b, MATMUL_SIZE)

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < MATMUL_TOLERANCE, (
            f"GPU 朴素矩阵乘法与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {MATMUL_TOLERANCE:.2e}"
        )

    def test_tiled_matmul_correctness(self, random_matmul_matrices, cpu_matmul):
        """测试 Tiled 矩阵乘法的正确性。

        将 GPU tiled kernel 的结果与 CPU numpy @ 运算的结果对比，
        验证最大误差在容差范围内。

        Args:
            random_matmul_matrices: conftest 提供的 fixture，返回 (a, b) 随机方阵
            cpu_matmul: conftest 提供的 fixture，返回 CPU 矩阵乘法函数
        """
        if _matmul is None:
            pytest.skip("跳过：Numba 未安装，无法导入 tiled_matmul 模块")
        require_cuda()

        a, b = random_matmul_matrices
        cpu_mul = cpu_matmul

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)

        gpu_result = self._run_tiled(d_a, d_b, MATMUL_SIZE)
        cpu_result = cpu_mul(a, b, MATMUL_SIZE)

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < MATMUL_TOLERANCE, (
            f"GPU Tiled 矩阵乘法与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {MATMUL_TOLERANCE:.2e}"
        )

    def test_naive_and_tiled_consistent(self, random_matmul_matrices):
        """测试 naive 和 tiled 两种 GPU 实现的结果一致。

        使用相同的输入矩阵，分别运行朴素版和 Tiled 版 kernel，
        验证两者的最大误差在容差范围内。这确保了 Tiled 优化
        没有引入计算错误。

        Args:
            random_matmul_matrices: conftest 提供的 fixture，返回 (a, b) 随机方阵
        """
        if _matmul is None:
            pytest.skip("跳过：Numba 未安装，无法导入 tiled_matmul 模块")
        require_cuda()

        a, b = random_matmul_matrices

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)

        naive_result = self._run_naive(d_a, d_b, MATMUL_SIZE)
        tiled_result = self._run_tiled(d_a, d_b, MATMUL_SIZE)

        max_diff = np.max(np.abs(naive_result - tiled_result))
        assert max_diff < MATMUL_TOLERANCE, (
            f"Naive 和 Tiled 结果不一致，最大误差: {max_diff:.2e}，"
            f"容差: {MATMUL_TOLERANCE:.2e}"
        )

    def test_matmul_known_values(self):
        """测试 GPU 矩阵乘法对已知小矩阵的正确性。

        使用单位矩阵验证 A * I = A，这是矩阵乘法最基本的性质。
        矩阵尺寸使用 TILE_WIDTH (16x16) 以确保 Tile 对齐。
        """
        if _matmul is None:
            pytest.skip("跳过：Numba 未安装，无法导入 tiled_matmul 模块")
        require_cuda()

        size = TILE_WIDTH  # 16x16，确保是 TILE_WIDTH 的整数倍

        # 构造随机矩阵 A 和单位矩阵 I
        np.random.seed(42)
        a = np.random.uniform(-0.5, 0.5, size=size * size).astype(np.float32)
        # 单位矩阵
        b = np.eye(size, dtype=np.float32).flatten()

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)

        # A * I 应等于 A
        tiled_result = self._run_tiled(d_a, d_b, size)

        max_diff = np.max(np.abs(tiled_result - a))
        assert max_diff < MATMUL_TOLERANCE, (
            f"A * I != A，最大误差: {max_diff:.2e}，容差: {MATMUL_TOLERANCE:.2e}"
        )