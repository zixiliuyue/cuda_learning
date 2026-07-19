#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 7 测试：全局内存合并访问 — 正确性验证
============================================

测试内容：
  1. 行优先读取 Kernel（合并访问）的行求和结果与 CPU numpy 一致
  2. 列优先读取 Kernel（非合并访问）的列求和结果与 CPU numpy 一致
  3. 合并访问与非合并访问是同一矩阵、两个方向的求和，
     验证 GPU 计算的总和（行求和之和 == 列求和之和）互相一致
  4. 使用 128x128 小规模矩阵，保证运行速度快

说明：
  合并 / 非合并访问指的是内存访问模式的差异，二者计算的是不同方向
  的求和（行 vs 列），因此不能直接逐元素比较。这里通过“行求和的总和”
  与“列求和的总和”应相等这一数学不变量来验证两个 Kernel 结果一致。
"""

import numpy as np
import pytest

# 导入 conftest 中的公共工具和常量
from conftest import (require_cuda, safe_import_day_module,  # noqa: E402
                      COALESCED_ROWS, COALESCED_COLS, TOLERANCE)

# 尝试安全导入被测模块
_coalesced = safe_import_day_module("coalesced_access", "day07_coalesced_access")

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None

# Block 尺寸（与源码保持一致）
BLOCK_SIZE = 256


class TestDay07CoalescedAccess:
    """Day 7: 全局内存合并访问的测试集合。"""

    def _run_sum_rows(self, d_data, rows, cols):
        """辅助函数：在 GPU 上执行行优先（合并访问）求和 Kernel。

        Args:
            d_data: 输入矩阵（一维行优先，GPU 设备数组）
            rows: 矩阵行数
            cols: 矩阵列数

        Returns:
            numpy.ndarray: 每一行的求和结果（长度为 rows）
        """
        d_result = cuda.device_array(rows, dtype=np.float32)
        grid = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE

        _coalesced.sum_rows_kernel[grid, BLOCK_SIZE](d_data, d_result, rows, cols)
        cuda.synchronize()

        return d_result.copy_to_host()

    def _run_sum_cols(self, d_data, rows, cols):
        """辅助函数：在 GPU 上执行列优先（非合并访问）求和 Kernel。

        Args:
            d_data: 输入矩阵（一维行优先，GPU 设备数组）
            rows: 矩阵行数
            cols: 矩阵列数

        Returns:
            numpy.ndarray: 每一列的求和结果（长度为 cols）
        """
        d_result = cuda.device_array(cols, dtype=np.float32)
        grid = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE

        # 注意源码 sum_cols_kernel 的参数顺序为 (data, result, cols, rows)
        _coalesced.sum_cols_kernel[grid, BLOCK_SIZE](d_data, d_result, cols, rows)
        cuda.synchronize()

        return d_result.copy_to_host()

    def test_coalesced_rows_correctness(self, random_coalesced_matrix):
        """测试行优先读取 Kernel（合并访问）的正确性。

        将 GPU 行求和结果与 CPU numpy 按行求和的结果对比，
        验证最大绝对误差在容差范围内。

        Args:
            random_coalesced_matrix: conftest 提供的 fixture，一维行优先矩阵
        """
        if _coalesced is None:
            pytest.skip("跳过：Numba 未安装，无法导入 coalesced_access 模块")
        require_cuda()

        flat = random_coalesced_matrix
        matrix = flat.reshape(COALESCED_ROWS, COALESCED_COLS)

        d_data = cuda.to_device(flat)
        gpu_result = self._run_sum_rows(d_data, COALESCED_ROWS, COALESCED_COLS)

        cpu_result = np.sum(matrix, axis=1)

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < TOLERANCE, (
            f"GPU 合并访问（行求和）与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_non_coalesced_cols_correctness(self, random_coalesced_matrix):
        """测试列优先读取 Kernel（非合并访问）的正确性。

        将 GPU 列求和结果与 CPU numpy 按列求和的结果对比，
        验证最大绝对误差在容差范围内。

        Args:
            random_coalesced_matrix: conftest 提供的 fixture，一维行优先矩阵
        """
        if _coalesced is None:
            pytest.skip("跳过：Numba 未安装，无法导入 coalesced_access 模块")
        require_cuda()

        flat = random_coalesced_matrix
        matrix = flat.reshape(COALESCED_ROWS, COALESCED_COLS)

        d_data = cuda.to_device(flat)
        gpu_result = self._run_sum_cols(d_data, COALESCED_ROWS, COALESCED_COLS)

        cpu_result = np.sum(matrix, axis=0)

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < TOLERANCE, (
            f"GPU 非合并访问（列求和）与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_coalesced_and_non_coalesced_consistent(self, random_coalesced_matrix):
        """测试合并访问与非合并访问两个 Kernel 的结果一致。

        合并访问按行求和、非合并访问按列求和，二者虽方向不同，
        但对同一矩阵而言“所有行之和”必然等于“所有列之和”（即矩阵总和）。
        这里以此不变量验证两个 Kernel 的结果相互一致。

        Args:
            random_coalesced_matrix: conftest 提供的 fixture，一维行优先矩阵
        """
        if _coalesced is None:
            pytest.skip("跳过：Numba 未安装，无法导入 coalesced_access 模块")
        require_cuda()

        flat = random_coalesced_matrix
        d_data = cuda.to_device(flat)

        row_result = self._run_sum_rows(d_data, COALESCED_ROWS, COALESCED_COLS)
        col_result = self._run_sum_cols(d_data, COALESCED_ROWS, COALESCED_COLS)

        row_total = float(np.sum(row_result))
        col_total = float(np.sum(col_result))

        diff = abs(row_total - col_total) / max(abs(row_total), 1e-10)
        assert diff < 1e-3, (
            f"合并访问与非合并访问的总和不一致，"
            f"行求和总和={row_total:.6f}，列求和总和={col_total:.6f}，"
            f"相对误差: {diff:.2e}"
        )

    def test_coalesced_known_values(self):
        """测试行 / 列求和 Kernel 对已知小矩阵的正确性。

        使用全 1 的 32x32 矩阵，每一行 / 每一列的求和结果都应等于列数 / 行数。
        """
        if _coalesced is None:
            pytest.skip("跳过：Numba 未安装，无法导入 coalesced_access 模块")
        require_cuda()

        rows, cols = 32, 32
        flat = np.ones(rows * cols, dtype=np.float32)
        d_data = cuda.to_device(flat)

        row_result = self._run_sum_rows(d_data, rows, cols)
        col_result = self._run_sum_cols(d_data, rows, cols)

        np.testing.assert_allclose(
            row_result, np.full(rows, cols, dtype=np.float32),
            atol=TOLERANCE, err_msg="全 1 矩阵行求和结果错误"
        )
        np.testing.assert_allclose(
            col_result, np.full(cols, rows, dtype=np.float32),
            atol=TOLERANCE, err_msg="全 1 矩阵列求和结果错误"
        )
