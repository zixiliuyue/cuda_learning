#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 4 测试：矩阵转置 — 正确性验证
===================================

测试内容：
  1. GPU 矩阵转置结果与 CPU numpy 结果一致
  2. 转置后再转置等于原矩阵（双重转置不变性）
"""

import numpy as np
import pytest

# 导入 conftest 中的公共工具和常量
from conftest import (require_cuda, safe_import_day_module,  # noqa: E402
                      MATRIX_SIZE, BLOCK_X, BLOCK_Y, TOLERANCE)

# 尝试安全导入被测模块
_transpose = safe_import_day_module("matrix_transpose", "day04_matrix_transpose")

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


class TestDay04MatrixTranspose:
    """Day 4: 矩阵转置的测试集合。"""

    def _run_transpose_kernel(self, d_in, width, height):
        """辅助函数：在 GPU 上执行转置 Kernel 并返回结果。

        Args:
            d_in: 输入数据（GPU 设备数组）
            width: 矩阵列数
            height: 矩阵行数

        Returns:
            numpy.ndarray: 转置后的结果（主机内存）
        """
        d_out = cuda.device_array_like(d_in)

        grid_x = (width + BLOCK_X - 1) // BLOCK_X
        grid_y = (height + BLOCK_Y - 1) // BLOCK_Y

        _transpose.matrix_transpose_kernel[(grid_x, grid_y), (BLOCK_X, BLOCK_Y)](
            d_in, d_out, width, height
        )
        cuda.synchronize()

        return d_out.copy_to_host()

    def test_gpu_transpose_correctness(self):
        """测试 GPU 矩阵转置正确性。

        使用顺序填充的矩阵，验证转置后每个元素位置正确。
        """
        if _transpose is None:
            pytest.skip("跳过：Numba 未安装，无法导入 matrix_transpose 模块")
        require_cuda()

        width = MATRIX_SIZE
        height = MATRIX_SIZE
        n = width * height

        # 使用顺序值填充，便于验证转置后的位置
        h_in = np.arange(n, dtype=np.float32)

        d_in = cuda.to_device(h_in)
        gpu_result = self._run_transpose_kernel(d_in, width, height)

        # CPU 参考转置
        cpu_result = h_in.reshape(height, width).T.flatten()

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < TOLERANCE, (
            f"GPU 矩阵转置结果与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_double_transpose_restores_original(self):
        """测试转置的逆不变性：转置两次应等于原矩阵。

        对同一矩阵连续执行两次 GPU 转置，验证结果与原矩阵一致。
        这是转置操作最重要的数学性质之一。
        """
        if _transpose is None:
            pytest.skip("跳过：Numba 未安装，无法导入 matrix_transpose 模块")
        require_cuda()

        width = MATRIX_SIZE
        height = MATRIX_SIZE
        n = width * height

        h_in = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)

        # 第一次转置：M x N -> N x M
        d_first = cuda.to_device(h_in)
        first_result = self._run_transpose_kernel(d_first, width, height)

        # 第二次转置：N x M -> M x N（还原）
        d_second = cuda.to_device(first_result)
        second_result = self._run_transpose_kernel(d_second, height, width)

        # 验证两次转置后恢复原矩阵
        max_diff = np.max(np.abs(second_result - h_in))
        assert max_diff < TOLERANCE, (
            f"双重转置未恢复原矩阵，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_transpose_known_values(self):
        """测试 GPU 矩阵转置对已知小矩阵的正确性。

        使用手工构造的 3x4 矩阵，验证转置后得到正确的 4x3 矩阵。
        """
        if _transpose is None:
            pytest.skip("跳过：Numba 未安装，无法导入 matrix_transpose 模块")
        require_cuda()

        rows, cols = 3, 4
        # 3x4 矩阵（行优先）：
        # [[1, 2, 3, 4],
        #  [5, 6, 7, 8],
        #  [9, 10, 11, 12]]
        h_in = np.array([1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12], dtype=np.float32)

        # 期望的转置结果（4x3，行优先）：
        # [[1, 5, 9],
        #  [2, 6, 10],
        #  [3, 7, 11],
        #  [4, 8, 12]]
        expected = np.array([1, 5, 9,
                              2, 6, 10,
                              3, 7, 11,
                              4, 8, 12], dtype=np.float32)

        d_in = cuda.to_device(h_in)
        gpu_result = self._run_transpose_kernel(d_in, cols, rows)

        np.testing.assert_allclose(gpu_result, expected, atol=TOLERANCE,
                                   err_msg="GPU 矩阵转置已知值验证失败")