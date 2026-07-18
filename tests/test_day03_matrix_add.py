#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3 测试：矩阵加法 — 正确性验证
===================================

测试内容：
  1. GPU 矩阵加法结果与 CPU numpy 结果一致（误差 < 1e-5）
  2. 测试使用小规模矩阵（64x64），确保运行速度快
"""

import numpy as np
import pytest

# 导入 conftest 中的公共工具和常量
from conftest import (require_cuda, safe_import_day_module,  # noqa: E402
                      MATRIX_SIZE, BLOCK_X, BLOCK_Y, TOLERANCE)

# 尝试安全导入被测模块
_matrix = safe_import_day_module("matrix_add", "day03_matrix_add")

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


class TestDay03MatrixAdd:
    """Day 3: 矩阵加法的测试集合。"""

    def test_gpu_matrix_add_correctness(self, random_matrices, cpu_matrix_add):
        """测试 GPU 矩阵加法正确性。

        使用随机生成的 float32 矩阵（一维行优先存储），
        分别用 GPU kernel 和 CPU numpy 计算加法，
        验证两者结果的最大绝对误差在容差范围内。

        Args:
            random_matrices: conftest 提供的 fixture，返回 (a, b) 随机矩阵
            cpu_matrix_add: conftest 提供的 fixture，返回 CPU 加法函数
        """
        if _matrix is None:
            pytest.skip("跳过：Numba 未安装，无法导入 matrix_add 模块")
        require_cuda()

        a, b = random_matrices
        cpu_add = cpu_matrix_add

        # 将数据传输到 GPU
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(a)

        # 计算 Grid 尺寸（2D）
        grid_x = (MATRIX_SIZE + BLOCK_X - 1) // BLOCK_X
        grid_y = (MATRIX_SIZE + BLOCK_Y - 1) // BLOCK_Y

        # 启动 Kernel
        _matrix.matrix_add_kernel[(grid_x, grid_y), (BLOCK_X, BLOCK_Y)](
            d_a, d_b, d_c, MATRIX_SIZE, MATRIX_SIZE
        )
        cuda.synchronize()

        # 将结果拷回主机
        gpu_result = d_c.copy_to_host()

        # CPU 参考计算
        cpu_result = cpu_add(a, b)

        # 验证：最大绝对误差应小于容差
        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < TOLERANCE, (
            f"GPU 矩阵加法结果与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_gpu_matrix_add_known_values(self):
        """测试 GPU 矩阵加法对已知输入的正确性。

        使用手工构造的 4x4 矩阵进行测试，验证每个元素都被正确相加。
        """
        if _matrix is None:
            pytest.skip("跳过：Numba 未安装，无法导入 matrix_add 模块")
        require_cuda()

        rows, cols = 4, 4
        a = np.array([1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 10, 11, 12,
                       13, 14, 15, 16], dtype=np.float32)
        b = np.array([16, 15, 14, 13,
                       12, 11, 10, 9,
                       8, 7, 6, 5,
                       4, 3, 2, 1], dtype=np.float32)
        expected = a + b  # 每个元素都是 17

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(a)

        _matrix.matrix_add_kernel[(1, 1), (4, 4)](d_a, d_b, d_c, cols, rows)
        cuda.synchronize()

        gpu_result = d_c.copy_to_host()

        # 逐元素对比
        np.testing.assert_allclose(gpu_result, expected, atol=TOLERANCE,
                                   err_msg="GPU 矩阵加法已知值验证失败")