#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2 测试：向量加法 — 正确性验证
===================================

测试内容：
  1. GPU 向量加法结果与 CPU numpy 结果一致（误差 < 1e-5）
  2. 测试使用小规模数据集（SIZE=1024），确保运行速度快
"""

import numpy as np
import pytest

# 导入 conftest 中的公共工具和常量
from conftest import (require_cuda, safe_import_day_module,  # noqa: E402
                      VECTOR_SIZE, THREADS_PER_BLOCK, TOLERANCE)

# 尝试安全导入被测模块
_vector = safe_import_day_module("vector_add", "day02_vector_add")

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


class TestDay02VectorAdd:
    """Day 2: 向量加法的测试集合。"""

    def test_gpu_vector_add_correctness(self, random_vectors, cpu_vector_add):
        """测试 GPU 向量加法正确性。

        使用随机生成的 float32 向量，分别用 GPU kernel 和 CPU numpy
        计算加法，验证两者结果的最大绝对误差在容差范围内。

        Args:
            random_vectors: conftest 提供的 fixture，返回 (a, b) 随机向量
            cpu_vector_add: conftest 提供的 fixture，返回 CPU 加法函数
        """
        if _vector is None:
            pytest.skip("跳过：Numba 未安装，无法导入 vector_add 模块")
        require_cuda()

        a, b = random_vectors
        cpu_add = cpu_vector_add

        # 将数据传输到 GPU
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(a)

        # 计算 Grid 尺寸
        grid_size = (VECTOR_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

        # 启动 Kernel
        _vector.vector_add_kernel[grid_size, THREADS_PER_BLOCK](
            d_a, d_b, d_c, VECTOR_SIZE
        )
        cuda.synchronize()

        # 将结果拷回主机
        gpu_result = d_c.copy_to_host()

        # CPU 参考计算
        cpu_result = cpu_add(a, b)

        # 验证：最大绝对误差应小于容差
        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < TOLERANCE, (
            f"GPU 向量加法结果与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_gpu_vector_add_element_wise(self):
        """测试 GPU 向量加法的逐元素正确性。

        使用手工构造的小向量 [1, 2, 3, ...] 和 [10, 20, 30, ...]，
        验证每个元素都被正确相加。
        """
        if _vector is None:
            pytest.skip("跳过：Numba 未安装，无法导入 vector_add 模块")
        require_cuda()

        n = 16
        a = np.arange(1, n + 1, dtype=np.float32)    # [1, 2, ..., 16]
        b = np.arange(10, 10 + n, dtype=np.float32)  # [10, 11, ..., 25]
        expected = a + b

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(a)

        threads = 16
        _vector.vector_add_kernel[1, threads](d_a, d_b, d_c, n)
        cuda.synchronize()

        gpu_result = d_c.copy_to_host()

        # 逐元素对比
        np.testing.assert_allclose(gpu_result, expected, atol=TOLERANCE,
                                   err_msg="GPU 向量加法逐元素验证失败")