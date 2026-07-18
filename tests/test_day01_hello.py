#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 1 测试：Hello CUDA — GPU 可用性检查与 Kernel 启动
=======================================================

测试内容：
  1. check_gpu() 函数能正常返回布尔值
  2. hello_cuda_kernel 可以正常启动且不抛出异常
"""

import os
import sys

import pytest

# 导入 conftest 中的公共工具
from conftest import require_cuda, CUDA_AVAILABLE, safe_import_day_module  # noqa: E402

# 尝试安全导入被测模块
_hello = safe_import_day_module("hello_cuda", "day01_hello_cuda")

# 尝试导入 numba.cuda（用于 synchronize 等）
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


class TestDay01HelloCUDA:
    """Day 1: Hello CUDA 的测试集合。"""

    def test_check_gpu_returns_bool(self):
        """测试 check_gpu 函数返回布尔值。

        无论 GPU 是否可用，函数都应返回 True 或 False，
        不应抛出异常。
        """
        if _hello is None:
            pytest.skip("跳过：Numba 未安装，无法导入 hello_cuda 模块")

        result = _hello.check_gpu()
        assert isinstance(result, bool), (
            f"check_gpu() 应返回布尔值，实际返回类型: {type(result)}"
        )

    def test_check_gpu_detects_gpu(self):
        """测试 check_gpu 在 CUDA 可用时返回 True。

        如果 CUDA 不可用则跳过此测试。
        """
        if _hello is None:
            pytest.skip("跳过：Numba 未安装，无法导入 hello_cuda 模块")
        require_cuda()

        assert _hello.check_gpu() is True, "GPU 可用时 check_gpu 应返回 True"

    def test_kernel_launch_no_exception(self):
        """测试 hello_cuda_kernel 启动不抛出异常。

        使用小规模 Grid/Block 配置（2 Block x 4 Thread），
        仅验证 Kernel 能正常编译和启动，不验证输出内容。
        """
        if _hello is None:
            pytest.skip("跳过：Numba 未安装，无法导入 hello_cuda 模块")
        require_cuda()

        num_blocks = 2
        threads_per_block = 4

        # 启动 Kernel，不应抛出任何异常
        _hello.hello_cuda_kernel[num_blocks, threads_per_block]()

        # 等待所有 GPU 线程执行完毕
        cuda.synchronize()