#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 5 测试：GPU 设备查询 — 基本功能验证
========================================

测试内容：
  1. cuda.is_available() 调用不抛出异常
  2. 获取设备信息函数不抛出异常
"""

import pytest

# 导入 conftest 中的公共工具
from conftest import require_cuda, CUDA_AVAILABLE  # noqa: E402

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


class TestDay05DeviceQuery:
    """Day 5: GPU 设备查询的测试集合。"""

    def test_cuda_is_available_no_exception(self):
        """测试 cuda.is_available() 调用不抛出异常。

        无论环境中是否有 GPU，此函数都应安全返回布尔值。
        """
        if cuda is None:
            pytest.skip("跳过：Numba 未安装")

        result = cuda.is_available()
        assert isinstance(result, bool), (
            f"cuda.is_available() 应返回布尔值，实际返回类型: {type(result)}"
        )

    def test_get_current_device_no_exception(self):
        """测试 cuda.get_current_device() 在 GPU 可用时不抛出异常。

        获取当前设备对象，并验证可以读取设备属性。
        """
        if cuda is None:
            pytest.skip("跳过：Numba 未安装")
        require_cuda()

        device = cuda.get_current_device()

        # 验证设备对象的基本属性可访问
        name = device.name
        assert name is not None, "设备名称不应为 None"

        cc = device.compute_capability
        assert isinstance(cc, tuple) and len(cc) == 2, (
            f"计算能力应为 (major, minor) 元组，实际: {cc}"
        )
        # 计算能力主版本号应大于 0
        assert cc[0] > 0, f"计算能力主版本号应 > 0，实际: {cc[0]}"

    def test_device_attributes_readable(self):
        """测试可以读取 GPU 设备的各项属性且不抛出异常。

        验证的属性包括：设备名称、计算能力、多处理器数量等。
        """
        if cuda is None:
            pytest.skip("跳过：Numba 未安装")
        require_cuda()

        device = cuda.get_current_device()

        # 以下属性访问均不应抛出异常
        _name = device.name.decode() if isinstance(device.name, bytes) else device.name
        _cc = device.compute_capability
        _max_threads = device.MAX_THREADS_PER_BLOCK

        # 验证多处理器数量为正整数
        sm_count = getattr(device, 'MULTIPROCESSOR_COUNT', None)
        if sm_count is not None:
            assert isinstance(sm_count, int) and sm_count > 0, (
                f"多处理器数量应为正整数，实际: {sm_count}"
            )