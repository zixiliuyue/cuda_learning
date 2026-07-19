#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 10 测试：CUDA 流与异步执行 — 正确性验证
==============================================

测试内容：
  1. 默认流（串行执行）的结果与 CPU 参考计算一致
  2. 多流并发执行的结果与 CPU 参考计算一致
  3. 串行执行与多流并发执行的结果互相一致

被测计算：result = (a + b + 1.0) * 2.0
  由源码 streams.py 的 vec_add_kernel（a + b + 1.0）与
  vec_scale_kernel（* 2.0）两级 Kernel 组合而成。
"""

import numpy as np
import pytest

# 导入 conftest 中的公共工具和常量
from conftest import (require_cuda, safe_import_day_module,  # noqa: E402
                      STREAM_SIZE, NUM_STREAMS, TOLERANCE)

# 尝试安全导入被测模块
_streams = safe_import_day_module("streams", "day10_streams")

# 尝试导入 numba.cuda
try:
    from numba import cuda  # noqa: E402
except ImportError:
    cuda = None


# 每个 Block 的线程数（与源码保持一致）
BLOCK_SIZE = 256


def _cpu_reference(a, b):
    """CPU 参考计算: (a + b + 1.0) * 2.0，与源码 vec_cpu 一致。"""
    return (a + b + 1.0) * 2.0


class TestDay10Streams:
    """Day 10: CUDA 流与异步执行的测试集合。"""

    def _run_serial(self, a, b, n):
        """辅助函数：在默认流上串行执行两级 Kernel。

        Args:
            a: 输入向量 a（主机数组）
            b: 输入向量 b（主机数组）
            n: 元素个数

        Returns:
            numpy.ndarray: 计算结果（主机内存）
        """
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(a)
        d_r = cuda.device_array_like(a)

        grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

        _streams.vec_add_kernel[grid, BLOCK_SIZE](d_a, d_b, d_c, n)
        cuda.synchronize()
        _streams.vec_scale_kernel[grid, BLOCK_SIZE](d_c, d_r, n)
        cuda.synchronize()

        return d_r.copy_to_host()

    def _run_multi_stream(self, a, b, n):
        """辅助函数：使用多个流分块并发执行两级 Kernel。

        将数据均分到 NUM_STREAMS 个流中，每个流独立完成
        传输 -> vec_add -> vec_scale -> 传回。

        Args:
            a: 输入向量 a（主机数组）
            b: 输入向量 b（主机数组）
            n: 元素个数

        Returns:
            numpy.ndarray: 拼接后的完整计算结果（主机内存）
        """
        chunk = n // NUM_STREAMS
        streams = [cuda.stream() for _ in range(NUM_STREAMS)]
        result = np.zeros(n, dtype=np.float32)

        grid_chunk = (chunk + BLOCK_SIZE - 1) // BLOCK_SIZE

        for i in range(NUM_STREAMS):
            offset = i * chunk
            a_chunk = np.ascontiguousarray(a[offset:offset + chunk])
            b_chunk = np.ascontiguousarray(b[offset:offset + chunk])

            with streams[i]:
                d_a_chunk = cuda.to_device(a_chunk, stream=streams[i])
                d_b_chunk = cuda.to_device(b_chunk, stream=streams[i])
                d_c_chunk = cuda.device_array_like(a_chunk)
                d_r_chunk = cuda.device_array_like(a_chunk)

                _streams.vec_add_kernel[grid_chunk, BLOCK_SIZE, streams[i]](
                    d_a_chunk, d_b_chunk, d_c_chunk, chunk
                )
                _streams.vec_scale_kernel[grid_chunk, BLOCK_SIZE, streams[i]](
                    d_c_chunk, d_r_chunk, chunk
                )

                h_r_chunk = d_r_chunk.copy_to_host(stream=streams[i])
                result[offset:offset + chunk] = h_r_chunk

        # 等待所有流完成
        for s in streams:
            s.synchronize()

        return result

    def test_serial_correctness(self, random_stream_vectors):
        """测试默认流（串行执行）结果的正确性。

        将 GPU 串行执行结果与 CPU 参考计算对比，验证最大误差在容差内。

        Args:
            random_stream_vectors: conftest 提供的 fixture，返回 (a, b)
        """
        if _streams is None:
            pytest.skip("跳过：Numba 未安装，无法导入 streams 模块")
        require_cuda()

        a, b = random_stream_vectors

        gpu_result = self._run_serial(a, b, STREAM_SIZE)
        cpu_result = _cpu_reference(a, b)

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < TOLERANCE, (
            f"串行执行结果与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_multi_stream_correctness(self, random_stream_vectors):
        """测试多流并发执行结果的正确性。

        将 GPU 多流执行结果与 CPU 参考计算对比，验证最大误差在容差内。

        Args:
            random_stream_vectors: conftest 提供的 fixture，返回 (a, b)
        """
        if _streams is None:
            pytest.skip("跳过：Numba 未安装，无法导入 streams 模块")
        require_cuda()

        a, b = random_stream_vectors

        gpu_result = self._run_multi_stream(a, b, STREAM_SIZE)
        cpu_result = _cpu_reference(a, b)

        max_diff = np.max(np.abs(gpu_result - cpu_result))
        assert max_diff < TOLERANCE, (
            f"多流并发结果与 CPU 不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )

    def test_serial_and_multi_stream_consistent(self, random_stream_vectors):
        """测试串行执行与多流并发执行的结果一致。

        使用相同输入，分别以默认流串行、以多流并发计算，
        验证两者逐元素结果的最大误差在容差范围内。
        这确保了多流分块处理没有引入计算错误。

        Args:
            random_stream_vectors: conftest 提供的 fixture，返回 (a, b)
        """
        if _streams is None:
            pytest.skip("跳过：Numba 未安装，无法导入 streams 模块")
        require_cuda()

        a, b = random_stream_vectors

        serial_result = self._run_serial(a, b, STREAM_SIZE)
        multi_result = self._run_multi_stream(a, b, STREAM_SIZE)

        max_diff = np.max(np.abs(serial_result - multi_result))
        assert max_diff < TOLERANCE, (
            f"串行执行与多流并发结果不一致，最大误差: {max_diff:.2e}，"
            f"容差: {TOLERANCE:.2e}"
        )
