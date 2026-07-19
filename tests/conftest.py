#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA 30 天学习项目 — pytest 公共配置与 fixture
================================================

本文件为整个测试套件提供：
  1. CUDA/GPU 可用性检查，不可用时自动跳过所有需要 GPU 的测试
  2. 安全的 day 模块导入函数（处理源码中 sys.exit 的情况）
  3. 公共 fixture：随机数组生成、CPU 参考计算
  4. 常量定义：测试容差、小数据集尺寸等

使用方式：
  在项目根目录下运行: python -m pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 测试容差 — 单精度浮点数（float32）的加法/矩阵运算误差上限
TOLERANCE = 1e-5

# 矩阵乘法容差 — 乘法累加误差稍大，使用较宽松的阈值
MATMUL_TOLERANCE = 1e-2

# 小规模向量长度（用于 Day2 向量加法测试）
VECTOR_SIZE = 1024

# 小规模矩阵行列数（用于 Day3 矩阵加法、Day4 矩阵转置）
MATRIX_SIZE = 64

# Day6 tiled matmul 使用的矩阵尺寸（必须是 TILE_WIDTH 的整数倍）
MATMUL_SIZE = 256

# 每个 Block 中的线程数
THREADS_PER_BLOCK = 256

# 2D Block 尺寸
BLOCK_X = 16
BLOCK_Y = 16

# Day6 Tiled 矩阵乘法的 Tile 宽度
TILE_WIDTH = 16

# Day7 合并访问测试使用的矩阵行列数（小规模，保证运行速度快）
COALESCED_ROWS = 128
COALESCED_COLS = 128

# Day9 归约测试使用的向量长度
# 使用 BLOCK_SIZE 的整数倍，方便 Block 级归约对齐
REDUCTION_SIZE = 4096

# 归约测试容差 — 大量浮点累加存在误差，使用相对误差阈值
REDUCTION_TOLERANCE = 1e-1

# Day9 / Day10 每个 Block 的线程数（须为 32 的倍数，供 Warp 归约使用）
REDUCTION_BLOCK_SIZE = 256

# Day10 CUDA 流测试使用的向量长度（须为 NUM_STREAMS 的整数倍）
STREAM_SIZE = 4096

# Day10 流数量
NUM_STREAMS = 4

# ---------------------------------------------------------------------------
# CUDA 可用性检查
# ---------------------------------------------------------------------------

# 尝试导入 numba.cuda，判断运行环境是否支持 CUDA
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    cuda = None
    CUDA_AVAILABLE = False


def require_cuda():
    """检查 CUDA 是否可用，不可用时抛出 SkipTest 跳过当前测试。

    在每个需要真实 GPU 执行的测试函数中调用此函数，
    确保在没有 GPU 的环境（如 CI 服务器、macOS）中优雅跳过。
    """
    if not CUDA_AVAILABLE:
        pytest.skip("跳过：未检测到可用的 NVIDIA GPU 或 Numba CUDA 不可用")


def require_numba():
    """检查 Numba 是否已安装，未安装时跳过当前测试。

    部分源码模块（如 hello_cuda.py）在 numba 不可用时会
    调用 sys.exit(1)，导致直接 import 会终止进程。
    此函数用于在导入源码模块之前先确认 numba 可用。
    """
    if not hasattr(sys.modules.get('numba', None), 'cuda'):
        pytest.skip("跳过：Numba 未安装，请运行 pip install numba")


def safe_import_day_module(module_name, day_dir_name):
    """安全导入指定 day 目录下的 Python 模块。

    源码模块（如 hello_cuda.py）在 numba 未安装时可能调用 sys.exit(1)，
    直接 import 会导致 pytest 收集阶段崩溃。此函数通过捕获 SystemExit
    异常来安全处理这种情况。

    Args:
        module_name: 要导入的模块名（不含 .py 后缀），如 'hello_cuda'
        day_dir_name: day 目录名，如 'day01_hello_cuda'

    Returns:
        module 对象，若导入失败则返回 None
    """
    # 先确认 numba 可用，否则源码模块会 sys.exit
    try:
        import numba  # noqa: F401
    except ImportError:
        return None

    # 将 day 的 python 目录加入搜索路径
    day_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # tests/
        "..", day_dir_name, "python"
    )
    abs_day_dir = os.path.abspath(day_dir)
    if abs_day_dir not in sys.path:
        sys.path.insert(0, abs_day_dir)

    try:
        module = __import__(module_name, fromlist=[''])
        return module
    except (ImportError, SystemExit):
        return None


# ---------------------------------------------------------------------------
# 公共 fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def random_vectors():
    """生成一对随机浮点向量（float32），用于向量加法等测试。

    返回:
        tuple: (a, b)，两个形状为 (VECTOR_SIZE,) 的 numpy 数组
    """
    a = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, size=VECTOR_SIZE).astype(np.float32)
    return a, b


@pytest.fixture
def random_matrices():
    """生成一对随机浮点矩阵（float32），用于矩阵加法/转置等测试。

    矩阵以一维数组形式返回（行优先），形状为 (MATRIX_SIZE * MATRIX_SIZE,)。

    返回:
        tuple: (a, b)，两个一维 numpy 数组
    """
    n = MATRIX_SIZE * MATRIX_SIZE
    a = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)
    return a, b


@pytest.fixture
def random_matmul_matrices():
    """生成一对用于矩阵乘法测试的随机方阵（float32）。

    矩阵以一维数组形式返回（行优先），形状为 (MATMUL_SIZE * MATMUL_SIZE,)。

    返回:
        tuple: (a, b)，两个一维 numpy 数组
    """
    n = MATMUL_SIZE * MATMUL_SIZE
    a = np.random.uniform(-0.5, 0.5, size=n).astype(np.float32)
    b = np.random.uniform(-0.5, 0.5, size=n).astype(np.float32)
    return a, b


@pytest.fixture
def cpu_vector_add():
    """返回 CPU 版向量加法函数，用于作为参考结果。

    返回:
        callable: 接受 (a, b) 两个 numpy 数组，返回 a + b
    """
    def _add(a, b):
        return a + b
    return _add


@pytest.fixture
def cpu_matrix_add():
    """返回 CPU 版矩阵加法函数，用于作为参考结果。

    返回:
        callable: 接受 (a, b) 两个一维 numpy 数组，返回 a + b
    """
    def _add(a, b):
        return a + b
    return _add


@pytest.fixture
def cpu_matrix_transpose():
    """返回 CPU 版矩阵转置函数，用于作为参考结果。

    矩阵以一维行优先数组表示，转置后仍为行优先。

    返回:
        callable: 接受 (matrix, width, height)，返回转置后的一维数组
    """
    def _transpose(matrix, width, height):
        return matrix.reshape(height, width).T.flatten()
    return _transpose


@pytest.fixture
def cpu_matmul():
    """返回 CPU 版矩阵乘法函数，用于作为参考结果。

    矩阵以一维行优先数组表示。

    返回:
        callable: 接受 (a, b, size)，返回乘积结果的一维数组
    """
    def _matmul(a, b, size):
        a_2d = a.reshape(size, size)
        b_2d = b.reshape(size, size)
        return (a_2d @ b_2d).flatten()
    return _matmul


@pytest.fixture
def random_coalesced_matrix():
    """生成用于合并访问测试的随机矩阵（float32）。

    以一维行优先数组形式返回，形状为 (COALESCED_ROWS * COALESCED_COLS,)，
    供 Day7 行优先（合并）/ 列优先（非合并）求和 Kernel 使用。

    返回:
        numpy.ndarray: 一维 float32 数组
    """
    n = COALESCED_ROWS * COALESCED_COLS
    return np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)


@pytest.fixture
def random_reduction_vector():
    """生成用于归约测试的随机正数向量（float32）。

    使用正数区间 [0, 10) 以减小浮点累加时的相消误差，
    供 Day9 原子 / 树形 / Warp Shuffle 归约测试使用。

    返回:
        numpy.ndarray: 形状为 (REDUCTION_SIZE,) 的一维 float32 数组
    """
    return np.random.uniform(0.0, 10.0, size=REDUCTION_SIZE).astype(np.float32)


@pytest.fixture
def random_stream_vectors():
    """生成一对用于 CUDA 流测试的随机向量（float32）。

    形状均为 (STREAM_SIZE,)，供 Day10 串行执行与多流并发执行测试使用。

    返回:
        tuple: (a, b)，两个一维 numpy 数组
    """
    a = np.random.uniform(-1.0, 1.0, size=STREAM_SIZE).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, size=STREAM_SIZE).astype(np.float32)
    return a, b