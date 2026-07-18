# -*- coding: utf-8 -*-
# cython: language_level=3
# distutils: language = c++
# =============================================================================
# 文件: cython_wrapper.pyx
# 说明: Day 38 —— Cython 封装层，把 src/main.cu 中的 CUDA C 函数暴露为 Python 函数
#
# Cython 混合编程要点：
#   1. `cdef extern from "cuda_wrapper.h"` 声明外部 C 函数原型；
#   2. `cimport numpy` 引入 NumPy 的 C 级 API（编译期），普通 `import numpy` 是运行期；
#   3. 使用 `<float*>` 强制类型转换获取数组底层指针，直接传给 CUDA 包装函数；
#   4. cdef 声明的变量是 C 级静态类型变量，避免 Python 对象开销，提升性能。
# =============================================================================

# --- 编译期导入 NumPy 的 C API（用于 ndarray 的类型标注与指针获取） ---
import numpy as np           # 运行期：普通 Python NumPy
cimport numpy as cnp         # 编译期：NumPy C API（需要 np.get_include()）

# 初始化 NumPy C-API（使用 <float*>PyArray_DATA 前必须调用，防止段错误）
cnp.import_array()


# =============================================================================
# 声明来自 cuda_wrapper.h 的 C 函数接口
#   - 这些函数在 src/main.cu 中以 extern "C" 实现
#   - 签名必须与头文件完全一致
# =============================================================================
cdef extern from "cuda_wrapper.h":
    int vector_add_wrapper(const float* a, const float* b, float* out, int n)
    int matmul_wrapper(const float* a, const float* b, float* c, int width)


# =============================================================================
# Python 可调用封装：向量加法
# =============================================================================
def vector_add(a, b):
    """向量加法（GPU 加速）。

    参数:
        a, b: 一维 numpy.float32 数组，长度相同。
    返回:
        out: 一维 numpy.float32 数组，等于 a + b。
    异常:
        ValueError:   输入形状不匹配。
        RuntimeError: CUDA 执行失败。
    """
    # 保证输入是连续的 float32 数组（Cython 直接取底层指针，要求内存连续）
    cdef cnp.ndarray[cnp.float32_t, ndim=1] a_arr = \
        np.ascontiguousarray(a, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] b_arr = \
        np.ascontiguousarray(b, dtype=np.float32)

    # 形状检查
    if a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError("向量长度不一致: %d vs %d"
                         % (a_arr.shape[0], b_arr.shape[0]))

    cdef int n = a_arr.shape[0]

    # 分配输出数组
    cdef cnp.ndarray[cnp.float32_t, ndim=1] out = \
        np.empty(n, dtype=np.float32)

    # 调用 CUDA 包装函数，<float*> 取得底层数据指针
    cdef int ret = vector_add_wrapper(
        <float*> cnp.PyArray_DATA(a_arr),
        <float*> cnp.PyArray_DATA(b_arr),
        <float*> cnp.PyArray_DATA(out),
        n
    )
    if ret != 0:
        raise RuntimeError("CUDA vector_add 执行失败 (ret=%d)" % ret)
    return out


# =============================================================================
# Python 可调用封装：方阵乘法
# =============================================================================
def matmul(a, b):
    """方阵乘法 C = A * B（GPU 加速）。

    参数:
        a, b: 二维 numpy.float32 方阵，形状为 (width, width)。
    返回:
        c: 二维 numpy.float32 数组，形状 (width, width)。
    异常:
        ValueError:   输入不是同尺寸方阵。
        RuntimeError: CUDA 执行失败。
    """
    # 转为连续 float32 二维数组
    cdef cnp.ndarray[cnp.float32_t, ndim=2] a_arr = \
        np.ascontiguousarray(a, dtype=np.float32)
    cdef cnp.ndarray[cnp.float32_t, ndim=2] b_arr = \
        np.ascontiguousarray(b, dtype=np.float32)

    # 必须是同尺寸方阵
    if a_arr.shape[0] != a_arr.shape[1]:
        raise ValueError("矩阵 A 必须是方阵")
    if b_arr.shape[0] != b_arr.shape[1]:
        raise ValueError("矩阵 B 必须是方阵")
    if a_arr.shape[0] != b_arr.shape[0]:
        raise ValueError("矩阵 A、B 尺寸不匹配")

    cdef int width = a_arr.shape[0]

    cdef cnp.ndarray[cnp.float32_t, ndim=2] c = \
        np.empty((width, width), dtype=np.float32)

    cdef int ret = matmul_wrapper(
        <float*> cnp.PyArray_DATA(a_arr),
        <float*> cnp.PyArray_DATA(b_arr),
        <float*> cnp.PyArray_DATA(c),
        width
    )
    if ret != 0:
        raise RuntimeError("CUDA matmul 执行失败 (ret=%d)" % ret)
    return c
