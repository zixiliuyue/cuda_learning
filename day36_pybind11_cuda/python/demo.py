#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 36: pybind11 CUDA 封装调用 — Python 演示脚本
=============================================================

本脚本演示如何调用由 pybind11 编译出来的 CUDA 扩展模块
（Windows 上是 cuda_pybind.pyd，Linux 上是 cuda_pybind.so）。

编译步骤（二选一）：
  方式 A：使用本项目的构建脚本
      Windows:  .\\build.bat
      Linux:    ./build.sh
    编译完成后，扩展模块会输出到 build/ 目录。

  方式 B：使用 pip 安装（若提供了 setup.py / pyproject）
      cd build && pip install .

运行本脚本：
  python demo.py

如果模块尚未编译，脚本会自动退回到纯 NumPy 实现，
以便你在没有编译环境时也能理解接口的语义。
"""

import os
import sys
import time

# ------------------------------------------------------------
# 尝试加载编译好的 CUDA 扩展模块
#   我们把 build/ 目录加入搜索路径，这样就能 import 到 .pyd / .so
# ------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'build'))

try:
    import cuda_pybind as _cuda
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False
    print("[提示] 模块未编译，请先运行 build.bat (Windows) 或 build.sh (Linux) 编译")
    print("[提示] 编译后可在 python/ 目录中 import cuda_pybind 即可使用")
    print("[提示] 当前将使用 NumPy fallback 演示接口语义")

# NumPy 是必需的（无论是否有 GPU 模块都会用到）
try:
    import numpy as np
except ImportError:
    print("[错误] 未安装 NumPy。请运行: pip install numpy")
    sys.exit(1)


# =============================================================================
# NumPy 参考实现（作为 fallback，同时用于验证 GPU 结果）
# =============================================================================

def vector_add_numpy(a, b):
    """向量加法的 NumPy 参考实现"""
    return a + b


def matmul_numpy(a, b, width):
    """矩阵乘法的 NumPy 参考实现（返回展平的一维数组）"""
    return (a.reshape(width, width) @ b.reshape(width, width)).reshape(-1)


# =============================================================================
# 演示：向量加法
# =============================================================================

def demo_vector_add():
    print()
    print("-" * 60)
    print("  演示 1: 向量加法 vector_add(a, b)")
    print("-" * 60)

    n = 1024 * 1024
    # 注意 dtype 必须与 C++ 端的 float（float32）一致
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    # NumPy 参考结果
    ref = vector_add_numpy(a, b)

    if HAS_MODULE:
        t0 = time.perf_counter()
        gpu = _cuda.vector_add(a, b)   # <-- 调用 GPU 扩展
        t1 = time.perf_counter()
        max_err = float(np.max(np.abs(gpu - ref)))
        print(f"  元素个数    : {n}")
        print(f"  GPU 耗时    : {(t1 - t0) * 1000:.3f} ms（含数据传输）")
        print(f"  最大误差    : {max_err:.2e}")
        print(f"  结果验证    : {'通过' if max_err < 1e-4 else '失败'}")
    else:
        print(f"  元素个数    : {n}")
        print(f"  [fallback] 前 5 个结果: {vector_add_numpy(a, b)[:5]}")


# =============================================================================
# 演示：矩阵乘法
# =============================================================================

def demo_matmul():
    print()
    print("-" * 60)
    print("  演示 2: 矩阵乘法 matmul(a, b, width)")
    print("-" * 60)

    width = 512
    a = np.random.rand(width * width).astype(np.float32)
    b = np.random.rand(width * width).astype(np.float32)

    ref = matmul_numpy(a, b, width)

    if HAS_MODULE:
        t0 = time.perf_counter()
        gpu = _cuda.matmul(a, b, width)   # <-- 调用 GPU 扩展
        t1 = time.perf_counter()
        max_err = float(np.max(np.abs(gpu - ref)))
        print(f"  矩阵尺寸    : {width} x {width}")
        print(f"  GPU 耗时    : {(t1 - t0) * 1000:.3f} ms（含数据传输）")
        print(f"  最大误差    : {max_err:.2e}")
        # 矩阵乘法累加误差略大，容差放宽到 1e-2
        print(f"  结果验证    : {'通过' if max_err < 1e-2 else '失败'}")
    else:
        print(f"  矩阵尺寸    : {width} x {width}")
        print(f"  [fallback] C[0][0] = {ref[0]:.4f}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print()
    print("=" * 60)
    print("     Day 36: pybind11 CUDA 封装调用 — Python 演示           ")
    print("=" * 60)

    if HAS_MODULE:
        print(f"  模块版本    : {getattr(_cuda, '__version__', '未知')}")
        try:
            print(f"  CUDA 设备   : {_cuda.device_name()}")
        except Exception as exc:
            print(f"  CUDA 设备   : 查询失败 ({exc})")
    else:
        print("  运行模式    : NumPy fallback（未加载 GPU 模块）")

    demo_vector_add()
    demo_matmul()

    # 知识点总结
    print()
    print("=" * 60)
    print("                        知识点总结                            ")
    print("=" * 60)
    print("  1. PYBIND11_MODULE    : 定义可被 Python import 的模块")
    print("  2. py::array_t<float> : 与 NumPy 数组零拷贝互操作")
    print("  3. buffer_info.ptr    : 获取 NumPy 底层数据指针")
    print("  4. import cuda_pybind : 像普通库一样调用 GPU 函数")
    print("  5. C++ 抛异常         : 自动转为 Python RuntimeError")
    print("  6. FetchContent       : CMake 自动拉取 pybind11 依赖")
    print("=" * 60)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
