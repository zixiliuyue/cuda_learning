# -*- coding: utf-8 -*-
"""Day 38: Cython + CUDA 扩展调用演示

演示流程:
    1. 尝试导入编译好的 cuda_cython 扩展;
    2. 若未编译, 给出编译提示并退回纯 NumPy 演示;
    3. 分别演示 GPU 向量加法、矩阵乘法, 并与 NumPy 对比正确性与耗时;
    4. 打印 60 字符分隔线 + 性能总结表格。

编译方法(在 python/ 目录):
    python setup.py build_ext --inplace
"""

import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

import numpy as np

# 尝试导入 Cython 编译出的 CUDA 扩展
try:
    import cuda_cython
    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False
    print("[提示] 模块未编译，请先在 python/ 目录运行:")
    print("  python setup.py build_ext --inplace")
    print("[提示] 本次将仅用 NumPy(CPU) 进行演示，用于展示预期结果。\n")

SEP = "=" * 60          # 60 字符分隔线


def bench(func, *args, repeat=5):
    """简单计时: 返回 (结果, 平均毫秒)。"""
    # 预热一次(触发首次 CUDA 上下文初始化)
    out = func(*args)
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = func(*args)
    dt = (time.perf_counter() - t0) / repeat * 1000.0
    return out, dt


def demo_vector_add():
    """向量加法演示。"""
    print(SEP)
    print("  演示 1: 向量加法 (Vector Add)")
    print(SEP)

    n = 1 << 20   # 1M 元素
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    # NumPy(CPU) 基准
    cpu_out, cpu_ms = bench(lambda x, y: x + y, a, b)
    print("  [CPU/NumPy]  规模=%d  平均耗时=%.3f ms" % (n, cpu_ms))

    if HAS_MODULE:
        gpu_out, gpu_ms = bench(cuda_cython.vector_add, a, b)
        ok = np.allclose(gpu_out, cpu_out, atol=1e-4)
        print("  [GPU/Cython] 规模=%d  平均耗时=%.3f ms" % (n, gpu_ms))
        print("  [验证] 与 NumPy 结果一致: %s" % ("是" if ok else "否"))
        return ("向量加法", n, cpu_ms, gpu_ms, ok)
    else:
        print("  [GPU/Cython] 未编译，跳过")
        return ("向量加法", n, cpu_ms, None, None)


def demo_matmul():
    """矩阵乘法演示。"""
    print(SEP)
    print("  演示 2: 矩阵乘法 (MatMul)")
    print(SEP)

    width = 512
    a = np.random.rand(width, width).astype(np.float32)
    b = np.random.rand(width, width).astype(np.float32)

    cpu_out, cpu_ms = bench(lambda x, y: x @ y, a, b, repeat=3)
    print("  [CPU/NumPy]  规模=%dx%d  平均耗时=%.3f ms" % (width, width, cpu_ms))

    if HAS_MODULE:
        gpu_out, gpu_ms = bench(cuda_cython.matmul, a, b, repeat=3)
        ok = np.allclose(gpu_out, cpu_out, atol=1e-2)
        print("  [GPU/Cython] 规模=%dx%d  平均耗时=%.3f ms" % (width, width, gpu_ms))
        print("  [验证] 与 NumPy 结果一致: %s" % ("是" if ok else "否"))
        return ("矩阵乘法", "%dx%d" % (width, width), cpu_ms, gpu_ms, ok)
    else:
        print("  [GPU/Cython] 未编译，跳过")
        return ("矩阵乘法", "%dx%d" % (width, width), cpu_ms, None, None)


def print_summary(rows):
    """打印性能总结表格。"""
    print(SEP)
    print("  性能总结")
    print(SEP)
    print("  %-10s | %-10s | %-12s | %-12s | %-6s"
          % ("测试项", "规模", "CPU(ms)", "GPU(ms)", "验证"))
    print("  " + "-" * 56)
    for name, size, cpu_ms, gpu_ms, ok in rows:
        gpu_str = "%.3f" % gpu_ms if gpu_ms is not None else "N/A"
        ok_str = ("通过" if ok else "失败") if ok is not None else "-"
        speed = "%.2fx" % (cpu_ms / gpu_ms) if gpu_ms else "-"
        print("  %-10s | %-10s | %-12.3f | %-12s | %-6s"
              % (name, str(size), cpu_ms, gpu_str, ok_str))
        if gpu_ms:
            print("             加速比(CPU/GPU): %s" % speed)
    print(SEP)


def main():
    print("\n" + SEP)
    print("     Day 38: Cython + CUDA 混合编程演示")
    print(SEP)
    print("  模块状态: %s\n" % ("已加载 cuda_cython" if HAS_MODULE else "未编译(仅 CPU 演示)"))

    rows = []
    rows.append(demo_vector_add())
    print()
    rows.append(demo_matmul())
    print()
    print_summary(rows)
    print("\n[完成] 演示结束。\n")


if __name__ == "__main__":
    main()
