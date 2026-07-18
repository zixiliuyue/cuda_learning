#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 37: ctypes 调用 CUDA 动态库 — Python 演示
=============================================================

本脚本演示如何用 Python【标准库 ctypes】直接调用编译好的 CUDA 动态库，
无需 PyCUDA / Numba / PyTorch 等任何第三方 GPU 库，真正做到「零外部依赖」
（NumPy 仅用于数据准备与结果验证，可选）。

核心步骤：
    1. 定位并加载动态库（Windows 的 .dll / Linux 的 .so）
    2. 声明每个导出函数的参数类型 (argtypes) 和返回类型 (restype)
    3. 准备 numpy 数组（连续、float32），直接把指针传给 C 函数
    4. 调用函数、检查返回码、验证结果

注意事项：
    - 传入的 numpy 数组必须是 C 连续 (C_CONTIGUOUS) 且 dtype=float32，
      否则指针语义会出错。这里用 np.ascontiguousarray 强制保证。
    - 使用 np.ctypeslib.ndpointer 声明参数类型，它会在调用时自动校验
      数组的 dtype / 维度 / 连续性，是比裸指针更安全的做法。
"""

import os
import sys
import time
import ctypes

try:
    import numpy as np
except ImportError:
    print("[错误] 未安装 NumPy。请运行: pip install numpy")
    sys.exit(1)


# ============================================================
#  第一步：定位并加载动态库
# ============================================================
def load_library():
    """
    根据操作系统定位对应的动态库文件并加载。

    CMake 的产物命名规则：
        - Windows : cuda_ctypes.dll
        - Linux   : libcuda_ctypes.so  (CMake 会自动加 lib 前缀)
        - macOS   : libcuda_ctypes.dylib
    这里对可能的输出目录（build、build/Release）都做了探测，
    以兼容不同平台/生成器的输出布局。
    """
    here = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(here, "..", "build")

    if os.name == "nt":
        lib_names = ["cuda_ctypes.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libcuda_ctypes.dylib", "cuda_ctypes.dylib"]
    else:
        lib_names = ["libcuda_ctypes.so", "cuda_ctypes.so"]

    # 候选目录：Windows 的 MSVC 生成器会把 dll 放进 Release/Debug 子目录
    candidate_dirs = [
        build_dir,
        os.path.join(build_dir, "Release"),
        os.path.join(build_dir, "Debug"),
    ]

    for d in candidate_dirs:
        for name in lib_names:
            path = os.path.normpath(os.path.join(d, name))
            if os.path.exists(path):
                print(f"[加载] 找到动态库: {path}")
                return ctypes.CDLL(path)

    print("[错误] 未找到动态库，请先编译：")
    print("       Windows -> 运行 build.bat")
    print("       Linux   -> 运行 ./build.sh")
    sys.exit(1)


# ============================================================
#  第二步：声明导出函数的参数与返回类型
# ============================================================
def setup_prototypes(lib):
    """
    为每个 C 函数设置 argtypes / restype。

    这一步至关重要：ctypes 默认把所有参数当作 int 处理，
    如果不显式声明，在 64 位系统上传指针会被截断为 32 位而崩溃。
    """
    # float32 一维指针（用于向量加法）
    f32_ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")

    # int vector_add_cuda(const float* a, const float* b, float* out, int n)
    lib.vector_add_cuda.argtypes = [f32_ptr, f32_ptr, f32_ptr, ctypes.c_int]
    lib.vector_add_cuda.restype = ctypes.c_int

    # int matmul_cuda(const float* a, const float* b, float* c, int width)
    lib.matmul_cuda.argtypes = [f32_ptr, f32_ptr, f32_ptr, ctypes.c_int]
    lib.matmul_cuda.restype = ctypes.c_int

    # int get_device_info(char* buffer, int buffer_size)
    lib.get_device_info.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.get_device_info.restype = ctypes.c_int

    # void free_cuda_mem(void* ptr)
    lib.free_cuda_mem.argtypes = [ctypes.c_void_p]
    lib.free_cuda_mem.restype = None


# ============================================================
#  第三步：各功能演示
# ============================================================
def demo_device_info(lib):
    """演示 get_device_info：由 Python 分配缓冲区，C 端填充。"""
    print("\n[演示 1] 查询 GPU 设备信息 (get_device_info)")
    print("-" * 60)

    # 分配 512 字节的可写字符缓冲区，交给 C 端填充
    buf = ctypes.create_string_buffer(512)
    ret = lib.get_device_info(buf, 512)

    if ret == 0:
        # C 端写入的是 UTF-8 字节串，解码为 Python str
        print(f"  {buf.value.decode('utf-8', errors='replace')}")
    else:
        print("  查询失败（可能没有可用的 GPU）")
    return ret


def demo_vector_add(lib):
    """演示 vector_add_cuda：百万级向量加法并验证正确性。"""
    print("\n[演示 2] GPU 向量加法 (vector_add_cuda)")
    print("-" * 60)

    n = 1024 * 1024  # 1,048,576 个元素

    # 准备输入数据；ascontiguousarray 确保内存连续、dtype 正确
    a = np.ascontiguousarray(np.random.uniform(-1, 1, n), dtype=np.float32)
    b = np.ascontiguousarray(np.random.uniform(-1, 1, n), dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)

    print(f"  向量长度: {n} 个元素 ({n * 4 / (1024*1024):.2f} MB/数组)")

    # 调用 GPU（含 H2D + Kernel + D2H 全流程）
    t0 = time.time()
    ret = lib.vector_add_cuda(a, b, out, n)
    gpu_ms = (time.time() - t0) * 1000.0

    # CPU 参考结果
    t0 = time.time()
    ref = a + b
    cpu_ms = (time.time() - t0) * 1000.0

    max_err = float(np.max(np.abs(out - ref)))
    ok = (ret == 0) and (max_err < 1e-4)

    print(f"  调用返回码: {ret} (0=成功)")
    print(f"  GPU 端到端: {gpu_ms:.4f} ms  |  CPU: {cpu_ms:.4f} ms")
    print(f"  最大误差  : {max_err:.2e}  ->  {'通过' if ok else '失败'}")
    return ok, gpu_ms, cpu_ms


def demo_matmul(lib):
    """演示 matmul_cuda：方阵乘法并与 numpy 对比。"""
    print("\n[演示 3] GPU 矩阵乘法 (matmul_cuda)")
    print("-" * 60)

    width = 512  # 512 x 512 方阵

    a = np.ascontiguousarray(np.random.uniform(-1, 1, (width, width)), dtype=np.float32)
    b = np.ascontiguousarray(np.random.uniform(-1, 1, (width, width)), dtype=np.float32)
    c = np.zeros((width, width), dtype=np.float32)

    print(f"  矩阵规模: {width} x {width}")

    # GPU 计算（传入的是 2D 数组，ndpointer 会自动取其数据指针）
    t0 = time.time()
    ret = lib.matmul_cuda(a, b, c, width)
    gpu_ms = (time.time() - t0) * 1000.0

    # CPU (numpy) 参考
    t0 = time.time()
    ref = a @ b
    cpu_ms = (time.time() - t0) * 1000.0

    # 矩阵乘法累加误差较大，用相对容差
    max_err = float(np.max(np.abs(c - ref)))
    ok = (ret == 0) and (max_err < 1e-2)

    print(f"  调用返回码: {ret} (0=成功)")
    print(f"  GPU 端到端: {gpu_ms:.4f} ms  |  CPU(numpy): {cpu_ms:.4f} ms")
    print(f"  最大误差  : {max_err:.2e}  ->  {'通过' if ok else '失败'}")
    return ok, gpu_ms, cpu_ms


# ============================================================
#  主流程
# ============================================================
def main():
    print()
    print("=" * 60)
    print("     Day 37: ctypes 调用 CUDA 动态库 — Python 演示        ")
    print("=" * 60)

    lib = load_library()
    setup_prototypes(lib)

    demo_device_info(lib)
    va_ok, va_gpu, va_cpu = demo_vector_add(lib)
    mm_ok, mm_gpu, mm_cpu = demo_matmul(lib)

    # ---------- 性能总结表格 ----------
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  {'任务':<12}{'GPU(ms)':>12}{'CPU(ms)':>12}{'结果':>10}")
    print("  " + "-" * 46)
    print(f"  {'向量加法':<12}{va_gpu:>12.4f}{va_cpu:>12.4f}{'通过' if va_ok else '失败':>10}")
    print(f"  {'矩阵乘法':<12}{mm_gpu:>12.4f}{mm_cpu:>12.4f}{'通过' if mm_ok else '失败':>10}")
    print("=" * 60)
    print("  说明: GPU 时间含 H2D/D2H 传输，小数据下可能不占优势；")
    print("        本演示的重点是「ctypes 零依赖调用 CUDA」这一机制。")
    print("=" * 60)
    print()

    return 0 if (va_ok and mm_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
