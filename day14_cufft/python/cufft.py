#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 14: cuFFT 快速傅里叶变换 — Python 版本

演示 FFT 的 GPU 加速实现。
优先使用 CuPy（封装 cuFFT），不可用时 fallback 到 NumPy FFT。
"""

import sys
import time
import numpy as np

# 尝试导入 CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("[提示] 未安装 CuPy，将使用 NumPy FFT fallback。")
    print("[提示] 安装 CuPy: pip install cupy-cuda12x")

# 常量定义
FFT_SIZE = 1024 * 1024  # 1M 点 FFT
TOLERANCE = 1e-2
ITERATIONS = 5


def fft_numpy(data):
    """NumPy FFT（CPU）"""
    return np.fft.fft(data)


def ifft_numpy(data):
    """NumPy IFFT（CPU）"""
    return np.fft.ifft(data)


def fft_cupy_timing(data_np, iterations):
    """CuPy FFT — 带计时"""
    d_data = cp.asarray(data_np)

    # 预热
    for _ in range(3):
        _ = cp.fft.fft(d_data)
    cp.cuda.Stream.null.synchronize()

    # 正向 FFT 计时
    start = time.time()
    for _ in range(iterations):
        result = cp.fft.fft(d_data)
    cp.cuda.Stream.null.synchronize()
    fft_time = (time.time() - start) / iterations * 1000

    # 逆 FFT 计时
    start = time.time()
    for _ in range(iterations):
        recovered = cp.fft.ifft(result)
    cp.cuda.Stream.null.synchronize()
    ifft_time = (time.time() - start) / iterations * 1000

    return cp.asnumpy(result), cp.asnumpy(recovered), fft_time, ifft_time


def numpy_fft_timing(data, iterations):
    """NumPy FFT — 带计时"""
    # 预热
    for _ in range(3):
        _ = np.fft.fft(data)

    # 正向 FFT
    start = time.time()
    for _ in range(iterations):
        result = np.fft.fft(data)
    fft_time = (time.time() - start) / iterations * 1000

    # 逆 FFT
    start = time.time()
    for _ in range(iterations):
        recovered = np.fft.ifft(result)
    ifft_time = (time.time() - start) / iterations * 1000

    return result, recovered, fft_time, ifft_time


def main():
    print()
    print("=" * 60)
    print("     Day 14: cuFFT 快速傅里叶变换 — Python 版本             ")
    print("=" * 60)
    print()

    # 数据准备
    print(f"[主机] FFT 大小: {FFT_SIZE:,} 点")
    print(f"[主机] 数据类型: complex64")

    # 生成测试信号: 50Hz + 120Hz 正弦波
    t = np.linspace(0, 1, FFT_SIZE, dtype=np.float32)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    signal = signal.astype(np.complex64)

    # NumPy FFT
    print("[主机] 运行 NumPy FFT...")
    np_result, np_recovered, np_fft_time, np_ifft_time = numpy_fft_timing(signal, ITERATIONS)

    # CuPy FFT
    if CUPY_AVAILABLE:
        print("[设备] 运行 CuPy FFT (cuFFT 后端)...")
        cp_result, cp_recovered, cp_fft_time, cp_ifft_time = fft_cupy_timing(signal, ITERATIONS)

    # 验证
    np_diff = np.max(np.abs(np_recovered - signal))
    np_ok = np_diff < TOLERANCE

    if CUPY_AVAILABLE:
        cp_diff = np.max(np.abs(cp_recovered - signal))
        cp_ok = cp_diff < TOLERANCE
        # FFT 结果的幅度谱对比
        cp_mag = np.abs(cp_result) / FFT_SIZE
        np_mag = np.abs(np_result) / FFT_SIZE
        mag_diff = np.max(np.abs(cp_mag - np_mag))
    else:
        cp_ok = True
        cp_diff = 0.0
        mag_diff = 0.0
        cp_fft_time = np_fft_time
        cp_ifft_time = np_ifft_time

    # 频率分析（展示 FFT 实用性）
    freqs = np.fft.fftfreq(FFT_SIZE, d=1.0/FFT_SIZE)
    magnitudes = np.abs(np_result) / FFT_SIZE
    # 找到主要的频率分量
    top_indices = np.argsort(magnitudes[:FFT_SIZE//2])[-5:][::-1]
    print()
    print("[分析] 主要频率分量:")
    for idx in top_indices:
        print(f"  频率 {freqs[idx]:>8.1f} Hz, 幅度 {magnitudes[idx]:>10.4f}")

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    if CUPY_AVAILABLE:
        print(f"  CuPy FFT (cuFFT GPU): {cp_fft_time:.4f} ms")
        print(f"  CuPy IFFT            : {cp_ifft_time:.4f} ms")
        print(f"  NumPy FFT (CPU)      : {np_fft_time:.4f} ms")
        print(f"  NumPy IFFT           : {np_ifft_time:.4f} ms")
        if cp_fft_time > 0:
            print(f"  FFT GPU 加速比      : {np_fft_time / cp_fft_time:.2f} x")
        print(f"  CuPy IFFT 验证       : {'通过' if cp_ok else '失败'} (误差: {cp_diff:.2e})")
    else:
        print(f"  NumPy FFT (CPU)      : {np_fft_time:.4f} ms")
        print(f"  NumPy IFFT           : {np_ifft_time:.4f} ms")
        print(f"  CuPy                 : 未安装（使用 NumPy fallback）")
    print(f"  NumPy IFFT 验证      : {'通过' if np_ok else '失败'} (误差: {np_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. cuFFT: NVIDIA 优化的 FFT 库，支持多种精度和维度")
    print("  2. CuPy 的 cp.fft.fft() 直接调用 cuFFT")
    print("  3. FFT 应用: 信号处理、图像滤波、偏微分方程求解")
    print("  4. 大规模 FFT (>=1M 点) GPU 优势更显著")
    print("  5. cuFFT 支持批量 FFT、实数到复数变换等")
    print("  6. 批处理多个小 FFT 时，CuPy 的优势更加明显")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
