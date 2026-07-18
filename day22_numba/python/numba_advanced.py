#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 22: Numba 高级特性 — Python 版本

汇总演示 Numba CUDA 的高级用法：
- CUDA 事件计时
- 设备函数
- 数学内建函数
- Warp 级别操作
- 自动类型推断
"""

import sys
import time
import numpy as np

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)

# 常量定义
ARRAY_SIZE = 1024 * 1024
BLOCK_SIZE = 256
TOLERANCE = 1e-5
ITERATIONS = 10


# =============================================================================
# 1. 设备函数 (device function)
# =============================================================================
@cuda.jit(device=True)
def device_sigmoid(x):
    """设备函数: 在 GPU 上被 Kernel 调用的辅助函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

# 需要 math 模块
import math

@cuda.jit(device=True)
def device_relu(x):
    """ReLU 激活函数（设备函数）"""
    return max(0.0, x)

@cuda.jit(device=True)
def device_softplus(x):
    """Softplus 激活函数: log(1 + exp(x))"""
    if x > 20:
        return x
    elif x < -20:
        return 0.0
    else:
        return math.log(1.0 + math.exp(x))


# =============================================================================
# 2. 使用设备函数的 Kernel
# =============================================================================
@cuda.jit
def activation_kernel(data, result, n, mode):
    """
    激活函数 Kernel: 根据 mode 选择不同的激活函数。
    mode=0: sigmoid, mode=1: relu, mode=2: softplus
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < n:
        x = data[tid]
        if mode == 0:
            result[tid] = device_sigmoid(x)
        elif mode == 1:
            result[tid] = device_relu(x)
        else:
            result[tid] = device_softplus(x)


# =============================================================================
# 3. 数学运算 Kernel
# =============================================================================
@cuda.jit
def math_ops_kernel(data, result, n):
    """
    数学运算 Kernel: 演示 Numba 支持的 CUDA 数学内建函数。
    包括: sin, cos, sqrt, log, exp, pow, fabs, fmin, fmax 等。
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < n:
        x = data[tid]
        # 链式数学运算
        result[tid] = math.sqrt(math.fabs(x)) + math.sin(x) * math.cos(x)


# =============================================================================
# 4. CUDA 事件计时演示
# =============================================================================
def event_timing_demo(d_data, d_result, n, grid_size):
    """演示使用 CUDA 事件进行精确计时"""
    start_event = cuda.event()
    end_event = cuda.event()

    start_event.record()
    activation_kernel[grid_size, BLOCK_SIZE](d_data, d_result, n, 1)
    end_event.record()
    end_event.synchronize()

    # 事件耗时（毫秒）
    elapsed = start_event.elapsed_time(end_event)
    return elapsed


def sigmoid_cpu(x):
    """CPU sigmoid"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def relu_cpu(x):
    """CPU relu"""
    return np.maximum(0, x)


def main():
    print()
    print("=" * 60)
    print("     Day 22: Numba 高级特性 — Python 版本                   ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 数据准备
    print(f"[主机] 数组大小: {ARRAY_SIZE:,} 元素")
    h_data = np.random.uniform(-5.0, 5.0, size=ARRAY_SIZE).astype(np.float32)
    d_data = cuda.to_device(h_data)
    d_result = cuda.device_array_like(h_data)

    grid_size = (ARRAY_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ---- 激活函数测试 ----
    print()
    print("[主机] 测试设备函数 (激活函数)...")

    # ReLU
    activation_kernel[grid_size, BLOCK_SIZE](d_data, d_result, ARRAY_SIZE, 1)
    cuda.synchronize()
    h_relu_gpu = d_result.copy_to_host()
    h_relu_cpu = relu_cpu(h_data)
    relu_diff = np.max(np.abs(h_relu_gpu - h_relu_cpu))

    # CUDA 事件计时
    print("[主机] CUDA 事件计时演示...")
    event_time = event_timing_demo(d_data, d_result, ARRAY_SIZE, grid_size)

    # 数学运算
    print("[主机] 测试 CUDA 数学内建函数...")
    start = time.time()
    for _ in range(ITERATIONS):
        math_ops_kernel[grid_size, BLOCK_SIZE](d_data, d_result, ARRAY_SIZE)
        cuda.synchronize()
    gpu_math_time = (time.time() - start) / ITERATIONS * 1000

    # CPU 数学运算
    start = time.time()
    for _ in range(ITERATIONS):
        _ = np.sqrt(np.abs(h_data)) + np.sin(h_data) * np.cos(h_data)
    cpu_math_time = (time.time() - start) / ITERATIONS * 1000

    # 验证
    h_math_gpu = d_result.copy_to_host()
    h_math_cpu = np.sqrt(np.abs(h_data)) + np.sin(h_data) * np.cos(h_data)
    math_diff = np.max(np.abs(h_math_gpu - h_math_cpu))

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  CUDA 事件计时       : {event_time:.4f} ms")
    print(f"  GPU 数学运算        : {gpu_math_time:.4f} ms")
    print(f"  CPU 数学运算        : {cpu_math_time:.4f} ms")
    if gpu_math_time > 0:
        print(f"  GPU vs CPU 数学     : {cpu_math_time / gpu_math_time:.2f}x")
    print(f"  ReLU 验证           : {'通过' if relu_diff < TOLERANCE else '失败'} (误差: {relu_diff:.2e})")
    print(f"  数学运算验证        : {'通过' if math_diff < TOLERANCE else '失败'} (误差: {math_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 设备函数: @cuda.jit(device=True), 在 GPU 上调用")
    print("  2. CUDA 事件: cuda.event() + elapsed_time(), 精确 GPU 计时")
    print("  3. 数学内建: math.sin/cos/exp/log/sqrt/pow/fabs")
    print("  4. Warp 操作: cuda.shfl_sync() (Warp Shuffle)")
    print("  5. 原子操作: cuda.atomic.add/min/max/cas")
    print("  6. 类型推断: Numba 自动推断，或用 dtype=np.float32 显式指定")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
