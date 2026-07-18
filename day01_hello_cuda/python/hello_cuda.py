#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 1: Hello CUDA — Python 版本 (Numba)
=============================================================

这是 Day 1 的 Python 等价实现，使用 Numba 的 CUDA 支持。
适合以下场景：
  - macOS 用户（没有 NVIDIA GPU）
  - 想快速验证概念而不想编译 C++
  - 通过 Google Colab 运行（免费 T4 GPU）

Numba 的 @cuda.jit 装饰器与 CUDA C++ 的对应关系：
  - @cuda.jit                 ≈ __global__
  - cuda.grid(1)              ≈ blockIdx.x * blockDim.x + threadIdx.x
  - cuda.blockIdx.x           ≈ blockIdx.x
  - cuda.threadIdx.x          ≈ threadIdx.x
  - cuda.blockDim.x           ≈ blockDim.x
  - cuda.gridDim.x            ≈ gridDim.x

安装依赖:
  pip install numba

如果在无 GPU 环境运行，Numba 会自动使用 CPU 后端并给出提示。
"""

import os
import sys

# 尝试导入 Numba CUDA
try:
    from numba import cuda
    import numpy as np
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)


def check_gpu():
    """检查系统中是否有可用的 NVIDIA GPU"""
    if not cuda.is_available():
        print("[警告] 未检测到 NVIDIA GPU。")
        print("[提示] 如果您在 macOS 上，这是正常的。")
        print("[提示] 可以通过 Google Colab 运行此代码获得免费 GPU。")
        return False

    # 获取设备信息
    device = cuda.get_current_device()
    print(f"[设备信息] 检测到 GPU: {device.name.decode() if isinstance(device.name, bytes) else device.name}")
    print(f"[设备信息] 计算能力: {device.compute_capability}")
    return True


# =============================================================================
# Kernel 函数（在 GPU 上执行）
# =============================================================================

@cuda.jit
def hello_cuda_kernel():
    """
    Hello CUDA Kernel — Numba 版本

    与 C++ 版本的 __global__ void hello_cuda_kernel() 对应。
    每个被调用的线程都会执行一次此函数体。
    """
    # 获取当前线程在 Block 内的索引
    tx = cuda.threadIdx.x
    # 获取当前 Block 在 Grid 内的索引
    bx = cuda.blockIdx.x
    # 获取 Block 维度（每个 Block 的线程数）
    bd = cuda.blockDim.x

    # 计算全局线程 ID
    global_tid = bx * bd + tx

    # 每个线程打印自己的身份信息
    # 注意：Numba 的 print 在 Kernel 中有一定限制，
    # 但简单的字符串格式化是支持的
    print(f"  [GPU线程] 全局ID={global_tid:02d} | Block={bx:02d} | Thread={tx:02d} | 你好，CUDA!")


# =============================================================================
# 主机代码（在 CPU 上执行）
# =============================================================================

def main():
    """主函数 — 程序入口"""
    print()
    print("=" * 60)
    print("     Day 1: Hello CUDA — Python/Numba 版本                  ")
    print("=" * 60)
    print()

    # 检查 GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print()
        print("[提示] 继续以演示模式运行（将在 CPU 上模拟执行）...")
        print()

    # 配置 Grid 和 Block 尺寸
    num_blocks = 4
    threads_per_block = 8

    print()
    print("[主机] 准备启动 Kernel ...")
    print(f"[主机] Grid 维度 : {num_blocks} 个 Block")
    print(f"[主机] Block 维度: {threads_per_block} 个 Thread")
    print(f"[主机] 总线程数  : {num_blocks * threads_per_block} 个")
    print()
    print(f"[主机] 启动 Kernel [{num_blocks}, {threads_per_block}] ...")
    print("-" * 60)

    # 启动 Kernel！
    # Numba 语法: kernel_function[grid_dim, block_dim](args...)
    # 这与 C++ 的 <<<grid_dim, block_dim>>> 完全对应
    hello_cuda_kernel[num_blocks, threads_per_block]()

    # 同步设备，等待所有 Kernel 执行完成
    cuda.synchronize()

    print("-" * 60)
    print("[主机] Kernel 执行完成！")

    # 知识点总结
    print()
    print("=" * 60)
    print("                        知识点总结                            ")
    print("=" * 60)
    print("  1. @cuda.jit        : Numba Kernel 装饰符，对应 __global__")
    print("  2. [G, B]           : Kernel 启动配置，对应 <<<G, B>>>")
    print("  3. cuda.threadIdx.x : 线程在 Block 内的索引")
    print("  4. cuda.blockIdx.x  : Block 在 Grid 内的索引")
    print("  5. cuda.blockDim.x  : Block 包含的线程总数")
    print("  6. cuda.grid(1)     : 计算全局线程 ID")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
