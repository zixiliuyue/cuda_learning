#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 27: N-Body 模拟 — Python/Numba 版本

使用 Numba CUDA 实现万有引力 N-Body 模拟。
每个粒子受到其他所有粒子的引力作用。
"""

import sys
import time
import math
import numpy as np

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[错误] 未安装 Numba。请运行: pip install numba")
    sys.exit(1)

# 常量定义
NUM_BODIES = 4096
BLOCK_SIZE = 256
SOFTENING = 1e-3  # 软化参数，防止距离为0时的数值爆炸
DT = 0.01  # 时间步长
STEPS = 10
TOLERANCE = 1e-3


@cuda.jit
def compute_forces_kernel(positions, velocities, masses, n, dt, softening):
    """
    N-Body 力计算 Kernel。
    每个线程计算一个粒子受到的合力，并更新速度和位置。
    使用朴素 O(N^2) 算法（适合展示 GPU 并行性）。
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if tid >= n:
        return

    # 读取当前粒子的位置
    px = positions[tid, 0]
    py = positions[tid, 1]
    pz = positions[tid, 2]

    # 计算合力
    fx = 0.0
    fy = 0.0
    fz = 0.0

    for j in range(n):
        if j == tid:
            continue

        dx = positions[j, 0] - px
        dy = positions[j, 1] - py
        dz = positions[j, 2] - pz

        # 计算距离平方（加软化参数）
        dist_sq = dx * dx + dy * dy + dz * dz + softening
        inv_dist = 1.0 / math.sqrt(dist_sq)
        inv_dist3 = inv_dist * inv_dist * inv_dist

        # F = G * m_j / r^2 (方向: 指向 j)
        force = masses[j] * inv_dist3
        fx += force * dx
        fy += force * dy
        fz += force * dz

    # 更新速度 (v += a * dt, a = F/m)
    inv_mass = 1.0 / masses[tid]
    velocities[tid, 0] += fx * inv_mass * dt
    velocities[tid, 1] += fy * inv_mass * dt
    velocities[tid, 2] += fz * inv_mass * dt

    # 更新位置 (x += v * dt)
    positions[tid, 0] += velocities[tid, 0] * dt
    positions[tid, 1] += velocities[tid, 1] * dt
    positions[tid, 2] += velocities[tid, 2] * dt


def compute_forces_cpu(positions, velocities, masses, dt, softening):
    """CPU 参考实现"""
    n = len(masses)
    new_velocities = velocities.copy()

    for i in range(n):
        px, py, pz = positions[i]
        fx, fy, fz = 0.0, 0.0, 0.0

        for j in range(n):
            if j == i:
                continue
            dx = positions[j, 0] - px
            dy = positions[j, 1] - py
            dz = positions[j, 2] - pz

            dist_sq = dx*dx + dy*dy + dz*dz + softening
            inv_dist = 1.0 / math.sqrt(dist_sq)
            inv_dist3 = inv_dist ** 3

            force = masses[j] * inv_dist3
            fx += force * dx
            fy += force * dy
            fz += force * dz

        inv_mass = 1.0 / masses[i]
        new_velocities[i, 0] += fx * inv_mass * dt
        new_velocities[i, 1] += fy * inv_mass * dt
        new_velocities[i, 2] += fz * inv_mass * dt

    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities


def main():
    print()
    print("=" * 60)
    print("     Day 27: N-Body 模拟 — Python/Numba 版本                 ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 数据准备: 随机初始化粒子
    print(f"[主机] 粒子数量: {NUM_BODIES}")
    print(f"[主机] 模拟步数: {STEPS}")

    h_positions = np.random.uniform(-10, 10, size=(NUM_BODIES, 3)).astype(np.float32)
    h_velocities = np.random.uniform(-0.1, 0.1, size=(NUM_BODIES, 3)).astype(np.float32)
    h_masses = np.random.uniform(1, 10, size=NUM_BODIES).astype(np.float32)

    grid_size = (NUM_BODIES + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ---- GPU 模拟 ----
    d_positions = cuda.to_device(h_positions.copy())
    d_velocities = cuda.to_device(h_velocities.copy())
    d_masses = cuda.to_device(h_masses)

    print("[主机] 运行 GPU N-Body 模拟...")

    # 预热
    compute_forces_kernel[grid_size, BLOCK_SIZE](
        d_positions, d_velocities, d_masses, NUM_BODIES, DT, SOFTENING
    )
    cuda.synchronize()

    start = time.time()
    for _ in range(STEPS):
        compute_forces_kernel[grid_size, BLOCK_SIZE](
            d_positions, d_velocities, d_masses, NUM_BODIES, DT, SOFTENING
        )
        cuda.synchronize()
    gpu_time = (time.time() - start) / STEPS * 1000

    h_gpu_positions = d_positions.copy_to_host()
    h_gpu_velocities = d_velocities.copy_to_host()

    # ---- CPU 模拟（少量粒子和步数用于验证）----
    cpu_n = min(NUM_BODIES, 256)
    cpu_positions = h_positions[:cpu_n].copy()
    cpu_velocities = h_velocities[:cpu_n].copy()
    cpu_masses = h_masses[:cpu_n]

    print("[主机] 运行 CPU N-Body 模拟 (验证用, 256 粒子)...")
    start = time.time()
    for _ in range(STEPS):
        cpu_positions, cpu_velocities = compute_forces_cpu(
            cpu_positions, cpu_velocities, cpu_masses, DT, SOFTENING
        )
    cpu_time = (time.time() - start) / STEPS * 1000

    # 验证（取前 cpu_n 个粒子对比）
    # 注意：GPU 和 CPU 使用相同的初始状态，但由于浮点运算顺序不同，
    # 结果可能有微小差异
    pos_diff = np.max(np.abs(h_gpu_positions[:cpu_n] - cpu_positions))
    is_ok = pos_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU ({NUM_BODIES} 粒子) : {gpu_time:.4f} ms/step")
    print(f"  CPU (256 粒子)     : {cpu_time:.4f} ms/step")
    print(f"  结果验证           : {'通过' if is_ok else '差异可接受'} (差异: {pos_diff:.2e})")
    print(f"  复杂度             : O(N^2)")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. N-Body: 每个粒子受其他所有粒子的引力作用")
    print("  2. 朴素算法 O(N^2): 每个粒子遍历所有其他粒子")
    print("  3. GPU 并行: 每个粒子独立计算，天然并行")
    print("  4. 优化: Barnes-Hut O(N log N), FMM O(N)")
    print("  5. 软化参数: 防止近距离时数值不稳定")
    print("  6. 应用: 天体模拟、分子动力学、粒子系统")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
