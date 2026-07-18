#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 30: 综合总结 — Python 版本

30天 CUDA 学习旅程的总结与回顾。
汇总各个主题的核心概念和性能基准。
"""

import sys
import time
import numpy as np

try:
    from numba import cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("[提示] 未安装 Numba。部分演示将使用 NumPy。")

# 常量定义
SIZE = 1024 * 1024
BLOCK_SIZE = 256
TOLERANCE = 1e-5
ITERATIONS = 5


def benchmark_vector_ops():
    """基准测试: 向量运算"""
    a = np.random.uniform(-1, 1, size=SIZE).astype(np.float32)
    b = np.random.uniform(-1, 1, size=SIZE).astype(np.float32)
    results = {}

    if NUMBA_AVAILABLE and cuda.is_available():
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_c = cuda.device_array_like(a)
        grid = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

        @cuda.jit
        def vec_add_k(da, db, dc, n):
            tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if tid < n:
                dc[tid] = da[tid] + db[tid]

        @cuda.jit
        def vec_mul_k(da, db, dc, n):
            tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            if tid < n:
                dc[tid] = da[tid] * db[tid]

        # GPU 向量加
        start = time.time()
        for _ in range(ITERATIONS):
            vec_add_k[grid, BLOCK_SIZE](d_a, d_b, d_c, SIZE)
            cuda.synchronize()
        results['gpu_vec_add'] = (time.time() - start) / ITERATIONS * 1000

        # GPU 向量乘
        start = time.time()
        for _ in range(ITERATIONS):
            vec_mul_k[grid, BLOCK_SIZE](d_a, d_b, d_c, SIZE)
            cuda.synchronize()
        results['gpu_vec_mul'] = (time.time() - start) / ITERATIONS * 1000

    # CPU NumPy
    start = time.time()
    for _ in range(ITERATIONS):
        _ = a + b
    results['cpu_vec_add'] = (time.time() - start) / ITERATIONS * 1000

    start = time.time()
    for _ in range(ITERATIONS):
        _ = a * b
    results['cpu_vec_mul'] = (time.time() - start) / ITERATIONS * 1000

    return results


def benchmark_matrix_ops():
    """基准测试: 矩阵运算"""
    n = 256
    a = np.random.uniform(-0.5, 0.5, size=(n, n)).astype(np.float32)
    b = np.random.uniform(-0.5, 0.5, size=(n, n)).astype(np.float32)
    results = {}

    if NUMBA_AVAILABLE and cuda.is_available():
        d_a = cuda.to_device(a.flatten())
        d_b = cuda.to_device(b.flatten())
        d_c = cuda.device_array_like(a.flatten())

        @cuda.jit
        def matmul_k(da, db, dc, width):
            col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
            if col < width and row < width:
                s = 0.0
                for k in range(width):
                    s += da[row * width + k] * db[k * width + col]
                dc[row * width + col] = s

        bs = 16
        grid = ((n + bs - 1) // bs, (n + bs - 1) // bs)

        start = time.time()
        for _ in range(ITERATIONS):
            matmul_k[grid, (bs, bs)](d_a, d_b, d_c, n)
            cuda.synchronize()
        results['gpu_matmul'] = (time.time() - start) / ITERATIONS * 1000

    start = time.time()
    for _ in range(ITERATIONS):
        _ = a @ b
    results['cpu_matmul'] = (time.time() - start) / ITERATIONS * 1000

    return results


def main():
    print()
    print("=" * 60)
    print("     Day 30: 综合总结 — Python 版本                          ")
    print("=" * 60)
    print()

    print("=" * 60)
    print("              30天 CUDA 学习回顾                              ")
    print("=" * 60)
    print()

    # 30天课程回顾
    curriculum = [
        ("Day 01", "Hello CUDA", "CUDA 编程模型基础"),
        ("Day 02", "向量加法", "GPU 并行计算入门"),
        ("Day 03", "矩阵加法", "2D Grid/Block 配置"),
        ("Day 04", "矩阵转置", "内存布局与合并访问"),
        ("Day 05", "设备查询", "GPU 架构与能力检测"),
        ("Day 06", "Tiled 矩阵乘", "共享内存优化"),
        ("Day 07", "合并访问", "全局内存访问模式"),
        ("Day 08", "占用率优化", "寄存器与 Block Size"),
        ("Day 09", "原子操作与归约", "并行归约算法"),
        ("Day 10", "CUDA 流", "异步执行与重叠"),
        ("Day 11", "常量/纹理内存", "特殊内存类型"),
        ("Day 12", "动态并行", "设备端 Kernel 启动"),
        ("Day 13", "cuBLAS", "线性代数库"),
        ("Day 14", "cuFFT", "快速傅里叶变换"),
        ("Day 15", "图像滤波", "卷积与图像处理"),
        ("Day 16", "并行排序", "Bitonic Sort"),
        ("Day 17", "前缀和扫描", "Blelloch Scan"),
        ("Day 18", "图 BFS", "GPU 图算法"),
        ("Day 19", "蒙特卡洛", "随机采样与估算"),
        ("Day 20", "性能调优", "多策略优化对比"),
        ("Day 21", "PyTorch 扩展", "自定义 CUDA 算子"),
        ("Day 22", "Numba 高级", "设备函数与事件"),
        ("Day 23", "OpenCV CUDA", "图像处理加速"),
        ("Day 24", "TensorRT", "推理优化引擎"),
        ("Day 25", "多 GPU", "数据与模型并行"),
        ("Day 26", "混合精度", "FP16/FP32 训练"),
        ("Day 27", "N-Body", "万有引力模拟"),
        ("Day 28", "光线追踪", "GPU 渲染"),
        ("Day 29", "推理引擎", "完整推理流程"),
        ("Day 30", "综合总结", "知识回顾与基准"),
    ]

    print(f"{'序号':>6} {'主题':<16} {'内容':<30} {'核心概念'}")
    print("-" * 60)
    for day, topic, desc in curriculum:
        print(f"{day:>6} {topic:<16} {desc:<30}")

    # 性能基准
    print()
    print("=" * 60)
    print("                        性能基准                            ")
    print("=" * 60)

    vec_results = benchmark_vector_ops()
    mat_results = benchmark_matrix_ops()

    print(f"\n{'操作':>20} {'CPU (ms)':>12} {'GPU (ms)':>12} {'加速比':>10}")
    print("-" * 60)

    if 'gpu_vec_add' in vec_results and 'cpu_vec_add' in vec_results:
        speedup = vec_results['cpu_vec_add'] / vec_results['gpu_vec_add']
        print(f"{'向量加法':>20} {vec_results['cpu_vec_add']:>12.4f} {vec_results['gpu_vec_add']:>12.4f} {speedup:>10.2f}x")

    if 'gpu_vec_mul' in vec_results and 'cpu_vec_mul' in vec_results:
        speedup = vec_results['cpu_vec_mul'] / vec_results['gpu_vec_mul']
        print(f"{'向量乘法':>20} {vec_results['cpu_vec_mul']:>12.4f} {vec_results['gpu_vec_mul']:>12.4f} {speedup:>10.2f}x")

    if 'gpu_matmul' in mat_results and 'cpu_matmul' in mat_results:
        speedup = mat_results['cpu_matmul'] / mat_results['gpu_matmul']
        print(f"{'矩阵乘法 256x256':>20} {mat_results['cpu_matmul']:>12.4f} {mat_results['gpu_matmul']:>12.4f} {speedup:>10.2f}x")

    if NUMBA_AVAILABLE and cuda.is_available():
        device = cuda.get_current_device()
        name = device.name.decode() if isinstance(device.name, bytes) else device.name
        print(f"\n[设备] GPU: {name}")

    print("=" * 60)
    print()

    # 学习路线建议
    print("进阶学习建议:")
    print("  1. 深入 CUDA C++: 阅读《CUDA C++ Programming Guide》")
    print("  2. 性能分析: Nsight Compute / Nsight Systems")
    print("  3. 深度学习: PyTorch CUDA 扩展 / Triton")
    print("  4. 库: cuDNN, cuBLAS, Thrust, CUB")
    print("  5. 实战: 实现一个完整的 GPU 项目")
    print("  6. 社区: NVIDIA Developer Forums / GitHub")
    print()

    print("=" * 60)
    print("          恭喜完成 30 天 CUDA 学习之旅！                      ")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
