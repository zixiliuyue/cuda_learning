#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 15: 图像滤波 — Python/Numba 版本

使用 Numba 实现简单的图像卷积滤波（3x3 均值滤波和高斯模糊），
对比 CPU 和 GPU 实现的性能差异。
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
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
BLOCK_SIZE = 16
TOLERANCE = 1e-3
FILTER_RADIUS = 1  # 3x3 滤波核

# 均值滤波核 (3x3)
KERNEL_MEAN = np.array([
    [1.0/9, 1.0/9, 1.0/9],
    [1.0/9, 1.0/9, 1.0/9],
    [1.0/9, 1.0/9, 1.0/9]
], dtype=np.float32)

# 高斯模糊核 (3x3)
KERNEL_GAUSS = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
], dtype=np.float32)


@cuda.jit
def conv2d_kernel(input_img, output_img, kernel, width, height, k_size):
    """
    2D 卷积 Kernel。
    每个线程处理输出图像的一个像素。
    使用边界复制（edge padding）处理边界像素。
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x >= width or y >= height:
        return

    half = k_size // 2
    sum_val = 0.0

    for ky in range(k_size):
        for kx in range(k_size):
            # 边界处理: clamp 到有效范围
            ix = min(max(x + kx - half, 0), width - 1)
            iy = min(max(y + ky - half, 0), height - 1)
            sum_val += input_img[iy * width + ix] * kernel[ky * k_size + kx]

    output_img[y * width + x] = sum_val


def conv2d_cpu(image, kernel, width, height, k_size):
    """CPU 版 2D 卷积"""
    half = k_size // 2
    output = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            sum_val = 0.0
            for ky in range(k_size):
                for kx in range(k_size):
                    ix = min(max(x + kx - half, 0), width - 1)
                    iy = min(max(y + ky - half, 0), height - 1)
                    sum_val += image[iy * width + ix] * kernel[ky, kx]
            output[y, x] = sum_val

    return output


def main():
    print()
    print("=" * 60)
    print("     Day 15: 图像滤波 — Python/Numba 版本                   ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 生成测试图像（随机灰度图）
    print(f"[主机] 图像尺寸: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    h_image = np.random.uniform(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
    h_kernel = KERNEL_GAUSS.flatten()  # 展平为1D用于 Kernel
    k_size = 3

    d_image = cuda.to_device(h_image.flatten())
    d_kernel = cuda.to_device(h_kernel)
    d_output = cuda.device_array_like(h_image.flatten())

    grid_x = (IMAGE_WIDTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (IMAGE_HEIGHT + BLOCK_SIZE - 1) // BLOCK_SIZE

    # ---- GPU 卷积 ----
    print("[主机] 运行 GPU 2D 卷积（高斯模糊）...")

    # 预热
    conv2d_kernel[(grid_x, grid_y), (BLOCK_SIZE, BLOCK_SIZE)](
        d_image, d_output, d_kernel, IMAGE_WIDTH, IMAGE_HEIGHT, k_size
    )
    cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(5):
        conv2d_kernel[(grid_x, grid_y), (BLOCK_SIZE, BLOCK_SIZE)](
            d_image, d_output, d_kernel, IMAGE_WIDTH, IMAGE_HEIGHT, k_size
        )
        cuda.synchronize()
    gpu_time = (time.time() - start) / 5 * 1000

    h_gpu_result = d_output.copy_to_host().reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    # ---- CPU 卷积 ----
    print("[主机] 运行 CPU 2D 卷积（高斯模糊）...")
    start = time.time()
    h_cpu_result = conv2d_cpu(h_image, KERNEL_GAUSS, IMAGE_WIDTH, IMAGE_HEIGHT, k_size)
    cpu_time = (time.time() - start) * 1000

    # 验证
    max_diff = np.max(np.abs(h_gpu_result - h_cpu_result))
    is_ok = max_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU 卷积时间 : {gpu_time:.4f} ms")
    print(f"  CPU 卷积时间 : {cpu_time:.4f} ms")
    if gpu_time > 0:
        print(f"  GPU 加速比   : {cpu_time / gpu_time:.2f} x")
    print(f"  结果验证     : {'通过' if is_ok else '失败'} (最大误差: {max_diff:.2e})")
    print(f"  滤波核类型   : 高斯模糊 (3x3)")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 图像卷积: 对每个像素应用滤波核进行加权求和")
    print("  2. 边界处理: clamp（边缘复制）、zero padding、wrap")
    print("  3. GPU 图像处理: 像素级并行，天然适合 GPU")
    print("  4. 共享内存优化: 将滤波核加载到共享内存")
    print("  5. 常见滤波核: 均值、高斯、Sobel、Laplacian")
    print("  6. 实际应用: OpenCV CUDA 模块提供大量预优化 Kernel")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
