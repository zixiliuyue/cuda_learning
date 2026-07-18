#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 23: OpenCV CUDA — Python 版本

演示使用 opencv-python 进行图像处理操作。
OpenCV 的 CUDA 模块提供 GPU 加速的图像处理函数。
不可用时用 NumPy 实现简单的图像处理操作。
"""

import sys
import time
import numpy as np

# 尝试导入 OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("[提示] 未安装 OpenCV，将使用 NumPy fallback。")
    print("[提示] 安装 OpenCV: pip install opencv-python")

# 常量定义
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
TOLERANCE = 1e-3
ITERATIONS = 10


def create_test_image(width, height):
    """创建测试图像（棋盘格 + 渐变背景）"""
    image = np.zeros((height, width), dtype=np.uint8)

    # 渐变背景
    for y in range(height):
        image[y, :] = (y * 255 // height).astype(np.uint8)

    # 棋盘格
    block = 64
    for y in range(0, height, block):
        for x in range(0, width, block):
            if (x // block + y // block) % 2 == 0:
                image[y:y+block, x:x+block] = 128

    # 添加随机噪声
    noise = np.random.randint(-30, 30, size=(height, width), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def gaussian_blur_numpy(image, ksize=5, sigma=1.0):
    """NumPy 实现高斯模糊（简化版）"""
    # 创建高斯核
    ax = np.arange(-ksize // 2 + 1, ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # 简单的卷积
    h, w = image.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float32), ((ph, ph), (pw, pw)), mode='edge')
    result = np.zeros_like(image, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            patch = padded[y:y+kh, x:x+kw]
            result[y, x] = np.sum(patch * kernel)

    return result.astype(np.uint8)


def sobel_numpy(image):
    """NumPy 实现 Sobel 边缘检测"""
    # Sobel 算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    h, w = image.shape
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = image[y-1:y+2, x-1:x+2].astype(np.float32)
            gx[y, x] = np.sum(patch * sobel_x)
            gy[y, x] = np.sum(patch * sobel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    return magnitude.astype(np.uint8)


def main():
    print()
    print("=" * 60)
    print("     Day 23: OpenCV CUDA — Python 版本                       ")
    print("=" * 60)
    print()

    # 创建测试图像
    image = create_test_image(IMAGE_WIDTH, IMAGE_HEIGHT)
    print(f"[主机] 图像尺寸: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")

    if OPENCV_AVAILABLE:
        print("[设备] 使用 OpenCV")

        # ---- OpenCV 高斯模糊 ----
        print("[主机] 运行 OpenCV 高斯模糊...")
        start = time.time()
        for _ in range(ITERATIONS):
            blurred_cv = cv2.GaussianBlur(image, (5, 5), 1.0)
        cv_blur_time = (time.time() - start) / ITERATIONS * 1000

        # ---- OpenCV Sobel 边缘检测 ----
        print("[主机] 运行 OpenCV Sobel 边缘检测...")
        start = time.time()
        for _ in range(ITERATIONS):
            sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            edges_cv = cv2.magnitude(sobel_x, sobel_y)
            edges_cv = np.uint8(np.clip(edges_cv, 0, 255))
        cv_sobel_time = (time.time() - start) / ITERATIONS * 1000

        # ---- NumPy 实现（对比）----
        print("[主机] 运行 NumPy 高斯模糊...")
        start = time.time()
        blurred_np = gaussian_blur_numpy(image, 5, 1.0)
        np_blur_time = (time.time() - start) * 1000

        print("[主机] 运行 NumPy Sobel 边缘检测...")
        start = time.time()
        edges_np = sobel_numpy(image)
        np_sobel_time = (time.time() - start) * 1000

        # 性能总结
        print()
        print("=" * 60)
        print("                        性能总结                            ")
        print("=" * 60)
        print(f"  OpenCV 高斯模糊   : {cv_blur_time:.4f} ms")
        print(f"  NumPy 高斯模糊    : {np_blur_time:.4f} ms")
        print(f"  OpenCV Sobel      : {cv_sobel_time:.4f} ms")
        print(f"  NumPy Sobel       : {np_sobel_time:.4f} ms")
        if np_blur_time > 0:
            print(f"  高斯模糊加速比    : {np_blur_time / cv_blur_time:.2f}x")
        print(f"  输出图像形状       : {blurred_cv.shape}")
        print("=" * 60)
        print()

        print("知识点总结:")
        print("  1. OpenCV CUDA: cv2.cuda 模块提供 GPU 加速图像处理")
        print("  2. cv2.cuda.GaussianBlur(): GPU 高斯模糊")
        print("  3. cv2.cuda.Sobel(): GPU 边缘检测")
        print("  4. cv2.cuda.resize(), threshold(), warpAffine() 等")
        print("  5. cuDNN 也可用于深度学习图像处理")
        print("  6. OpenCV + CUDA: pip install opencv-contrib-python")
        print()
    else:
        # NumPy fallback
        print("[设备] 使用 NumPy fallback")
        print("[主机] 运行 NumPy Sobel 边缘检测...")
        start = time.time()
        edges_np = sobel_numpy(image)
        np_time = (time.time() - start) * 1000

        print()
        print("=" * 60)
        print("                        性能总结                            ")
        print("=" * 60)
        print(f"  NumPy Sobel (CPU) : {np_time:.4f} ms")
        print(f"  输出图像形状       : {edges_np.shape}")
        print("=" * 60)
        print()

        print("知识点总结:")
        print("  1. OpenCV CUDA: GPU 加速的计算机视觉库")
        print("  2. 常用操作: 高斯模糊、边缘检测、特征提取")
        print("  3. pip install opencv-contrib-python 安装完整版")
        print("  4. cv2.cuda 命名空间下为 GPU 版本")
        print("  5. UMat 类型支持透明 GPU 加速")
        print("  6. 应用: 实时视频处理、自动驾驶、工业检测")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
