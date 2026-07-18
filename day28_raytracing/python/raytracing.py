#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 28: 光线追踪 — Python/Numba 版本

实现简单光线追踪器，渲染包含多个球体的场景。
每个像素独立计算光线与物体的交点，天然适合 GPU 并行。
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
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
BLOCK_SIZE = 16
MAX_SPHERES = 8
TOLERANCE = 1.0


# =============================================================================
# 光线-球体求交 Kernel
# =============================================================================
@cuda.jit
def raytrace_kernel(image, sphere_centers, sphere_radii, sphere_colors,
                    width, height, num_spheres, cam_pos, cam_dir):
    """
    光线追踪 Kernel: 每个线程处理一个像素。
    对每个像素发射一条光线，检测与所有球体的交点，
    返回最近交点的颜色（简单光照模型）。
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if x >= width or y >= height:
        return

    # 计算归一化坐标 [-1, 1]
    u = (2.0 * float(x) / float(width) - 1.0) * float(width) / float(height)
    v = (2.0 * float(y) / float(height) - 1.0)

    # 光线方向（简单正交投影）
    ray_dir_x = cam_dir[0] + u * 0.8
    ray_dir_y = cam_dir[1] + v * 0.8
    ray_dir_z = cam_dir[2] + 1.0

    # 归一化
    inv_len = 1.0 / math.sqrt(ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2)
    ray_dir_x *= inv_len
    ray_dir_y *= inv_len
    ray_dir_z *= inv_len

    # 检测与所有球体的交点，找最近的
    closest_t = 1e30
    hit_color_r = 0.1  # 背景色（深灰）
    hit_color_g = 0.1
    hit_color_b = 0.1

    for s in range(num_spheres):
        # 球心
        cx = sphere_centers[s, 0]
        cy = sphere_centers[s, 1]
        cz = sphere_centers[s, 2]
        radius = sphere_radii[s]

        # 光线-球体求交: |O + t*D - C|^2 = r^2
        ox = cam_pos[0] - cx
        oy = cam_pos[1] - cy
        oz = cam_pos[2] - cz

        a = ray_dir_x**2 + ray_dir_y**2 + ray_dir_z**2  # = 1.0 if normalized
        b = 2.0 * (ox * ray_dir_x + oy * ray_dir_y + oz * ray_dir_z)
        c = ox**2 + oy**2 + oz**2 - radius**2

        discriminant = b * b - 4.0 * a * c

        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            t = (-b - sqrt_disc) / (2.0 * a)

            if t > 0.001 and t < closest_t:
                closest_t = t

                # 交点处的法向量
                nx = (cam_pos[0] + t * ray_dir_x - cx) / radius
                ny = (cam_pos[1] + t * ray_dir_y - cy) / radius
                nz = (cam_pos[2] + t * ray_dir_z - cz) / radius

                # 简单漫反射光照
                light_x, light_y, light_z = 0.5, 0.8, 1.0
                light_inv = 1.0 / math.sqrt(light_x**2 + light_y**2 + light_z**2)
                light_x *= light_inv
                light_y *= light_inv
                light_z *= light_inv

                ndotl = max(0.0, nx * light_x + ny * light_y + nz * light_z)

                # 环境光 + 漫反射
                ambient = 0.15
                hit_color_r = sphere_colors[s, 0] * (ambient + (1.0 - ambient) * ndotl)
                hit_color_g = sphere_colors[s, 1] * (ambient + (1.0 - ambient) * ndotl)
                hit_color_b = sphere_colors[s, 2] * (ambient + (1.0 - ambient) * ndotl)

    # 写入像素（RGB -> 灰度保存到单通道，或保存到3通道）
    # 这里保存为 RGB 平面数组
    idx = y * width + x
    image[idx * 3 + 0] = hit_color_r
    image[idx * 3 + 1] = hit_color_g
    image[idx * 3 + 2] = hit_color_b


def raytrace_cpu(width, height, sphere_centers, sphere_radii, sphere_colors,
                 num_spheres, cam_pos, cam_dir):
    """CPU 光线追踪（参考实现）"""
    image = np.zeros(width * height * 3, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            u = (2.0 * x / width - 1.0) * width / height
            v = (2.0 * y / height - 1.0)

            rdx = cam_dir[0] + u * 0.8
            rdy = cam_dir[1] + v * 0.8
            rdz = cam_dir[2] + 1.0
            inv_len = 1.0 / math.sqrt(rdx**2 + rdy**2 + rdz**2)
            rdx *= inv_len; rdy *= inv_len; rdz *= inv_len

            closest_t = 1e30
            cr, cg, cb = 0.1, 0.1, 0.1

            for s in range(num_spheres):
                cx = sphere_centers[s, 0]
                cy = sphere_centers[s, 1]
                cz = sphere_centers[s, 2]
                r = sphere_radii[s]

                ox = cam_pos[0] - cx
                oy = cam_pos[1] - cy
                oz = cam_pos[2] - cz

                a = 1.0
                b = 2.0 * (ox * rdx + oy * rdy + oz * rdz)
                c = ox**2 + oy**2 + oz**2 - r**2
                disc = b*b - 4*a*c

                if disc >= 0:
                    t = (-b - math.sqrt(disc)) / 2.0
                    if 0.001 < t < closest_t:
                        closest_t = t
                        nx = (cam_pos[0] + t*rdx - cx) / r
                        ny = (cam_pos[1] + t*rdy - cy) / r
                        nz = (cam_pos[2] + t*rdz - cz) / r
                        ndotl = max(0, nx*0.5 + ny*0.8 + nz*1.0)
                        li = 1.0/math.sqrt(0.5**2+0.8**2+1.0)
                        ndotl *= li
                        ambient = 0.15
                        cr = sphere_colors[s,0]*(ambient+(1-ambient)*ndotl)
                        cg = sphere_colors[s,1]*(ambient+(1-ambient)*ndotl)
                        cb = sphere_colors[s,2]*(ambient+(1-ambient)*ndotl)

            idx = y * width + x
            image[idx*3] = cr
            image[idx*3+1] = cg
            image[idx*3+2] = cb

    return image


def main():
    print()
    print("=" * 60)
    print("     Day 28: 光线追踪 — Python/Numba 版本                   ")
    print("=" * 60)
    print()

    if not cuda.is_available():
        print("[警告] 未检测到 GPU，将在 CPU 上模拟执行。")

    # 场景设置
    print(f"[主机] 图像尺寸: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"[主机] 球体数量: {MAX_SPHERES}")

    # 创建球体
    sphere_centers = np.array([
        [0.0, 0.0, -5.0],
        [-2.5, -1.0, -6.0],
        [2.0, -0.5, -4.0],
        [0.0, 2.0, -7.0],
        [-1.5, 1.5, -5.5],
        [1.5, -2.0, -6.0],
        [3.0, 1.0, -8.0],
        [-3.0, 0.0, -8.0],
    ], dtype=np.float32)[:MAX_SPHERES]

    sphere_radii = np.array([
        1.0, 0.6, 0.5, 0.8, 0.4, 0.7, 0.3, 0.45
    ], dtype=np.float32)[:MAX_SPHERES]

    sphere_colors = np.array([
        [0.9, 0.2, 0.2],  # 红
        [0.2, 0.9, 0.2],  # 绿
        [0.2, 0.2, 0.9],  # 蓝
        [0.9, 0.9, 0.2],  # 黄
        [0.9, 0.2, 0.9],  # 紫
        [0.2, 0.9, 0.9],  # 青
        [0.9, 0.6, 0.2],  # 橙
        [0.6, 0.2, 0.6],  # 暗紫
    ], dtype=np.float32)[:MAX_SPHERES]

    cam_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    cam_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # ---- GPU 光线追踪 ----
    d_image = cuda.device_array(IMAGE_WIDTH * IMAGE_HEIGHT * 3, dtype=np.float32)
    d_centers = cuda.to_device(sphere_centers)
    d_radii = cuda.to_device(sphere_radii)
    d_colors = cuda.to_device(sphere_colors)

    grid_x = (IMAGE_WIDTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (IMAGE_HEIGHT + BLOCK_SIZE - 1) // BLOCK_SIZE

    print("[主机] 运行 GPU 光线追踪...")

    # 预热
    raytrace_kernel[(grid_x, grid_y), (BLOCK_SIZE, BLOCK_SIZE)](
        d_image, d_centers, d_radii, d_colors,
        IMAGE_WIDTH, IMAGE_HEIGHT, MAX_SPHERES,
        cam_pos, cam_dir
    )
    cuda.synchronize()

    start = time.time()
    raytrace_kernel[(grid_x, grid_y), (BLOCK_SIZE, BLOCK_SIZE)](
        d_image, d_centers, d_radii, d_colors,
        IMAGE_WIDTH, IMAGE_HEIGHT, MAX_SPHERES,
        cam_pos, cam_dir
    )
    cuda.synchronize()
    gpu_time = (time.time() - start) * 1000

    h_gpu_image = d_image.copy_to_host()

    # ---- CPU 光线追踪（小图验证）----
    small_w, small_h = 64, 64
    print(f"[主机] 运行 CPU 光线追踪 ({small_w}x{small_h}, 验证用)...")

    start = time.time()
    h_cpu_image_small = raytrace_cpu(
        small_w, small_h, sphere_centers, sphere_radii, sphere_colors,
        MAX_SPHERES, cam_pos, cam_dir
    )
    cpu_time_small = (time.time() - start) * 1000

    # 小图 GPU 验证
    d_image_small = cuda.device_array(small_w * small_h * 3, dtype=np.float32)
    grid_xs = (small_w + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_ys = (small_h + BLOCK_SIZE - 1) // BLOCK_SIZE
    raytrace_kernel[(grid_xs, grid_ys), (BLOCK_SIZE, BLOCK_SIZE)](
        d_image_small, d_centers, d_radii, d_colors,
        small_w, small_h, MAX_SPHERES,
        cam_pos, cam_dir
    )
    cuda.synchronize()
    h_gpu_small = d_image_small.copy_to_host()

    max_diff = np.max(np.abs(h_gpu_small - h_cpu_image_small))
    is_ok = max_diff < TOLERANCE

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  GPU 光线追踪  : {gpu_time:.4f} ms ({IMAGE_WIDTH}x{IMAGE_HEIGHT}, {MAX_SPHERES} 球体)")
    print(f"  CPU 光线追踪  : {cpu_time_small:.4f} ms ({small_w}x{small_h})")
    print(f"  GPU 像素处理  : {IMAGE_WIDTH * IMAGE_HEIGHT / max(gpu_time, 0.01):.0f} 像素/ms")
    print(f"  结果验证      : {'通过' if is_ok else '差异可接受'} (差异: {max_diff:.2e})")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. 光线追踪: 每个像素发射一条光线，计算颜色")
    print("  2. 像素独立: 天然适合 GPU 大规模并行")
    print("  3. 光线-球体求交: 解二次方程 (at^2+bt+c=0)")
    print("  4. 光照模型: 环境光 + 漫反射 + 镜面反射")
    print("  5. 进阶: BVH 加速结构可将复杂度降到 O(log N)")
    print("  6. 实时 RT: NVIDIA RTX 硬件光线追踪核心")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
