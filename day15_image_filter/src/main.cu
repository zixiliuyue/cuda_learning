/******************************************************************************
 * @file    main.cu
 * @brief   Day 15: GPU 图像滤波 - 卷积与边缘检测
 * @details 本程序在 GPU 上实现两种经典图像滤波算子：
 *          1. 高斯模糊（Gaussian Blur）—— 平滑图像，去除噪声
 *          2. Sobel 边缘检测（Sobel Edge Detection）—— 提取图像边缘
 *
 *          核心学习目标：
 *          1. 理解 2D 卷积在 GPU 上的并行化实现
 *          2. 掌握边界处理技巧（clamp / 镜像 / 补零）
 *          3. 学会使用 __constant__ 常量内存存储滤波器核
 *          4. 理解高斯模糊和 Sobel 算子的数学原理
 *          5. 分析访存模式对性能的影响
 *
 *          卷积原理：
 *          输出像素 (i,j) = Σ(输入像素(i+fx, j+fy) * 核(fx, fy))
 *          即滑动窗口加权求和，每个输出像素独立可并行计算。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

/**
 * @brief 测试图像尺寸（宽 x 高）
 * @note  1024x1024 = 约 4 MB 数据（float 单通道），适合快速验证
 */
#define IMAGE_WIDTH 1024
#define IMAGE_HEIGHT 1024

/**
 * @brief 卷积核大小（5x5）
 * @note  核越大越模糊但计算量越大（O(FILTER_SIZE^2) 每像素）
 *        FILTER_SIZE=5 时每个像素需要 25 次乘加
 */
#define FILTER_SIZE 5

/**
 * @brief 每个 Block 的线程数（2D: 16x16 = 256 线程）
 * @note  16x16 的 Block 尺寸在大多数 GPU 上能获得较好的占用率
 *        同时也方便处理边界——每个 Block 处理 16x16 像素区域
 */
#define BLOCK_SIZE 16

/**
 * @brief 高斯核（5x5, sigma=1.0，归一化后）
 * @details 高斯核用于图像平滑，权重由二维高斯函数计算：
 *          G(x,y) = (1/(2πσ²)) * exp(-(x²+y²)/(2σ²))
 *          核内所有元素之和 = 1.0（归一化），保证图像亮度不变。
 *
 *          为什么用 __constant__ 内存？
 *          1. 所有线程读取相同的数据（只读），完美匹配常量内存的 broadcast 机制
 *          2. 常量内存有专用的 L1 缓存（8KB 每 SM），延迟低
 *          3. 相比全局内存读取，常量内存的带宽利用率更高（一个 warp 的线程读同一地址只需一次事务）
 */
__constant__ float gaussian_kernel[FILTER_SIZE * FILTER_SIZE] = {
    0.0037f, 0.0147f, 0.0256f, 0.0147f, 0.0037f,
    0.0147f, 0.0585f, 0.0952f, 0.0585f, 0.0147f,
    0.0256f, 0.0952f, 0.1502f, 0.0952f, 0.0256f,
    0.0147f, 0.0585f, 0.0952f, 0.0585f, 0.0147f,
    0.0037f, 0.0147f, 0.0256f, 0.0147f, 0.0037f
};

/**
 * @brief  高斯模糊 Kernel
 * @param  d_in    输入图像（行优先存储，每个像素一个 float 灰度值）
 * @param  d_out   输出图像（平滑后的结果）
 * @param  width   图像宽度（像素数）
 * @param  height  图像高度（像素数）
 *
 * @details 每个线程处理一个输出像素，对该像素周围的 FILTER_SIZE x FILTER_SIZE
 *          邻域做加权平均。权重来自高斯核，中心像素权重最大，越远权重越小。
 *
 *          边界处理策略：clamp（钳制到边界像素）
 *          当卷积核超出图像边界时，将采样坐标限制在 [0, width-1] 范围内。
 *          这相当于用边界像素"外扩"图像，避免了零补边（zero padding）导致的
 *          黑边伪影和镜像补边（mirror padding）的额外计算开销。
 *
 *          访存模式分析：
 *          相邻线程访问相邻像素（col 连续），满足合并访问条件。
 *          但每个线程需要读取 FILTER_SIZE^2 个邻域像素，导致输入图像
 *          被重复读取（FILTER_SIZE^2 次），这是性能瓶颈所在。
 *          优化方向：使用共享内存 tile 缓存邻域数据（类似 Day 6 的 Tiled MatMul）。
 */
__global__ void gaussian_blur_kernel(const float* d_in, float* d_out, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 只处理图像范围内的像素（超出部分不处理）
    if (col < width && row < height)
    {
        float sum = 0.0f;
        int half = FILTER_SIZE / 2;  // 核半径 = 2（对 5x5 核）

        // 遍历卷积核的每个元素
        // 时间复杂度：O(width * height * FILTER_SIZE^2) = O(N * K^2)
        for (int fy = 0; fy < FILTER_SIZE; fy++)
        {
            for (int fx = 0; fx < FILTER_SIZE; fx++)
            {
                // 计算采样坐标：核中心偏移 (fx - half, fy - half)
                // 用 min(max(...)) 实现 clamp 边界处理
                // 当坐标超出 [0, width-1] 时，钳制到最近的边界像素
                int img_x = min(max(col + fx - half, 0), width - 1);
                int img_y = min(max(row + fy - half, 0), height - 1);

                // 读取输入像素值（行优先索引）
                float pixel = d_in[img_y * width + img_x];
                // 从常量内存读取对应权重
                float weight = gaussian_kernel[fy * FILTER_SIZE + fx];
                sum += pixel * weight;
            }
        }
        d_out[row * width + col] = sum;
    }
}

/**
 * @brief  Sobel 边缘检测 Kernel
 * @param  d_in    输入图像（灰度图）
 * @param  d_out   输出图像（梯度幅值，范围 [0, 1]）
 * @param  width   图像宽度
 * @param  height  图像高度
 *
 * @details Sobel 算子通过计算图像亮度的梯度来检测边缘。
 *          它使用两个 3x3 卷积核分别检测水平（Gx）和垂直（Gy）方向的变化：
 *
 *          Gx（水平边缘检测）：         Gy（垂直边缘检测）：
 *          [-1  0 +1]                  [-1 -2 -1]
 *          [-2  0 +2]                  [ 0  0  0]
 *          [-1  0 +1]                  [+1 +2 +1]
 *
 *          梯度幅值 = sqrt(Gx² + Gy²)
 *          - 幅值大 → 该像素位于边缘上
 *          - 幅值小 → 该像素位于平坦区域
 *
 *          边界处理：Sobel 3x3 核需要至少 1 像素的边界，
 *          超出边界的像素直接设为 0（黑色），不计算梯度。
 *
 * @note   实现要点：
 *          1. 核固定为 3x3，因此未用 __constant__ 而是硬编码
 *          2. 3x3 是固定大小，不需要 FILTER_SIZE 宏——Sobel 核就是 3x3
 *          3. 使用 sqrtf() 计算梯度幅值，fminf() 限制到 [0, 1]
 */
__global__ void sobel_kernel(const float* d_in, float* d_out, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Sobel 3x3 核需要至少 1 像素的边界，边界像素不计算梯度
    if (col >= 1 && col < width - 1 && row >= 1 && row < height - 1)
    {
        // === 水平梯度 Gx ===
        // 用 Sobel X 核与 3x3 邻域做卷积
        // 系数含义：左边列负（-1, -2, -1），右边列正（+1, +2, +1），中间列无贡献（0）
        // 对水平边缘（上下变化）敏感
        float gx =
            // 第一行（row-1）
            -1.0f * d_in[(row - 1) * width + (col - 1)] +
             1.0f * d_in[(row - 1) * width + (col + 1)] +
            // 第二行（row）
            -2.0f * d_in[row * width + (col - 1)] +
             2.0f * d_in[row * width + (col + 1)] +
            // 第三行（row+1）
            -1.0f * d_in[(row + 1) * width + (col - 1)] +
             1.0f * d_in[(row + 1) * width + (col + 1)];

        // === 垂直梯度 Gy ===
        // 用 Sobel Y 核与 3x3 邻域做卷积
        // 系数含义：上边行负（-1, -2, -1），下边行正（+1, +2, +1），中间行无贡献（0）
        // 对垂直边缘（左右变化）敏感
        float gy =
            // 第一行（row-1）
            -1.0f * d_in[(row - 1) * width + (col - 1)] +
            -2.0f * d_in[(row - 1) * width + col] +
            -1.0f * d_in[(row - 1) * width + (col + 1)] +
            // 第三行（row+1）
             1.0f * d_in[(row + 1) * width + (col - 1)] +
             2.0f * d_in[(row + 1) * width + col] +
             1.0f * d_in[(row + 1) * width + (col + 1)];

        // 梯度幅值 = sqrt(Gx^2 + Gy^2)
        // 幅值越大，该像素越可能是边缘
        float magnitude = sqrtf(gx * gx + gy * gy);
        // 钳制到 [0, 1] 范围，避免溢出
        d_out[row * width + col] = fminf(magnitude, 1.0f);
    }
    else if (col < width && row < height)
    {
        // 边界像素直接设为 0（黑色，表示非边缘）
        d_out[row * width + col] = 0.0f;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 15: GPU 图像滤波 - 卷积与边缘检测                    \n");
    printf("=============================================================\n\n");

    print_device_info();

    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    int num_pixels = width * height;
    size_t bytes = num_pixels * sizeof(float);

    printf("[主机] 图像尺寸: %d x %d (%.2f MB)\n", width, height, bytes / (1024.0 * 1024.0));

    // 生成测试图像（棋盘格图案）
    // 棋盘格包含丰富的水平和垂直边缘，适合演示 Sobel 检测效果
    float* h_image = (float*)malloc(bytes);
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int block_size = 64;
            int block_x = col / block_size;
            int block_y = row / block_size;
            // 交替黑白方块：0.2（深灰）和 0.8（浅灰），形成边缘
            h_image[row * width + col] = ((block_x + block_y) % 2 == 0) ? 0.2f : 0.8f;
        }
    }

    float *d_in, *d_out;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_image, bytes, cudaMemcpyHostToDevice));

    // 2D Grid-Block 配置
    // 使用 2D 索引直接映射到图像坐标，无需手动转换
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 高斯模糊
    printf("\n[主机] 执行高斯模糊 ...\n");
    GpuTimer timer_blur;
    gpu_timer_init(&timer_blur);
    gpu_timer_start(&timer_blur);
    gaussian_blur_kernel<<<grid_dim, block_dim>>>(d_in, d_out, width, height);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_blur);
    gpu_timer_print(&timer_blur, "高斯模糊 ");

    // Sobel 边缘检测
    printf("\n[主机] 执行 Sobel 边缘检测 ...\n");
    GpuTimer timer_sobel;
    gpu_timer_init(&timer_sobel);
    gpu_timer_start(&timer_sobel);
    sobel_kernel<<<grid_dim, block_dim>>>(d_in, d_out, width, height);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_sobel);
    gpu_timer_print(&timer_sobel, "Sobel 边缘检测 ");

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. 2D 卷积: 滑动窗口加权求和，每个输出像素独立可并行\n");
    printf("  2. 高斯模糊: 高斯核 = exp(-(x^2+y^2)/2sigma^2)，归一化到和为 1\n");
    printf("  3. Sobel 算子: Gx 检测水平边缘，Gy 检测垂直边缘\n");
    printf("  4. 边界处理: clamp 策略用 min(max()) 实现，避免分支\n");
    printf("  5. 常量内存: 所有线程读相同数据，用 broadcast 提升效率\n");
    printf("  6. 访存模式: 相邻线程访问相邻像素，满足合并访问\n");
    printf("  7. 优化方向: 用共享内存 tile 减少全局内存重复读取\n");
    printf("=============================================================\n\n");

    cudaFree(d_in); cudaFree(d_out);
    free(h_image);
    gpu_timer_destroy(&timer_blur);
    gpu_timer_destroy(&timer_sobel);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}