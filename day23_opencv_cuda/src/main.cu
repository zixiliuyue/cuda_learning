/******************************************************************************
 * @file    main.cu
 * @brief   Day 23: OpenCV CUDA 模块概念演示
 * @details 完整 OpenCV CUDA 模块需要 OpenCV 编译时启用 CUDA 支持。
 *          本项目展示等效的 CUDA Kernel 实现，对应 OpenCV CUDA 的功能。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define BLOCK_SIZE 16

/**
 * @brief  RGB 转灰度 - 对应 cv::cuda::cvtColor
 */
__global__ void rgb_to_gray_kernel(const uchar3* d_rgb, unsigned char* d_gray, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height)
    {
        int idx = row * width + col;
        uchar3 pixel = d_rgb[idx];
        // BT.601 标准: Y = 0.299*R + 0.587*G + 0.114*B
        d_gray[idx] = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    }
}

/**
 * @brief  高斯模糊 - 对应 cv::cuda::GaussianBlur
 */
__global__ void gaussian_blur_3x3_kernel(const unsigned char* d_in, unsigned char* d_out,
                                          int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= 1 && col < width - 1 && row >= 1 && row < height - 1)
    {
        float sum = 0.0f;
        float kernel[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int idx = (row + dy) * width + (col + dx);
                sum += d_in[idx] * kernel[(dy + 1) * 3 + (dx + 1)];
            }
        }
        d_out[row * width + col] = (unsigned char)(sum / 16.0f);
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 23: OpenCV CUDA 模块概念                             \n");
    printf("=============================================================\n");
    printf("\n");
    printf("  [说明] 完整 OpenCV CUDA 需要编译时启用 WITH_CUDA 选项。\n");
    printf("  [说明] 本项目展示等效的 CUDA Kernel 实现。\n");
    printf("\n");

    print_device_info();

    int width = IMAGE_WIDTH;
    int height = IMAGE_HEIGHT;
    int num_pixels = width * height;
    printf("[主机] 图像尺寸: %d x %d\n", width, height);

    // 分配数据
    uchar3* h_rgb = (uchar3*)malloc(num_pixels * sizeof(uchar3));
    unsigned char* h_gray = (unsigned char*)malloc(num_pixels);
    for (int i = 0; i < num_pixels; i++)
    {
        h_rgb[i].x = (unsigned char)(rand() % 256);
        h_rgb[i].y = (unsigned char)(rand() % 256);
        h_rgb[i].z = (unsigned char)(rand() % 256);
    }

    uchar3* d_rgb = NULL;
    unsigned char *d_gray, *d_blur;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_rgb, num_pixels * sizeof(uchar3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_gray, num_pixels));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_blur, num_pixels));
    CUDA_SAFE_CALL(cudaMemcpy(d_rgb, h_rgb, num_pixels * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // RGB -> Gray
    printf("\n[主机] GPU RGB 转灰度 ...\n");
    GpuTimer timer_gray;
    gpu_timer_init(&timer_gray);
    gpu_timer_start(&timer_gray);
    rgb_to_gray_kernel<<<grid_dim, block_dim>>>(d_rgb, d_gray, width, height);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_gray);
    gpu_timer_print(&timer_gray, "RGB 转灰度 ");

    // Gaussian Blur
    printf("\n[主机] GPU 高斯模糊 ...\n");
    GpuTimer timer_blur;
    gpu_timer_init(&timer_blur);
    gpu_timer_start(&timer_blur);
    gaussian_blur_3x3_kernel<<<grid_dim, block_dim>>>(d_gray, d_blur, width, height);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_blur);
    gpu_timer_print(&timer_blur, "高斯模糊 ");

    printf("\n=============================================================\n");
    printf("     OpenCV CUDA 常用函数                                     \n");
    printf("=============================================================\n");
    printf("  cv::cuda::GpuMat       - GPU 图像数据\n");
    printf("  cv::cuda::cvtColor     - 颜色空间转换\n");
    printf("  cv::cuda::GaussianBlur - 高斯模糊\n");
    printf("  cv::cuda::Sobel        - Sobel 边缘检测\n");
    printf("  cv::cuda::resize       - 图像缩放\n");
    printf("=============================================================\n\n");

    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_blur);
    free(h_rgb); free(h_gray);
    gpu_timer_destroy(&timer_gray);
    gpu_timer_destroy(&timer_blur);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
