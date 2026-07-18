/******************************************************************************
 * @file    main.cu
 * @brief   Day 26: 混合精度训练概念
 * @details 展示 FP16 和 FP32 计算的对比，对应 PyTorch AMP 的底层原理。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuda_utils.h"
#include "timer.h"

#define ARRAY_SIZE (1024 * 1024 * 8)
#define BLOCK_SIZE 256

/**
 * @brief  FP32 向量加法
 */
__global__ void vector_add_fp32(const float* a, const float* b, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}

/**
 * @brief  FP16 向量加法（需要 Compute Capability 5.3+）
 */
__global__ void vector_add_fp16(const half* a, const half* b, half* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = __hadd(a[tid], b[tid]);
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 26: 混合精度训练概念                                 \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = ARRAY_SIZE;
    printf("[主机] 数组大小: %d 个元素\n", n);

    // FP32 数据
    float* h_a = (float*)malloc(n * sizeof(float));
    float* h_b = (float*)malloc(n * sizeof(float));
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++)
    {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // FP32 测试
    float *d_a32, *d_b32, *d_c32;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a32, n * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b32, n * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c32, n * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_a32, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b32, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n[主机] FP32 向量加法 ...\n");
    GpuTimer timer_fp32;
    gpu_timer_init(&timer_fp32);
    gpu_timer_start(&timer_fp32);
    vector_add_fp32<<<grid_size, BLOCK_SIZE>>>(d_a32, d_b32, d_c32, n);
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer_fp32);
    gpu_timer_print(&timer_fp32, "FP32 向量加法 ");

    // FP16 测试
    half* h_a16 = (half*)malloc(n * sizeof(half));
    half* h_b16 = (half*)malloc(n * sizeof(half));
    for (int i = 0; i < n; i++)
    {
        h_a16[i] = __float2half(h_a[i]);
        h_b16[i] = __float2half(h_b[i]);
    }

    half *d_a16, *d_b16, *d_c16;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a16, n * sizeof(half)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b16, n * sizeof(half)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c16, n * sizeof(half)));
    CUDA_SAFE_CALL(cudaMemcpy(d_a16, h_a16, n * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b16, h_b16, n * sizeof(half), cudaMemcpyHostToDevice));

    printf("\n[主机] FP16 向量加法 ...\n");
    GpuTimer timer_fp16;
    gpu_timer_init(&timer_fp16);
    gpu_timer_start(&timer_fp16);
    vector_add_fp16<<<grid_size, BLOCK_SIZE>>>(d_a16, d_b16, d_c16, n);
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer_fp16);
    gpu_timer_print(&timer_fp16, "FP16 向量加法 ");

    printf("\n=============================================================\n");
    printf("     混合精度训练要点                                         \n");
    printf("=============================================================\n");
    printf("  1. FP16: 2 字节，带宽需求减半，Tensor Core 加速\n");
    printf("  2. 损失缩放: 防止梯度下溢到零\n");
    printf("  3. 主权重保持 FP32: 确保精度\n");
    printf("  4. 自动混合精度: 自动选择 FP16/FP32\n");
    printf("  5. PyTorch: torch.cuda.amp.autocast()\n");
    printf("=============================================================\n\n");

    cudaFree(d_a32); cudaFree(d_b32); cudaFree(d_c32);
    cudaFree(d_a16); cudaFree(d_b16); cudaFree(d_c16);
    free(h_a); free(h_b); free(h_a16); free(h_b16);
    gpu_timer_destroy(&timer_fp32);
    gpu_timer_destroy(&timer_fp16);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
