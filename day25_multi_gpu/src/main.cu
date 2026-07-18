/******************************************************************************
 * @file    main.cu
 * @brief   Day 25: 多 GPU 编程基础
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define ARRAY_SIZE (1024 * 1024 * 4)
#define BLOCK_SIZE 256

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] = a[tid] + b[tid];
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 25: 多 GPU 编程基础                                  \n");
    printf("=============================================================\n\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("[系统] 检测到 %d 个 GPU 设备\n", device_count);

    if (device_count < 2)
    {
        printf("[提示] 系统中只有 %d 个 GPU，多 GPU 演示使用数据并行分割。\n", device_count);
        printf("[提示] 逻辑仍然适用，实际加速需要多个物理 GPU。\n\n");
    }

    int num_gpus = (device_count < 2) ? 1 : 2;
    int n_per_gpu = ARRAY_SIZE / num_gpus;
    size_t bytes_per_gpu = n_per_gpu * sizeof(float);

    printf("[主机] 数据分割: %d 元素 / GPU\n", n_per_gpu);

    float* h_a = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* h_b = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* h_c = (float*)malloc(ARRAY_SIZE * sizeof(float));
    srand((unsigned int)time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);

    // 在每个 GPU 上执行部分计算
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        CUDA_SAFE_CALL(cudaSetDevice(gpu));

        float *d_a, *d_b, *d_c;
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes_per_gpu));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes_per_gpu));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes_per_gpu));

        int offset = gpu * n_per_gpu;
        CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a + offset, bytes_per_gpu, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b + offset, bytes_per_gpu, cudaMemcpyHostToDevice));

        int grid_size = (n_per_gpu + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, n_per_gpu);

        CUDA_SAFE_CALL(cudaMemcpy(h_c + offset, d_c, bytes_per_gpu, cudaMemcpyDeviceToHost));

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    gpu_timer_stop(&timer);
    gpu_timer_print(&timer, "多 GPU 向量加法 ");

    printf("\n=============================================================\n");
    printf("     多 GPU 编程要点                                          \n");
    printf("=============================================================\n");
    printf("  1. cudaSetDevice: 切换当前 GPU 上下文\n");
    printf("  2. 数据并行: 将数据分割到多个 GPU\n");
    printf("  3. Peer-to-Peer: GPU 间直接内存访问\n");
    printf("  4. NCCL: 多 GPU 通信库（AllReduce 等）\n");
    printf("  5. 负载均衡: 确保每个 GPU 工作量相当\n");
    printf("=============================================================\n\n");

    free(h_a); free(h_b); free(h_c);
    gpu_timer_destroy(&timer);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
