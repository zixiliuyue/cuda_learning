/******************************************************************************
 * @file    main.cu
 * @brief   Day 22: Numba CUDA 编程 - 概念对应演示
 * @details Numba 的 CUDA 支持与原生 CUDA C++ 高度对应。
 *          本项目展示相同算法的 C++ 版本，可与 python/ 目录中的 Numba 版本对比。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

#define ARRAY_SIZE (1024 * 1024)
#define BLOCK_SIZE 256
#define K 5  // K-Means 聚类数

/**
 * @brief  K-Means 一次迭代 - GPU 实现
 * @details 对应 Numba 版本的 @cuda.jit Kernel
 */
__global__ void kmeans_assign_kernel(const float* d_points, const float* d_centroids,
                                      int* d_labels, int n, int k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float min_dist = 1e20f;
        int best_label = 0;
        float px = d_points[tid * 2];
        float py = d_points[tid * 2 + 1];

        for (int c = 0; c < k; c++)
        {
            float cx = d_centroids[c * 2];
            float cy = d_centroids[c * 2 + 1];
            float dist = (px - cx) * (px - cx) + (py - cy) * (py - cy);
            if (dist < min_dist)
            {
                min_dist = dist;
                best_label = c;
            }
        }
        d_labels[tid] = best_label;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 22: Numba CUDA 编程 - 概念对应                       \n");
    printf("=============================================================\n");
    printf("\n");
    printf("  [说明] Numba 的 @cuda.jit 与 CUDA C++ 的对应关系:\n");
    printf("  - @cuda.jit          -> __global__\n");
    printf("  - cuda.threadIdx.x   -> threadIdx.x\n");
    printf("  - cuda.blockIdx.x    -> blockIdx.x\n");
    printf("  - cuda.blockDim.x    -> blockDim.x\n");
    printf("  - cuda.syncthreads() -> __syncthreads()\n");
    printf("\n");

    print_device_info();

    int n = ARRAY_SIZE;
    printf("[主机] 数据点数量: %d\n", n);

    // 生成 2D 数据点
    float* h_points = (float*)malloc(n * 2 * sizeof(float));
    int* h_labels = (int*)malloc(n * sizeof(int));
    float h_centroids[K * 2];

    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++)
    {
        h_points[i * 2] = (float)rand() / RAND_MAX;
        h_points[i * 2 + 1] = (float)rand() / RAND_MAX;
    }
    for (int c = 0; c < K; c++)
    {
        h_centroids[c * 2] = (float)rand() / RAND_MAX;
        h_centroids[c * 2 + 1] = (float)rand() / RAND_MAX;
    }

    float *d_points, *d_centroids;
    int* d_labels;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_points, n * 2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_centroids, K * 2 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_labels, n * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(d_points, h_points, n * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_centroids, h_centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n[主机] GPU K-Means 一次迭代 ...\n");
    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);
    kmeans_assign_kernel<<<grid_size, BLOCK_SIZE>>>(d_points, d_centroids, d_labels, n, K);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer);
    gpu_timer_print(&timer, "K-Means 分配 ");

    CUDA_SAFE_CALL(cudaMemcpy(h_labels, d_labels, n * sizeof(int), cudaMemcpyDeviceToHost));

    printf("\n=============================================================\n");
    printf("     Numba 与 CUDA C++ 对比                                   \n");
    printf("=============================================================\n");
    printf("  Numba 优点:\n");
    printf("    - 无需编译，即时编译（JIT）\n");
    printf("    - Python 语法，学习曲线平缓\n");
    printf("    - 自动内存管理\n");
    printf("  Numba 缺点:\n");
    printf("    - 调试信息有限\n");
    printf("    - 性能通常不如手写 CUDA C++\n");
    printf("    - 不支持所有 CUDA 特性\n");
    printf("=============================================================\n\n");

    cudaFree(d_points); cudaFree(d_centroids); cudaFree(d_labels);
    free(h_points); free(h_labels);
    gpu_timer_destroy(&timer);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
