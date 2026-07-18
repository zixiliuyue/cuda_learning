/******************************************************************************
 * @file    main.cu
 * @brief   Day 16: 并行排序 - Bitonic Sort
 * @details 实现 Bitonic Sort（适合 GPU 的并行排序算法）。
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
#define BLOCK_SIZE 512

/**
 * @brief  Bitonic Sort 核心 Kernel
 * @details 两两比较交换，形成 bitonic 序列后合并
 */
__global__ void bitonic_sort_kernel(float* d_data, int j, int k, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ixj = tid ^ j;

    if (ixj > tid && tid < n)
    {
        // 判断是否需要交换
        if ((tid & k) == 0)
        {
            if (d_data[tid] > d_data[ixj])
            {
                float temp = d_data[tid];
                d_data[tid] = d_data[ixj];
                d_data[ixj] = temp;
            }
        }
        else
        {
            if (d_data[tid] < d_data[ixj])
            {
                float temp = d_data[tid];
                d_data[tid] = d_data[ixj];
                d_data[ixj] = temp;
            }
        }
    }
}

int compare_float(const void* a, const void* b)
{
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 16: 并行排序 - Bitonic Sort                          \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);
    printf("[主机] 数组大小: %d 个元素\n", n);

    float* h_data = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    init_random_float(h_data, n, 0.0f, 1000.0f);
    memcpy(h_ref, h_data, bytes);

    float* d_data = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // GPU Bitonic Sort
    printf("\n[主机] GPU Bitonic Sort ...\n");
    GpuTimer timer_gpu;
    gpu_timer_init(&timer_gpu);
    gpu_timer_start(&timer_gpu);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int k = 2; k <= n; k *= 2)
    {
        for (int j = k / 2; j > 0; j /= 2)
        {
            bitonic_sort_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, j, k, n);
        }
    }
    cudaDeviceSynchronize();

    gpu_timer_stop(&timer_gpu);
    gpu_timer_print(&timer_gpu, "GPU Bitonic Sort");

    CUDA_SAFE_CALL(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // CPU qsort 验证
    printf("\n[主机] CPU qsort ...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    qsort(h_ref, n, sizeof(float), compare_float);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU qsort");

    // 验证
    int correct = 1;
    for (int i = 0; i < n; i++)
    {
        if (fabsf(h_data[i] - h_ref[i]) > 1e-4f)
        {
            correct = 0;
            break;
        }
    }
    printf("  排序结果验证: %s\n", correct ? "通过 " : "失败 ");

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. Bitonic Sort: 适合并行实现的排序网络\n");
    printf("  2. 复杂度: O(log^2 N)，并行度很高\n");
    printf("  3. 比较-交换: 核心操作，无分支发散\n");
    printf("  4. 实际应用: Thrust 库提供更高效的实现\n");
    printf("=============================================================\n\n");

    cudaFree(d_data);
    free(h_data); free(h_ref);
    gpu_timer_destroy(&timer_gpu);
    gpu_timer_destroy(&timer_cpu);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
