/******************************************************************************
 * @file    main.cu
 * @brief   Day 20: 性能分析与调优综合
 * @details 对一个向量加法 Kernel 进行多轮优化，展示调优方法论。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

#define ARRAY_SIZE (1024 * 1024 * 16)
#define TOLERANCE 1e-5f

/* ============================================================
 *  版本 1: 最朴素的实现
 * ============================================================ */
__global__ void v1_naive(const float* a, const float* b, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = a[tid] + b[tid];
    }
}

/* ============================================================
 *  版本 2: 循环展开（每个线程处理 4 个元素）
 * ============================================================ */
__global__ void v2_unroll(const float* a, const float* b, float* c, int n)
{
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (tid + 3 < n)
    {
        c[tid] = a[tid] + b[tid];
        c[tid + 1] = a[tid + 1] + b[tid + 1];
        c[tid + 2] = a[tid + 2] + b[tid + 2];
        c[tid + 3] = a[tid + 3] + b[tid + 3];
    }
}

/* ============================================================
 *  版本 3: 使用 __restrict__ 和 const 提示
 * ============================================================ */
__global__ void v3_restrict(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = a[tid] + b[tid];
    }
}

/* ============================================================
 *  版本 4: 使用 float4 向量化加载
 * ============================================================ */
__global__ void v4_float4(const float4* a, const float4* b, float4* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n / 4)
    {
        float4 av = a[tid];
        float4 bv = b[tid];
        float4 cv;
        cv.x = av.x + bv.x;
        cv.y = av.y + bv.y;
        cv.z = av.z + bv.z;
        cv.w = av.w + bv.w;
        c[tid] = cv;
    }
}

double benchmark_kernel(void (*kernel)(const float*, const float*, float*, int),
                        const float* d_a, const float* d_b, float* d_c,
                        int n, int grid, int block, const char* name)
{
    GpuTimer timer;
    gpu_timer_init(&timer);

    // 预热
    kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    int iterations = 50;
    gpu_timer_start(&timer);
    for (int i = 0; i < iterations; i++)
    {
        kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer);

    double avg_ms = gpu_timer_get_elapsed_ms(&timer) / iterations;
    double bandwidth = (3.0 * n * sizeof(float) / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);
    printf("  %-20s | %8.3f ms | %8.2f GB/s\n", name, avg_ms, bandwidth);

    gpu_timer_destroy(&timer);
    return avg_ms;
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 20: 性能分析与调优综合                               \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);
    printf("[主机] 数组大小: %d 个元素 (%.2f MB)\n", n, bytes / (1024.0 * 1024.0));

    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    init_random_float(h_a, n, -1.0f, 1.0f);
    init_random_float(h_b, n, -1.0f, 1.0f);

    float *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    printf("\n=============================================================\n");
    printf("     优化版本对比                                             \n");
    printf("=============================================================\n");
    printf("  %-20s | %10s | %10s\n", "版本 ", "时间 ", "带宽 ");
    printf("  -------------------- | ---------- | ----------\n");

    int grid1 = (n + 255) / 256;
    benchmark_kernel(v1_naive, d_a, d_b, d_c, n, grid1, 256, "V1 朴素 ");
    benchmark_kernel(v2_unroll, d_a, d_b, d_c, n, grid1 / 4 + 1, 256, "V2 循环展开 ");
    benchmark_kernel(v3_restrict, d_a, d_b, d_c, n, grid1, 256, "V3 restrict");

    // float4 版本需要不同的参数类型
    printf("  %-20s | %8s | %8s\n", "V4 float4", "需要类型转换 ", "向量化加载 ");

    printf("\n=============================================================\n");
    printf("                        调优方法论                            \n");
    printf("=============================================================\n");
    printf("  1. 基准测试: 先测量原始性能\n");
    printf("  2. Roofline 分析: 判断计算受限还是内存受限\n");
    printf("  3. 内存优化: 合并访问、使用共享内存\n");
    printf("  4. 计算优化: 循环展开、向量化\n");
    printf("  5. 占用率优化: 调整 Block 大小和寄存器使用\n");
    printf("  6. 迭代验证: 每次优化后验证结果正确性\n");
    printf("=============================================================\n\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
