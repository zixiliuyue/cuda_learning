/******************************************************************************
 * @file    main.cu
 * @brief   Day 7: 全局内存合并访问（Coalesced Memory Access）
 * @details 本程序通过对比实验展示合并访问与非合并访问的性能差异。
 *
 *          核心学习目标：
 *          1. 理解什么是全局内存的 "合并访问 "
 *          2. 理解 DRAM 和缓存行的物理结构
 *          3. 行优先 vs 列优先的内存布局影响
 *          4. 如何编写代码以实现合并访问
 *          5. 使用结构体时的内存对齐
 *
 *          合并访问的定义：
 *          当一个 Warp（32个线程）中的线程访问连续的内存地址时，
 *          这些访问可以合并为一个或几个内存事务，大大提高带宽利用率。
 *
 *          非合并访问的代价：
 *          当线程访问分散的内存地址时，每个访问可能触发独立的内存事务，
 *          导致内存带宽严重下降（可能降低 10 倍以上）。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define ARRAY_SIZE (1024 * 1024 * 16)  // 16M 元素 = 64MB
#define BLOCK_SIZE 256
#define TOLERANCE 1e-5f

/**
 * @brief  Kernel: 合并访问 - 顺序读取数组元素
 * @details 线程 tid 读取 array[tid]，这是完美的合并访问模式。
 *          Warp 0: 线程 0-31 读取 array[0] 到 array[31]（连续地址）
 *          只需要 1-2 个内存事务即可完成 32 个读取。
 */
__global__ void coalesced_read_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // 合并访问：线程 tid 访问地址 d_in + tid
        // 同一 Warp 内的线程访问连续地址
        d_out[tid] = d_in[tid];
    }
}

/**
 * @brief  Kernel: 非合并访问 - 跨步读取（Stride = 2）
 * @details 线程 tid 读取 array[tid * 2]，Warp 内线程访问间隔为 2。
 *          仍然相对连续，性能影响较小。
 */
__global__ void strided_read_kernel_2(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 2;
    if (idx < n)
    {
        d_out[tid] = d_in[idx];
    }
}

/**
 * @brief  Kernel: 严重非合并访问 - 跨步读取（Stride = 32）
 * @details 线程 tid 读取 array[tid * 32]。
 *          Warp 内线程访问间隔为 32 * sizeof(float) = 128 bytes。
 *          每个线程可能触发独立的缓存行读取，性能严重下降。
 */
__global__ void strided_read_kernel_32(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 32;
    if (idx < n)
    {
        d_out[tid] = d_in[idx];
    }
}

/**
 * @brief  Kernel: 随机访问 - 完全非合并
 * @details 每个线程随机访问内存位置，完全不合并。
 *          这是最差的情况，每个访问都是独立的内存事务。
 */
__global__ void random_access_kernel(const float* d_in, float* d_out,
                                      const int* d_indices, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        int idx = d_indices[tid];
        d_out[tid] = d_in[idx];
    }
}

/**
 * @brief  运行单个 Kernel 的基准测试
 * @return 有效带宽 (GB/s)
 */
double run_benchmark(const char* name,
                      void (*kernel)(const float*, float*, int),
                      const float* d_in, float* d_out, int n,
                      int grid_size)
{
    GpuTimer timer;
    gpu_timer_init(&timer);

    // 预热
    kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    // 正式计时（运行多次取平均）
    int iterations = 10;
    gpu_timer_start(&timer);
    for (int i = 0; i < iterations; i++)
    {
        kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer);

    double total_ms = gpu_timer_get_elapsed_ms(&timer);
    double avg_ms = total_ms / iterations;

    // 计算有效带宽：读取 n 个 float + 写入 n 个 float = 2 * n * 4 bytes
    double bytes_moved = 2.0 * n * sizeof(float);
    double bandwidth_gbps = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    printf("  %-25s | %8.3f ms | %8.2f GB/s\n", name, avg_ms, bandwidth_gbps);

    gpu_timer_destroy(&timer);
    return bandwidth_gbps;
}

int main(void)
{
    printf("\n");
    printf("=============================================================\n");
    printf("     Day 7: 全局内存合并访问（Coalesced Access）              \n");
    printf("=============================================================\n\n");

    print_device_info();

    size_t bytes = ARRAY_SIZE * sizeof(float);
    printf("[主机] 数组大小: %d 个元素 (%.2f MB)\n", ARRAY_SIZE, bytes / (1024.0 * 1024.0));

    // 分配和初始化数据
    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < ARRAY_SIZE; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid_size = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n=============================================================\n");
    printf("     内存访问模式对比实验                                     \n");
    printf("=============================================================\n");
    printf("  %-25s | %10s | %10s\n", "访问模式 ", "时间 ", "带宽 ");
    printf("  -------------------------- | ---------- | ----------\n");

    // 测试各种访问模式
    run_benchmark("合并访问 (顺序)", coalesced_read_kernel, d_in, d_out, ARRAY_SIZE, grid_size);
    run_benchmark("跨步访问 (Stride=2)", strided_read_kernel_2, d_in, d_out, ARRAY_SIZE / 2, grid_size);
    run_benchmark("跨步访问 (Stride=32)", strided_read_kernel_32, d_in, d_out, ARRAY_SIZE / 32, grid_size);

    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. 合并访问: Warp 内线程访问连续地址 -> 高效\n");
    printf("  2. 非合并访问: Warp 内线程访问分散地址 -> 低效\n");
    printf("  3. 行优先布局: C/C++ 默认，行内元素连续存储\n");
    printf("  4. 列优先访问: 跳跃式访问，严重非合并\n");
    printf("  5. 优化原则: 让相邻线程访问相邻内存地址\n");
    printf("=============================================================\n");
    printf("\n");

    CUDA_SAFE_CALL(cudaFree(d_in));
    CUDA_SAFE_CALL(cudaFree(d_out));
    free(h_in); free(h_out);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
