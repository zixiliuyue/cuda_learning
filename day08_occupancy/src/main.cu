/******************************************************************************
 * @file    main.cu
 * @brief   Day 8: 寄存器、局部内存与占用率优化
 * @details 本程序通过不同寄存器使用量的 Kernel 对比，展示占用率概念。
 *
 *          核心学习目标：
 *          1. 理解寄存器文件的结构与分配策略
 *          2. 理解寄存器溢出到局部内存的性能代价
 *          3. 理解占用率（Occupancy）概念与计算
 *          4. 使用 __launch_bounds__ 限制寄存器使用
 *          5. 编译器优化等级的选择
 *
 *          占用率 (Occupancy) 定义：
 *          每个 SM 上实际驻留的 Warp 数量 / 最大可能驻留的 Warp 数量
 *          高占用率可以隐藏内存延迟，但过高不一定更好。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define ARRAY_SIZE (1024 * 1024)
#define BLOCK_SIZE 256

/**
 * @brief  简单 Kernel（使用少量寄存器）
 */
__global__ void simple_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float val = d_in[tid];
        d_out[tid] = val * val;
    }
}

/**
 * @brief  复杂 Kernel（使用大量寄存器）
 * @details 通过大量中间变量增加寄存器使用量
 */
__global__ void heavy_register_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // 大量中间变量，增加寄存器使用
        float a1 = d_in[tid];
        float a2 = a1 * a1;
        float a3 = a2 * a1;
        float a4 = a3 * a2;
        float a5 = a4 * a3;
        float a6 = a5 * a4;
        float a7 = a6 * a5;
        float a8 = a7 * a6;
        float a9 = a8 * a7;
        float a10 = a9 * a8;
        float a11 = a10 * a9;
        float a12 = a11 * a10;
        float a13 = a12 * a11;
        float a14 = a13 * a12;
        float a15 = a14 * a13;
        float a16 = a15 * a14;
        d_out[tid] = a16;
    }
}

/**
 * @brief  限制寄存器的 Kernel（使用 __launch_bounds__）
 * @details __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
 *          提示编译器：每个 Block 最多 256 线程，每个 SM 至少 2 个 Block
 *          编译器会根据此提示减少寄存器分配，可能使用局部内存代替
 */
__launch_bounds__(256, 2)
__global__ void bounded_register_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // 与 heavy_register_kernel 相同逻辑
        float a1 = d_in[tid];
        float a2 = a1 * a1;
        float a3 = a2 * a1;
        float a4 = a3 * a2;
        float a5 = a4 * a3;
        float a6 = a5 * a4;
        float a7 = a6 * a5;
        float a8 = a7 * a6;
        float a9 = a8 * a7;
        float a10 = a9 * a8;
        float a11 = a10 * a9;
        float a12 = a11 * a10;
        float a13 = a12 * a11;
        float a14 = a13 * a12;
        float a15 = a14 * a13;
        float a16 = a15 * a14;
        d_out[tid] = a16;
    }
}

/**
 * @brief  计算并打印理论占用率
 */
void print_occupancy_info(void (*kernel)(const float*, float*, int), int block_size)
{
    cudaFuncAttributes attr;
    CUDA_SAFE_CALL(cudaFuncGetAttributes(&attr, kernel));

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // 计算占用率
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int warps_per_block = (block_size + prop.warpSize - 1) / prop.warpSize;
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;

    // 寄存器限制
    int max_blocks_by_regs = prop.regsPerMultiprocessor / (attr.numRegs * block_size);
    if (max_blocks_by_regs == 0) max_blocks_by_regs = 1;

    int actual_blocks = (max_blocks_per_sm < max_blocks_by_regs) ?
                        max_blocks_per_sm : max_blocks_by_regs;
    int actual_warps = actual_blocks * warps_per_block;
    float occupancy = (float)actual_warps / (float)max_warps_per_sm * 100.0f;

    printf("    寄存器/线程     : %d\n", attr.numRegs);
    printf("    共享内存/Block  : %zu bytes\n", attr.sharedSizeBytes);
    printf("    理论占用率      : %.1f%% (%d / %d Warps)\n",
           occupancy, actual_warps, max_warps_per_sm);
}

double run_kernel_test(const char* name,
                        void (*kernel)(const float*, float*, int),
                        const float* d_in, float* d_out, int n, int grid_size)
{
    GpuTimer timer;
    gpu_timer_init(&timer);

    // 预热
    kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    int iterations = 20;
    gpu_timer_start(&timer);
    for (int i = 0; i < iterations; i++)
    {
        kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer);

    double avg_ms = gpu_timer_get_elapsed_ms(&timer) / iterations;
    printf("  %-30s | %8.3f ms\n", name, avg_ms);

    gpu_timer_destroy(&timer);
    return avg_ms;
}

int main(void)
{
    printf("\n");
    printf("=============================================================\n");
    printf("     Day 8: 寄存器、局部内存与占用率优化                      \n");
    printf("=============================================================\n\n");

    print_device_info();

    size_t bytes = ARRAY_SIZE * sizeof(float);
    printf("[主机] 数组大小: %d 个元素\n", ARRAY_SIZE);

    float* h_in = (float*)malloc(bytes);
    for (int i = 0; i < ARRAY_SIZE; i++) h_in[i] = 1.0f + (float)i * 0.0001f;

    float *d_in, *d_out;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int grid_size = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n=============================================================\n");
    printf("     Kernel 占用率分析                                        \n");
    printf("=============================================================\n\n");

    printf("[分析] 简单 Kernel（少量寄存器）:\n");
    print_occupancy_info(simple_kernel, BLOCK_SIZE);
    printf("\n");

    printf("[分析] 复杂 Kernel（大量寄存器）:\n");
    print_occupancy_info(heavy_register_kernel, BLOCK_SIZE);
    printf("\n");

    printf("[分析] 限制寄存器的 Kernel（launch_bounds）:\n");
    print_occupancy_info(bounded_register_kernel, BLOCK_SIZE);
    printf("\n");

    printf("=============================================================\n");
    printf("     性能对比                                                 \n");
    printf("=============================================================\n");
    printf("  %-30s | %10s\n", "Kernel", "平均时间 ");
    printf("  ------------------------------ | ----------\n");

    run_kernel_test("简单 Kernel", simple_kernel, d_in, d_out, ARRAY_SIZE, grid_size);
    run_kernel_test("复杂 Kernel（大量寄存器） ", heavy_register_kernel, d_in, d_out, ARRAY_SIZE, grid_size);
    run_kernel_test("限制寄存器 Kernel", bounded_register_kernel, d_in, d_out, ARRAY_SIZE, grid_size);

    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. 寄存器: GPU 上最快的存储，数量有限\n");
    printf("  2. 局部内存: 寄存器溢出时自动使用，速度慢\n");
    printf("  3. 占用率: 实际驻留 Warp / 最大可能 Warp\n");
    printf("  4. __launch_bounds__: 限制寄存器使用，提高占用率\n");
    printf("  5. 高占用率可以隐藏内存延迟\n");
    printf("=============================================================\n");
    printf("\n");

    CUDA_SAFE_CALL(cudaFree(d_in));
    CUDA_SAFE_CALL(cudaFree(d_out));
    free(h_in);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
