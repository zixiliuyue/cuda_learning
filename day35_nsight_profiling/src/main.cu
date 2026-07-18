/******************************************************************************
 * @file    main.cu
 * @brief   Day 35: Nsight 深度性能分析
 * @details 构造多个具有不同性能特征的 Kernel，并自动计算其带宽 / 计算吞吐 /
 *          占用率 / 利用率，帮助理解如何判断一个 Kernel 是内存受限
 *          (memory-bound) 还是计算受限 (compute-bound)。
 *
 *          本程序配合 Nsight Compute / Nsight Systems 使用效果最佳：
 *              ncu --set full ./nsight_profiling.exe
 *              nsys profile ./nsight_profiling.exe
 *
 *          涉及的四类 Kernel：
 *          1. 内存带宽受限 Kernel —— 向量加法（算术强度极低）
 *          2. 计算受限 Kernel     —— 每元素大量 FMA 运算（算术强度极高）
 *          3. 低占用率 Kernel     —— 使用大量寄存器，限制并行度
 *          4. 高占用率 Kernel     —— 轻量寄存器，SM 可容纳更多 Warp
 *
 * @note    使用 CUDA 13 兼容 API。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

#define N (1024 * 1024 * 16)     /* 16M 个元素 */
#define BLOCK_SIZE 256
#define ITERATIONS 50            /* 计时平均次数 */
#define COMPUTE_ITERS 512        /* 计算受限 Kernel 内部循环次数 */

/* ============================================================
 *  Kernel 1: 内存带宽受限（memory-bound）
 *  ------------------------------------------------------------
 *  每个线程只做一次加法，却要读 2 个 float、写 1 个 float。
 *  算术强度 = 1 FLOP / 12 Bytes ≈ 0.083 FLOP/Byte，极低，
 *  性能几乎完全由显存带宽决定。
 * ============================================================ */
__global__ void kernel_memory_bound(const float* a, const float* b,
                                     float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = a[tid] + b[tid];
    }
}

/* ============================================================
 *  Kernel 2: 计算受限（compute-bound）
 *  ------------------------------------------------------------
 *  只读 1 个 float、写 1 个 float，却在寄存器中做大量 FMA。
 *  算术强度很高，性能由 GPU 的浮点计算能力决定。
 * ============================================================ */
__global__ void kernel_compute_bound(const float* a, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float x = a[tid];
        float acc = 0.0f;
        /* 大量乘加运算，全部在寄存器中完成，不访问全局内存 */
        #pragma unroll 8
        for (int i = 0; i < COMPUTE_ITERS; i++)
        {
            acc = acc * 1.000001f + x;   /* 每次迭代 2 FLOP（乘 + 加） */
        }
        c[tid] = acc;
    }
}

/* ============================================================
 *  Kernel 3: 低占用率（high register usage）
 *  ------------------------------------------------------------
 *  故意使用大量本地变量占用寄存器，使得每个 SM 能同时驻留的
 *  Warp 数量下降，占用率降低，从而暴露占用率对性能的影响。
 * ============================================================ */
__global__ void __launch_bounds__(BLOCK_SIZE)
kernel_low_occupancy(const float* a, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        /* 使用一组独立累加器，迫使编译器分配大量寄存器 */
        float r0 = a[tid], r1 = r0 + 1.0f, r2 = r0 + 2.0f, r3 = r0 + 3.0f;
        float r4 = r0 + 4.0f, r5 = r0 + 5.0f, r6 = r0 + 6.0f, r7 = r0 + 7.0f;
        float r8 = r0 + 8.0f, r9 = r0 + 9.0f, ra = r0 + 10.0f, rb = r0 + 11.0f;
        float rc = r0 + 12.0f, rd = r0 + 13.0f, re = r0 + 14.0f, rf = r0 + 15.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++)
        {
            r0 = r0 * 1.01f + r1; r1 = r1 * 1.01f + r2; r2 = r2 * 1.01f + r3;
            r3 = r3 * 1.01f + r4; r4 = r4 * 1.01f + r5; r5 = r5 * 1.01f + r6;
            r6 = r6 * 1.01f + r7; r7 = r7 * 1.01f + r8; r8 = r8 * 1.01f + r9;
            r9 = r9 * 1.01f + ra; ra = ra * 1.01f + rb; rb = rb * 1.01f + rc;
            rc = rc * 1.01f + rd; rd = rd * 1.01f + re; re = re * 1.01f + rf;
            rf = rf * 1.01f + r0;
        }
        c[tid] = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 +
                 r8 + r9 + ra + rb + rc + rd + re + rf;
    }
}

/* ============================================================
 *  Kernel 4: 高占用率（low register usage）
 *  ------------------------------------------------------------
 *  完成相近的计算量，但使用极少的寄存器，SM 可容纳更多 Warp，
 *  占用率高，通常能更好地隐藏内存与计算延迟。
 * ============================================================ */
__global__ void kernel_high_occupancy(const float* a, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float acc = a[tid];
        #pragma unroll 8
        for (int i = 0; i < 512; i++)
        {
            acc = acc * 1.01f + 0.5f;   /* 单一累加器，寄存器占用极低 */
        }
        c[tid] = acc;
    }
}

/* ============================================================
 *  占用率分析：使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor
 * ============================================================ */
static double analyze_occupancy(const void* kernel, int block_size,
                                const char* name)
{
    int max_active_blocks = 0;
    /* 查询在给定 Block 大小下，每个 SM 最多可驻留多少个 Block */
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, kernel, block_size, 0));

    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, 0));

    /* 每个 SM 的活跃 Warp 数 = 活跃 Block 数 * 每 Block 的 Warp 数 */
    int active_warps  = max_active_blocks * (block_size / prop.warpSize);
    int max_warps     = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    double occupancy  = (double)active_warps / (double)max_warps;

    printf("  %-22s | %10d | %10d | %8.1f%%\n",
           name, max_active_blocks, active_warps, occupancy * 100.0);
    return occupancy;
}

/* ============================================================
 *  计时辅助：运行 Kernel 多次取平均（毫秒）
 * ============================================================ */
static double time_kernel_memory(const float* d_a, const float* d_b,
                                 float* d_c, int n, int grid, int block)
{
    GpuTimer timer;
    gpu_timer_init(&timer);
    kernel_memory_bound<<<grid, block>>>(d_a, d_b, d_c, n);  /* 预热 */
    cudaDeviceSynchronize();
    gpu_timer_start(&timer);
    for (int i = 0; i < ITERATIONS; i++)
        kernel_memory_bound<<<grid, block>>>(d_a, d_b, d_c, n);
    gpu_timer_stop(&timer);
    double ms = gpu_timer_get_elapsed_ms(&timer) / ITERATIONS;
    gpu_timer_destroy(&timer);
    return ms;
}

static double time_kernel_compute(const float* d_a, float* d_c,
                                  int n, int grid, int block)
{
    GpuTimer timer;
    gpu_timer_init(&timer);
    kernel_compute_bound<<<grid, block>>>(d_a, d_c, n);  /* 预热 */
    cudaDeviceSynchronize();
    gpu_timer_start(&timer);
    for (int i = 0; i < ITERATIONS; i++)
        kernel_compute_bound<<<grid, block>>>(d_a, d_c, n);
    gpu_timer_stop(&timer);
    double ms = gpu_timer_get_elapsed_ms(&timer) / ITERATIONS;
    gpu_timer_destroy(&timer);
    return ms;
}

static double time_kernel_low(const float* d_a, float* d_c,
                              int n, int grid, int block)
{
    GpuTimer timer;
    gpu_timer_init(&timer);
    kernel_low_occupancy<<<grid, block>>>(d_a, d_c, n);  /* 预热 */
    cudaDeviceSynchronize();
    gpu_timer_start(&timer);
    for (int i = 0; i < ITERATIONS; i++)
        kernel_low_occupancy<<<grid, block>>>(d_a, d_c, n);
    gpu_timer_stop(&timer);
    double ms = gpu_timer_get_elapsed_ms(&timer) / ITERATIONS;
    gpu_timer_destroy(&timer);
    return ms;
}

static double time_kernel_high(const float* d_a, float* d_c,
                               int n, int grid, int block)
{
    GpuTimer timer;
    gpu_timer_init(&timer);
    kernel_high_occupancy<<<grid, block>>>(d_a, d_c, n);  /* 预热 */
    cudaDeviceSynchronize();
    gpu_timer_start(&timer);
    for (int i = 0; i < ITERATIONS; i++)
        kernel_high_occupancy<<<grid, block>>>(d_a, d_c, n);
    gpu_timer_stop(&timer);
    double ms = gpu_timer_get_elapsed_ms(&timer) / ITERATIONS;
    gpu_timer_destroy(&timer);
    return ms;
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 35: Nsight 深度性能分析                             \n");
    printf("=============================================================\n\n");

    print_device_info();

    /* 查询理论显存带宽（GB/s），用于计算利用率 */
    int mem_clock_khz = 0, mem_bus_width = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, 0);
    double theoretical_bw = 2.0 * (double)mem_clock_khz * 1000.0 *
                            (mem_bus_width / 8.0) / 1e9;

    /* 估算理论计算吞吐（GFLOPS）：SM 数 * 每 SM 每周期 FMA * 2 * 频率 */
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
    /* 现代 GPU 每个 SM 通常有 128 个 FP32 CUDA Core，每周期 2 FLOP(FMA) */
    double theoretical_gflops = (double)prop.multiProcessorCount * 128.0 *
                                2.0 * (double)clock_khz * 1000.0 / 1e9;

    int n = N;
    size_t bytes = (size_t)n * sizeof(float);
    printf("\n[主机] 数组大小: %d 个元素 (%.2f MB)\n",
           n, bytes / (1024.0 * 1024.0));
    printf("[主机] 理论显存带宽: %.1f GB/s\n", theoretical_bw);
    printf("[主机] 理论计算吞吐(FP32): %.1f GFLOPS\n", theoretical_gflops);

    /* 分配设备内存 */
    float *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes));

    float* h_a = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    init_random_float(h_a, n, -1.0f, 1.0f);
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_a, bytes, cudaMemcpyHostToDevice));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* --------------------------------------------------------
     *  占用率分析
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     占用率分析 (cudaOccupancyMaxActiveBlocks...)            \n");
    printf("=============================================================\n");
    printf("  %-22s | %10s | %10s | %8s\n",
           "Kernel", "活跃Block", "活跃Warp", "占用率 ");
    printf("  ---------------------- | ---------- | ---------- | --------\n");
    analyze_occupancy((const void*)kernel_memory_bound,  BLOCK_SIZE, "内存受限 ");
    analyze_occupancy((const void*)kernel_compute_bound, BLOCK_SIZE, "计算受限 ");
    double occ_low  = analyze_occupancy((const void*)kernel_low_occupancy,
                                        BLOCK_SIZE, "低占用率 ");
    double occ_high = analyze_occupancy((const void*)kernel_high_occupancy,
                                        BLOCK_SIZE, "高占用率 ");

    /* --------------------------------------------------------
     *  性能实测
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     性能实测与瓶颈分析                                      \n");
    printf("=============================================================\n");

    /* Kernel 1: 内存受限 —— 3 次内存访问(2读1写) */
    double t_mem = time_kernel_memory(d_a, d_b, d_c, n, grid, BLOCK_SIZE);
    double bw_mem = (3.0 * n * sizeof(float) / 1e9) / (t_mem / 1000.0);
    printf("\n[Kernel 1] 内存带宽受限 (memory-bound)\n");
    printf("  实测耗时     : %.4f ms\n", t_mem);
    printf("  实测带宽     : %.1f GB/s\n", bw_mem);
    printf("  带宽利用率   : %.1f%%\n", bw_mem / theoretical_bw * 100.0);
    printf("  瓶颈判定     : 算术强度极低 -> 受显存带宽限制\n");

    /* Kernel 2: 计算受限 —— 每元素 2*COMPUTE_ITERS FLOP */
    double t_cmp = time_kernel_compute(d_a, d_c, n, grid, BLOCK_SIZE);
    double gflops_cmp = ((double)n * 2.0 * COMPUTE_ITERS / 1e9) /
                        (t_cmp / 1000.0);
    printf("\n[Kernel 2] 计算受限 (compute-bound)\n");
    printf("  实测耗时     : %.4f ms\n", t_cmp);
    printf("  实测吞吐     : %.1f GFLOPS\n", gflops_cmp);
    printf("  计算利用率   : %.1f%%\n", gflops_cmp / theoretical_gflops * 100.0);
    printf("  瓶颈判定     : 算术强度极高 -> 受计算吞吐限制\n");

    /* Kernel 3 & 4: 占用率对比 */
    double t_low  = time_kernel_low(d_a, d_c, n, grid, BLOCK_SIZE);
    double t_high = time_kernel_high(d_a, d_c, n, grid, BLOCK_SIZE);
    printf("\n[Kernel 3/4] 占用率对性能的影响\n");
    printf("  低占用率 Kernel: %.4f ms  (占用率 %.1f%%)\n",
           t_low, occ_low * 100.0);
    printf("  高占用率 Kernel: %.4f ms  (占用率 %.1f%%)\n",
           t_high, occ_high * 100.0);
    if (t_high > 0.0)
        printf("  相对差异       : %.2f x\n", t_low / t_high);

    /* --------------------------------------------------------
     *  性能总结表格
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("                     性能分析总结表格                        \n");
    printf("=============================================================\n");
    printf("  %-14s | %10s | %14s | %8s\n",
           "Kernel", "耗时(ms)", "指标 ", "利用率 ");
    printf("  -------------- | ---------- | -------------- | --------\n");
    printf("  %-14s | %10.4f | %8.1f GB/s | %6.1f%%\n",
           "内存受限 ", t_mem, bw_mem, bw_mem / theoretical_bw * 100.0);
    printf("  %-14s | %10.4f | %7.1f GFLOPS | %6.1f%%\n",
           "计算受限 ", t_cmp, gflops_cmp,
           gflops_cmp / theoretical_gflops * 100.0);
    printf("  %-14s | %10.4f | %10.1f%% 占用 | %8s\n",
           "低占用率 ", t_low, occ_low * 100.0, "-");
    printf("  %-14s | %10.4f | %10.1f%% 占用 | %8s\n",
           "高占用率 ", t_high, occ_high * 100.0, "-");
    printf("=============================================================\n");

    printf("\n[Roofline 提示]\n");
    printf("  算术强度 = FLOP / 访存字节数 (FLOP/Byte)\n");
    printf("  拐点强度 = 理论算力 / 理论带宽 = %.2f FLOP/Byte\n",
           theoretical_gflops / theoretical_bw);
    printf("  低于拐点 -> memory-bound；高于拐点 -> compute-bound\n");
    printf("  建议使用: ncu --set full ./nsight_profiling.exe 深入分析\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a);

    printf("\n[主机] 程序正常退出。\n\n");
    return 0;
}
