/******************************************************************************
 * @file    main.cu
 * @brief   Day 10: CUDA 流与异步执行
 * @details 本程序演示 CUDA 流的创建和使用，实现计算与数据传输的重叠。
 *
 *          核心学习目标：
 *          1. CUDA 流（Stream）的概念与创建
 *          2. 默认流（Stream 0）vs 显式流
 *          3. 主机-设备数据传输与 Kernel 执行的重叠
 *          4. 事件（Event）与精确计时
 *          5. 多流并行执行的最佳实践
 *
 *          流的关键概念：
 *          - 流是 GPU 上操作的 FIFO 队列
 *          - 同一流中的操作按顺序执行
 *          - 不同流中的操作可以并行执行（如果资源允许）
 *          - 默认流（0）会与其他流同步
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define SEGMENT_SIZE (1024 * 1024)  // 每段 1M 元素
#define NUM_SEGMENTS 4
#define BLOCK_SIZE 256

__global__ void vector_scale_kernel(const float* d_in, float* d_out, float scale, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        d_out[tid] = d_in[tid] * scale;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 10: CUDA 流与异步执行                                \n");
    printf("=============================================================\n\n");

    print_device_info();

    size_t segment_bytes = SEGMENT_SIZE * sizeof(float);
    printf("[主机] 每段大小: %d 个元素 (%.2f MB)\n", SEGMENT_SIZE, segment_bytes / (1024.0 * 1024.0));
    printf("[主机] 总段数  : %d\n", NUM_SEGMENTS);

    // 分配主机内存（使用页锁定内存以实现异步传输）
    float* h_in = NULL;
    float* h_out = NULL;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_in, NUM_SEGMENTS * segment_bytes));
    CUDA_SAFE_CALL(cudaMallocHost((void**)&h_out, NUM_SEGMENTS * segment_bytes));

    for (int i = 0; i < NUM_SEGMENTS * SEGMENT_SIZE; i++)
    {
        h_in[i] = (float)(i % 100) * 0.01f;
    }

    // 分配设备内存
    float* d_in = NULL;
    float* d_out = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_in, NUM_SEGMENTS * segment_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, NUM_SEGMENTS * segment_bytes));

    int grid_size = (SEGMENT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* --------------------------------------------------------
     *  串行执行（单流）
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     串行执行（默认流）                                       \n");
    printf("=============================================================\n");

    GpuTimer timer_serial;
    gpu_timer_init(&timer_serial);
    gpu_timer_start(&timer_serial);

    for (int i = 0; i < NUM_SEGMENTS; i++)
    {
        float* h_in_seg = h_in + i * SEGMENT_SIZE;
        float* h_out_seg = h_out + i * SEGMENT_SIZE;
        float* d_in_seg = d_in + i * SEGMENT_SIZE;
        float* d_out_seg = d_out + i * SEGMENT_SIZE;

        // H2D -> Kernel -> D2H，全部串行
        CUDA_SAFE_CALL(cudaMemcpy(d_in_seg, h_in_seg, segment_bytes, cudaMemcpyHostToDevice));
        vector_scale_kernel<<<grid_size, BLOCK_SIZE>>>(d_in_seg, d_out_seg, 2.0f, SEGMENT_SIZE);
        CUDA_SAFE_CALL(cudaMemcpy(h_out_seg, d_out_seg, segment_bytes, cudaMemcpyDeviceToHost));
    }

    gpu_timer_stop(&timer_serial);
    gpu_timer_print(&timer_serial, "串行执行（默认流） ");

    /* --------------------------------------------------------
     *  多流并行执行
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     多流并行执行                                             \n");
    printf("=============================================================\n");

    // 创建多个 CUDA 流
    cudaStream_t streams[NUM_SEGMENTS];
    for (int i = 0; i < NUM_SEGMENTS; i++)
    {
        CUDA_SAFE_CALL(cudaStreamCreate(&streams[i]));
    }

    GpuTimer timer_parallel;
    gpu_timer_init(&timer_parallel);
    gpu_timer_start(&timer_parallel);

    for (int i = 0; i < NUM_SEGMENTS; i++)
    {
        float* h_in_seg = h_in + i * SEGMENT_SIZE;
        float* h_out_seg = h_out + i * SEGMENT_SIZE;
        float* d_in_seg = d_in + i * SEGMENT_SIZE;
        float* d_out_seg = d_out + i * SEGMENT_SIZE;

        // 每个段使用独立的流，可以并行执行
        CUDA_SAFE_CALL(cudaMemcpyAsync(d_in_seg, h_in_seg, segment_bytes,
                                        cudaMemcpyHostToDevice, streams[i]));
        vector_scale_kernel<<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
            d_in_seg, d_out_seg, 2.0f, SEGMENT_SIZE);
        CUDA_SAFE_CALL(cudaMemcpyAsync(h_out_seg, d_out_seg, segment_bytes,
                                        cudaMemcpyDeviceToHost, streams[i]));
    }

    // 同步所有流
    for (int i = 0; i < NUM_SEGMENTS; i++)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(streams[i]));
    }

    gpu_timer_stop(&timer_parallel);
    gpu_timer_print(&timer_parallel, "多流并行执行 ");

    double serial_ms = gpu_timer_get_elapsed_ms(&timer_serial);
    double parallel_ms = gpu_timer_get_elapsed_ms(&timer_parallel);

    printf("\n=============================================================\n");
    printf("                        性能总结                              \n");
    printf("=============================================================\n");
    printf("  串行执行时间    : %.2f ms\n", serial_ms);
    printf("  多流并行时间    : %.2f ms\n", parallel_ms);
    printf("  加速比          : %.2fx\n", serial_ms / parallel_ms);
    printf("=============================================================\n\n");

    // 清理流
    for (int i = 0; i < NUM_SEGMENTS; i++)
    {
        CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]));
    }

    CUDA_SAFE_CALL(cudaFreeHost(h_in));
    CUDA_SAFE_CALL(cudaFreeHost(h_out));
    CUDA_SAFE_CALL(cudaFree(d_in));
    CUDA_SAFE_CALL(cudaFree(d_out));

    gpu_timer_destroy(&timer_serial);
    gpu_timer_destroy(&timer_parallel);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
