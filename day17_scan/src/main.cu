/******************************************************************************
 * @file    main.cu
 * @brief   Day 17: 并行前缀和（Scan / Prefix Sum）
 * @details 实现 Blelloch 并行 Scan 算法。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define ARRAY_SIZE (1024 * 1024)
#define BLOCK_SIZE 512

/**
 * @brief  Blelloch 上扫（Up-sweep）阶段
 */
__global__ void scan_up_sweep_kernel(float* d_data, int n, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n)
    {
        d_data[idx] += d_data[idx - stride];
    }
}

/**
 * @brief  Blelloch 下扫（Down-sweep）阶段
 */
__global__ void scan_down_sweep_kernel(float* d_data, int n, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = (tid + 1) * stride * 2 - 1;
    if (idx < n)
    {
        float temp = d_data[idx - stride];
        d_data[idx - stride] = d_data[idx];
        d_data[idx] += temp;
    }
}

void scan_cpu(const float* in, float* out, int n)
{
    out[0] = in[0];
    for (int i = 1; i < n; i++)
    {
        out[i] = out[i - 1] + in[i];
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 17: 并行前缀和（Scan）                               \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);
    printf("[主机] 数组大小: %d 个元素\n", n);

    float* h_in = (float*)malloc(bytes);
    float* h_out_gpu = (float*)malloc(bytes);
    float* h_out_cpu = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++) h_in[i] = (float)(rand() % 10);

    float* d_data = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_in, bytes, cudaMemcpyHostToDevice));

    // GPU Scan
    printf("\n[主机] GPU Blelloch Scan ...\n");
    GpuTimer timer_gpu;
    gpu_timer_init(&timer_gpu);
    gpu_timer_start(&timer_gpu);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Up-sweep
    for (int stride = 1; stride < n; stride *= 2)
    {
        scan_up_sweep_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, n, stride);
    }

    // 将最后一个元素置零（Exclusive Scan）
    float zero = 0.0f;
    cudaMemcpy(&d_data[n - 1], &zero, sizeof(float), cudaMemcpyHostToDevice);

    // Down-sweep
    for (int stride = n / 2; stride >= 1; stride /= 2)
    {
        scan_down_sweep_kernel<<<num_blocks, BLOCK_SIZE>>>(d_data, n, stride);
    }

    cudaDeviceSynchronize();
    gpu_timer_stop(&timer_gpu);
    gpu_timer_print(&timer_gpu, "GPU Blelloch Scan");

    CUDA_SAFE_CALL(cudaMemcpy(h_out_gpu, d_data, bytes, cudaMemcpyDeviceToHost));

    // CPU Scan
    printf("\n[主机] CPU 串行 Scan ...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    scan_cpu(h_in, h_out_cpu, n);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU 串行 Scan");

    // 验证
    int correct = 1;
    for (int i = 0; i < n; i++)
    {
        if (fabsf(h_out_gpu[i] - h_out_cpu[i]) > 1e-4f)
        {
            correct = 0;
            printf("  不匹配: idx=%d, GPU=%.4f, CPU=%.4f\n", i, h_out_gpu[i], h_out_cpu[i]);
            if (i > 5) break;
        }
    }
    printf("  验证结果: %s\n", correct ? "通过 " : "失败 ");

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. Scan: 并行前缀和，Inclusive / Exclusive\n");
    printf("  2. Blelloch 算法: 上扫 + 下扫两阶段\n");
    printf("  3. 复杂度: O(log N)，需要多轮 Kernel 启动\n");
    printf("  4. 应用场景: 直方图、流压缩、排序\n");
    printf("=============================================================\n\n");

    cudaFree(d_data);
    free(h_in); free(h_out_gpu); free(h_out_cpu);
    gpu_timer_destroy(&timer_gpu);
    gpu_timer_destroy(&timer_cpu);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
