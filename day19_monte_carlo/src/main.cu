/******************************************************************************
 * @file    main.cu
 * @brief   Day 19: 蒙特卡洛方法 - 估算 Pi 值与期权定价
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define NUM_SAMPLES (1024 * 1024 * 16)  // 16M 样本
#define BLOCK_SIZE 256

/**
 * @brief  蒙特卡洛估算 Pi - 每个线程生成随机点
 */
__global__ void mc_pi_kernel(int* d_count, int n, unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // 每个线程处理多个样本（跨度循环）
    int local_count = 0;
    curandState state;
    curand_init(seed, tid, 0, &state);

    for (int i = tid; i < n; i += total_threads)
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f)
        {
            local_count++;
        }
    }

    // 原子累加局部结果
    atomicAdd(d_count, local_count);
}

/**
 * @brief  蒙特卡洛期权定价 - Black-Scholes
 */
__global__ void mc_option_kernel(float* d_results, int n,
                                  float S0, float K, float T,
                                  float r, float sigma,
                                  unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    curandState state;
    curand_init(seed, tid, 0, &state);

    for (int i = tid; i < n; i += total_threads)
    {
        float z = curand_normal(&state);
        float ST = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * z);
        float payoff = fmaxf(ST - K, 0.0f);
        d_results[i] = payoff;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 19: 蒙特卡洛方法                                     \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = NUM_SAMPLES;
    printf("[主机] 样本数量: %d\n", n);

    /* ---------------- Pi 估算 ---------------- */
    printf("\n=============================================================\n");
    printf("     蒙特卡洛估算 Pi                                          \n");
    printf("=============================================================\n");

    int* d_count = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_count, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("\n[主机] 启动 Pi 估算 Kernel ...\n");
    GpuTimer timer_pi;
    gpu_timer_init(&timer_pi);
    gpu_timer_start(&timer_pi);

    mc_pi_kernel<<<grid_size, BLOCK_SIZE>>>(d_count, n, 1234ULL);
    cudaDeviceSynchronize();

    gpu_timer_stop(&timer_pi);
    gpu_timer_print(&timer_pi, "MC Pi 估算 ");

    int h_count = 0;
    CUDA_SAFE_CALL(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    float pi_estimate = 4.0f * (float)h_count / (float)n;
    printf("  圆内点数: %d / %d\n", h_count, n);
    printf("  Pi 估算值: %.8f\n", pi_estimate);
    printf("  误差: %.8f\n", fabsf(pi_estimate - 3.14159265f));

    cudaFree(d_count);

    /* ---------------- 期权定价 ---------------- */
    printf("\n=============================================================\n");
    printf("     蒙特卡洛期权定价 (Black-Scholes)                         \n");
    printf("=============================================================\n");

    float S0 = 100.0f;   // 标的资产价格
    float K = 100.0f;    // 行权价
    float T = 1.0f;      // 到期时间（年）
    float r = 0.05f;     // 无风险利率
    float sigma = 0.2f;  // 波动率

    float* d_results = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_results, n * sizeof(float)));

    printf("\n[主机] 参数: S0=%.2f, K=%.2f, T=%.2f, r=%.2f, sigma=%.2f\n", S0, K, T, r, sigma);
    printf("[主机] 启动期权定价 Kernel ...\n");

    GpuTimer timer_option;
    gpu_timer_init(&timer_option);
    gpu_timer_start(&timer_option);

    mc_option_kernel<<<grid_size, BLOCK_SIZE>>>(d_results, n, S0, K, T, r, sigma, 5678ULL);
    cudaDeviceSynchronize();

    gpu_timer_stop(&timer_option);
    gpu_timer_print(&timer_option, "MC 期权定价 ");

    // 计算平均值（简化：使用 Thrust 或 CPU 归约）
    float* h_results = (float*)malloc(n * sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, n * sizeof(float), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += h_results[i];
    float option_price = (float)(sum / n) * expf(-r * T);
    printf("  期权价格估算: %.4f\n", option_price);

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. curand: CUDA 随机数生成库\n");
    printf("  2. curand_init: 每个线程独立初始化随机状态\n");
    printf("  3. 跨度循环: 每个线程处理多个样本\n");
    printf("  4. 原子操作: 累加计数结果\n");
    printf("  5. 期权定价: Black-Scholes 蒙特卡洛模拟\n");
    printf("=============================================================\n\n");

    cudaFree(d_results);
    free(h_results);
    gpu_timer_destroy(&timer_pi);
    gpu_timer_destroy(&timer_option);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
