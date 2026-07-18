/******************************************************************************
 * @file    main.cu
 * @brief   Day 14: cuFFT - GPU 加速的快速傅里叶变换
 * @details 演示使用 cuFFT 进行 1D FFT 和逆 FFT。
 *
 *          FFT（快速傅里叶变换）是信号处理的核心算法，
 *          cuFFT 提供了高度优化的 GPU 实现。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda_utils.h"
#include "timer.h"

#define SIGNAL_SIZE (1024 * 1024)  // 1M 点

/**
 * @brief  检查 cuFFT 错误
 */
#define CUFFT_SAFE_CALL(call)                                               \
    do {                                                                    \
        cufftResult result = (call);                                        \
        if (result != CUFFT_SUCCESS) {                                      \
            fprintf(stderr, "[cuFFT 错误] 状态码: %d | 文件: %s | 行: %d\n", \
                    (int)result, __FILE__, __LINE__);                       \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 14: cuFFT - GPU 加速的快速傅里叶变换                 \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = SIGNAL_SIZE;
    printf("[主机] 信号长度: %d 点\n", n);

    // 分配主机内存
    cufftComplex* h_signal = (cufftComplex*)malloc(n * sizeof(cufftComplex));
    cufftComplex* h_result = (cufftComplex*)malloc(n * sizeof(cufftComplex));

    // 生成测试信号：两个正弦波的叠加
    for (int i = 0; i < n; i++)
    {
        float t = (float)i / (float)n;
        float real = sinf(2.0f * 3.14159265f * 10.0f * t) +
                     0.5f * sinf(2.0f * 3.14159265f * 50.0f * t);
        h_signal[i].x = real;  // 实部
        h_signal[i].y = 0.0f;  // 虚部
    }

    // 分配设备内存
    cufftComplex* d_signal = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_signal, n * sizeof(cufftComplex)));
    CUDA_SAFE_CALL(cudaMemcpy(d_signal, h_signal, n * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // 创建 cuFFT 计划
    cufftHandle plan;
    CUFFT_SAFE_CALL(cufftPlan1d(&plan, n, CUFFT_C2C, 1));

    // 执行正向 FFT
    printf("\n[主机] 执行正向 FFT ...\n");
    GpuTimer timer_fft;
    gpu_timer_init(&timer_fft);
    gpu_timer_start(&timer_fft);

    CUFFT_SAFE_CALL(cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD));

    gpu_timer_stop(&timer_fft);
    gpu_timer_print(&timer_fft, "cuFFT 正向变换 ");

    // 执行逆 FFT
    printf("\n[主机] 执行逆 FFT ...\n");
    GpuTimer timer_ifft;
    gpu_timer_init(&timer_ifft);
    gpu_timer_start(&timer_ifft);

    CUFFT_SAFE_CALL(cufftExecC2C(plan, d_signal, d_signal, CUFFT_INVERSE));

    gpu_timer_stop(&timer_ifft);
    gpu_timer_print(&timer_ifft, "cuFFT 逆变换 ");

    // 复制结果回主机
    CUDA_SAFE_CALL(cudaMemcpy(h_result, d_signal, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // 验证：逆 FFT 后应该恢复原始信号（需要除以 N）
    printf("\n[主机] 验证 FFT 结果 ...\n");
    float max_error = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float reconstructed = h_result[i].x / (float)n;
        float error = fabsf(reconstructed - h_signal[i].x);
        if (error > max_error) max_error = error;
    }
    printf("  最大重建误差: %.6e\n", max_error);
    printf("  验证结果: %s\n", (max_error < 1e-4f) ? "通过 " : "失败 ");

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. cufftPlan1d: 创建 1D FFT 计划\n");
    printf("  2. cufftExecC2C: 执行复数到复数 FFT\n");
    printf("  3. CUFFT_FORWARD / CUFFT_INVERSE: 正向/逆变换\n");
    printf("  4. 逆变换后需要除以 N 恢复原始幅度\n");
    printf("  5. cuFFT 自动处理最优线程配置\n");
    printf("=============================================================\n\n");

    cufftDestroy(plan);
    cudaFree(d_signal);
    free(h_signal);
    free(h_result);
    gpu_timer_destroy(&timer_fft);
    gpu_timer_destroy(&timer_ifft);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
