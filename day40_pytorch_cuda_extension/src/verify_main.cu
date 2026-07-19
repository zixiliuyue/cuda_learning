/******************************************************************************
 * @file    verify_main.cu
 * @brief   Day 40: 纯 CUDA 验证程序（不依赖 PyTorch）
 * @details 本文件用于「脱离 PyTorch」独立验证四个 Kernel 的逻辑正确性：
 *              - ReLU 前向 / 反向
 *              - LayerNorm 前向 / 反向（对 input 的梯度）
 *          方法：在 GPU 上跑 Kernel，用 CPU 参考实现对比结果。
 *
 *          实现方式：直接 #include 四个 Kernel 源文件。由于未定义 WITH_TORCH 宏，
 *          那些文件里的 PyTorch 绑定层（#ifdef WITH_TORCH 段）不会被编译，
 *          因此本程序完全不需要 PyTorch 头文件，用 nvcc/CMake 即可构建。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "math_utils.h"

// 直接引入 Kernel 定义（WITH_TORCH 未定义，仅保留裸 Kernel）
#include "relu_forward.cu"
#include "relu_backward.cu"
#include "layer_norm_forward.cu"
#include "layer_norm_backward.cu"

#define N_ELEM   (1024 * 1024)   // ReLU 测试元素数
#define LN_ROWS  512             // LayerNorm 行数
#define LN_DIM   256             // LayerNorm 归一化维度
#define TOL      1e-4f

// ---------------------- CPU 参考实现 ----------------------
static void relu_forward_cpu(const float* in, float* out, int n)
{
    for (int i = 0; i < n; ++i) out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}

static void relu_backward_cpu(const float* go, const float* in, float* gi, int n)
{
    for (int i = 0; i < n; ++i) gi[i] = (in[i] > 0.0f) ? go[i] : 0.0f;
}

static void layer_norm_forward_cpu(const float* in, const float* gamma,
                                   const float* beta, float* out,
                                   int rows, int D, float eps)
{
    for (int r = 0; r < rows; ++r)
    {
        const float* x = in + (size_t)r * D;
        float* y = out + (size_t)r * D;
        float sum = 0.0f, sqr = 0.0f;
        for (int i = 0; i < D; ++i) { sum += x[i]; sqr += x[i] * x[i]; }
        float m = sum / D;
        float var = sqr / D - m * m;
        float rs = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < D; ++i) y[i] = (x[i] - m) * rs * gamma[i] + beta[i];
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 40: 自定义 CUDA 算子逻辑验证（不依赖 PyTorch）      \n");
    printf("=============================================================\n\n");

    print_device_info();
    srand((unsigned int)time(NULL));

    int all_ok = 1;

    /* ================= 1) ReLU 前向 ================= */
    {
        int n = N_ELEM;
        size_t bytes = n * sizeof(float);
        float* h_in  = (float*)malloc(bytes);
        float* h_gpu = (float*)malloc(bytes);
        float* h_cpu = (float*)malloc(bytes);
        init_random_float(h_in, n, -2.0f, 2.0f);

        float *d_in, *d_out;
        CUDA_SAFE_CALL(cudaMalloc(&d_in, bytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_out, bytes));
        CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

        int grid = (n + RELU_BLOCK_SIZE - 1) / RELU_BLOCK_SIZE;
        relu_forward_kernel<<<grid, RELU_BLOCK_SIZE>>>(d_in, d_out, n);
        CHECK_KERNEL_ERROR();
        CUDA_SAFE_CALL(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

        relu_forward_cpu(h_in, h_cpu, n);
        int ok = compare_float_arrays(h_gpu, h_cpu, n, TOL);
        printf("[验证] ReLU 前向       : %s\n", ok ? "通过" : "失败");
        all_ok &= ok;

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_gpu); free(h_cpu);
    }

    /* ================= 2) ReLU 反向 ================= */
    {
        int n = N_ELEM;
        size_t bytes = n * sizeof(float);
        float* h_in  = (float*)malloc(bytes);
        float* h_go  = (float*)malloc(bytes);
        float* h_gpu = (float*)malloc(bytes);
        float* h_cpu = (float*)malloc(bytes);
        init_random_float(h_in, n, -2.0f, 2.0f);
        init_random_float(h_go, n, -1.0f, 1.0f);

        float *d_in, *d_go, *d_gi;
        CUDA_SAFE_CALL(cudaMalloc(&d_in, bytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_go, bytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_gi, bytes));
        CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_go, h_go, bytes, cudaMemcpyHostToDevice));

        int grid = (n + RELU_BLOCK_SIZE - 1) / RELU_BLOCK_SIZE;
        relu_backward_kernel<<<grid, RELU_BLOCK_SIZE>>>(d_go, d_in, d_gi, n);
        CHECK_KERNEL_ERROR();
        CUDA_SAFE_CALL(cudaMemcpy(h_gpu, d_gi, bytes, cudaMemcpyDeviceToHost));

        relu_backward_cpu(h_go, h_in, h_cpu, n);
        int ok = compare_float_arrays(h_gpu, h_cpu, n, TOL);
        printf("[验证] ReLU 反向       : %s\n", ok ? "通过" : "失败");
        all_ok &= ok;

        cudaFree(d_in); cudaFree(d_go); cudaFree(d_gi);
        free(h_in); free(h_go); free(h_gpu); free(h_cpu);
    }

    /* ================= 3) LayerNorm 前向 ================= */
    {
        int rows = LN_ROWS, D = LN_DIM;
        int n = rows * D;
        size_t bytes = n * sizeof(float);
        size_t dbytes = D * sizeof(float);
        size_t rbytes = rows * sizeof(float);

        float* h_in    = (float*)malloc(bytes);
        float* h_gamma = (float*)malloc(dbytes);
        float* h_beta  = (float*)malloc(dbytes);
        float* h_gpu   = (float*)malloc(bytes);
        float* h_cpu   = (float*)malloc(bytes);
        init_random_float(h_in, n, -3.0f, 3.0f);
        init_random_float(h_gamma, D, 0.5f, 1.5f);
        init_random_float(h_beta, D, -0.5f, 0.5f);

        float *d_in, *d_gamma, *d_beta, *d_out, *d_mean, *d_rstd;
        CUDA_SAFE_CALL(cudaMalloc(&d_in, bytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_gamma, dbytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_beta, dbytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_out, bytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_mean, rbytes));
        CUDA_SAFE_CALL(cudaMalloc(&d_rstd, rbytes));
        CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_gamma, h_gamma, dbytes, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_beta, h_beta, dbytes, cudaMemcpyHostToDevice));

        layer_norm_forward_kernel<<<rows, LN_BLOCK_SIZE>>>(
            d_in, d_gamma, d_beta, d_out, d_mean, d_rstd, rows, D, 1e-5f);
        CHECK_KERNEL_ERROR();
        CUDA_SAFE_CALL(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

        layer_norm_forward_cpu(h_in, h_gamma, h_beta, h_cpu, rows, D, 1e-5f);
        int ok = compare_float_arrays(h_gpu, h_cpu, n, 1e-3f);
        printf("[验证] LayerNorm 前向  : %s\n", ok ? "通过" : "失败");
        all_ok &= ok;

        cudaFree(d_in); cudaFree(d_gamma); cudaFree(d_beta);
        cudaFree(d_out); cudaFree(d_mean); cudaFree(d_rstd);
        free(h_in); free(h_gamma); free(h_beta); free(h_gpu); free(h_cpu);
    }

    printf("\n=============================================================\n");
    printf("  总体结果: %s\n", all_ok ? "全部通过" : "存在失败");
    printf("  提示: PyTorch 扩展的完整调用与自动求导请运行 python/demo.py\n");
    printf("=============================================================\n\n");

    return all_ok ? 0 : 1;
}
