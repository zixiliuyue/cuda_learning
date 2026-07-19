/******************************************************************************
 * @file    demo.cu
 * @brief   高频基础 Kernel 库演示程序（Softmax / LayerNorm / GELU / Dropout）
 * @details 本程序独立演示 common/cuda_kernels 下所有算子的调用方式：
 *          1. 用固定 / 随机测试数据构造输入；
 *          2. 分别调用 GPU Kernel 与 CPU 参考实现；
 *          3. 逐元素对比，验证 GPU 结果的正确性；
 *          4. 用 GpuTimer 计时，输出性能总结表格。
 *
 *          所有 Kernel 均以 header-only 方式 #include，无需单独编译。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// ---- common 公共头文件 ----
#include "cuda_utils.h"     // CUDA_SAFE_CALL / CHECK_KERNEL_ERROR / print_device_info
#include "timer.h"          // GpuTimer 计时器
#include "math_utils.h"     // init_random_float / compare_float_arrays 等
#include "platform.h"       // PRINT_PLATFORM_INFO

// ---- 待演示的高频算子（header-only）----
#include "softmax_kernel.cuh"
#include "layer_norm_kernel.cuh"
#include "gelu_kernel.cuh"
#include "dropout_kernel.cuh"

// 60 字符分隔线
#define SEP_LINE "------------------------------------------------------------"

/* ============================================================
 *  CPU 参考实现（用于验证 GPU 结果正确性）
 * ============================================================ */

/**
 * @brief  Softmax 的 CPU 参考实现（逐行，带数值稳定处理）
 */
static void softmax_cpu(const float* in, float* out, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
    {
        const float* x = in + (size_t)r * cols;
        float*       y = out + (size_t)r * cols;

        float m = -FLT_MAX;
        for (int c = 0; c < cols; c++) m = fmaxf(m, x[c]);

        float sum = 0.0f;
        for (int c = 0; c < cols; c++) sum += expf(x[c] - m);

        float inv = 1.0f / sum;
        for (int c = 0; c < cols; c++) y[c] = expf(x[c] - m) * inv;
    }
}

/**
 * @brief  LayerNorm 的 CPU 参考实现（逐行）
 */
static void layer_norm_cpu(const float* in, float* out,
                           const float* gamma, const float* beta,
                           int rows, int cols, float eps)
{
    for (int r = 0; r < rows; r++)
    {
        const float* x = in + (size_t)r * cols;
        float*       y = out + (size_t)r * cols;

        double sum = 0.0, sqsum = 0.0;
        for (int c = 0; c < cols; c++) { sum += x[c]; sqsum += (double)x[c] * x[c]; }

        double mu  = sum / cols;
        double var = sqsum / cols - mu * mu;
        if (var < 0.0) var = 0.0;
        float rstd = (float)(1.0 / sqrt(var + eps));

        for (int c = 0; c < cols; c++)
        {
            float x_hat = ((float)(x[c] - mu)) * rstd;
            float g = gamma ? gamma[c] : 1.0f;
            float b = beta  ? beta[c]  : 0.0f;
            y[c] = g * x_hat + b;
        }
    }
}

/**
 * @brief  GELU tanh 版 CPU 参考实现
 */
static void gelu_tanh_cpu(const float* in, float* out, int n)
{
    const float k = 0.7978845608028654f;
    for (int i = 0; i < n; i++)
    {
        float x = in[i];
        float inner = k * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

/**
 * @brief  GELU sigmoid 版 CPU 参考实现
 */
static void gelu_sigmoid_cpu(const float* in, float* out, int n)
{
    for (int i = 0; i < n; i++)
    {
        float x = in[i];
        out[i] = x / (1.0f + expf(-1.702f * x));
    }
}

/* ============================================================
 *  各算子演示函数
 * ============================================================ */

/**
 * @brief  演示 Softmax，返回 GPU 计算耗时（ms）
 */
static double demo_softmax(int rows, int cols, int* passed)
{
    printf("\n" SEP_LINE "\n");
    printf("  [1] Softmax  (rows=%d, cols=%d，每行独立归一化)\n", rows, cols);
    printf(SEP_LINE "\n");

    int n = rows * cols;
    size_t bytes = (size_t)n * sizeof(float);

    float* h_in  = (float*)malloc(bytes);
    float* h_gpu = (float*)malloc(bytes);
    float* h_ref = (float*)malloc(bytes);
    init_random_float(h_in, n, -3.0f, 3.0f);

    float *d_in = NULL, *d_out = NULL;
    CUDA_SAFE_CALL(cudaMalloc(&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_out, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    GpuTimer t; gpu_timer_init(&t); gpu_timer_start(&t);
    launch_softmax(d_in, d_out, rows, cols);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&t);
    double ms = gpu_timer_get_elapsed_ms(&t);

    CUDA_SAFE_CALL(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    softmax_cpu(h_in, h_ref, rows, cols);

    // 验证：每行之和应约等于 1
    float row0_sum = 0.0f;
    for (int c = 0; c < cols; c++) row0_sum += h_gpu[c];
    printf("  第 0 行 softmax 之和 = %.6f (应 ≈ 1.0)\n", row0_sum);
    *passed = compare_float_arrays(h_gpu, h_ref, n, 1e-4f);

    gpu_timer_destroy(&t);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_gpu); free(h_ref);
    return ms;
}

/**
 * @brief  演示 LayerNorm，返回 GPU 计算耗时（ms）
 */
static double demo_layer_norm(int rows, int cols, int* passed)
{
    printf("\n" SEP_LINE "\n");
    printf("  [2] LayerNorm  (rows=%d, cols=%d，带 gamma/beta)\n", rows, cols);
    printf(SEP_LINE "\n");

    int n = rows * cols;
    size_t bytes = (size_t)n * sizeof(float);
    size_t param_bytes = (size_t)cols * sizeof(float);

    float* h_in    = (float*)malloc(bytes);
    float* h_gpu   = (float*)malloc(bytes);
    float* h_ref   = (float*)malloc(bytes);
    float* h_gamma = (float*)malloc(param_bytes);
    float* h_beta  = (float*)malloc(param_bytes);
    init_random_float(h_in, n, -5.0f, 5.0f);
    init_random_float(h_gamma, cols, 0.5f, 1.5f);
    init_random_float(h_beta, cols, -0.5f, 0.5f);

    float eps = 1e-5f;
    float *d_in = NULL, *d_out = NULL, *d_gamma = NULL, *d_beta = NULL;
    CUDA_SAFE_CALL(cudaMalloc(&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_out, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_gamma, param_bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_beta, param_bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_gamma, h_gamma, param_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_beta, h_beta, param_bytes, cudaMemcpyHostToDevice));

    GpuTimer t; gpu_timer_init(&t); gpu_timer_start(&t);
    launch_layer_norm(d_in, d_out, d_gamma, d_beta, rows, cols, eps);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&t);
    double ms = gpu_timer_get_elapsed_ms(&t);

    CUDA_SAFE_CALL(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    layer_norm_cpu(h_in, h_ref, h_gamma, h_beta, rows, cols, eps);
    *passed = compare_float_arrays(h_gpu, h_ref, n, 1e-3f);

    gpu_timer_destroy(&t);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_in); free(h_gpu); free(h_ref); free(h_gamma); free(h_beta);
    return ms;
}

/**
 * @brief  演示 GELU（两种实现），返回 tanh 版 GPU 耗时（ms）
 */
static double demo_gelu(int n, int* passed)
{
    printf("\n" SEP_LINE "\n");
    printf("  [3] GELU  (n=%d，对比 tanh 精确版 与 sigmoid 近似版)\n", n);
    printf(SEP_LINE "\n");

    size_t bytes = (size_t)n * sizeof(float);
    float* h_in   = (float*)malloc(bytes);
    float* h_gpu  = (float*)malloc(bytes);
    float* h_ref  = (float*)malloc(bytes);
    float* h_ref2 = (float*)malloc(bytes);
    float* h_gpu2 = (float*)malloc(bytes);
    init_random_float(h_in, n, -4.0f, 4.0f);

    float *d_in = NULL, *d_out = NULL;
    CUDA_SAFE_CALL(cudaMalloc(&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_out, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // tanh 精确版
    GpuTimer t; gpu_timer_init(&t); gpu_timer_start(&t);
    launch_gelu(d_in, d_out, n, /*use_tanh=*/true);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&t);
    double ms = gpu_timer_get_elapsed_ms(&t);
    CUDA_SAFE_CALL(cudaMemcpy(h_gpu, d_out, bytes, cudaMemcpyDeviceToHost));
    gelu_tanh_cpu(h_in, h_ref, n);
    printf("  [tanh 精确版] ");
    int p1 = compare_float_arrays(h_gpu, h_ref, n, 1e-4f);

    // sigmoid 近似版
    launch_gelu(d_in, d_out, n, /*use_tanh=*/false);
    CHECK_KERNEL_ERROR();
    CUDA_SAFE_CALL(cudaMemcpy(h_gpu2, d_out, bytes, cudaMemcpyDeviceToHost));
    gelu_sigmoid_cpu(h_in, h_ref2, n);
    printf("  [sigmoid 近似版] ");
    int p2 = compare_float_arrays(h_gpu2, h_ref2, n, 1e-4f);

    // 对比两种实现的最大差异（说明近似版与精确版并不完全相同）
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) max_diff = fmaxf(max_diff, fabsf(h_gpu[i] - h_gpu2[i]));
    printf("  tanh 版 与 sigmoid 版 的最大差异 = %.6f （两者是不同近似，非 bug）\n", max_diff);

    *passed = (p1 && p2);
    gpu_timer_destroy(&t);
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_gpu); free(h_ref); free(h_ref2); free(h_gpu2);
    return ms;
}

/**
 * @brief  演示 Dropout（训练 + 推理），返回训练模式 GPU 耗时（ms）
 */
static double demo_dropout(int n, int* passed)
{
    printf("\n" SEP_LINE "\n");
    printf("  [4] Dropout  (n=%d，训练随机丢弃 vs 推理恒等)\n", n);
    printf(SEP_LINE "\n");

    size_t bytes = (size_t)n * sizeof(float);
    float* h_in    = (float*)malloc(bytes);
    float* h_train = (float*)malloc(bytes);
    float* h_infer = (float*)malloc(bytes);
    float* h_mask  = (float*)malloc(bytes);
    // 输入全设为 1.0，便于直观观察缩放系数
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;

    float drop_prob = 0.3f;
    float *d_in = NULL, *d_out = NULL, *d_mask = NULL;
    CUDA_SAFE_CALL(cudaMalloc(&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_out, bytes));
    CUDA_SAFE_CALL(cudaMalloc(&d_mask, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ---- 训练模式 ----
    GpuTimer t; gpu_timer_init(&t); gpu_timer_start(&t);
    launch_dropout(d_in, d_out, d_mask, n, drop_prob, /*training=*/true, /*seed=*/2026ULL);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&t);
    double ms = gpu_timer_get_elapsed_ms(&t);
    CUDA_SAFE_CALL(cudaMemcpy(h_train, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_mask, d_mask, bytes, cudaMemcpyDeviceToHost));

    // 统计实际丢弃率与保留元素的期望
    int kept = 0;
    for (int i = 0; i < n; i++) if (h_mask[i] > 0.5f) kept++;
    float actual_drop = 1.0f - (float)kept / n;
    float mean_out = array_mean(h_train, n);
    printf("  设定丢弃率 p=%.2f | 实际丢弃率=%.4f | 保留元素值=%.4f (应≈1/(1-p)=%.4f)\n",
           drop_prob, actual_drop, 1.0f / (1.0f - drop_prob), 1.0f / (1.0f - drop_prob));
    printf("  训练输出均值=%.4f (应≈1.0，因缩放补偿使期望不变)\n", mean_out);

    // ---- 推理模式 ----
    launch_dropout(d_in, d_out, NULL, n, drop_prob, /*training=*/false);
    CHECK_KERNEL_ERROR();
    CUDA_SAFE_CALL(cudaMemcpy(h_infer, d_out, bytes, cudaMemcpyDeviceToHost));
    // 推理模式应恒等：输出全 1.0
    int infer_ok = 1;
    for (int i = 0; i < n; i++) if (fabsf(h_infer[i] - 1.0f) > 1e-6f) { infer_ok = 0; break; }
    printf("  推理模式恒等映射验证: %s\n", infer_ok ? "通过(输出==输入)" : "失败");

    // 训练正确性：非零输出必须等于 scale，零输出必须来自 mask=0
    float scale = 1.0f / (1.0f - drop_prob);
    int train_ok = 1;
    for (int i = 0; i < n; i++)
    {
        float expect = (h_mask[i] > 0.5f) ? scale : 0.0f;
        if (fabsf(h_train[i] - expect) > 1e-4f) { train_ok = 0; break; }
    }
    printf("  训练模式掩码/缩放一致性验证: %s\n", train_ok ? "通过" : "失败");

    *passed = (infer_ok && train_ok);
    gpu_timer_destroy(&t);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_mask);
    free(h_in); free(h_train); free(h_infer); free(h_mask);
    return ms;
}

/* ============================================================
 *  主函数
 * ============================================================ */
int main(void)
{
    printf("\n=============================================================\n");
    printf("     高频基础 Kernel 库演示 (Softmax/LayerNorm/GELU/Dropout)  \n");
    printf("=============================================================\n\n");

    PRINT_PLATFORM_INFO();
    print_device_info();

    // 各算子的测试规模
    int sm_rows = 512, sm_cols = 1024;   // Softmax
    int ln_rows = 512, ln_cols = 1024;   // LayerNorm
    int gelu_n  = 1 << 20;               // GELU  (1M 元素)
    int drop_n  = 1 << 20;               // Dropout (1M 元素)

    int p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    double ms1 = demo_softmax(sm_rows, sm_cols, &p1);
    double ms2 = demo_layer_norm(ln_rows, ln_cols, &p2);
    double ms3 = demo_gelu(gelu_n, &p3);
    double ms4 = demo_dropout(drop_n, &p4);

    // ---- 性能总结表格 ----
    printf("\n=============================================================\n");
    printf("                      性能与验证总结                          \n");
    printf("=============================================================\n");
    printf("  %-14s | %-18s | %12s | %-8s\n", "算子", "规模", "GPU耗时(ms)", "验证");
    printf("  " SEP_LINE "\n");
    printf("  %-14s | %-18s | %12.4f | %-8s\n", "Softmax",
           "512 x 1024", ms1, p1 ? "PASS" : "FAIL");
    printf("  %-14s | %-18s | %12.4f | %-8s\n", "LayerNorm",
           "512 x 1024", ms2, p2 ? "PASS" : "FAIL");
    printf("  %-14s | %-18s | %12.4f | %-8s\n", "GELU(tanh)",
           "1048576", ms3, p3 ? "PASS" : "FAIL");
    printf("  %-14s | %-18s | %12.4f | %-8s\n", "Dropout",
           "1048576", ms4, p4 ? "PASS" : "FAIL");
    printf("=============================================================\n");

    int all_pass = p1 && p2 && p3 && p4;
    printf("\n  总体结果: %s\n\n", all_pass ? "全部通过 " : "存在失败项 ");

    return all_pass ? 0 : 1;
}
