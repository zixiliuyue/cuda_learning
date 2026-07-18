/******************************************************************************
 * @file    main.cu
 * @brief   Day 21: PyTorch CUDA 扩展概念演示
 * @details 由于完整 PyTorch C++ 扩展需要 PyTorch 头文件和库，
 *          本项目展示核心 CUDA Kernel 逻辑，可作为 PyTorch Op 的实现参考。
 *          实际集成请参考 python/ 目录中的 setup.py 和 PyTorch 官方文档。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

#define ARRAY_SIZE (1024 * 1024 * 4)
#define BLOCK_SIZE 256
#define TOLERANCE 1e-4f

/**
 * @brief  ReLU Activation Kernel（PyTorch 中 torch.nn.ReLU 的 CUDA 实现）
 * @details ReLU(x) = max(0, x)
 *          这是深度学习中最常用的激活函数之一。
 */
__global__ void relu_kernel(const float* input, float* output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float val = input[tid];
        output[tid] = (val > 0.0f) ? val : 0.0f;
    }
}

/**
 * @brief  Softmax Kernel（简化版，仅展示概念）
 * @details Softmax(x_i) = exp(x_i) / sum(exp(x_j))
 */
__global__ void softmax_kernel(const float* input, float* output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // 简化为元素级 exp（完整实现需要全局归约）
        output[tid] = expf(input[tid]);
    }
}

void relu_cpu(const float* input, float* output, int n)
{
    for (int i = 0; i < n; i++)
    {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 21: PyTorch CUDA 扩展概念                            \n");
    printf("=============================================================\n");
    printf("\n");
    printf("  [说明] 完整 PyTorch C++ 扩展需要 PyTorch 开发环境。\n");
    printf("  [说明] 本项目展示可作为 PyTorch Op 的 CUDA Kernel 逻辑。\n");
    printf("  [说明] 实际集成请参考 python/ 目录。\n");
    printf("\n");

    print_device_info();

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);
    printf("[主机] 数组大小: %d 个元素\n", n);

    float* h_input = (float*)malloc(bytes);
    float* h_output_gpu = (float*)malloc(bytes);
    float* h_output_cpu = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    init_random_float(h_input, n, -2.0f, 2.0f);

    float *d_input, *d_output;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_input, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_output, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // GPU ReLU
    printf("\n[主机] GPU ReLU Kernel ...\n");
    GpuTimer timer_gpu;
    gpu_timer_init(&timer_gpu);
    gpu_timer_start(&timer_gpu);
    relu_kernel<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_gpu);
    gpu_timer_print(&timer_gpu, "GPU ReLU");
    CUDA_SAFE_CALL(cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost));

    // CPU ReLU
    printf("\n[主机] CPU ReLU ...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    relu_cpu(h_input, h_output_cpu, n);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU ReLU");

    // 验证
    int correct = compare_float_arrays(h_output_gpu, h_output_cpu, n, TOLERANCE);

    printf("\n=============================================================\n");
    printf("     PyTorch CUDA 扩展集成要点                                \n");
    printf("=============================================================\n");
    printf("  1. 使用 torch.utils.cpp_extension 编译 CUDA 扩展\n");
    printf("  2. 使用 ATen 张量库处理 PyTorch Tensor\n");
    printf("  3. 注册自定义 Op: TORCH_LIBRARY\n");
    printf("  4. 支持自动求导: 实现 backward Kernel\n");
    printf("  5. 使用 torch.cuda.amp 支持混合精度\n");
    printf("=============================================================\n\n");

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output_gpu); free(h_output_cpu);
    gpu_timer_destroy(&timer_gpu);
    gpu_timer_destroy(&timer_cpu);

    printf("[主机] 程序正常退出。\n\n");
    return correct ? 0 : 1;
}
