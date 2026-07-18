/******************************************************************************
 * @file    main.cu
 * @brief   Day 24: TensorRT 推理优化概念
 * @details 完整 TensorRT 需要 TensorRT SDK。
 *          本项目展示推理引擎的核心算子实现，对应 TensorRT 的底层原理。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define BATCH_SIZE 32
#define INPUT_SIZE 784   // 28x28
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10

/**
 * @brief  全连接层 + ReLU - 对应 TensorRT 中的 FC 层
 */
__global__ void fc_relu_kernel(const float* input, const float* weights,
                                const float* bias, float* output,
                                int in_size, int out_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < out_size)
    {
        float sum = bias[tid];
        for (int i = 0; i < in_size; i++)
        {
            sum += input[i] * weights[i * out_size + tid];
        }
        output[tid] = fmaxf(sum, 0.0f);
    }
}

/**
 * @brief  Softmax 输出层
 */
__global__ void softmax_output_kernel(const float* input, float* output, int size)
{
    // 简化版：假设单线程处理
    float max_val = input[0];
    for (int i = 1; i < size; i++)
    {
        if (input[i] > max_val) max_val = input[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; i++)
    {
        output[i] /= sum;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 24: TensorRT 推理优化概念                            \n");
    printf("=============================================================\n");
    printf("\n");
    printf("  [说明] 完整 TensorRT 需要 NVIDIA TensorRT SDK。\n");
    printf("  [说明] 本项目展示推理引擎的核心算子 CUDA 实现。\n");
    printf("\n");

    print_device_info();

    printf("[主机] 网络结构: FC(%d->%d) -> ReLU -> FC(%d->%d) -> Softmax\n",
           INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // 分配数据
    float* h_input = (float*)malloc(INPUT_SIZE * sizeof(float));
    float* h_w1 = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float* h_b1 = (float*)malloc(HIDDEN_SIZE * sizeof(float));
    float* h_w2 = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float* h_b2 = (float*)malloc(OUTPUT_SIZE * sizeof(float));

    srand((unsigned int)time(NULL));
    for (int i = 0; i < INPUT_SIZE; i++) h_input[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) h_w1[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < HIDDEN_SIZE; i++) h_b1[i] = 0.0f;
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) h_w2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < OUTPUT_SIZE; i++) h_b2[i] = 0.0f;

    float *d_input, *d_w1, *d_b1, *d_hidden, *d_w2, *d_b2, *d_output;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(float)));

    CUDA_SAFE_CALL(cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_w1, h_w1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_w2, h_w2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    printf("\n[主机] 推理执行 ...\n");
    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);

    // Layer 1: FC + ReLU
    fc_relu_kernel<<<(HIDDEN_SIZE + 255) / 256, 256>>>(
        d_input, d_w1, d_b1, d_hidden, INPUT_SIZE, HIDDEN_SIZE);

    // Layer 2: FC
    fc_relu_kernel<<<(OUTPUT_SIZE + 255) / 256, 256>>>(
        d_hidden, d_w2, d_b2, d_output, HIDDEN_SIZE, OUTPUT_SIZE);

    // Softmax
    softmax_output_kernel<<<1, 1>>>(d_output, d_output, OUTPUT_SIZE);

    cudaDeviceSynchronize();
    gpu_timer_stop(&timer);
    gpu_timer_print(&timer, "推理执行 ");

    float* h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));
    CUDA_SAFE_CALL(cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  输出概率: ");
    for (int i = 0; i < OUTPUT_SIZE; i++) printf("%.4f ", h_output[i]);
    printf("\n");

    printf("\n=============================================================\n");
    printf("     TensorRT 优化技术                                        \n");
    printf("=============================================================\n");
    printf("  1. 层融合: 合并 Conv+BN+ReLU 等层\n");
    printf("  2. 精度校准: FP32 -> FP16 -> INT8\n");
    printf("  3. Kernel 自动调优: 选择最优实现\n");
    printf("  4. 动态张量内存: 减少内存分配\n");
    printf("  5. 多流执行: 并行处理多个请求\n");
    printf("=============================================================\n\n");

    cudaFree(d_input); cudaFree(d_w1); cudaFree(d_b1);
    cudaFree(d_hidden); cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_output);
    free(h_input); free(h_w1); free(h_b1); free(h_w2); free(h_b2); free(h_output);
    gpu_timer_destroy(&timer);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
