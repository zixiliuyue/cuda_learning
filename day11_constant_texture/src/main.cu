/******************************************************************************
 * @file    main.cu
 * @brief   Day 11: 常量内存与纹理内存
 * @details 本程序演示常量内存和纹理内存的使用场景。
 *
 *          核心学习目标：
 *          1. 常量内存（Constant Memory）的缓存机制
 *          2. __constant__ 关键字与 cudaMemcpyToSymbol
 *          3. 纹理内存（Texture Memory）的硬件插值与缓存
 *          4. 只读数据的缓存优化
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define LUT_SIZE 256
#define ARRAY_SIZE (1024 * 1024 * 4)
#define BLOCK_SIZE 256

// 常量内存声明（全局作用域）
// 常量内存大小限制：64 KB
// 所有线程读取相同地址时，广播给所有线程，效率极高
__constant__ float c_lut[LUT_SIZE];

/**
 * @brief  使用常量内存的 LUT 查找 Kernel
 */
__global__ void lut_constant_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        int idx = (int)(d_in[tid] * (LUT_SIZE - 1)) & (LUT_SIZE - 1);
        // 从常量内存读取：所有线程可能读取相同的 lut 元素
        // 常量内存在这种情况下通过广播机制效率极高
        d_out[tid] = c_lut[idx];
    }
}

/**
 * @brief  使用全局内存的 LUT 查找 Kernel（对比）
 */
__global__ void lut_global_kernel(const float* d_in, const float* d_lut, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        int idx = (int)(d_in[tid] * (LUT_SIZE - 1)) & (LUT_SIZE - 1);
        d_out[tid] = d_lut[idx];
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 11: 常量内存与纹理内存                               \n");
    printf("=============================================================\n\n");

    print_device_info();

    // 初始化 LUT 数据
    float h_lut[LUT_SIZE];
    for (int i = 0; i < LUT_SIZE; i++)
    {
        h_lut[i] = (float)i / (float)LUT_SIZE;
    }

    // 初始化输入数据
    float* h_in = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float* h_out = (float*)malloc(ARRAY_SIZE * sizeof(float));
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        h_in[i] = (float)(i % LUT_SIZE) / (float)LUT_SIZE;
    }

    float *d_in, *d_out, *d_lut;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_in, ARRAY_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, ARRAY_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_lut, LUT_SIZE * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_lut, h_lut, LUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // 将 LUT 复制到常量内存
    // cudaMemcpyToSymbol: 将主机数据复制到设备的常量内存
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_lut, h_lut, LUT_SIZE * sizeof(float)));

    int grid_size = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 测试全局内存版本
    printf("\n[主机] 全局内存 LUT 查找 ...\n");
    GpuTimer timer_global;
    gpu_timer_init(&timer_global);
    gpu_timer_start(&timer_global);
    lut_global_kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_lut, d_out, ARRAY_SIZE);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_global);
    gpu_timer_print(&timer_global, "全局内存 LUT");

    // 测试常量内存版本
    printf("\n[主机] 常量内存 LUT 查找 ...\n");
    GpuTimer timer_const;
    gpu_timer_init(&timer_const);
    gpu_timer_start(&timer_const);
    lut_constant_kernel<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, ARRAY_SIZE);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_const);
    gpu_timer_print(&timer_const, "常量内存 LUT");

    double global_ms = gpu_timer_get_elapsed_ms(&timer_global);
    double const_ms = gpu_timer_get_elapsed_ms(&timer_const);

    printf("\n=============================================================\n");
    printf("                        性能总结                              \n");
    printf("=============================================================\n");
    printf("  全局内存 LUT : %.3f ms\n", global_ms);
    printf("  常量内存 LUT : %.3f ms\n", const_ms);
    printf("  加速比       : %.2fx\n", global_ms / const_ms);
    printf("=============================================================\n\n");

    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. __constant__: 声明常量内存变量\n");
    printf("  2. cudaMemcpyToSymbol: 复制数据到常量内存\n");
    printf("  3. 常量内存优点: 广播机制，所有线程读同一地址极快\n");
    printf("  4. 常量内存限制: 64KB，只读\n");
    printf("  5. 纹理内存: 适合2D空间局部性访问，带硬件缓存\n");
    printf("=============================================================\n\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_lut);
    free(h_in); free(h_out);
    gpu_timer_destroy(&timer_global);
    gpu_timer_destroy(&timer_const);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
