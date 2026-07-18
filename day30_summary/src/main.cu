/******************************************************************************
 * @file    main.cu
 * @brief   Day 30: 30天 CUDA 学习总结
 * @details 运行一个综合性能基准测试，回顾所有学过的概念。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "platform.h"

#define TEST_SIZE (1024 * 1024)
#define BLOCK_SIZE 256

__global__ void final_benchmark_kernel(const float* a, const float* b, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = a[tid] * b[tid] + a[tid];
    }
}

int main(void)
{
    printf("\n");
    printf("=============================================================\n");
    printf("                                                             \n");
    printf("     CUDA 30天学习项目 - 总结与回顾                         \n");
    printf("                                                             \n");
    printf("=============================================================\n\n");

    PRINT_PLATFORM_INFO();
    printf("\n");
    print_device_info();

    printf("\n=============================================================\n");
    printf("     30天学习路线图回顾                                       \n");
    printf("=============================================================\n\n");

    printf("  【第一阶段】CUDA 基础与核心概念 (Day 1-5)\n");
    printf("    Day 01: 环境搭建与第一个 CUDA 程序\n");
    printf("    Day 02: 向量加法 - 线程层次与执行模型\n");
    printf("    Day 03: 矩阵加法 - 内存模型与数据传输\n");
    printf("    Day 04: 矩阵转置 - 错误处理与调试\n");
    printf("    Day 05: GPU 设备查询与性能基准\n\n");

    printf("  【第二阶段】内存优化与高级特性 (Day 6-12)\n");
    printf("    Day 06: 共享内存优化 - Tiled 矩阵乘法\n");
    printf("    Day 07: 全局内存合并访问\n");
    printf("    Day 08: 寄存器与占用率优化\n");
    printf("    Day 09: 原子操作与并行归约\n");
    printf("    Day 10: CUDA 流与异步执行\n");
    printf("    Day 11: 常量内存与纹理内存\n");
    printf("    Day 12: 动态并行与 CUDA Graphs\n\n");

    printf("  【第三阶段】算法实战与优化技巧 (Day 13-20)\n");
    printf("    Day 13: cuBLAS 矩阵运算库\n");
    printf("    Day 14: cuFFT 快速傅里叶变换\n");
    printf("    Day 15: GPU 图像滤波\n");
    printf("    Day 16: 并行排序\n");
    printf("    Day 17: 并行前缀和 (Scan)\n");
    printf("    Day 18: GPU 图算法 BFS\n");
    printf("    Day 19: 蒙特卡洛方法\n");
    printf("    Day 20: 性能分析与调优综合\n\n");

    printf("  【第四阶段】主流框架集成 (Day 21-26)\n");
    printf("    Day 21: PyTorch CUDA 扩展\n");
    printf("    Day 22: Numba CUDA 编程\n");
    printf("    Day 23: OpenCV CUDA 模块\n");
    printf("    Day 24: TensorRT 推理优化\n");
    printf("    Day 25: 多 GPU 编程\n");
    printf("    Day 26: 混合精度训练\n\n");

    printf("  【第五阶段】综合实战 (Day 27-30)\n");
    printf("    Day 27: N-Body 物理模拟\n");
    printf("    Day 28: GPU 光线追踪\n");
    printf("    Day 29: 简易推理引擎\n");
    printf("    Day 30: 学习总结与进阶路线图\n\n");

    /* ---------------- 综合基准测试 ---------------- */
    printf("=============================================================\n");
    printf("     综合基准测试                                             \n");
    printf("=============================================================\n\n");

    int n = TEST_SIZE;
    size_t bytes = n * sizeof(float);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++)
    {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }

    float *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);
    final_benchmark_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer);
    gpu_timer_print(&timer, "综合基准 (FMA)");

    double elapsed_ms = gpu_timer_get_elapsed_ms(&timer);
    double bandwidth = (3.0 * bytes / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);
    printf("  有效带宽: %.2f GB/s\n", bandwidth);

    /* ---------------- 进阶路线图 ---------------- */
    printf("\n=============================================================\n");
    printf("     进阶学习路线图                                           \n");
    printf("=============================================================\n\n");

    printf("  1. CUDA 底层优化\n");
    printf("     - PTX/SASS 汇编优化\n");
    printf("     - CUTLASS 模板库\n");
    printf("     - warp-level 原语深入\n\n");

    printf("  2. 深度学习加速\n");
    printf("     - cuDNN / cuTensor\n");
    printf("     - Triton 编译器\n");
    printf("     - FlashAttention 实现\n\n");

    printf("  3. 高级话题\n");
    printf("     - GPU 驱动开发\n");
    printf("     - RDMA / GPUDirect\n");
    printf("     - MIG (Multi-Instance GPU)\n\n");

    printf("=============================================================\n");
    printf("                                                             \n");
    printf("     恭喜完成 CUDA 30天学习项目！                             \n");
    printf("                                                             \n");
    printf("=============================================================\n\n");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b);
    gpu_timer_destroy(&timer);

    return 0;
}
