/******************************************************************************
 * @file    main.cu
 * @brief   Day 13: cuBLAS - NVIDIA 官方线性代数库
 * @details 演示使用 cuBLAS 进行矩阵乘法，并与手写 CUDA Kernel 对比性能。
 *
 *          cuBLAS 是 NVIDIA 高度优化的 BLAS 库，支持：
 *          - 向量运算（Level 1）
 *          - 矩阵-向量运算（Level 2）
 *          - 矩阵-矩阵运算（Level 3，如 SGEMM）
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

#define MATRIX_SIZE 2048
#define TOLERANCE 1e-2f

/**
 * @brief  检查 cuBLAS 错误
 */
#define CUBLAS_SAFE_CALL(call)                                              \
    do {                                                                    \
        cublasStatus_t status = (call);                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            fprintf(stderr, "[cuBLAS 错误] 状态码: %d | 文件: %s | 行: %d\n", \
                    (int)status, __FILE__, __LINE__);                       \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/**
 * @brief  手写 CUDA 矩阵乘法 Kernel（Tiled 版本）
 */
#define TILE_WIDTH 16
__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n)
{
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    float sum = 0.0f;
    for (int m = 0; m < (n + TILE_WIDTH - 1) / TILE_WIDTH; m++)
    {
        int a_col = m * TILE_WIDTH + threadIdx.x;
        int b_row = m * TILE_WIDTH + threadIdx.y;
        ds_a[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        ds_b[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) sum += ds_a[threadIdx.y][k] * ds_b[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = sum;
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 13: cuBLAS - NVIDIA 官方线性代数库                 \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = MATRIX_SIZE;
    size_t bytes = n * n * sizeof(float);
    printf("[主机] 矩阵尺寸: %d x %d (%.2f MB x 3)\n", n, n, bytes / (1024.0 * 1024.0));

    // 分配主机内存
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c_cuda = (float*)malloc(bytes);
    float* h_c_cublas = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    init_random_float(h_a, n * n, -0.5f, 0.5f);
    init_random_float(h_b, n * n, -0.5f, 0.5f);

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    CUBLAS_SAFE_CALL(cublasCreate(&handle));

    // cuBLAS SGEMM 参数
    // C = alpha * A * B + beta * C
    float alpha = 1.0f;
    float beta = 0.0f;

    // cuBLAS 使用列优先（Column-Major）存储
    // 需要对参数进行调整以适应行优先（Row-Major）数据
    // C^T = B^T * A^T -> 在列优先视角下就是 C = A * B

    printf("\n[主机] cuBLAS SGEMM 执行 ...\n");
    GpuTimer timer_cublas;
    gpu_timer_init(&timer_cublas);
    gpu_timer_start(&timer_cublas);

    CUBLAS_SAFE_CALL(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置
        n, n, n,                   // M, N, K
        &alpha,
        d_b, n,                    // B 矩阵（由于列优先调整）
        d_a, n,                    // A 矩阵
        &beta,
        d_c, n                     // C 矩阵
    ));

    gpu_timer_stop(&timer_cublas);
    gpu_timer_print(&timer_cublas, "cuBLAS SGEMM");
    CUDA_SAFE_CALL(cudaMemcpy(h_c_cublas, d_c, bytes, cudaMemcpyDeviceToHost));

    // 手写 CUDA Kernel
    printf("\n[主机] 手写 CUDA Kernel 执行 ...\n");
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

    GpuTimer timer_cuda;
    gpu_timer_init(&timer_cuda);
    gpu_timer_start(&timer_cuda);
    matmul_tiled_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, n);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_cuda);
    gpu_timer_print(&timer_cuda, "手写 CUDA Kernel");
    CUDA_SAFE_CALL(cudaMemcpy(h_c_cuda, d_c, bytes, cudaMemcpyDeviceToHost));

    // 对比结果
    int is_match = compare_float_arrays(h_c_cublas, h_c_cuda, n * n, TOLERANCE);

    printf("\n=============================================================\n");
    printf("                        性能总结                              \n");
    printf("=============================================================\n");
    printf("  cuBLAS SGEMM     : %.2f ms\n", gpu_timer_get_elapsed_ms(&timer_cublas));
    printf("  手写 CUDA Kernel : %.2f ms\n", gpu_timer_get_elapsed_ms(&timer_cuda));
    printf("  结果一致性       : %s\n", is_match ? "通过 " : "失败 ");
    printf("=============================================================\n\n");

    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c_cuda); free(h_c_cublas);
    gpu_timer_destroy(&timer_cublas);
    gpu_timer_destroy(&timer_cuda);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
