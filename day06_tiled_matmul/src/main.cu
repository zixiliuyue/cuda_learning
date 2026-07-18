/******************************************************************************
 * @file    main.cu
 * @brief   Day 6: 共享内存优化 — Tiled Matrix Multiplication
 * @details 本程序实现矩阵乘法 C = A x B，对比三种实现：
 *          1. CPU 参考实现（三重循环）
 *          2. GPU 朴素实现（每个线程计算一个元素，重复读取全局内存）
 *          3. GPU Tiled 实现（使用共享内存缓存子矩阵，减少全局内存访问）
 *
 *          核心学习目标：
 *          1. 共享内存（Shared Memory）的概念和使用方法
 *          2. __shared__ 关键字与内存分配
 *          3. Bank Conflict 原理与避免策略
 *          4. __syncthreads() 线程同步的作用
 *          5. Tiling 技术：分块计算减少全局内存访问
 *          6. 对比朴素实现和 Tiled 实现的性能差异
 *
 *          矩阵乘法回顾：
 *          C[i][j] = sum(A[i][k] * B[k][j]) for k = 0 to N-1
 *          计算复杂度: O(N^3)
 *          内存访问: 每个元素 C[i][j] 需要读取 A 的一行和 B 的一列
 *
 *          Tiling 原理：
 *          - 将大矩阵分成小块（Tile），每块大小为 TILE_WIDTH x TILE_WIDTH
 *          - 将 Tile 加载到共享内存中，Block 内所有线程共享
 *          - 每个线程只需从共享内存读取数据，而不是全局内存
 *          - 全局内存访问次数从 O(N^3) 降低到 O(N^3 / TILE_WIDTH)
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

/* ============================================================
 *  配置参数
 * ============================================================ */

/** @brief 矩阵维度（方阵） */
#define MATRIX_SIZE 1024

/** @brief Tile 宽度（必须 <= 最大共享内存限制） */
#define TILE_WIDTH 16

/** @brief 每个 Block 的线程数（二维） */
#define BLOCK_SIZE 16

/** @brief 浮点数比较容差 */
#define TOLERANCE 1e-3f

/* ============================================================
 *  Kernel 1: 朴素矩阵乘法（无共享内存优化）
 * ============================================================ */

/**
 * @brief  朴素矩阵乘法 Kernel
 * @param  d_a  输入矩阵 A（M x K）
 * @param  d_b  输入矩阵 B（K x N）
 * @param  d_c  输出矩阵 C（M x N）
 * @param  width  矩阵维度（假设三方阵）
 * @return void
 *
 * @details 每个线程计算输出矩阵 C 的一个元素。
 *          对于 C[i][j]，需要读取 A 的第 i 行和 B 的第 j 列。
 *          问题：A 的第 i 行被所有计算同一行的线程重复读取，
 *                B 的第 j 列被所有计算同一列的线程重复读取。
 *          全局内存访问次数：每个元素需要 2*width 次读取
 *          总全局内存读取：2 * width^3（非常庞大！）
 */
__global__ void matmul_naive_kernel(const float* d_a, const float* d_b, float* d_c,
                                     int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < width)
    {
        float sum = 0.0f;

        // 每个线程执行 width 次乘加运算
        // 每次运算需要从全局内存读取 A[row][k] 和 B[k][col]
        for (int k = 0; k < width; k++)
        {
            sum += d_a[row * width + k] * d_b[k * width + col];
        }

        d_c[row * width + col] = sum;
    }
}

/* ============================================================
 *  Kernel 2: Tiled 矩阵乘法（共享内存优化）
 * ============================================================ */

/**
 * @brief  Tiled 矩阵乘法 Kernel（使用共享内存）
 * @param  d_a  输入矩阵 A
 * @param  d_b  输入矩阵 B
 * @param  d_c  输出矩阵 C
 * @param  width  矩阵维度
 * @return void
 *
 * @details 使用 Tiling 技术减少全局内存访问。
 *
 *          共享内存分配：
 *          - ds_a[TILE_WIDTH][TILE_WIDTH]: 缓存 A 的子矩阵
 *          - ds_b[TILE_WIDTH][TILE_WIDTH]: 缓存 B 的子矩阵
 *          共享内存大小：2 * TILE_WIDTH^2 * sizeof(float)
 *          当 TILE_WIDTH=16 时：2 * 256 * 4 = 2048 bytes = 2 KB
 *          远小于通常的 48KB 共享内存限制。
 *
 *          Tiling 算法流程：
 *          1. 将 A 和 B 都分成 TILE_WIDTH x TILE_WIDTH 的小块
 *          2. 对于每个 Tile 位置 m:
 *             a. 将 A 的 Tile (row, m) 加载到 ds_a
 *             b. 将 B 的 Tile (m, col) 加载到 ds_b
 *             c. __syncthreads() 等待所有线程完成加载
 *             d. 使用 ds_a 和 ds_b 中的数据计算部分和
 *             e. __syncthreads() 等待所有线程完成计算（防止覆盖共享内存）
 *          3. 将最终结果写入全局内存
 *
 *          全局内存访问优化效果：
 *          - 朴素实现：每个元素读取 2*width 次全局内存
 *          - Tiled 实现：每个元素读取 2*width/TILE_WIDTH 次全局内存
 *          - 加速比：约 TILE_WIDTH 倍（实际受其他因素影响）
 */
__global__ void matmul_tiled_kernel(const float* d_a, const float* d_b, float* d_c,
                                     int width)
{
    // 声明共享内存数组
    // __shared__ 修饰符表示该变量存储在共享内存中
    // 每个 Block 有一份独立的共享内存副本
    // 该 Block 内的所有线程都可以访问和修改这些数组
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];

    // 计算当前线程在输出矩阵中的位置
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // 局部累加变量（存储在寄存器中）
    float sum = 0.0f;

    // 遍历所有 Tile
    // 对于 width x width 的矩阵，每行/列有 width/TILE_WIDTH 个 Tile
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; m++)
    {
        // 计算当前 Tile 在 A 和 B 中的列/行偏移
        int a_col = m * TILE_WIDTH + threadIdx.x;
        int b_row = m * TILE_WIDTH + threadIdx.y;

        // 协作加载：Block 内所有线程合作将数据从全局内存加载到共享内存
        // 每个线程负责加载 A 和 B 中的一个元素到共享内存
        // 这样每个全局内存元素只被读取一次，而不是被每个线程重复读取

        // 加载 A 的 Tile 元素到 ds_a
        // ds_a[threadIdx.y][threadIdx.x] = A[row][a_col]
        if (row < width && a_col < width)
        {
            ds_a[threadIdx.y][threadIdx.x] = d_a[row * width + a_col];
        }
        else
        {
            ds_a[threadIdx.y][threadIdx.x] = 0.0f;  // 越界填充 0
        }

        // 加载 B 的 Tile 元素到 ds_b
        // ds_b[threadIdx.y][threadIdx.x] = B[b_row][col]
        if (b_row < width && col < width)
        {
            ds_b[threadIdx.y][threadIdx.x] = d_b[b_row * width + col];
        }
        else
        {
            ds_b[threadIdx.y][threadIdx.x] = 0.0f;  // 越界填充 0
        }

        // 线程同步：确保所有线程都已完成共享内存的加载
        // 这是 Tiling 算法的关键步骤！
        // 如果没有同步，某些线程可能还在读取共享内存时，
        // 其他线程已经开始覆盖共享内存中的数据
        __syncthreads();

        // 使用共享内存中的数据计算部分积
        // 每个线程计算 TILE_WIDTH 个元素的乘积和
        for (int k = 0; k < TILE_WIDTH; k++)
        {
            // 从共享内存读取数据（比全局内存快 ~100 倍）
            sum += ds_a[threadIdx.y][k] * ds_b[k][threadIdx.x];
        }

        // 再次同步：确保所有线程都已完成计算
        // 防止某些线程提前进入下一轮循环，覆盖共享内存
        __syncthreads();
    }

    // 将最终结果写入全局内存
    if (row < width && col < width)
    {
        d_c[row * width + col] = sum;
    }
}

/* ============================================================
 *  主机辅助函数
 * ============================================================ */

void matmul_cpu(const float* a, const float* b, float* c, int width)
{
    for (int row = 0; row < width; row++)
    {
        for (int col = 0; col < width; col++)
        {
            float sum = 0.0f;
            for (int k = 0; k < width; k++)
            {
                sum += a[row * width + k] * b[k * width + col];
            }
            c[row * width + col] = sum;
        }
    }
}

/* ============================================================
 *  主函数
 * ============================================================ */

int main(void)
{
    printf("\n");
    printf("=============================================================\n");
    printf("     Day 6: 共享内存优化 — Tiled Matrix Multiplication        \n");
    printf("=============================================================\n\n");

    print_device_info();

    int num_elements = MATRIX_SIZE * MATRIX_SIZE;
    size_t bytes = num_elements * sizeof(float);

    printf("[主机] 矩阵尺寸: %d x %d (%.2f MB x 3)\n",
           MATRIX_SIZE, MATRIX_SIZE, bytes / (1024.0 * 1024.0));
    printf("[主机] Tile 宽度: %d\n", TILE_WIDTH);

    /* --------------------------------------------------------
     *  分配和初始化数据
     * -------------------------------------------------------- */
    printf("\n[主机] 分配主机内存 ...\n");
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c_naive = (float*)malloc(bytes);
    float* h_c_tiled = (float*)malloc(bytes);
    float* h_c_ref = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c_naive || !h_c_tiled || !h_c_ref)
    {
        fprintf(stderr, "[错误] 主机内存分配失败！\n");
        return EXIT_FAILURE;
    }

    printf("[主机] 初始化矩阵数据 ...\n");
    srand((unsigned int)time(NULL));
    init_random_float(h_a, num_elements, -0.5f, 0.5f);
    init_random_float(h_b, num_elements, -0.5f, 0.5f);

    // 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 配置 Grid 和 Block
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("[主机] Grid: (%d, %d), Block: (%d, %d)\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    /* --------------------------------------------------------
     *  CPU 参考计算
     * -------------------------------------------------------- */
    printf("\n[主机] CPU 参考计算（可能需要几秒）...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    matmul_cpu(h_a, h_b, h_c_ref, MATRIX_SIZE);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU 矩阵乘法");

    /* --------------------------------------------------------
     *  GPU 朴素实现
     * -------------------------------------------------------- */
    printf("\n[主机] GPU 朴素实现 ...\n");
    GpuTimer timer_naive;
    gpu_timer_init(&timer_naive);
    gpu_timer_start(&timer_naive);
    matmul_naive_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, MATRIX_SIZE);
    gpu_timer_stop(&timer_naive);
    CHECK_KERNEL_ERROR();
    gpu_timer_print(&timer_naive, "GPU 朴素实现");

    CUDA_SAFE_CALL(cudaMemcpy(h_c_naive, d_c, bytes, cudaMemcpyDeviceToHost));
    int correct_naive = compare_float_arrays(h_c_naive, h_c_ref, num_elements, TOLERANCE);

    /* --------------------------------------------------------
     *  GPU Tiled 实现
     * -------------------------------------------------------- */
    printf("\n[主机] GPU Tiled 实现（共享内存优化）...\n");
    GpuTimer timer_tiled;
    gpu_timer_init(&timer_tiled);
    gpu_timer_start(&timer_tiled);
    matmul_tiled_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, MATRIX_SIZE);
    gpu_timer_stop(&timer_tiled);
    CHECK_KERNEL_ERROR();
    gpu_timer_print(&timer_tiled, "GPU Tiled 实现");

    CUDA_SAFE_CALL(cudaMemcpy(h_c_tiled, d_c, bytes, cudaMemcpyDeviceToHost));
    int correct_tiled = compare_float_arrays(h_c_tiled, h_c_ref, num_elements, TOLERANCE);

    /* --------------------------------------------------------
     *  性能总结
     * -------------------------------------------------------- */
    double cpu_ms = gpu_timer_get_elapsed_ms(&timer_cpu);
    double naive_ms = gpu_timer_get_elapsed_ms(&timer_naive);
    double tiled_ms = gpu_timer_get_elapsed_ms(&timer_tiled);

    printf("\n");
    printf("=============================================================\n");
    printf("                        性能总结                              \n");
    printf("=============================================================\n");
    printf("  矩阵尺寸        : %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("  CPU 时间        : %.2f ms\n", cpu_ms);
    printf("  GPU 朴素实现    : %.2f ms (加速比: %.1fx)\n", naive_ms, cpu_ms / naive_ms);
    printf("  GPU Tiled 实现  : %.2f ms (加速比: %.1fx)\n", tiled_ms, cpu_ms / tiled_ms);
    printf("  Tiled vs 朴素   : %.2fx 更快\n", naive_ms / tiled_ms);
    printf("  朴素结果验证    : %s\n", correct_naive ? "通过" : "失败");
    printf("  Tiled 结果验证  : %s\n", correct_tiled ? "通过" : "失败");
    printf("=============================================================\n");

    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. __shared__: 声明共享内存变量\n");
    printf("  2. Tiling: 将大矩阵分块，减少全局内存访问\n");
    printf("  3. __syncthreads(): Block 内线程同步，关键！\n");
    printf("  4. Bank Conflict: 避免同一 Warp 线程访问同一 bank\n");
    printf("  5. 协作加载: Block 内线程合作加载数据到共享内存\n");
    printf("  6. 共享内存速度比全局内存快约 100 倍\n");
    printf("=============================================================\n");
    printf("\n");

    // 释放资源
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c_naive); free(h_c_tiled); free(h_c_ref);
    gpu_timer_destroy(&timer_cpu);
    gpu_timer_destroy(&timer_naive);
    gpu_timer_destroy(&timer_tiled);

    printf("[主机] 资源已释放，程序正常退出。\n\n");
    return (correct_naive && correct_tiled) ? 0 : 1;
}
