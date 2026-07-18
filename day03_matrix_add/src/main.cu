/******************************************************************************
 * @file    main.cu
 * @brief   Day 3: 矩阵加法 — 理解 CUDA 内存模型与数据传输
 * @details 本程序实现两个二维矩阵的逐元素相加: C[row][col] = A[row][col] + B[row][col]
 *
 *          核心学习目标：
 *          1. 二维线程组织的实际应用 — 每个线程处理矩阵的一个元素
 * *          2. 二维索引的正确计算:
 *             row = blockIdx.y * blockDim.y + threadIdx.y
 *             col = blockIdx.x * blockDim.x + threadIdx.x
 *          3. 行优先（Row-Major）内存布局的理解
 *          4. 主机↔设备内存传输的完整流程
 *          5. 统一内存（Unified Memory / cudaMallocManaged）的初步使用
 *          6. 二维数据的边界检查
 *
 *          内存模型关键概念：
 *          - 主机内存（Host Memory）: CPU 可访问的 RAM，通过 malloc/calloc 分配
 *          - 设备内存（Device Memory）: GPU 显存，通过 cudaMalloc 分配
 *          - 数据传输: cudaMemcpy(H2D/D2H) 是性能瓶颈之一
 *          - 统一内存: cudaMallocManaged 让 CPU 和 GPU 共享同一块内存
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

/** @brief 矩阵行数 */
#define MATRIX_ROWS 1024

/** @brief 矩阵列数 */
#define MATRIX_COLS 1024

/** @brief 二维 Block 的 X 维度（每行 16 个线程） */
#define BLOCK_SIZE_X 16

/** @brief 二维 Block 的 Y 维度（每列 16 个线程） */
#define BLOCK_SIZE_Y 16

/** @brief 浮点数比较容差 */
#define TOLERANCE 1e-5f

/* ============================================================
 *  Kernel 函数（在 GPU 上执行）
 * ============================================================ */

/**
 * @brief  矩阵加法 Kernel（显式内存管理版本）
 * @param  d_a  输入矩阵 A（设备内存，行优先展平）
 * @param  d_b  输入矩阵 B（设备内存，行优先展平）
 * @param  d_c  输出矩阵 C（设备内存，行优先展平）
 * @param  width   矩阵列数
 * @param  height  矩阵行数
 * @return void
 *
 * @details 使用二维 Grid 和二维 Block 组织线程。
 *          每个线程负责计算输出矩阵中的一个元素 C[row][col]。
 *
 *          二维索引计算：
 *          - col = blockIdx.x * blockDim.x + threadIdx.x  (列索引)
 *          - row = blockIdx.y * blockDim.y + threadIdx.y  (行索引)
 *
 *          行优先展平访问：
 *          - 二维矩阵在内存中按行优先顺序存储为一维数组
 *          - 元素 (row, col) 的一维索引 = row * width + col
 *          - 这是 C/C++ 数组的默认布局方式
 */
__global__ void matrix_add_kernel(const float* d_a, const float* d_b, float* d_c,
                                   int width, int height)
{
    // 计算当前线程对应的矩阵列索引（x 方向）
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算当前线程对应的矩阵行索引（y 方向）
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 二维边界检查：确保线程不会访问矩阵范围外的元素
    if (col < width && row < height)
    {
        // 将二维坐标转换为一维行优先索引
        // 这是理解 CUDA 中二维数据处理的关键
        int idx = row * width + col;

        // 执行矩阵加法
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

/**
 * @brief  矩阵加法 Kernel（统一内存版本）
 * @param  a  输入矩阵 A（统一内存，CPU和GPU均可访问）
 * @param  b  输入矩阵 B（统一内存）
 * @param  c  输出矩阵 C（统一内存）
 * @param  width   矩阵列数
 * @param  height  矩阵行数
 * @return void
 *
 * @details 使用 cudaMallocManaged 分配的内存称为"统一内存"（Unified Memory）。
 *          统一内存的优势：
 *          1. 不需要显式调用 cudaMemcpy 进行数据传输
 *          2. 系统自动在 CPU 和 GPU 之间迁移数据页
 *          3. 代码更简洁，适合快速原型开发
 *
 *          统一内存的缺点：
 *          1. 数据传输发生在首次访问时，可能造成不可预测的性能开销
 *          2. 对数据迁移的控制粒度较粗
 *          3. 老版本 GPU 支持有限
 */
__global__ void matrix_add_kernel_unified(const float* a, const float* b, float* c,
                                           int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}

/* ============================================================
 *  主机辅助函数
 * ============================================================ */

/**
 * @brief  CPU 版本的矩阵加法（用于验证）
 * @param  a  输入矩阵 A（主机内存）
 * @param  b  输入矩阵 B（主机内存）
 * @param  c  输出矩阵 C（主机内存）
 * @param  width   矩阵列数
 * @param  height  矩阵行数
 * @return void
 */
void matrix_add_cpu(const float* a, const float* b, float* c, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int idx = row * width + col;
            c[idx] = a[idx] + b[idx];
        }
    }
}

/**
 * @brief  打印矩阵的左上角预览
 * @param  mat     矩阵数据指针
 * @param  width   矩阵列数
 * @param  height  矩阵行数
 * @param  label   打印标签
 * @param  preview_size  预览大小（默认 4）
 * @return void
 */
void print_matrix_preview(const float* mat, int width, int height,
                           const char* label, int preview_size)
{
    printf("%s (%dx%d, 左上角 %dx%d):\n", label, height, width, preview_size, preview_size);
    for (int row = 0; row < preview_size && row < height; row++)
    {
        printf("  ");
        for (int col = 0; col < preview_size && col < width; col++)
        {
            printf("%8.4f ", mat[row * width + col]);
        }
        printf("...\n");
    }
}

/* ============================================================
 *  主函数
 * ============================================================ */

int main(void)
{
    printf("\n");
    printf("=============================================================\n");
    printf("     Day 3: 矩阵加法 — CUDA 内存模型与数据传输                \n");
    printf("=============================================================\n\n");

    print_device_info();

    /* --------------------------------------------------------
     *  第一部分：显式内存管理版本
     * -------------------------------------------------------- */

    printf("\n=============================================================\n");
    printf("     第一部分：显式内存管理（cudaMalloc + cudaMemcpy）        \n");
    printf("=============================================================\n\n");

    // 计算数据大小
    int num_elements = MATRIX_ROWS * MATRIX_COLS;
    size_t bytes = num_elements * sizeof(float);

    printf("[主机] 矩阵尺寸: %d x %d = %d 个元素 (%.2f MB)\n",
           MATRIX_ROWS, MATRIX_COLS, num_elements, bytes / (1024.0 * 1024.0));

    // 在主机内存中分配矩阵
    printf("[主机] 分配主机内存 ...\n");
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    float* h_c_ref = (float*)malloc(bytes);

    if (!h_a || !h_b || !h_c || !h_c_ref)
    {
        fprintf(stderr, "[错误] 主机内存分配失败！\n");
        return EXIT_FAILURE;
    }

    // 初始化矩阵
    printf("[主机] 初始化输入矩阵 ...\n");
    srand((unsigned int)time(NULL));
    init_random_float(h_a, num_elements, -1.0f, 1.0f);
    init_random_float(h_b, num_elements, -1.0f, 1.0f);

    print_matrix_preview(h_a, MATRIX_COLS, MATRIX_ROWS, "[主机] 矩阵 A", 4);
    print_matrix_preview(h_b, MATRIX_COLS, MATRIX_ROWS, "[主机] 矩阵 B", 4);

    // 在设备内存中分配矩阵
    printf("\n[主机] 分配设备内存 (cudaMalloc) ...\n");
    float *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes));

    // 将数据从主机复制到设备
    printf("[主机] 数据传输 H2D ...\n");
    GpuTimer timer_h2d;
    gpu_timer_init(&timer_h2d);
    gpu_timer_start(&timer_h2d);
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    gpu_timer_stop(&timer_h2d);
    gpu_timer_print(&timer_h2d, "H2D 数据传输");

    // 配置二维 Grid 和 Block
    // dim3(x, y) 创建二维维度，z 默认为 1
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_dim(
        (MATRIX_COLS + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,  // X 方向 Block 数
        (MATRIX_ROWS + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y   // Y 方向 Block 数
    );

    printf("\n[主机] Kernel 配置:\n");
    printf("  Block 维度: (%d, %d) = %d 线程/Block\n",
           BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_X * BLOCK_SIZE_Y);
    printf("  Grid 维度 : (%d, %d) = %d Blocks\n",
           grid_dim.x, grid_dim.y, grid_dim.x * grid_dim.y);
    printf("  总线程数  : %d\n", grid_dim.x * grid_dim.y * BLOCK_SIZE_X * BLOCK_SIZE_Y);

    // 启动 Kernel
    printf("\n[主机] 启动矩阵加法 Kernel ...\n");
    GpuTimer timer_kernel;
    gpu_timer_init(&timer_kernel);
    gpu_timer_start(&timer_kernel);

    matrix_add_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, MATRIX_COLS, MATRIX_ROWS);

    gpu_timer_stop(&timer_kernel);
    CHECK_KERNEL_ERROR();
    gpu_timer_print(&timer_kernel, "Kernel 执行 (显式内存)");

    // 复制结果回主机
    printf("\n[主机] 数据传输 D2H ...\n");
    GpuTimer timer_d2h;
    gpu_timer_init(&timer_d2h);
    gpu_timer_start(&timer_d2h);
    CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    gpu_timer_stop(&timer_d2h);
    gpu_timer_print(&timer_d2h, "D2H 数据传输");

    // CPU 验证
    printf("\n[主机] CPU 参考计算 ...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    matrix_add_cpu(h_a, h_b, h_c_ref, MATRIX_COLS, MATRIX_ROWS);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU 矩阵加法");

    // 验证结果
    printf("\n[主机] 验证 GPU 结果 ...\n");
    int is_correct = compare_float_arrays(h_c, h_c_ref, num_elements, TOLERANCE);
    print_matrix_preview(h_c, MATRIX_COLS, MATRIX_ROWS, "[主机] GPU 结果 C", 4);

    // 释放显式管理内存
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));

    /* --------------------------------------------------------
     *  第二部分：统一内存版本（cudaMallocManaged）
     * -------------------------------------------------------- */

    printf("\n=============================================================\n");
    printf("     第二部分：统一内存（cudaMallocManaged）                  \n");
    printf("=============================================================\n\n");

    // 使用统一内存分配 — 不需要显式 cudaMemcpy！
    printf("[主机] 分配统一内存 (cudaMallocManaged) ...\n");
    float *u_a, *u_b, *u_c;
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&u_a, bytes));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&u_b, bytes));
    CUDA_SAFE_CALL(cudaMallocManaged((void**)&u_c, bytes));

    // 直接在 CPU 上初始化统一内存
    printf("[主机] 初始化统一内存数据 ...\n");
    init_random_float(u_a, num_elements, -1.0f, 1.0f);
    init_random_float(u_b, num_elements, -1.0f, 1.0f);

    // 启动 Kernel — 不需要显式数据传输！
    printf("[主机] 启动 Kernel（无需显式数据传输）...\n");
    GpuTimer timer_unified;
    gpu_timer_init(&timer_unified);
    gpu_timer_start(&timer_unified);

    // 注意：首次在 GPU 上访问统一内存时，系统会自动将数据页迁移到 GPU
    matrix_add_kernel_unified<<<grid_dim, block_dim>>>(u_a, u_b, u_c, MATRIX_COLS, MATRIX_ROWS);

    gpu_timer_stop(&timer_unified);
    CHECK_KERNEL_ERROR();

    // 同步设备，确保 GPU 完成写入
    // 在读取统一内存前需要同步，否则可能读到旧数据
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    gpu_timer_print(&timer_unified, "Kernel + 自动数据迁移");

    // 直接在 CPU 上读取结果 — 系统会自动将数据页迁移回 CPU
    printf("\n[主机] 统一内存结果（直接在 CPU 上访问）:\n");
    print_matrix_preview(u_c, MATRIX_COLS, MATRIX_ROWS, "[主机] 统一内存结果 C", 4);

    // 验证统一内存结果
    printf("\n[主机] 验证统一内存结果 ...\n");
    int is_unified_correct = compare_float_arrays(u_c, h_c_ref, num_elements, TOLERANCE);

    // 释放统一内存
    CUDA_SAFE_CALL(cudaFree(u_a));
    CUDA_SAFE_CALL(cudaFree(u_b));
    CUDA_SAFE_CALL(cudaFree(u_c));

    /* --------------------------------------------------------
     *  性能总结
     * -------------------------------------------------------- */

    printf("\n");
    printf("=============================================================\n");
    printf("                        性能总结                              \n");
    printf("=============================================================\n");
    printf("  矩阵尺寸            : %d x %d\n", MATRIX_ROWS, MATRIX_COLS);
    printf("  显式内存 Kernel     : %.4f ms\n", gpu_timer_get_elapsed_ms(&timer_kernel));
    printf("  显式内存 H2D 传输   : %.4f ms\n", gpu_timer_get_elapsed_ms(&timer_h2d));
    printf("  显式内存 D2H 传输   : %.4f ms\n", gpu_timer_get_elapsed_ms(&timer_d2h));
    printf("  统一内存 Kernel+迁移: %.4f ms\n", gpu_timer_get_elapsed_ms(&timer_unified));
    printf("  CPU 计算时间        : %.4f ms\n", gpu_timer_get_elapsed_ms(&timer_cpu));
    printf("  显式内存结果验证    : %s\n", is_correct ? "通过" : "失败");
    printf("  统一内存结果验证    : %s\n", is_unified_correct ? "通过" : "失败");
    printf("=============================================================\n");

    /* --------------------------------------------------------
     *  知识点总结
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. 二维索引: row = blockIdx.y * blockDim.y + threadIdx.y\n");
    printf("              col = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("  2. 行优先布局: idx = row * width + col\n");
    printf("  3. 显式内存: cudaMalloc + cudaMemcpy(H2D/D2H)\n");
    printf("  4. 统一内存: cudaMallocManaged，自动数据迁移\n");
    printf("  5. 统一内存优点: 代码简洁，无需显式拷贝\n");
    printf("  6. 统一内存缺点: 迁移开销可能不可预测\n");
    printf("=============================================================\n");
    printf("\n");

    // 释放主机内存
    free(h_a); free(h_b); free(h_c); free(h_c_ref);
    gpu_timer_destroy(&timer_h2d);
    gpu_timer_destroy(&timer_kernel);
    gpu_timer_destroy(&timer_d2h);
    gpu_timer_destroy(&timer_cpu);
    gpu_timer_destroy(&timer_unified);

    printf("[主机] 资源已释放，程序正常退出。\n\n");
    return (is_correct && is_unified_correct) ? 0 : 1;
}
