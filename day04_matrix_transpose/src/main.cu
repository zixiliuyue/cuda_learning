/******************************************************************************
 * @file    main.cu
 * @brief   Day 4: 矩阵转置 — 错误处理与调试基础
 * @details 本程序实现矩阵转置: out[col][row] = in[row][col]
 *
 *          核心学习目标：
 *          1. 完整的 CUDA 错误处理流程（从分配内存到 Kernel 执行）
 *          2. 使用 CUDA_SAFE_CALL 宏包裹所有 CUDA API 调用
 *          3. 使用 CHECK_KERNEL_ERROR 检查异步 Kernel 错误
 *          4. 理解非合并内存访问（矩阵转置天然存在）
 *          5. 使用 printf 在 Kernel 中进行简单调试
 *          6. 编写防御性代码，优雅处理各种错误情况
 *
 *          转置操作的内存访问模式：
 *          - 读取输入矩阵：按行读取（连续访问，合并访问）
 *          - 写入输出矩阵：按列写入（跨行跳跃，非合并访问）
 *          这是理解内存优化重要性的经典案例。
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

#define MATRIX_ROWS 1024
#define MATRIX_COLS 1024
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define TOLERANCE 1e-5f

/* ============================================================
 *  Kernel 函数
 * ============================================================ */

/**
 * @brief  矩阵转置 Kernel（简单版本）
 * @param  d_in   输入矩阵（行优先展平）
 * @param  d_out  输出矩阵（行优先展平）
 * @param  width   输入矩阵列数（输出矩阵行数）
 * @param  height  输入矩阵行数（输出矩阵列数）
 * @return void
 *
 * @details 转置逻辑: out[col][row] = in[row][col]
 *          一维索引映射: out[col * height + row] = in[row * width + col]
 *
 *          内存访问模式分析：
 *          - 读取 d_in:  线程 (row, col) 读取 in[row*width + col]
 *                       同一 Warp 内的线程 col 递增，访问连续内存 -> 合并访问
 *          - 写入 d_out: 线程 (row, col) 写入 out[col*height + row]
 *                       同一 Warp 内的线程写入位置间隔 height -> 非合并访问
 */
__global__ void matrix_transpose_kernel(const float* d_in, float* d_out,
                                         int width, int height)
{
    // 计算当前线程负责的输入矩阵位置
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查
    if (col < width && row < height)
    {
        // 读取输入矩阵元素（行优先，连续访问）
        float val = d_in[row * width + col];

        // 写入输出矩阵的转置位置（列优先映射到行优先）
        // out[col][row] 在行优先展平中的索引 = col * height + row
        d_out[col * height + row] = val;
    }
}

/**
 * @brief  矩阵转置 Kernel（带调试输出的版本）
 * @details 仅在 DEBUG 模式下启用 printf，展示如何在 Kernel 中调试
 */
__global__ void matrix_transpose_kernel_debug(const float* d_in, float* d_out,
                                               int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int in_idx = row * width + col;
        int out_idx = col * height + row;
        float val = d_in[in_idx];
        d_out[out_idx] = val;

        // 仅在特定条件下打印，避免输出过多
        // 条件：前几个线程且是第一个 Block
        if (blockIdx.x == 0 && blockIdx.y == 0 &&
            threadIdx.x < 2 && threadIdx.y < 2)
        {
            printf("  [Debug] Thread(%d,%d): in[%d]=%.4f -> out[%d]\n",
                   threadIdx.x, threadIdx.y, in_idx, val, out_idx);
        }
    }
}

/* ============================================================
 *  主机辅助函数
 * ============================================================ */

void matrix_transpose_cpu(const float* in, float* out, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            out[col * height + row] = in[row * width + col];
        }
    }
}

void print_matrix_corner(const float* mat, int width, int height,
                          const char* label, int n)
{
    printf("%s (%dx%d, 左上角 %dx%d):\n", label, height, width, n, n);
    for (int row = 0; row < n && row < height; row++)
    {
        printf("  ");
        for (int col = 0; col < n && col < width; col++)
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
    printf("     Day 4: 矩阵转置 — 错误处理与调试基础                     \n");
    printf("=============================================================\n\n");

    print_device_info();

    int num_elements = MATRIX_ROWS * MATRIX_COLS;
    size_t bytes = num_elements * sizeof(float);

    printf("[主机] 矩阵尺寸: %d x %d (%.2f MB)\n",
           MATRIX_ROWS, MATRIX_COLS, bytes / (1024.0 * 1024.0));

    /* --------------------------------------------------------
     *  第一步：分配主机内存（带错误检查）
     * -------------------------------------------------------- */
    printf("\n[主机] 分配主机内存 ...\n");

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    float* h_out_ref = (float*)malloc(bytes);

    // 防御性编程：检查所有指针是否分配成功
    if (h_in == NULL || h_out == NULL || h_out_ref == NULL)
    {
        fprintf(stderr, "[错误] 主机内存分配失败！请检查系统内存是否充足。\n");
        // 释放已分配的内存，避免泄漏
        free(h_in); free(h_out); free(h_out_ref);
        return EXIT_FAILURE;
    }
    printf("[主机] 主机内存分配成功。\n");

    /* --------------------------------------------------------
     *  第二步：初始化数据
     * -------------------------------------------------------- */
    printf("[主机] 初始化输入矩阵 ...\n");
    srand((unsigned int)time(NULL));

    // 使用模式初始化，使转置结果容易验证
    // in[row][col] = row * width + col，转置后 out[col][row] 应该等于相同的值
    for (int row = 0; row < MATRIX_ROWS; row++)
    {
        for (int col = 0; col < MATRIX_COLS; col++)
        {
            h_in[row * MATRIX_COLS + col] = (float)(row * MATRIX_COLS + col);
        }
    }

    print_matrix_corner(h_in, MATRIX_COLS, MATRIX_ROWS, "[主机] 输入矩阵", 4);

    /* --------------------------------------------------------
     *  第三步：分配设备内存（使用 CUDA_SAFE_CALL 宏）
     * -------------------------------------------------------- */
    printf("\n[主机] 分配设备内存 ...\n");

    float* d_in = NULL;
    float* d_out = NULL;

    // CUDA_SAFE_CALL 宏会自动检查 cudaMalloc 的返回值
    // 如果失败，会打印错误信息（文件名、行号、错误描述）并终止程序
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_in, bytes));
    printf("[主机] 设备内存 d_in 分配成功: %zu bytes\n", bytes);

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, bytes));
    printf("[主机] 设备内存 d_out 分配成功: %zu bytes\n", bytes);

    /* --------------------------------------------------------
     *  第四步：数据传输 H2D
     * -------------------------------------------------------- */
    printf("\n[主机] 数据传输 H2D ...\n");
    GpuTimer timer_h2d;
    gpu_timer_init(&timer_h2d);
    gpu_timer_start(&timer_h2d);

    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    gpu_timer_stop(&timer_h2d);
    gpu_timer_print(&timer_h2d, "H2D 数据传输");

    /* --------------------------------------------------------
     *  第五步：启动 Kernel（带调试版本）
     * -------------------------------------------------------- */
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_dim(
        (MATRIX_COLS + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (MATRIX_ROWS + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );

    printf("\n[主机] Kernel 配置: Grid(%d,%d), Block(%d,%d)\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);

    // 先运行调试版本（只打印少量信息）
    printf("\n[主机] 运行调试版本 Kernel（观察线程行为）...\n");
    matrix_transpose_kernel_debug<<<grid_dim, block_dim>>>(d_in, d_out, MATRIX_COLS, MATRIX_ROWS);
    CHECK_KERNEL_ERROR();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    printf("[主机] 调试 Kernel 完成。\n");

    // 再运行正式版本（计时）
    printf("\n[主机] 运行正式版本 Kernel ...\n");
    GpuTimer timer_kernel;
    gpu_timer_init(&timer_kernel);
    gpu_timer_start(&timer_kernel);

    matrix_transpose_kernel<<<grid_dim, block_dim>>>(d_in, d_out, MATRIX_COLS, MATRIX_ROWS);

    gpu_timer_stop(&timer_kernel);
    CHECK_KERNEL_ERROR();
    gpu_timer_print(&timer_kernel, "转置 Kernel");

    /* --------------------------------------------------------
     *  第六步：数据传输 D2H
     * -------------------------------------------------------- */
    printf("\n[主机] 数据传输 D2H ...\n");
    GpuTimer timer_d2h;
    gpu_timer_init(&timer_d2h);
    gpu_timer_start(&timer_d2h);

    CUDA_SAFE_CALL(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    gpu_timer_stop(&timer_d2h);
    gpu_timer_print(&timer_d2h, "D2H 数据传输");

    /* --------------------------------------------------------
     *  第七步：验证结果
     * -------------------------------------------------------- */
    printf("\n[主机] CPU 参考计算 ...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    matrix_transpose_cpu(h_in, h_out_ref, MATRIX_COLS, MATRIX_ROWS);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU 转置");

    printf("\n[主机] 验证 GPU 结果 ...\n");
    int is_correct = compare_float_arrays(h_out, h_out_ref, num_elements, TOLERANCE);

    print_matrix_corner(h_out, MATRIX_ROWS, MATRIX_COLS, "[主机] 转置结果", 4);

    /* --------------------------------------------------------
     *  第八步：故意触发一个错误（教学演示）
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     错误处理演示（教学目的）                                 \n");
    printf("=============================================================\n\n");

    printf("[教学] 下面演示如何捕获和处理 CUDA 错误 ...\n\n");

    // 演示 1: 捕获 cudaMemcpy 方向错误
    printf("[演示 1] 尝试用错误的方向进行 cudaMemcpy ...\n");
    cudaError_t err_test;

    // 故意使用错误的方向：应该用 DeviceToHost，但这里用 HostToDevice
    // 注意：这不是致命错误，cudaMemcpy 会返回错误码但不会崩溃
    err_test = cudaMemcpy(h_out, d_out, 10, cudaMemcpyHostToDevice);
    if (err_test != cudaSuccess)
    {
        printf("  [捕获成功] 预期错误: %s\n", cudaGetErrorString(err_test));
        printf("  [说明] 将设备指针作为 HostToDevice 的目标是无效的\n");
    }
    // 清除错误状态，避免影响后续操作
    cudaGetLastError();

    // 演示 2: 检查 Kernel 配置是否合法
    printf("\n[演示 2] 检查 Kernel 启动配置的合法性 ...\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  设备最大 Block 维度: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  当前配置 Block 维度: (%d, %d, 1)\n", BLOCK_SIZE_X, BLOCK_SIZE_Y);
    if (BLOCK_SIZE_X <= prop.maxThreadsDim[0] && BLOCK_SIZE_Y <= prop.maxThreadsDim[1])
    {
        printf("  [检查通过] 当前配置在设备限制范围内\n");
    }
    else
    {
        printf("  [检查失败] 当前配置超出设备限制！\n");
    }

    /* --------------------------------------------------------
     *  性能总结
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("                        性能总结                              \n");
    printf("=============================================================\n");
    printf("  矩阵尺寸        : %d x %d\n", MATRIX_ROWS, MATRIX_COLS);
    printf("  GPU Kernel      : %.4f ms\n", gpu_timer_get_elapsed_ms(&timer_kernel));
    printf("  CPU 计算        : %.4f ms\n", gpu_timer_get_elapsed_ms(&timer_cpu));
    printf("  结果验证        : %s\n", is_correct ? "通过" : "失败");
    printf("=============================================================\n");

    /* --------------------------------------------------------
     *  知识点总结
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. 转置索引: out[col*H + row] = in[row*W + col]\n");
    printf("  2. 读取合并: 同一 Warp 线程连续读取 -> 高效\n");
    printf("  3. 写入非合并: 同一 Warp 线程跨行写入 -> 低效\n");
    printf("  4. CUDA_SAFE_CALL: 包裹所有 CUDA API 调用\n");
    printf("  5. CHECK_KERNEL_ERROR: 同步并检查 Kernel 错误\n");
    printf("  6. Kernel printf: 可用于简单调试（需同步）\n");
    printf("  7. 防御性编程: 检查所有分配和边界\n");
    printf("=============================================================\n");
    printf("\n");

    /* --------------------------------------------------------
     *  释放资源
     * -------------------------------------------------------- */
    CUDA_SAFE_CALL(cudaFree(d_in));
    CUDA_SAFE_CALL(cudaFree(d_out));
    free(h_in); free(h_out); free(h_out_ref);
    gpu_timer_destroy(&timer_h2d);
    gpu_timer_destroy(&timer_kernel);
    gpu_timer_destroy(&timer_d2h);
    gpu_timer_destroy(&timer_cpu);

    printf("[主机] 资源已释放，程序正常退出。\n\n");
    return is_correct ? 0 : 1;
}
