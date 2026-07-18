/******************************************************************************
 * @file    main.cu
 * @brief   Day 2: 向量加法 — 理解 CUDA 执行模型与线程层次
 * @details 本程序实现两个向量的逐元素相加: C[i] = A[i] + B[i]
 *
 *          核心学习目标：
 *          1. 一维线程组织的实际应用 — 每个线程处理一个数组元素
 *          2. 全局线程 ID 的正确计算: tid = blockIdx.x * blockDim.x + threadIdx.x
 *          3. 边界检查 — 防止线程访问超出数组范围的内存
 *          4. GPU 结果与 CPU 结果对比验证
 *          5. 使用 CUDA_SAFE_CALL 宏进行错误检查
 *          6. 使用 GpuTimer 测量 Kernel 执行时间
 *
 *          线程分配策略：
 *          - 向量长度 N = 1,000,000
 *          - 每个 Block 256 个线程（ Warp 大小的整数倍，效率最高）
 *          - 需要的 Block 数量 = (N + 255) / 256 = 3907
 *          - 总线程数 = 3907 * 256 = 1,000,192（略大于 N，多余线程通过边界检查跳过）
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

/** @brief 向量元素总数（100万个元素） */
#define VECTOR_SIZE 1000000

/** @brief 每个 Block 的线程数（256 = 8 * WarpSize，是 Warp 的整数倍） */
#define THREADS_PER_BLOCK 256

/** @brief 浮点数比较的容差 */
#define TOLERANCE 1e-5f

/* ============================================================
 *  Kernel 函数（在 GPU 上执行）
 * ============================================================ */

/**
 * @brief  向量加法 Kernel
 * @param  d_a  输入向量 A（设备内存）
 * @param  d_b  输入向量 B（设备内存）
 * @param  d_c  输出向量 C（设备内存）
 * @param  n    向量元素个数
 * @return void
 *
 * @details 每个线程计算一个元素的加法: d_c[tid] = d_a[tid] + d_b[tid]
 *
 *          线程索引计算：
 *          - blockIdx.x:  当前 Block 在 Grid 中的索引
 *          - blockDim.x:  每个 Block 包含的线程数（此处为 256）
 *          - threadIdx.x: 当前线程在 Block 中的索引
 *          - 全局线程 ID = blockIdx.x * blockDim.x + threadIdx.x
 *
 *          边界检查：
 *          - 由于 Block 数量是向上取整的，最后几个 Block 中的部分线程
 *            可能会超出数组范围，必须通过 if (tid < n) 进行边界保护。
 *          - 缺少边界检查是导致 CUDA 程序崩溃（非法内存访问）的常见原因。
 */
__global__ void vector_add_kernel(const float* d_a, const float* d_b, float* d_c, int n)
{
    // 计算当前线程的全局唯一索引
    // 这是 CUDA 编程中最常用的索引计算方式
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保线程不会访问超出数组范围的内存
    // 这是 CUDA 编程的安全最佳实践，必须始终进行
    if (tid < n)
    {
        // 执行实际的向量加法运算
        // 每个线程独立计算一个元素，32个线程（一个Warp）同时执行
        d_c[tid] = d_a[tid] + d_b[tid];
    }
    // 超出范围的线程什么都不做，直接返回
}

/* ============================================================
 *  主机辅助函数（在 CPU 上执行）
 * ============================================================ */

/**
 * @brief  CPU 版本的向量加法（用于验证 GPU 结果）
 * @param  a  输入向量 A（主机内存）
 * @param  b  输入向量 B（主机内存）
 * @param  c  输出向量 C（主机内存）
 * @param  n  向量元素个数
 * @return void
 */
void vector_add_cpu(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief  计算 Grid 维度（Block 数量）
 * @param  total_threads  需要处理的总元素数
 * @param  block_size     每个 Block 的线程数
 * @return int            需要的 Block 数量
 *
 * @details 使用向上取整除法: (n + block_size - 1) / block_size
 *          这样可以确保即使 n 不是 block_size 的整数倍，
 *          也能覆盖所有元素。
 */
int calculate_grid_size(int total_threads, int block_size)
{
    return (total_threads + block_size - 1) / block_size;
}

/* ============================================================
 *  主函数
 * ============================================================ */

int main(void)
{
    /* --------------------------------------------------------
     *  第一步：打印程序信息
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("     Day 2: 向量加法 — CUDA 执行模型与线程层次                \n");
    printf("=============================================================\n\n");

    print_device_info();

    /* --------------------------------------------------------
     *  第二步：在主机内存中分配和初始化数据
     * -------------------------------------------------------- */

    // 计算数据大小（字节）
    size_t bytes = VECTOR_SIZE * sizeof(float);

    printf("[主机] 向量大小: %d 个元素 (%.2f MB)\n", VECTOR_SIZE, bytes / (1024.0 * 1024.0));
    printf("[主机] 正在分配主机内存 ...\n");

    // 在主机（CPU）内存中分配三个向量
    // h_ 前缀表示 Host（主机）内存
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);        // GPU 结果
    float* h_c_ref = (float*)malloc(bytes);    // CPU 参考结果（用于验证）

    if (h_a == NULL || h_b == NULL || h_c == NULL || h_c_ref == NULL)
    {
        fprintf(stderr, "[错误] 主机内存分配失败！\n");
        return EXIT_FAILURE;
    }

    // 初始化输入向量 A 和 B
    // 使用随机数填充，范围 [-1.0, 1.0]
    printf("[主机] 正在初始化输入数据 ...\n");
    srand((unsigned int)time(NULL));
    init_random_float(h_a, VECTOR_SIZE, -1.0f, 1.0f);
    init_random_float(h_b, VECTOR_SIZE, -1.0f, 1.0f);

    // 打印输入向量的前 5 个元素（用于直观检查）
    printf("[主机] 向量 A 预览: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_a[i]);
    printf("...\n");
    printf("[主机] 向量 B 预览: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", h_b[i]);
    printf("...\n");

    /* --------------------------------------------------------
     *  第三步：在设备内存中分配空间
     * -------------------------------------------------------- */

    printf("\n[主机] 正在分配设备内存 ...\n");

    // d_ 前缀表示 Device（设备/GPU）内存
    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;

    // cudaMalloc: 在 GPU 显存中分配指定大小的内存
    // 返回的指针是设备指针，只能在 GPU 代码中使用，不能在 CPU 代码中解引用
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_a, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_c, bytes));

    printf("[主机] 设备内存分配完成 (%.2f MB x 3)\n", bytes / (1024.0 * 1024.0));

    /* --------------------------------------------------------
     *  第四步：将数据从主机复制到设备
     * -------------------------------------------------------- */

    printf("[主机] 正在将数据从主机复制到设备 (H2D) ...\n");

    GpuTimer timer_h2d;
    gpu_timer_init(&timer_h2d);
    gpu_timer_start(&timer_h2d);

    // cudaMemcpy: 内存拷贝函数
    // 方向 cudaMemcpyHostToDevice: 从主机内存复制到设备内存
    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    gpu_timer_stop(&timer_h2d);
    gpu_timer_print(&timer_h2d, "主机→设备数据传输");

    /* --------------------------------------------------------
     *  第五步：启动 CUDA Kernel
     * -------------------------------------------------------- */

    // 计算 Grid 维度（Block 数量）
    int grid_size = calculate_grid_size(VECTOR_SIZE, THREADS_PER_BLOCK);

    printf("\n[主机] Kernel 启动配置:\n");
    printf("  Block 维度 : %d 线程/Block\n", THREADS_PER_BLOCK);
    printf("  Grid 维度  : %d Blocks\n", grid_size);
    printf("  总线程数   : %d (覆盖 %d 个元素)\n", grid_size * THREADS_PER_BLOCK, VECTOR_SIZE);
    printf("  多余线程   : %d 个（将被边界检查跳过）\n",
           grid_size * THREADS_PER_BLOCK - VECTOR_SIZE);

    // 准备计时器
    GpuTimer timer_kernel;
    gpu_timer_init(&timer_kernel);

    printf("\n[主机] 启动向量加法 Kernel ...\n");

    // 启动 Kernel
    // <<<grid_size, THREADS_PER_BLOCK>>> 配置了线程组织结构
    // 所有线程并行执行 vector_add_kernel 函数
    gpu_timer_start(&timer_kernel);
    vector_add_kernel<<<grid_size, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, VECTOR_SIZE);
    gpu_timer_stop(&timer_kernel);

    // 同步并检查 Kernel 错误
    CHECK_KERNEL_ERROR();

    gpu_timer_print(&timer_kernel, "Kernel 执行 (向量加法)");

    /* --------------------------------------------------------
     *  第六步：将结果从设备复制回主机
     * -------------------------------------------------------- */

    printf("\n[主机] 正在将结果从设备复制回主机 (D2H) ...\n");

    GpuTimer timer_d2h;
    gpu_timer_init(&timer_d2h);
    gpu_timer_start(&timer_d2h);

    // 方向 cudaMemcpyDeviceToHost: 从设备内存复制到主机内存
    CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    gpu_timer_stop(&timer_d2h);
    gpu_timer_print(&timer_d2h, "设备→主机数据传输");

    /* --------------------------------------------------------
     *  第七步：验证 GPU 结果的正确性
     * -------------------------------------------------------- */

    printf("\n[主机] 正在用 CPU 计算参考结果以验证 ...\n");

    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);

    // 在 CPU 上计算相同的向量加法作为参考
    vector_add_cpu(h_a, h_b, h_c_ref, VECTOR_SIZE);

    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU 参考计算");

    // 对比 GPU 和 CPU 的结果
    printf("\n[主机] 验证 GPU 计算结果 ...\n");
    int is_correct = compare_float_arrays(h_c, h_c_ref, VECTOR_SIZE, TOLERANCE);

    // 打印结果预览
    printf("\n[主机] 结果预览 (前 10 个元素):\n");
    printf("  索引    |  A[i]   |  B[i]   | GPU C[i] | CPU C[i] |\n");
    printf("  --------|---------|---------|----------|----------|\n");
    for (int i = 0; i < 10; i++)
    {
        printf("  %6d  | %7.4f | %7.4f | %8.4f | %8.4f |\n",
               i, h_a[i], h_b[i], h_c[i], h_c_ref[i]);
    }

    /* --------------------------------------------------------
     *  第八步：性能总结
     * -------------------------------------------------------- */

    double kernel_ms = gpu_timer_get_elapsed_ms(&timer_kernel);
    double cpu_ms = gpu_timer_get_elapsed_ms(&timer_cpu);

    printf("\n");
    printf("=============================================================\n");
    printf("                        性能总结                              \n");
    printf("=============================================================\n");
    printf("  数据量              : %d 个元素 (%.2f MB)\n", VECTOR_SIZE, bytes / (1024.0 * 1024.0));
    printf("  GPU Kernel 时间     : %.4f ms\n", kernel_ms);
    printf("  CPU 计算时间        : %.4f ms\n", cpu_ms);
    if (kernel_ms > 0 && cpu_ms > 0)
    {
        printf("  GPU 加速比          : %.2f x\n", cpu_ms / kernel_ms);
    }
    printf("  结果验证            : %s\n", is_correct ? "通过" : "失败");
    printf("=============================================================\n");

    /* --------------------------------------------------------
     *  第九步：知识点总结
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. 全局线程 ID:  tid = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("  2. 边界检查:    if (tid < n) 是防止越界访问的必要保护\n");
    printf("  3. Grid 计算:   (N + block_size - 1) / block_size 向上取整\n");
    printf("  4. Warp 对齐:   block_size 取 256（Warp 大小的整数倍）效率最高\n");
    printf("  5. 内存传输:    cudaMemcpy(H2D/D2H) 是 GPU 计算的固定开销\n");
    printf("  6. 结果验证:    始终用 CPU 参考实现验证 GPU 结果的正确性\n");
    printf("=============================================================\n");
    printf("\n");

    /* --------------------------------------------------------
     *  第十步：释放资源
     * -------------------------------------------------------- */

    // 释放设备内存
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));

    // 释放主机内存
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);

    // 销毁计时器
    gpu_timer_destroy(&timer_h2d);
    gpu_timer_destroy(&timer_kernel);
    gpu_timer_destroy(&timer_d2h);
    gpu_timer_destroy(&timer_cpu);

    printf("[主机] 资源已释放，程序正常退出。\n\n");

    return is_correct ? 0 : 1;
}
