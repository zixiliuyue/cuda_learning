/******************************************************************************
 * @file    main.cu
 * @brief   Day 1: Hello CUDA — 我的第一个 GPU 程序
 * @details 这是 CUDA 30天学习项目的第一个程序，目标是理解：
 *          1. CUDA 程序的基本结构（主机代码 + 设备代码）
 *          2. Kernel 函数的语法 <<<grid, block>>>
 *          3. 线程索引的含义：threadIdx, blockIdx
 *          4. nvcc 编译流程
 *
 *          程序启动一个包含 4 个 Block、每个 Block 包含 8 个 Thread 的 Kernel，
 *          总共 32 个线程并行执行，每个线程打印自己的身份信息。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include "cuda_utils.h"
#include "platform.h"

/* ============================================================
 *  Kernel 函数（在 GPU 上执行）
 * ============================================================ */

/**
 * @brief  Hello CUDA Kernel 函数
 * @param  none
 * @return void
 *
 * @details 这是最简单的 CUDA Kernel 函数示例。
 *          每个被调用的线程都会执行一次此函数体。
 *
 *          关键概念：
 *          - __global__ 修饰符：表示这是一个 Kernel 函数，
 *            由主机（CPU）调用，在设备（GPU）上执行。
 *          - threadIdx.x：当前线程在其所在 Block 中的索引（0 到 blockDim.x-1）
 *          - blockIdx.x：当前 Block 在其所在 Grid 中的索引（0 到 gridDim.x-1）
 *          - blockDim.x：每个 Block 中包含的线程数量
 *
 * @note   Kernel 中可以使用 printf，但输出会被缓冲，
 *         需要 cudaDeviceSynchronize() 才能在主机端看到完整输出。
 */
__global__ void hello_cuda_kernel(void)
{
    // 计算全局线程 ID（在简单演示中可选，但有助于理解）
    // 全局 ID = 当前 Block 起始位置 + 当前线程在 Block 内的偏移
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程打印自己的身份信息
    // 注意：由于 32 个线程同时执行，输出的顺序是不确定的！
    printf("  [GPU线程] 全局ID=%02d | Block=%02d | Thread=%02d | 你好，CUDA!\n",
           global_tid, blockIdx.x, threadIdx.x);
}

/* ============================================================
 *  主机代码（在 CPU 上执行）
 * ============================================================ */

/**
 * @brief  主函数 — 程序入口
 * @return int  程序退出码（0=成功）
 */
int main(void)
{
    /* --------------------------------------------------------
     *  第一步：打印程序标题和平台信息
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("     Day 1: Hello CUDA — 我的第一个 GPU 程序                  \n");
    printf("=============================================================\n\n");

    // 使用 platform.h 中的宏打印当前平台信息
    PRINT_PLATFORM_INFO();
    printf("\n");

    /* --------------------------------------------------------
     *  第二步：查询并打印 GPU 设备信息
     * -------------------------------------------------------- */
    print_device_info();

    /* --------------------------------------------------------
     *  第三步：配置并启动 CUDA Kernel
     * -------------------------------------------------------- */

    // 定义 Grid 和 Block 的尺寸
    // dim3 是 CUDA 提供的三维整数向量类型，用于描述维度
    // 这里我们只使用一维，所以 y 和 z 默认为 1

    int num_blocks = 4;   // Grid 中包含 4 个 Block
    int threads_per_block = 8;  // 每个 Block 包含 8 个 Thread

    printf("\n[主机] 准备启动 Kernel ...\n");
    printf("[主机] Grid 维度 : %d 个 Block\n", num_blocks);
    printf("[主机] Block 维度: %d 个 Thread\n", threads_per_block);
    printf("[主机] 总线程数  : %d 个\n", num_blocks * threads_per_block);
    printf("\n");

    // 创建 dim3 类型的维度变量
    // dim3(x, y, z) — 这里 y 和 z 省略，默认为 1
    dim3 grid_dim(num_blocks);          // Grid:  4 个 Block
    dim3 block_dim(threads_per_block);  // Block: 8 个 Thread

    // 启动 Kernel！这是 CUDA 最核心的语法：<<< >>>
    // <<<grid_dim, block_dim>>> 指定了 Kernel 的启动配置
    // 所有 32 个线程将同时（并行）执行 hello_cuda_kernel 函数
    printf("[主机] 启动 Kernel <<< %d, %d >>> ...\n", num_blocks, threads_per_block);
    printf("------------------------------------------------------------\n");

    hello_cuda_kernel<<<grid_dim, block_dim>>>();

    // 同步设备，等待所有 Kernel 执行完成
    // 这是必要的，因为 Kernel 启动是异步的：
    // CPU 发出启动命令后会立即继续执行后续代码，
    // 如果不调用 cudaDeviceSynchronize()，可能看不到 Kernel 的输出
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    printf("------------------------------------------------------------\n");
    printf("[主机] Kernel 执行完成！\n");

    /* --------------------------------------------------------
     *  第四步：总结与解释
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. __global__  : Kernel 函数修饰符，CPU调用、GPU执行\n");
    printf("  2. <<<G, B>>>  : Kernel 启动配置，G=Grid维度，B=Block维度\n");
    printf("  3. threadIdx   : 线程在 Block 内的索引\n");
    printf("  4. blockIdx    : Block 在 Grid 内的索引\n");
    printf("  5. blockDim    : Block 包含的线程总数\n");
    printf("  6. gridDim     : Grid 包含的 Block 总数\n");
    printf("  7. 全局线程 ID : blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("\n");
    printf("  注意: 由于 32 个线程并行执行，printf 的输出顺序\n");
    printf("        是不确定的，每次运行可能不同！\n");
    printf("=============================================================\n");
    printf("\n");

    return 0;
}
