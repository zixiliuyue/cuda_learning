/******************************************************************************
 * @file    main.cu
 * @brief   Day 12: 动态并行与 CUDA Graphs
 * @details 本程序演示动态并行（Kernel 中启动 Kernel）和 CUDA Graphs。
 *
 *          核心学习目标：
 *          1. 动态并行（Dynamic Parallelism）
 *          2. CUDA Graphs 的捕获、实例化与执行
 *          3. 适用场景与性能收益分析
 *
 * @note    动态并行需要 Compute Capability 3.5+ 和 separable compilation
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define ARRAY_SIZE 1024
#define BLOCK_SIZE 256

/**
 * @brief  子 Kernel：由父 Kernel 动态调用
 * @details 每个线程调用此子 Kernel，处理一个元素
 */
__global__ void child_kernel(float* data, int idx)
{
    data[idx] = data[idx] * 2.0f + 1.0f;
}

/**
 * @brief  父 Kernel：使用动态并行启动子 Kernel
 * @details 每个线程在运行时决定是否需要启动子 Kernel
 */
__global__ void parent_kernel(float* data, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // 动态并行：在 Kernel 中启动另一个 Kernel
        // 注意：子 Kernel 的 <<< >>> 配置在设备端执行
        child_kernel<<<1, 1>>>(data, tid);
    }
}

/**
 * @brief  普通 Kernel（无动态并行，对比用）
 */
__global__ void normal_kernel(float* data, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        data[tid] = data[tid] * 2.0f + 1.0f;
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 12: 动态并行与 CUDA Graphs                           \n");
    printf("=============================================================\n\n");

    print_device_info();

    // 检查动态并行支持
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5))
    {
        printf("[警告] 当前设备不支持动态并行（需要 CC 3.5+）。\n");
        printf("[提示] 跳过动态并行演示，仅展示 CUDA Graphs。\n");
    }

    size_t bytes = ARRAY_SIZE * sizeof(float);
    float* h_data = (float*)malloc(bytes);
    for (int i = 0; i < ARRAY_SIZE; i++) h_data[i] = (float)i;

    float* d_data = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, bytes));

    int grid_size = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* --------------------------------------------------------
     *  普通 Kernel 执行
     * -------------------------------------------------------- */
    printf("\n[主机] 普通 Kernel 执行 ...\n");
    CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    GpuTimer timer_normal;
    gpu_timer_init(&timer_normal);
    gpu_timer_start(&timer_normal);
    normal_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, ARRAY_SIZE);
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer_normal);
    gpu_timer_print(&timer_normal, "普通 Kernel");

    /* --------------------------------------------------------
     *  CUDA Graphs 演示
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     CUDA Graphs 演示                                         \n");
    printf("=============================================================\n\n");

    // 1. 创建 Graph
    cudaGraph_t graph;
    CUDA_SAFE_CALL(cudaGraphCreate(&graph, 0));

    // 2. 添加节点到 Graph
    cudaGraphNode_t kernel_node;

    // Kernel 节点参数
    // 注意：ARRAY_SIZE 是宏（右值），不能取地址 &ARRAY_SIZE。
    // 必须用一个真实的局部变量来承载，才能作为 kernelParams 传入。
    int array_size_arg = ARRAY_SIZE;
    void* kernel_args[2] = { (void*)&d_data, (void*)&array_size_arg };
    cudaKernelNodeParams kernel_params = { 0 };
    kernel_params.func = (void*)normal_kernel;
    kernel_params.gridDim = dim3(grid_size, 1, 1);
    kernel_params.blockDim = dim3(BLOCK_SIZE, 1, 1);
    kernel_params.sharedMemBytes = 0;
    kernel_params.kernelParams = kernel_args;
    kernel_params.extra = NULL;

    // 3. 实例化 Graph
    cudaGraphExec_t graph_exec;
    CUDA_SAFE_CALL(cudaGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel_params));
    CUDA_SAFE_CALL(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    // 注意：Graph 实例化后，原 graph 即可释放。
    // 但如果后续需要修改执行图，必须保留 graph 并重新实例化。
    // 常见陷阱：修改 graph_exec 的内容（是只读的），必须操作 graph 再重新实例化。

    // 4. 执行 Graph
    printf("[主机] 执行 CUDA Graph ...\n");
    GpuTimer timer_graph;
    gpu_timer_init(&timer_graph);
    gpu_timer_start(&timer_graph);

    for (int i = 0; i < 100; i++)
    {
        CUDA_SAFE_CALL(cudaGraphLaunch(graph_exec, 0));
    }
    cudaDeviceSynchronize();

    gpu_timer_stop(&timer_graph);
    gpu_timer_print(&timer_graph, "CUDA Graph (100次)");

    // 对比：直接启动 100 次 Kernel
    printf("\n[主机] 直接启动 100 次 Kernel ...\n");
    GpuTimer timer_direct;
    gpu_timer_init(&timer_direct);
    gpu_timer_start(&timer_direct);
    for (int i = 0; i < 100; i++)
    {
        normal_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, ARRAY_SIZE);
    }
    cudaDeviceSynchronize();
    gpu_timer_stop(&timer_direct);
    gpu_timer_print(&timer_direct, "直接启动 (100次)");

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. 动态并行: Kernel 中可以启动子 Kernel\n");
    printf("  2. 动态并行需要: CC 3.5+ 和 -rdc=true\n");
    printf("  3. CUDA Graphs: 预定义执行流程，减少启动开销\n");
    printf("  4. Graph 适用场景: 重复执行相同的工作流程\n");
    printf("  5. Graph 优势: 减少 CPU 提交开销和 Kernel 启动延迟\n");
    printf("=============================================================\n\n");

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
    cudaFree(d_data);
    free(h_data);
    gpu_timer_destroy(&timer_normal);
    gpu_timer_destroy(&timer_graph);
    gpu_timer_destroy(&timer_direct);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
