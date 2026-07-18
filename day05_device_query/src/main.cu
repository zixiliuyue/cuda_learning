/******************************************************************************
 * @file    main.cu
 * @brief   Day 5: GPU 设备查询与性能基准
 * @details 本程序全面查询 CUDA 设备的硬件特性，并运行微基准测试。
 *
 *          核心学习目标：
 *          1. 使用 cudaGetDeviceProperties 获取设备详细信息
 *          2. 理解 GPU 硬件参数的含义（SM数、Warp大小、寄存器等）
 *          3. 计算 GPU 的理论峰值性能
 *          4. 运行内存带宽基准测试
 *          5. 了解 Compute Capability 与功能特性的关系
 *          6. 编写通用设备信息查询工具
 *
 *          关键硬件参数解读：
 *          - 计算能力 (Compute Capability): 决定支持的 CUDA 功能版本
 *          - SM (Streaming Multiprocessor): GPU 的核心计算单元
 *          - Warp: 32 个线程组成的执行单元，是 GPU 调度的基本单位
 *          - 共享内存: 每个 SM 上的高速缓存，供 Block 内线程共享
 *          - 寄存器: 每个线程的私有高速存储
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "platform.h"

/* ============================================================
 *  设备信息查询函数
 * ============================================================ */

/**
 * @brief  打印单个 CUDA 设备的详细信息
 * @param  prop       设备属性结构体
 * @param  device_id  设备编号
 * @return void
 */
void print_device_details(const cudaDeviceProp& prop, int device_id)
{
    printf("\n");
    printf("=============================================================\n");
    printf("  设备 #%d: %s\n", device_id, prop.name);
    printf("=============================================================\n");

    /* ---------------- 基本信息 ---------------- */
    printf("\n【基本信息】\n");
    printf("  设备名称              : %s\n", prop.name);
    printf("  计算能力 (CC)         : %d.%d\n", prop.major, prop.minor);
    printf("  总显存                : %.2f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  显存类型              : %s\n",
           prop.integrated ? "集成显卡（共享系统内存）" : "独立显卡（专用显存）");

    /* ---------------- 计算能力 ---------------- */
    printf("\n【计算资源】\n");
    printf("  多处理器数量 (SM)     : %d\n", prop.multiProcessorCount);

    // 根据 Compute Capability 计算每个 SM 的 CUDA 核心数
    int cores_per_sm = 0;
    if (prop.major == 3) cores_per_sm = 192;       // Kepler
    else if (prop.major == 5) cores_per_sm = 128;  // Maxwell
    else if (prop.major == 6) cores_per_sm = 64;   // Pascal (GP100: 64)
    else if (prop.major == 7) cores_per_sm = 64;   // Turing/Volta
    else if (prop.major == 8) cores_per_sm = 128;  // Ampere
    else if (prop.major == 9) cores_per_sm = 128;  // Ada Lovelace
    else if (prop.major >= 10) cores_per_sm = 128; // Blackwell+
    else cores_per_sm = 128;

    printf("  每个 SM 的 CUDA 核心  : ~%d\n", cores_per_sm);
    printf("  总 CUDA 核心数（估算）: ~%d\n", prop.multiProcessorCount * cores_per_sm);
    printf("  最大线程 / Block      : %d\n", prop.maxThreadsPerBlock);
    printf("  最大线程 / SM         : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Warp 大小             : %d\n", prop.warpSize);

    /* ---------------- 内存信息 ---------------- */
    printf("\n【内存资源】\n");
    printf("  共享内存 / Block      : %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  共享内存 / SM         : %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  寄存器 / Block        : %d\n", prop.regsPerBlock);
    printf("  寄存器 / SM           : %d\n", prop.regsPerMultiprocessor);
    printf("  L2 缓存大小           : %zu KB\n", prop.l2CacheSize / 1024);
    printf("  显存总线宽度          : %d bits\n", prop.memoryBusWidth);
    printf("  显存时钟频率          : %.0f MHz\n", prop.memoryClockRate / 1000.0f);

    // 计算理论显存带宽
    double bandwidth_gbps = 2.0 * prop.memoryClockRate * 1000.0 *
                            (prop.memoryBusWidth / 8.0) / 1e9;
    printf("  理论显存带宽          : %.2f GB/s\n", bandwidth_gbps);

    /* ---------------- 维度限制 ---------------- */
    printf("\n【维度限制】\n");
    printf("  最大 Block 维度       : (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  最大 Grid 维度        : (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    /* ---------------- 特性支持 ---------------- */
    printf("\n【特性支持】\n");
    printf("  统一内存              : %s\n", prop.managedMemory ? "支持" : "不支持");
    printf("  并发内核执行          : %s\n", prop.concurrentKernels ? "支持" : "不支持");
    printf("  双精度浮点性能        : %s\n",
           prop.singleToDoublePrecisionPerfRatio > 0 ? "支持" : "不支持");
    printf("  ECC 内存纠错          : %s\n", prop.ECCEnabled ? "已启用" : "未启用");
    printf("  PCIe 总线 ID          : %d:%d.%d\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    /* ---------------- Compute Capability 特性 ---------------- */
    printf("\n【Compute Capability %d.%d 特性】\n", prop.major, prop.minor);

    // 根据 CC 版本列出支持的关键特性
    printf("  - 动态并行 (Dynamic Parallelism)   : %s\n",
           (prop.major >= 3) ? "支持" : "不支持");
    printf("  - 统一内存 (Unified Memory)        : %s\n",
           (prop.major >= 3) ? "支持" : "不支持");
    printf("  - Warp Shuffle 指令                : %s\n",
           (prop.major >= 3) ? "支持" : "不支持");
    printf("  - FP16 (半精度浮点)                : %s\n",
           (prop.major >= 5) ? "支持" : "不支持");
    printf("  - Tensor Core (第一代)             : %s\n",
           (prop.major >= 7) ? "支持" : "不支持");
    printf("  - 独立线程调度                     : %s\n",
           (prop.major >= 7) ? "支持" : "不支持");
    printf("  - Tensor Core (第三代, Ampere)     : %s\n",
           (prop.major >= 8) ? "支持" : "不支持");
    printf("  - FP8 / FP4 支持 (Blackwell)       : %s\n",
           (prop.major >= 10) ? "支持" : "不支持");
}

/* ============================================================
 *  内存带宽基准测试
 * ============================================================ */

/**
 * @brief  内存拷贝基准测试 Kernel
 * @details 简单的设备内内存拷贝，用于测量显存带宽
 */
__global__ void bandwidth_test_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        d_out[tid] = d_in[tid];
    }
}

/**
 * @brief  运行显存带宽基准测试
 * @return void
 */
void run_bandwidth_benchmark(void)
{
    printf("\n=============================================================\n");
    printf("     显存带宽基准测试                                         \n");
    printf("=============================================================\n");

    // 测试不同数据大小
    int sizes_mb[] = {16, 64, 256, 1024};
    int num_sizes = sizeof(sizes_mb) / sizeof(sizes_mb[0]);

    printf("\n  数据大小    |  H2D 带宽    |  D2H 带宽    |  设备内拷贝  \n");
    printf("  ------------|--------------|--------------|--------------\n");

    for (int i = 0; i < num_sizes; i++)
    {
        int size_mb = sizes_mb[i];
        int num_elements = (size_mb * 1024 * 1024) / sizeof(float);
        size_t bytes = num_elements * sizeof(float);

        float* h_src = (float*)malloc(bytes);
        float* h_dst = (float*)malloc(bytes);
        float* d_src = NULL;
        float* d_dst = NULL;

        CUDA_SAFE_CALL(cudaMalloc((void**)&d_src, bytes));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_dst, bytes));

        // 初始化数据
        for (int j = 0; j < num_elements; j++) h_src[j] = (float)j;

        // H2D 带宽测试
        GpuTimer timer;
        gpu_timer_init(&timer);
        gpu_timer_start(&timer);
        CUDA_SAFE_CALL(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
        gpu_timer_stop(&timer);
        double h2d_ms = gpu_timer_get_elapsed_ms(&timer);
        double h2d_bw = (bytes / (1024.0 * 1024.0 * 1024.0)) / (h2d_ms / 1000.0);

        // D2H 带宽测试
        gpu_timer_start(&timer);
        CUDA_SAFE_CALL(cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost));
        gpu_timer_stop(&timer);
        double d2h_ms = gpu_timer_get_elapsed_ms(&timer);
        double d2h_bw = (bytes / (1024.0 * 1024.0 * 1024.0)) / (d2h_ms / 1000.0);

        // 设备内拷贝带宽测试
        int threads = 256;
        int blocks = (num_elements + threads - 1) / threads;
        gpu_timer_start(&timer);
        bandwidth_test_kernel<<<blocks, threads>>>(d_src, d_dst, num_elements);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        gpu_timer_stop(&timer);
        double device_ms = gpu_timer_get_elapsed_ms(&timer);
        double device_bw = (bytes * 2.0 / (1024.0 * 1024.0 * 1024.0)) / (device_ms / 1000.0);

        printf("  %6d MB   | %8.2f GB/s | %8.2f GB/s | %8.2f GB/s\n",
               size_mb, h2d_bw, d2h_bw, device_bw);

        gpu_timer_destroy(&timer);
        free(h_src); free(h_dst);
        CUDA_SAFE_CALL(cudaFree(d_src));
        CUDA_SAFE_CALL(cudaFree(d_dst));
    }
}

/* ============================================================
 *  主函数
 * ============================================================ */

int main(void)
{
    printf("\n");
    printf("=============================================================\n");
    printf("     Day 5: GPU 设备查询与性能基准                            \n");
    printf("=============================================================\n\n");

    PRINT_PLATFORM_INFO();
    printf("\n");

    /* --------------------------------------------------------
     *  查询 CUDA 设备数量
     * -------------------------------------------------------- */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess)
    {
        printf("[错误] 无法获取 CUDA 设备数量: %s\n", cudaGetErrorString(err));
        printf("[提示] 可能原因:\n");
        printf("  1. NVIDIA 显卡驱动未安装或版本过旧\n");
        printf("  2. 系统中没有 NVIDIA GPU\n");
        printf("  3. CUDA Toolkit 安装不完整\n");
        printf("\n[提示] macOS 用户请使用 Python 版本或通过 Google Colab 运行。\n");
        return EXIT_FAILURE;
    }

    printf("[系统] 检测到 %d 个 CUDA 设备\n\n", device_count);

    if (device_count == 0)
    {
        printf("[警告] 系统中未检测到 NVIDIA GPU。\n");
        printf("[提示] 您仍然可以学习 CUDA 编程概念，但无法运行 GPU 代码。\n");
        printf("[提示] 推荐使用 Google Colab 获得免费 GPU。\n");
        return 0;
    }

    /* --------------------------------------------------------
     *  遍历并打印每个设备的信息
     * -------------------------------------------------------- */
    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, i));
        print_device_details(prop, i);
    }

    /* --------------------------------------------------------
     *  运行基准测试
     * -------------------------------------------------------- */
    run_bandwidth_benchmark();

    /* --------------------------------------------------------
     *  知识点总结
     * -------------------------------------------------------- */
    printf("\n");
    printf("=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. cudaGetDeviceCount: 获取系统中 CUDA 设备数量\n");
    printf("  2. cudaGetDeviceProperties: 获取设备详细属性\n");
    printf("  3. Compute Capability: 决定支持的 CUDA 功能\n");
    printf("  4. SM (Streaming Multiprocessor): GPU 的核心计算单元\n");
    printf("  5. Warp: 32 个线程，GPU 调度的基本单位\n");
    printf("  6. 显存带宽 = 2 * 显存频率 * 总线宽度 / 8\n");
    printf("  7. 理论峰值算力 = SM数 * 每SM核心数 * 2 * 频率\n");
    printf("=============================================================\n");
    printf("\n");

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
