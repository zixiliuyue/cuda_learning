/******************************************************************************
 * @file    cuda_utils.h
 * @brief   CUDA 错误检查与辅助工具宏
 * @details 提供统一的 CUDA API 错误检查封装，帮助初学者快速定位 GPU 程序中的错误。
 *          所有 CUDA 函数调用都建议通过 CUDA_SAFE_CALL 宏包裹，这样在发生错误时
 *          可以自动输出文件名、行号和错误描述信息。
 *
 * @note    此头文件被所有 Day 项目共享，是跨项目的基础工具库
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  将 CUDA 错误码转换为可读字符串并输出到 stderr
 * @param  error   CUDA 错误码（cudaError_t 类型）
 * @param  file    发生错误的源文件名（通常用 __FILE__ 宏传入）
 * @param  line    发生错误的行号（通常用 __LINE__ 宏传入）
 * @return void
 *
 * @usage  此函数通常不直接调用，而是通过 CUDA_SAFE_CALL 宏自动调用。
 *         当 CUDA API 返回非 cudaSuccess 的错误码时，会打印如下格式信息：
 *         CUDA error at [文件名]:[行号] - [错误描述] (错误码数值)
 */
static inline void check_cuda_error(cudaError_t error, const char* file, int line)
{
    // 如果错误码不是 cudaSuccess（即 0），说明 CUDA API 调用失败
    if (error != cudaSuccess)
    {
        // 输出错误信息到标准错误流
        // cudaGetErrorString(error) 将错误码转换为人类可读字符串
        fprintf(stderr,
                "[CUDA 错误] 文件: %s | 行号: %d | 错误码: %d | 描述: %s\n",
                file, line, (int)error, cudaGetErrorString(error));

        // 刷新标准错误流，确保错误信息立即显示
        fflush(stderr);

        // 终止程序运行，防止错误状态继续传播导致难以调试的问题
        // 在生产环境中，可以考虑使用异常代替 exit
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief  CUDA API 调用安全封装宏
 * @param  call  任意返回 cudaError_t 的 CUDA API 调用表达式
 *
 * @details 此宏会自动将 CUDA API 调用包装在 do-while(0) 结构中，
 *          确保宏在任何上下文中都能安全使用（如 if-else 语句中）。
 *          如果调用失败，会立即输出错误信息并终止程序。
 *
 * @example 使用示例：
 *          CUDA_SAFE_CALL(cudaMalloc(&d_ptr, size));
 *          CUDA_SAFE_CALL(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
 */
#define CUDA_SAFE_CALL(call)                                              \
    do {                                                                  \
        cudaError_t __cuda_err = (call);                                  \
        check_cuda_error(__cuda_err, __FILE__, __LINE__);                 \
    } while (0)

/**
 * @brief  同步设备并检查 Kernel 执行错误
 * @param  file    发生错误的源文件名
 * @param  line    发生错误的行号
 * @return void
 *
 * @details CUDA Kernel 的启动是异步的，错误可能在后续同步时才被发现。
 *          此函数调用 cudaDeviceSynchronize() 强制等待所有 Kernel 完成，
 *          然后检查是否有异步错误发生。
 *
 * @note   在调试阶段建议每次 Kernel 启动后都调用此函数，
 *         但在发布版本中应减少同步以提高性能。
 */
static inline void check_kernel_error(const char* file, int line)
{
    // 同步设备：阻塞主机线程，直到设备上所有之前的 CUDA 调用完成
    cudaError_t err = cudaDeviceSynchronize();

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "[Kernel 错误] 文件: %s | 行号: %d | 错误码: %d | 描述: %s\n",
                file, line, (int)err, cudaGetErrorString(err));
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief  Kernel 启动后同步并检查错误的便捷宏
 * @usage  在 Kernel 启动后使用：
 *         my_kernel<<<grid, block>>>(args);
 *         CHECK_KERNEL_ERROR();
 */
#define CHECK_KERNEL_ERROR() check_kernel_error(__FILE__, __LINE__)

/**
 * @brief  打印当前 CUDA 设备的简要信息
 * @return void
 *
 * @details 输出设备名称、计算能力、总显存大小等关键信息，
 *          方便在程序开始时确认运行环境。
 */
static inline void print_device_info(void)
{
    int device_count = 0;

    // 获取系统中可用的 CUDA 设备数量
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        printf("[设备信息] 无法获取 CUDA 设备数量: %s\n", cudaGetErrorString(err));
        return;
    }

    if (device_count == 0)
    {
        printf("[设备信息] 系统中未检测到 NVIDIA GPU 设备！\n");
        printf("[设备信息] 如果您在 macOS 上运行，请使用 Python 版本或通过 Google Colab 运行。\n");
        return;
    }

    // 获取当前活动的 CUDA 设备编号（默认是 0）
    int current_device = 0;
    cudaGetDevice(&current_device);

    // 查询该设备的详细属性
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, current_device));

    printf("=================================================================\n");
    printf("                     CUDA 设备信息摘要                          \n");
    printf("=================================================================\n");
    printf("  设备编号       : %d / %d\n", current_device, device_count - 1);
    printf("  设备名称       : %s\n", prop.name);
    printf("  计算能力       : %d.%d\n", prop.major, prop.minor);
    printf("  总显存         : %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  多处理器(SM)数 : %d\n", prop.multiProcessorCount);
    printf("  最大线程/块    : %d\n", prop.maxThreadsPerBlock);
    printf("  最大块维度     : (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  最大网格维度   : (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  共享内存/块    : %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  寄存器/块      : %d\n", prop.regsPerBlock);
    printf("  Warp 大小      : %d\n", prop.warpSize);

    /*
     * 显存带宽的理论值 = 2 (DDR双倍速率) * 显存频率 * 总线位宽 / 8
     *
     * 【重要兼容性说明】
     * 在 CUDA 12.x 及更早版本中，可以直接通过 prop.memoryClockRate 和
     * prop.memoryBusWidth 获取显存频率和位宽。
     * 但从 CUDA 13.0 起，这两个字段已从 cudaDeviceProp 结构体中【移除】。
     * 官方推荐改用 cudaDeviceGetAttribute() 配合属性枚举来查询，
     * 该 API 在 CUDA 11 / 12 / 13 上均可用，是跨版本最稳妥的写法。
     */
    int mem_clock_khz = 0;   // 显存时钟频率（单位：千赫兹 kHz）
    int mem_bus_width = 0;   // 显存总线位宽（单位：比特 bit）
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, current_device);
    cudaDeviceGetAttribute(&mem_bus_width, cudaDevAttrGlobalMemoryBusWidth, current_device);

    if (mem_clock_khz > 0 && mem_bus_width > 0)
    {
        // mem_clock_khz 单位为 kHz，乘以 1000 转换为 Hz
        double bandwidth_gbps = 2.0 * (double)mem_clock_khz * 1000.0 * (mem_bus_width / 8.0) / 1e9;
        printf("  显存带宽(理论): %.1f GB/s\n", bandwidth_gbps);
    }
    else
    {
        printf("  显存带宽(理论): 无法查询（驱动不支持该属性）\n");
    }
    printf("=================================================================\n");
}

#ifdef __cplusplus
}
#endif

#endif /* CUDA_UTILS_H */
