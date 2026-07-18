/******************************************************************************
 * @file    timer.h
 * @brief   CUDA GPU 事件计时器
 * @details 提供基于 CUDA Event 的高精度计时功能，用于测量 Kernel 执行时间
 *          和设备内存传输时间。计时精度可达微秒级别。
 *
 *          使用 CUDA Event 计时的优势：
 *          1. 自动处理 GPU 异步执行，确保计时的准确性
 *          2. 不受 CPU 调度影响，只测量 GPU 实际工作时间
 *          3. 精度高于 CPU 计时器（如 std::chrono）
 *
 * @note    此计时器要求系统中至少有一个可用的 CUDA 设备
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#ifndef TIMER_H
#define TIMER_H

#include <cuda_runtime.h>
#include "cuda_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct GpuTimer
 * @brief  GPU 计时器结构体
 *
 * @field  start_event   CUDA 开始事件标记
 * @field  stop_event    CUDA 结束事件标记
 * @field  is_running    标记计时器是否正在运行（防止重复启动）
 */
typedef struct {
    cudaEvent_t start_event;   /**< 记录开始时刻的 CUDA 事件 */
    cudaEvent_t stop_event;    /**< 记录结束时刻的 CUDA 事件 */
    int is_running;            /**< 计时器状态：0=停止, 1=运行中 */
} GpuTimer;

/**
 * @brief  初始化 GPU 计时器
 * @param  timer  指向 GpuTimer 结构体的指针
 * @return void
 *
 * @details 创建两个 CUDA Event 对象，用于标记开始和结束时间点。
 *          CUDA Event 是轻量级的 GPU 时间戳，创建开销极小。
 */
static inline void gpu_timer_init(GpuTimer* timer)
{
    if (timer == NULL) return;

    timer->is_running = 0;

    // 创建 CUDA Event，使用默认标志（0）
    // CUDA Event 会在创建时自动与当前设备关联
    CUDA_SAFE_CALL(cudaEventCreate(&(timer->start_event)));
    CUDA_SAFE_CALL(cudaEventCreate(&(timer->stop_event)));
}

/**
 * @brief  销毁 GPU 计时器，释放 CUDA Event 资源
 * @param  timer  指向 GpuTimer 结构体的指针
 * @return void
 *
 * @note   程序结束前应调用此函数，避免 CUDA Event 资源泄漏
 */
static inline void gpu_timer_destroy(GpuTimer* timer)
{
    if (timer == NULL) return;

    CUDA_SAFE_CALL(cudaEventDestroy(timer->start_event));
    CUDA_SAFE_CALL(cudaEventDestroy(timer->stop_event));
    timer->is_running = 0;
}

/**
 * @brief  开始计时
 * @param  timer  指向 GpuTimer 结构体的指针
 * @return void
 *
 * @details 在当前 CUDA 流中记录一个开始事件。
 *          由于 CUDA 的异步特性，此事件会在流中之前的所有操作完成后才触发。
 */
static inline void gpu_timer_start(GpuTimer* timer)
{
    if (timer == NULL) return;

    // 标记计时器为运行状态
    timer->is_running = 1;

    // 在当前 CUDA 流中记录开始事件
    // 这会自动等待流中之前的所有操作完成
    CUDA_SAFE_CALL(cudaEventRecord(timer->start_event, 0));
}

/**
 * @brief  停止计时
 * @param  timer  指向 GpuTimer 结构体的指针
 * @return void
 */
static inline void gpu_timer_stop(GpuTimer* timer)
{
    if (timer == NULL || !timer->is_running) return;

    // 在当前 CUDA 流中记录结束事件
    CUDA_SAFE_CALL(cudaEventRecord(timer->stop_event, 0));

    timer->is_running = 0;
}

/**
 * @brief  获取计时结果（毫秒）
 * @param  timer  指向 GpuTimer 结构体的指针
 * @return double  经过的时间，单位为毫秒（ms）
 *
 * @details 此函数会自动同步两个事件，确保 GPU 上所有操作已完成后再计算时间差。
 *          如果计时器尚未停止，会先等待 stop_event 完成。
 */
static inline double gpu_timer_get_elapsed_ms(GpuTimer* timer)
{
    if (timer == NULL) return 0.0;

    float elapsed_ms = 0.0f;

    // cudaEventSynchronize 阻塞主机线程，直到 stop_event 在 GPU 上被记录
    // 这确保了我们测量的时间段内，GPU 上的所有工作都已完成
    CUDA_SAFE_CALL(cudaEventSynchronize(timer->stop_event));

    // 计算两个事件之间的时间差，单位为毫秒
    // 此函数在设备端有专门的硬件支持，精度很高
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_ms, timer->start_event, timer->stop_event));

    return (double)elapsed_ms;
}

/**
 * @brief  获取计时结果（秒）
 * @param  timer  指向 GpuTimer 结构体的指针
 * @return double  经过的时间，单位为秒（s）
 */
static inline double gpu_timer_get_elapsed_sec(GpuTimer* timer)
{
    return gpu_timer_get_elapsed_ms(timer) / 1000.0;
}

/**
 * @brief  打印计时结果（带格式化输出）
 * @param  timer       指向 GpuTimer 结构体的指针
 * @param  operation   操作描述字符串（如 "Kernel 执行 "、 "内存传输 "）
 * @return void
 *
 * @example
 *          GpuTimer timer;
 *          gpu_timer_init(&timer);
 *          gpu_timer_start(&timer);
 *          my_kernel<<<grid, block>>>(...);
 *          gpu_timer_stop(&timer);
 *          gpu_timer_print(&timer, "MyKernel");
 */
static inline void gpu_timer_print(GpuTimer* timer, const char* operation)
{
    double ms = gpu_timer_get_elapsed_ms(timer);
    printf("[计时] %-30s : %10.4f ms  (%10.6f s)\n", operation, ms, ms / 1000.0);
}

#ifdef __cplusplus
}
#endif

#endif /* TIMER_H */
