/******************************************************************************
 * @file    main.cu
 * @brief   Day 33: CUDA IPC - 进程间共享 GPU 显存
 * @details 演示 CUDA IPC（Inter-Process Communication，进程间通信）机制，
 *          用于在【不同进程】之间共享同一块 GPU 显存，避免昂贵的
 *          "设备 -> 主机 -> 设备" 数据拷贝。
 *
 *          真实的 IPC 场景需要两个独立进程：
 *          - 进程 A（生产者）：分配显存并写入数据，导出 IPC 句柄
 *          - 进程 B（消费者）：接收句柄，打开同一块显存并读取
 *          句柄通常通过共享内存、Unix Domain Socket、命名管道等方式传递。
 *
 *          本示例在【单进程】内模拟完整 IPC 流程（教学目的），核心 API：
 *          - cudaIpcGetMemHandle()   : 从设备指针获取可共享的内存句柄
 *          - cudaIpcOpenMemHandle()  : 从句柄映射出本进程可用的设备指针
 *          - cudaIpcCloseMemHandle() : 关闭（取消映射）打开的共享内存
 *          - cudaIpcGetEventHandle() / cudaIpcOpenEventHandle() : IPC 事件同步
 *
 *          同时演示（如存在多 GPU）：
 *          - cudaMemcpyPeer : 同一进程内不同设备之间的直接拷贝
 *
 * @note    IPC 内存句柄 (cudaIpcMemHandle_t) 是一段固定长度的字节序列，
 *          可以直接按字节序列化后通过任意 IPC 通道传给另一个进程。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define ARRAY_SIZE (1024 * 1024)   // 共享数组元素个数（约 4 MB float 数据）
#define BLOCK_SIZE 256
#define MAGIC_VALUE 3.14159f       // 生产者写入的校验值基准

/**
 * @brief  生产者 Kernel：向共享显存写入可校验的数据
 * @param  data  设备指针（指向将被共享的显存）
 * @param  n     元素个数
 *
 * @details 写入 data[i] = MAGIC_VALUE + i，方便消费者端逐元素校验。
 */
__global__ void producer_kernel(float* data, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        data[tid] = MAGIC_VALUE + (float)tid;
    }
}

/**
 * @brief  消费者 Kernel：对共享显存中的数据做原地变换（每个元素 *2）
 * @param  data  设备指针（通过 IPC 句柄映射得到，指向同一块物理显存）
 * @param  n     元素个数
 *
 * @details 消费者的写入会直接反映在生产者看到的同一块显存上，
 *          这正是 IPC 共享内存 "零拷贝" 的价值所在。
 */
__global__ void consumer_kernel(float* data, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        data[tid] = data[tid] * 2.0f;
    }
}

/**
 * @brief  以十六进制打印 IPC 句柄的前若干字节，帮助理解句柄的 "序列化" 本质
 * @param  handle  IPC 内存句柄
 * @param  n_bytes 打印的字节数
 */
static void print_ipc_handle(const cudaIpcMemHandle_t& handle, int n_bytes)
{
    const unsigned char* bytes = (const unsigned char*)&handle;
    printf("  IPC 句柄 (前 %d 字节): ", n_bytes);
    for (int i = 0; i < n_bytes; i++)
    {
        printf("%02x ", bytes[i]);
    }
    printf("...\n");
    printf("  IPC 句柄总大小: %zu 字节\n", sizeof(cudaIpcMemHandle_t));
}

/**
 * @brief  演示 IPC 事件同步机制（单进程模拟）
 *
 * @details 真实场景中，进程 A 在流上记录一个 IPC 事件并导出句柄，
 *          进程 B 打开该事件句柄后可以用 cudaStreamWaitEvent 等待 A 的工作完成，
 *          从而实现跨进程的 GPU 端同步（无需 CPU 介入）。
 */
static void demo_ipc_event(void)
{
    printf("\n-------------------------------------------------------------\n");
    printf("[演示 3] IPC 事件同步 (cudaIpcGetEventHandle)\n");
    printf("-------------------------------------------------------------\n");

    // 创建带 IPC 能力的事件（必须同时指定 cudaEventInterprocess 与 cudaEventDisableTiming）
    cudaEvent_t event;
    CUDA_SAFE_CALL(cudaEventCreateWithFlags(
        &event, cudaEventInterprocess | cudaEventDisableTiming));

    // 生产者端：导出事件句柄
    cudaIpcEventHandle_t event_handle;
    CUDA_SAFE_CALL(cudaIpcGetEventHandle(&event_handle, event));
    printf("  [生产者] 已导出 IPC 事件句柄 (大小 %zu 字节)\n", sizeof(cudaIpcEventHandle_t));

    // 消费者端：打开事件句柄（单进程中得到的是指向同一事件的句柄）
    cudaEvent_t imported_event;
    CUDA_SAFE_CALL(cudaIpcOpenEventHandle(&imported_event, event_handle));
    printf("  [消费者] 已打开 IPC 事件句柄，可用于 cudaStreamWaitEvent 跨进程同步\n");

    // 在默认流记录事件，并同步等待
    CUDA_SAFE_CALL(cudaEventRecord(event, 0));
    CUDA_SAFE_CALL(cudaEventSynchronize(event));
    printf("  [同步] 事件已触发，跨进程同步流程演示完成\n");

    CUDA_SAFE_CALL(cudaEventDestroy(imported_event));
    CUDA_SAFE_CALL(cudaEventDestroy(event));
}

/**
 * @brief  演示 cudaMemcpyPeer（同一进程内多 GPU 之间的直接拷贝）
 * @param  device_count 系统中的 GPU 数量
 */
static void demo_peer_copy(int device_count)
{
    printf("\n-------------------------------------------------------------\n");
    printf("[演示 2] 多 GPU 间点对点拷贝 (cudaMemcpyPeer)\n");
    printf("-------------------------------------------------------------\n");

    if (device_count < 2)
    {
        printf("  [提示] 仅检测到 %d 个 GPU，跳过 P2P 拷贝演示。\n", device_count);
        printf("  [提示] cudaMemcpyPeer 需要至少 2 个物理 GPU 才能体现价值。\n");
        return;
    }

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);

    // 在 GPU 0 上分配并填充数据
    CUDA_SAFE_CALL(cudaSetDevice(0));
    float* d0 = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d0, bytes));
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    producer_kernel<<<grid, BLOCK_SIZE>>>(d0, n);
    CHECK_KERNEL_ERROR();

    // 在 GPU 1 上分配目标缓冲
    CUDA_SAFE_CALL(cudaSetDevice(1));
    float* d1 = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d1, bytes));

    // 尝试开启 P2P 访问（若硬件支持）
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, 1, 0);
    if (can_access)
    {
        cudaDeviceEnablePeerAccess(0, 0);
    }

    // 直接从 GPU 0 拷贝到 GPU 1，无需经过主机内存
    CUDA_SAFE_CALL(cudaMemcpyPeer(d1, 1, d0, 0, bytes));
    printf("  [P2P] 已将 %.2f MB 数据从 GPU 0 直接拷贝到 GPU 1\n",
           bytes / (1024.0 * 1024.0));
    printf("  [P2P] P2P 直连访问: %s\n", can_access ? "支持 " : "不支持（走 PCIe 中转） ");

    cudaSetDevice(0);
    cudaFree(d0);
    cudaSetDevice(1);
    cudaFree(d1);
    cudaSetDevice(0);
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 33: CUDA IPC - 进程间共享 GPU 显存                 \n");
    printf("=============================================================\n\n");

    // 打印 GPU 设备信息
    print_device_info();

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    int n = ARRAY_SIZE;
    size_t bytes = n * sizeof(float);
    printf("\n[主机] 共享数组元素: %d 个 (%.2f MB)\n", n, bytes / (1024.0 * 1024.0));

    // ============================================================
    // 演示 1：IPC 共享显存的完整流程（单进程模拟）
    // ============================================================
    printf("\n-------------------------------------------------------------\n");
    printf("[演示 1] IPC 共享显存 (Get/Open/Close MemHandle)\n");
    printf("-------------------------------------------------------------\n");

    // ---- 生产者：分配显存并写入数据 ----
    float* d_producer = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_producer, bytes));

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    producer_kernel<<<grid, BLOCK_SIZE>>>(d_producer, n);
    CHECK_KERNEL_ERROR();
    printf("  [生产者] 已在设备指针 %p 写入数据\n", (void*)d_producer);

    // ---- 生产者：导出 IPC 内存句柄（这一步是序列化的关键）----
    cudaIpcMemHandle_t mem_handle;
    CUDA_SAFE_CALL(cudaIpcGetMemHandle(&mem_handle, d_producer));
    printf("  [生产者] 已通过 cudaIpcGetMemHandle 导出句柄:\n");
    print_ipc_handle(mem_handle, 8);

    // ==== 此处模拟句柄通过 socket / 共享内存 传递给另一个进程 ====
    // 真实场景：另一个进程收到这段字节后，反序列化成 cudaIpcMemHandle_t
    printf("  [传递] 句柄经由 IPC 通道（socket/共享内存）传给消费者进程...\n");

    // ---- 消费者：打开 IPC 句柄，得到本进程可用的设备指针 ----
    float* d_consumer = nullptr;
    CUDA_SAFE_CALL(cudaIpcOpenMemHandle(
        (void**)&d_consumer, mem_handle, cudaIpcMemLazyEnablePeerAccess));
    printf("  [消费者] 已通过 cudaIpcOpenMemHandle 映射到设备指针 %p\n", (void*)d_consumer);
    printf("  [消费者] 注意：该指针与生产者指向【同一块物理显存】\n");

    // ---- 消费者：对共享显存做原地变换（*2）----
    consumer_kernel<<<grid, BLOCK_SIZE>>>(d_consumer, n);
    CHECK_KERNEL_ERROR();
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // ---- 校验：从生产者指针读回，验证消费者的修改可见 ----
    float* h_result = (float*)malloc(bytes);
    CUDA_SAFE_CALL(cudaMemcpy(h_result, d_producer, bytes, cudaMemcpyDeviceToHost));

    // 期望值 = (MAGIC_VALUE + i) * 2
    int mismatch = 0;
    for (int i = 0; i < n; i++)
    {
        float expected = (MAGIC_VALUE + (float)i) * 2.0f;
        if (fabsf(h_result[i] - expected) > 1e-2f)
        {
            if (mismatch < 5)
            {
                printf("  [不匹配] 索引 %d: 实际=%.4f, 期望=%.4f\n", i, h_result[i], expected);
            }
            mismatch++;
        }
    }
    int is_ok = (mismatch == 0);
    printf("  [校验] 消费者写入对生产者可见: %s\n", is_ok ? "通过 " : "失败 ");

    // ---- 消费者：关闭 IPC 句柄 ----
    CUDA_SAFE_CALL(cudaIpcCloseMemHandle(d_consumer));
    printf("  [消费者] 已通过 cudaIpcCloseMemHandle 关闭共享内存映射\n");

    // ============================================================
    // 演示 2：多 GPU P2P 拷贝
    // ============================================================
    demo_peer_copy(device_count);

    // ============================================================
    // 演示 3：IPC 事件同步
    // ============================================================
    demo_ipc_event();

    // ============================================================
    // 总结
    // ============================================================
    printf("\n=============================================================\n");
    printf("                        性能与要点总结                       \n");
    printf("=============================================================\n");
    printf("  %-30s | %-12s\n", "演示项 ", "状态 ");
    printf("  ------------------------------ | ------------\n");
    printf("  %-30s | %10s\n", "IPC 共享显存读写校验", is_ok ? "通过 " : "失败 ");
    printf("  %-30s | %10s\n", "多 GPU P2P 拷贝", device_count >= 2 ? "已演示 " : "跳过(单GPU) ");
    printf("  %-30s | %10s\n", "IPC 事件同步", "已演示 ");
    printf("=============================================================\n");
    printf("  关键教学点:\n");
    printf("  1. cudaIpcGetMemHandle: 导出可跨进程共享的显存句柄 \n");
    printf("  2. cudaIpcOpenMemHandle: 另一进程映射到同一块物理显存 \n");
    printf("  3. cudaIpcCloseMemHandle: 用完及时关闭映射 \n");
    printf("  4. IPC 句柄本质是定长字节序列，可经 socket/共享内存 传递 \n");
    printf("  5. IPC 事件用于跨进程 GPU 端同步（无需 CPU 介入） \n");
    printf("  6. 应用场景: 分布式训练、多进程推理服务、零拷贝数据管道 \n");
    printf("=============================================================\n\n");

    // ---- 清理 ----
    cudaFree(d_producer);
    free(h_result);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}