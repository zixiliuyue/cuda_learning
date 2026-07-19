/******************************************************************************
 * @file    main.cu
 * @brief   Day 39: NCCL 多 GPU AllReduce 通信实战
 * @author  CUDA 30天学习项目
 * @date     2026-07-18
 *
 * 本示例演示如何使用 NVIDIA NCCL 在单进程多 GPU 场景下执行 AllReduce（求和）。
 *
 * AllReduce 语义：
 *   假设有 K 个 GPU，每个 GPU 各持有一份长度为 N 的向量 x_i，
 *   AllReduce(sum) 之后，每个 GPU 都得到 sum(x_0 + x_1 + ... + x_{K-1})。
 *   即：所有 GPU 上的结果完全一致，且等于所有输入之和。
 *
 * 关键技术点：
 *   1. ncclUniqueId 的生成与广播（多进程场景需要用 MPI/socket 广播）
 *   2. 单进程多 GPU 使用 ncclCommInitAll 一次性初始化所有通信器
 *   3. ncclAllReduce API 的使用（数据类型、规约操作、流绑定）
 *   4. NCCL 操作必须绑定到 CUDA 流，用 cudaStreamSynchronize 等待完成
 *
 * 条件编译：
 *   - 若系统安装了 NCCL（能找到 <nccl.h>），则运行真实的 NCCL AllReduce；
 *   - 否则退化为 CPU 模拟，讲清 AllReduce 的语义与安装方法。
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ *
 * 尝试包含 NCCL 头文件。__has_include 是 C++17 / 现代编译器特性，
 * 找不到 <nccl.h> 时自动退化到 CPU 模拟路径，保证无 NCCL 也能编译。
 * ------------------------------------------------------------------ */
#if defined(__has_include)
#  if __has_include(<nccl.h>)
#    include <nccl.h>
#    ifndef HAS_NCCL
#      define HAS_NCCL 1
#    endif
#  else
#    ifndef HAS_NCCL
#      define HAS_NCCL 0
#    endif
#  endif
#else
/* 极老编译器不支持 __has_include，则依赖 CMake 传入的 -DHAS_NCCL */
#  ifndef HAS_NCCL
#    define HAS_NCCL 0
#  endif
#endif

/* ================================================================== *
 * 通用宏与工具
 * ================================================================== */

/* CUDA 运行时错误检查宏 */
#define CUDA_CHECK(cmd)                                                       \
    do {                                                                      \
        cudaError_t e = (cmd);                                                \
        if (e != cudaSuccess) {                                               \
            printf("[CUDA 错误] %s:%d '%s'\n",                                \
                   __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#if HAS_NCCL
/* NCCL 错误检查宏 */
#define NCCL_CHECK(cmd)                                                       \
    do {                                                                      \
        ncclResult_t r = (cmd);                                               \
        if (r != ncclSuccess) {                                               \
            printf("[NCCL 错误] %s:%d '%s'\n",                                \
                   __FILE__, __LINE__, ncclGetErrorString(r));                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)
#endif

#define N_ELEMENTS (1024 * 1024)   /* 每个 GPU 的向量长度（1M 个 float） */

static void print_line(void)
{
    /* 60 字符分隔线 */
    printf("------------------------------------------------------------\n");
}

/* ================================================================== *
 * CPU 参考实现：把所有 GPU 的输入求和，作为正确性基准
 * ================================================================== */
static void cpu_allreduce_reference(float** host_inputs, int num_gpus,
                                    int n, float* out_sum)
{
    for (int i = 0; i < n; ++i) {
        float acc = 0.0f;
        for (int g = 0; g < num_gpus; ++g)
            acc += host_inputs[g][i];
        out_sum[i] = acc;
    }
}

#if HAS_NCCL
/* ================================================================== *
 * 真实 NCCL AllReduce 演示（单进程多 GPU）
 * ================================================================== */
static int run_nccl_demo(int num_gpus)
{
    printf("[模式] 检测到 NCCL，执行真实的 GPU AllReduce。\n");
    printf("[配置] GPU 数量 = %d，每 GPU 向量长度 = %d（%.1f MB）\n\n",
           num_gpus, N_ELEMENTS, N_ELEMENTS * sizeof(float) / (1024.0 * 1024.0));

    int*   devs         = (int*)   malloc(sizeof(int)     * num_gpus);
    float** h_in        = (float**)malloc(sizeof(float*)  * num_gpus);
    float** d_send      = (float**)malloc(sizeof(float*)  * num_gpus);
    float** d_recv      = (float**)malloc(sizeof(float*)  * num_gpus);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * num_gpus);

    /* 每个 GPU 填充不同的值：GPU g 的元素全部初始化为 (g + 1)。
       这样 AllReduce(sum) 后每个元素应为 1+2+...+num_gpus。 */
    for (int g = 0; g < num_gpus; ++g) {
        devs[g] = g;
        CUDA_CHECK(cudaSetDevice(g));

        h_in[g] = (float*)malloc(sizeof(float) * N_ELEMENTS);
        for (int i = 0; i < N_ELEMENTS; ++i)
            h_in[g][i] = (float)(g + 1);

        CUDA_CHECK(cudaMalloc((void**)&d_send[g], sizeof(float) * N_ELEMENTS));
        CUDA_CHECK(cudaMalloc((void**)&d_recv[g], sizeof(float) * N_ELEMENTS));
        CUDA_CHECK(cudaMemcpy(d_send[g], h_in[g],
                              sizeof(float) * N_ELEMENTS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamCreate(&streams[g]));
    }

    /* --------------------------------------------------------------
     * 初始化 NCCL 通信器。
     * 单进程多 GPU：直接用 ncclCommInitAll 一次性初始化 num_gpus 个通信器，
     * 内部会自动生成并分发 ncclUniqueId，无需手动广播。
     * -------------------------------------------------------------- */
    ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * num_gpus);
    NCCL_CHECK(ncclCommInitAll(comms, num_gpus, devs));
    printf("[NCCL] ncclCommInitAll 完成，%d 个通信器就绪。\n", num_gpus);

    /* --------------------------------------------------------------
     * 计时：NCCL AllReduce。
     * 用 GPU 0 的事件计时；ncclGroupStart/End 将多 GPU 调用合并为一组，
     * 避免单进程多流互相死锁。
     * -------------------------------------------------------------- */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, streams[0]));

    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        /* ncclAllReduce(发送缓冲, 接收缓冲, 元素数, 数据类型, 规约操作, 通信器, 流) */
        NCCL_CHECK(ncclAllReduce((const void*)d_send[g],
                                 (void*)d_recv[g],
                                 N_ELEMENTS,
                                 ncclFloat,
                                 ncclSum,
                                 comms[g],
                                 streams[g]));
    }
    NCCL_CHECK(ncclGroupEnd());

    /* 等待所有 GPU 的流完成 NCCL 操作 */
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamSynchronize(streams[g]));
    }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(stop, streams[0]));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float nccl_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&nccl_ms, start, stop));

    /* --------------------------------------------------------------
     * 验证：每个 GPU 的结果应完全一致，且等于 1+2+...+num_gpus。
     * -------------------------------------------------------------- */
    float expected = 0.0f;
    for (int g = 0; g < num_gpus; ++g) expected += (float)(g + 1);

    int all_ok = 1;
    float* h_out = (float*)malloc(sizeof(float) * N_ELEMENTS);
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMemcpy(h_out, d_recv[g],
                              sizeof(float) * N_ELEMENTS, cudaMemcpyDeviceToHost));
        /* 抽样检查首、中、尾元素 */
        int idxs[3] = {0, N_ELEMENTS / 2, N_ELEMENTS - 1};
        for (int k = 0; k < 3; ++k) {
            if (h_out[idxs[k]] != expected) {
                all_ok = 0;
                printf("[验证] GPU %d 索引 %d 结果 %.1f 期望 %.1f\n",
                       g, idxs[k], h_out[idxs[k]], expected);
            }
        }
        printf("[验证] GPU %d 结果[0] = %.1f（期望 %.1f）\n",
               g, h_out[0], expected);
    }
    printf("[验证] %s\n\n", all_ok ? "全部 GPU 结果一致，AllReduce 正确！"
                                   : "结果不一致，请检查！");

    /* --------------------------------------------------------------
     * 对比：逐 GPU 拷回 CPU + CPU 求和（朴素做法）
     * -------------------------------------------------------------- */
    cudaEvent_t cstart, cstop;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventCreate(&cstart));
    CUDA_CHECK(cudaEventCreate(&cstop));
    CUDA_CHECK(cudaEventRecord(cstart, 0));

    float* cpu_sum = (float*)malloc(sizeof(float) * N_ELEMENTS);
    cpu_allreduce_reference(h_in, num_gpus, N_ELEMENTS, cpu_sum);

    CUDA_CHECK(cudaEventRecord(cstop, 0));
    CUDA_CHECK(cudaEventSynchronize(cstop));
    float cpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cpu_ms, cstart, cstop));

    /* --------------------------------------------------------------
     * 性能总结表格
     * -------------------------------------------------------------- */
    printf("=============================================================\n");
    printf("     性能对比（%d GPU，每 GPU %.1f MB）                       \n",
           num_gpus, N_ELEMENTS * sizeof(float) / (1024.0 * 1024.0));
    printf("=============================================================\n");
    printf("  %-32s | %12s\n", "方法", "耗时 (ms)");
    print_line();
    printf("  %-32s | %12.3f\n", "NCCL AllReduce (GPU)", nccl_ms);
    printf("  %-32s | %12.3f\n", "拷回 CPU + CPU 求和", cpu_ms);
    print_line();
    if (nccl_ms > 0.0f)
        printf("  加速比（CPU / NCCL）: %.2fx\n", cpu_ms / nccl_ms);
    printf("=============================================================\n\n");
    printf("  注：CPU 路径还未计入 %d 次 DtoH 拷贝的完整开销，实际差距更大。\n\n",
           num_gpus);

    /* 清理资源 */
    for (int g = 0; g < num_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        ncclCommDestroy(comms[g]);
        CUDA_CHECK(cudaStreamDestroy(streams[g]));
        CUDA_CHECK(cudaFree(d_send[g]));
        CUDA_CHECK(cudaFree(d_recv[g]));
        free(h_in[g]);
    }
    free(comms); free(devs); free(h_in); free(d_send); free(d_recv);
    free(streams); free(h_out); free(cpu_sum);
    return 0;
}
#endif /* HAS_NCCL */

/* ================================================================== *
 * CPU 模拟演示（无 NCCL 时的退化路径）
 * ================================================================== */
static int run_cpu_simulation(void)
{
    const int num_gpus = 4;    /* 逻辑上模拟 4 个 GPU */
    const int n        = 8;    /* 用小向量便于打印 */

    printf("[模式] 未检测到 NCCL，使用 CPU 模拟 AllReduce 概念。\n\n");
    print_line();
    printf(" 如需运行真实多 GPU AllReduce，请先安装 NCCL：\n");
    printf("   Linux : sudo apt install libnccl2 libnccl-dev\n");
    printf("           或 conda install -c nvidia nccl\n");
    printf("   源码  : https://github.com/NVIDIA/nccl\n");
    printf("   Windows: NCCL 官方仅支持 Linux；Windows 下可用 WSL2，\n");
    printf("            或用本项目 python/nccl_demo.py 体验概念。\n");
    print_line();
    printf("\n");

    /* 构造 num_gpus 份输入：GPU g 的元素全部为 (g + 1) */
    float** h_in = (float**)malloc(sizeof(float*) * num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        h_in[g] = (float*)malloc(sizeof(float) * n);
        for (int i = 0; i < n; ++i)
            h_in[g][i] = (float)(g + 1);
    }

    printf("[输入] 每个 GPU 持有的向量（长度 %d）:\n", n);
    for (int g = 0; g < num_gpus; ++g) {
        printf("  GPU %d: ", g);
        for (int i = 0; i < n; ++i) printf("%4.0f", h_in[g][i]);
        printf("\n");
    }
    printf("\n");

    /* CPU 版 AllReduce：等价于 NCCL 的 ncclSum 结果 */
    float* result = (float*)malloc(sizeof(float) * n);
    cpu_allreduce_reference(h_in, num_gpus, n, result);

    printf("[AllReduce(sum) 后] 每个 GPU 都得到相同的求和结果:\n");
    for (int g = 0; g < num_gpus; ++g) {
        printf("  GPU %d: ", g);
        for (int i = 0; i < n; ++i) printf("%4.0f", result[i]);
        printf("\n");
    }

    float expected = 0.0f;
    for (int g = 0; g < num_gpus; ++g) expected += (float)(g + 1);
    printf("\n[说明] 每个元素 = 1+2+...+%d = %.0f，所有 GPU 结果一致。\n\n",
           num_gpus, expected);

    /* 性能总结表格（模拟值仅作示意，展示表格格式） */
    printf("=============================================================\n");
    printf("     AllReduce 概念对比（示意）                              \n");
    printf("=============================================================\n");
    printf("  %-30s | %-22s\n", "方法", "特点");
    print_line();
    printf("  %-30s | %-22s\n", "NCCL Ring AllReduce", "带宽最优, 大数据首选 ");
    printf("  %-30s | %-22s\n", "NCCL Tree AllReduce", "延迟最优, 多节点小数据 ");
    printf("  %-30s | %-22s\n", "朴素: 拷回 CPU 求和", "PCIe 瓶颈, 串行 ");
    print_line();
    printf("=============================================================\n\n");

    for (int g = 0; g < num_gpus; ++g) free(h_in[g]);
    free(h_in); free(result);
    return 0;
}

/* ================================================================== *
 * 多进程场景说明（仅打印，不实际运行）：
 *   1. 由 rank 0 调用 ncclGetUniqueId 生成 ncclUniqueId；
 *   2. 通过 MPI_Bcast / socket 把该 id 广播给所有进程；
 *   3. 每个进程用相同 id 调用 ncclCommInitRank(&comm, nranks, id, myRank)；
 *   4. 之后每个进程各管理一个 GPU，调用 ncclAllReduce 即可。
 * ================================================================== */
static void print_multiprocess_notes(void)
{
    print_line();
    printf(" 多进程（每进程 1 GPU）初始化流程：\n");
    printf("   rank0: ncclGetUniqueId(&id)\n");
    printf("   广播 : MPI_Bcast(&id, ...) 或 socket 分发 id\n");
    printf("   各rank: ncclCommInitRank(&comm, nRanks, id, myRank)\n");
    printf("   通信 : ncclAllReduce(sendbuf, recvbuf, n, type, op, comm, stream)\n");
    print_line();
    printf("\n");
}

/* ================================================================== *
 * 主函数
 * ================================================================== */
int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 39: NCCL 多 GPU AllReduce 通信实战                   \n");
    printf("=============================================================\n\n");

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        printf("[提示] 未检测到可用 CUDA 设备：%s\n", cudaGetErrorString(err));
        device_count = 0;
    }
    printf("[系统] 检测到 %d 个 CUDA 设备\n", device_count);
    printf("[编译] HAS_NCCL = %d（%s）\n\n", HAS_NCCL,
           HAS_NCCL ? "已链接 NCCL" : "未链接 NCCL，运行 CPU 模拟");

    print_multiprocess_notes();

#if HAS_NCCL
    if (device_count >= 1) {
        /* NCCL 单进程多 GPU；只有 1 个 GPU 时 AllReduce 结果就是自身，
           仍然演示完整 API 流程。 */
        int num_gpus = device_count;
        run_nccl_demo(num_gpus);
    } else {
        printf("[提示] 没有可用 GPU，退化为 CPU 模拟。\n\n");
        run_cpu_simulation();
    }
#else
    run_cpu_simulation();
#endif

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
