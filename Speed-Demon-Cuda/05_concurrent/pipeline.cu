/**
 * @file pipeline.cu
 * @brief CUDA 流水线示例
 *        演示多阶段流水线处理，将传输、计算、传输回完全重叠
 */

#include "../common/cuda_utils.h"
#include "../common/benchmark.h"
#include <stdio.h>

// 向量大小
const int N = 1 << 23;  // 8M 元素

// 流水线配置
const int NUM_STREAMS = 4;  // 流水线阶段数

/**
 * @brief 向量加法核函数（计算密集型）
 */
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 100; i++) {
            sum += a[idx] + b[idx];
        }
        c[idx] = sum / 100.0f;
    }
}

/**
 * @brief 版本 1：完全串行处理
 *        所有数据一次性传输、计算、传输回
 */
void serial_processing(const float* h_a, const float* h_b, float* h_c) {
    float *d_a, *d_b, *d_c;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // 同步传输所有数据
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // 核函数执行
    dim3 block_dim(256);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);
    vector_add<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());

    // 同步传输结果回主机
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

/**
 * @brief 版本 2：2 阶段流水线
 *        将数据分成 2 块，重叠传输和计算
 */
void pipeline_2stage(const float* h_a, const float* h_b, float* h_c) {
    const int chunk_size = N / 2;
    float *d_a[2], *d_b[2], *d_c[2];
    cudaStream_t streams[2];

    // 分配设备内存和流
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc(&d_a[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    dim3 block_dim(256);
    dim3 grid_dim((chunk_size + block_dim.x - 1) / block_dim.x);

    // ========== 阶段 1：传输 chunk 0 ==========
    CUDA_CHECK(cudaMemcpyAsync(d_a[0], h_a, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_b[0], h_b, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    // ========== 阶段 2：传输 chunk 1 & 计算 chunk 0 ==========
    // 在 stream1 中传输 chunk 1
    CUDA_CHECK(cudaMemcpyAsync(d_a[1], h_a + chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[1]));
    CUDA_CHECK(cudaMemcpyAsync(d_b[1], h_b + chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[1]));

    // 在 stream0 中计算 chunk 0
    vector_add<<<grid_dim, block_dim, 0, streams[0]>>>(d_a[0], d_b[0], d_c[0], chunk_size);
    CUDA_CHECK(cudaGetLastError());

    // 等待两个流都完成
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    // ========== 阶段 3：计算 chunk 1 & 传输回 chunk 0 ==========
    // 在 stream0 中传输回 chunk 0 的结果
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c[0], chunk_size * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[0]));

    // 在 stream1 中计算 chunk 1
    vector_add<<<grid_dim, block_dim, 0, streams[1]>>>(d_a[1], d_b[1], d_c[1], chunk_size);
    CUDA_CHECK(cudaGetLastError());

    // 等待两个流都完成
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    // ========== 阶段 4：传输回 chunk 1 的结果 ==========
    CUDA_CHECK(cudaMemcpyAsync(h_c + chunk_size, d_c[1], chunk_size * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[1]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    // 释放资源
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFree(d_a[i]));
        CUDA_CHECK(cudaFree(d_b[i]));
        CUDA_CHECK(cudaFree(d_c[i]));
    }
}

/**
 * @brief 版本 3：4 阶段流水线（完全流水线）
 *        将数据分成 4 块，实现最大程度的重叠
 */
void pipeline_4stage(const float* h_a, const float* h_b, float* h_c) {
    const int num_chunks = 4;
    const int chunk_size = N / num_chunks;
    float *d_a[num_chunks], *d_b[num_chunks], *d_c[num_chunks];
    cudaStream_t streams[num_chunks];

    // 分配设备内存和流
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaMalloc(&d_a[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    dim3 block_dim(256);
    dim3 grid_dim((chunk_size + block_dim.x - 1) / block_dim.x);

    // ========== 流水线填充阶段 ==========
    // 阶段 0：传输 chunk 0
    CUDA_CHECK(cudaMemcpyAsync(d_a[0], h_a, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_b[0], h_b, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));

    // 阶段 1：计算 chunk 0, 传输 chunk 1
    vector_add<<<grid_dim, block_dim, 0, streams[0]>>>(d_a[0], d_b[0], d_c[0], chunk_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(d_a[1], h_a + chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[1]));
    CUDA_CHECK(cudaMemcpyAsync(d_b[1], h_b + chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[1]));

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));

    // 阶段 2：传输回 chunk 0, 计算 chunk 1, 传输 chunk 2
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c[0], chunk_size * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[0]));

    vector_add<<<grid_dim, block_dim, 0, streams[1]>>>(d_a[1], d_b[1], d_c[1], chunk_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(d_a[2], h_a + 2 * chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[2]));
    CUDA_CHECK(cudaMemcpyAsync(d_b[2], h_b + 2 * chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[2]));

    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));

    // ========== 流水线满负荷阶段 ==========
    // 阶段 3：传输回 chunk 1, 计算 chunk 2, 传输 chunk 3
    CUDA_CHECK(cudaMemcpyAsync(h_c + chunk_size, d_c[1], chunk_size * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[1]));

    vector_add<<<grid_dim, block_dim, 0, streams[2]>>>(d_a[2], d_b[2], d_c[2], chunk_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(d_a[3], h_a + 3 * chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[3]));
    CUDA_CHECK(cudaMemcpyAsync(d_b[3], h_b + 3 * chunk_size, chunk_size * sizeof(float),
                                cudaMemcpyHostToDevice, streams[3]));

    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    CUDA_CHECK(cudaStreamSynchronize(streams[2]));
    CUDA_CHECK(cudaStreamSynchronize(streams[3]));

    // ========== 流水线排空阶段 ==========
    // 阶段 4：传输回 chunk 2, 计算 chunk 3
    CUDA_CHECK(cudaMemcpyAsync(h_c + 2 * chunk_size, d_c[2], chunk_size * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[2]));

    vector_add<<<grid_dim, block_dim, 0, streams[3]>>>(d_a[3], d_b[3], d_c[3], chunk_size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(streams[2]));
    CUDA_CHECK(cudaStreamSynchronize(streams[3]));

    // 阶段 5：传输回 chunk 3
    CUDA_CHECK(cudaMemcpyAsync(h_c + 3 * chunk_size, d_c[3], chunk_size * sizeof(float),
                                cudaMemcpyDeviceToHost, streams[3]));
    CUDA_CHECK(cudaStreamSynchronize(streams[3]));

    // 释放资源
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFree(d_a[i]));
        CUDA_CHECK(cudaFree(d_b[i]));
        CUDA_CHECK(cudaFree(d_c[i]));
    }
}

/**
 * @brief 版本 4：通用流水线实现（可配置阶段数）
 *        使用循环实现通用的 N 阶段流水线
 */
void pipeline_generic(const float* h_a, const float* h_b, float* h_c) {
    const int num_chunks = NUM_STREAMS;
    const int chunk_size = N / num_chunks;
    float *d_a[num_chunks], *d_b[num_chunks], *d_c[num_chunks];
    cudaStream_t streams[num_chunks];

    // 分配设备内存和流
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaMalloc(&d_a[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    dim3 block_dim(256);
    dim3 grid_dim((chunk_size + block_dim.x - 1) / block_dim.x);

    // ========== 流水线填充阶段 ==========
    for (int step = 0; step < num_chunks; step++) {
        // 传输 step 到设备
        CUDA_CHECK(cudaMemcpyAsync(d_a[step], h_a + step * chunk_size,
                                    chunk_size * sizeof(float),
                                    cudaMemcpyHostToDevice, streams[step]));
        CUDA_CHECK(cudaMemcpyAsync(d_b[step], h_b + step * chunk_size,
                                    chunk_size * sizeof(float),
                                    cudaMemcpyHostToDevice, streams[step]));

        // 计算前面的 step
        for (int i = 0; i < step; i++) {
            vector_add<<<grid_dim, block_dim, 0, streams[i]>>>(
                d_a[i], d_b[i], d_c[i], chunk_size);
            CUDA_CHECK(cudaGetLastError());
        }

        // 同步所有相关流
        for (int i = 0; i <= step; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    // ========== 流水线排空阶段 ==========
    for (int step = 0; step < num_chunks; step++) {
        // 传输回 step 的结果
        CUDA_CHECK(cudaMemcpyAsync(h_c + step * chunk_size, d_c[step],
                                    chunk_size * sizeof(float),
                                    cudaMemcpyDeviceToHost, streams[step]));

        // 计算后面的 step
        for (int i = step + 1; i < num_chunks; i++) {
            vector_add<<<grid_dim, block_dim, 0, streams[i]>>>(
                d_a[i], d_b[i], d_c[i], chunk_size);
            CUDA_CHECK(cudaGetLastError());
        }

        // 同步所有相关流
        for (int i = step; i < num_chunks; i++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    // 释放资源
    for (int i = 0; i < num_chunks; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaFree(d_a[i]));
        CUDA_CHECK(cudaFree(d_b[i]));
        CUDA_CHECK(cudaFree(d_c[i]));
    }
}

int main() {
    printf("=== CUDA 流水线处理示例 ===\n\n");

    // 打印设备信息
    print_device_info();

    // 分配 Pinned 主机内存
    float *h_a, *h_b, *h_c1, *h_c2, *h_c3, *h_c4;
    CUDA_CHECK(cudaMallocHost(&h_a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c1, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c2, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c3, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c4, N * sizeof(float)));

    // 初始化数据
    printf("\n初始化数据...\n");
    srand(42);
    initialize_array(h_a, N, 0.1f, 0.9f);
    initialize_array(h_b, N, 0.1f, 0.9f);

    printf("向量大小: %d 个浮点数 (%.2f MB)\n",
           N, (N * sizeof(float)) / (1024.0 * 1024.0));
    printf("流水线阶段数: %d\n", NUM_STREAMS);

    // 创建基准测试对象
    Benchmark bench(2, 10);
    size_t data_size = N * sizeof(float) * 2;
    size_t flops = (size_t)N * 200;

    // 运行版本 1：完全串行
    printf("\n--- 版本 1：完全串行处理 ---\n");
    bench.run("完全串行", [&]() {
        serial_processing(h_a, h_b, h_c1);
    }, data_size, flops);

    // 运行版本 2：2 阶段流水线
    printf("\n--- 版本 2：2 阶段流水线 ---\n");
    bench.run("2阶段流水线", [&]() {
        pipeline_2stage(h_a, h_b, h_c2);
    }, data_size, flops);

    // 运行版本 3：4 阶段流水线
    printf("\n--- 版本 3：4 阶段流水线 ---\n");
    bench.run("4阶段流水线", [&]() {
        pipeline_4stage(h_a, h_b, h_c3);
    }, data_size, flops);

    // 运行版本 4：通用流水线
    printf("\n--- 版本 4：%d 阶段通用流水线 ---\n", NUM_STREAMS);
    bench.run("通用流水线", [&]() {
        pipeline_generic(h_a, h_b, h_c4);
    }, data_size, flops);

    // 验证结果
    printf("\n验证结果...\n");
    bool correct2 = verify_results(h_c1, h_c2, N, 1e-3f);
    bool correct3 = verify_results(h_c1, h_c3, N, 1e-3f);
    bool correct4 = verify_results(h_c1, h_c4, N, 1e-3f);

    if (!correct2 || !correct3 || !correct4) {
        printf("错误：部分结果验证失败！\n");
    }

    // 打印汇总报告
    bench.print_summary();

    // 分析性能提升
    auto results = bench.get_results();
    if (results.size() >= 4) {
        float base_time = results[0].avg_time_ms;
        printf("\n性能分析（以完全串行为基准）:\n");
        for (int i = 0; i < results.size(); i++) {
            float speedup = base_time / results[i].avg_time_ms;
            printf("  %s: 加速比 = %.2fx\n", results[i].name.c_str(), speedup);
        }

        float pipeline_4stage_speedup = base_time / results[2].avg_time_ms;
        float efficiency = (pipeline_4stage_speedup / 3.0f) * 100.0f;
        printf("\n关键观察:\n");
        printf("  4 阶段流水线理论最大加速比: ~3.0x\n");
        printf("  4 阶段流水线实际加速比: %.2fx\n", pipeline_4stage_speedup);
        printf("  流水线效率: %.1f%%\n", efficiency);
    }

    // 释放主机内存
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c1));
    CUDA_CHECK(cudaFreeHost(h_c2));
    CUDA_CHECK(cudaFreeHost(h_c3));
    CUDA_CHECK(cudaFreeHost(h_c4));

    // 重置设备
    CUDA_CHECK(cudaDeviceReset());

    printf("\n程序执行完成。\n");
    return 0;
}
