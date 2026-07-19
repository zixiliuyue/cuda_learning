/**
 * @file overlap_transfer.cu
 * @brief CUDA 数据传输与计算重叠示例
 *        演示 Pinned Memory、异步传输、以及传输与计算重叠的性能
 */

#include "../common/cuda_utils.h"
#include "../common/benchmark.h"
#include <stdio.h>

// 向量大小
const int N = 1 << 22;  // 4M 元素

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
 * @brief 版本 1：使用可分页内存 + 同步传输
 *        所有操作完全串行
 */
void pageable_sync_transfer(const float* h_a, const float* h_b, float* h_c) {
    float *d_a, *d_b, *d_c;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // 同步传输数据（使用可分页内存）
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
 * @brief 版本 2：使用 Pinned Memory + 同步传输
 *        传输速度更快，但仍然串行
 */
void pinned_sync_transfer(const float* h_a, const float* h_b, float* h_c) {
    float *d_a, *d_b, *d_c;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // 同步传输数据（使用 Pinned Memory，传输更快）
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
 * @brief 版本 3：使用 Pinned Memory + 异步传输（无重叠）
 *        使用异步传输，但传输和计算仍然串行
 */
void pinned_async_no_overlap(const float* h_a, const float* h_b, float* h_c) {
    float *d_a, *d_b, *d_c;
    cudaStream_t stream;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 异步传输数据到设备
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, N * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, N * sizeof(float),
                                cudaMemcpyHostToDevice, stream));

    // 等待传输完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 核函数执行（在同一个流中）
    dim3 block_dim(256);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);
    vector_add<<<grid_dim, block_dim, 0, stream>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // 等待计算完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 异步传输结果回主机
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, N * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    // 等待传输完成
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 释放资源
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

/**
 * @brief 版本 4：使用 Pinned Memory + 传输与计算重叠
 *        将数据分成两部分：一部分在计算时，另一部分在传输
 */
void pinned_async_with_overlap(const float* h_a, const float* h_b, float* h_c) {
    const int half_n = N / 2;
    float *d_a1, *d_a2, *d_b1, *d_b2, *d_c1, *d_c2;
    cudaStream_t stream1, stream2;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a1, half_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a2, half_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, half_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, half_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1, half_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2, half_n * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    dim3 block_dim(256);
    dim3 grid_dim((half_n + block_dim.x - 1) / block_dim.x);

    // ========== 第一阶段：传输第一部分数据 ==========
    CUDA_CHECK(cudaMemcpyAsync(d_a1, h_a, half_n * sizeof(float),
                                cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_b1, h_b, half_n * sizeof(float),
                                cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // ========== 第二阶段：传输第二部分 & 计算第一部分（重叠） ==========
    // 在 stream2 中传输第二部分数据
    CUDA_CHECK(cudaMemcpyAsync(d_a2, h_a + half_n, half_n * sizeof(float),
                                cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_b2, h_b + half_n, half_n * sizeof(float),
                                cudaMemcpyHostToDevice, stream2));

    // 在 stream1 中计算第一部分数据
    vector_add<<<grid_dim, block_dim, 0, stream1>>>(d_a1, d_b1, d_c1, half_n);
    CUDA_CHECK(cudaGetLastError());

    // 等待两个流都完成
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // ========== 第三阶段：计算第二部分 & 传输第一部分结果（重叠） ==========
    // 在 stream1 中传输第一部分结果回主机
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c1, half_n * sizeof(float),
                                cudaMemcpyDeviceToHost, stream1));

    // 在 stream2 中计算第二部分数据
    vector_add<<<grid_dim, block_dim, 0, stream2>>>(d_a2, d_b2, d_c2, half_n);
    CUDA_CHECK(cudaGetLastError());

    // 等待两个流都完成
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // ========== 第四阶段：传输第二部分结果 ==========
    CUDA_CHECK(cudaMemcpyAsync(h_c + half_n, d_c2, half_n * sizeof(float),
                                cudaMemcpyDeviceToHost, stream2));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    // 释放资源
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_a2));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaFree(d_c2));
}

int main() {
    printf("=== CUDA 数据传输与计算重叠示例 ===\n\n");

    // 打印设备信息
    print_device_info();

    // 分配可分页主机内存（用于版本 1）
    float* h_a_pageable = new float[N];
    float* h_b_pageable = new float[N];
    float* h_c_pageable = new float[N];

    // 分配 Pinned 主机内存（用于版本 2, 3, 4）
    float *h_a_pinned, *h_b_pinned;
    float *h_c1, *h_c2, *h_c3, *h_c4;
    CUDA_CHECK(cudaMallocHost(&h_a_pinned, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b_pinned, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c1, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c2, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c3, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c4, N * sizeof(float)));

    // 初始化数据
    printf("\n初始化数据...\n");
    srand(42);
    initialize_array(h_a_pageable, N, 0.1f, 0.9f);
    initialize_array(h_b_pageable, N, 0.1f, 0.9f);

    // 复制数据到 Pinned Memory
    memcpy(h_a_pinned, h_a_pageable, N * sizeof(float));
    memcpy(h_b_pinned, h_b_pageable, N * sizeof(float));

    printf("向量大小: %d 个浮点数 (%.2f MB)\n",
           N, (N * sizeof(float)) / (1024.0 * 1024.0));

    // 创建基准测试对象
    Benchmark bench(2, 10);
    size_t data_size = N * sizeof(float) * 2;
    size_t flops = (size_t)N * 200;

    // 运行版本 1：可分页内存 + 同步传输
    printf("\n--- 版本 1：可分页内存 + 同步传输 ---\n");
    bench.run("可分页内存+同步传输", [&]() {
        pageable_sync_transfer(h_a_pageable, h_b_pageable, h_c1);
    }, data_size, flops);

    // 运行版本 2：Pinned Memory + 同步传输
    printf("\n--- 版本 2：Pinned Memory + 同步传输 ---\n");
    bench.run("Pinned内存+同步传输", [&]() {
        pinned_sync_transfer(h_a_pinned, h_b_pinned, h_c2);
    }, data_size, flops);

    // 运行版本 3：Pinned Memory + 异步传输（无重叠）
    printf("\n--- 版本 3：Pinned Memory + 异步传输（无重叠） ---\n");
    bench.run("Pinned内存+异步无重叠", [&]() {
        pinned_async_no_overlap(h_a_pinned, h_b_pinned, h_c3);
    }, data_size, flops);

    // 运行版本 4：Pinned Memory + 传输与计算重叠
    printf("\n--- 版本 4：Pinned Memory + 传输与计算重叠 ---\n");
    bench.run("Pinned内存+传输计算重叠", [&]() {
        pinned_async_with_overlap(h_a_pinned, h_b_pinned, h_c4);
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
        printf("\n性能分析（以可分页内存+同步传输为基准）:\n");
        for (int i = 0; i < results.size(); i++) {
            float speedup = base_time / results[i].avg_time_ms;
            printf("  %s: 加速比 = %.2fx\n", results[i].name.c_str(), speedup);
        }

        float pinned_improvement = results[0].avg_time_ms / results[1].avg_time_ms;
        float overlap_improvement = results[1].avg_time_ms / results[3].avg_time_ms;
        printf("\n关键观察:\n");
        printf("  Pinned Memory 带来的传输提升: %.2fx\n", pinned_improvement);
        printf("  传输与计算重叠带来的额外提升: %.2fx\n", overlap_improvement);
    }

    // 释放内存
    delete[] h_a_pageable;
    delete[] h_b_pageable;
    delete[] h_c_pageable;
    CUDA_CHECK(cudaFreeHost(h_a_pinned));
    CUDA_CHECK(cudaFreeHost(h_b_pinned));
    CUDA_CHECK(cudaFreeHost(h_c1));
    CUDA_CHECK(cudaFreeHost(h_c2));
    CUDA_CHECK(cudaFreeHost(h_c3));
    CUDA_CHECK(cudaFreeHost(h_c4));

    // 重置设备
    CUDA_CHECK(cudaDeviceReset());

    printf("\n程序执行完成。\n");
    return 0;
}
