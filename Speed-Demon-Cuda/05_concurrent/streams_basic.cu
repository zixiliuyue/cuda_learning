/**
 * @file streams_basic.cu
 * @brief CUDA 流基本使用示例
 *        演示串行执行 vs 并发执行的性能对比
 */

#include "../common/cuda_utils.h"
#include "../common/benchmark.h"
#include <stdio.h>

// 向量大小
const int N = 1 << 20;  // 1M 元素
const int NUM_STREAMS = 4;

/**
 * @brief 向量加法核函数
 */
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        // 增加计算量，使核函数执行时间更长
        float sum = 0.0f;
        for (int i = 0; i < 100; i++) {
            sum += a[idx] + b[idx];
        }
        c[idx] = sum / 100.0f;
    }
}

/**
 * @brief 串行执行版本（默认流）
 *        所有操作在同一个流中顺序执行
 */
void serial_execution(const float* h_a, const float* h_b, float* h_c) {
    float *d_a, *d_b, *d_c;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // 同步传输数据（阻塞）
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // 核函数执行（在默认流中）
    dim3 block_dim(256);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);
    vector_add<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    // 同步传输结果回主机
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 同步设备（确保所有操作完成）
    CUDA_CHECK(cudaDeviceSynchronize());

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

/**
 * @brief 并发执行版本（使用多个流）
 *        将数据分成多个块，在不同流中并发执行
 */
void concurrent_execution(const float* h_a, const float* h_b, float* h_c) {
    const int chunk_size = N / NUM_STREAMS;

    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];

    // 为每个流分配设备内存
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMalloc(&d_a[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c[i], chunk_size * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    dim3 block_dim(256);
    dim3 grid_dim((chunk_size + block_dim.x - 1) / block_dim.x);

    // 在每个流中执行：传输 -> 计算 -> 传输回
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;

        // 1. 异步传输数据到设备
        CUDA_CHECK(cudaMemcpyAsync(d_a[i], h_a + offset,
                                    chunk_size * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    streams[i]));
        CUDA_CHECK(cudaMemcpyAsync(d_b[i], h_b + offset,
                                    chunk_size * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    streams[i]));

        // 2. 在该流中异步执行核函数
        vector_add<<<grid_dim, block_dim, 0, streams[i]>>>(
            d_a[i], d_b[i], d_c[i], chunk_size);
        CUDA_CHECK(cudaGetLastError());

        // 3. 异步传输结果回主机
        CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c[i],
                                    chunk_size * sizeof(float),
                                    cudaMemcpyDeviceToHost,
                                    streams[i]));
    }

    // 等待所有流完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // 释放资源
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaFree(d_a[i]));
        CUDA_CHECK(cudaFree(d_b[i]));
        CUDA_CHECK(cudaFree(d_c[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
}

int main() {
    printf("=== CUDA 流基本使用示例 ===\n\n");

    // 打印设备信息
    print_device_info();

    // 分配主机内存（使用 Pinned Memory 支持异步传输）
    float *h_a, *h_b, *h_c_serial, *h_c_concurrent;
    CUDA_CHECK(cudaMallocHost(&h_a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c_serial, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c_concurrent, N * sizeof(float)));

    // 初始化数据
    printf("\n初始化数据...\n");
    srand(42);
    initialize_array(h_a, N, 0.0f, 1.0f);
    initialize_array(h_b, N, 0.0f, 1.0f);
    printf("数据大小: %d 个浮点数 (%.2f MB)\n",
           N, (N * sizeof(float)) / (1024.0 * 1024.0));
    printf("流数量: %d\n", NUM_STREAMS);

    // 创建基准测试对象
    Benchmark bench(2, 10);

    // 运行串行版本
    bench.run("串行执行（默认流）", [&]() {
        serial_execution(h_a, h_b, h_c_serial);
    }, N * sizeof(float) * 2, (size_t)N * 100 * 2);

    // 运行并发版本
    bench.run("并发执行（多流）", [&]() {
        concurrent_execution(h_a, h_b, h_c_concurrent);
    }, N * sizeof(float) * 2, (size_t)N * 100 * 2);

    // 验证结果
    printf("\n验证结果...\n");
    bool correct = verify_results(h_c_serial, h_c_concurrent, N);
    if (!correct) {
        printf("错误：并发版本结果不正确！\n");
    }

    // 打印汇总报告
    bench.print_summary();

    // 计算加速比
    auto results = bench.get_results();
    if (results.size() >= 2) {
        float speedup = results[0].avg_time_ms / results[1].avg_time_ms;
        printf("加速比: %.2fx\n", speedup);
        if (speedup > 1.0f) {
            printf("✓ 并发执行获得了性能提升\n");
        } else {
            printf("  并发执行未获得明显提升（可能原因：核函数太小、硬件限制等）\n");
        }
    }

    // 释放主机内存
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c_serial));
    CUDA_CHECK(cudaFreeHost(h_c_concurrent));

    // 重置设备
    CUDA_CHECK(cudaDeviceReset());

    printf("\n程序执行完成。\n");
    return 0;
}
