/**
 * @file concurrent_kernels.cu
 * @brief CUDA 核函数并发执行示例
 *        演示多个核函数在不同流中并发执行的性能对比
 */

#include "../common/cuda_utils.h"
#include "../common/benchmark.h"
#include <stdio.h>

// 向量大小和核函数数量
const int N = 1 << 18;  // 256K 元素
const int NUM_KERNELS = 4;

/**
 * @brief 向量加法核函数（计算密集型）
 */
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < 200; i++) {
            sum += a[idx] + b[idx];
        }
        c[idx] = sum / 200.0f;
    }
}

/**
 * @brief 向量乘法核函数（计算密集型）
 */
__global__ void vector_mul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float product = 1.0f;
        for (int i = 0; i < 50; i++) {
            product *= a[idx] * b[idx];
            product = fmin(product, 1e10f);  // 防止溢出
        }
        c[idx] = product;
    }
}

/**
 * @brief 向量缩放核函数（计算密集型）
 */
__global__ void vector_scale(const float* a, float* c, float scale, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float result = a[idx];
        for (int i = 0; i < 100; i++) {
            result = result * scale + a[idx];
        }
        c[idx] = result;
    }
}

/**
 * @brief 向量正弦变换核函数（计算密集型）
 */
__global__ void vector_sin(const float* a, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        float result = 0.0f;
        for (int i = 0; i < 50; i++) {
            result += sin(a[idx] * (i + 1) * 0.1f);
        }
        c[idx] = result;
    }
}

/**
 * @brief 串行执行版本（所有核函数在默认流中顺序执行）
 */
void serial_execution(const float* h_a, const float* h_b,
                      float* h_c1, float* h_c2, float* h_c3, float* h_c4) {
    float *d_a, *d_b, *d_c1, *d_c2, *d_c3, *d_c4;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c3, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c4, N * sizeof(float)));

    // 传输输入数据
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);

    // 串行执行 4 个核函数（都在默认流中）
    vector_add<<<grid_dim, block_dim>>>(d_a, d_b, d_c1, N);
    CUDA_CHECK(cudaGetLastError());

    vector_mul<<<grid_dim, block_dim>>>(d_a, d_b, d_c2, N);
    CUDA_CHECK(cudaGetLastError());

    vector_scale<<<grid_dim, block_dim>>>(d_a, d_c3, 1.1f, N);
    CUDA_CHECK(cudaGetLastError());

    vector_sin<<<grid_dim, block_dim>>>(d_a, d_c4, N);
    CUDA_CHECK(cudaGetLastError());

    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());

    // 传输结果回主机
    CUDA_CHECK(cudaMemcpy(h_c1, d_c1, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c2, d_c2, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c3, d_c3, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c4, d_c4, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaFree(d_c2));
    CUDA_CHECK(cudaFree(d_c3));
    CUDA_CHECK(cudaFree(d_c4));
}

/**
 * @brief 并发执行版本（每个核函数在独立的流中执行）
 */
void concurrent_execution(const float* h_a, const float* h_b,
                           float* h_c1, float* h_c2, float* h_c3, float* h_c4) {
    float *d_a, *d_b, *d_c1, *d_c2, *d_c3, *d_c4;
    cudaStream_t streams[NUM_KERNELS];

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c2, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c3, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c4, N * sizeof(float)));

    // 创建流
    for (int i = 0; i < NUM_KERNELS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // 传输输入数据（在默认流中，作为依赖）
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block_dim(256);
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x);

    // 在不同流中并发执行 4 个核函数
    vector_add<<<grid_dim, block_dim, 0, streams[0]>>>(d_a, d_b, d_c1, N);
    CUDA_CHECK(cudaGetLastError());

    vector_mul<<<grid_dim, block_dim, 0, streams[1]>>>(d_a, d_b, d_c2, N);
    CUDA_CHECK(cudaGetLastError());

    vector_scale<<<grid_dim, block_dim, 0, streams[2]>>>(d_a, d_c3, 1.1f, N);
    CUDA_CHECK(cudaGetLastError());

    vector_sin<<<grid_dim, block_dim, 0, streams[3]>>>(d_a, d_c4, N);
    CUDA_CHECK(cudaGetLastError());

    // 等待所有流中的核函数执行完成
    for (int i = 0; i < NUM_KERNELS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    // 传输结果回主机
    CUDA_CHECK(cudaMemcpy(h_c1, d_c1, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c2, d_c2, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c3, d_c3, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_c4, d_c4, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放资源
    for (int i = 0; i < NUM_KERNELS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaFree(d_c2));
    CUDA_CHECK(cudaFree(d_c3));
    CUDA_CHECK(cudaFree(d_c4));
}

int main() {
    printf("=== CUDA 核函数并发执行示例 ===\n\n");

    // 打印设备信息
    print_device_info();

    // 检查设备是否支持并发核函数
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("\n设备并发能力:\n");
    printf("  支持并发核函数: %s\n", prop.concurrentKernels ? "是" : "否");
    printf("  异步引擎数量: %d\n", prop.asyncEngineCount);
    printf("  多处理器数量: %d\n", prop.multiProcessorCount);

    // 分配主机内存（使用 Pinned Memory）
    float *h_a, *h_b, *h_c1_serial, *h_c2_serial, *h_c3_serial, *h_c4_serial;
    float *h_c1_concurrent, *h_c2_concurrent, *h_c3_concurrent, *h_c4_concurrent;
    CUDA_CHECK(cudaMallocHost(&h_a, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_b, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c1_serial, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c2_serial, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c3_serial, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c4_serial, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c1_concurrent, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c2_concurrent, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c3_concurrent, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_c4_concurrent, N * sizeof(float)));

    // 初始化数据
    printf("\n初始化数据...\n");
    srand(42);
    initialize_array(h_a, N, 0.1f, 0.9f);
    initialize_array(h_b, N, 0.1f, 0.9f);
    printf("每个向量大小: %d 个浮点数 (%.2f MB)\n",
           N, (N * sizeof(float)) / (1024.0 * 1024.0));
    printf("并发核函数数量: %d\n", NUM_KERNELS);

    // 创建基准测试对象
    Benchmark bench(2, 10);

    // 计算总 FLOPs
    size_t flops_per_kernel = N * 400;  // 每个核函数的大致运算量
    size_t total_flops = flops_per_kernel * NUM_KERNELS;

    // 运行串行版本
    bench.run("串行执行（4个核函数顺序）", [&]() {
        serial_execution(h_a, h_b, h_c1_serial, h_c2_serial,
                         h_c3_serial, h_c4_serial);
    }, N * sizeof(float) * 6, total_flops);

    // 运行并发版本
    bench.run("并发执行（4个核函数独立流）", [&]() {
        concurrent_execution(h_a, h_b, h_c1_concurrent, h_c2_concurrent,
                             h_c3_concurrent, h_c4_concurrent);
    }, N * sizeof(float) * 6, total_flops);

    // 验证结果
    printf("\n验证结果...\n");
    bool correct1 = verify_results(h_c1_serial, h_c1_concurrent, N, 1e-3f);
    bool correct2 = verify_results(h_c2_serial, h_c2_concurrent, N, 1e-3f);
    bool correct3 = verify_results(h_c3_serial, h_c3_concurrent, N, 1e-3f);
    bool correct4 = verify_results(h_c4_serial, h_c4_concurrent, N, 1e-3f);

    if (!correct1 || !correct2 || !correct3 || !correct4) {
        printf("错误：部分结果验证失败！\n");
    }

    // 打印汇总报告
    bench.print_summary();

    // 计算加速比
    auto results = bench.get_results();
    if (results.size() >= 2) {
        float speedup = results[0].avg_time_ms / results[1].avg_time_ms;
        printf("核函数并发加速比: %.2fx\n", speedup);
        printf("理论最大加速比: %.2fx\n", (float)NUM_KERNELS);
        printf("效率: %.1f%%\n", (speedup / NUM_KERNELS) * 100.0f);

        if (speedup > 1.5f) {
            printf("✓ 核函数并发获得了显著性能提升\n");
        } else if (speedup > 1.0f) {
            printf("  核函数并发获得了一定性能提升\n");
        } else {
            printf("  核函数并发未获得明显提升（可能原因：核函数太大、硬件限制等）\n");
        }
    }

    // 释放主机内存
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c1_serial));
    CUDA_CHECK(cudaFreeHost(h_c2_serial));
    CUDA_CHECK(cudaFreeHost(h_c3_serial));
    CUDA_CHECK(cudaFreeHost(h_c4_serial));
    CUDA_CHECK(cudaFreeHost(h_c1_concurrent));
    CUDA_CHECK(cudaFreeHost(h_c2_concurrent));
    CUDA_CHECK(cudaFreeHost(h_c3_concurrent));
    CUDA_CHECK(cudaFreeHost(h_c4_concurrent));

    // 重置设备
    CUDA_CHECK(cudaDeviceReset());

    printf("\n程序执行完成。\n");
    return 0;
}
