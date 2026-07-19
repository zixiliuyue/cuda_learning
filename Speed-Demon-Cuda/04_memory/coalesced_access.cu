#include <iostream>
#include <cuda_runtime.h>
#include "../common/benchmark.h"
#include "../common/cuda_utils.h"

#define BLOCK_SIZE 256
#define MATRIX_SIZE 2048

__global__ void v1_bad_uncoalesced_access(const float* input, float* output, int width, int height)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < width * height) {
        int col = idx / height;
        int row = idx % height;

        // 非合并访问：同一 warp 内线程访问步长为 height 的地址
        // 线程 0 访问 input[0], 线程 1 访问 input[height], 等等
        // 导致 32 个独立的内存事务
        output[idx] = input[row * width + col] * 2.0f;
    }
}

__global__ void v2_good_coalesced_access(const float* input, float* output, int width, int height)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx < width * height) {
        // 合并访问：同一 warp 内线程访问连续的内存地址
        // 线程 k 访问 input[idx + k]，实现完全合并
        // 只需要 1-2 个内存事务
        output[idx] = input[idx] * 2.0f;
    }
}

void initialize_matrix(float* matrix, int width, int height)
{
    for (int i = 0; i < width * height; ++i) {
        matrix[i] = static_cast<float>(i) / 100.0f;
    }
}

bool verify_results(const float* output1, const float* output2, int size)
{
    for (int i = 0; i < size; ++i) {
        if (std::abs(output1[i] - output2[i]) > 1e-5f) {
            printf("验证失败！索引 %d: 预期 %f, 实际 %f\n", i, output1[i], output2[i]);
            return false;
        }
    }
    printf("验证通过！所有结果匹配 ✓\n");
    return true;
}

int main()
{
    print_device_info();

    const int width = MATRIX_SIZE;
    const int height = MATRIX_SIZE;
    const int data_size = width * height * sizeof(float);

    printf("\n=== 合并内存访问性能对比 ===\n");
    printf("矩阵大小: %d x %d = %d 元素\n", width, height, width * height);
    printf("数据大小: %.2f MB\n\n", data_size / (1024.0f * 1024.0f));

    float* h_input = new float[width * height];
    float* h_output_bad = new float[width * height];
    float* h_output_good = new float[width * height];

    initialize_matrix(h_input, width, height);

    float* d_input = nullptr;
    float* d_output_bad = nullptr;
    float* d_output_good = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, data_size));
    CUDA_CHECK(cudaMalloc(&d_output_bad, data_size));
    CUDA_CHECK(cudaMalloc(&d_output_good, data_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((width * height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Grid 维度: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("Block 维度: (%d, %d, %d)\n\n", block_dim.x, block_dim.y, block_dim.z);

    Benchmark bench(3, 20);

    bench.run("v1_bad - 非合并访问", [&]() {
        v1_bad_uncoalesced_access<<<grid_dim, block_dim>>>(d_input, d_output_bad, width, height);
    }, data_size, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bench.run("v2_good - 合并访问", [&]() {
        v2_good_coalesced_access<<<grid_dim, block_dim>>>(d_input, d_output_good, width, height);
    }, data_size, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_bad, d_output_bad, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_good, d_output_good, data_size, cudaMemcpyDeviceToHost));

    printf("\n=== 结果验证 ===\n");
    verify_results(h_output_bad, h_output_good, width * height);

    bench.print_summary();

    const auto& results = bench.get_results();
    if (results.size() >= 2) {
        float speedup = results[0].avg_time_ms / results[1].avg_time_ms;
        printf("性能提升: %.2fx\n\n", speedup);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_bad));
    CUDA_CHECK(cudaFree(d_output_good));

    delete[] h_input;
    delete[] h_output_bad;
    delete[] h_output_good;

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
