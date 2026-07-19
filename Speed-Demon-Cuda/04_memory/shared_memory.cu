#include <iostream>
#include <cuda_runtime.h>
#include "../common/benchmark.h"
#include "../common/cuda_utils.h"

#define BLOCK_SIZE 128
#define ARRAY_SIZE (1 << 20)

__global__ void v1_bad_bank_conflict(const float* input, float* output)
{
    __shared__ float sh[32][32];

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;

    int row = tid / 32;
    int col = tid % 32;

    // 将数据加载到共享内存
    if (tid < 1024) {
        sh[row][col] = input[blockIdx.x * 1024 + tid];
    }
    __syncthreads();

    if (idx < ARRAY_SIZE) {
        // Bank 冲突：同一 warp 内多个线程访问同一个 Bank
        // 线程 0, 32, 64, 96 访问 sh[0][0], sh[1][0], 都在 Bank 0
        float val = 0.0f;
        for (int i = 0; i < 32; ++i) {
            val += sh[i][col];  // 所有线程同时访问同一列，严重 Bank 冲突
        }
        output[idx] = val;
    }
}

__global__ void v2_good_no_bank_conflict(const float* input, float* output)
{
    // 使用填充避免 Bank 冲突
    __shared__ float sh[32][33];

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;

    int row = tid / 32;
    int col = tid % 32;

    // 将数据加载到共享内存
    if (tid < 1024) {
        sh[row][col] = input[blockIdx.x * 1024 + tid];
    }
    __syncthreads();

    if (idx < ARRAY_SIZE) {
        // 无 Bank 冲突：由于填充，不同行映射到不同 Bank
        float val = 0.0f;
        for (int i = 0; i < 32; ++i) {
            val += sh[i][col];  // 由于填充，sh[i][col] 在不同 Bank
        }
        output[idx] = val;
    }
}

__global__ void v3_bad_conflict_reduction(const float* input, float* output)
{
    __shared__ float sh[128];

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;

    // 加载数据到共享内存
    if (idx < ARRAY_SIZE) {
        sh[tid] = input[idx];
    } else {
        sh[tid] = 0.0f;
    }
    __syncthreads();

    // 归约求和 - 有 Bank 冲突的版本
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh[tid] += sh[tid + s];  // stride 访问，有 Bank 冲突
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sh[0];
    }
}

__global__ void v4_good_no_conflict_reduction(const float* input, float* output)
{
    __shared__ float sh[128];

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;

    // 加载数据到共享内存
    if (idx < ARRAY_SIZE) {
        sh[tid] = input[idx];
    } else {
        sh[tid] = 0.0f;
    }
    __syncthreads();

    // 归约求和 - 优化 Bank 冲突
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // 使用顺序访问模式减少冲突
            float sum = sh[tid] + sh[tid + s];
            __syncthreads();
            sh[tid] = sum;
            __syncthreads();
        }
    }

    if (tid == 0) {
        output[blockIdx.x] = sh[0];
    }
}

void initialize_array(float* array, int size)
{
    for (int i = 0; i < size; ++i) {
        array[i] = static_cast<float>(i) / 1000.0f;
    }
}

bool verify_results(const float* output1, const float* output2, int size)
{
    for (int i = 0; i < size; ++i) {
        if (std::abs(output1[i] - output2[i]) > 1e-3f) {
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

    const int data_size = ARRAY_SIZE * sizeof(float);
    const int reduction_blocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int reduction_size = reduction_blocks * sizeof(float);

    printf("\n=== 共享内存 Bank 冲突性能对比 ===\n");
    printf("数组大小: %d 元素\n", ARRAY_SIZE);
    printf("数据大小: %.2f MB\n\n", data_size / (1024.0f * 1024.0f));

    float* h_input = new float[ARRAY_SIZE];
    float* h_output_bad = new float[ARRAY_SIZE];
    float* h_output_good = new float[ARRAY_SIZE];
    float* h_reduction_bad = new float[reduction_blocks];
    float* h_reduction_good = new float[reduction_blocks];

    initialize_array(h_input, ARRAY_SIZE);

    float* d_input = nullptr;
    float* d_output_bad = nullptr;
    float* d_output_good = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, data_size));
    CUDA_CHECK(cudaMalloc(&d_output_bad, data_size));
    CUDA_CHECK(cudaMalloc(&d_output_good, data_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Grid 维度: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("Block 维度: (%d, %d, %d)\n\n", block_dim.x, block_dim.y, block_dim.z);

    Benchmark bench(3, 15);

    printf("=== 测试 1: 列访问（Bank 冲突对比） ===\n");

    bench.run("v1_bad - 有 Bank 冲突", [&]() {
        v1_bad_bank_conflict<<<grid_dim, block_dim>>>(d_input, d_output_bad);
    }, data_size, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bench.run("v2_good - 无 Bank 冲突", [&]() {
        v2_good_no_bank_conflict<<<grid_dim, block_dim>>>(d_input, d_output_good);
    }, data_size, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_bad, d_output_bad, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_good, d_output_good, data_size, cudaMemcpyDeviceToHost));

    printf("\n=== 结果验证 ===\n");
    verify_results(h_output_bad, h_output_good, ARRAY_SIZE);

    const auto& results = bench.get_results();
    if (results.size() >= 2) {
        float speedup = results[0].avg_time_ms / results[1].avg_time_ms;
        printf("Bank 冲突优化提升: %.2fx\n\n", speedup);
    }

    bench.clear();

    printf("=== 测试 2: 归约操作（Bank 冲突对比） ===\n");

    bench.run("v3_bad - 归约(有冲突)", [&]() {
        v3_bad_conflict_reduction<<<grid_dim, block_dim>>>(d_input, d_output_bad);
    }, data_size, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    bench.run("v4_good - 归约(优化冲突)", [&]() {
        v4_good_no_conflict_reduction<<<grid_dim, block_dim>>>(d_input, d_output_good);
    }, data_size, 0);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_reduction_bad, d_output_bad, reduction_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_reduction_good, d_output_good, reduction_size, cudaMemcpyDeviceToHost));

    printf("\n=== 结果验证 ===\n");
    verify_results(h_reduction_bad, h_reduction_good, reduction_blocks);

    bench.print_summary();

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_bad));
    CUDA_CHECK(cudaFree(d_output_good));

    delete[] h_input;
    delete[] h_output_bad;
    delete[] h_output_good;
    delete[] h_reduction_bad;
    delete[] h_reduction_good;

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
