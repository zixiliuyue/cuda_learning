#include <iostream>
#include <cuda_runtime.h>
#include "../common/benchmark.h"
#include "../common/cuda_utils.h"

#define BLOCK_SIZE 256
#define KERNEL_SIZE 15
#define HALF_KERNEL (KERNEL_SIZE / 2)
#define ARRAY_SIZE (1 << 20)

__constant__ float d_const_kernel[KERNEL_SIZE];

__global__ void v1_bad_global_memory(const float* input, const float* kernel, float* output)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx >= HALF_KERNEL && idx < ARRAY_SIZE - HALF_KERNEL) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            // 每次都从全局内存读取 kernel 数据
            // 没有利用缓存和广播机制
            sum += input[idx + k - HALF_KERNEL] * kernel[k];
        }
        output[idx] = sum;
    }
}

__global__ void v2_good_constant_memory(const float* input, float* output)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (idx >= HALF_KERNEL && idx < ARRAY_SIZE - HALF_KERNEL) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            // 从常量内存读取 kernel 数据
            // 自动利用常量缓存和 warp 广播机制
            sum += input[idx + k - HALF_KERNEL] * d_const_kernel[k];
        }
        output[idx] = sum;
    }
}

__global__ void v3_best_shared_constant(const float* input, float* output)
{
    __shared__ float sh_input[BLOCK_SIZE + KERNEL_SIZE - 1];

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;

    // 加载数据到共享内存，包含 halo 区域
    if (idx < ARRAY_SIZE) {
        sh_input[tid + HALF_KERNEL] = input[idx];
    } else {
        sh_input[tid + HALF_KERNEL] = 0.0f;
    }

    // 加载左边界
    if (tid < HALF_KERNEL && idx >= HALF_KERNEL) {
        sh_input[tid] = input[idx - HALF_KERNEL];
    }

    // 加载右边界
    if (tid >= BLOCK_SIZE - HALF_KERNEL && idx + HALF_KERNEL < ARRAY_SIZE) {
        sh_input[tid + KERNEL_SIZE - 1] = input[idx + HALF_KERNEL];
    }

    __syncthreads();

    if (idx >= HALF_KERNEL && idx < ARRAY_SIZE - HALF_KERNEL) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            // 共享内存读取输入，常量内存读取 kernel
            sum += sh_input[tid + k] * d_const_kernel[k];
        }
        output[idx] = sum;
    }
}

void initialize_gaussian_kernel(float* kernel, int size, float sigma = 1.0f)
{
    float sum = 0.0f;
    int half = size / 2;

    for (int i = 0; i < size; ++i) {
        float x = static_cast<float>(i - half);
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }
}

void initialize_array(float* array, int size)
{
    for (int i = 0; i < size; ++i) {
        array[i] = static_cast<float>(i % 256) / 256.0f;
    }
}

bool verify_results(const float* expected, const float* actual, int size, float epsilon = 1e-5f)
{
    for (int i = HALF_KERNEL; i < size - HALF_KERNEL; ++i) {
        if (std::abs(expected[i] - actual[i]) > epsilon) {
            printf("验证失败！索引 %d: 预期 %f, 实际 %f, 差值 %f\n", 
                   i, expected[i], actual[i], std::abs(expected[i] - actual[i]));
            return false;
        }
    }
    printf("验证通过！所有结果匹配 ✓\n");
    return true;
}

void cpu_convolution(const float* input, const float* kernel, float* output, int size)
{
    for (int i = HALF_KERNEL; i < size - HALF_KERNEL; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; ++k) {
            sum += input[i + k - HALF_KERNEL] * kernel[k];
        }
        output[i] = sum;
    }
}

int main()
{
    print_device_info();

    const int data_size = ARRAY_SIZE * sizeof(float);
    const int kernel_size = KERNEL_SIZE * sizeof(float);

    printf("\n=== 常量内存性能对比（一维卷积） ===\n");
    printf("数组大小: %d 元素\n", ARRAY_SIZE);
    printf("数据大小: %.2f MB\n", data_size / (1024.0f * 1024.0f));
    printf("卷积核大小: %d\n", KERNEL_SIZE);
    printf("常量内存使用: %.2f KB\n\n", kernel_size / 1024.0f);

    float* h_input = new float[ARRAY_SIZE];
    float* h_kernel = new float[KERNEL_SIZE];
    float* h_output_cpu = new float[ARRAY_SIZE];
    float* h_output_gpu = new float[ARRAY_SIZE];

    initialize_array(h_input, ARRAY_SIZE);
    initialize_gaussian_kernel(h_kernel, KERNEL_SIZE, 2.0f);

    printf("高斯卷积核: ");
    for (int i = 0; i < KERNEL_SIZE; ++i) {
        printf("%.4f ", h_kernel[i]);
    }
    printf("\n\n");

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, data_size));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size));
    CUDA_CHECK(cudaMalloc(&d_output, data_size));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));

    // 初始化常量内存
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_kernel, h_kernel, kernel_size));

    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Grid 维度: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("Block 维度: (%d, %d, %d)\n\n", block_dim.x, block_dim.y, block_dim.z);

    Benchmark bench(3, 20);
    const size_t flops = 2LL * (ARRAY_SIZE - KERNEL_SIZE + 1) * KERNEL_SIZE;

    printf("=== v1_bad - 全局内存版本 ===\n");
    bench.run("v1_bad - 全局内存", [&]() {
        v1_bad_global_memory<<<grid_dim, block_dim>>>(d_input, d_kernel, d_output);
    }, data_size * 2 + kernel_size, flops);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, data_size, cudaMemcpyDeviceToHost));
    cpu_convolution(h_input, h_kernel, h_output_cpu, ARRAY_SIZE);
    printf("结果验证: ");
    verify_results(h_output_cpu, h_output_gpu, ARRAY_SIZE);

    printf("\n=== v2_good - 常量内存版本 ===\n");
    bench.run("v2_good - 常量内存", [&]() {
        v2_good_constant_memory<<<grid_dim, block_dim>>>(d_input, d_output);
    }, data_size + kernel_size, flops);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, data_size, cudaMemcpyDeviceToHost));
    printf("结果验证: ");
    verify_results(h_output_cpu, h_output_gpu, ARRAY_SIZE);

    printf("\n=== v3_best - 常量内存 + 共享内存 ===\n");
    bench.run("v3_best - 常量+共享内存", [&]() {
        v3_best_shared_constant<<<grid_dim, block_dim>>>(d_input, d_output);
    }, data_size + kernel_size, flops);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, data_size, cudaMemcpyDeviceToHost));
    printf("结果验证: ");
    verify_results(h_output_cpu, h_output_gpu, ARRAY_SIZE);

    printf("\n");
    bench.print_summary();

    const auto& results = bench.get_results();
    if (results.size() >= 3) {
        float speedup_const = results[0].avg_time_ms / results[1].avg_time_ms;
        float speedup_best = results[0].avg_time_ms / results[2].avg_time_ms;
        printf("\n=== 性能总结 ===\n");
        printf("常量内存相对全局内存: %.2fx 加速\n", speedup_const);
        printf("常量+共享内存相对全局内存: %.2fx 加速\n", speedup_best);
        printf("共享内存进一步提升: %.2fx\n\n", speedup_best / speedup_const);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output_cpu;
    delete[] h_output_gpu;

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
