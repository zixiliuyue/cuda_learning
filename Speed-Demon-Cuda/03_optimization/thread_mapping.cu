/**
 * @file thread_mapping.cu
 * @brief 线程映射优化对比示例
 * 
 * 本文件演示不同线程映射策略的性能差异，包括：
 * - 1D 线程映射（基础版 vs 优化版）
 * - 2D 线程映射
 * - 不同 block 大小对比
 * - 边界处理方式对比
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/benchmark.h"

// ============================================
// 1D 线程映射 - 基础版
// ============================================

/**
 * @brief 1D 线程映射基础版本
 * 
 * 特点：
 * - 简单的 1D 索引计算
 * - 每次处理一个元素
 * - 无跨步循环
 */
__global__ void vector_add_1d_basic(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================
// 1D 线程映射 - 优化版（跨步循环）
// ============================================

/**
 * @brief 1D 线程映射优化版本（跨步循环）
 * 
 * 特点：
 * - 使用跨步循环，减少 grid 大小
 * - 每个线程处理多个元素
 * - 提高指令级并行
 */
__global__ void vector_add_1d_stride(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

// ============================================
// 2D 线程映射 - 矩阵加法
// ============================================

/**
 * @brief 2D 线程映射 - 矩阵加法
 */
__global__ void matrix_add_2d(float* a, float* b, float* c, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief 2D 线程映射 - 使用 1D 索引（优化）
 * 
 * 将矩阵视为 1D 数组，使用 1D 索引
 */
__global__ void matrix_add_1d_as_2d(float* a, float* b, float* c, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    
    if (idx < total_elements) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================
// 不同 block 大小的 kernel
// ============================================

__global__ void vector_add_block64(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 64 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_block128(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 128 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_block256(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_block512(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 512 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_block1024(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 1024 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================
// 边界处理方式对比
// ============================================

/**
 * @brief 边界处理 - 简单 if 检查（每次循环都检查）
 */
__global__ void boundary_check_each(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    for (int i = idx; i < n; i += gridDim.x * 256) {
        data[i] = data[i] * 2.0f;
    }
}

/**
 * @brief 边界处理 - 提前返回
 */
__global__ void boundary_early_return(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= n) return;
    
    int step = gridDim.x * 256;
    for (int i = idx; i < n; i += step) {
        data[i] = data[i] * 2.0f;
    }
}

/**
 * @brief 边界处理 - 掩码方式
 */
__global__ void boundary_masked(float* data, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    for (int i = idx; i < n + 256 - 1; i += step) {
        float val = (i < n) ? data[i] : 0.0f;
        val = val * 2.0f;
        if (i < n) data[i] = val;
    }
}

// ============================================
// 运行测试
// ============================================

void run_thread_mapping_tests() {
    const int N = 1 << 24;  // 16M 元素
    const int data_size = N * sizeof(float);
    const size_t data_size_bytes = data_size;
    
    std::cout << "========================================" << std::endl;
    std::cout << "      线程映射优化对比测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "数据大小: " << N << " 元素 (" << (float)data_size / (1024*1024) << " MB)" << std::endl;
    
    // 分配内存
    float *h_a, *h_b, *h_c, *h_c_ref;
    float *d_a, *d_b, *d_c;
    
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    h_c_ref = new float[N];
    
    CUDA_CHECK(cudaMalloc(&d_a, data_size));
    CUDA_CHECK(cudaMalloc(&d_b, data_size));
    CUDA_CHECK(cudaMalloc(&d_c, data_size));
    
    // 初始化数据
    initialize_array(h_a, N, 0.0f, 1.0f);
    initialize_array(h_b, N, 0.0f, 1.0f);
    cpu_vector_add(h_a, h_b, h_c_ref, N);
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, data_size, cudaMemcpyHostToDevice));
    
    Benchmark bench(2, 10);
    
    // ============================================
    // 测试 1: 1D 基础 vs 跨步
    // ============================================
    std::cout << "\n--- 测试 1: 1D 基础 vs 跨步 ---\n" << std::endl;
    
    bench.run("1D 基础版 (256x1024)", [&]() {
        vector_add_1d_basic<<<1024, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    bench.run("1D 跨步版 (256x64)", [&]() {
        vector_add_1d_stride<<<64, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    // ============================================
    // 测试 2: 不同 block 大小对比
    // ============================================
    std::cout << "\n--- 测试 2: 不同 Block 大小对比 ---\n" << std::endl;
    
    bench.run("Block 大小 = 64", [&]() {
        vector_add_block64<<<(N + 64 - 1)/64, 64>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    bench.run("Block 大小 = 128", [&]() {
        vector_add_block128<<<(N + 128 - 1)/128, 128>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    bench.run("Block 大小 = 256", [&]() {
        vector_add_block256<<<(N + 256 - 1)/256, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    bench.run("Block 大小 = 512", [&]() {
        vector_add_block512<<<(N + 512 - 1)/512, 512>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    bench.run("Block 大小 = 1024", [&]() {
        vector_add_block1024<<<(N + 1024 - 1)/1024, 1024>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    // ============================================
    // 测试 3: 2D 线程映射对比
    // ============================================
    std::cout << "\n--- 测试 3: 2D 线程映射对比 ---\n" << std::endl;
    
    const int rows = 4096;
    const int cols = 4096;
    const int matrix_elements = rows * cols;
    const size_t matrix_data_size = matrix_elements * sizeof(float);
    
    float *d_ma, *d_mb, *d_mc;
    CUDA_CHECK(cudaMalloc(&d_ma, matrix_data_size));
    CUDA_CHECK(cudaMalloc(&d_mb, matrix_data_size));
    CUDA_CHECK(cudaMalloc(&d_mc, matrix_data_size));
    
    CUDA_CHECK(cudaMemcpy(d_ma, h_a, matrix_data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mb, h_b, matrix_data_size, cudaMemcpyHostToDevice));
    
    dim3 block2d(16, 16);
    dim3 grid2d((cols + block2d.x - 1)/block2d.x, (rows + block2d.y - 1)/block2d.y);
    
    bench.run("2D 映射 (16x16)", [&]() {
        matrix_add_2d<<<grid2d, block2d>>>(d_ma, d_mb, d_mc, rows, cols);
    }, matrix_data_size * 3, matrix_elements * 2);
    
    bench.run("1D 映射模拟 2D", [&]() {
        matrix_add_1d_as_2d<<<(matrix_elements + 256 - 1)/256, 256>>>(
            d_ma, d_mb, d_mc, rows, cols);
    }, matrix_data_size * 3, matrix_elements * 2);
    
    CUDA_CHECK(cudaFree(d_ma));
    CUDA_CHECK(cudaFree(d_mb));
    CUDA_CHECK(cudaFree(d_mc));
    
    // ============================================
    // 测试 4: 边界处理方式对比
    // ============================================
    std::cout << "\n--- 测试 4: 边界处理方式对比 ---\n" << std::endl;
    
    const int N_non_divisible = N + 123;  // 非 256 倍数
    const size_t data_size_non_divisible = N_non_divisible * sizeof(float);
    
    float *d_data_boundary;
    CUDA_CHECK(cudaMalloc(&d_data_boundary, data_size_non_divisible));
    CUDA_CHECK(cudaMemcpy(d_data_boundary, h_a, data_size_non_divisible, cudaMemcpyHostToDevice));
    
    bench.run("边界检查每次循环", [&]() {
        boundary_check_each<<<64, 256>>>(d_data_boundary, N_non_divisible);
    }, data_size_non_divisible * 2, N_non_divisible);
    
    bench.run("提前返回边界检查", [&]() {
        boundary_early_return<<<64, 256>>>(d_data_boundary, N_non_divisible);
    }, data_size_non_divisible * 2, N_non_divisible);
    
    bench.run("掩码边界处理", [&]() {
        boundary_masked<<<64, 256>>>(d_data_boundary, N_non_divisible);
    }, data_size_non_divisible * 2, N_non_divisible);
    
    CUDA_CHECK(cudaFree(d_data_boundary));
    
    // 打印汇总报告
    bench.print_summary();
    
    // 清理
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    std::cout << "\n测试完成！" << std::endl;
}

// ============================================
// 主函数
// ============================================

int main() {
    print_device_info();
    run_thread_mapping_tests();
    return 0;
}
