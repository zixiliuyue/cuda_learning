/**
 * @file loop_unrolling.cu
 * @brief 循环展开优化对比示例
 * 
 * 本文件演示不同循环展开策略的性能差异，包括：
 * - 无展开 vs 手动展开（2/4/8/16 倍）
 * - #pragma unroll 编译器自动展开
 * - 跨步循环 + 展开
 * - 不同运算类型的展开效果
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/benchmark.h"

// ============================================
// 无展开 - 基础版本
// ============================================

/**
 * @brief 向量加法 - 无展开
 */
__global__ void vector_add_no_unroll(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================
// 手动展开 - 2 倍
// ============================================

/**
 * @brief 向量加法 - 手动展开 2 倍
 */
__global__ void vector_add_unroll2_manual(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
    if (idx + 256 < n) {
        c[idx + 256] = a[idx + 256] + b[idx + 256];
    }
}

// ============================================
// 手动展开 - 4 倍
// ============================================

/**
 * @brief 向量加法 - 手动展开 4 倍
 */
__global__ void vector_add_unroll4_manual(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    for (int i = idx; i < n; i += step * 4) {
        c[i] = a[i] + b[i];
        c[i + step] = a[i + step] + b[i + step];
        c[i + step * 2] = a[i + step * 2] + b[i + step * 2];
        c[i + step * 3] = a[i + step * 3] + b[i + step * 3];
    }
}

// ============================================
// 使用 #pragma unroll 展开
// ============================================

/**
 * @brief 向量加法 - #pragma unroll 2 倍
 */
__global__ void vector_add_unroll2_pragma(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    #pragma unroll 2
    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief 向量加法 - #pragma unroll 4 倍
 */
__global__ void vector_add_unroll4_pragma(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    #pragma unroll 4
    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief 向量加法 - #pragma unroll 8 倍
 */
__global__ void vector_add_unroll8_pragma(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    #pragma unroll 8
    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief 向量加法 - #pragma unroll 16 倍
 */
__global__ void vector_add_unroll16_pragma(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    #pragma unroll 16
    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief 向量加法 - #pragma unroll 完全展开
 */
__global__ void vector_add_unroll_full(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    #pragma unroll
    for (int i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

// ============================================
// SAXPY 运算展开对比
// ============================================

/**
 * @brief SAXPY - 无展开
 */
__global__ void saxpy_no_unroll(float* a, float* b, float alpha, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        c[idx] = alpha * a[idx] + b[idx];
    }
}

/**
 * @brief SAXPY - 展开 4 倍
 */
__global__ void saxpy_unroll4(float* a, float* b, float alpha, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    #pragma unroll 4
    for (int i = idx; i < n; i += step) {
        c[i] = alpha * a[i] + b[i];
    }
}

/**
 * @brief SAXPY - 展开 8 倍
 */
__global__ void saxpy_unroll8(float* a, float* b, float alpha, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    int step = gridDim.x * 256;
    
    #pragma unroll 8
    for (int i = idx; i < n; i += step) {
        c[i] = alpha * a[i] + b[i];
    }
}

// ============================================
// 点积运算展开对比
// ============================================

/**
 * @brief 点积运算 - 无展开
 */
__global__ void dot_product_no_unroll(float* a, float* b, float* c, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 512 + threadIdx.x;
    int step = gridDim.x * 512;
    
    float sum = 0.0f;
    for (int i = idx; i < n; i += step) {
        sum += a[i] * b[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        c[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief 点积运算 - 展开 4 倍
 */
__global__ void dot_product_unroll4(float* a, float* b, float* c, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 512 + threadIdx.x;
    int step = gridDim.x * 512;
    
    float sum = 0.0f;
    #pragma unroll 4
    for (int i = idx; i < n; i += step) {
        sum += a[i] * b[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        c[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief 点积运算 - 展开 8 倍
 */
__global__ void dot_product_unroll8(float* a, float* b, float* c, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 512 + threadIdx.x;
    int step = gridDim.x * 512;
    
    float sum = 0.0f;
    #pragma unroll 8
    for (int i = idx; i < n; i += step) {
        sum += a[i] * b[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        c[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 矩阵转置展开对比
// ============================================

__global__ void transpose_no_unroll(float* a, float* b, int n) {
    __shared__ float block[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (threadIdx.x < 32 && threadIdx.y < 32) {
        block[threadIdx.y][threadIdx.x] = a[y * n + x];
    }
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (threadIdx.x < 32 && threadIdx.y < 32 && x < n && y < n) {
        b[y * n + x] = block[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_unroll4(float* a, float* b, int n) {
    __shared__ float block[32][32];
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    #pragma unroll 4
    for (int i = 0; i < 32; i += 8) {
        if (threadIdx.x < 32 && threadIdx.y + i < 32) {
            block[threadIdx.y + i][threadIdx.x] = a[(y + i) * n + x];
        }
    }
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    #pragma unroll 4
    for (int i = 0; i < 32; i += 8) {
        if (threadIdx.x < 32 && threadIdx.y + i < 32 && x < n && y + i < n) {
            b[(y + i) * n + x] = block[threadIdx.x][threadIdx.y + i];
        }
    }
}

// ============================================
// 运行测试
// ============================================

void run_loop_unrolling_tests() {
    const int N = 1 << 24;  // 16M 元素
    const int data_size = N * sizeof(float);
    const size_t data_size_bytes = data_size;
    
    std::cout << "========================================" << std::endl;
    std::cout << "      循环展开优化对比测试" << std::endl;
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
    // 测试 1: 向量加法 - 不同展开因子
    // ============================================
    std::cout << "\n--- 测试 1: 向量加法 - 不同展开因子 ---\n" << std::endl;
    
    bench.run("向量加法 - 无展开", [&]() {
        vector_add_no_unroll<<<1024, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    bench.run("向量加法 - 手动展开 2 倍", [&]() {
        vector_add_unroll2_manual<<<1024, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    bench.run("向量加法 - pragma 展开 2 倍", [&]() {
        vector_add_unroll2_pragma<<<64, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    bench.run("向量加法 - pragma 展开 4 倍", [&]() {
        vector_add_unroll4_pragma<<<64, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    bench.run("向量加法 - pragma 展开 8 倍", [&]() {
        vector_add_unroll8_pragma<<<64, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    bench.run("向量加法 - pragma 展开 16 倍", [&]() {
        vector_add_unroll16_pragma<<<64, 256>>>(d_a, d_b, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    CUDA_CHECK(cudaMemcpy(h_c, d_c, data_size, cudaMemcpyDeviceToHost));
    verify_results(h_c_ref, h_c, N);
    
    // ============================================
    // 测试 2: SAXPY 运算展开对比
    // ============================================
    std::cout << "\n--- 测试 2: SAXPY 运算展开对比 ---\n" << std::endl;
    
    float alpha = 2.0f;
    
    bench.run("SAXPY - 无展开", [&]() {
        saxpy_no_unroll<<<1024, 256>>>(d_a, d_b, alpha, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    bench.run("SAXPY - 展开 4 倍", [&]() {
        saxpy_unroll4<<<64, 256>>>(d_a, d_b, alpha, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    bench.run("SAXPY - 展开 8 倍", [&]() {
        saxpy_unroll8<<<64, 256>>>(d_a, d_b, alpha, d_c, N);
    }, data_size_bytes * 3, N * 2);
    
    // ============================================
    // 测试 3: 点积运算展开对比
    // ============================================
    std::cout << "\n--- 测试 3: 点积运算展开对比 ---\n" << std::endl;
    
    const int blocks_per_sm = 8;
    const int num_blocks = 64 * blocks_per_sm;
    
    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    
    bench.run("点积 - 无展开", [&]() {
        dot_product_no_unroll<<<num_blocks, 256>>>(d_a, d_b, d_partial, N);
    }, data_size_bytes * 2, N * 2);
    
    bench.run("点积 - 展开 4 倍", [&]() {
        dot_product_unroll4<<<num_blocks, 256>>>(d_a, d_b, d_partial, N);
    }, data_size_bytes * 2, N * 2);
    
    bench.run("点积 - 展开 8 倍", [&]() {
        dot_product_unroll8<<<num_blocks, 256>>>(d_a, d_b, d_partial, N);
    }, data_size_bytes * 2, N * 2);
    
    CUDA_CHECK(cudaFree(d_partial));
    
    // ============================================
    // 测试 4: 矩阵转置展开对比
    // ============================================
    std::cout << "\n--- 测试 4: 矩阵转置展开对比 ---\n" << std::endl;
    
    const int matrix_size = 4096;
    const int matrix_elements = matrix_size * matrix_size;
    const size_t matrix_data_size = matrix_elements * sizeof(float);
    
    float *d_ma, *d_mb;
    CUDA_CHECK(cudaMalloc(&d_ma, matrix_data_size));
    CUDA_CHECK(cudaMalloc(&d_mb, matrix_data_size));
    CUDA_CHECK(cudaMemcpy(d_ma, h_a, matrix_data_size, cudaMemcpyHostToDevice));
    
    dim3 block_t(32, 32);
    dim3 grid_t((matrix_size + 32 - 1)/32, (matrix_size + 32 - 1)/32);
    
    bench.run("矩阵转置 - 无展开", [&]() {
        transpose_no_unroll<<<grid_t, block_t>>>(d_ma, d_mb, matrix_size);
    }, matrix_data_size * 2, matrix_elements);
    
    bench.run("矩阵转置 - 展开 4 倍", [&]() {
        transpose_unroll4<<<grid_t, block_t>>>(d_ma, d_mb, matrix_size);
    }, matrix_data_size * 2, matrix_elements);
    
    CUDA_CHECK(cudaFree(d_ma));
    CUDA_CHECK(cudaFree(d_mb));
    
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
    run_loop_unrolling_tests();
    return 0;
}
