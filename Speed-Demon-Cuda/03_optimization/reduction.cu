/**
 * @file reduction.cu
 * @brief 规约算法优化对比示例
 * 
 * 本文件演示规约算法的逐步优化过程，包括：
 * - 朴素规约
 * - 减少分支发散
 * - 减少共享内存 bank 冲突
 * - 增加指令级并行
 * - 循环展开
 * - Warp 原语优化
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "../common/cuda_utils.h"
#include "../common/benchmark.h"

// ============================================
// 版本 1: 朴素规约 (Naive Reduction)
// ============================================

/**
 * @brief 朴素规约
 * 
 * 特点：
 * - 简单直接的实现
 * - 大量的分支发散
 * - 不使用跨步循环
 */
__global__ void reduction_naive(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sdata[tid] = g_idata[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 版本 2: 减少分支发散
// ============================================

/**
 * @brief 减少分支发散的规约
 * 
 * 优化：
 * - 使用连续的线程索引
 * - 减少 warp 内分支
 */
__global__ void reduction_no_divergence(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sdata[tid] = g_idata[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 版本 3: 跨步循环读取多个元素
// ============================================

/**
 * @brief 跨步循环读取多个元素
 * 
 * 优化：
 * - 每个线程读取多个元素
 * - 减少 block 数量
 * - 增加工作负载
 */
__global__ void reduction_stride(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) {
        sum += g_idata[idx];
    }
    if (idx + blockDim.x < n) {
        sum += g_idata[idx + blockDim.x];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 版本 4: 消除 bank 冲突
// ============================================

/**
 * @brief 消除 bank 冲突的规约
 * 
 * 优化：
 * - 使用 volatile 避免编译器优化
 * - 直接操作共享内存
 */
__global__ void reduction_no_bank_conflict(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) {
        sum += g_idata[idx];
    }
    if (idx + blockDim.x < n) {
        sum += g_idata[idx + blockDim.x];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 版本 5: 循环展开
// ============================================

/**
 * @brief 循环展开的规约
 * 
 * 优化：
 * - 手动展开循环
 * - 减少循环控制开销
 * - 增加指令级并行
 */
__global__ void reduction_unroll(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) {
        sum += g_idata[idx];
    }
    if (idx + blockDim.x < n) {
        sum += g_idata[idx + blockDim.x];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 版本 6: Warp 原语优化
// ============================================

/**
 * @brief Warp 归约函数
 */
__device__ __forceinline__ float warp_reduce(float val) {
    unsigned int tid = threadIdx.x;
    __shared__ float sdata[32];
    
    if (tid < 32) {
        sdata[tid] = val;
    }
    __syncthreads();
    
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    return sdata[tid];
}

/**
 * @brief 使用 Warp 原语优化的规约
 * 
 * 优化：
 * - 最后一个 warp 内直接归约
 * - 减少同步开销
 */
__global__ void reduction_warp_optimized(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) {
        sum += g_idata[idx];
    }
    if (idx + blockDim.x < n) {
        sum += g_idata[idx + blockDim.x];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];
        __syncthreads();
        
        // Warp 内归约
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, s);
        }
        
        sdata[tid] = val;
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 版本 7: 完全优化版本
// ============================================

/**
 * @brief 完全优化的规约
 * 
 * 综合所有优化：
 * - 跨步读取多个元素
 * - 循环展开
 * - Warp 原语
 * - 减少同步
 */
__global__ void reduction_complete(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    
    float sum = 0.0f;
    int step = gridDim.x * blockDim.x * 4;
    
    // 每个线程读取 4 个元素
    for (int i = idx; i < n; i += step) {
        sum += g_idata[i];
        sum += g_idata[i + blockDim.x];
        sum += g_idata[i + blockDim.x * 2];
        sum += g_idata[i + blockDim.x * 3];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 循环展开
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    
    if (tid < 32) {
        float val = sdata[tid] + sdata[tid + 32];
        __syncthreads();
        
        // 使用 warp shuffle
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            val += __shfl_xor_sync(0xffffffff, val, s);
        }
        
        sdata[tid] = val;
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// 最大值规约对比
// ============================================

/**
 * @brief 最大值规约 - 基础版
 */
__global__ void max_reduction_basic(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sdata[tid] = g_idata[idx];
    } else {
        sdata[tid] = -INFINITY;
    }
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief 最大值规约 - 优化版
 */
__global__ void max_reduction_optimized(float* g_idata, float* g_odata, int n) {
    __shared__ float sdata[1024];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    
    float max_val = -INFINITY;
    int step = gridDim.x * blockDim.x * 4;
    
    for (int i = idx; i < n; i += step) {
        max_val = max(max_val, g_idata[i]);
        max_val = max(max_val, g_idata[i + blockDim.x]);
        max_val = max(max_val, g_idata[i + blockDim.x * 2]);
        max_val = max(max_val, g_idata[i + blockDim.x * 3]);
    }
    
    sdata[tid] = max_val;
    __syncthreads();
    
    if (blockDim.x >= 1024) { if (tid < 512) { sdata[tid] = max(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    
    if (tid < 32) {
        float val = max(sdata[tid], sdata[tid + 32]);
        __syncthreads();
        
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            float other = __shfl_xor_sync(0xffffffff, val, s);
            val = max(val, other);
        }
        
        sdata[tid] = val;
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ============================================
// CPU 规约（验证用）
// ============================================

float cpu_reduction_sum(float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

float cpu_reduction_max(float* data, int n) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max_val = max(max_val, data[i]);
    }
    return max_val;
}

// ============================================
// 运行测试
// ============================================

void run_reduction_tests() {
    const int N = 1 << 24;  // 16M 元素
    const int data_size = N * sizeof(float);
    
    std::cout << "========================================" << std::endl;
    std::cout << "      规约算法优化对比测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "数据大小: " << N << " 元素 (" << (float)data_size / (1024*1024) << " MB)" << std::endl;
    
    // 分配内存
    float* h_data = new float[N];
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_size));
    
    // 初始化数据
    initialize_array(h_data, N, 0.0f, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));
    
    // CPU 计算（验证）
    float cpu_sum = cpu_reduction_sum(h_data, N);
    float cpu_max = cpu_reduction_max(h_data, N);
    std::cout << "CPU Sum: " << cpu_sum << std::endl;
    std::cout << "CPU Max: " << cpu_max << std::endl;
    
    Benchmark bench(2, 10);
    
    const int block_size = 1024;
    const int num_blocks = 64;
    
    float* d_partial;
    float h_partial[num_blocks];
    CUDA_CHECK(cudaMalloc(&d_partial, num_blocks * sizeof(float)));
    
    // ============================================
    // 测试 1: 求和规约 - 不同版本对比
    // ============================================
    std::cout << "\n--- 测试 1: 求和规约 - 不同版本对比 ---\n" << std::endl;
    
    // 版本 1: 朴素规约
    bench.run("1. 朴素规约", [&]() {
        reduction_naive<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float gpu_sum = 0.0f;
    for (int i = 0; i < num_blocks; ++i) gpu_sum += h_partial[i];
    std::cout << "   验证: " << (fabs(gpu_sum - cpu_sum) < 1.0f ? "通过" : "失败") << std::endl;
    
    // 版本 2: 减少分支发散
    bench.run("2. 减少分支发散", [&]() {
        reduction_no_divergence<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    // 版本 3: 跨步循环
    bench.run("3. 跨步循环 (2x)", [&]() {
        reduction_stride<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    // 版本 4: 无 bank 冲突
    bench.run("4. 消除 bank 冲突", [&]() {
        reduction_no_bank_conflict<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    // 版本 5: 循环展开
    bench.run("5. 循环展开", [&]() {
        reduction_unroll<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    // 版本 6: Warp 原语优化
    bench.run("6. Warp 原语优化", [&]() {
        reduction_warp_optimized<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    // 版本 7: 完全优化
    bench.run("7. 完全优化 (4x)", [&]() {
        reduction_complete<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    gpu_sum = 0.0f;
    for (int i = 0; i < num_blocks; ++i) gpu_sum += h_partial[i];
    std::cout << "   验证: " << (fabs(gpu_sum - cpu_sum) < 1.0f ? "通过" : "失败") << std::endl;
    
    // ============================================
    // 测试 2: 最大值规约对比
    // ============================================
    std::cout << "\n--- 测试 2: 最大值规约对比 ---\n" << std::endl;
    
    bench.run("最大值规约 - 基础版", [&]() {
        max_reduction_basic<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    float gpu_max = -INFINITY;
    for (int i = 0; i < num_blocks; ++i) gpu_max = max(gpu_max, h_partial[i]);
    std::cout << "   验证: " << (fabs(gpu_max - cpu_max) < 1e-5f ? "通过" : "失败") << std::endl;
    
    bench.run("最大值规约 - 优化版", [&]() {
        max_reduction_optimized<<<num_blocks, block_size>>>(d_data, d_partial, N);
    }, data_size, N);
    
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    gpu_max = -INFINITY;
    for (int i = 0; i < num_blocks; ++i) gpu_max = max(gpu_max, h_partial[i]);
    std::cout << "   验证: " << (fabs(gpu_max - cpu_max) < 1e-5f ? "通过" : "失败") << std::endl;
    
    // ============================================
    // 测试 3: 不同 block 大小对比
    // ============================================
    std::cout << "\n--- 测试 3: 不同 Block 大小对比 ---\n" << std::endl;
    
    const int blocks_per_test = 128;
    
    bench.run("Block = 128", [&]() {
        reduction_complete<<<blocks_per_test, 128>>>(d_data, d_partial, N);
    }, data_size, N);
    
    bench.run("Block = 256", [&]() {
        reduction_complete<<<blocks_per_test, 256>>>(d_data, d_partial, N);
    }, data_size, N);
    
    bench.run("Block = 512", [&]() {
        reduction_complete<<<blocks_per_test, 512>>>(d_data, d_partial, N);
    }, data_size, N);
    
    bench.run("Block = 1024", [&]() {
        reduction_complete<<<blocks_per_test, 1024>>>(d_data, d_partial, N);
    }, data_size, N);
    
    // 打印汇总报告
    bench.print_summary();
    
    // 清理
    delete[] h_data;
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_partial));
    
    std::cout << "\n测试完成！" << std::endl;
}

// ============================================
// 主函数
// ============================================

int main() {
    print_device_info();
    run_reduction_tests();
    return 0;
}
