#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA错误: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

template<typename F>
double measure_time(F&& func, int iterations = 10) {
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func();
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count() / iterations;
}

// ============================================
// 1. 向量加法基准测试
// ============================================

__global__ void vector_add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void benchmark_vector_add(int size) {
    std::cout << "\n[1] 向量加法测试 (大小: " << size << ")" << std::endl;
    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    h_a = new float[size];
    h_b = new float[size];
    h_c = new float[size];
    
    for (int i = 0; i < size; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i + 1);
    }
    
    CHECK_CUDA(cudaMalloc(&d_a, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, size * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((size + 255) / 256);
    
    double avg_time = measure_time([&]() {
        vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, size);
    }, 100);
    
    double bandwidth = (3.0 * size * sizeof(float)) / (avg_time * 1e6) / 1024;
    std::cout << "    平均时间: " << std::fixed << std::setprecision(4) << avg_time << " ms" << std::endl;
    std::cout << "    带宽: " << std::fixed << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
    
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < std::min(1000, size); i++) {
        if (std::abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "    正确性: " << (correct ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    delete[] h_a; delete[] h_b; delete[] h_c;
    CHECK_CUDA(cudaFree(d_a)); CHECK_CUDA(cudaFree(d_b)); CHECK_CUDA(cudaFree(d_c));
}

// ============================================
// 2. 简单矩阵乘法基准测试
// ============================================

__global__ void matmul_simple_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void benchmark_matmul_simple(int N) {
    std::cout << "\n[2] 简单矩阵乘法测试 (N: " << N << ")" << std::endl;
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    
    double avg_time = measure_time([&]() {
        matmul_simple_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }, 10);
    
    double flops = 2.0 * N * N * N / (avg_time * 1e-3) / 1e9;
    std::cout << "    平均时间: " << std::fixed << std::setprecision(4) << avg_time << " ms" << std::endl;
    std::cout << "    GFLOPS: " << std::fixed << std::setprecision(2) << flops << std::endl;
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < std::min(1000, N * N); i++) {
        if (std::abs(h_C[i] - static_cast<float>(N)) > 1e-3) {
            correct = false;
            break;
        }
    }
    std::cout << "    正确性: " << (correct ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    delete[] h_A; delete[] h_B; delete[] h_C;
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
}

// ============================================
// 3. 分块矩阵乘法基准测试
// ============================================

__global__ void matmul_tiled_kernel(float* A, float* B, float* C, int N) {
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 32 + ty;
    int col = blockIdx.x * 32 + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + 31) / 32; t++) {
        if (row < N && t * 32 + tx < N) {
            As[ty][tx] = A[row * N + t * 32 + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * 32 + ty < N) {
            Bs[ty][tx] = B[(t * 32 + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < 32; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void benchmark_matmul_tiled(int N) {
    std::cout << "\n[3] 分块矩阵乘法测试 (N: " << N << ")" << std::endl;
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    h_A = new float[N * N];
    h_B = new float[N * N];
    h_C = new float[N * N];
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    
    CHECK_CUDA(cudaMalloc(&d_A, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N * N * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    
    double avg_time = measure_time([&]() {
        matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }, 20);
    
    double flops = 2.0 * N * N * N / (avg_time * 1e-3) / 1e9;
    std::cout << "    平均时间: " << std::fixed << std::setprecision(4) << avg_time << " ms" << std::endl;
    std::cout << "    GFLOPS: " << std::fixed << std::setprecision(2) << flops << std::endl;
    
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < std::min(1000, N * N); i++) {
        if (std::abs(h_C[i] - static_cast<float>(N)) > 1e-3) {
            correct = false;
            break;
        }
    }
    std::cout << "    正确性: " << (correct ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    delete[] h_A; delete[] h_B; delete[] h_C;
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
}

// ============================================
// 4. 规约求和基准测试
// ============================================

__global__ void reduce_sum_kernel(float* data, float* result, int n) {
    __shared__ float sdata[512];
    int tid = threadIdx.x;
    int idx = blockIdx.x * 1024 + threadIdx.x;
    
    sdata[tid] = 0.0f;
    if (idx < n) sdata[tid] += data[idx];
    if (idx + 512 < n) sdata[tid] += data[idx + 512];
    __syncthreads();
    
    for (int s = 256; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) result[blockIdx.x] = sdata[0];
}

void benchmark_reduction(int size) {
    std::cout << "\n[4] 规约求和测试 (大小: " << size << ")" << std::endl;
    
    float *h_data, *h_result;
    float *d_data, *d_result;
    
    h_data = new float[size];
    for (int i = 0; i < size; i++) {
        h_data[i] = 1.0f;
    }
    
    int blocks = (size + 1023) / 1024;
    h_result = new float[blocks];
    
    CHECK_CUDA(cudaMalloc(&d_data, size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, blocks * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(512);
    dim3 grid(blocks);
    
    double avg_time = measure_time([&]() {
        reduce_sum_kernel<<<grid, block>>>(d_data, d_result, size);
    }, 50);
    
    double throughput = (size * sizeof(float)) / (avg_time * 1e6) / 1024;
    std::cout << "    平均时间: " << std::fixed << std::setprecision(4) << avg_time << " ms" << std::endl;
    std::cout << "    吞吐量: " << std::fixed << std::setprecision(2) << throughput << " GB/s" << std::endl;
    
    CHECK_CUDA(cudaMemcpy(h_result, d_result, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum = 0.0f;
    for (int i = 0; i < blocks; i++) sum += h_result[i];
    bool correct = (std::abs(sum - static_cast<float>(size)) / size < 1e-4);
    std::cout << "    正确性: " << (correct ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    delete[] h_data; delete[] h_result;
    CHECK_CUDA(cudaFree(d_data)); CHECK_CUDA(cudaFree(d_result));
}

// ============================================
// 5. 卷积基准测试
// ============================================

__global__ void convolution_kernel(float* input, float* kernel, float* output, 
                                     int width, int height, int kernel_size) {
    int idx = blockIdx.x * 64 + threadIdx.x;
    int output_width = width - kernel_size + 1;
    int output_height = height - kernel_size + 1;
    
    if (idx < output_width * output_height) {
        int out_row = idx / output_width;
        int out_col = idx % output_width;
        
        float sum = 0.0f;
        for (int k_row = 0; k_row < kernel_size; k_row++) {
            for (int k_col = 0; k_col < kernel_size; k_col++) {
                int in_row = out_row + k_row;
                int in_col = out_col + k_col;
                sum += input[in_row * width + in_col] * kernel[k_row * kernel_size + k_col];
            }
        }
        output[idx] = sum;
    }
}

void benchmark_convolution(int width, int height, int kernel_size) {
    std::cout << "\n[5] 卷积测试 (" << width << "x" << height << ", 核大小: " << kernel_size << "x" << kernel_size << ")" << std::endl;
    
    float *h_input, *h_kernel, *h_output;
    float *d_input, *d_kernel, *d_output;
    
    h_input = new float[width * height];
    h_kernel = new float[kernel_size * kernel_size];
    int output_width = width - kernel_size + 1;
    int output_height = height - kernel_size + 1;
    h_output = new float[output_width * output_height];
    
    for (int i = 0; i < width * height; i++) h_input[i] = 1.0f;
    for (int i = 0; i < kernel_size * kernel_size; i++) h_kernel[i] = 1.0f;
    
    CHECK_CUDA(cudaMalloc(&d_input, width * height * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, output_width * output_height * sizeof(float)));
    
    CHECK_CUDA(cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(64);
    dim3 grid((output_width * output_height + 63) / 64);
    
    double avg_time = measure_time([&]() {
        convolution_kernel<<<grid, block>>>(d_input, d_kernel, d_output, width, height, kernel_size);
    }, 20);
    
    double operations = 2.0 * output_width * output_height * kernel_size * kernel_size;
    double gflops = operations / (avg_time * 1e-3) / 1e9;
    std::cout << "    平均时间: " << std::fixed << std::setprecision(4) << avg_time << " ms" << std::endl;
    std::cout << "    GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
    
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_width * output_height * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    float expected = static_cast<float>(kernel_size * kernel_size);
    for (int i = 0; i < std::min(1000, output_width * output_height); i++) {
        if (std::abs(h_output[i] - expected) > 1e-3) {
            correct = false;
            break;
        }
    }
    std::cout << "    正确性: " << (correct ? "PASS ✓" : "FAIL ✗") << std::endl;
    
    delete[] h_input; delete[] h_kernel; delete[] h_output;
    CHECK_CUDA(cudaFree(d_input)); CHECK_CUDA(cudaFree(d_kernel)); CHECK_CUDA(cudaFree(d_output));
}

// ============================================
// 6. 内存复制带宽测试
// ============================================

void benchmark_memory_copy(int size) {
    std::cout << "\n[6] 内存复制带宽测试 (大小: " << size << " MB)" << std::endl;
    
    float *h_data, *d_data;
    int bytes = size * 1024 * 1024;
    
    CHECK_CUDA(cudaMallocHost(&h_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    
    for (int i = 0; i < bytes / (int)sizeof(float); i++) {
        h_data[i] = static_cast<float>(i);
    }
    
    double h2d_time = measure_time([&]() {
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    }, 10);
    
    double d2h_time = measure_time([&]() {
        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    }, 10);
    
    double d2d_time = measure_time([&]() {
        cudaMemcpy(d_data, d_data, bytes, cudaMemcpyDeviceToDevice);
    }, 10);
    
    double h2d_bw = static_cast<double>(size) / h2d_time * 1000 / 1024;
    double d2h_bw = static_cast<double>(size) / d2h_time * 1000 / 1024;
    double d2d_bw = static_cast<double>(size) / d2d_time * 1000 / 1024;
    
    std::cout << "    H2D 带宽: " << std::fixed << std::setprecision(2) << h2d_bw << " GB/s" << std::endl;
    std::cout << "    D2H 带宽: " << std::fixed << std::setprecision(2) << d2h_bw << " GB/s" << std::endl;
    std::cout << "    D2D 带宽: " << std::fixed << std::setprecision(2) << d2d_bw << " GB/s" << std::endl;
    
    CHECK_CUDA(cudaFreeHost(h_data));
    CHECK_CUDA(cudaFree(d_data));
}

// ============================================
// 主函数
// ============================================

void print_device_info() {
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    
    std::cout << "========================================" << std::endl;
    std::cout << "   CUDA 基准测试套件" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "检测到 " << device_count << " 个 CUDA 设备" << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        std::cout << "\n设备 " << i << ": " << prop.name << std::endl;
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  全局内存: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  SM 数量: " << prop.multiProcessorCount << std::endl;
        std::cout << "  显存带宽: ~" << prop.memoryClockRate * prop.memoryBusWidth / 8 / 1024 / 1024 << " GB/s" << std::endl;
    }
    std::cout << "========================================" << std::endl;
}

int main() {
    print_device_info();
    
    std::cout << "\n开始运行基准测试..." << std::endl;
    
    benchmark_vector_add(1 << 24);
    benchmark_matmul_simple(512);
    benchmark_matmul_tiled(1024);
    benchmark_reduction(1 << 24);
    benchmark_convolution(2048, 2048, 5);
    benchmark_memory_copy(128);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "所有基准测试完成!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
