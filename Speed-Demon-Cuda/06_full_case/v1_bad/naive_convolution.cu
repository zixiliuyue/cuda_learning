#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "benchmark.h"

const int KERNEL_RADIUS = 2;
const int KERNEL_SIZE = 2 * KERNEL_RADIUS + 1;

__global__ void naive_convolution_kernel(float* input, float* output, 
                                         float* kernel, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        float sum = 0.0f;
        for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            for (int l = -KERNEL_RADIUS; l <= KERNEL_RADIUS; l++) {
                int input_row = row + k;
                int input_col = col + l;
                if (input_row >= 0 && input_row < height && 
                    input_col >= 0 && input_col < width) {
                    sum += input[input_row * width + input_col] * 
                           kernel[(k + KERNEL_RADIUS) * KERNEL_SIZE + 
                                  (l + KERNEL_RADIUS)];
                }
            }
        }
        output[row * width + col] = sum;
    }
}

void cpu_convolution(const float* input, float* output, const float* kernel,
                     int width, int height) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            float sum = 0.0f;
            for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
                for (int l = -KERNEL_RADIUS; l <= KERNEL_RADIUS; l++) {
                    int input_row = row + k;
                    int input_col = col + l;
                    if (input_row >= 0 && input_row < height && 
                        input_col >= 0 && input_col < width) {
                        sum += input[input_row * width + input_col] * 
                               kernel[(k + KERNEL_RADIUS) * KERNEL_SIZE + 
                                      (l + KERNEL_RADIUS)];
                    }
                }
            }
            output[row * width + col] = sum;
        }
    }
}

void generate_gaussian_kernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int radius = size / 2;
    
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            float value = expf(-(i * i + j * j) / (2 * sigma * sigma));
            kernel[(i + radius) * size + (j + radius)] = value;
            sum += value;
        }
    }
    
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
}

int main() {
    print_device_info();
    
    const int width = 1920;
    const int height = 1080;
    const int image_size = width * height;
    const int kernel_size = KERNEL_SIZE;
    
    printf("=== 图像卷积 v1: 朴素实现 ===\n");
    printf("图像大小: %d x %d = %d 像素\n", width, height, image_size);
    printf("卷积核: %d x %d 高斯模糊\n\n", kernel_size, kernel_size);
    
    std::vector<float> h_input(image_size);
    std::vector<float> h_output_cpu(image_size);
    std::vector<float> h_output_gpu(image_size);
    std::vector<float> h_kernel(kernel_size * kernel_size);
    
    for (int i = 0; i < image_size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    generate_gaussian_kernel(h_kernel.data(), kernel_size, 1.0f);
    
    printf("=== CPU 参考实现 ===\n");
    CpuTimer cpu_timer;
    cpu_timer.start();
    cpu_convolution(h_input.data(), h_output_cpu.data(), h_kernel.data(), 
                    width, height);
    float cpu_time = cpu_timer.stop();
    printf("CPU 处理时间: %.3f ms\n\n", cpu_time);
    
    float* d_input;
    float* d_output;
    float* d_kernel;
    
    CUDA_CHECK(cudaMalloc(&d_input, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), 
                          image_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), 
                          kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);
    
    printf("=== GPU 朴素实现 ===\n");
    printf("Grid 大小: (%d, %d), Block 大小: (%d, %d)\n", 
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    
    CudaTimer gpu_timer;
    gpu_timer.start();
    
    naive_convolution_kernel<<<grid_dim, block_dim>>>(
        d_input, d_output, d_kernel, width, height
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float gpu_time = gpu_timer.stop();
    printf("GPU 处理时间: %.3f ms\n", gpu_time);
    printf("加速比 (CPU/GPU): %.1fx\n\n", cpu_time / gpu_time);
    
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, 
                          image_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("=== 结果验证 ===\n");
    bool verified = verify_results(h_output_cpu.data(), h_output_gpu.data(), 
                                   image_size, 1e-4f);
    
    if (verified) {
        printf("✅ GPU 结果与 CPU 参考一致！\n\n");
    } else {
        printf("❌ GPU 结果与 CPU 参考不一致！\n\n");
    }
    
    size_t total_ops = static_cast<size_t>(image_size) * kernel_size * kernel_size * 2;
    float gflops = static_cast<float>(total_ops) / (gpu_time * 1e-3f) / 1e9f;
    float memory_access = static_cast<float>(image_size) * (1 + kernel_size * kernel_size) * sizeof(float);
    float bandwidth = memory_access / (gpu_time * 1e-3f) / (1024 * 1024 * 1024);
    
    printf("=== 性能分析 ===\n");
    printf("总浮点操作数: %.2f GFLOPs\n", static_cast<float>(total_ops) / 1e9f);
    printf("GFLOPS: %.2f\n", gflops);
    printf("内存访问量: %.2f GB\n", memory_access / (1024 * 1024 * 1024));
    printf("内存带宽需求: %.2f GB/s\n", bandwidth);
    printf("理论 GPU 带宽利用率估计: ~%.0f%% (基于典型 GPU 300GB/s)\n", 
           (bandwidth / 300.0f) * 100);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_kernel));
    
    printf("\n=== v1_bad 总结 ===\n");
    printf("✅ 功能正确\n");
    printf("❌ 性能瓶颈：全局内存访问效率低，大量重复读取\n");
    printf("❌ 每个像素都读整个卷积核，相邻像素重复读相同数据\n");
    printf("💡 下一步优化：使用共享内存减少全局内存访问\n");
    
    return 0;
}
