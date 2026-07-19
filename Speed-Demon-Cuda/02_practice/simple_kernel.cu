#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

__global__ void simple_add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void simple_multiply_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void saxpy_kernel(float* a, float* b, float* c, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = alpha * a[idx] + b[idx];
    }
}

void initialize_data(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    const int n = 1 << 24;
    const int bytes = n * sizeof(float);
    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    
    initialize_data(h_a, n);
    initialize_data(h_b, n);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    printf("=== 执行向量加法核函数 ===\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    simple_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float add_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&add_time, start, stop));
    printf("向量加法执行时间: %.3f ms\n", add_time);
    
    printf("\n=== 执行向量乘法核函数 ===\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    simple_multiply_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float multiply_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&multiply_time, start, stop));
    printf("向量乘法执行时间: %.3f ms\n", multiply_time);
    
    printf("\n=== 执行 SAXPY 核函数 ===\n");
    float alpha = 2.0f;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    saxpy_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, alpha, n);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float saxpy_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&saxpy_time, start, stop));
    printf("SAXPY 执行时间: %.3f ms\n", saxpy_time);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < 100; ++i) {
        float expected = alpha * h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            correct = false;
            printf("结果验证失败！索引 %d: 预期 %f, 实际 %f\n", i, expected, h_c[i]);
            break;
        }
    }
    if (correct) {
        printf("\n结果验证通过！\n");
    }
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
