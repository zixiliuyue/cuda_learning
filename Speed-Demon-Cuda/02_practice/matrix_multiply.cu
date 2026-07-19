#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define TILE_SIZE 32

__global__ void naive_matmul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void tiled_matmul(float* A, float* B, float* C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + tx < N) {
            s_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < N && col < N) {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void tiled_matmul_bank_conflict(float* A, float* B, float* C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + tx < N) {
            s_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < N && col < N) {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void initialize_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

bool verify_result(float* C_naive, float* C_optimized, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (fabs(C_naive[i] - C_optimized[i]) > 1e-3) {
            printf("验证失败！索引 %d: 朴素 %.6f, 优化 %.6f, 差值 %.6f\n", 
                   i, C_naive[i], C_optimized[i], fabs(C_naive[i] - C_optimized[i]));
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1024;
    const int bytes = N * N * sizeof(float);
    
    float *h_A, *h_B, *h_C_naive, *h_C_tiled, *h_C_tiled_no_bank;
    float *d_A, *d_B, *d_C;
    
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C_naive = (float*)malloc(bytes);
    h_C_tiled = (float*)malloc(bytes);
    h_C_tiled_no_bank = (float*)malloc(bytes);
    
    initialize_matrix(h_A, N);
    initialize_matrix(h_B, N);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    printf("=== 朴素矩阵乘法 (无优化) ===\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    naive_matmul<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float naive_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&naive_time, start, stop));
    printf("执行时间: %.3f ms\n", naive_time);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));
    
    printf("\n=== 使用共享内存分块的矩阵乘法 ===\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    tiled_matmul<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float tiled_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&tiled_time, start, stop));
    printf("执行时间: %.3f ms\n", tiled_time);
    printf("加速比: %.2fx\n", naive_time / tiled_time);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));
    
    printf("\n=== 无Bank冲突的分块矩阵乘法 ===\n");
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    tiled_matmul_bank_conflict<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float no_bank_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&no_bank_time, start, stop));
    printf("执行时间: %.3f ms\n", no_bank_time);
    printf("相对于朴素版本加速比: %.2fx\n", naive_time / no_bank_time);
    printf("相对于有bank冲突版本加速比: %.2fx\n", tiled_time / no_bank_time);
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_tiled_no_bank, d_C, bytes, cudaMemcpyDeviceToHost));
    
    printf("\n=== 结果验证 ===\n");
    bool tiled_correct = verify_result(h_C_naive, h_C_tiled, N);
    bool no_bank_correct = verify_result(h_C_naive, h_C_tiled_no_bank, N);
    
    if (tiled_correct && no_bank_correct) {
        printf("所有版本结果验证通过！\n");
    } else {
        printf("验证失败！\n");
    }
    
    printf("\n=== 性能分析 ===\n");
    float operations = 2.0f * N * N * N;
    printf("总浮点运算次数: %.2f GFLOPs\n", operations / 1e9);
    printf("朴素版本 GFLOPS: %.2f\n", operations / (naive_time / 1000.0f) / 1e9);
    printf("分块版本 GFLOPS: %.2f\n", operations / (tiled_time / 1000.0f) / 1e9);
    printf("无Bank冲突版本 GFLOPS: %.2f\n", operations / (no_bank_time / 1000.0f) / 1e9);
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_tiled_no_bank);
    
    return 0;
}
