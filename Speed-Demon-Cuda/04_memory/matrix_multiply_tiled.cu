#include <iostream>
#include <cuda_runtime.h>
#include "../common/benchmark.h"
#include "../common/cuda_utils.h"

#define MATRIX_SIZE 1024
#define TILE_SIZE 32

__global__ void v1_baseline_naive(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            // 每次迭代都访问全局内存
            // 数据复用率低，O(N³) 次全局访问
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void v2_good_tiled(const float* A, const float* B, float* C, int N)
{
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历所有 Tile
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载 Tile A 到共享内存
        if (row < N && t * TILE_SIZE + tx < N) {
            sh_A[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            sh_A[ty][tx] = 0.0f;
        }

        // 加载 Tile B 到共享内存
        if (t * TILE_SIZE + ty < N && col < N) {
            sh_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            sh_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 在共享内存中计算
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void v3_best_optimized(const float* A, const float* B, float* C, int N)
{
    // 使用填充避免 Bank 冲突
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 向量化加载：一次加载两个元素
    int row_A = row;
    int col_A = tx;
    int row_B = ty;
    int col_B = col;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载 Tile A 到共享内存（带填充）
        if (row_A < N && t * TILE_SIZE + col_A < N) {
            sh_A[ty][tx] = A[row_A * N + t * TILE_SIZE + col_A];
        } else {
            sh_A[ty][tx] = 0.0f;
        }

        // 加载 Tile B 到共享内存（带填充）
        if (t * TILE_SIZE + row_B < N && col_B < N) {
            sh_B[ty][tx] = B[(t * TILE_SIZE + row_B) * N + col_B];
        } else {
            sh_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 在共享内存中计算，无 Bank 冲突
        for (int k = 0; k < TILE_SIZE; ++k) {
            // 由于填充，sh_A[ty][k] 和 sh_B[k][tx] 都不会产生 Bank 冲突
            sum += sh_A[ty][k] * sh_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void initialize_matrix(float* matrix, int size)
{
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

bool verify_results(const float* expected, const float* actual, int size, float epsilon = 1e-2f)
{
    for (int i = 0; i < size * size; ++i) {
        if (std::abs(expected[i] - actual[i]) > epsilon) {
            printf("验证失败！索引 %d: 预期 %f, 实际 %f, 差值 %f\n", 
                   i, expected[i], actual[i], std::abs(expected[i] - actual[i]));
            return false;
        }
    }
    printf("验证通过！所有结果匹配 ✓\n");
    return true;
}

void cpu_matrix_multiply(const float* A, const float* B, float* C, int N)
{
    printf("正在计算 CPU 基准结果（可能需要几分钟）...\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    printf("CPU 计算完成！\n");
}

int main()
{
    print_device_info();

    const int N = MATRIX_SIZE;
    const int data_size = N * N * sizeof(float);

    printf("\n=== 分块矩阵乘法性能对比 ===\n");
    printf("矩阵大小: %d x %d = %d 元素\n", N, N, N * N);
    printf("数据大小: %.2f MB\n", data_size / (1024.0f * 1024.0f));
    printf("浮点运算量: %.2f GFLOPS\n", (2.0f * N * N * N) / 1e9f);
    printf("分块大小: %d x %d\n\n", TILE_SIZE, TILE_SIZE);

    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C_cpu = new float[N * N];
    float* h_C_gpu = new float[N * N];

    srand(42);
    initialize_matrix(h_A, N);
    initialize_matrix(h_B, N);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, data_size));
    CUDA_CHECK(cudaMalloc(&d_B, data_size));
    CUDA_CHECK(cudaMalloc(&d_C, data_size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, data_size, cudaMemcpyHostToDevice));

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    printf("Grid 维度: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("Block 维度: (%d, %d, %d)\n\n", block_dim.x, block_dim.y, block_dim.z);

    Benchmark bench(3, 10);
    const size_t flops = 2LL * N * N * N;

    printf("=== v1_baseline - 朴素矩阵乘法 ===\n");
    bench.run("v1_baseline - 朴素实现", [&]() {
        v1_baseline_naive<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
    }, data_size * 3, flops);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, data_size, cudaMemcpyDeviceToHost));
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, N);
    printf("结果验证: ");
    verify_results(h_C_cpu, h_C_gpu, N);

    printf("\n=== v2_good - 基础分块矩阵乘法 ===\n");
    bench.run("v2_good - 基础分块", [&]() {
        v2_good_tiled<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
    }, data_size * 3, flops);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, data_size, cudaMemcpyDeviceToHost));
    printf("结果验证: ");
    verify_results(h_C_cpu, h_C_gpu, N);

    printf("\n=== v3_best - 优化分块（无 Bank 冲突） ===\n");
    bench.run("v3_best - 优化分块", [&]() {
        v3_best_optimized<<<grid_dim, block_dim>>>(d_A, d_B, d_C, N);
    }, data_size * 3, flops);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, data_size, cudaMemcpyDeviceToHost));
    printf("结果验证: ");
    verify_results(h_C_cpu, h_C_gpu, N);

    printf("\n");
    bench.print_summary();

    const auto& results = bench.get_results();
    if (results.size() >= 3) {
        float speedup_tiled = results[0].avg_time_ms / results[1].avg_time_ms;
        float speedup_optimized = results[0].avg_time_ms / results[2].avg_time_ms;
        printf("\n=== 性能总结 ===\n");
        printf("分块相对朴素实现: %.2fx 加速\n", speedup_tiled);
        printf("优化分块相对朴素实现: %.2fx 加速\n", speedup_optimized);
        printf("优化分块相对基础分块: %.2fx 加速\n\n", speedup_optimized / speedup_tiled);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
