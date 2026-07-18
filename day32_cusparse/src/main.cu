/******************************************************************************
 * @file    main.cu
 * @brief   Day 32: cuSPARSE - 稀疏矩阵运算库
 * @details 演示使用 cuSPARSE 库进行稀疏矩阵-向量乘法（SpMV），
 *          并与同等规模的稠密矩阵-向量乘法进行性能对比。
 *
 *          稀疏矩阵在科学计算、图算法、机器学习（如推荐系统、GNN）中广泛存在，
 *          当矩阵中绝大多数元素为 0 时，使用稀疏存储格式（如 CSR）可以：
 *          - 大幅减少显存占用（只存储非零元素）
 *          - 显著提升计算速度（跳过 0 元素的乘加运算）
 *
 *          本示例使用 cuSPARSE 的 Generic API（CUDA 11+ 推荐，CUDA 13 兼容）：
 *          - cusparseCreateCsr()   : 创建 CSR 格式稀疏矩阵描述符
 *          - cusparseCreateDnVec() : 创建稠密向量描述符
 *          - cusparseSpMV()        : 执行稀疏矩阵-向量乘法 y = alpha*A*x + beta*y
 *
 *          性能对比维度：
 *          - CSR 稀疏 SpMV（不同稀疏度 90% / 95% / 99%）
 *          - 同等规模的稠密矩阵-向量乘法（手写 Kernel）
 *
 * @note    编译需链接 cusparse 库： -lcusparse
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>          // cuSPARSE 库头文件
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

// 矩阵规模：N x N 方阵
#define MATRIX_SIZE 4096
// 稠密 Kernel 的线程块大小
#define BLOCK_SIZE 256

/**
 * @brief  cuSPARSE API 调用安全封装宏
 * @param  call  任意返回 cusparseStatus_t 的 cuSPARSE API 调用表达式
 *
 * @details 与 CUDA_SAFE_CALL 类似，如果 cuSPARSE 调用返回非
 *          CUSPARSE_STATUS_SUCCESS 的状态码，则打印文件名、行号并终止程序。
 *          cusparseGetErrorString() 从 CUDA 11 起可用，用于获取可读错误描述。
 */
#define CUSPARSE_SAFE_CALL(call)                                              \
    do {                                                                      \
        cusparseStatus_t __status = (call);                                   \
        if (__status != CUSPARSE_STATUS_SUCCESS) {                            \
            fprintf(stderr,                                                   \
                    "[cuSPARSE 错误] 状态码: %d | 描述: %s | 文件: %s | 行: %d\n", \
                    (int)__status, cusparseGetErrorString(__status),          \
                    __FILE__, __LINE__);                                      \
            fflush(stderr);                                                   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * @brief  稠密矩阵-向量乘法 Kernel（对比基准，行优先存储）
 * @param  mat  稠密矩阵，尺寸 n x n（行优先）
 * @param  vec  输入向量，长度 n
 * @param  out  输出向量，长度 n
 * @param  n    矩阵维度
 *
 * @details 每个线程负责计算输出向量的一个元素 out[row] = sum_j(mat[row][j] * vec[j])。
 *          这里对包含大量 0 的稠密矩阵做完整遍历，浪费了大量算力，
 *          正好用来凸显稀疏格式跳过 0 元素带来的优势。
 */
__global__ void dense_spmv_kernel(const float* mat, const float* vec, float* out, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n)
    {
        float sum = 0.0f;
        // 遍历该行所有列（包括所有 0 元素）
        for (int col = 0; col < n; col++)
        {
            sum += mat[(size_t)row * n + col] * vec[col];
        }
        out[row] = sum;
    }
}

/**
 * @brief  在主机端生成指定稀疏度的 CSR 格式稀疏矩阵
 * @param  n            矩阵维度 (n x n)
 * @param  sparsity     稀疏度（0.0~1.0），例如 0.90 表示 90% 元素为 0
 * @param  h_values     [输出] 非零元素值数组
 * @param  h_col_idx    [输出] 每个非零元素的列索引
 * @param  h_row_ptr    [输出] 行指针数组，长度 n+1
 * @return int          非零元素总数 nnz
 *
 * @details CSR（Compressed Sparse Row，压缩稀疏行）格式由三个数组构成：
 *          - values ：按行优先顺序存储的所有非零元素值
 *          - col_idx：每个非零元素对应的列号
 *          - row_ptr：长度 n+1，row_ptr[i] 表示第 i 行的第一个非零元素
 *                     在 values/col_idx 中的起始下标，row_ptr[n] = nnz
 */
static int generate_csr_matrix(int n, float sparsity,
                               std::vector<float>& h_values,
                               std::vector<int>& h_col_idx,
                               std::vector<int>& h_row_ptr)
{
    h_values.clear();
    h_col_idx.clear();
    h_row_ptr.clear();
    h_row_ptr.push_back(0);  // 第 0 行起始下标恒为 0

    // 每行期望保留的非零元素个数
    int nnz_per_row = (int)(n * (1.0 - sparsity));
    if (nnz_per_row < 1) nnz_per_row = 1;  // 至少保留 1 个非零元素

    for (int row = 0; row < n; row++)
    {
        // 简单策略：在该行随机挑选 nnz_per_row 个不重复的列位置放置非零值
        // 为保证性能测试稳定，这里用等距步长 + 随机偏移生成列索引
        int step = n / nnz_per_row;
        if (step < 1) step = 1;

        int count = 0;
        for (int col = row % step; col < n && count < nnz_per_row; col += step)
        {
            float val = (float)rand() / (float)RAND_MAX - 0.5f;  // [-0.5, 0.5)
            h_values.push_back(val);
            h_col_idx.push_back(col);
            count++;
        }
        // 记录累积非零元素个数，作为下一行的起始位置
        h_row_ptr.push_back((int)h_values.size());
    }

    return (int)h_values.size();  // nnz
}

/**
 * @brief  将 CSR 稀疏矩阵还原为稠密矩阵（行优先），用于稠密对比 Kernel
 * @param  n          矩阵维度
 * @param  h_values   非零值数组
 * @param  h_col_idx  列索引数组
 * @param  h_row_ptr  行指针数组
 * @param  h_dense    [输出] 稠密矩阵，需预先分配 n*n 大小并清零
 */
static void csr_to_dense(int n,
                         const std::vector<float>& h_values,
                         const std::vector<int>& h_col_idx,
                         const std::vector<int>& h_row_ptr,
                         float* h_dense)
{
    for (int i = 0; i < (size_t)n * n; i++) h_dense[i] = 0.0f;
    for (int row = 0; row < n; row++)
    {
        for (int k = h_row_ptr[row]; k < h_row_ptr[row + 1]; k++)
        {
            int col = h_col_idx[k];
            h_dense[(size_t)row * n + col] = h_values[k];
        }
    }
}

/**
 * @brief  针对给定稀疏度执行一次完整的 SpMV 性能测试
 * @param  handle     cuSPARSE 库句柄
 * @param  n          矩阵维度
 * @param  sparsity   稀疏度
 * @param  d_x        输入向量（设备端，长度 n）
 * @return double     cusparseSpMV 的执行耗时（毫秒）
 */
static double run_sparse_spmv(cusparseHandle_t handle, int n, float sparsity, const float* d_x)
{
    // ---- 1. 主机端生成 CSR 矩阵 ----
    std::vector<float> h_values;
    std::vector<int>   h_col_idx;
    std::vector<int>   h_row_ptr;
    int nnz = generate_csr_matrix(n, sparsity, h_values, h_col_idx, h_row_ptr);

    double density = 100.0 * (double)nnz / ((double)n * n);
    printf("  [稀疏度 %.0f%%] 非零元素 nnz = %d (实际密度 %.3f%%)\n",
           sparsity * 100.0, nnz, density);

    // ---- 2. 拷贝 CSR 三数组到设备 ----
    float* d_values = nullptr;
    int*   d_col_idx = nullptr;
    int*   d_row_ptr = nullptr;
    float* d_y = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_col_idx, nnz * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_y, n * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(d_y, 0, n * sizeof(float)));

    // ---- 3. 创建 cuSPARSE 描述符（Generic API）----
    // 稀疏矩阵 A 采用 CSR 格式描述
    cusparseSpMatDescr_t matA;
    CUSPARSE_SAFE_CALL(cusparseCreateCsr(
        &matA,
        n, n, nnz,                       // 行数、列数、非零元素数
        d_row_ptr, d_col_idx, d_values,  // CSR 三数组
        CUSPARSE_INDEX_32I,              // row_ptr 索引类型
        CUSPARSE_INDEX_32I,              // col_idx 索引类型
        CUSPARSE_INDEX_BASE_ZERO,        // 索引从 0 开始
        CUDA_R_32F));                    // 值类型为 float

    // 稠密向量 x 与 y
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_SAFE_CALL(cusparseCreateDnVec(&vecX, n, (void*)d_x, CUDA_R_32F));
    CUSPARSE_SAFE_CALL(cusparseCreateDnVec(&vecY, n, (void*)d_y, CUDA_R_32F));

    // ---- 4. 计算 y = alpha * A * x + beta * y ----
    float alpha = 1.0f;
    float beta  = 0.0f;

    // 查询所需的外部工作缓冲区大小
    size_t buffer_size = 0;
    CUSPARSE_SAFE_CALL(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size));

    void* d_buffer = nullptr;
    if (buffer_size > 0)
    {
        CUDA_SAFE_CALL(cudaMalloc(&d_buffer, buffer_size));
    }

    // 先做一次预热调用（避免首次调用的初始化开销污染计时）
    CUSPARSE_SAFE_CALL(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_buffer));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // 正式计时
    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);
    CUSPARSE_SAFE_CALL(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_buffer));
    gpu_timer_stop(&timer);
    double elapsed = gpu_timer_get_elapsed_ms(&timer);

    // ---- 5. 释放资源 ----
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    if (d_buffer) cudaFree(d_buffer);
    cudaFree(d_values);
    cudaFree(d_col_idx);
    cudaFree(d_row_ptr);
    cudaFree(d_y);
    gpu_timer_destroy(&timer);

    return elapsed;
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 32: cuSPARSE - 稀疏矩阵-向量乘法（SpMV）           \n");
    printf("=============================================================\n\n");

    // 打印 GPU 设备信息
    print_device_info();

    int n = MATRIX_SIZE;
    printf("\n[主机] 矩阵规模: %d x %d\n", n, n);
    printf("[主机] 稠密矩阵显存需求: %.2f MB\n", (double)n * n * sizeof(float) / (1024.0 * 1024.0));

    srand((unsigned int)time(NULL));

    // ---- 准备输入向量 x（所有测试共用）----
    float* h_x = (float*)malloc(n * sizeof(float));
    init_random_float(h_x, n, -1.0f, 1.0f);

    float* d_x = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_x, n * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice));

    // ---- 创建 cuSPARSE 句柄 ----
    cusparseHandle_t handle;
    CUSPARSE_SAFE_CALL(cusparseCreate(&handle));

    // ============================================================
    // 测试 1：不同稀疏度下的 cuSPARSE SpMV 性能
    // ============================================================
    printf("\n-------------------------------------------------------------\n");
    printf("[测试 1] cuSPARSE CSR SpMV（不同稀疏度）\n");
    printf("-------------------------------------------------------------\n");
    float sparsities[3] = {0.90f, 0.95f, 0.99f};
    double spmv_times[3];
    for (int i = 0; i < 3; i++)
    {
        spmv_times[i] = run_sparse_spmv(handle, n, sparsities[i], d_x);
    }

    // ============================================================
    // 测试 2：同等规模的稠密矩阵-向量乘法（手写 Kernel）
    //   以 99% 稀疏度的矩阵还原为稠密矩阵进行对比
    // ============================================================
    printf("\n-------------------------------------------------------------\n");
    printf("[测试 2] 稠密矩阵-向量乘法（手写 Kernel，遍历全部元素）\n");
    printf("-------------------------------------------------------------\n");

    std::vector<float> h_values, h_row_ptr_dummy_f;
    std::vector<int> h_col_idx, h_row_ptr;
    generate_csr_matrix(n, 0.99f, h_values, h_col_idx, h_row_ptr);

    // 将 CSR 还原为稠密矩阵
    float* h_dense = (float*)malloc((size_t)n * n * sizeof(float));
    csr_to_dense(n, h_values, h_col_idx, h_row_ptr, h_dense);

    float* d_mat = nullptr;
    float* d_out = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_mat, (size_t)n * n * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, n * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_mat, h_dense, (size_t)n * n * sizeof(float), cudaMemcpyHostToDevice));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 预热
    dense_spmv_kernel<<<grid_size, BLOCK_SIZE>>>(d_mat, d_x, d_out, n);
    CHECK_KERNEL_ERROR();

    GpuTimer dense_timer;
    gpu_timer_init(&dense_timer);
    gpu_timer_start(&dense_timer);
    dense_spmv_kernel<<<grid_size, BLOCK_SIZE>>>(d_mat, d_x, d_out, n);
    gpu_timer_stop(&dense_timer);
    double dense_time = gpu_timer_get_elapsed_ms(&dense_timer);
    printf("  [稠密 Kernel] 遍历 %d x %d 全部元素\n", n, n);
    gpu_timer_print(&dense_timer, "稠密矩阵-向量乘法");

    // ============================================================
    // 性能总结表格
    // ============================================================
    printf("\n=============================================================\n");
    printf("                        性能总结                             \n");
    printf("=============================================================\n");
    printf("  %-28s | %-12s\n", "方法", "耗时(ms)");
    printf("  ---------------------------- | ------------\n");
    printf("  %-28s | %10.4f\n", "CSR SpMV (稀疏度 90%)", spmv_times[0]);
    printf("  %-28s | %10.4f\n", "CSR SpMV (稀疏度 95%)", spmv_times[1]);
    printf("  %-28s | %10.4f\n", "CSR SpMV (稀疏度 99%)", spmv_times[2]);
    printf("  %-28s | %10.4f\n", "稠密矩阵-向量乘法", dense_time);
    printf("  ---------------------------- | ------------\n");
    if (spmv_times[2] > 0.0)
    {
        printf("  稀疏(99%%) 相对稠密加速比    | %9.2fx\n", dense_time / spmv_times[2]);
    }
    printf("=============================================================\n");
    printf("  结论: 稀疏度越高，CSR SpMV 越能跳过 0 元素，优势越明显。\n");
    printf("=============================================================\n\n");

    // ---- 清理资源 ----
    cusparseDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_mat);
    cudaFree(d_out);
    free(h_x);
    free(h_dense);
    gpu_timer_destroy(&dense_timer);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
