/******************************************************************************
 * @file    main.cu
 * @brief   Day 38: Cython + CUDA 混合编程 —— 纯 CUDA Kernel 与 C 风格包装函数
 * @details 本文件仅包含纯粹的 CUDA Kernel 实现以及供 Cython 调用的
 *          C 风格（extern "C"）主机端包装函数，不包含任何 Python 绑定代码。
 *
 *          设计要点：
 *          1. Kernel 与包装函数解耦，Kernel 只关心计算逻辑；
 *          2. 包装函数负责显存分配、H2D/D2H 拷贝、Kernel 启动与释放；
 *          3. extern "C" 保证函数名不被 C++ 编译器 mangling（名称修饰），
 *             这样 Cython（本质是 C/C++）才能按名字正确链接到这些符号；
 *          4. 包装函数返回 int：0=成功, -1=失败，方便 Python 侧抛异常。
 *
 *          同时提供 main() 入口，可作为独立可执行程序验证 Kernel 正确性
 *          （不依赖 Python 环境），编译方式见 CMakeLists.txt / build 脚本。
 *
 * @author  CUDA 30天学习项目
 * @date     2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* ---------------------------------------------------------------------------
 * 说明：
 *   本文件被 Cython 的 setup.py 直接作为源文件编译（与 .pyx 一起链接），
 *   为了减少对外部头文件的依赖，这里不强制包含 common/include 中的工具头，
 *   而是内联一份最简单的错误检查逻辑。独立编译（CMake）时同样适用。
 * ------------------------------------------------------------------------- */

#define BLOCK_SIZE 256      /* 一维 Kernel 每个 Block 的线程数 */
#define TILE_WIDTH 16       /* 矩阵乘法分块（Tile）宽度 */

/* 内部错误检查宏：一旦 CUDA API 出错，打印信息并跳转到 cleanup 返回 -1 */
#define CU_CHECK(call)                                                        \
    do {                                                                     \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA 错误] %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            goto cleanup;                                                   \
        }                                                                  \
    } while (0)

/* =========================================================================
 * 1. CUDA Kernel 实现（纯计算逻辑，与主机代码无耦合）
 * ========================================================================= */

/**
 * @brief  向量加法 Kernel：out[i] = a[i] + b[i]
 * @param  a   输入向量 A（设备指针）
 * @param  b   输入向量 B（设备指针）
 * @param  out 输出向量（设备指针）
 * @param  n   元素个数
 */
__global__ void vector_add_kernel(const float* a, const float* b,
                                  float* out, int n)
{
    /* 计算全局线程索引 */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /* 边界检查：防止越界访问 */
    if (tid < n)
    {
        out[tid] = a[tid] + b[tid];
    }
}

/**
 * @brief  矩阵乘法 Kernel（共享内存分块 / Tiled 版本）：C = A * B
 * @param  a     输入矩阵 A（width x width，行主序，设备指针）
 * @param  b     输入矩阵 B（width x width，行主序，设备指针）
 * @param  c     输出矩阵 C（width x width，行主序，设备指针）
 * @param  width 方阵边长
 *
 * @details 每个 Block 负责计算 C 的一个 TILE_WIDTH x TILE_WIDTH 子块，
 *          通过共享内存缓存 A、B 的分块，减少全局内存访问次数。
 */
__global__ void matmul_kernel(const float* a, const float* b,
                              float* c, int width)
{
    /* 共享内存：分别缓存 A、B 的一个 Tile */
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /* 当前线程负责的输出元素坐标 (row, col) */
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    /* 沿 K 维度分块遍历，逐块累加 */
    int num_tiles = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < num_tiles; ++t)
    {
        /* 将 A 的 Tile 加载到共享内存（含边界保护） */
        int a_col = t * TILE_WIDTH + tx;
        if (row < width && a_col < width)
            ds_a[ty][tx] = a[row * width + a_col];
        else
            ds_a[ty][tx] = 0.0f;

        /* 将 B 的 Tile 加载到共享内存（含边界保护） */
        int b_row = t * TILE_WIDTH + ty;
        if (b_row < width && col < width)
            ds_b[ty][tx] = b[b_row * width + col];
        else
            ds_b[ty][tx] = 0.0f;

        /* 同步：确保整个 Tile 加载完成后再计算 */
        __syncthreads();

        /* 计算当前 Tile 对结果的贡献 */
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += ds_a[ty][k] * ds_b[k][tx];

        /* 同步：确保计算完成后再覆盖共享内存 */
        __syncthreads();
    }

    /* 写回结果（含边界保护） */
    if (row < width && col < width)
        c[row * width + col] = sum;
}

/* =========================================================================
 * 2. 主机端 C 风格包装函数（extern "C" 供 Cython 链接调用）
 * ========================================================================= */

extern "C" {

/**
 * @brief  在 GPU 上执行向量加法（完整的显存管理 + Kernel 调用）
 * @param  a   主机端输入向量 A（长度 n）
 * @param  b   主机端输入向量 B（长度 n）
 * @param  out 主机端输出向量（长度 n，由调用方分配）
 * @param  n   元素个数
 * @return 0=成功, -1=失败
 */
int vector_add_wrapper(const float* a, const float* b, float* out, int n)
{
    float *d_a = NULL, *d_b = NULL, *d_out = NULL;
    size_t bytes = (size_t)n * sizeof(float);
    int ret = -1;

    /* 分配设备内存 */
    CU_CHECK(cudaMalloc((void**)&d_a, bytes));
    CU_CHECK(cudaMalloc((void**)&d_b, bytes));
    CU_CHECK(cudaMalloc((void**)&d_out, bytes));

    /* 主机 -> 设备 拷贝输入 */
    CU_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    /* 配置一维网格并启动 Kernel */
    {
        int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_out, n);
        CU_CHECK(cudaGetLastError());        /* 检查 Kernel 启动错误 */
        CU_CHECK(cudaDeviceSynchronize());   /* 等待 Kernel 执行完成 */
    }

    /* 设备 -> 主机 拷贝结果 */
    CU_CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));

    ret = 0;  /* 全部成功 */

cleanup:
    /* 无论成功与否都释放设备内存（NULL 传入 cudaFree 是安全的） */
    if (d_a)   cudaFree(d_a);
    if (d_b)   cudaFree(d_b);
    if (d_out) cudaFree(d_out);
    return ret;
}

/**
 * @brief  在 GPU 上执行方阵乘法 C = A * B
 * @param  a     主机端矩阵 A（width x width，行主序）
 * @param  b     主机端矩阵 B（width x width，行主序）
 * @param  c     主机端输出矩阵 C（width x width，由调用方分配）
 * @param  width 方阵边长
 * @return 0=成功, -1=失败
 */
int matmul_wrapper(const float* a, const float* b, float* c, int width)
{
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    size_t bytes = (size_t)width * width * sizeof(float);
    int ret = -1;

    CU_CHECK(cudaMalloc((void**)&d_a, bytes));
    CU_CHECK(cudaMalloc((void**)&d_b, bytes));
    CU_CHECK(cudaMalloc((void**)&d_c, bytes));

    CU_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    /* 配置二维网格（每个 Block 计算一个 TILE_WIDTH x TILE_WIDTH 子块） */
    {
        dim3 block(TILE_WIDTH, TILE_WIDTH);
        dim3 grid((width + TILE_WIDTH - 1) / TILE_WIDTH,
                  (width + TILE_WIDTH - 1) / TILE_WIDTH);
        matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, width);
        CU_CHECK(cudaGetLastError());
        CU_CHECK(cudaDeviceSynchronize());
    }

    CU_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

    ret = 0;

cleanup:
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    return ret;
}

}  /* extern "C" */

/* =========================================================================
 * 3. 独立可执行入口（可选）：不依赖 Python，直接验证 Kernel 正确性
 * ========================================================================= */
#ifndef CYTHON_BUILD  /* 被 Cython 编译时可通过宏跳过 main，避免重复符号 */

/* CPU 参考实现：向量加法 */
static void vector_add_cpu(const float* a, const float* b, float* out, int n)
{
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

/* CPU 参考实现：方阵乘法 */
static void matmul_cpu(const float* a, const float* b, float* c, int width)
{
    for (int i = 0; i < width; ++i)
        for (int j = 0; j < width; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < width; ++k)
                sum += a[i * width + k] * b[k * width + j];
            c[i * width + j] = sum;
        }
}

int main(void)
{
    printf("\n============================================================\n");
    printf("     Day 38: Cython + CUDA 混合编程（Kernel 自检） \n");
    printf("============================================================\n\n");
    printf("  [说明] 此可执行文件用于验证纯 CUDA Kernel 的正确性， \n");
    printf("  [说明] 真正的 Cython 扩展请在 python/ 目录用 setup.py 编译。 \n\n");

    /* -------- 向量加法自检 -------- */
    const int N = 1 << 20;   /* 1M 元素 */
    size_t vbytes = (size_t)N * sizeof(float);
    float* va = (float*)malloc(vbytes);
    float* vb = (float*)malloc(vbytes);
    float* vgpu = (float*)malloc(vbytes);
    float* vcpu = (float*)malloc(vbytes);
    for (int i = 0; i < N; ++i) { va[i] = (float)(i % 100); vb[i] = (float)(i % 7); }

    int rc1 = vector_add_wrapper(va, vb, vgpu, N);
    vector_add_cpu(va, vb, vcpu, N);
    int vec_ok = (rc1 == 0);
    for (int i = 0; i < N && vec_ok; ++i)
        if (fabsf(vgpu[i] - vcpu[i]) > 1e-4f) vec_ok = 0;

    /* -------- 矩阵乘法自检 -------- */
    const int W = 256;
    size_t mbytes = (size_t)W * W * sizeof(float);
    float* ma = (float*)malloc(mbytes);
    float* mb = (float*)malloc(mbytes);
    float* mgpu = (float*)malloc(mbytes);
    float* mcpu = (float*)malloc(mbytes);
    for (int i = 0; i < W * W; ++i) { ma[i] = (float)((i % 13) * 0.1f); mb[i] = (float)((i % 5) * 0.2f); }

    int rc2 = matmul_wrapper(ma, mb, mgpu, W);
    matmul_cpu(ma, mb, mcpu, W);
    int mat_ok = (rc2 == 0);
    for (int i = 0; i < W * W && mat_ok; ++i)
        if (fabsf(mgpu[i] - mcpu[i]) > 1e-2f) mat_ok = 0;

    /* -------- 结果汇总表 -------- */
    printf("============================================================\n");
    printf("     Kernel 正确性汇总 \n");
    printf("============================================================\n");
    printf("  %-20s | %-10s | %-10s\n", "测试项 ", "规模 ", "结果 ");
    printf("  ------------------------------------------------------\n");
    printf("  %-20s | %-10d | %-10s\n", "向量加法 ", N, vec_ok ? "通过 " : "失败 ");
    printf("  %-20s | %-10s | %-10s\n", "矩阵乘法 ", "256x256", mat_ok ? "通过 " : "失败 ");
    printf("============================================================\n\n");

    free(va); free(vb); free(vgpu); free(vcpu);
    free(ma); free(mb); free(mgpu); free(mcpu);

    printf("[主机] 程序正常退出。 \n\n");
    return (vec_ok && mat_ok) ? 0 : 1;
}

#endif /* CYTHON_BUILD */
