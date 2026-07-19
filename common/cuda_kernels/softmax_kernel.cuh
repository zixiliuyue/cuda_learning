/******************************************************************************
 * @file    softmax_kernel.cuh
 * @brief   Softmax 前向 Kernel（LLM / 深度学习高频基础算子）
 * @details 本头文件提供 header-only 的 Softmax 前向实现，可直接 #include 使用。
 *
 *          Softmax 数学定义（对某一行向量 x 的第 i 个元素）：
 *
 *              softmax(x)_i = exp(x_i) / sum_j( exp(x_j) )
 *
 *          【数值稳定性】
 *          直接计算 exp(x_i) 在 x_i 较大时会上溢（inf），较小时会下溢（0）。
 *          工程上采用「减去每行最大值」的等价变形，保证指数参数 <= 0：
 *
 *              m       = max_j(x_j)
 *              softmax(x)_i = exp(x_i - m) / sum_j( exp(x_j - m) )
 *
 *          由于分子分母同乘 exp(-m)，结果在数学上完全等价，但不会溢出。
 *
 *          【并行策略】
 *          每一行分配「一个 Block」处理，Block 内所有线程协作完成三步归约：
 *            1) 求该行最大值 m（max 归约）
 *            2) 求 sum_j( exp(x_j - m) )（sum 归约）
 *            3) 逐元素写出 exp(x_j - m) / sum
 *          归约采用 warp-level 的 __shfl_down_sync + 共享内存跨 warp 归约。
 *
 *          【维度支持】
 *          - 一维：视为 rows=1, cols=N 的单行；
 *          - 二维：形状 [rows, cols]，每一行独立做 softmax（最常见于 attention 分数矩阵）。
 *
 * @note    header-only，任何 .cu 文件 #include 后即可调用 launch_softmax(...)
 * @author  CUDA 30天学习项目
 * @date     2026-07-18
 ******************************************************************************/

#ifndef SOFTMAX_KERNEL_CUH
#define SOFTMAX_KERNEL_CUH

#include <cuda_runtime.h>
#include <float.h>   // FLT_MAX
#include <math.h>    // expf

/*
 * 共享的 warp-level 归约辅助函数。
 * 用 CUDA_KERNELS_WARP_REDUCE_DEFINED 宏做守卫，
 * 保证 softmax_kernel.cuh 与 layer_norm_kernel.cuh 同时被 #include 时
 * 这两个 __device__ 函数只定义一次，避免重复定义（redefinition）编译错误。
 */
#ifndef CUDA_KERNELS_WARP_REDUCE_DEFINED
#define CUDA_KERNELS_WARP_REDUCE_DEFINED

/**
 * @brief  Warp 内 max 归约（32 线程协作求最大值）
 * @param  val  本线程持有的候选值
 * @return float  归约后的最大值（lane 0 持有正确结果，其余 lane 也会得到相同值的一部分）
 *
 * @details __shfl_down_sync(mask, val, offset)：
 *          让当前 lane 读取 "lane + offset" 号线程寄存器中的 val。
 *          通过 offset = 16, 8, 4, 2, 1 的折半规约，
 *          经过 5 步后 lane 0 即持有整个 warp 的最大值。
 *          0xFFFFFFFF 表示 warp 内全部 32 个线程都参与本次 shuffle。
 */
__device__ __forceinline__ float warp_reduce_max(float val)
{
    // 折半规约：每一步比较距离 offset 的两个线程的值，取较大者
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/**
 * @brief  Warp 内 sum 归约（32 线程协作求和）
 * @param  val  本线程持有的部分和
 * @return float  归约后的总和（lane 0 持有正确结果）
 */
__device__ __forceinline__ float warp_reduce_sum(float val)
{
    // 折半规约：每一步把距离 offset 的线程的值累加进来
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

#endif /* CUDA_KERNELS_WARP_REDUCE_DEFINED */

/**
 * @brief  Softmax 前向 Kernel（每个 Block 负责一行）
 * @param  input   输入矩阵指针，形状 [rows, cols]，行优先（row-major）存储
 * @param  output  输出矩阵指针，形状与 input 相同
 * @param  rows    行数（一维输入时传 1）
 * @param  cols    每行元素个数（softmax 的归约维度）
 * @return void
 *
 * @details 线程组织：
 *          - grid.x = rows：每一行由一个 Block 独占；
 *          - block.x = blockDim.x（建议 128/256，且为 32 的倍数）；
 *          - 若 cols > blockDim.x，则每个线程用 grid-stride 方式处理多个元素。
 *
 *          共享内存布局：warp_shared[warpsPerBlock]，
 *          用于把各 warp 的局部归约结果汇总到第 0 号 warp 再做一次归约。
 */
__global__ void softmax_forward_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols)
{
    // 当前 Block 负责的行号
    int row = blockIdx.x;
    if (row >= rows) return;

    // 定位到该行的起始地址（行优先存储：第 row 行偏移 row * cols）
    const float* row_in = input + (size_t)row * cols;
    float*       row_out = output + (size_t)row * cols;

    int tid   = threadIdx.x;                 // Block 内线程编号
    int lane  = tid & 31;                    // warp 内 lane 编号（tid % 32）
    int wid   = tid >> 5;                     // warp 编号（tid / 32）
    int nWarp = (blockDim.x + 31) >> 5;       // 本 Block 的 warp 数量

    // 共享内存：先存 max 的 warp 结果，随后复用来存 sum 的 warp 结果
    extern __shared__ float warp_shared[];

    // ---------------- 第 1 步：求该行最大值 m ----------------
    // 每个线程用 grid-stride 遍历自己负责的列，维护线程局部最大值
    float thread_max = -FLT_MAX;
    for (int col = tid; col < cols; col += blockDim.x)
    {
        thread_max = fmaxf(thread_max, row_in[col]);
    }

    // warp 内先归约，再把各 warp 的结果写入共享内存
    thread_max = warp_reduce_max(thread_max);
    if (lane == 0) warp_shared[wid] = thread_max;
    __syncthreads();

    // 由第 0 号 warp 对所有 warp 的局部最大值再归约一次
    if (wid == 0)
    {
        float v = (lane < nWarp) ? warp_shared[lane] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane == 0) warp_shared[0] = v;   // 最终最大值放入 warp_shared[0]
    }
    __syncthreads();
    float row_max = warp_shared[0];          // 该行的最大值 m（所有线程共享读取）

    // ---------------- 第 2 步：求 sum = Σ exp(x - m) ----------------
    float thread_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x)
    {
        // 减去最大值再取指数，保证指数参数 <= 0，避免上溢
        thread_sum += expf(row_in[col] - row_max);
    }

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane == 0) warp_shared[wid] = thread_sum;
    __syncthreads();

    if (wid == 0)
    {
        float v = (lane < nWarp) ? warp_shared[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) warp_shared[0] = v;   // 最终求和结果放入 warp_shared[0]
    }
    __syncthreads();
    float row_sum = warp_shared[0];          // 该行的归一化分母 Σexp

    // ---------------- 第 3 步：逐元素写出归一化结果 ----------------
    // 用倒数乘法替代除法（除法较慢），inv_sum = 1 / Σexp
    float inv_sum = 1.0f / row_sum;
    for (int col = tid; col < cols; col += blockDim.x)
    {
        row_out[col] = expf(row_in[col] - row_max) * inv_sum;
    }
}

/**
 * @brief  Softmax 前向的主机端启动封装
 * @param  d_input   设备端输入指针，形状 [rows, cols]
 * @param  d_output  设备端输出指针，形状 [rows, cols]
 * @param  rows      行数（一维输入传 1）
 * @param  cols      每行元素个数
 * @param  block_size  每个 Block 的线程数（默认 256，须为 32 的倍数）
 * @param  stream    CUDA 流（默认 0，即默认流）
 * @return void
 *
 * @details 便于在其他 .cu 中一行调用。共享内存大小按 warp 数分配。
 */
static inline void launch_softmax(const float* d_input,
                                  float* d_output,
                                  int rows,
                                  int cols,
                                  int block_size = 256,
                                  cudaStream_t stream = 0)
{
    // 保证 block_size 至少为 32，且不超过硬件上限
    if (block_size < 32) block_size = 32;
    if (block_size > 1024) block_size = 1024;

    dim3 grid(rows);
    dim3 block(block_size);

    // 共享内存需容纳每个 warp 一个 float
    int nWarp = (block_size + 31) / 32;
    size_t shmem = (size_t)nWarp * sizeof(float);

    softmax_forward_kernel<<<grid, block, shmem, stream>>>(d_input, d_output, rows, cols);
}

#endif /* SOFTMAX_KERNEL_CUH */
