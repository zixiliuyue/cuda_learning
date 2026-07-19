/******************************************************************************
 * @file    layer_norm_kernel.cuh
 * @brief   Layer Normalization 前向 Kernel（Transformer / LLM 高频基础算子）
 * @details 本头文件提供 header-only 的 LayerNorm 前向实现，可直接 #include 使用。
 *
 *          LayerNorm 数学定义（对每一行向量 x，长度为 D）：
 *
 *              mu       = (1/D) * Σ_i x_i                 // 均值
 *              var      = (1/D) * Σ_i (x_i - mu)^2        // 有偏方差
 *              x_hat_i  = (x_i - mu) / sqrt(var + eps)    // 标准化
 *              y_i      = gamma_i * x_hat_i + beta_i      // 仿射变换（可学习）
 *
 *          其中：
 *          - eps   ：数值稳定项，防止方差为 0 时除零（典型 1e-5）；
 *          - gamma ：可学习缩放参数（scale），长度为 D；
 *          - beta  ：可学习偏置参数（bias），长度为 D。
 *
 *          【与 BatchNorm 的区别】
 *          LayerNorm 在「特征维度（每一行内部）」做归一化，与 batch 大小无关，
 *          因此在序列长度可变的 NLP / Transformer 中比 BatchNorm 更常用。
 *
 *          【并行策略】
 *          每一行分配一个 Block。Block 内线程协作：
 *            1) 求 Σ x 与 Σ x^2（一次遍历同时累计，避免两遍数据读取）；
 *            2) 由 Σx、Σx² 反推 mu 与 var：
 *                  mu  = Σx / D
 *                  var = Σx²/D - mu²   （E[x²] - E[x]² 的定义式）
 *            3) 逐元素写出 y = gamma * (x - mu) * rstd + beta，
 *               其中 rstd = 1 / sqrt(var + eps)。
 *          归约采用 warp-level 的 __shfl_down_sync + 共享内存跨 warp 归约。
 *
 * @note    header-only，可与 softmax_kernel.cuh 一起 #include（warp 归约函数已做守卫）
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#ifndef LAYER_NORM_KERNEL_CUH
#define LAYER_NORM_KERNEL_CUH

#include <cuda_runtime.h>
#include <math.h>    // sqrtf

/*
 * 与 softmax_kernel.cuh 共享同一组 warp-level 归约辅助函数。
 * 若上游已定义（先 include 了 softmax_kernel.cuh），此处不再重复定义。
 */
#ifndef CUDA_KERNELS_WARP_REDUCE_DEFINED
#define CUDA_KERNELS_WARP_REDUCE_DEFINED

/**
 * @brief  Warp 内 max 归约（本文件未直接用到，与 softmax 保持一致的工具集）
 */
__device__ __forceinline__ float warp_reduce_max(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

/**
 * @brief  Warp 内 sum 归约（32 线程协作求和）
 */
__device__ __forceinline__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

#endif /* CUDA_KERNELS_WARP_REDUCE_DEFINED */

/**
 * @brief  LayerNorm 前向 Kernel（每个 Block 负责一行）
 * @param  input   输入矩阵，形状 [rows, cols]，行优先存储
 * @param  output  输出矩阵，形状与 input 相同
 * @param  gamma   可学习缩放参数，长度 cols；传 nullptr 时等价于全 1（不缩放）
 * @param  beta    可学习偏置参数，长度 cols；传 nullptr 时等价于全 0（不偏置）
 * @param  rows    行数（归一化的独立样本个数）
 * @param  cols    每行特征维度 D（归一化的维度）
 * @param  eps     数值稳定项，防止除零（典型 1e-5f）
 * @return void
 *
 * @details 线程组织：grid.x = rows，block.x = blockDim.x（建议 256）。
 *          共享内存需要 2 * nWarp 个 float：前半段汇总 Σx，后半段汇总 Σx²。
 */
__global__ void layer_norm_forward_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          int rows,
                                          int cols,
                                          float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_in  = input  + (size_t)row * cols;
    float*       row_out = output + (size_t)row * cols;

    int tid   = threadIdx.x;
    int lane  = tid & 31;               // warp 内 lane 编号
    int wid   = tid >> 5;               // warp 编号
    int nWarp = (blockDim.x + 31) >> 5; // Block 内 warp 数量

    // 共享内存：[0, nWarp)  存各 warp 的 Σx
    //           [nWarp, 2*nWarp) 存各 warp 的 Σx²
    extern __shared__ float smem[];
    float* s_sum  = smem;
    float* s_sqsum = smem + nWarp;

    // ---------------- 第 1 步：一次遍历同时累加 Σx 与 Σx² ----------------
    float thread_sum   = 0.0f;   // Σ x
    float thread_sqsum = 0.0f;   // Σ x²
    for (int col = tid; col < cols; col += blockDim.x)
    {
        float v = row_in[col];
        thread_sum   += v;
        thread_sqsum += v * v;
    }

    // warp 内归约
    thread_sum   = warp_reduce_sum(thread_sum);
    thread_sqsum = warp_reduce_sum(thread_sqsum);
    if (lane == 0)
    {
        s_sum[wid]   = thread_sum;
        s_sqsum[wid] = thread_sqsum;
    }
    __syncthreads();

    // 由第 0 号 warp 对各 warp 的部分和再归约一次
    if (wid == 0)
    {
        float vs = (lane < nWarp) ? s_sum[lane]   : 0.0f;
        float vq = (lane < nWarp) ? s_sqsum[lane] : 0.0f;
        vs = warp_reduce_sum(vs);
        vq = warp_reduce_sum(vq);
        if (lane == 0)
        {
            s_sum[0]   = vs;   // 整行 Σx
            s_sqsum[0] = vq;   // 整行 Σx²
        }
    }
    __syncthreads();

    // ---------------- 第 2 步：由 Σx、Σx² 计算 mu 与 rstd ----------------
    float inv_d = 1.0f / (float)cols;
    float mu    = s_sum[0] * inv_d;                 // 均值 mu = Σx / D
    // 方差 var = E[x²] - E[x]² = Σx²/D - mu²
    float var   = s_sqsum[0] * inv_d - mu * mu;
    // 浮点误差可能导致 var 出现极小负值，用 fmaxf 夹到 0 保证 sqrt 有效
    var = fmaxf(var, 0.0f);
    float rstd  = rsqrtf(var + eps);                // 1 / sqrt(var + eps)

    // ---------------- 第 3 步：逐元素写出仿射标准化结果 ----------------
    for (int col = tid; col < cols; col += blockDim.x)
    {
        float x_hat = (row_in[col] - mu) * rstd;    // 标准化
        // gamma / beta 允许为空指针：为空时退化为不缩放 / 不偏置
        float g = (gamma != nullptr) ? gamma[col] : 1.0f;
        float b = (beta  != nullptr) ? beta[col]  : 0.0f;
        row_out[col] = g * x_hat + b;
    }
}

/**
 * @brief  LayerNorm 前向的主机端启动封装
 * @param  d_input   设备端输入，形状 [rows, cols]
 * @param  d_output  设备端输出，形状 [rows, cols]
 * @param  d_gamma   设备端缩放参数（长度 cols），可为 nullptr
 * @param  d_beta    设备端偏置参数（长度 cols），可为 nullptr
 * @param  rows      行数
 * @param  cols      每行特征维度
 * @param  eps       数值稳定项（默认 1e-5f）
 * @param  block_size  每个 Block 线程数（默认 256）
 * @param  stream    CUDA 流（默认 0）
 * @return void
 */
static inline void launch_layer_norm(const float* d_input,
                                     float* d_output,
                                     const float* d_gamma,
                                     const float* d_beta,
                                     int rows,
                                     int cols,
                                     float eps = 1e-5f,
                                     int block_size = 256,
                                     cudaStream_t stream = 0)
{
    if (block_size < 32) block_size = 32;
    if (block_size > 1024) block_size = 1024;

    dim3 grid(rows);
    dim3 block(block_size);

    // 共享内存需 2 * nWarp 个 float（Σx 与 Σx² 各一段）
    int nWarp = (block_size + 31) / 32;
    size_t shmem = (size_t)2 * nWarp * sizeof(float);

    layer_norm_forward_kernel<<<grid, block, shmem, stream>>>(
        d_input, d_output, d_gamma, d_beta, rows, cols, eps);
}

#endif /* LAYER_NORM_KERNEL_CUH */
