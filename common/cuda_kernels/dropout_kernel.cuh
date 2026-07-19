/******************************************************************************
 * @file    dropout_kernel.cuh
 * @brief   Dropout Kernel（深度学习训练正则化高频算子）
 * @details 本头文件提供 header-only 的 Dropout 前向实现，可直接 #include 使用。
 *          随机数由 cuRAND 的设备端 API（curand_kernel.h）在 Kernel 内生成。
 *
 *          Dropout 原理：
 *          训练时，以概率 p 随机「丢弃」（置零）神经元输出，
 *          用于打破神经元之间的共适应，缓解过拟合。
 *          为了让训练与推理的期望输出一致，采用「Inverted Dropout」：
 *
 *              训练模式：
 *                  keep_prob = 1 - p
 *                  scale     = 1 / keep_prob
 *                  mask_i    = (rand_i > p) ? 1 : 0      // 以概率 p 丢弃
 *                  y_i       = x_i * mask_i * scale      // 保留者放大补偿
 *
 *              推理模式：
 *                  y_i = x_i                              // 恒等映射，不做任何丢弃
 *
 *          由于训练时对保留元素乘了 1/keep_prob 做补偿，
 *          推理时无需再缩放，直接透传即可，这就是 Inverted Dropout 的好处。
 *
 *          【种子管理】
 *          curand_init(seed, subsequence, offset, &state) 初始化每个线程的
 *          随机状态。为保证：
 *            - 不同线程产生「不相关」的随机序列 —— 用全局线程 ID 作为 subsequence；
 *            - 每次迭代（step）产生「不同」的掩码 —— 让上层每步传入递增的 seed
 *              （或固定 seed + 递增 offset）。
 *          若希望结果可复现，固定 seed 即可得到确定的掩码。
 *
 * @note    需要链接 cudart 即可（curand_kernel.h 为 header-only 设备端 API，
 *          无需额外链接 curand 库）。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#ifndef DROPOUT_KERNEL_CUH
#define DROPOUT_KERNEL_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>   // curandState, curand_init, curand_uniform

/**
 * @brief  Dropout 训练模式 Kernel（生成随机掩码并缩放）
 * @param  input     输入数组
 * @param  output    输出数组
 * @param  mask      输出掩码数组（可为 nullptr）：记录每个元素是否被保留（1.0 保留 / 0.0 丢弃）
 *                   反向传播时需要复用同一掩码，故通常需要保存。
 * @param  n         元素总数
 * @param  drop_prob 丢弃概率 p（取值 [0,1)）
 * @param  seed      随机种子（同一 seed 得到可复现的掩码）
 * @return void
 *
 * @details 每个线程用 grid-stride 处理多个元素，
 *          为每个线程独立初始化一份 curandState，
 *          用全局线程 ID 作为 subsequence 保证线程间随机序列互不相关。
 */
__global__ void dropout_train_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     float* __restrict__ mask,
                                     int n,
                                     float drop_prob,
                                     unsigned long long seed)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // keep_prob = 1 - p；scale = 1 / keep_prob（Inverted Dropout 补偿系数）
    float keep_prob = 1.0f - drop_prob;
    // 防御性处理：keep_prob 过小时避免除零 / 数值爆炸
    float scale = (keep_prob > 1e-8f) ? (1.0f / keep_prob) : 0.0f;

    // 为当前线程初始化随机状态：
    //   seed        —— 全局种子，控制整体可复现性
    //   subsequence —— 用线程 ID，使不同线程走不同的子序列（互不相关）
    //   offset      —— 0，从子序列起点开始取数
    curandState state;
    curand_init(seed, (unsigned long long)idx, 0ULL, &state);

    for (int i = idx; i < n; i += stride)
    {
        // curand_uniform 返回 (0, 1] 均匀分布随机数
        float r = curand_uniform(&state);
        // r > drop_prob 则保留（掩码=1），否则丢弃（掩码=0）
        float m = (r > drop_prob) ? 1.0f : 0.0f;
        if (mask != nullptr) mask[i] = m;
        // 训练输出 = 输入 * 掩码 * 缩放
        output[i] = input[i] * m * scale;
    }
}

/**
 * @brief  Dropout 推理模式 Kernel（恒等映射）
 * @param  input   输入数组
 * @param  output  输出数组
 * @param  n       元素总数
 * @return void
 *
 * @details 推理时不做任何丢弃，直接拷贝。
 *          因训练用的是 Inverted Dropout（已在训练时补偿缩放），此处无需缩放。
 */
__global__ void dropout_infer_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
    {
        output[i] = input[i];   // 恒等映射
    }
}

/**
 * @brief  Dropout 前向的主机端启动封装
 * @param  d_input     设备端输入数组
 * @param  d_output    设备端输出数组
 * @param  d_mask      设备端掩码数组（可为 nullptr），仅训练模式使用
 * @param  n           元素总数
 * @param  drop_prob   丢弃概率 p（[0,1)）
 * @param  training    true=训练模式（随机丢弃）；false=推理模式（恒等）
 * @param  seed        随机种子（默认 1234），固定值可复现掩码
 * @param  block_size  每个 Block 线程数（默认 256）
 * @param  stream      CUDA 流（默认 0）
 * @return void
 */
static inline void launch_dropout(const float* d_input,
                                  float* d_output,
                                  float* d_mask,
                                  int n,
                                  float drop_prob,
                                  bool training,
                                  unsigned long long seed = 1234ULL,
                                  int block_size = 256,
                                  cudaStream_t stream = 0)
{
    if (block_size < 32) block_size = 32;
    if (block_size > 1024) block_size = 1024;

    int grid = (n + block_size - 1) / block_size;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;

    if (training)
        dropout_train_kernel<<<grid, block_size, 0, stream>>>(
            d_input, d_output, d_mask, n, drop_prob, seed);
    else
        dropout_infer_kernel<<<grid, block_size, 0, stream>>>(
            d_input, d_output, n);
}

#endif /* DROPOUT_KERNEL_CUH */
