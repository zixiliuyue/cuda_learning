/******************************************************************************
 * @file    layer_norm_forward.cu
 * @brief   Day 40: Layer Normalization 前向 CUDA Kernel（更复杂的例子）
 * @details LayerNorm 对每一行（最后一个维度 D）做归一化：
 *              mean = (1/D) * sum(x_i)
 *              var  = (1/D) * sum((x_i - mean)^2)
 *              y_i  = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
 *          其中 gamma / beta 是可学习的仿射参数（长度 D）。
 *
 *          实现策略：一个 Block 负责一行，Block 内线程做并行归约求 sum
 *          和 sum_sq，再广播 mean / rstd 给所有线程完成归一化。
 *          同时输出 mean 和 rstd(=1/sqrt(var+eps))，供反向传播复用，避免重算。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <cuda_runtime.h>

// 每行使用的线程数（一个 Block 处理一行）。
// 256 是折中值：足够并行，又不至于线程过多空转。
#define LN_BLOCK_SIZE 256

/**
 * @brief  LayerNorm 前向 Kernel
 * @param  input   输入 [rows, D]（行主序连续）
 * @param  gamma   缩放参数 [D]
 * @param  beta    偏移参数 [D]
 * @param  output  输出 [rows, D]
 * @param  mean    每行均值 [rows]（输出，供反向使用）
 * @param  rstd    每行 1/sqrt(var+eps) [rows]（输出，供反向使用）
 * @param  rows    行数
 * @param  D       每行元素个数（归一化维度）
 * @param  eps     防止除零的小常数
 *
 * @details 归约过程：
 *          1) 每个线程按 stride 累加自己负责的元素，得到局部 sum / sum_sq。
 *          2) 写入共享内存后做树形归约，thread 0 得到整行的 sum / sum_sq。
 *          3) 由 sum / sum_sq 计算 mean 和 var（E[x^2]-E[x]^2），存入共享内存广播。
 *          4) 所有线程用 mean / rstd 完成 (x-mean)*rstd*gamma+beta。
 */
__global__ void layer_norm_forward_kernel(const float* input,
                                          const float* gamma,
                                          const float* beta,
                                          float* output,
                                          float* mean,
                                          float* rstd,
                                          int rows,
                                          int D,
                                          float eps)
{
    int row = blockIdx.x;              // 当前 Block 负责的行号
    if (row >= rows) return;

    int tid = threadIdx.x;
    const float* x = input + (size_t)row * D;   // 指向本行起始
    float*       y = output + (size_t)row * D;

    // 两块共享内存：分别用于归约 sum 和 sum_sq
    __shared__ float s_sum[LN_BLOCK_SIZE];
    __shared__ float s_sqr[LN_BLOCK_SIZE];
    // 用于把 mean / rstd 从 thread 0 广播给全体线程
    __shared__ float s_mean;
    __shared__ float s_rstd;

    // ---- 步骤 1：每个线程累加自己负责的元素（网格跨步循环） ----
    float local_sum = 0.0f;
    float local_sqr = 0.0f;
    for (int i = tid; i < D; i += blockDim.x)
    {
        float v = x[i];
        local_sum += v;
        local_sqr += v * v;
    }
    s_sum[tid] = local_sum;
    s_sqr[tid] = local_sqr;
    __syncthreads();

    // ---- 步骤 2：树形归约，得到整行的 sum 与 sum_sq ----
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_sum[tid] += s_sum[tid + stride];
            s_sqr[tid] += s_sqr[tid + stride];
        }
        __syncthreads();
    }

    // ---- 步骤 3：thread 0 计算 mean / var / rstd ----
    if (tid == 0)
    {
        float m   = s_sum[0] / D;                 // 均值
        float var = s_sqr[0] / D - m * m;         // 方差 = E[x^2] - (E[x])^2
        float rs  = rsqrtf(var + eps);            // 1/sqrt(var+eps)
        s_mean = m;
        s_rstd = rs;
        mean[row] = m;   // 保存供反向使用
        rstd[row] = rs;
    }
    __syncthreads();

    // ---- 步骤 4：归一化 + 仿射变换 ----
    float m  = s_mean;
    float rs = s_rstd;
    for (int i = tid; i < D; i += blockDim.x)
    {
        float norm = (x[i] - m) * rs;
        y[i] = norm * gamma[i] + beta[i];
    }
}

/* ==========================================================================
 *  PyTorch 绑定层（仅 WITH_TORCH 时编译）
 * ========================================================================== */
#ifdef WITH_TORCH

#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x)       TORCH_CHECK((x).is_cuda(),        #x " 必须是 CUDA 张量")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(),  #x " 必须是内存连续的张量")
#define CHECK_FLOAT(x)      TORCH_CHECK((x).scalar_type() == at::kFloat, #x " 必须是 float32 张量")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

/**
 * @brief  LayerNorm 前向的 PyTorch 封装函数
 * @param  input  输入张量，最后一维为归一化维度 D
 * @param  gamma  缩放参数 [D]
 * @param  beta   偏移参数 [D]
 * @param  eps    数值稳定项
 * @return {output, mean, rstd} 三个张量的列表：
 *              output —— 归一化结果，与 input 同形状
 *              mean   —— 每行均值（展平后形状 [rows]）
 *              rstd   —— 每行 1/sqrt(var+eps)
 *
 * @details 把 input 视为 [rows, D]：D 取最后一维，rows = numel / D。
 *          mean / rstd 一并返回，反向传播直接复用，避免重复计算统计量。
 */
std::vector<torch::Tensor> layer_norm_forward_cuda(torch::Tensor input,
                                                   torch::Tensor gamma,
                                                   torch::Tensor beta,
                                                   double eps)
{
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);

    int D = static_cast<int>(input.size(-1));
    TORCH_CHECK(gamma.numel() == D, "gamma 长度必须等于归一化维度 D");
    TORCH_CHECK(beta.numel()  == D, "beta 长度必须等于归一化维度 D");

    int rows = static_cast<int>(input.numel() / D);

    auto output = torch::empty_like(input);
    // mean / rstd 每行一个标量
    auto opts = input.options();
    auto mean = torch::empty({rows}, opts);
    auto rstd = torch::empty({rows}, opts);

    // 每行一个 Block，Block 内 LN_BLOCK_SIZE 个线程
    dim3 grid(rows);
    dim3 block(LN_BLOCK_SIZE);

    layer_norm_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        rows, D, static_cast<float>(eps));

    C10_CUDA_CHECK(cudaGetLastError());
    return {output, mean, rstd};
}

#endif /* WITH_TORCH */
