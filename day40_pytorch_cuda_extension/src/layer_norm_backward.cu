/******************************************************************************
 * @file    layer_norm_backward.cu
 * @brief   Day 40: Layer Normalization 反向 CUDA Kernel（简化版，演示概念）
 * @details LayerNorm 反向比前向复杂，因为 mean 和 var 都依赖整行输入，
 *          所以每个 x_i 的梯度都会牵连整行。设归一化维度为 D，令
 *              x_hat_i = (x_i - mean) * rstd
 *          对输入的梯度公式（对 gamma 加权后的上游梯度记为 g_i = grad_out_i * gamma_i）：
 *              grad_x_i = rstd * ( g_i - mean(g) - x_hat_i * mean(g * x_hat) )
 *          其中 mean(.) 表示对该行 D 个元素求平均。
 *
 *          本简化版：
 *            - 一个 Block 处理一行，用共享内存归约求 sum(g) 与 sum(g * x_hat)。
 *            - 只计算对 input 的梯度 grad_input（gamma / beta 的梯度在
 *              Python 封装层用 PyTorch 张量运算完成，聚焦演示核心概念）。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <cuda_runtime.h>

#define LN_BLOCK_SIZE 256

/**
 * @brief  LayerNorm 反向 Kernel（仅计算对 input 的梯度）
 * @param  grad_output  上游梯度 [rows, D]
 * @param  input        前向输入 [rows, D]
 * @param  gamma        缩放参数 [D]
 * @param  mean         前向保存的每行均值 [rows]
 * @param  rstd         前向保存的每行 1/sqrt(var+eps) [rows]
 * @param  grad_input   输出：对 input 的梯度 [rows, D]
 * @param  rows         行数
 * @param  D            归一化维度
 *
 * @details 步骤：
 *          1) 计算 x_hat_i、g_i = grad_out_i * gamma_i。
 *          2) 归约得到 sum(g) 与 sum(g * x_hat)。
 *          3) grad_x_i = rstd * ( g_i - sum(g)/D - x_hat_i * sum(g*x_hat)/D )。
 */
__global__ void layer_norm_backward_kernel(const float* grad_output,
                                           const float* input,
                                           const float* gamma,
                                           const float* mean,
                                           const float* rstd,
                                           float* grad_input,
                                           int rows,
                                           int D)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const float* go = grad_output + (size_t)row * D;
    const float* x  = input       + (size_t)row * D;
    float*       gx = grad_input  + (size_t)row * D;

    float m  = mean[row];
    float rs = rstd[row];

    __shared__ float s_g[LN_BLOCK_SIZE];      // 归约 sum(g)
    __shared__ float s_gx[LN_BLOCK_SIZE];     // 归约 sum(g * x_hat)
    __shared__ float s_sum_g;                 // 广播：整行 sum(g)
    __shared__ float s_sum_gx;                // 广播：整行 sum(g*x_hat)

    // ---- 步骤 1 + 2 局部累加 ----
    float local_g  = 0.0f;
    float local_gx = 0.0f;
    for (int i = tid; i < D; i += blockDim.x)
    {
        float x_hat = (x[i] - m) * rs;
        float g     = go[i] * gamma[i];
        local_g  += g;
        local_gx += g * x_hat;
    }
    s_g[tid]  = local_g;
    s_gx[tid] = local_gx;
    __syncthreads();

    // ---- 树形归约 ----
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_g[tid]  += s_g[tid + stride];
            s_gx[tid] += s_gx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        s_sum_g  = s_g[0];
        s_sum_gx = s_gx[0];
    }
    __syncthreads();

    // ---- 步骤 3：写回每个元素的梯度 ----
    float mean_g  = s_sum_g  / D;
    float mean_gx = s_sum_gx / D;
    for (int i = tid; i < D; i += blockDim.x)
    {
        float x_hat = (x[i] - m) * rs;
        float g     = go[i] * gamma[i];
        gx[i] = rs * (g - mean_g - x_hat * mean_gx);
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
 * @brief  LayerNorm 反向的 PyTorch 封装函数（简化版）
 * @param  grad_output  上游梯度
 * @param  input        前向输入
 * @param  gamma        缩放参数 [D]
 * @param  mean         前向保存的均值 [rows]
 * @param  rstd         前向保存的 rstd [rows]
 * @return {grad_input, grad_gamma, grad_beta}
 *
 * @details grad_input 由自定义 Kernel 计算；
 *          grad_gamma / grad_beta 用 PyTorch 张量算子直接求出（教学简化）：
 *              grad_gamma = sum_over_rows( grad_output * x_hat )
 *              grad_beta  = sum_over_rows( grad_output )
 */
std::vector<torch::Tensor> layer_norm_backward_cuda(torch::Tensor grad_output,
                                                    torch::Tensor input,
                                                    torch::Tensor gamma,
                                                    torch::Tensor mean,
                                                    torch::Tensor rstd)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(gamma);
    CHECK_INPUT(mean);
    CHECK_INPUT(rstd);

    int D    = static_cast<int>(input.size(-1));
    int rows = static_cast<int>(input.numel() / D);

    auto grad_input = torch::empty_like(input);

    dim3 grid(rows);
    dim3 block(LN_BLOCK_SIZE);

    layer_norm_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        rows, D);

    C10_CUDA_CHECK(cudaGetLastError());

    // grad_gamma / grad_beta 用 ATen 张量运算完成（简化演示）
    auto input_2d = input.reshape({rows, D});
    auto go_2d    = grad_output.reshape({rows, D});
    auto mean_col = mean.reshape({rows, 1});
    auto rstd_col = rstd.reshape({rows, 1});
    auto x_hat    = (input_2d - mean_col) * rstd_col;         // [rows, D]

    auto grad_gamma = (go_2d * x_hat).sum(0);                  // [D]
    auto grad_beta  = go_2d.sum(0);                            // [D]

    return {grad_input, grad_gamma, grad_beta};
}

#endif /* WITH_TORCH */
