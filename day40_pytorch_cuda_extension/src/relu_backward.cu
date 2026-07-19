/******************************************************************************
 * @file    relu_backward.cu
 * @brief   Day 40: ReLU 反向传播 CUDA Kernel（PyTorch 自定义算子）
 * @details ReLU 的导数是一个「门控」：
 *              d/dx ReLU(x) = 1  (x > 0)
 *              d/dx ReLU(x) = 0  (x <= 0)
 *          因此反向传播时，把上游梯度 grad_output 按 (input > 0) 做遮罩：
 *              grad_input = (input > 0) ? grad_output : 0
 *          这演示了自动求导中「反向 Kernel 与前向 Kernel 一一对应」的思想。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <cuda_runtime.h>

#define RELU_BLOCK_SIZE 256

/**
 * @brief  ReLU 反向 Kernel
 * @param  grad_output  上游传来的梯度（对 ReLU 输出的梯度）
 * @param  input        前向时的输入（决定门控是否打开）
 * @param  grad_input   计算得到的对输入的梯度（输出）
 * @param  n            元素总数
 *
 * @details 每个线程处理一个元素。注意这里判断用的是「前向的 input」而非
 *          「前向的 output」——虽然对 ReLU 两者判断等价（output>0 <=> input>0），
 *          但保存 input 更贴合通用写法，反向时用它决定梯度是否通过。
 */
__global__ void relu_backward_kernel(const float* grad_output,
                                     const float* input,
                                     float* grad_input,
                                     int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        // 门控：input > 0 时梯度原样通过，否则被截断为 0
        grad_input[tid] = (input[tid] > 0.0f) ? grad_output[tid] : 0.0f;
    }
}

/* ==========================================================================
 *  PyTorch 绑定层（仅 WITH_TORCH 时编译）
 * ========================================================================== */
#ifdef WITH_TORCH

#include <torch/extension.h>

#define CHECK_CUDA(x)       TORCH_CHECK((x).is_cuda(),        #x " 必须是 CUDA 张量")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(),  #x " 必须是内存连续的张量")
#define CHECK_FLOAT(x)      TORCH_CHECK((x).scalar_type() == at::kFloat, #x " 必须是 float32 张量")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

/**
 * @brief  ReLU 反向的 PyTorch 封装函数
 * @param  grad_output  上游梯度
 * @param  input        前向保存的输入张量
 * @return grad_input   对输入的梯度
 *
 * @details 对应 Python 侧 cuda_custom_ops.relu_backward(grad_output, input)。
 *          在 autograd.Function.backward 中被调用。
 */
torch::Tensor relu_backward_cuda(torch::Tensor grad_output, torch::Tensor input)
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    TORCH_CHECK(grad_output.numel() == input.numel(),
                "grad_output 与 input 元素数量必须一致");

    auto grad_input = torch::empty_like(input);
    int n = static_cast<int>(input.numel());
    int grid = (n + RELU_BLOCK_SIZE - 1) / RELU_BLOCK_SIZE;

    relu_backward_kernel<<<grid, RELU_BLOCK_SIZE>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        n);

    C10_CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

#endif /* WITH_TORCH */
