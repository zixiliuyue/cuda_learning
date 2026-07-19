/******************************************************************************
 * @file    relu_forward.cu
 * @brief   Day 40: ReLU 前向传播 CUDA Kernel（PyTorch 自定义算子）
 * @details ReLU(x) = max(0, x) 是深度学习中最常用的激活函数。
 *          本文件包含两部分：
 *            1) 纯 CUDA Kernel（relu_forward_kernel）——不依赖 PyTorch，
 *               可被 CMake 编译成独立可执行文件用于验证逻辑。
 *            2) PyTorch 绑定层（仅在定义了 WITH_TORCH 宏时编译）——
 *               把 Kernel 包装成可被 Python 调用的算子，并在本文件统一
 *               注册整个扩展模块（PYBIND11_MODULE）。
 *
 * @note    编译方式：
 *            - PyTorch 扩展：由 setup.py 传入 -DWITH_TORCH 宏，编译绑定层。
 *            - 独立验证：CMake 不定义 WITH_TORCH，只编译裸 Kernel。
 * @author  CUDA 30天学习项目
 * @date     2026-07-18
 ******************************************************************************/

#include <cuda_runtime.h>

// 每个 Block 的线程数（256 是常用的经验值）
#define RELU_BLOCK_SIZE 256

/**
 * @brief  ReLU 前向 Kernel
 * @param  input   输入张量数据指针（device 内存）
 * @param  output  输出张量数据指针（device 内存）
 * @param  n       元素总数
 *
 * @details 采用「每个线程处理一个元素」的最简单映射方式：
 *          全局线程编号 tid = blockIdx.x * blockDim.x + threadIdx.x。
 *          若 tid 越界（tid >= n）则不做任何事，防止访问非法内存。
 *          计算逻辑：output[tid] = max(0, input[tid])。
 */
__global__ void relu_forward_kernel(const float* input, float* output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        float val = input[tid];
        // 大于 0 保留原值，否则置 0
        output[tid] = (val > 0.0f) ? val : 0.0f;
    }
}

/* ==========================================================================
 *  以下为 PyTorch 绑定层，仅在编译 PyTorch 扩展（定义 WITH_TORCH）时启用。
 *  CMake 独立验证编译时不会进入此段，因此不依赖 PyTorch 头文件。
 * ========================================================================== */
#ifdef WITH_TORCH

#include <torch/extension.h>

// ---- 张量合法性检查宏（教学用，实际项目可放到公共头文件） ----
#define CHECK_CUDA(x)       TORCH_CHECK((x).is_cuda(),        #x " 必须是 CUDA 张量")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(),  #x " 必须是内存连续的张量")
#define CHECK_FLOAT(x)      TORCH_CHECK((x).scalar_type() == at::kFloat, #x " 必须是 float32 张量")
#define CHECK_INPUT(x)      CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

/**
 * @brief  ReLU 前向的 PyTorch 封装函数
 * @param  input  输入张量（要求 CUDA、连续、float32）
 * @return 与 input 同形状的输出张量
 *
 * @details 这是 Python 侧 cuda_custom_ops.relu_forward(input) 的 C++ 实现。
 *          流程：检查输入 -> 分配输出 -> 计算 grid/block -> 启动 Kernel。
 */
torch::Tensor relu_forward_cuda(torch::Tensor input)
{
    CHECK_INPUT(input);

    // 分配与输入同形状、同类型、同设备的输出张量
    auto output = torch::empty_like(input);
    int n = static_cast<int>(input.numel());

    // 计算需要多少个 Block（向上取整）
    int grid = (n + RELU_BLOCK_SIZE - 1) / RELU_BLOCK_SIZE;

    relu_forward_kernel<<<grid, RELU_BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n);

    // 检查 Kernel 启动是否出错
    C10_CUDA_CHECK(cudaGetLastError());
    return output;
}

// ---- 其他源文件中定义的封装函数，在此前向声明以便统一注册 ----
torch::Tensor relu_backward_cuda(torch::Tensor grad_output, torch::Tensor input);
std::vector<torch::Tensor> layer_norm_forward_cuda(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps);
std::vector<torch::Tensor> layer_norm_backward_cuda(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor gamma,
    torch::Tensor mean, torch::Tensor rstd);

/**
 * @brief  注册整个扩展模块
 * @details 因为 setup.py 把多个 .cu 一起编译成同一个模块，
 *          PYBIND11_MODULE 只能出现一次，这里统一放在 relu_forward.cu 中。
 *          TORCH_EXTENSION_NAME 由编译系统自动定义为扩展名 cuda_custom_ops。
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Day 40: 自定义 CUDA 算子（ReLU / LayerNorm）前向与反向";
    m.def("relu_forward",        &relu_forward_cuda,        "ReLU 前向 (CUDA)");
    m.def("relu_backward",       &relu_backward_cuda,       "ReLU 反向 (CUDA)");
    m.def("layer_norm_forward",  &layer_norm_forward_cuda,  "LayerNorm 前向 (CUDA)");
    m.def("layer_norm_backward", &layer_norm_backward_cuda, "LayerNorm 反向 (CUDA)");
}

#endif /* WITH_TORCH */
