/******************************************************************************
 * @file    gelu_kernel.cuh
 * @brief   GELU 激活函数 Kernel（Transformer / LLM 高频基础算子）
 * @details 本头文件提供 header-only 的 GELU 前向实现，可直接 #include 使用。
 *
 *          GELU（Gaussian Error Linear Unit）定义：
 *
 *              GELU(x) = x * Phi(x)
 *
 *          其中 Phi(x) 是标准正态分布的累积分布函数（CDF）。
 *          直接计算 Phi 需要误差函数 erf，代价较高，实践中有两种常见近似：
 *
 *          ┌──────────────┬────────────────────────────────────────────────┐
 *          │ 版本         │ 公式                                            │
 *          ├──────────────┼────────────────────────────────────────────────┤
 *          │ 精确/tanh 版 │ 0.5*x*(1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3)))│
 *          │ 近似/sigmoid │ x * sigmoid(1.702 * x)                          │
 *          └──────────────┴────────────────────────────────────────────────┘
 *
 *          【两种实现的差异】
 *          - tanh 版：即 GPT/BERT 论文使用的 "gelu_new"，精度更高，
 *            与真实 erf 版 GELU 几乎无差别，但含 tanh 与三次项，计算量略大；
 *          - sigmoid 版：也叫 "SiLU 风格近似"（1.702 是拟合常数），
 *            计算最快，但在两侧尾部与真实 GELU 有微小偏差，
 *            对推理精度不敏感的场景（如加速推理）常用。
 *
 *          【逐元素算子】
 *          GELU 是逐元素（element-wise）操作，无需归约，
 *          直接用 grid-stride 循环让每个线程处理若干元素即可。
 *
 * @note    header-only，可直接 #include 使用
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#ifndef GELU_KERNEL_CUH
#define GELU_KERNEL_CUH

#include <cuda_runtime.h>
#include <math.h>    // tanhf, expf

/**
 * @brief  GELU 精确（tanh）版单元素求值
 * @param  x  输入标量
 * @return float  GELU(x)
 *
 * @details 公式：0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
 *          常量 sqrt(2/pi) ≈ 0.7978845608。
 */
__device__ __forceinline__ float gelu_tanh(float x)
{
    const float kSqrt2OverPi = 0.7978845608028654f; // sqrt(2/pi)
    const float kCoeff        = 0.044715f;           // 三次项拟合系数
    float x3    = x * x * x;                         // x^3
    float inner = kSqrt2OverPi * (x + kCoeff * x3);  // tanh 内部参数
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * @brief  GELU 近似（sigmoid）版单元素求值
 * @param  x  输入标量
 * @return float  x * sigmoid(1.702 * x)
 *
 * @details sigmoid(z) = 1 / (1 + exp(-z))，此处 z = 1.702 * x。
 *          1.702 是使 x*sigmoid(1.702x) 逼近真实 GELU 的经验拟合常数。
 */
__device__ __forceinline__ float gelu_sigmoid(float x)
{
    const float kScale = 1.702f;
    float sig = 1.0f / (1.0f + expf(-kScale * x));   // sigmoid(1.702 * x)
    return x * sig;
}

/**
 * @brief  GELU 精确（tanh）版 Kernel
 * @param  input   输入数组
 * @param  output  输出数组
 * @param  n       元素总数
 * @return void
 *
 * @details 采用 grid-stride 循环，兼容任意大小的 n：
 *          每个线程从自己的全局索引出发，步长为总线程数，遍历所有归属元素。
 */
__global__ void gelu_tanh_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x; // 全局线程 ID
    int stride = gridDim.x * blockDim.x;                // 总线程数（grid-stride 步长）
    for (int i = idx; i < n; i += stride)
    {
        output[i] = gelu_tanh(input[i]);
    }
}

/**
 * @brief  GELU 近似（sigmoid）版 Kernel
 * @param  input   输入数组
 * @param  output  输出数组
 * @param  n       元素总数
 * @return void
 */
__global__ void gelu_sigmoid_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride)
    {
        output[i] = gelu_sigmoid(input[i]);
    }
}

/**
 * @brief  GELU 前向的主机端启动封装
 * @param  d_input   设备端输入数组
 * @param  d_output  设备端输出数组
 * @param  n         元素总数
 * @param  use_tanh  true=精确 tanh 版；false=近似 sigmoid 版（默认 true）
 * @param  block_size  每个 Block 线程数（默认 256）
 * @param  stream    CUDA 流（默认 0）
 * @return void
 *
 * @details grid 数量按元素数除以 block_size 上取整，并限制在合理范围内。
 */
static inline void launch_gelu(const float* d_input,
                               float* d_output,
                               int n,
                               bool use_tanh = true,
                               int block_size = 256,
                               cudaStream_t stream = 0)
{
    if (block_size < 32) block_size = 32;
    if (block_size > 1024) block_size = 1024;

    int grid = (n + block_size - 1) / block_size;
    if (grid < 1) grid = 1;
    // 限制 grid 上限，配合 grid-stride 循环仍可覆盖超大数组
    if (grid > 65535) grid = 65535;

    if (use_tanh)
        gelu_tanh_kernel<<<grid, block_size, 0, stream>>>(d_input, d_output, n);
    else
        gelu_sigmoid_kernel<<<grid, block_size, 0, stream>>>(d_input, d_output, n);
}

#endif /* GELU_KERNEL_CUH */
