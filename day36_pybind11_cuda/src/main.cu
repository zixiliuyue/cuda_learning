/******************************************************************************
 * @file    main.cu
 * @brief   Day 36: pybind11 CUDA 封装调用
 * @details 本项目演示如何使用 pybind11 将 CUDA Kernel 编译为一个
 *          Python 可以直接 import 的扩展模块（Windows 上是 .pyd，
 *          Linux 上是 .so）。这样 Python 端就能像调用普通函数一样，
 *          直接把 NumPy 数组传给 GPU 计算并拿回结果。
 *
 *          核心学习目标：
 *          1. pybind11 的基本用法：PYBIND11_MODULE 宏导出函数
 *          2. py::array_t<float> 与 NumPy 数组的零拷贝互操作
 *          3. 主机端包装函数：从 NumPy 取指针 -> 拷贝到 GPU
 *             -> 启动 Kernel -> 拷回结果 -> 封装为 NumPy 返回
 *          4. 两个经典 Kernel：向量加法 + Tiled 矩阵乘法（共享内存）
 *          5. 用 FetchContent 自动拉取 pybind11，无需手动安装依赖
 *
 *          三种 CUDA + Python 绑定方式对比（详见 README.md）：
 *          - Numba（@cuda.jit）        ：纯 Python，最易上手
 *          - PyTorch cpp_extension      ：深度学习框架集成，支持自动求导
 *          - pybind11（本项目）         ：通用 C++/CUDA 封装，最灵活
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>   // py::array_t 及 NumPy 互操作支持

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdexcept>
#include <string>

// pybind11 命名空间惯用别名
namespace py = pybind11;

/* ============================================================
 *  配置参数
 * ============================================================ */

/** @brief 一维 Kernel 每个 Block 的线程数 */
#define BLOCK_SIZE 256

/** @brief 矩阵乘法 Tile 宽度（共享内存分块） */
#define TILE_WIDTH 16

/* ============================================================
 *  CUDA 错误检查（本项目内联版本）
 *
 *  说明：这里不直接用 common/cuda_utils.h 中的 exit() 版本，
 *  因为在 Python 扩展中直接 exit() 会让整个解释器崩溃。
 *  更友好的做法是抛出 C++ 异常，pybind11 会自动把它转换成
 *  Python 的 RuntimeError，用户可以在 Python 里 try/except。
 * ============================================================ */

/**
 * @brief  检查 CUDA API 返回值，出错时抛出 std::runtime_error
 * @param  err   CUDA 错误码
 * @param  msg   附加的上下文描述
 */
static inline void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        // 拼接错误信息，抛出异常交给 pybind11 转成 Python 异常
        std::string full = std::string("[CUDA 错误] ") + msg +
                           " : " + cudaGetErrorString(err);
        throw std::runtime_error(full);
    }
}

/** @brief 便捷宏：包裹任意返回 cudaError_t 的调用 */
#define CUDA_CHECK(call) cuda_check((call), #call)

/* ============================================================
 *  Kernel 1: 向量加法
 * ============================================================ */

/**
 * @brief  向量加法 Kernel：c[i] = a[i] + b[i]
 * @param  a  输入向量 A
 * @param  b  输入向量 B
 * @param  c  输出向量 C
 * @param  n  元素个数
 *
 * @details 每个线程负责计算输出向量中的一个元素。
 *          通过全局线程 ID 定位到自己要处理的下标，
 *          边界判断 (tid < n) 防止越界访问。
 */
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = a[tid] + b[tid];
    }
}

/* ============================================================
 *  Kernel 2: Tiled 矩阵乘法（共享内存优化）
 * ============================================================ */

/**
 * @brief  Tiled 矩阵乘法 Kernel：C = A x B（均为 width x width 方阵）
 * @param  a      输入矩阵 A（行主序）
 * @param  b      输入矩阵 B（行主序）
 * @param  c      输出矩阵 C（行主序）
 * @param  width  矩阵维度
 *
 * @details 采用 Tiling（分块）技术，把 A、B 的子块先加载到共享内存，
 *          Block 内所有线程共享这份数据，从而大幅减少对全局内存的
 *          重复读取。全局内存访问次数从 O(N^3) 降到 O(N^3/TILE_WIDTH)。
 *
 *          两次 __syncthreads() 是关键：
 *          - 第一次：保证所有线程都完成共享内存加载后再开始计算
 *          - 第二次：保证所有线程都算完后再进入下一轮（防止数据被覆盖）
 */
__global__ void matmul_kernel(const float* a, const float* b, float* c, int width)
{
    // 每个 Block 独有一份共享内存副本
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];

    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    float sum = 0.0f;

    // 遍历所有 Tile
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; m++)
    {
        int a_col = m * TILE_WIDTH + threadIdx.x;
        int b_row = m * TILE_WIDTH + threadIdx.y;

        // 协作加载 A 的子块（越界填 0）
        if (row < width && a_col < width)
            ds_a[threadIdx.y][threadIdx.x] = a[row * width + a_col];
        else
            ds_a[threadIdx.y][threadIdx.x] = 0.0f;

        // 协作加载 B 的子块（越界填 0）
        if (b_row < width && col < width)
            ds_b[threadIdx.y][threadIdx.x] = b[b_row * width + col];
        else
            ds_b[threadIdx.y][threadIdx.x] = 0.0f;

        // 等待所有线程完成加载
        __syncthreads();

        // 使用共享内存中的数据累加部分积
        for (int k = 0; k < TILE_WIDTH; k++)
            sum += ds_a[threadIdx.y][k] * ds_b[k][threadIdx.x];

        // 等待所有线程完成计算再进入下一轮
        __syncthreads();
    }

    if (row < width && col < width)
        c[row * width + col] = sum;
}

/* ============================================================
 *  主机端包装函数（供 Python 调用）
 * ============================================================ */

/**
 * @brief  向量加法的主机包装函数
 * @param  a  Python 传入的 NumPy 一维数组（float32）
 * @param  b  Python 传入的 NumPy 一维数组（float32）
 * @return    计算结果，作为新的 NumPy 数组返回
 *
 * @details 流程：
 *          1. 通过 request() 拿到 NumPy 缓冲区描述（指针、维度、大小）
 *          2. 校验形状是否一致
 *          3. cudaMalloc 分配显存，cudaMemcpy 上传数据
 *          4. 启动 Kernel
 *          5. 分配一个新的 py::array_t 作为输出，把结果拷回
 *          6. 释放显存，返回结果
 */
py::array_t<float> vector_add(py::array_t<float> a, py::array_t<float> b)
{
    // buffer_info 描述了 NumPy 数组的底层内存布局
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();

    // 校验：两个输入必须一维且长度一致
    if (buf_a.ndim != 1 || buf_b.ndim != 1)
        throw std::runtime_error("[输入错误] vector_add 需要一维数组");
    if (buf_a.size != buf_b.size)
        throw std::runtime_error("[输入错误] 两个向量长度必须相同");

    int n = static_cast<int>(buf_a.size);
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // 分配输出 NumPy 数组（长度为 n 的一维数组）
    auto result = py::array_t<float>(n);
    py::buffer_info buf_c = result.request();

    // 拿到三个数组的原始指针
    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_c = static_cast<float*>(buf_c.ptr);

    // 分配显存
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 上传数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_a, ptr_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, ptr_b, bytes, cudaMemcpyHostToDevice));

    // 启动 Kernel
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());       // 检查启动配置错误
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待执行完成并捕获运行时错误

    // 拷回结果
    CUDA_CHECK(cudaMemcpy(ptr_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // 释放显存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}

/**
 * @brief  矩阵乘法的主机包装函数
 * @param  a      NumPy 数组，表示 width x width 矩阵 A（可为一维展平或二维）
 * @param  b      NumPy 数组，表示 width x width 矩阵 B
 * @param  width  矩阵维度
 * @return        结果矩阵 C，作为一维 NumPy 数组返回（长度 width*width）
 *
 * @details 为了接口简单，这里约定输入按行主序展平，输出也返回展平的
 *          一维数组，Python 端可用 numpy.reshape(width, width) 还原。
 */
py::array_t<float> matmul(py::array_t<float> a, py::array_t<float> b, int width)
{
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();

    size_t expected = static_cast<size_t>(width) * width;
    if (static_cast<size_t>(buf_a.size) != expected ||
        static_cast<size_t>(buf_b.size) != expected)
    {
        throw std::runtime_error(
            "[输入错误] matmul 要求 A、B 元素数均为 width*width");
    }

    size_t bytes = expected * sizeof(float);

    // 输出为长度 width*width 的一维数组
    auto result = py::array_t<float>(static_cast<py::ssize_t>(expected));
    py::buffer_info buf_c = result.request();

    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_c = static_cast<float*>(buf_c.ptr);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, ptr_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, ptr_b, bytes, cudaMemcpyHostToDevice));

    // 配置二维 Grid / Block
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    dim3 grid_dim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                  (width + TILE_WIDTH - 1) / TILE_WIDTH);

    matmul_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(ptr_c, d_c, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return result;
}

/**
 * @brief  返回当前 GPU 设备名称（演示如何导出返回字符串的函数）
 * @return 设备名字符串；若无 GPU 则返回提示信息
 */
std::string device_name()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return std::string("未检测到 CUDA 设备");

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        return std::string("无法查询设备属性");

    return std::string(prop.name);
}

/* ============================================================
 *  PYBIND11_MODULE：定义 Python 模块
 *
 *  第一个参数 cuda_pybind 必须与 CMake 中 pybind11_add_module()
 *  的目标名一致，也就是最终 import 时使用的模块名。
 *  第二个参数 m 是代表该模块的对象。
 * ============================================================ */
PYBIND11_MODULE(cuda_pybind, m)
{
    // 模块级文档字符串（Python 中 help(cuda_pybind) 可见）
    m.doc() = "Day 36: 使用 pybind11 封装的 CUDA 计算模块";

    // 导出函数：m.def("Python中的名字", 函数指针, "文档", 参数名...)
    m.def("vector_add", &vector_add,
          "GPU 向量加法：返回 a + b（NumPy float32 一维数组）",
          py::arg("a"), py::arg("b"));

    m.def("matmul", &matmul,
          "GPU Tiled 矩阵乘法：返回展平的 C = A x B（长度 width*width）",
          py::arg("a"), py::arg("b"), py::arg("width"));

    m.def("device_name", &device_name,
          "返回当前 CUDA 设备名称");

    // 也可以导出常量属性
    m.attr("__version__") = "1.0.0";
}
