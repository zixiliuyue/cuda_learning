/******************************************************************************
 * @file    main.cu
 * @brief   Day 37: ctypes 调用 CUDA 动态库 - 将 CUDA Kernel 导出为 C 风格 API
 * @details 本文件不再包含 main() 入口函数，而是把 CUDA 计算能力封装成一组
 *          C 风格（extern "C"）的导出函数，编译为标准动态库：
 *              - Windows : cuda_ctypes.dll
 *              - Linux   : libcuda_ctypes.so
 *
 *          这样做的关键价值在于：Python 只需依赖【标准库 ctypes】即可直接
 *          调用 GPU 计算，无需安装 PyCUDA / Numba / PyTorch 等重量级依赖，
 *          实现「零外部依赖」的 GPU 加速调用。
 *
 *          核心学习目标：
 *          1. extern "C" —— 关闭 C++ 名称修饰（name mangling），保证符号名可预测
 *          2. __declspec(dllexport) / visibility("default") —— 显式导出符号
 *          3. 用 #ifdef _WIN32 宏统一跨平台导出关键字（EXPORT_API）
 *          4. 设计「以主机指针为输入输出」的稳定 ABI 接口，屏蔽 GPU 内存细节
 *          5. 返回错误码（0=成功, -1=失败）而不是抛异常，方便 ctypes 处理
 *
 *          导出的四个函数：
 *          - vector_add_cuda : GPU 向量加法 out = a + b
 *          - matmul_cuda     : GPU 方阵乘法   C = A * B
 *          - get_device_info : 把设备信息写入调用方提供的字符串缓冲区
 *          - free_cuda_mem   : 释放由本库分配的 GPU 内存（预留接口）
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

/* ============================================================
 *  跨平台导出宏 EXPORT_API
 * ------------------------------------------------------------
 *  Windows (MSVC/NVCC)：需要 __declspec(dllexport) 才会把符号
 *      写进 .dll 的导出表，否则外部无法通过名字找到该函数。
 *  Linux/macOS (GCC/Clang)：默认所有符号可见，但配合
 *      -fvisibility=hidden 时需要 visibility("default") 显式导出，
 *      这样能得到更干净、更小的导出符号表。
 * ============================================================ */
#ifdef _WIN32
    #define EXPORT_API __declspec(dllexport)
#else
    #define EXPORT_API __attribute__((visibility("default")))
#endif

/* ============================================================
 *  配置参数
 * ============================================================ */

/** @brief 每个 Block 的线程数（1D，用于向量加法） */
#define THREADS_PER_BLOCK 256

/** @brief 矩阵乘法使用的 2D Block 边长（16x16 = 256 线程/Block） */
#define TILE_DIM 16

/* ============================================================
 *  内部错误检查辅助（库内部使用，不导出）
 * ------------------------------------------------------------
 *  与其他 Day 项目里的 CUDA_SAFE_CALL 不同：动态库【绝不能】
 *  在出错时直接 exit() 杀死进程，否则会连带把调用方的 Python
 *  解释器一起终止。这里改为「打印错误 + 返回错误码」的策略。
 * ============================================================ */

/**
 * @brief  检查 CUDA 调用结果，出错时打印信息并返回非零
 * @param  err   CUDA API 返回的错误码
 * @param  msg   出错时附带的上下文描述
 * @return 0 表示成功，-1 表示失败
 */
static int cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA 错误] %s : %s\n", msg, cudaGetErrorString(err));
        fflush(stderr);
        return -1;
    }
    return 0;
}

/* ============================================================
 *  Kernel 函数（在 GPU 上执行，不导出）
 * ============================================================ */

/**
 * @brief  向量加法 Kernel: out[i] = a[i] + b[i]
 * @param  a    输入向量 A（设备内存）
 * @param  b    输入向量 B（设备内存）
 * @param  out  输出向量（设备内存）
 * @param  n    向量元素个数
 */
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n)
{
    // 全局线程 ID = 块索引 * 块内线程数 + 块内线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：多余线程直接跳过，防止越界访问
    if (tid < n)
    {
        out[tid] = a[tid] + b[tid];
    }
}

/**
 * @brief  朴素方阵乘法 Kernel: C = A * B（A、B、C 均为 width x width）
 * @param  a      输入矩阵 A（设备内存，行主序）
 * @param  b      输入矩阵 B（设备内存，行主序）
 * @param  c      输出矩阵 C（设备内存，行主序）
 * @param  width  方阵边长
 *
 * @details 每个线程负责计算输出矩阵 C 中的一个元素 C[row][col]，
 *          遍历 A 的第 row 行与 B 的第 col 列做点积。
 *          此处使用最直观的实现，重点在演示 ctypes 调用而非极致性能，
 *          若追求性能可参考 Day 6 的共享内存分块（tiled）实现。
 */
__global__ void matmul_kernel(const float* a, const float* b, float* c, int width)
{
    // 2D 线程索引映射到输出矩阵的 (row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width)
    {
        float sum = 0.0f;
        // 点积：A 的第 row 行 · B 的第 col 列
        for (int k = 0; k < width; ++k)
        {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

/* ============================================================
 *  导出的 C 风格 API（供 Python ctypes 调用）
 * ------------------------------------------------------------
 *  用 extern "C" 包裹，关闭 C++ 名称修饰。否则 C++ 编译器会把
 *  vector_add_cuda 变成类似 ?vector_add_cuda@@YAHPEBM0PEAMH@Z
 *  这样带类型信息的符号名，ctypes 将无法用原始函数名找到它。
 * ============================================================ */
extern "C" {

/**
 * @brief  在 GPU 上执行向量加法 out = a + b
 * @param  a    主机端输入向量 A（长度 n）
 * @param  b    主机端输入向量 B（长度 n）
 * @param  out  主机端输出缓冲区（长度 n），结果写入此处
 * @param  n    向量元素个数
 * @return 0 = 成功；-1 = 发生 CUDA 错误
 *
 * @details 完整流程（对调用方完全透明）：
 *          1. 在 GPU 上分配 d_a / d_b / d_out
 *          2. 把主机数据拷贝到设备 (H2D)
 *          3. 启动 Kernel 并行计算
 *          4. 把结果拷回主机 (D2H)
 *          5. 释放设备内存
 *          调用方（Python）只看到普通的主机指针，无需关心 GPU 细节。
 */
EXPORT_API int vector_add_cuda(const float* a, const float* b, float* out, int n)
{
    if (a == NULL || b == NULL || out == NULL || n <= 0)
    {
        fprintf(stderr, "[参数错误] vector_add_cuda: 空指针或 n<=0\n");
        return -1;
    }

    size_t bytes = (size_t)n * sizeof(float);
    float* d_a = NULL;
    float* d_b = NULL;
    float* d_out = NULL;
    int ret = -1;   // 默认失败，成功时置 0

    // 第 1 步：分配设备内存
    if (cuda_check(cudaMalloc((void**)&d_a, bytes), "cudaMalloc d_a")) goto cleanup;
    if (cuda_check(cudaMalloc((void**)&d_b, bytes), "cudaMalloc d_b")) goto cleanup;
    if (cuda_check(cudaMalloc((void**)&d_out, bytes), "cudaMalloc d_out")) goto cleanup;

    // 第 2 步：主机 -> 设备 (H2D)
    if (cuda_check(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "H2D a")) goto cleanup;
    if (cuda_check(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "H2D b")) goto cleanup;

    // 第 3 步：启动 Kernel
    {
        int grid = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        vector_add_kernel<<<grid, THREADS_PER_BLOCK>>>(d_a, d_b, d_out, n);
        if (cuda_check(cudaGetLastError(), "启动 vector_add_kernel")) goto cleanup;
        if (cuda_check(cudaDeviceSynchronize(), "同步 vector_add_kernel")) goto cleanup;
    }

    // 第 4 步：设备 -> 主机 (D2H)
    if (cuda_check(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost), "D2H out")) goto cleanup;

    ret = 0;   // 全部成功

cleanup:
    // 第 5 步：无论成功失败都释放设备内存，防止显存泄漏
    if (d_a)   cudaFree(d_a);
    if (d_b)   cudaFree(d_b);
    if (d_out) cudaFree(d_out);
    return ret;
}

/**
 * @brief  在 GPU 上执行方阵乘法 C = A * B
 * @param  a      主机端输入矩阵 A（width x width，行主序）
 * @param  b      主机端输入矩阵 B（width x width，行主序）
 * @param  c      主机端输出矩阵 C（width x width，行主序）
 * @param  width  方阵边长
 * @return 0 = 成功；-1 = 发生 CUDA 错误
 */
EXPORT_API int matmul_cuda(const float* a, const float* b, float* c, int width)
{
    if (a == NULL || b == NULL || c == NULL || width <= 0)
    {
        fprintf(stderr, "[参数错误] matmul_cuda: 空指针或 width<=0\n");
        return -1;
    }

    size_t elems = (size_t)width * (size_t)width;
    size_t bytes = elems * sizeof(float);
    float* d_a = NULL;
    float* d_b = NULL;
    float* d_c = NULL;
    int ret = -1;

    if (cuda_check(cudaMalloc((void**)&d_a, bytes), "cudaMalloc d_a")) goto cleanup;
    if (cuda_check(cudaMalloc((void**)&d_b, bytes), "cudaMalloc d_b")) goto cleanup;
    if (cuda_check(cudaMalloc((void**)&d_c, bytes), "cudaMalloc d_c")) goto cleanup;

    if (cuda_check(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice), "H2D a")) goto cleanup;
    if (cuda_check(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice), "H2D b")) goto cleanup;

    {
        // 2D 线程组织：每个 Block 处理 TILE_DIM x TILE_DIM 的输出子块
        dim3 block(TILE_DIM, TILE_DIM);
        dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
                  (width + TILE_DIM - 1) / TILE_DIM);
        matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, width);
        if (cuda_check(cudaGetLastError(), "启动 matmul_kernel")) goto cleanup;
        if (cuda_check(cudaDeviceSynchronize(), "同步 matmul_kernel")) goto cleanup;
    }

    if (cuda_check(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost), "D2H c")) goto cleanup;

    ret = 0;

cleanup:
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    return ret;
}

/**
 * @brief  查询当前 CUDA 设备信息，并写入调用方提供的字符串缓冲区
 * @param  buffer       调用方分配的字符缓冲区（由 Python 端 create_string_buffer 提供）
 * @param  buffer_size  缓冲区容量（字节）
 * @return 0 = 成功；-1 = 失败（无设备 / 缓冲区无效 / 查询出错）
 *
 * @details 这里演示「由调用方分配缓冲、库负责填充」的经典 C ABI 模式：
 *          库不返回堆指针，从而避免跨模块 malloc/free 不匹配的问题，
 *          内存所有权始终归调用方，最安全也最易被 ctypes 使用。
 *          使用 snprintf 严格限制写入长度，杜绝缓冲区溢出。
 */
EXPORT_API int get_device_info(char* buffer, int buffer_size)
{
    if (buffer == NULL || buffer_size <= 0)
    {
        return -1;
    }

    int device_count = 0;
    if (cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount"))
    {
        snprintf(buffer, buffer_size, "无法获取 CUDA 设备数量");
        return -1;
    }

    if (device_count == 0)
    {
        snprintf(buffer, buffer_size, "系统中未检测到 NVIDIA GPU 设备");
        return -1;
    }

    int dev = 0;
    cudaGetDevice(&dev);

    cudaDeviceProp prop;
    if (cuda_check(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties"))
    {
        snprintf(buffer, buffer_size, "查询设备属性失败");
        return -1;
    }

    // 把关键信息格式化进缓冲区（snprintf 保证不会溢出）
    snprintf(buffer, buffer_size,
             "设备名称=%s | 计算能力=%d.%d | 显存=%.2f GB | SM数=%d | Warp大小=%d",
             prop.name, prop.major, prop.minor,
             prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
             prop.multiProcessorCount, prop.warpSize);

    return 0;
}

/**
 * @brief  释放由本库分配并返回给调用方的 GPU 内存（预留接口）
 * @param  ptr  之前由本库通过 cudaMalloc 返回的设备指针
 *
 * @details 当前的三个计算函数都采用「内部分配、内部释放」的自洽模式，
 *          调用方无需管理 GPU 内存。此函数作为扩展预留：
 *          若未来提供「返回设备指针供多次复用」的接口，
 *          调用方必须通过本函数释放，而【不能】用 Python 的 del /
 *          C 的 free() ——因为 GPU 显存只能由 cudaFree 释放。
 *          这体现了动态库设计中「谁分配、谁提供释放接口」的原则。
 */
EXPORT_API void free_cuda_mem(void* ptr)
{
    if (ptr != NULL)
    {
        cudaFree(ptr);
    }
}

} /* extern "C" */
