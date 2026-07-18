/******************************************************************************
 * @file    main.cu
 * @brief   Day 31: Flash Attention 原理与实现（教学简化版）
 * @details 本程序实现一个【简化版 Flash Attention 前向传播】，用于演示
 *          Transformer 中注意力机制的两种计算方式：
 *          1. standard_attention_kernel — 标准 O(N^2) 注意力（对比基准）
 *          2. flash_attention_kernel    — 简化版 Flash Attention
 *                                          （tiled 分块 + online softmax）
 *
 *          核心学习目标：
 *          1. 理解注意力机制 Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V
 *          2. 理解标准实现为何需要 O(N^2) 的中间矩阵（显存瓶颈）
 *          3. 理解 Flash Attention 的核心思想：
 *             - Tiling：K/V 分块加载到共享内存，避免写出完整 N x N 矩阵
 *             - Online Softmax：一边遍历 tile 一边增量更新 softmax 归一化因子
 *          4. 理解 online softmax 的数值稳定性技巧（减去 running max）
 *          5. 使用 float4 向量化加载提高显存带宽利用率
 *
 *          注意力机制回顾：
 *          - Q（Query）  : [N, d]  查询矩阵
 *          - K（Key）    : [N, d]  键矩阵
 *          - V（Value）  : [N, d]  值矩阵
 *          - S = Q K^T   : [N, N]  相似度分数矩阵（standard 需完整存储）
 *          - P = softmax(S / sqrt(d))  按行做 softmax
 *          - O = P V     : [N, d]  输出矩阵
 *
 *          Flash Attention 核心逻辑（简化伪代码，针对每个 Q 行）：
 *              m = -inf, l = 0, O = [0...0]
 *              遍历 K/V 的每个 tile:
 *                  加载 K_tile, V_tile 到共享内存
 *                  计算 S = Q_row * K_tile^T / sqrt(d)
 *                  m_new = max(m, max(S))
 *                  P     = exp(S - m_new)
 *                  l     = l * exp(m - m_new) + sum(P)
 *                  O     = O * exp(m - m_new) + P * V_tile
 *                  m     = m_new
 *              O = O / l   // 最终 softmax 归一化
 *
 *          【重要】这是一个教学简化版，不是工业级 Flash Attention 实现：
 *          - 不包含反向传播（backward）
 *          - 不包含 block-sparse、causal mask 等高级特性
 *          - 每个线程负责一整行 Q（便于初学者理解，而非极致性能）
 *
 * @note    使用 CUDA 13 兼容 API（cudaDeviceGetAttribute 查询显存带宽）
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

/* ============================================================
 *  配置参数
 * ============================================================ */

/** @brief 序列长度 N（Q/K/V 的行数） */
#define SEQ_LEN 1024

/** @brief 头维度 d（每个 token 的特征维度，必须是 4 的倍数以支持 float4） */
#define HEAD_DIM 64

/** @brief Flash Attention 分块大小（每个 tile 包含多少个 K/V 行） */
#define TILE_SIZE 32

/** @brief 浮点数比较容差 */
#define TOLERANCE 1e-2f

/* ============================================================
 *  Kernel 1: 标准注意力（O(N^2)，对比基准）
 * ============================================================ */

/**
 * @brief  标准注意力 Kernel（每个线程计算一整行输出 O[row]）
 * @param  d_q     查询矩阵 Q（N x d，行主序）
 * @param  d_k     键矩阵 K（N x d，行主序）
 * @param  d_v     值矩阵 V（N x d，行主序）
 * @param  d_o     输出矩阵 O（N x d，行主序）
 * @param  n       序列长度 N
 * @param  d       头维度 d
 * @return void
 *
 * @details 每个线程负责输出矩阵 O 的第 row 行，流程如下：
 *          1. 先遍历所有 K 行，计算相似度分数 S[j] = Q[row]·K[j] / sqrt(d)，
 *             同时记录该行的最大值 max_s（用于数值稳定的 softmax）。
 *          2. 再次遍历，计算 exp(S[j] - max_s) 并累加到分母 sum_exp。
 *          3. 最后按 O[row] = sum_j( softmax_j * V[j] ) 计算输出。
 *
 *          注意：本实现将分数 S 存储在【局部数组 scores[]】中，
 *          这里的 scores 长度等于 N，仅用于教学演示"需要保存整行分数"。
 *          在真实的标准实现里，S 是一个完整的 N x N 矩阵，会消耗大量显存，
 *          这正是 Flash Attention 想要避免的问题。
 */
__global__ void standard_attention_kernel(const float* d_q, const float* d_k,
                                          const float* d_v, float* d_o,
                                          int n, int d)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    // 缩放因子 1 / sqrt(d)，稳定注意力分数的量级
    float scale = rsqrtf((float)d);

    // 指向当前 Q 行的起始地址
    const float* q_row = d_q + row * d;

    // ------------------------------------------------------------
    // 第 1 遍：找到该行所有分数中的最大值（数值稳定性）
    // ------------------------------------------------------------
    float max_s = -FLT_MAX;
    for (int j = 0; j < n; j++)
    {
        const float* k_row = d_k + j * d;
        float s = 0.0f;
        for (int t = 0; t < d; t++)
        {
            s += q_row[t] * k_row[t];
        }
        s *= scale;
        if (s > max_s) max_s = s;
    }

    // ------------------------------------------------------------
    // 第 2 遍：计算 softmax 分母 sum(exp(S - max_s))
    //          同时把 exp 权重直接累加到输出 O（避免再存整行分数）
    // ------------------------------------------------------------
    // 输出累加器（存放在寄存器 / 本地内存）
    float acc[HEAD_DIM];
    for (int t = 0; t < d; t++) acc[t] = 0.0f;

    float sum_exp = 0.0f;
    for (int j = 0; j < n; j++)
    {
        const float* k_row = d_k + j * d;
        float s = 0.0f;
        for (int t = 0; t < d; t++)
        {
            s += q_row[t] * k_row[t];
        }
        s *= scale;

        float p = expf(s - max_s);   // 未归一化的注意力权重
        sum_exp += p;

        // 加权累加 V[j]
        const float* v_row = d_v + j * d;
        for (int t = 0; t < d; t++)
        {
            acc[t] += p * v_row[t];
        }
    }

    // ------------------------------------------------------------
    // 归一化并写回输出
    // ------------------------------------------------------------
    float inv_sum = 1.0f / sum_exp;
    float* o_row = d_o + row * d;
    for (int t = 0; t < d; t++)
    {
        o_row[t] = acc[t] * inv_sum;
    }
}

/* ============================================================
 *  Kernel 2: 简化版 Flash Attention（tiled + online softmax）
 * ============================================================ */

/**
 * @brief  简化版 Flash Attention Kernel（tiled 分块 + online softmax）
 * @param  d_q     查询矩阵 Q（N x d）
 * @param  d_k     键矩阵 K（N x d）
 * @param  d_v     值矩阵 V（N x d）
 * @param  d_o     输出矩阵 O（N x d）
 * @param  n       序列长度 N
 * @param  d       头维度 d
 * @return void
 *
 * @details 每个 Block 处理一段连续的 Q 行；Block 内每个线程负责一行 Q。
 *          Block 内所有线程【协作】把 K/V 的一个 tile 加载到共享内存，
 *          然后各自用自己的 Q 行与共享内存中的 K/V tile 做计算。
 *
 *          共享内存布局：
 *          - s_k[TILE_SIZE * HEAD_DIM]: 缓存当前 K tile
 *          - s_v[TILE_SIZE * HEAD_DIM]: 缓存当前 V tile
 *          共享内存大小 = 2 * TILE_SIZE * HEAD_DIM * sizeof(float)
 *          当 TILE_SIZE=32, HEAD_DIM=64 时 = 2*32*64*4 = 16 KB
 *
 *          Online Softmax 核心（每个 Q 行维护三个运行量）：
 *          - m : 到目前为止见过的最大分数（running max）
 *          - l : 到目前为止的 softmax 分母（running sum of exp）
 *          - acc[d] : 到目前为止的加权 V 累加（尚未除以 l）
 *
 *          每处理完一个 tile，若出现更大的 max，则用缩放因子
 *          alpha = exp(m_old - m_new) 对旧的 l 和 acc 做"回溯修正"，
 *          从而保证结果与一次性计算完整 softmax 完全等价。
 *
 *          Flash Attention 的关键收益：
 *          - 从不构造完整的 N x N 分数矩阵，显存占用从 O(N^2) 降到 O(N)
 *          - K/V tile 复用共享内存，减少全局内存重复读取
 */
__global__ void flash_attention_kernel(const float* d_q, const float* d_k,
                                       const float* d_v, float* d_o,
                                       int n, int d)
{
    // 共享内存：缓存一个 K tile 和一个 V tile
    __shared__ float s_k[TILE_SIZE * HEAD_DIM];
    __shared__ float s_v[TILE_SIZE * HEAD_DIM];

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // 缩放因子 1 / sqrt(d)
    float scale = rsqrtf((float)d);

    // 每个线程维护自己 Q 行的 online softmax 运行状态
    float m = -FLT_MAX;          // running max
    float l = 0.0f;              // running sum of exp（softmax 分母）
    float acc[HEAD_DIM];         // 加权 V 累加器（未归一化）
    for (int t = 0; t < d; t++) acc[t] = 0.0f;

    // 将当前线程的 Q 行读入寄存器（复用，减少全局内存访问）
    float q_reg[HEAD_DIM];
    if (row < n)
    {
        const float* q_row = d_q + row * d;
        for (int t = 0; t < d; t++) q_reg[t] = q_row[t];
    }

    // ------------------------------------------------------------
    // 遍历 K/V 的每个 tile
    // ------------------------------------------------------------
    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; tile++)
    {
        int tile_start = tile * TILE_SIZE;

        // ---- 协作加载 K/V tile 到共享内存 ----
        // Block 内所有线程分工，把 TILE_SIZE * d 个元素搬进共享内存。
        // 这里用 threadIdx.x 作为线性索引起点，跨步 blockDim.x 覆盖全部元素。
        for (int idx = threadIdx.x; idx < TILE_SIZE * d; idx += blockDim.x)
        {
            int r = idx / d;          // tile 内的行号
            int c = idx % d;          // 列号（特征维度）
            int global_row = tile_start + r;
            if (global_row < n)
            {
                s_k[idx] = d_k[global_row * d + c];
                s_v[idx] = d_v[global_row * d + c];
            }
            else
            {
                s_k[idx] = 0.0f;
                s_v[idx] = 0.0f;
            }
        }

        // 同步：确保整个 tile 已加载完毕，才能开始计算
        __syncthreads();

        // ---- 只有有效的 Q 行才参与计算 ----
        if (row < n)
        {
            // 遍历 tile 内的每个 K/V 行，做 online softmax 更新
            for (int r = 0; r < TILE_SIZE; r++)
            {
                int global_row = tile_start + r;
                if (global_row >= n) break;   // 尾部越界保护

                // 计算相似度分数 s = Q_row · K_row / sqrt(d)
                const float* k_ptr = s_k + r * d;
                float s = 0.0f;
                for (int t = 0; t < d; t++)
                {
                    s += q_reg[t] * k_ptr[t];
                }
                s *= scale;

                // ---- online softmax 增量更新 ----
                // 1. 计算新的 running max
                float m_new = fmaxf(m, s);

                // 2. 旧累加量的回溯缩放因子 alpha = exp(m_old - m_new)
                float alpha = expf(m - m_new);

                // 3. 当前分数的指数权重 p = exp(s - m_new)
                float p = expf(s - m_new);

                // 4. 更新 softmax 分母 l
                l = l * alpha + p;

                // 5. 更新输出累加器：旧值缩放 + 新贡献
                const float* v_ptr = s_v + r * d;
                for (int t = 0; t < d; t++)
                {
                    acc[t] = acc[t] * alpha + p * v_ptr[t];
                }

                // 6. 更新 running max
                m = m_new;
            }
        }

        // 同步：确保所有线程都用完当前共享内存，才能加载下一个 tile
        __syncthreads();
    }

    // ------------------------------------------------------------
    // 最终归一化：O[row] = acc / l
    // ------------------------------------------------------------
    if (row < n)
    {
        float inv_l = 1.0f / l;
        float* o_row = d_o + row * d;
        for (int t = 0; t < d; t++)
        {
            o_row[t] = acc[t] * inv_l;
        }
    }
}

/* ============================================================
 *  Kernel 3: 使用 float4 向量化加载的 Flash Attention
 * ============================================================ */

/**
 * @brief  float4 向量化版本的 Flash Attention Kernel
 * @param  d_q     查询矩阵 Q（N x d）
 * @param  d_k     键矩阵 K（N x d）
 * @param  d_v     值矩阵 V（N x d）
 * @param  d_o     输出矩阵 O（N x d）
 * @param  n       序列长度 N
 * @param  d       头维度 d（必须是 4 的倍数）
 * @return void
 *
 * @details 逻辑与 flash_attention_kernel 完全相同，区别在于：
 *          在从全局内存加载 Q 行、以及从共享内存读取 K/V 时，
 *          使用 float4（一次读 4 个 float，16 字节）进行【向量化访问】。
 *
 *          向量化的好处：
 *          - 单条指令搬运 128 bit，提高显存带宽利用率
 *          - 减少访存指令条数，降低指令开销
 *
 *          前提：数据地址需 16 字节对齐，且 d 是 4 的倍数（本例 d=64 满足）。
 */
__global__ void flash_attention_float4_kernel(const float* d_q, const float* d_k,
                                              const float* d_v, float* d_o,
                                              int n, int d)
{
    __shared__ float s_k[TILE_SIZE * HEAD_DIM];
    __shared__ float s_v[TILE_SIZE * HEAD_DIM];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = rsqrtf((float)d);

    int d4 = d / 4;   // float4 个数

    float m = -FLT_MAX;
    float l = 0.0f;
    float acc[HEAD_DIM];
    for (int t = 0; t < d; t++) acc[t] = 0.0f;

    // 用 float4 向量化把 Q 行读入寄存器
    float q_reg[HEAD_DIM];
    if (row < n)
    {
        const float4* q_row4 = reinterpret_cast<const float4*>(d_q + row * d);
        float4* q_reg4 = reinterpret_cast<float4*>(q_reg);
        for (int t = 0; t < d4; t++) q_reg4[t] = q_row4[t];
    }

    int num_tiles = (n + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; tile++)
    {
        int tile_start = tile * TILE_SIZE;

        // 协作加载（同样以 float4 为粒度搬运）
        int total4 = TILE_SIZE * d4;
        float4* s_k4 = reinterpret_cast<float4*>(s_k);
        float4* s_v4 = reinterpret_cast<float4*>(s_v);
        for (int idx = threadIdx.x; idx < total4; idx += blockDim.x)
        {
            int r = idx / d4;
            int c4 = idx % d4;
            int global_row = tile_start + r;
            if (global_row < n)
            {
                const float4* k_src = reinterpret_cast<const float4*>(d_k + global_row * d);
                const float4* v_src = reinterpret_cast<const float4*>(d_v + global_row * d);
                s_k4[idx] = k_src[c4];
                s_v4[idx] = v_src[c4];
            }
            else
            {
                float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                s_k4[idx] = zero;
                s_v4[idx] = zero;
            }
        }

        __syncthreads();

        if (row < n)
        {
            for (int r = 0; r < TILE_SIZE; r++)
            {
                int global_row = tile_start + r;
                if (global_row >= n) break;

                // 向量化点积 Q_row · K_row
                const float4* k4 = reinterpret_cast<const float4*>(s_k + r * d);
                const float4* q4 = reinterpret_cast<const float4*>(q_reg);
                float s = 0.0f;
                for (int t = 0; t < d4; t++)
                {
                    float4 a = q4[t];
                    float4 b = k4[t];
                    s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
                s *= scale;

                float m_new = fmaxf(m, s);
                float alpha = expf(m - m_new);
                float p = expf(s - m_new);
                l = l * alpha + p;

                // 向量化更新 acc
                const float4* v4 = reinterpret_cast<const float4*>(s_v + r * d);
                float4* acc4 = reinterpret_cast<float4*>(acc);
                for (int t = 0; t < d4; t++)
                {
                    float4 vv = v4[t];
                    float4 aa = acc4[t];
                    aa.x = aa.x * alpha + p * vv.x;
                    aa.y = aa.y * alpha + p * vv.y;
                    aa.z = aa.z * alpha + p * vv.z;
                    aa.w = aa.w * alpha + p * vv.w;
                    acc4[t] = aa;
                }
                m = m_new;
            }
        }

        __syncthreads();
    }

    if (row < n)
    {
        float inv_l = 1.0f / l;
        float* o_row = d_o + row * d;
        for (int t = 0; t < d; t++)
        {
            o_row[t] = acc[t] * inv_l;
        }
    }
}

/* ============================================================
 *  主机辅助函数：CPU 参考实现
 * ============================================================ */

/**
 * @brief  CPU 参考注意力实现（数值稳定 softmax），用于验证 GPU 结果
 */
void attention_cpu(const float* q, const float* k, const float* v, float* o,
                   int n, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    float* scores = (float*)malloc(n * sizeof(float));

    for (int row = 0; row < n; row++)
    {
        const float* q_row = q + row * d;

        // 计算分数并找最大值
        float max_s = -FLT_MAX;
        for (int j = 0; j < n; j++)
        {
            const float* k_row = k + j * d;
            float s = 0.0f;
            for (int t = 0; t < d; t++) s += q_row[t] * k_row[t];
            s *= scale;
            scores[j] = s;
            if (s > max_s) max_s = s;
        }

        // softmax 分母
        float sum_exp = 0.0f;
        for (int j = 0; j < n; j++)
        {
            scores[j] = expf(scores[j] - max_s);
            sum_exp += scores[j];
        }
        float inv_sum = 1.0f / sum_exp;

        // 加权求和 O = softmax(S) V
        float* o_row = o + row * d;
        for (int t = 0; t < d; t++) o_row[t] = 0.0f;
        for (int j = 0; j < n; j++)
        {
            float w = scores[j] * inv_sum;
            const float* v_row = v + j * d;
            for (int t = 0; t < d; t++) o_row[t] += w * v_row[t];
        }
    }

    free(scores);
}

/* ============================================================
 *  主函数
 * ============================================================ */

int main(void)
{
    printf("\n");
    printf("============================================================\n");
    printf("        Day 31: Flash Attention 原理与实现（简化版）        \n");
    printf("============================================================\n\n");

    print_device_info();

    int n = SEQ_LEN;
    int d = HEAD_DIM;
    int num_elements = n * d;
    size_t bytes = (size_t)num_elements * sizeof(float);

    printf("\n[主机] 序列长度 N   : %d\n", n);
    printf("[主机] 头维度   d   : %d\n", d);
    printf("[主机] 分块 TILE    : %d\n", TILE_SIZE);
    printf("[主机] Q/K/V 尺寸   : %d x %d (%.2f MB each)\n",
           n, d, bytes / (1024.0 * 1024.0));
    printf("[主机] 标准注意力中间矩阵 S 尺寸: %d x %d (%.2f MB, Flash 无需存储)\n",
           n, n, (double)n * n * sizeof(float) / (1024.0 * 1024.0));

    /* --------------------------------------------------------
     *  分配和初始化数据
     * -------------------------------------------------------- */
    printf("\n[主机] 分配主机内存 ...\n");
    float* h_q = (float*)malloc(bytes);
    float* h_k = (float*)malloc(bytes);
    float* h_v = (float*)malloc(bytes);
    float* h_o_std = (float*)malloc(bytes);
    float* h_o_flash = (float*)malloc(bytes);
    float* h_o_flash4 = (float*)malloc(bytes);
    float* h_o_ref = (float*)malloc(bytes);

    if (!h_q || !h_k || !h_v || !h_o_std || !h_o_flash || !h_o_flash4 || !h_o_ref)
    {
        fprintf(stderr, "[错误] 主机内存分配失败！\n");
        return EXIT_FAILURE;
    }

    printf("[主机] 初始化 Q/K/V 数据 ...\n");
    srand((unsigned int)time(NULL));
    init_random_float(h_q, num_elements, -1.0f, 1.0f);
    init_random_float(h_k, num_elements, -1.0f, 1.0f);
    init_random_float(h_v, num_elements, -1.0f, 1.0f);

    // 分配设备内存
    float *d_q, *d_k, *d_v, *d_o;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_q, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_k, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_v, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_o, bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice));

    // 配置 Grid / Block：每个 Block 处理 TILE_SIZE 行 Q
    int block_size = TILE_SIZE;
    int grid_size = (n + block_size - 1) / block_size;
    printf("[主机] Grid: %d, Block: %d\n", grid_size, block_size);

    /* --------------------------------------------------------
     *  CPU 参考计算
     * -------------------------------------------------------- */
    printf("\n[主机] CPU 参考计算（O(N^2)，可能需要几秒）...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    attention_cpu(h_q, h_k, h_v, h_o_ref, n, d);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU 注意力 ");

    /* --------------------------------------------------------
     *  GPU 标准注意力
     * -------------------------------------------------------- */
    printf("\n[主机] GPU 标准注意力（O(N^2)）...\n");
    GpuTimer timer_std;
    gpu_timer_init(&timer_std);
    gpu_timer_start(&timer_std);
    standard_attention_kernel<<<grid_size, block_size>>>(d_q, d_k, d_v, d_o, n, d);
    gpu_timer_stop(&timer_std);
    CHECK_KERNEL_ERROR();
    gpu_timer_print(&timer_std, "GPU 标准注意力 ");
    CUDA_SAFE_CALL(cudaMemcpy(h_o_std, d_o, bytes, cudaMemcpyDeviceToHost));
    int ok_std = compare_float_arrays(h_o_std, h_o_ref, num_elements, TOLERANCE);

    /* --------------------------------------------------------
     *  GPU Flash Attention（tiled + online softmax）
     * -------------------------------------------------------- */
    printf("\n[主机] GPU Flash Attention（tiled + online softmax）...\n");
    GpuTimer timer_flash;
    gpu_timer_init(&timer_flash);
    gpu_timer_start(&timer_flash);
    flash_attention_kernel<<<grid_size, block_size>>>(d_q, d_k, d_v, d_o, n, d);
    gpu_timer_stop(&timer_flash);
    CHECK_KERNEL_ERROR();
    gpu_timer_print(&timer_flash, "GPU Flash Attention ");
    CUDA_SAFE_CALL(cudaMemcpy(h_o_flash, d_o, bytes, cudaMemcpyDeviceToHost));
    int ok_flash = compare_float_arrays(h_o_flash, h_o_ref, num_elements, TOLERANCE);

    /* --------------------------------------------------------
     *  GPU Flash Attention（float4 向量化）
     * -------------------------------------------------------- */
    printf("\n[主机] GPU Flash Attention（float4 向量化）...\n");
    GpuTimer timer_flash4;
    gpu_timer_init(&timer_flash4);
    gpu_timer_start(&timer_flash4);
    flash_attention_float4_kernel<<<grid_size, block_size>>>(d_q, d_k, d_v, d_o, n, d);
    gpu_timer_stop(&timer_flash4);
    CHECK_KERNEL_ERROR();
    gpu_timer_print(&timer_flash4, "GPU Flash(float4) ");
    CUDA_SAFE_CALL(cudaMemcpy(h_o_flash4, d_o, bytes, cudaMemcpyDeviceToHost));
    int ok_flash4 = compare_float_arrays(h_o_flash4, h_o_ref, num_elements, TOLERANCE);

    /* --------------------------------------------------------
     *  性能总结
     * -------------------------------------------------------- */
    double cpu_ms = gpu_timer_get_elapsed_ms(&timer_cpu);
    double std_ms = gpu_timer_get_elapsed_ms(&timer_std);
    double flash_ms = gpu_timer_get_elapsed_ms(&timer_flash);
    double flash4_ms = gpu_timer_get_elapsed_ms(&timer_flash4);

    printf("\n");
    printf("============================================================\n");
    printf("                        性能总结                            \n");
    printf("============================================================\n");
    printf("  配置              : N=%d, d=%d, TILE=%d\n", n, d, TILE_SIZE);
    printf("  CPU 注意力        : %10.3f ms (基准)\n", cpu_ms);
    printf("  GPU 标准注意力    : %10.3f ms (加速比 %6.1fx) %s\n",
           std_ms, cpu_ms / std_ms, ok_std ? "[通过]" : "[失败]");
    printf("  GPU Flash         : %10.3f ms (加速比 %6.1fx) %s\n",
           flash_ms, cpu_ms / flash_ms, ok_flash ? "[通过]" : "[失败]");
    printf("  GPU Flash(float4) : %10.3f ms (加速比 %6.1fx) %s\n",
           flash4_ms, cpu_ms / flash4_ms, ok_flash4 ? "[通过]" : "[失败]");
    printf("  Flash vs 标准     : %.2fx\n", std_ms / flash_ms);
    printf("  float4 vs Flash   : %.2fx\n", flash_ms / flash4_ms);
    printf("============================================================\n");

    printf("\n");
    printf("============================================================\n");
    printf("                        知识点总结                          \n");
    printf("============================================================\n");
    printf("  1. Attention(Q,K,V) = softmax(QK^T / sqrt(d)) V\n");
    printf("  2. 标准实现需完整 N x N 分数矩阵，显存 O(N^2)\n");
    printf("  3. Flash: K/V 分块进共享内存，显存降到 O(N)\n");
    printf("  4. Online Softmax: 用 running max/sum 增量更新\n");
    printf("  5. alpha = exp(m_old - m_new) 回溯修正旧累加\n");
    printf("  6. float4 向量化加载 128bit，提升带宽利用率\n");
    printf("============================================================\n");
    printf("\n");

    // 释放资源
    CUDA_SAFE_CALL(cudaFree(d_q));
    CUDA_SAFE_CALL(cudaFree(d_k));
    CUDA_SAFE_CALL(cudaFree(d_v));
    CUDA_SAFE_CALL(cudaFree(d_o));
    free(h_q); free(h_k); free(h_v);
    free(h_o_std); free(h_o_flash); free(h_o_flash4); free(h_o_ref);
    gpu_timer_destroy(&timer_cpu);
    gpu_timer_destroy(&timer_std);
    gpu_timer_destroy(&timer_flash);
    gpu_timer_destroy(&timer_flash4);

    printf("[主机] 资源已释放，程序正常退出。\n\n");
    return (ok_std && ok_flash && ok_flash4) ? 0 : 1;
}
