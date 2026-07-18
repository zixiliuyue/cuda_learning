/******************************************************************************
 * @file    main.cu
 * @brief   Day 27: N-Body 物理模拟 — 万有引力全对力计算
 * @details 本程序在 GPU 上模拟 N 个天体在万有引力作用下的运动。
 *          每个天体受到其他所有天体的引力，复杂度 O(N^2)。
 *
 *          核心学习目标：
 *          1. 理解 N-Body 问题的物理模型和并行化方法
 *          2. 掌握 float4 向量类型的优势（128-bit 对齐加载）
 *          3. 学会使用 rsqrtf() 快速倒数平方根优化除法
 *          4. 理解软化因子（SOFTENING）防止除零的物理意义
 *          5. 综合应用 CUDA 优化技术（合并访问、向量化、循环展开）
 *
 *          物理模型：
 *          牛顿万有引力定律：F = G * m1 * m2 / r^2
 *          加速度：a = F / m = G * m_other / r^2
 *          力的分量：F_x = G * m1 * m2 * (dx / r^3)  ← 注意是 r^3 不是 r^2
 *                    因为 F = G*m1*m2/r^2, 方向沿 dx, 投影到坐标轴需再除以 r
 *                    所以 F_x = F * (dx/r) = G*m1*m2*dx/r^3
 *
 *          优化技巧：
 *          1. 将质量存在 float4.w 中，一次 128-bit 加载获得位置+质量
 *          2. 使用 rsqrtf() 代替 1/sqrtf()，减少一次除法
 *          3. 计算 inv_dist3 = inv_dist^3，避免重复计算
 *          4. 力累加用寄存器变量，避免频繁全局内存访问
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

/**
 * @brief 天体数量（粒子数）
 * @note  4096 个天体 = O(N^2) = 16.7M 对相互作用力计算
 *        每个线程遍历所有其他天体，计算量随 N 平方增长。
 *        对于更大的 N（如 10^6），需要改用 Barnes-Hut 近似算法。
 */
#define NUM_BODIES 4096

/**
 * @brief 每个 Block 的线程数
 * @note 需要保证每个天体被一个线程处理，即 grid_size * BLOCK_SIZE >= NUM_BODIES
 *       4096 / 256 = 16 个 Block，每个 Block 256 线程
 */
#define BLOCK_SIZE 256

/**
 * @brief 软化因子（Softening Parameter）
 * @details 当两个天体距离极近时，r^2 接近 0，导致引力趋于无穷大（奇点）。
 *          SOFTENING 在 r^2 上加一个很小的常数，避免除零：
 *          corrected_dist_sqr = dx^2 + dy^2 + dz^2 + SOFTENING^2
 *          物理意义：模拟天体不是质点，而是有空间分布的"软球体"
 *          当距离远大于 SOFTENING 时，影响可忽略不计。
 * @note   SOFTENING 的值需要根据系统尺度选择：
 *         - 太小：奇点附近数值不稳定
 *         - 太大：近距离的物理精度降低
 *         1e-9 在 [-1, 1] 的三维空间内是合理的。
 */
#define SOFTENING 1e-9f

/**
 * @brief 时间步长（秒）
 * @details 欧拉积分的时间步长，控制模拟精度。
 *          DT 越小，模拟越精确但计算量越大。
 *          0.01 秒在 [-1, 1] 空间尺度下是合理的平衡值。
 *          注意：如果 DT 太大，系统会变得不稳定（能量不守恒）。
 */
#define DT 0.01f

/**
 * @brief 模拟步数
 * @note  10 步用于验证正确性，实际模拟需要数千步
 */
#define NUM_STEPS 10

/**
 * @brief  N-Body 力计算与位置更新 Kernel
 * @param  d_pos      输入：天体位置 (x, y, z) + 质量 (w)，每个天体用 float4 存储
 * @param  d_vel      输入/输出：天体速度 (x, y, z)
 * @param  d_out_pos  输出：更新后的位置 (x, y, z, w)
 * @param  n          天体数量
 *
 * @details 每个线程处理一个天体，计算该天体受到的所有其他天体的引力，
 *          然后更新速度和位置。
 *
 *          算法步骤：
 *          1. 加载本天体的位置和质量（一次 float4 加载）
 *          2. 遍历所有其他天体，累加引力
 *             - 计算距离向量 (dx, dy, dz)
 *             - 计算距离平方 dist_sqr = dx^2 + dy^2 + dz^2 + SOFTENING
 *             - 计算 inv_dist = rsqrtf(dist_sqr)  ← 快速倒数平方根
 *             - 计算 inv_dist3 = inv_dist^3        ← 力的分量需要 1/r^3
 *             - 力分量 = 其他天体质量 * inv_dist3 * 距离分量
 *             - 注意：这里省略了万有引力常数 G（设为 1），因为只做演示
 *          3. 更新速度：v += F * DT
 *          4. 更新位置：pos += v * DT
 *
 *          访存模式分析：
 *          - 所有线程同时读取 d_pos[0..n-1]（全局内存），
 *            这是典型的 broadcast 模式（所有线程读同一片数据）
 *          - 写入 d_out_pos[tid] 和 d_vel[tid] 是合并的
 *          - 性能瓶颈：O(N^2) 的全局内存读取
 *
 *          优化方向：
 *          1. 共享内存 Tiling：将天体分块读入共享内存，减少全局内存访问
 *          2. 循环展开：手动展开内层循环，增加指令级并行
 *          3. Barnes-Hut：将远处的天体用 Quadrupole 近似，降为 O(N log N)
 */
__global__ void nbody_kernel(const float4* d_pos, float3* d_vel, float4* d_out_pos, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 步骤 1：加载本天体的位置和质量
    // float4 是 128-bit 对齐的向量类型，一次加载 4 个 float
    // my_pos.w 存储质量（见初始化时 h_pos[i].w = 1.0f/n）
    float4 my_pos = d_pos[tid];

    // 力累加器（寄存器变量，不需要全局内存访问）
    float3 force = {0.0f, 0.0f, 0.0f};

    // 步骤 2：遍历所有其他天体，累加引力
    // 时间复杂度 O(N)，每个线程执行 N 次迭代
    for (int j = 0; j < n; j++)
    {
        // 加载其他天体的位置和质量
        float4 other = d_pos[j];

        // 计算距离向量
        float dx = other.x - my_pos.x;
        float dy = other.y - my_pos.y;
        float dz = other.z - my_pos.z;

        // 计算距离平方，加上软化因子防止除零
        // 注意：dist_sqr 是标量，用于计算引力大小
        float dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;

        // 使用 rsqrtf() 计算倒数平方根：inv_dist = 1 / sqrt(dist_sqr)
        // rsqrtf() 是硬件加速指令，比 sqrtf() + 除法 快得多
        // 精度大约 2^-23 相对误差，对物理模拟足够
        float inv_dist = rsqrtf(dist_sqr);

        // inv_dist^3 = 1 / (r^3)
        // 为什么需要 r^3？因为力分量 F_x = G * m_i * m_j * dx / r^3
        // 推导：|F| = G*m1*m2/r^2, 方向分量 = dx/r, 所以 F_x = |F| * dx/r = G*m1*m2*dx/r^3
        float inv_dist3 = inv_dist * inv_dist * inv_dist;

        // 力的大小（省略 G）：f = m_other / r^3
        // other.w = 其他天体的质量
        float f = other.w * inv_dist3;

        // 累加力的分量
        force.x += dx * f;
        force.y += dy * f;
        force.z += dz * f;
    }

    // 步骤 3：更新速度
    // v_new = v_old + F * DT （欧拉积分）
    float3 my_vel = d_vel[tid];
    my_vel.x += force.x * DT;
    my_vel.y += force.y * DT;
    my_vel.z += force.z * DT;

    // 步骤 4：更新位置
    // pos_new = pos_old + v_new * DT （欧拉积分）
    my_pos.x += my_vel.x * DT;
    my_pos.y += my_vel.y * DT;
    my_pos.z += my_vel.z * DT;

    // 写回更新后的位置和速度
    d_out_pos[tid] = my_pos;
    d_vel[tid] = my_vel;
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 27: N-Body 物理模拟                                  \n");
    printf("=============================================================\n\n");

    print_device_info();

    int n = NUM_BODIES;
    printf("[主机] 粒子数量: %d\n", n);
    printf("[主机] 计算复杂度: O(N^2) = %.1f 万对相互作用\n", (double)n * n / 1e4);

    // 初始化天体位置和质量
    float4* h_pos = (float4*)malloc(n * sizeof(float4));
    float3* h_vel = (float3*)malloc(n * sizeof(float3));
    srand((unsigned int)time(NULL));
    for (int i = 0; i < n; i++)
    {
        // 位置在 [-1, 1] 范围内随机分布
        h_pos[i].x = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_pos[i].y = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        h_pos[i].z = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        // 质量均匀分布，总质量归一化为 1
        h_pos[i].w = 1.0f / (float)n;
        // 初始速度为零
        h_vel[i].x = h_vel[i].y = h_vel[i].z = 0.0f;
    }

    float4 *d_pos, *d_out_pos;
    float3* d_vel;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pos, n * sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out_pos, n * sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_vel, n * sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemcpy(d_pos, h_pos, n * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_vel, h_vel, n * sizeof(float3), cudaMemcpyHostToDevice));

    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("\n[主机] 模拟 %d 步 ...\n", NUM_STEPS);
    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);

    for (int step = 0; step < NUM_STEPS; step++)
    {
        nbody_kernel<<<grid_size, BLOCK_SIZE>>>(d_pos, d_vel, d_out_pos, n);
        // 交换输入输出缓冲区（双缓冲技术）
        // 避免每次迭代重新分配内存，只需交换指针
        float4* temp = d_pos;
        d_pos = d_out_pos;
        d_out_pos = temp;
    }
    // 注意：Kernel 是异步的，必须同步才能准确计时
    // 如果忘记同步，计时器会记录错误的（过短的）时间
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    gpu_timer_stop(&timer);
    gpu_timer_print(&timer, "N-Body 模拟 (10步)");

    // 预期性能（RTX 5060 Ti）：
    // 4096 个天体，每步 O(N^2) = 16.7M 力计算
    // 10 步约 167M 次力计算，预期约 50-100 ms

    printf("\n=============================================================\n");
    printf("     N-Body 优化方向                                          \n");
    printf("=============================================================\n");
    printf("  1. 共享内存 Tiling: 将天体分块加载到共享内存，减少全局内存访问\n");
    printf("  2. Barnes-Hut: 远处天体用四极矩近似，O(N^2) -> O(N log N)\n");
    printf("  3. 多 GPU: 空间分解到多个 GPU，每个 GPU 计算部分天体\n");
    printf("  4. 循环展开: 手动展开内层循环，增加指令级并行\n");
    printf("  5. 混合精度: 用 fp16 存储位置，fp32 累加力\n");
    printf("=============================================================\n\n");

    cudaFree(d_pos); cudaFree(d_out_pos); cudaFree(d_vel);
    free(h_pos); free(h_vel);
    gpu_timer_destroy(&timer);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}