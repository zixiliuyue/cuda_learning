/******************************************************************************
 * @file    main.cu
 * @brief   Day 9: 原子操作与并行归约（Parallel Reduction）
 * @details 本程序实现数组元素的并行求和，对比多种实现方式。
 *
 *          核心学习目标：
 *          1. 原子操作：atomicAdd 的使用与性能代价
 *          2. 并行归约算法：树形归约与共享内存归约
 *          3. Warp Shuffle 指令：__shfl_down_sync
 *          4. 避免线程发散的分支策略
 *          5. 多阶段归约：Block 级别 + Grid 级别
 *
 *          归约算法原理：
 *          将 N 个元素的和转化为两两相加，形成二叉树结构。
 *          串行复杂度 O(N)，并行复杂度 O(log N)。
 *
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

/**
 * @brief 数组元素个数（4M 个 float = 16 MB 数据）
 * @note 必须是 2 的幂次（归约算法要求），且足够大以体现并行优势
 */
#define ARRAY_SIZE (1024 * 1024 * 4)

/**
 * @brief 每个 Block 的线程数
 * @note 必须是 32 的倍数（Warp 大小），且 <= 1024（CUDA 限制）
 *       256 在大多数 GPU 上能获得较好的占用率
 */
#define BLOCK_SIZE 256

/**
 * @brief  原子操作归约（最简单但最慢）
 * @details 每个线程直接 atomicAdd 到全局结果。
 *          问题：所有线程竞争同一个内存地址，严重串行化。
 */
__global__ void reduce_atomic_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        atomicAdd(d_out, d_in[tid]);
    }
}

/**
 * @brief  共享内存归约 Kernel（每个 Block 归约到自己的共享内存）
 * @details 两阶段归约：
 *          1. 每个 Block 在共享内存中完成局部归约
 *          2. 将每个 Block 的部分和写回全局内存
 *          需要第二轮 Kernel 将 Block 结果再归约。
 */
__global__ void reduce_shared_kernel(const float* d_in, float* d_out, int n)
{
    // 共享内存用于存储当前 Block 的数据
    __shared__ float sdata[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // 将全局内存数据加载到共享内存
    if (tid < n)
    {
        sdata[local_tid] = d_in[tid];
    }
    else
    {
        sdata[local_tid] = 0.0f;
    }
    __syncthreads();

    // 树形归约：每一步将距离为 stride 的元素相加
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (local_tid < stride)
        {
            sdata[local_tid] += sdata[local_tid + stride];
        }
        __syncthreads();
    }

    // 线程 0 将 Block 的部分和写回全局内存
    if (local_tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief  Warp Shuffle 归约（更高效）
 * @details 使用 __shfl_down_sync 在 Warp 内进行归约，
 *          无需共享内存同步，效率更高。
 */
__global__ void reduce_warp_shuffle_kernel(const float* d_in, float* d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;  // Warp 内线程编号
    int warp_id = threadIdx.x / 32;

    // 每个线程读取一个元素
    float val = (tid < n) ? d_in[tid] : 0.0f;

    // Warp 内归约使用 shuffle 指令
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Warp 结果写入共享内存
    __shared__ float warp_sums[BLOCK_SIZE / 32];
    if (lane == 0)
    {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // 最后由第一个 Warp 归约所有 Warp 的和
    if (warp_id == 0)
    {
        val = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane == 0)
        {
            d_out[blockIdx.x] = val;
        }
    }
}

float reduce_cpu(const float* h_in, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += h_in[i];
    return sum;
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 9: 原子操作与并行归约                                \n");
    printf("=============================================================\n\n");

    print_device_info();

    size_t bytes = ARRAY_SIZE * sizeof(float);
    printf("[主机] 数组大小: %d 个元素 (%.2f MB)\n", ARRAY_SIZE, bytes / (1024.0 * 1024.0));

    float* h_in = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    init_random_float(h_in, ARRAY_SIZE, 0.0f, 1.0f);

    float* d_in = NULL;
    float* d_out = NULL;
    float* d_block_sums = NULL;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_in, bytes));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_out, sizeof(float)));
    int num_blocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // CPU 参考
    printf("\n[主机] CPU 归约 ...\n");
    GpuTimer timer_cpu;
    gpu_timer_init(&timer_cpu);
    gpu_timer_start(&timer_cpu);
    float cpu_sum = reduce_cpu(h_in, ARRAY_SIZE);
    gpu_timer_stop(&timer_cpu);
    gpu_timer_print(&timer_cpu, "CPU 归约 ");
    printf("  CPU 结果: %.6f\n", cpu_sum);

    // 原子操作版本（慢）
    printf("\n[主机] GPU 原子操作归约 ...\n");
    CUDA_SAFE_CALL(cudaMemset(d_out, 0, sizeof(float)));
    GpuTimer timer_atomic;
    gpu_timer_init(&timer_atomic);
    gpu_timer_start(&timer_atomic);
    reduce_atomic_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, ARRAY_SIZE);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_atomic);
    gpu_timer_print(&timer_atomic, "原子操作归约 ");
    float atomic_sum = 0.0f;
    CUDA_SAFE_CALL(cudaMemcpy(&atomic_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  GPU 结果: %.6f (误差: %.6e)\n", atomic_sum, fabsf(atomic_sum - cpu_sum));

    // 共享内存归约
    printf("\n[主机] GPU 共享内存归约 ...\n");
    GpuTimer timer_shared;
    gpu_timer_init(&timer_shared);
    gpu_timer_start(&timer_shared);
    reduce_shared_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_block_sums, ARRAY_SIZE);
    // 第二轮：归约 Block 结果
    reduce_shared_kernel<<<1, BLOCK_SIZE>>>(d_block_sums, d_out, num_blocks);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_shared);
    gpu_timer_print(&timer_shared, "共享内存归约 ");
    float shared_sum = 0.0f;
    CUDA_SAFE_CALL(cudaMemcpy(&shared_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  GPU 结果: %.6f (误差: %.6e)\n", shared_sum, fabsf(shared_sum - cpu_sum));

    // Warp Shuffle 归约
    printf("\n[主机] GPU Warp Shuffle 归约 ...\n");
    GpuTimer timer_shuffle;
    gpu_timer_init(&timer_shuffle);
    gpu_timer_start(&timer_shuffle);
    reduce_warp_shuffle_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_block_sums, ARRAY_SIZE);
    reduce_warp_shuffle_kernel<<<1, BLOCK_SIZE>>>(d_block_sums, d_out, num_blocks);
    CHECK_KERNEL_ERROR();
    gpu_timer_stop(&timer_shuffle);
    gpu_timer_print(&timer_shuffle, "Warp Shuffle 归约 ");
    float shuffle_sum = 0.0f;
    CUDA_SAFE_CALL(cudaMemcpy(&shuffle_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  GPU 结果: %.6f (误差: %.6e)\n", shuffle_sum, fabsf(shuffle_sum - cpu_sum));

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. atomicAdd: 原子操作，线程安全但性能低\n");
    printf("  2. 共享内存归约: 两阶段归约，减少全局竞争\n");
    printf("  3. __shfl_down_sync: Warp 内高效数据交换\n");
    printf("  4. 树形归约: 复杂度从 O(N) 降到 O(log N)\n");
    printf("=============================================================\n\n");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_block_sums);
    free(h_in);
    return 0;
}
