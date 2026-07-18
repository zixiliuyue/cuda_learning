/******************************************************************************
 * @file    main.cu
 * @brief   Day 34: 自定义 GPU 内存池
 * @details 实现一个简单的 GPU 内存池管理器（bump allocator），
 *          用于减少 cudaMalloc / cudaFree 频繁调用带来的开销。
 *
 *          核心思想：
 *          1. 程序启动时一次性向驱动申请一大块显存（cudaMalloc）；
 *          2. 后续所有的“分配”都只是移动一个偏移指针（bump），几乎零开销；
 *          3. “释放”在 bump allocator 中不逐个归还，而是通过 reset 一次性回收；
 *          4. 程序结束时统一 cudaFree 释放整块显存。
 *
 *          这种策略非常适合“分配模式固定 / 生命周期一致”的场景（如每帧
 *          重新分配大量临时缓冲区），是深度学习框架显存管理的简化教学版。
 *
 * @note    使用 CUDA 13 兼容 API，仅使用 C 风格结构体 + 函数以保持简单。
 * @author  CUDA 30天学习项目
 * @date     2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"
#include "math_utils.h"

/* 每次测试分配的元素个数与迭代次数 */
#define ALLOC_ELEMENTS (256 * 1024)      /* 单次分配 256K 个 float = 1MB */
#define TEST_ITERATIONS 1000             /* 分配/释放循环次数 */
#define TOLERANCE 1e-5f

/* 内存对齐字节数（256 字节对齐，满足绝大多数 GPU 访问要求） */
#define POOL_ALIGNMENT 256

/* ============================================================
 *  内存池数据结构（简化版 bump allocator）
 * ============================================================ */
/**
 * @struct MemPool
 * @brief  简单的 GPU 显存池
 *
 * @field  base_ptr        预分配显存的基地址（设备指针）
 * @field  total_size      池的总大小（字节）
 * @field  offset          当前分配偏移量（bump allocator 的“水位线”）
 * @field  num_allocations 累计分配次数统计（用于观察行为）
 */
typedef struct {
    char*  base_ptr;        /* 预分配显存的基地址 */
    size_t total_size;      /* 总大小（字节） */
    size_t offset;          /* 当前分配偏移（bump allocator） */
    int    num_allocations; /* 分配次数统计 */
} MemPool;

/* ------------------------------------------------------------
 *  工具函数：将 size 向上对齐到 POOL_ALIGNMENT 的整数倍
 * ------------------------------------------------------------ */
static inline size_t align_up(size_t size, size_t alignment)
{
    /* (size + alignment - 1) 向上取整后再屏蔽低位，得到对齐后的大小 */
    return (size + alignment - 1) & ~(alignment - 1);
}

/* ============================================================
 *  内存池 API 实现
 * ============================================================ */

/**
 * @brief  初始化内存池：一次性预分配一大块显存
 * @param  pool        指向 MemPool 结构体的指针
 * @param  total_size  希望预分配的总字节数
 * @return void
 */
static void init_memory_pool(MemPool* pool, size_t total_size)
{
    if (pool == NULL) return;

    /* 关键：整个程序生命周期内，cudaMalloc 只调用这一次 */
    CUDA_SAFE_CALL(cudaMalloc((void**)&pool->base_ptr, total_size));

    pool->total_size      = total_size;
    pool->offset          = 0;
    pool->num_allocations = 0;

    printf("[内存池] 初始化成功，预分配显存: %.2f MB (基地址=%p)\n",
           total_size / (1024.0 * 1024.0), (void*)pool->base_ptr);
}

/**
 * @brief  从内存池中分配一块显存（bump allocator）
 * @param  pool  指向 MemPool 结构体的指针
 * @param  size  需要分配的字节数
 * @return void* 分配到的设备指针；若空间不足返回 NULL
 *
 * @details 分配过程仅仅是：对齐 -> 检查剩余空间 -> 返回当前偏移位置 ->
 *          移动偏移量。整个过程没有任何系统调用，因此速度极快。
 */
static void* pool_allocate(MemPool* pool, size_t size)
{
    if (pool == NULL || pool->base_ptr == NULL) return NULL;

    /* 将请求大小对齐，保证返回的指针满足对齐要求 */
    size_t aligned_size = align_up(size, POOL_ALIGNMENT);

    /* 检查池中剩余空间是否足够 */
    if (pool->offset + aligned_size > pool->total_size)
    {
        fprintf(stderr, "[内存池] 空间不足！请求 %zu 字节，剩余 %zu 字节\n",
                aligned_size, pool->total_size - pool->offset);
        return NULL;
    }

    /* 当前偏移处即为本次分配的起始地址 */
    void* ptr = (void*)(pool->base_ptr + pool->offset);

    /* “bump”：把水位线向前推进 */
    pool->offset += aligned_size;
    pool->num_allocations++;

    return ptr;
}

/**
 * @brief  归还一块显存到内存池
 * @param  pool  指向 MemPool 结构体的指针
 * @param  ptr   之前从池中分配得到的指针
 * @return void
 *
 * @note   bump allocator 无法单独释放某一块内存（因为它不记录每块的边界）。
 *         这里的 pool_free 仅作接口占位，真正的回收依赖 pool_reset。
 *         这正是 bump allocator “分配快、但不支持随机释放”的取舍。
 */
static void pool_free(MemPool* pool, void* ptr)
{
    /* 对 bump allocator 来说，单块释放是空操作（no-op） */
    (void)pool;
    (void)ptr;
    /* 教学说明：如需支持单块释放，应改用 free-list 或分段(slab)分配器 */
}

/**
 * @brief  重置内存池：一次性回收所有已分配的内存
 * @param  pool  指向 MemPool 结构体的指针
 * @return void
 *
 * @details 只需把偏移量清零，之前分配的所有内存瞬间“全部释放”，
 *          O(1) 复杂度，这是 bump allocator 最大的优势。
 */
static void pool_reset(MemPool* pool)
{
    if (pool == NULL) return;
    pool->offset = 0;
    /* num_allocations 累计统计保留，便于观察整体行为 */
}

/**
 * @brief  销毁内存池：释放整块预分配的显存
 * @param  pool  指向 MemPool 结构体的指针
 * @return void
 */
static void destroy_memory_pool(MemPool* pool)
{
    if (pool == NULL || pool->base_ptr == NULL) return;

    /* 整个程序生命周期内，cudaFree 也只调用这一次 */
    CUDA_SAFE_CALL(cudaFree(pool->base_ptr));

    printf("[内存池] 已销毁，累计分配次数: %d\n", pool->num_allocations);

    pool->base_ptr        = NULL;
    pool->total_size      = 0;
    pool->offset          = 0;
    pool->num_allocations = 0;
}

/* ============================================================
 *  验证用 Kernel：向量缩放 c[i] = a[i] * scale
 * ============================================================ */
__global__ void vector_scale(const float* a, float* c, float scale, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        c[tid] = a[tid] * scale;
    }
}

/* ============================================================
 *  测试 1: 直接使用 cudaMalloc / cudaFree 的性能
 * ============================================================ */
static double benchmark_cuda_malloc(size_t bytes, int iterations)
{
    GpuTimer timer;
    gpu_timer_init(&timer);

    gpu_timer_start(&timer);
    for (int i = 0; i < iterations; i++)
    {
        void* d_ptr = NULL;
        CUDA_SAFE_CALL(cudaMalloc(&d_ptr, bytes));  /* 每次都向驱动申请 */
        CUDA_SAFE_CALL(cudaFree(d_ptr));            /* 每次都向驱动归还 */
    }
    gpu_timer_stop(&timer);

    double ms = gpu_timer_get_elapsed_ms(&timer);
    gpu_timer_destroy(&timer);
    return ms;
}

/* ============================================================
 *  测试 2: 使用内存池分配 / 释放的性能
 * ============================================================ */
static double benchmark_memory_pool(MemPool* pool, size_t bytes, int iterations)
{
    GpuTimer timer;
    gpu_timer_init(&timer);

    gpu_timer_start(&timer);
    for (int i = 0; i < iterations; i++)
    {
        void* d_ptr = pool_allocate(pool, bytes);  /* 仅移动偏移，极快 */
        pool_free(pool, d_ptr);                    /* bump allocator 下为 no-op */
        pool_reset(pool);                          /* O(1) 回收，供下轮复用 */
    }
    gpu_timer_stop(&timer);

    double ms = gpu_timer_get_elapsed_ms(&timer);
    gpu_timer_destroy(&timer);
    return ms;
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 34: 自定义 GPU 内存池                               \n");
    printf("=============================================================\n\n");

    /* 打印 GPU 设备信息 */
    print_device_info();

    int    n     = ALLOC_ELEMENTS;
    size_t bytes = (size_t)n * sizeof(float);
    printf("\n[主机] 单次分配大小: %d 个 float (%.2f MB)\n",
           n, bytes / (1024.0 * 1024.0));
    printf("[主机] 测试迭代次数: %d 次\n", TEST_ITERATIONS);

    /* --------------------------------------------------------
     *  第一步：功能验证 —— 用内存池分配的显存执行 Kernel
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     功能验证: 使用内存池分配的显存执行 Kernel               \n");
    printf("=============================================================\n");

    MemPool pool;
    /* 预分配一块足够大的池（这里申请 8MB，足以容纳两个 1MB 缓冲 + 余量） */
    init_memory_pool(&pool, (size_t)8 * 1024 * 1024);

    /* 从池中分配输入和输出两个缓冲区 */
    float* d_a = (float*)pool_allocate(&pool, bytes);
    float* d_c = (float*)pool_allocate(&pool, bytes);

    /* 准备主机数据 */
    float* h_a = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);
    srand((unsigned int)time(NULL));
    init_random_float(h_a, n, -1.0f, 1.0f);

    CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    /* 启动 vector_scale Kernel，验证内存可用 */
    float scale = 2.5f;
    int block = 256;
    int grid  = (n + block - 1) / block;
    vector_scale<<<grid, block>>>(d_a, d_c, scale, n);
    CHECK_KERNEL_ERROR();

    CUDA_SAFE_CALL(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* 校验结果正确性 */
    int correct = 1;
    for (int i = 0; i < n; i++)
    {
        if (fabsf(h_c[i] - h_a[i] * scale) > TOLERANCE)
        {
            correct = 0;
            break;
        }
    }
    printf("  Kernel 计算结果: %s\n", correct ? "正确 (PASS)" : "错误 (FAIL)");
    printf("  当前池水位线   : %.2f MB / %.2f MB\n",
           pool.offset / (1024.0 * 1024.0),
           pool.total_size / (1024.0 * 1024.0));

    /* 用完后重置池，回收所有内存供后续测试复用 */
    pool_reset(&pool);

    /* --------------------------------------------------------
     *  第二步：性能对比 —— cudaMalloc/Free vs 内存池
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("     性能对比: 直接 cudaMalloc/Free  vs  内存池              \n");
    printf("=============================================================\n");

    /* 预热一次，避免首次调用的初始化开销影响测量 */
    benchmark_cuda_malloc(bytes, 1);
    benchmark_memory_pool(&pool, bytes, 1);
    pool_reset(&pool);

    double time_malloc = benchmark_cuda_malloc(bytes, TEST_ITERATIONS);
    double time_pool   = benchmark_memory_pool(&pool, bytes, TEST_ITERATIONS);

    printf("  %-28s | %12s | %14s\n", "方案 ", "总耗时(ms)", "平均每次(us)");
    printf("  ---------------------------- | ------------ | --------------\n");
    printf("  %-28s | %12.3f | %14.3f\n",
           "直接 cudaMalloc/cudaFree", time_malloc,
           time_malloc / TEST_ITERATIONS * 1000.0);
    printf("  %-28s | %12.3f | %14.3f\n",
           "自定义内存池 ", time_pool,
           time_pool / TEST_ITERATIONS * 1000.0);

    /* --------------------------------------------------------
     *  第三步：性能总结
     * -------------------------------------------------------- */
    printf("\n=============================================================\n");
    printf("                        性能总结                             \n");
    printf("=============================================================\n");
    if (time_pool > 0.0)
    {
        printf("  内存池相对加速比 : %.1f x\n", time_malloc / time_pool);
    }
    printf("  结论: 频繁分配/释放场景下，内存池可显著降低分配开销。\n");
    printf("  原理: cudaMalloc/cudaFree 涉及驱动与内核态交互，开销较大；\n");
    printf("        内存池将其摊销为一次性预分配 + O(1) 的偏移移动。\n");
    printf("=============================================================\n");

    /* --------------------------------------------------------
     *  清理资源
     * -------------------------------------------------------- */
    destroy_memory_pool(&pool);
    free(h_a);
    free(h_c);

    printf("\n[主机] 程序正常退出。\n\n");
    return 0;
}
