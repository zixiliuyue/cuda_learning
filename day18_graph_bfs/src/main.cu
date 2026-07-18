/******************************************************************************
 * @file    main.cu
 * @brief   Day 18: GPU 并行 BFS 图遍历
 * @details 使用 CSR 格式表示图，实现前沿扩展（Frontier Expansion）策略。
 * @author  CUDA 30天学习项目
 * @date    2026-07-18
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "timer.h"

#define NUM_NODES 10000
#define NUM_EDGES 50000

/**
 * @brief  BFS Kernel - 前沿扩展
 * @param  csr_offsets  CSR 行偏移数组
 * @param  csr_edges    CSR 列索引数组（邻居节点）
 * @param  distances    距离数组
 * @param  visited      访问标记数组
 * @param  frontier     当前前沿节点数组
 * @param  frontier_size 当前前沿大小
 * @param  next_frontier 下一前沿节点数组
 * @param  next_size     下一前沿大小（原子累加）
 * @param  current_level 当前 BFS 层级
 */
__global__ void bfs_kernel(const int* csr_offsets, const int* csr_edges,
                            int* distances, int* visited,
                            const int* frontier, int frontier_size,
                            int* next_frontier, int* next_size,
                            int current_level)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size)
    {
        int node = frontier[tid];
        int start = csr_offsets[node];
        int end = csr_offsets[node + 1];

        for (int e = start; e < end; e++)
        {
            int neighbor = csr_edges[e];
            // 原子 CAS 尝试标记未访问节点
            if (atomicCAS(&visited[neighbor], 0, 1) == 0)
            {
                distances[neighbor] = current_level + 1;
                int pos = atomicAdd(next_size, 1);
                next_frontier[pos] = neighbor;
            }
        }
    }
}

int main(void)
{
    printf("\n=============================================================\n");
    printf("     Day 18: GPU 并行 BFS 图遍历                              \n");
    printf("=============================================================\n\n");

    print_device_info();

    int num_nodes = NUM_NODES;
    int num_edges = NUM_EDGES;
    printf("[主机] 图规模: %d 节点, %d 边\n", num_nodes, num_edges);

    // 生成随机图（CSR 格式）
    int* h_csr_offsets = (int*)malloc((num_nodes + 1) * sizeof(int));
    int* h_csr_edges = (int*)malloc(num_edges * sizeof(int));
    int* h_distances = (int*)malloc(num_nodes * sizeof(int));
    int* h_visited = (int*)malloc(num_nodes * sizeof(int));

    h_csr_offsets[0] = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        int edges_per_node = num_edges / num_nodes;
        h_csr_offsets[i + 1] = h_csr_offsets[i] + edges_per_node;
    }
    for (int i = 0; i < num_edges; i++)
    {
        h_csr_edges[i] = rand() % num_nodes;
    }
    for (int i = 0; i < num_nodes; i++)
    {
        h_distances[i] = -1;
        h_visited[i] = 0;
    }
    h_distances[0] = 0;
    h_visited[0] = 1;

    // 分配设备内存
    int *d_csr_offsets, *d_csr_edges, *d_distances, *d_visited;
    int *d_frontier, *d_next_frontier, *d_next_size;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_csr_offsets, (num_nodes + 1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_csr_edges, num_edges * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_distances, num_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_visited, num_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_frontier, num_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_next_frontier, num_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_next_size, sizeof(int)));

    CUDA_SAFE_CALL(cudaMemcpy(d_csr_offsets, h_csr_offsets, (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_csr_edges, h_csr_edges, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_distances, h_distances, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_visited, h_visited, num_nodes * sizeof(int), cudaMemcpyHostToDevice));

    // 初始化前沿
    int h_frontier[1] = {0};
    int frontier_size = 1;
    CUDA_SAFE_CALL(cudaMemcpy(d_frontier, h_frontier, sizeof(int), cudaMemcpyHostToDevice));

    printf("\n[主机] GPU BFS 遍历 ...\n");
    GpuTimer timer;
    gpu_timer_init(&timer);
    gpu_timer_start(&timer);

    int level = 0;
    while (frontier_size > 0)
    {
        int zero = 0;
        CUDA_SAFE_CALL(cudaMemcpy(d_next_size, &zero, sizeof(int), cudaMemcpyHostToDevice));

        int blocks = (frontier_size + 255) / 256;
        bfs_kernel<<<blocks, 256>>>(d_csr_offsets, d_csr_edges,
                                     d_distances, d_visited,
                                     d_frontier, frontier_size,
                                     d_next_frontier, d_next_size,
                                     level);
        cudaDeviceSynchronize();

        // 交换前沿
        int* temp = d_frontier;
        d_frontier = d_next_frontier;
        d_next_frontier = temp;
        CUDA_SAFE_CALL(cudaMemcpy(&frontier_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
    }

    gpu_timer_stop(&timer);
    gpu_timer_print(&timer, "GPU BFS");
    printf("  BFS 层数: %d\n", level);

    printf("\n=============================================================\n");
    printf("                        知识点总结                            \n");
    printf("=============================================================\n");
    printf("  1. CSR: 压缩稀疏行格式存储图\n");
    printf("  2. 前沿扩展: 每层并行处理前沿节点\n");
    printf("  3. atomicCAS: 确保节点只被访问一次\n");
    printf("  4. 工作负载不均衡: 图算法常见问题\n");
    printf("=============================================================\n\n");

    cudaFree(d_csr_offsets); cudaFree(d_csr_edges);
    cudaFree(d_distances); cudaFree(d_visited);
    cudaFree(d_frontier); cudaFree(d_next_frontier); cudaFree(d_next_size);
    free(h_csr_offsets); free(h_csr_edges);
    free(h_distances); free(h_visited);
    gpu_timer_destroy(&timer);

    printf("[主机] 程序正常退出。\n\n");
    return 0;
}
