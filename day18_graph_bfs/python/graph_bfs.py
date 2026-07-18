#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 18: 图 BFS — Python/NumPy 版本

使用邻接表表示法实现广度优先搜索（BFS），
演示 GPU 概念在图算法中的应用。
GPU BFS 是典型的 frontier-based 并行算法。
"""

import sys
import time
import numpy as np
from collections import deque

# 常量定义
NUM_NODES = 1024
MAX_EDGES_PER_NODE = 8
BLOCK_SIZE = 256
TOLERANCE = 1.0  # BFS 结果为整数，验证用


def generate_random_graph(num_nodes, max_edges):
    """
    生成随机无向图（邻接表表示）。
    返回: offsets (CSR 格式的偏移数组), edges (边数组)
    CSR 格式是 GPU 图算法的常用表示法。
    """
    edge_list = []
    for i in range(num_nodes):
        # 每个节点随机连接 1~max_edges 个邻居
        num_edges = np.random.randint(1, max_edges + 1)
        neighbors = np.random.choice(num_nodes, size=num_edges, replace=False)
        for j in neighbors:
            if j != i:  # 避免自环
                edge_list.append((i, j))
                edge_list.append((j, i))  # 无向图

    # 去重
    edge_list = list(set(edge_list))

    # 转为 CSR 格式
    adj = [[] for _ in range(num_nodes)]
    for u, v in edge_list:
        adj[u].append(v)

    # 构建 CSR
    offsets = np.zeros(num_nodes + 1, dtype=np.int32)
    edges = []
    for i in range(num_nodes):
        offsets[i + 1] = offsets[i] + len(adj[i])
        edges.extend(sorted(adj[i]))

    edges = np.array(edges, dtype=np.int32)
    return offsets, edges, adj


def bfs_serial(adj_list, start_node, num_nodes):
    """串行 BFS（CPU 参考）"""
    visited = np.full(num_nodes, -1, dtype=np.int32)
    queue = deque([start_node])
    visited[start_node] = 0

    while queue:
        node = queue.popleft()
        for neighbor in adj_list[node]:
            if visited[neighbor] == -1:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)

    return visited


def bfs_frontier_simulation(offsets, edges, start_node, num_nodes):
    """
    模拟 GPU Frontier-based BFS。
    GPU BFS 的核心思想是: 每层所有 frontier 节点的邻居展开是独立的，可以并行。
    """
    distance = np.full(num_nodes, -1, dtype=np.int32)
    distance[start_node] = 0
    frontier = np.array([start_node], dtype=np.int32)

    iteration = 0
    while len(frontier) > 0:
        # 模拟并行展开: 所有 frontier 节点的邻居同时处理
        next_frontier = []
        for node in frontier:
            start = offsets[node]
            end = offsets[node + 1]
            for edge_idx in range(start, end):
                neighbor = edges[edge_idx]
                if distance[neighbor] == -1:
                    distance[neighbor] = iteration + 1
                    next_frontier.append(neighbor)

        frontier = np.array(next_frontier, dtype=np.int32) if next_frontier else np.array([], dtype=np.int32)
        iteration += 1

    return distance


def main():
    print()
    print("=" * 60)
    print("     Day 18: 图 BFS — Python/NumPy 版本                     ")
    print("=" * 60)
    print()

    # 图生成
    print(f"[主机] 生成随机图: {NUM_NODES} 个节点, 最大度数 {MAX_EDGES_PER_NODE}")
    offsets, edges, adj_list = generate_random_graph(NUM_NODES, MAX_EDGES_PER_NODE)
    total_edges = len(edges) // 2  # 无向图
    print(f"[主机] CSR 格式: offsets[{len(offsets)}], edges[{len(edges)}]")
    print(f"[主机] 总边数: {total_edges}")

    start_node = 0

    # 串行 BFS
    print()
    print("[主机] 运行串行 BFS...")
    start = time.time()
    result_serial = bfs_serial(adj_list, start_node, NUM_NODES)
    serial_time = (time.time() - start) * 1000

    # Frontier-based BFS（模拟 GPU）
    print("[主机] 运行 Frontier-based BFS（GPU 概念模拟）...")
    start = time.time()
    result_frontier = bfs_frontier_simulation(offsets, edges, start_node, NUM_NODES)
    frontier_time = (time.time() - start) * 1000

    # 验证
    diff_count = np.sum(result_serial != result_frontier)
    is_ok = diff_count == 0

    # 统计
    reachable = np.sum(result_serial >= 0)
    max_depth = np.max(result_serial)
    avg_degree = len(edges) / NUM_NODES

    # 性能总结
    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  串行 BFS           : {serial_time:.4f} ms")
    print(f"  Frontier BFS       : {frontier_time:.4f} ms")
    print(f"  结果一致性         : {'通过' if is_ok else f'失败 ({diff_count} 个节点不同)'}")
    print(f"  可达节点           : {reachable}/{NUM_NODES}")
    print(f"  最大深度           : {max_depth}")
    print(f"  平均度数           : {avg_degree:.1f}")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. CSR 格式: offsets[i]..offsets[i+1] 是节点 i 的邻居范围")
    print("  2. Frontier-based BFS: 每层展开所有 frontier 节点")
    print("  3. GPU 并行性: frontier 内各节点展开独立，可并行")
    print("  4. 挑战: 不规则内存访问（邻接表访问不连续）")
    print("  5. 负载均衡: 度数大的节点需要更多线程")
    print("  6. 实际 GPU BFS 库: Gunrock (UC Davis)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
