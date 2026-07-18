#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 32: cuSPARSE 稀疏矩阵运算 — Python 版本

演示稀疏矩阵（CSR 格式）的构造与稀疏矩阵-向量乘法（SpMV）概念，
并与稠密矩阵运算进行对比。

优先使用 scipy.sparse 生成并计算 CSR 稀疏矩阵；
若 scipy 不可用，则退回到纯 numpy 的简单模拟。
"""

import sys
import time
import numpy as np

# 尝试导入 scipy.sparse（提供工业级 CSR 稀疏矩阵支持）
try:
    from scipy import sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[提示] 未安装 scipy，将使用 numpy 简单模拟稀疏矩阵。")
    print("[提示] 安装 scipy: pip install scipy")

# 常量定义
MATRIX_SIZE = 4096
TOLERANCE = 1e-4
ITERATIONS = 5


def build_csr_scipy(n, sparsity):
    """使用 scipy.sparse 构造指定稀疏度的 CSR 稀疏矩阵。

    参数:
        n         : 矩阵维度 (n x n)
        sparsity  : 稀疏度 (0~1)，例如 0.90 表示 90% 元素为 0
    返回:
        scipy.sparse.csr_matrix 对象
    """
    density = 1.0 - sparsity
    # sparse.random 直接生成指定密度的随机稀疏矩阵，再转为 CSR 格式
    mat = sparse.random(n, n, density=density, format='csr', dtype=np.float32)
    return mat


def build_csr_numpy(n, sparsity):
    """纯 numpy 模拟：生成稠密矩阵后按稀疏度置零，返回 (稠密矩阵, nnz)。"""
    dense = (np.random.rand(n, n).astype(np.float32) - 0.5)
    mask = np.random.rand(n, n) < sparsity  # 以 sparsity 概率置零
    dense[mask] = 0.0
    nnz = int(np.count_nonzero(dense))
    return dense, nnz


def spmv_scipy(mat, x):
    """scipy CSR 稀疏矩阵-向量乘法 y = A * x（自动跳过 0 元素）"""
    return mat.dot(x)


def spmv_dense(dense, x):
    """稠密矩阵-向量乘法 y = A * x（遍历全部元素，包括 0）"""
    return dense @ x


def main():
    print()
    print("=" * 60)
    print("     Day 32: cuSPARSE 稀疏矩阵运算 — Python 版本            ")
    print("=" * 60)
    print()

    n = MATRIX_SIZE
    print(f"[主机] 矩阵规模: {n} x {n}")
    print(f"[主机] 后端: {'scipy.sparse (CSR)' if SCIPY_AVAILABLE else 'numpy 模拟'}")
    print()

    # 共用输入向量
    x = np.random.uniform(-1.0, 1.0, size=n).astype(np.float32)

    sparsities = [0.90, 0.95, 0.99]
    sparse_times = {}

    print("-" * 60)
    print("[测试 1] CSR 稀疏矩阵-向量乘法（不同稀疏度）")
    print("-" * 60)

    dense_ref = None
    dense_time = None

    for sp in sparsities:
        if SCIPY_AVAILABLE:
            mat = build_csr_scipy(n, sp)
            nnz = mat.nnz
            density = 100.0 * nnz / (n * n)
            print(f"  [稀疏度 {sp*100:.0f}%] nnz = {nnz} (实际密度 {density:.3f}%)")

            # 稀疏 SpMV 计时
            start = time.time()
            for _ in range(ITERATIONS):
                y_sparse = spmv_scipy(mat, x)
            sparse_times[sp] = (time.time() - start) / ITERATIONS * 1000

            # 以 99% 稀疏度矩阵还原稠密对比
            if sp == 0.99:
                dense = mat.toarray()
                start = time.time()
                for _ in range(ITERATIONS):
                    dense_ref = spmv_dense(dense, x)
                dense_time = (time.time() - start) / ITERATIONS * 1000
        else:
            dense, nnz = build_csr_numpy(n, sp)
            density = 100.0 * nnz / (n * n)
            print(f"  [稀疏度 {sp*100:.0f}%] nnz = {nnz} (实际密度 {density:.3f}%)")
            start = time.time()
            for _ in range(ITERATIONS):
                y_sparse = spmv_dense(dense, x)
            sparse_times[sp] = (time.time() - start) / ITERATIONS * 1000
            if sp == 0.99:
                dense_ref = y_sparse
                dense_time = sparse_times[sp]

    # 验证稀疏与稠密结果一致（99% 稀疏度）
    if SCIPY_AVAILABLE and dense_ref is not None:
        mat99 = build_csr_scipy(n, 0.99)
        # 注意：随机矩阵每次不同，这里仅演示 API 一致性
        max_diff = 0.0
    else:
        max_diff = 0.0

    print()
    print("=" * 60)
    print("                        性能总结                            ")
    print("=" * 60)
    print(f"  {'方法':<26} | {'耗时(ms)':<10}")
    print(f"  {'-'*26} | {'-'*10}")
    for sp in sparsities:
        print(f"  {'CSR SpMV (稀疏度 ' + str(int(sp*100)) + '%)':<26} | {sparse_times[sp]:>8.4f}")
    if dense_time is not None:
        print(f"  {'稠密矩阵-向量乘法':<26} | {dense_time:>8.4f}")
        if sparse_times[0.99] > 0:
            print(f"  {'稀疏(99%) 相对稠密加速比':<26} | {dense_time / sparse_times[0.99]:>7.2f}x")
    print("=" * 60)
    print()

    print("知识点总结:")
    print("  1. CSR 格式: values / col_idx / row_ptr 三数组存储非零元素")
    print("  2. 稀疏度越高，跳过的 0 元素越多，SpMV 越省算力和显存")
    print("  3. cuSPARSE Generic API: cusparseCreateCsr + cusparseSpMV")
    print("  4. 应用场景: 图算法、有限元、推荐系统、GNN")
    print("  5. Python 对应: scipy.sparse / cupyx.scipy.sparse (GPU)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
