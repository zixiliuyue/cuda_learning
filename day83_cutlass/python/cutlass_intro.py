#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 83: CUTLASS 入门 — GEMM 模板库的分层 tiling

CUTLASS 是 NVIDIA 开源的 C++ 模板库，用「分层 tiling + 模板组合」把
Day6（Tiling 矩阵乘）、Day13（cuBLAS）背后的高性能 GEMM 拆成可复用组件。
接前 40 天大纲遗留的「用 CUTLASS 写 GEMM」项。

CUTLASS 本体是 C++ 模板（GPU 上编译运行），本天以**原理为主**：
  1. 讲清 CUTLASS 的三层 tile：ThreadBlock → Warp → Thread（+ 主循环软件流水）；
  2. 用纯 Python/NumPy **复现分层分块 GEMM 的概念**，验证与朴素/naive GEMM 一致；
  3. 打印各层 tile 的形状与数据搬运层级（GMEM→SMEM→REG）；
  4. 给出 CUTLASS C++ 使用骨架代码（在 README / 本文件字符串中）。

无需 GPU；有 torch 时用 torch 做参考对照，否则用 numpy。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TORCH_CUDA = TORCH_AVAILABLE and torch.cuda.is_available()
MULTI_GPU = TORCH_CUDA and torch.cuda.device_count() > 1

SEP = "=" * 64


# ============================================================
#  CUTLASS 分层结构讲解
# ============================================================
def explain_cutlass_hierarchy():
    print(SEP)
    print("CUTLASS 的分层 GEMM：把大矩阵乘拆成三层 tile")
    print(SEP)
    print(r"""
  C[M,N] = A[M,K] × B[K,N]

  第 1 层 ThreadBlock Tile（一个 block 负责 C 的一块）
    ┌───────────── C ─────────────┐
    │  Mtile×Ntile 的输出子块      │  典型 128×128
    │  沿 K 维分 Ktile（如 32）循环 │
    └──────────────────────────────┘
    从 GMEM 把 A 的 [Mtile,Ktile]、B 的 [Ktile,Ntile] 搬进 SMEM（共享内存）。

  第 2 层 Warp Tile（block 内每个 warp 负责一小块）
    把 128×128 再切成若干 64×64 交给各 warp，数据从 SMEM 读到寄存器。

  第 3 层 Thread / Instruction Tile（每线程/每条 MMA 指令）
    每个线程算 8×8 之类的微块；Tensor Core 上用 mma.sync 指令，
    如 16×8×16 的 warp 级矩阵乘累加（WMMA/MMA）。

  主循环软件流水（Software Pipelining）
    双缓冲：算当前 Ktile 的同时，异步预取下一个 Ktile（cp.async），
    把访存延迟藏在计算后面——这是 CUTLASS 高性能的关键之一。

  数据搬运层级：GMEM → SMEM → Register → Tensor Core
                （慢/大）        （快/小）    （最快）
""")


def explain_template_composition():
    print(SEP)
    print("模板组合：CUTLASS 用 C++ 模板参数「拼」出一个 GEMM kernel")
    print(SEP)
    print(r"""
  你只需指定这些「积木」，编译器实例化出专用高性能 kernel：
    - 元素类型      ElementA/B/C（half, tf32, int8, float ...）
    - 布局          LayoutA/B/C（RowMajor / ColumnMajor）
    - ThreadBlock   Shape<128,128,32>
    - Warp          Shape<64,64,32>
    - Instruction   Shape<16,8,16>（对应 Tensor Core MMA）
    - 计算架构      SM70/SM80/SM90（Volta/Ampere/Hopper）
    - 流水级数      Stages = 2/3/4（软件流水深度）
    - Epilogue      如 LinearCombination（做 alpha*AB + beta*C、加偏置、激活）

  换句话说：CUTLASS 把「怎样高效做 GEMM」参数化，
            换精度/换架构/换融合尾操作，只改模板参数即可。
""")


# ============================================================
#  用 NumPy 复现「分层分块 GEMM」概念
# ============================================================
def naive_gemm(a, b):
    """朴素三重循环 GEMM（作为正确性基准，规模小才用）。"""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    c = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i, p] * b[p, j]
            c[i, j] = acc
    return c


def tiled_gemm(a, b, block_tile, warp_tile, verbose=True):
    """模拟 CUTLASS 分层 tiling 的分块 GEMM（数值上等价于普通矩阵乘）。

    - block_tile=(BM,BN,BK)：ThreadBlock 层，一次搬 A[BM,BK]、B[BK,BN] 到「SMEM」
    - warp_tile=(WM,WN)    ：Warp 层，把 block tile 再切给各 warp（这里只演示切分与累加）

    我们不追求性能，只演示「三层循环嵌套 + 沿 K 分块累加」的层级结构，
    并统计各层 tile 数，验证结果与 naive_gemm 完全一致。
    """
    m, k = a.shape
    _, n = b.shape
    bm, bn, bk = block_tile
    wm, wn = warp_tile
    c = np.zeros((m, n), dtype=np.float64)

    n_block_tiles = 0
    n_warp_tiles = 0
    n_ktiles = 0

    # ---- ThreadBlock 层：遍历 C 的 (BM×BN) 输出块 ----
    for i0 in range(0, m, bm):
        for j0 in range(0, n, bn):
            n_block_tiles += 1
            i1 = min(i0 + bm, m)
            j1 = min(j0 + bn, n)
            # 该 block 的累加器（相当于寄存器里的 C 子块）
            acc = np.zeros((i1 - i0, j1 - j0), dtype=np.float64)
            # ---- 沿 K 维分 Ktile 循环（主循环，对应软件流水的每一拍）----
            for p0 in range(0, k, bk):
                n_ktiles += 1
                p1 = min(p0 + bk, k)
                a_smem = a[i0:i1, p0:p1]   # 搬到「SMEM」的 A 分块
                b_smem = b[p0:p1, j0:j1]   # 搬到「SMEM」的 B 分块
                # ---- Warp 层：把 block tile 再切给各 warp 累加 ----
                for wi in range(0, i1 - i0, wm):
                    for wj in range(0, j1 - j0, wn):
                        n_warp_tiles += 1
                        wi1 = min(wi + wm, i1 - i0)
                        wj1 = min(wj + wn, j1 - j0)
                        # 每个 warp 块做一次小矩阵乘累加（模拟 Tensor Core MMA）
                        acc[wi:wi1, wj:wj1] += (
                            a_smem[wi:wi1, :] @ b_smem[:, wj:wj1]
                        )
            c[i0:i1, j0:j1] = acc

    if verbose:
        print(f"  ThreadBlock tiles = {n_block_tiles}"
              f"（每块 {bm}×{bn}，沿 K 分 {bk}）")
        print(f"  K-loop 迭代总数   = {n_ktiles}"
              f"（主循环拍数，对应软件流水/双缓冲）")
        print(f"  Warp tiles 总数   = {n_warp_tiles}（每块 {wm}×{wn}）")
    return c


def cutlass_cpp_skeleton():
    print(SEP)
    print("CUTLASS C++ 使用骨架（在 Linux + NVIDIA GPU 上编译运行）")
    print(SEP)
    print(r"""
  #include "cutlass/gemm/device/gemm.h"

  // 用模板参数「拼」出一个 fp16 输入 / fp32 累加的 Ampere GEMM
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, cutlass::layout::RowMajor,      // A: half, 行主序
      cutlass::half_t, cutlass::layout::ColumnMajor,   // B: half, 列主序
      float,          cutlass::layout::RowMajor,       // C: float
      float,                                            // 累加类型
      cutlass::arch::OpClassTensorOp,                   // 用 Tensor Core
      cutlass::arch::Sm80,                              // Ampere (SM80)
      cutlass::gemm::GemmShape<128, 128, 32>,           // ThreadBlock Tile
      cutlass::gemm::GemmShape< 64,  64, 32>,           // Warp Tile
      cutlass::gemm::GemmShape< 16,   8, 16>            // Instruction (MMA) Tile
  >;

  Gemm gemm_op;
  Gemm::Arguments args({M, N, K},
                       {dA, lda}, {dB, ldb}, {dC, ldc}, {dC, ldc},
                       {alpha, beta});   // Epilogue: alpha*AB + beta*C
  cutlass::Status st = gemm_op(args);    // 启动 kernel

  编译（示意）：
    nvcc -I<cutlass>/include -arch=sm_80 gemm.cu -o gemm

  也可用 Python 封装 nvidia-cutlass（pip install nvidia-cutlass）在 GPU 上跑。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_cutlass_hierarchy()
    print()
    explain_template_composition()

    print()
    print(SEP)
    print("演示：用 NumPy 复现 CUTLASS 式分层分块 GEMM 并验证正确性")
    print(SEP)
    m, k, n = 96, 80, 64
    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))

    print(f"  矩阵：A[{m},{k}] × B[{k},{n}] = C[{m},{n}]")
    print(f"  Tile 配置：ThreadBlock=(32,32,16), Warp=(16,16)")
    print()
    c_tiled = tiled_gemm(a, b, block_tile=(32, 32, 16), warp_tile=(16, 16))

    # 正确性基准：优先 torch，其次 numpy matmul
    if TORCH_AVAILABLE:
        ref = (torch.tensor(a) @ torch.tensor(b)).numpy()
        ref_name = "torch.matmul"
    else:
        ref = a @ b
        ref_name = "numpy.matmul"
    max_err = float(np.max(np.abs(c_tiled - ref)))
    print()
    print(f"  与 {ref_name} 最大误差 = {max_err:.2e}  "
          + ("✓ 一致" if max_err < 1e-9 else "✗ 不一致"))

    # 再与朴素三重循环对照（小规模验证 tiling 不改变数学结果）
    ms, ks, ns = 8, 6, 5
    sa = rng.standard_normal((ms, ks))
    sb = rng.standard_normal((ks, ns))
    c_naive = naive_gemm(sa, sb)
    c_small = tiled_gemm(sa, sb, block_tile=(4, 4, 3), warp_tile=(2, 2), verbose=False)
    err2 = float(np.max(np.abs(c_naive - c_small)))
    print(f"  小规模 tiled vs 朴素三重循环 最大误差 = {err2:.2e}  "
          + ("✓ 一致" if err2 < 1e-9 else "✗ 不一致"))

    print()
    cutlass_cpp_skeleton()

    print()
    print(SEP)
    if TORCH_CUDA:
        print("环境：有 GPU。真实 CUTLASS 需 C++ 编译或 `pip install nvidia-cutlass`；")
        print("      本天用 NumPy 复现分层 tiling 概念，理解结构后再上手 C++ 模板。")
    else:
        print("环境：无 GPU。CUTLASS 本体是 C++ 模板；已用 NumPy 复现分层 tiling 概念。")
    print(SEP)


if __name__ == "__main__":
    main()
