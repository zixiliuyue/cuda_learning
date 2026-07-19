#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 76: 认识 Megatron-LM 架构 — 张量并行（Tensor Parallel）如何切 Transformer

Megatron-LM 是 NVIDIA 提出的大模型训练框架，其最核心的思想是
**张量并行（TP）**：把一个 Transformer 层里巨大的权重矩阵按行/列切到
多张 GPU 上，让每张卡只存一部分参数、只算一部分计算，再用极少量的
集合通信（all-reduce）把结果拼回来。

本天以「原理为主 + 单机可跑的数值验证」呈现：
  - 用【单进程模拟 world_size 个 rank】的方式，把权重真的切开，
    分别计算，再手动模拟 all-reduce / all-gather，
    最后验证「切分并行的结果」与「不切分的完整计算」在数值上完全一致；
  - 讲清 MLP（列并行→行并行）与 MHA（QKV 列并行、输出投影行并行）的切法；
  - 讲清前向/反向的两个通信算子 f 与 g（Megatron 论文里的核心记号）。

为什么单进程模拟：单机单卡上起 torch.distributed + mp.spawn 容易卡死，
而张量并行的「数学等价性」用 numpy 单进程即可完整复现，学习效果一致。
"""

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量：torch / CUDA / 分布式）——仅用于打印环境说明，
# 本天数值演示统一用 numpy 单进程模拟，保证任何环境稳定退出。
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CUDA_AVAILABLE = bool(TORCH_AVAILABLE and torch.cuda.is_available())

DIST_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import torch.distributed as dist  # noqa: F401
        DIST_AVAILABLE = True
    except ImportError:
        DIST_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  0. 基础工具：softmax / gelu（numpy 实现，数值稳定）
# ============================================================
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    # tanh 近似的 GELU，与主流实现一致
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


# ============================================================
#  1. 架构要点讲解
# ============================================================
def explain_megatron_arch():
    print(SEP)
    print("Megatron-LM 张量并行：一个 Transformer 层怎么切？")
    print(SEP)
    print("""
  一个标准 Transformer 层 = 多头注意力(MHA) + 前馈网络(MLP)，
  Megatron 的做法是把两块里的大 Linear 沿不同维度切开：

  ┌───────────────── MLP（两层 Linear）─────────────────┐
   Y = GELU(X · A) · B
     A: [h, 4h]  —— 【列并行】沿输出维(4h)切成 world 份
                    每个 rank 只算 4h/world 列，天然按列独立，无需通信
     GELU 是逐元素运算，切开后各 rank 各算各的，仍然正确
     B: [4h, h]  —— 【行并行】沿输入维(4h)切成 world 份
                    每个 rank 得到部分和，最后 all-reduce 求和拼回
  └──────────────────────────────────────────────────────┘

  ┌───────────────── MHA（多头注意力）─────────────────┐
   Q=X·Wq, K=X·Wk, V=X·Wv ;  O = Attention(Q,K,V) · Wo
     Wq/Wk/Wv: 【列并行】——按 head 切！每个 rank 拿 nheads/world 个头，
                注意力天然按头独立，每个 rank 完整算自己那些头的注意力
     Wo:        【行并行】——按输入维切，各 rank 出部分和，all-reduce 求和
  └──────────────────────────────────────────────────────┘

  关键洞察：
   - 列并行的输出可以直接喂给行并行的输入，中间【不需要通信】，
     一整个 MLP / MHA 块只在【最后】做一次 all-reduce —— 通信极省。
   - 切 head 而不是切 hidden 内部，保证注意力的数学结构不被破坏。
""")


def explain_fg_operators():
    print(SEP)
    print("通信算子 f 与 g（Megatron 论文核心记号）")
    print(SEP)
    print("""
  一个「列并行 Linear → 行并行 Linear」的组合，只有两个通信点：

        输入 X                              输出 Y
          │                                   ▲
        [ f ]  前向: 广播/恒等                 │
          │    反向: all-reduce 梯度         [ g ] 前向: all-reduce
          ▼                                   │    反向: 恒等
     列并行 Linear ──(无通信)──► 行并行 Linear ┘

   f 算子：前向把输入送进各 rank（恒等/复制），
           反向时对各 rank 传回的输入梯度做 all-reduce。
   g 算子：前向对各 rank 的部分输出做 all-reduce 求和，
           反向时是恒等（各 rank 直接拿到相同的输出梯度）。

   记忆口诀：一层 TP 块 = 前向 1 次 all-reduce(g) + 反向 1 次 all-reduce(f)。
""")


# ============================================================
#  2. MLP 的张量并行：数值等价验证
#     完整:  Y = GELU(X·A)·B
#     并行:  A 列并行, B 行并行, 最后 all-reduce
# ============================================================
def demo_mlp_tp(world_size=4, seq=6, h=8):
    print(SEP)
    print(f"演示 1：MLP 张量并行（列并行 + 行并行），world_size={world_size}")
    print(SEP)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((seq, h)).astype(np.float64)
    A = rng.standard_normal((h, 4 * h)).astype(np.float64)   # 第一层：h -> 4h
    B = rng.standard_normal((4 * h, h)).astype(np.float64)   # 第二层：4h -> h

    # --- 参考：不切分的完整计算 ---
    Y_full = gelu(X @ A) @ B

    # --- 张量并行：单进程模拟 world_size 个 rank ---
    assert (4 * h) % world_size == 0, "隐藏维需能被 world_size 整除"
    col = (4 * h) // world_size  # 每个 rank 负责的列/行数

    partials = []
    for r in range(world_size):
        # 列并行：rank r 只持有 A 的第 r 段列 -> 得到 4h/world 列的中间结果
        A_r = A[:, r * col:(r + 1) * col]        # [h, col]
        Z_r = gelu(X @ A_r)                       # [seq, col]  GELU 逐元素，切开也对
        # 行并行：rank r 只持有 B 的对应行段 -> 得到一个「部分和」
        B_r = B[r * col:(r + 1) * col, :]        # [col, h]
        Y_r = Z_r @ B_r                           # [seq, h] 部分和
        partials.append(Y_r)

    # g 算子：all-reduce（对各 rank 的部分和求和），这里用手动求和模拟
    Y_tp = np.sum(partials, axis=0)

    max_err = np.abs(Y_tp - Y_full).max()
    print(f"  每个 rank 持有 A 的 {col}/{4*h} 列、B 的 {col}/{4*h} 行")
    print(f"  中间结果不通信，块末 all-reduce 一次（模拟 {world_size} 份部分和求和）")
    print(f"  与完整计算的最大误差 = {max_err:.2e}  ->  "
          + ("等价 ✓" if max_err < 1e-9 else "不等价 ✗"))


# ============================================================
#  3. MHA 的张量并行：数值等价验证
#     QKV 按 head 列并行，各 rank 独立算注意力，Wo 行并行 all-reduce
# ============================================================
def demo_mha_tp(world_size=2, seq=5, h=8, nheads=4):
    print()
    print(SEP)
    print(f"演示 2：MHA 张量并行（QKV 按 head 列并行 + Wo 行并行），world_size={world_size}")
    print(SEP)

    assert nheads % world_size == 0, "头数需能被 world_size 整除"
    assert h % nheads == 0, "hidden 需能被头数整除"
    dh = h // nheads                      # 每个头的维度
    heads_per_rank = nheads // world_size

    rng = np.random.default_rng(1)
    X = rng.standard_normal((seq, h)).astype(np.float64)
    Wq = rng.standard_normal((h, h)).astype(np.float64)
    Wk = rng.standard_normal((h, h)).astype(np.float64)
    Wv = rng.standard_normal((h, h)).astype(np.float64)
    Wo = rng.standard_normal((h, h)).astype(np.float64)

    def attention_all_heads(X, Wq, Wk, Wv, head_ids):
        """只计算 head_ids 指定的那些头，输出对应的拼接结果 [seq, len(head_ids)*dh]。"""
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        outs = []
        for hid in head_ids:
            sl = slice(hid * dh, (hid + 1) * dh)
            q, k, v = Q[:, sl], K[:, sl], V[:, sl]
            scores = (q @ k.T) / np.sqrt(dh)
            ctx = softmax(scores, axis=-1) @ v       # [seq, dh]
            outs.append(ctx)
        return np.concatenate(outs, axis=1)          # [seq, len*dh]

    # --- 参考：完整 MHA ---
    O_full_ctx = attention_all_heads(X, Wq, Wk, Wv, list(range(nheads)))  # [seq, h]
    Y_full = O_full_ctx @ Wo

    # --- 张量并行 ---
    partials = []
    for r in range(world_size):
        head_ids = list(range(r * heads_per_rank, (r + 1) * heads_per_rank))
        # 列并行：QKV 只切出该 rank 负责的头对应的列（等价于切 Wq/Wk/Wv 的列）
        ctx_r = attention_all_heads(X, Wq, Wk, Wv, head_ids)  # [seq, heads_per_rank*dh]
        # 行并行 Wo：取 Wo 中对应这些头的行段
        rows = slice(r * heads_per_rank * dh, (r + 1) * heads_per_rank * dh)
        Wo_r = Wo[rows, :]                                    # [heads_per_rank*dh, h]
        partials.append(ctx_r @ Wo_r)                         # 部分和 [seq, h]

    Y_tp = np.sum(partials, axis=0)                            # g 算子 all-reduce
    max_err = np.abs(Y_tp - Y_full).max()
    print(f"  头数={nheads}, 每 rank 负责 {heads_per_rank} 个头（各头维度 dh={dh}）")
    print(f"  每 rank 完整计算自己的头的注意力，Wo 行并行后 all-reduce 求和")
    print(f"  与完整 MHA 的最大误差 = {max_err:.2e}  ->  "
          + ("等价 ✓" if max_err < 1e-9 else "不等价 ✗"))


# ============================================================
#  主流程
# ============================================================
def main():
    explain_megatron_arch()
    explain_fg_operators()
    demo_mlp_tp(world_size=4)
    demo_mha_tp(world_size=2)

    print()
    print(SEP)
    if CUDA_AVAILABLE and DIST_AVAILABLE:
        print("环境：torch + CUDA + 分布式就绪。")
        print("      本天为原理与数值等价演示，故仍用单进程模拟切分（避免 mp.spawn 卡死）；")
        print("      真正多卡运行请参考 Megatron-LM 官方仓库的 tensor_parallel 实现。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch，但无 CUDA GPU / 未用分布式。已用 numpy 单进程模拟张量并行。")
    else:
        print("环境：未安装 PyTorch。已用 numpy 单进程模拟张量并行原理。")
    print("要点：TP 把大矩阵按列/行切到多卡，一整块只需 1 次 all-reduce，通信极省。")
    print(SEP)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
