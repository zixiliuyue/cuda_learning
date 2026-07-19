#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 63: 剪枝 — 结构化 vs 非结构化

模型压缩三板斧（量化 / 剪枝 / 蒸馏）中的「剪枝」：把冗余的权重置零或直接删除，
换取更小的模型体积与更快的推理。本天覆盖两大流派：

  1. 非结构化剪枝（Unstructured / 权重级）：
       按权重绝对值大小把最小的一批权重「置零」，得到稀疏矩阵。
       优点：同样稀疏度下精度损失最小；缺点：稀疏分布不规则，
       普通 GPU/CPU 难以直接加速（需专用稀疏 kernel，如 Ampere 2:4）。

  2. 结构化剪枝（Structured / 通道级）：
       整行/整列/整个通道/整个神经元一起删掉，模型真实「变小」。
       优点：剩下的仍是稠密小矩阵，任何硬件都能直接加速；
       缺点：粒度粗，同样压缩率下精度损失通常更大。

再演示「剪枝 -> 微调恢复精度」这一关键工程环节，并画出
「稀疏度 vs 精度」权衡曲线（文本版）。

运行策略（能不回退就不回退）：
  - 有 PyTorch：无论有无 GPU 都真实训练/剪枝/微调完整跑通
    （device = "cuda" if torch.cuda.is_available() else "cpu"）；
  - 没有 PyTorch：退回 NumPy，演示非结构化剪枝的核心原理（按幅值置零 + 稀疏度统计）。
所有分支都保证 exit code 0。
"""

import sys
import numpy as np

# ------------------------------------------------------------
# 环境探测三布尔量
# ------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

SEP = "=" * 64


# ============================================================
#  合成数据集：两个高斯团的二分类（可分性强，几十步就能收敛）
# ============================================================
def make_dataset(n_per_class=512, dim=32, seed=0):
    """构造两类高斯分布数据，返回 (X, y)。"""
    rng = np.random.default_rng(seed)
    # 两个类中心分开一段距离
    center = np.zeros(dim, dtype=np.float32)
    center[:8] = 0.7
    x0 = rng.normal(-center, 1.0, size=(n_per_class, dim)).astype(np.float32)
    x1 = rng.normal(+center, 1.0, size=(n_per_class, dim)).astype(np.float32)
    X = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int64)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ============================================================
#  PyTorch 部分：真实训练 + 剪枝 + 微调
# ============================================================
if TORCH_AVAILABLE:

    class MLP(nn.Module):
        """一个刻意「偏大」的三层 MLP，方便展示剪枝后仍能保持精度。"""

        def __init__(self, dim=32, hidden=128, n_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, n_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def train_model(model, X, y, epochs=60, lr=1e-2):
        """标准训练循环，返回训练后的模型。"""
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        return model

    @torch.no_grad()
    def evaluate(model, X, y):
        """返回分类准确率（%）。"""
        model.eval()
        pred = model(X).argmax(dim=1)
        return (pred == y).float().mean().item() * 100.0

    @torch.no_grad()
    def global_sparsity(model):
        """统计所有 Linear 权重中 0 元素占比（%）——真实稀疏度。"""
        total, zeros = 0, 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                w = m.weight
                total += w.numel()
                zeros += (w == 0).sum().item()
        return 100.0 * zeros / max(total, 1)

    @torch.no_grad()
    def unstructured_prune(model, amount):
        """非结构化剪枝：对每个 Linear，按 |w| 从小到大置零 amount 比例的权重。

        用「掩码」实现：mask 中被剪掉的位置为 0，保留的为 1。
        我们把 mask 存到 module 上，微调时用它保持这些位置恒为 0。
        """
        for m in model.modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data.abs().flatten()
                k = int(w.numel() * amount)
                if k > 0:
                    # 第 k 小的幅值作为阈值
                    threshold = torch.kthvalue(w, k).values
                    mask = (m.weight.data.abs() > threshold).float()
                else:
                    mask = torch.ones_like(m.weight.data)
                m.weight.data.mul_(mask)
                m.weight_mask = mask  # 记录掩码供微调时复用

    @torch.no_grad()
    def channel_l2_norms(linear):
        """结构化剪枝的重要性度量：每个输出神经元（权重矩阵的一行）的 L2 范数。"""
        return linear.weight.data.norm(p=2, dim=1)

    def structured_prune_fc2(model, keep_ratio):
        """结构化剪枝：把 fc2 的输出神经元按 L2 范数排序，只保留 keep_ratio 比例。

        与非结构化不同——这里真的「删掉」神经元，重建更小的 fc2 与 fc3，
        模型参数量真实下降，任何硬件都能直接受益。
        返回一个新的、更瘦的模型（注意：只有 fc2 输出维度变小，fc1 保持不变）。
        """
        device = model.fc1.weight.device
        hidden = model.fc2.out_features
        keep = max(1, int(hidden * keep_ratio))
        importance = channel_l2_norms(model.fc2)          # 每个输出神经元的重要性
        keep_idx = torch.topk(importance, keep).indices.sort().values

        # 用与 MLP 相同结构手工搭建，但让 fc2 的输出维度缩小为 keep
        new_model = MLP(dim=model.fc1.in_features, hidden=hidden,
                        n_classes=model.fc3.out_features).to(device)
        # fc2：输入维度不变(=hidden)，删掉未保留的输出神经元(行)
        new_model.fc2 = nn.Linear(hidden, keep).to(device)
        # fc3：输入维度 = fc2 的输出 = keep
        new_model.fc3 = nn.Linear(keep, model.fc3.out_features).to(device)

        # fc1 完全拷贝（输出仍是 hidden，喂给瘦身后的 fc2 的输入）
        new_model.fc1.load_state_dict(model.fc1.state_dict())
        # fc2：保留选中的输出神经元(行)
        new_model.fc2.weight.data = model.fc2.weight.data[keep_idx].clone()
        new_model.fc2.bias.data = model.fc2.bias.data[keep_idx].clone()
        # fc3：删掉对应的输入列
        new_model.fc3.weight.data = model.fc3.weight.data[:, keep_idx].clone()
        new_model.fc3.bias.data = model.fc3.bias.data.clone()
        return new_model, keep

    def finetune_with_mask(model, X, y, epochs=30, lr=5e-3):
        """带掩码的微调：每步更新后重新套用掩码，保持被剪权重恒为 0。"""
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for m in model.modules():
                    if isinstance(m, nn.Linear) and hasattr(m, "weight_mask"):
                        m.weight.data.mul_(m.weight_mask)
        return model

    def count_params(model):
        """统计非零参数量（结构化剪枝后即真实参数量）。"""
        return sum(p.numel() for p in model.parameters())


# ============================================================
#  NumPy 部分：无 torch 时演示非结构化剪枝核心原理
# ============================================================
def numpy_unstructured_prune_demo():
    print(SEP)
    print("NumPy 原理演示：非结构化剪枝（按幅值置零）")
    print(SEP)
    rng = np.random.default_rng(0)
    W = rng.normal(size=(64, 128)).astype(np.float32)
    for amount in (0.0, 0.5, 0.8, 0.9):
        k = int(W.size * amount)
        if k > 0:
            threshold = np.partition(np.abs(W).ravel(), k - 1)[k - 1]
            mask = (np.abs(W) > threshold).astype(np.float32)
        else:
            mask = np.ones_like(W)
        Wp = W * mask
        sparsity = 100.0 * (Wp == 0).sum() / Wp.size
        frob = np.linalg.norm(Wp) / np.linalg.norm(W)
        print(f"  目标稀疏度={amount*100:4.0f}% -> 实际={sparsity:5.1f}% | "
              f"保留能量(Frobenius比)={frob:.3f}")
    print("  观察：置零幅值最小的权重时，矩阵整体能量损失很小 —— 这正是")
    print("        非结构化剪枝在高稀疏度下仍能保持精度的直觉来源。")


# ============================================================
#  概念讲解
# ============================================================
def explain_pruning():
    print(SEP)
    print("剪枝概念：结构化 vs 非结构化")
    print(SEP)
    print("""
  维度            非结构化剪枝                结构化剪枝
  ------------------------------------------------------------
  粒度            单个权重                    整通道/整神经元/整行列
  结果            稀疏矩阵(带很多 0)          更小的稠密矩阵
  重要性度量      |w| 幅值                    通道 L2 范数 / BN gamma
  精度损失        同稀疏度下更小              同压缩率下通常更大
  硬件加速        需专用稀疏 kernel(2:4)      任意硬件直接受益
  模型是否变小    否(只是变 0)                是(参数真实减少)

  工程范式：train -> prune -> fine-tune（微调恢复精度）
    直接大比例剪枝会掉点，微调让剩余权重重新适配，找回大部分精度。
    可迭代进行：剪一点 -> 微调 -> 再剪 -> 再微调（iterative pruning）。
""")


# ============================================================
#  主流程
# ============================================================
def run_torch():
    torch.manual_seed(0)
    X_np, y_np = make_dataset()
    n_train = int(len(X_np) * 0.8)
    X = torch.from_numpy(X_np).to(DEVICE)
    y = torch.from_numpy(y_np).to(DEVICE)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    print(SEP)
    print(f"环境：PyTorch 就绪，device = {DEVICE}")
    print(SEP)

    # 1) 训练基线大模型
    base = MLP().to(DEVICE)
    train_model(base, Xtr, ytr)
    base_acc = evaluate(base, Xte, yte)
    base_params = count_params(base)
    print(f"[基线] 稠密模型  参数量={base_params:>6d}  测试精度={base_acc:5.2f}%")

    # 2) 非结构化剪枝：稀疏度-精度权衡曲线
    print()
    print(SEP)
    print("实验 A：非结构化剪枝（权重置零）—— 稀疏度 vs 精度")
    print(SEP)
    print(f"{'目标稀疏度':>10} | {'实际稀疏度':>10} | {'剪枝后':>8} | {'微调后':>8}")
    print("-" * 52)
    for amount in (0.5, 0.7, 0.8, 0.9, 0.95):
        import copy
        m = copy.deepcopy(base)
        unstructured_prune(m, amount)
        acc_pruned = evaluate(m, Xte, yte)
        finetune_with_mask(m, Xtr, ytr)
        acc_ft = evaluate(m, Xte, yte)
        sp = global_sparsity(m)
        print(f"{amount*100:9.0f}% | {sp:9.1f}% | {acc_pruned:7.2f}% | {acc_ft:7.2f}%")
    print("  观察：剪枝直后可能掉点，微调后基本追回基线 —— 微调是剪枝的关键一步。")

    # 3) 结构化剪枝：真实删神经元，模型变小
    print()
    print(SEP)
    print("实验 B：结构化剪枝（删 fc2 神经元）—— 压缩率 vs 精度")
    print(SEP)
    print(f"{'保留比例':>8} | {'隐藏单元':>8} | {'参数量':>8} | {'剪枝后':>8} | {'微调后':>8}")
    print("-" * 56)
    for keep_ratio in (1.0, 0.5, 0.25, 0.125):
        import copy
        small, keep = structured_prune_fc2(copy.deepcopy(base), keep_ratio)
        acc_pruned = evaluate(small, Xte, yte)
        train_model(small, Xtr, ytr, epochs=30, lr=5e-3)  # 微调
        acc_ft = evaluate(small, Xte, yte)
        params = count_params(small)
        print(f"{keep_ratio*100:7.1f}% | {keep:8d} | {params:8d} | "
              f"{acc_pruned:7.2f}% | {acc_ft:7.2f}%")
    print("  观察：结构化剪枝后参数量真实下降（稠密小模型），任何硬件都能直接加速。")

    return 0


def main():
    print()
    print(SEP)
    print("Day 63: 剪枝 — 结构化 vs 非结构化")
    print(SEP)

    explain_pruning()

    if TORCH_AVAILABLE:
        run_torch()
    else:
        print(SEP)
        print("环境：未安装 PyTorch，使用 NumPy 演示非结构化剪枝核心原理。")
        print(SEP)
        numpy_unstructured_prune_demo()

    print()
    print(SEP)
    print("知识点总结")
    print(SEP)
    print("""  1. 非结构化剪枝：按 |w| 幅值置零，稀疏度高、精度损失小，但需专用稀疏 kernel。
  2. 结构化剪枝：删整通道/神经元，模型真实变小，任意硬件受益，粒度粗。
  3. 重要性度量：非结构化用 |w|；结构化常用通道 L2 范数或 BN 的 gamma。
  4. train -> prune -> fine-tune 是恢复精度的标准范式，可迭代进行。
  5. NVIDIA Ampere 的 2:4 结构化稀疏是「半结构化」，兼顾精度与硬件加速。""")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
