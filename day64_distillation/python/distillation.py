#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 64: 知识蒸馏（Knowledge Distillation）

模型压缩三板斧的最后一块：不改数值精度（量化）、不删参数（剪枝），
而是让一个**小模型（student）**去模仿一个**大模型（teacher）**的行为，
从而用更少的参数逼近大模型的精度。

核心思想（Hinton et al., 2015）：
  teacher 输出的 softmax 概率里藏着「暗知识（dark knowledge）」——
  比如「这张 3 有点像 8，但完全不像 1」这类类间相似性信息。
  硬标签（one-hot）丢掉了这些信息，而 teacher 的 soft label 保留了它们。

温度 T 的作用：
  soft(z) = softmax(z / T)
  T 越大，概率分布越「软」（更平缓），类间的相对关系被放大，
  student 能学到更丰富的暗知识。蒸馏时 teacher/student 都用同一个 T 软化。

蒸馏损失（加权）：
  L = alpha * T^2 * KL(student_soft || teacher_soft)   # 蒸馏项（软标签）
    + (1 - alpha) * CE(student_logits, hard_label)      # 标准项（硬标签）
  其中 T^2 用来补偿软化后梯度幅值的缩放。

对比实验：
  1) 直接训练 student（只用硬标签）
  2) 蒸馏训练 student（软标签 + 硬标签）
  看蒸馏能否让同样大小的 student 拿到更高精度。

运行策略：有 torch 就真实训练（有 CUDA 用 GPU，否则用 CPU）；
无 torch 用 NumPy 演示「温度如何软化概率分布」这一核心机制。所有分支 exit code 0。
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
#  合成数据集：多类可分但有重叠（让蒸馏的「类间相似性」有意义）
# ============================================================
def make_dataset(n_per_class=400, dim=20, n_classes=4, seed=0):
    """构造 n_classes 个高斯团，中心排成一圈，相邻类有一定重叠。"""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in range(n_classes):
        center = np.zeros(dim, dtype=np.float32)
        angle = 2 * np.pi * c / n_classes
        center[0] = 2.8 * np.cos(angle)
        center[1] = 2.8 * np.sin(angle)
        Xs.append(rng.normal(center, 1.1, size=(n_per_class, dim)).astype(np.float32))
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ============================================================
#  PyTorch 部分
# ============================================================
if TORCH_AVAILABLE:

    class TeacherNet(nn.Module):
        """大模型：宽而深，容量充足。"""

        def __init__(self, dim=20, n_classes=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    class StudentNet(nn.Module):
        """小模型：窄而浅，参数量远小于 teacher。"""

        def __init__(self, dim=20, n_classes=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 32), nn.ReLU(),
                nn.Linear(32, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def train_plain(model, X, y, epochs=120, lr=1e-2):
        """标准训练：仅硬标签交叉熵。"""
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
        return model

    def distill_loss(student_logits, teacher_logits, hard_label, T, alpha):
        """KD 损失：软标签 KL 散度 + 硬标签 CE，按 alpha 加权。

        - 软标签项：teacher/student 都用温度 T 软化，KL(student || teacher)。
          乘以 T^2 补偿软化带来的梯度缩放。
        - 硬标签项：student 原始 logits 与真值标签的标准交叉熵。
        """
        soft_teacher = F.softmax(teacher_logits / T, dim=1)
        log_soft_student = F.log_softmax(student_logits / T, dim=1)
        kd = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (T * T)
        ce = F.cross_entropy(student_logits, hard_label)
        return alpha * kd + (1.0 - alpha) * ce, kd.item(), ce.item()

    def train_distill(student, teacher, X, y, T=4.0, alpha=0.7,
                      epochs=120, lr=1e-2):
        """蒸馏训练：软标签 + 硬标签联合优化。"""
        teacher.eval()
        with torch.no_grad():
            teacher_logits_all = teacher(X)  # teacher 输出可提前算好（固定）
        student.train()
        opt = torch.optim.Adam(student.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            s_logits = student(X)
            loss, _, _ = distill_loss(s_logits, teacher_logits_all, y, T, alpha)
            loss.backward()
            opt.step()
        return student

    @torch.no_grad()
    def evaluate(model, X, y):
        model.eval()
        return (model(X).argmax(dim=1) == y).float().mean().item() * 100.0

    @torch.no_grad()
    def show_dark_knowledge(teacher, X, y, T):
        """打印一个样本在 T=1 与 T>1 下的 teacher 概率分布，直观看「暗知识」。"""
        teacher.eval()
        idx = 0
        logits = teacher(X[idx:idx + 1])
        p_hard = F.softmax(logits, dim=1).cpu().numpy().ravel()
        p_soft = F.softmax(logits / T, dim=1).cpu().numpy().ravel()
        print(f"  样本真值类别 = {y[idx].item()}")
        print(f"  T=1   概率分布: {np.array2string(p_hard, precision=3, floatmode='fixed')}")
        print(f"  T={T:g} 概率分布: {np.array2string(p_soft, precision=3, floatmode='fixed')}")
        print("  观察：升高温度后分布变「软」，非目标类的相对概率被放大 ——")
        print("        这些类间相似性正是 student 要从 teacher 学到的暗知识。")


# ============================================================
#  NumPy 部分：无 torch 时演示温度软化机制
# ============================================================
def numpy_temperature_demo():
    print(SEP)
    print("NumPy 原理演示：温度 T 如何软化概率分布")
    print(SEP)

    def softmax_T(z, T):
        z = z / T
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    logits = np.array([4.0, 2.0, 1.0, 0.5], dtype=np.float32)
    print(f"  原始 logits = {logits}")
    for T in (1.0, 2.0, 4.0, 8.0):
        p = softmax_T(logits, T)
        print(f"  T={T:>3.0f}  softmax = {np.array2string(p, precision=3, floatmode='fixed')}"
              f"   (熵={-np.sum(p*np.log(p+1e-12)):.3f})")
    print("  观察：T 越大分布越平缓（熵越大），类间相对关系越突出。")
    print("        student 用同一个 T 去拟合 teacher 的软标签，即可学到暗知识。")


# ============================================================
#  概念讲解
# ============================================================
def explain_distillation():
    print(SEP)
    print("知识蒸馏概念")
    print(SEP)
    print("""
  角色         说明
  ------------------------------------------------------------
  teacher      大模型/集成模型，精度高但推理慢
  student      小模型，参数少推理快，目标是逼近 teacher 精度
  soft label   teacher 经温度 T 软化的概率分布（含暗知识）
  hard label   数据集真值 one-hot 标签

  损失函数（加权）：
    L = alpha * T^2 * KL(student_soft || teacher_soft)  # 蒸馏项
      + (1-alpha) * CE(student_logits, hard_label)      # 硬标签项

  关键点：
   1. 温度 T：软化 softmax，放大类间相似性（暗知识）。蒸馏项乘 T^2
      补偿梯度缩放；推理时 student 用 T=1。
   2. alpha：软/硬标签权重。alpha 偏大更信任 teacher，偏小更信任真值。
   3. 直觉：teacher 的「3 像 8 不像 1」比 one-hot 提供更多监督信号，
      student 因此能用更少参数学得更好。
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

    # 1) 训练 teacher
    teacher = TeacherNet().to(DEVICE)
    train_plain(teacher, Xtr, ytr, epochs=150)
    acc_teacher = evaluate(teacher, Xte, yte)
    print(f"[teacher] 参数量={count_params(teacher):>6d}  测试精度={acc_teacher:5.2f}%")

    # 展示暗知识
    print()
    print(SEP)
    print("暗知识（dark knowledge）演示")
    print(SEP)
    show_dark_knowledge(teacher, Xte, yte, T=4.0)

    # 2) 直接训练 student（基线）
    print()
    print(SEP)
    print("对比实验：直接训练 vs 知识蒸馏（相同 student 结构）")
    print(SEP)
    student_plain = StudentNet().to(DEVICE)
    train_plain(student_plain, Xtr, ytr, epochs=120)
    acc_plain = evaluate(student_plain, Xte, yte)

    # 3) 蒸馏训练 student
    student_kd = StudentNet().to(DEVICE)
    train_distill(student_kd, teacher, Xtr, ytr, T=4.0, alpha=0.7, epochs=120)
    acc_kd = evaluate(student_kd, Xte, yte)

    sp = count_params(student_plain)
    tp = count_params(teacher)
    print(f"  teacher (大)          : 参数量={tp:>6d}  精度={acc_teacher:5.2f}%")
    print(f"  student 直接训练      : 参数量={sp:>6d}  精度={acc_plain:5.2f}%")
    print(f"  student 知识蒸馏      : 参数量={sp:>6d}  精度={acc_kd:5.2f}%")
    print(f"  压缩比 (teacher/student) = {tp / sp:.1f}x")
    gain = acc_kd - acc_plain
    print(f"  蒸馏相对直接训练精度变化 = {gain:+.2f} 个百分点")

    # 4) 扫描温度 T 的影响
    print()
    print(SEP)
    print("温度 T 扫描（alpha=0.7 固定）")
    print(SEP)
    print(f"{'温度 T':>8} | {'student 蒸馏精度':>16}")
    print("-" * 30)
    for T in (1.0, 2.0, 4.0, 8.0):
        s = StudentNet().to(DEVICE)
        train_distill(s, teacher, Xtr, ytr, T=T, alpha=0.7, epochs=120)
        print(f"{T:7.0f} | {evaluate(s, Xte, yte):15.2f}%")
    print("  观察：适中的 T（常见 2~5）通常最好；过大过小都可能损失暗知识。")

    return 0


def main():
    print()
    print(SEP)
    print("Day 64: 知识蒸馏（Knowledge Distillation）")
    print(SEP)

    explain_distillation()

    if TORCH_AVAILABLE:
        run_torch()
    else:
        print(SEP)
        print("环境：未安装 PyTorch，使用 NumPy 演示温度软化核心机制。")
        print(SEP)
        numpy_temperature_demo()

    print()
    print(SEP)
    print("知识点总结")
    print(SEP)
    print("""  1. 知识蒸馏：让小 student 模仿大 teacher 的软标签，用更少参数逼近其精度。
  2. 暗知识：teacher softmax 里的类间相似性，比 one-hot 硬标签信息更丰富。
  3. 温度 T：软化分布放大暗知识；蒸馏项乘 T^2 补偿梯度；推理时 T=1。
  4. 损失：alpha·T²·KL(soft) + (1-alpha)·CE(hard)，权衡 teacher 与真值。
  5. 蒸馏常与量化/剪枝叠加使用，是端侧小模型的常用「涨点」手段。""")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
