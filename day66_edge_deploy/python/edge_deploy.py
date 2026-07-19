#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 66: 端侧部署实战 — FP16/INT8 + 精度验证（第十一阶段收官）

把本阶段学到的东西串成一条完整链路，用一个真实可跑的小模型走一遍：

    训练(FP32)  ->  量化(FP16 / INT8 模拟)  ->  导出  ->  部署对比

并对三种精度做「精度 / 模型大小 / 推理耗时」三维对比 + 精度验证。

精度方案（都用可跑的 PyTorch 模拟，无需 TensorRT）：
  - FP32：基线，float32 权重与激活。
  - FP16：权重与激活转 half（有 GPU 时真半精度；CPU 上 half 支持有限，
          用「float16 存储 + float32 计算」模拟其数值影响）。
  - INT8：对称 per-tensor 量化模拟 —— 用校准集统计激活范围求 scale，
          权重/激活量化到 int8 再反量化（fake quant），观察精度损失。

「模型大小」按每种精度的字节数估算（FP32=4B, FP16=2B, INT8=1B）。
「推理耗时」实测多次取均值（相对趋势为主，绝对值随硬件而变）。

运行策略：有 torch 就真实训练/推理（有 CUDA 用 GPU，否则 CPU）；
无 torch 用 NumPy 演示 INT8 对称量化的核心数值流程。所有分支 exit code 0。
"""

import sys
import time
import copy
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
#  合成数据集：多类分类
# ============================================================
def make_dataset(n_per_class=400, dim=64, n_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for c in range(n_classes):
        center = rng.normal(0, 1.0, size=dim).astype(np.float32) * 0.9
        Xs.append(rng.normal(center, 1.6, size=(n_per_class, dim)).astype(np.float32))
        ys.append(np.full(n_per_class, c, dtype=np.int64))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ============================================================
#  PyTorch 部分
# ============================================================
if TORCH_AVAILABLE:

    class MLPClassifier(nn.Module):
        def __init__(self, dim=64, hidden=256, n_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, n_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    def train_fp32(model, X, y, epochs=120, lr=1e-2):
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
        return model

    @torch.no_grad()
    def accuracy(model, X, y):
        model.eval()
        return (model(X).argmax(1) == y).float().mean().item() * 100.0

    # ---------- INT8 对称 per-tensor 量化模拟 ----------
    def symmetric_quant_params(t, num_bits=8):
        """由张量范围求对称量化 scale（zero_point=0）。"""
        qmax = 2 ** (num_bits - 1) - 1          # int8 -> 127
        max_abs = t.abs().max().clamp(min=1e-8)
        scale = max_abs / qmax
        return scale, qmax

    def fake_quant(t, scale, qmax):
        """量化到整数再反量化（模拟 INT8 推理的数值精度损失）。"""
        q = torch.clamp(torch.round(t / scale), -qmax - 1, qmax)
        return q * scale

    class Int8SimLinear(nn.Module):
        """用 fake-quant 模拟 INT8 线性层：权重按 per-tensor 量化，
        激活用校准得到的 scale 量化。"""

        def __init__(self, linear, act_scale, act_qmax):
            super().__init__()
            w = linear.weight.data
            w_scale, w_qmax = symmetric_quant_params(w)
            self.register_buffer("qweight", fake_quant(w, w_scale, w_qmax))
            self.register_buffer("bias", linear.bias.data.clone())
            self.act_scale = act_scale
            self.act_qmax = act_qmax

        def forward(self, x):
            x = fake_quant(x, self.act_scale, self.act_qmax)  # 量化激活
            return F.linear(x, self.qweight, self.bias)

    @torch.no_grad()
    def calibrate_act_scales(model, calib_X):
        """PTQ 校准：前向一遍，统计每个 Linear 输入激活的范围，求 scale。"""
        scales = {}
        acts = {}

        def make_hook(name):
            def hook(module, inp, out):
                acts[name] = inp[0].detach()
            return hook

        handles = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(make_hook(name)))
        model.eval()
        model(calib_X)
        for h in handles:
            h.remove()
        for name, a in acts.items():
            s, qmax = symmetric_quant_params(a)
            scales[name] = (s, qmax)
        return scales

    @torch.no_grad()
    def build_int8_model(fp32_model, calib_X):
        """把 FP32 模型转成 INT8 模拟模型（PTQ）。"""
        scales = calibrate_act_scales(fp32_model, calib_X)
        m = copy.deepcopy(fp32_model)
        for name, sub in list(m.named_modules()):
            if isinstance(sub, nn.Linear):
                s, qmax = scales[name]
                parent = m
                *path, last = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, Int8SimLinear(sub, s, qmax))
        return m

    @torch.no_grad()
    def bench_latency(model, X, iters=50):
        """实测推理耗时（ms/次），取均值。"""
        model.eval()
        # 预热
        for _ in range(5):
            model(X)
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model(X)
        if CUDA_AVAILABLE:
            torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000.0

    def model_size_bytes(model, bytes_per_param):
        """按每参数字节数估算模型大小（KB），统计权重参数 + buffer 量化权重。"""
        n = 0
        for p in model.parameters():
            n += p.numel()
        for b in model.buffers():
            n += b.numel()
        return n * bytes_per_param / 1024.0


# ============================================================
#  NumPy 部分：无 torch 时演示 INT8 对称量化数值流程
# ============================================================
def numpy_int8_demo():
    print(SEP)
    print("NumPy 原理演示：INT8 对称 per-tensor 量化")
    print(SEP)
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.5, size=(4, 6)).astype(np.float32)
    qmax = 127
    scale = np.abs(w).max() / qmax
    q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    deq = q.astype(np.float32) * scale
    err = np.abs(w - deq).mean()
    print(f"  权重范围 [{w.min():.3f}, {w.max():.3f}]  scale={scale:.5f}")
    print(f"  量化后 int8 示例（第一行）: {q[0]}")
    print(f"  反量化平均绝对误差 = {err:.5f}（远小于 scale，说明量化可用）")
    print(f"  存储：FP32={w.nbytes} B  ->  INT8={q.nbytes} B（4x 压缩）")


# ============================================================
#  概念讲解
# ============================================================
def explain_edge_deploy():
    print(SEP)
    print("端侧部署：精度 vs 大小 vs 速度的三角权衡")
    print(SEP)
    print("""
  精度     每参数字节   相对大小   典型速度   精度损失    使用条件
  ------------------------------------------------------------------
  FP32     4 B          1.0x       1.0x       无(基线)    训练/高精度需求
  FP16     2 B          0.5x       ~2x        极小        主流部署默认
  INT8     1 B          0.25x      ~2-4x      小(需校准)  端侧/高吞吐

  端侧部署完整链路：
    训练(FP32) -> 量化(FP16/INT8) -> 导出(ONNX) -> 引擎(TensorRT) -> 部署
                    │
                    ├─ FP16：直接转半精度，几乎无损，一行 flag。
                    └─ INT8：PTQ 需校准集统计激活范围求 scale；
                             掉点多时上 QAT（量化感知训练）。

  精度验证：部署后必须用验证集核对 top-1 / 关键指标，
            确认量化掉点在可接受范围（通常要求 < 1%）。
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
    calib = Xtr[:256]  # 校准集（PTQ 用）

    print(SEP)
    print(f"环境：PyTorch 就绪，device = {DEVICE}")
    print(SEP)

    # 1) 训练 FP32 基线
    fp32 = MLPClassifier().to(DEVICE)
    train_fp32(fp32, Xtr, ytr)
    acc_fp32 = accuracy(fp32, Xte, yte)
    print(f"[训练] FP32 基线训练完成，测试精度 = {acc_fp32:.2f}%")

    # 2) FP16 模型
    if CUDA_AVAILABLE:
        fp16 = copy.deepcopy(fp32).half()      # 真半精度（GPU）
        Xte16 = Xte.half()
        acc_fp16 = accuracy(fp16, Xte16, yte)
        lat_fp16 = bench_latency(fp16, Xte16)
    else:
        # CPU 上 half 计算支持有限：float16 存储 + float32 计算，模拟数值影响
        fp16 = copy.deepcopy(fp32)
        with torch.no_grad():
            for p in fp16.parameters():
                p.copy_(p.half().float())      # 经过一次 fp16 舍入
        acc_fp16 = accuracy(fp16, Xte, yte)
        lat_fp16 = bench_latency(fp16, Xte)

    # 3) INT8 模拟模型（PTQ + 校准）
    int8 = build_int8_model(fp32, calib)
    acc_int8 = accuracy(int8, Xte, yte)

    # 4) 三种精度对比
    lat_fp32 = bench_latency(fp32, Xte)
    lat_int8 = bench_latency(int8, Xte)
    size_fp32 = model_size_bytes(fp32, 4)
    size_fp16 = model_size_bytes(fp32, 2)
    size_int8 = model_size_bytes(fp32, 1)

    print()
    print(SEP)
    print("三精度对比：精度 / 模型大小 / 推理耗时")
    print(SEP)
    header = f"{'精度':>6} | {'测试精度':>8} | {'掉点':>7} | {'模型大小':>10} | {'耗时(ms)':>9}"
    print(header)
    print("-" * len(header))
    print(f"{'FP32':>6} | {acc_fp32:7.2f}% | {'--':>7} | {size_fp32:8.1f}KB | {lat_fp32:9.3f}")
    print(f"{'FP16':>6} | {acc_fp16:7.2f}% | {acc_fp32-acc_fp16:+6.2f}% | "
          f"{size_fp16:8.1f}KB | {lat_fp16:9.3f}")
    print(f"{'INT8':>6} | {acc_int8:7.2f}% | {acc_fp32-acc_int8:+6.2f}% | "
          f"{size_int8:8.1f}KB | {lat_int8:9.3f}")

    # 5) 精度验证结论
    print()
    print(SEP)
    print("精度验证（部署前必做）")
    print(SEP)
    drop_fp16 = acc_fp32 - acc_fp16
    drop_int8 = acc_fp32 - acc_int8
    thr = 1.0
    print(f"  FP16 掉点 {drop_fp16:+.2f}%  ->  "
          f"{'通过 ✓ (<1%)' if abs(drop_fp16) < thr else '需关注 (>=1%)'}")
    print(f"  INT8 掉点 {drop_int8:+.2f}%  ->  "
          f"{'通过 ✓ (<1%)' if abs(drop_int8) < thr else '偏大，建议 QAT 或 per-channel'}")
    print("  结论：FP16 通常无损可直接上；INT8 掉点大时改 per-channel 量化或 QAT。")
    print("  注：本例耗时为 PyTorch 模拟值 —— INT8 走 fake-quant（含额外量化算子）")
    print("      故看起来更慢；真实 INT8/FP16 的加速来自 TensorRT 的专用低精度 kernel。")
    if not CUDA_AVAILABLE:
        print("  注：当前为 CPU，FP16/INT8 均为数值模拟，真实加速需 GPU + TensorRT。")

    return 0


def main():
    print()
    print(SEP)
    print("Day 66: 端侧部署实战 — FP16/INT8 + 精度验证（第十一阶段收官）")
    print(SEP)

    explain_edge_deploy()

    if TORCH_AVAILABLE:
        run_torch()
    else:
        print(SEP)
        print("环境：未安装 PyTorch，使用 NumPy 演示 INT8 对称量化数值流程。")
        print(SEP)
        numpy_int8_demo()

    print()
    print(SEP)
    print("知识点总结")
    print(SEP)
    print("""  1. 部署三角：精度 / 模型大小 / 推理速度需权衡，端侧优先小而快。
  2. FP16：字节减半、约 2x 加速、几乎无损，主流部署默认精度。
  3. INT8：4x 压缩、更高吞吐，但需校准集统计 scale，掉点大时上 QAT。
  4. PTQ 流程：训练 -> 校准(统计激活范围) -> 量化 -> 精度验证。
  5. 精度验证是部署最后一道关：掉点须控制在指标可接受范围（常 <1%）。
  6. 完整链路：训练 -> 量化 -> ONNX 导出 -> TensorRT Engine -> 端侧推理。""")
    print(SEP)

    # ---- 第十一阶段回顾 ----
    print()
    print(SEP)
    print("第十一阶段（Day 59-66）模型压缩与部署 · 回顾")
    print(SEP)
    print("""  Day 59  量化基础       对称/非对称、per-tensor/per-channel、量化/反量化
  Day 60  PTQ + 校准       训练后量化，用校准集统计范围求 scale
  Day 61  QAT              量化感知训练，插入伪量化节点让模型适应量化
  Day 62  大模型量化       GPTQ / AWQ / SmoothQuant 权重量化原理
  Day 63  剪枝             结构化 vs 非结构化 + 剪枝后微调恢复精度
  Day 64  知识蒸馏         teacher->student，软标签+温度 T 的 KD loss
  Day 65  TensorRT 深入    ONNX -> Engine -> Plugin 完整部署链路
  Day 66  端侧部署实战     FP32/FP16/INT8 对比 + 精度验证（本天收官）

  主线：量化(59-62) + 剪枝(63) + 蒸馏(64) = 模型压缩三板斧；
        TensorRT(65) + 端侧部署(66) = 把压缩后的模型高效落地。
  承接：Day24 TensorRT 概念、Day53 推理量化；
  下一阶段：第十二阶段 分布式训练（Day 67-80）。""")
    print(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
