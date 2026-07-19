#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 88: Linux 工程化 — GPU 异常排查 / coredump / compute-sanitizer

真实训练/推理里最耗时的往往不是写 kernel，而是排查线上问题：显存 OOM、
非法内存访问(illegal memory access)、hang(卡死)、loss 变 NaN。本天系统梳理
这些问题的**症状 → 定位工具 → 修复思路**，并用可运行的诊断工具演示：

  1. NaN/Inf 检测钩子：训练中自动定位「哪一层最先产生 NaN」；
  2. OOM 模拟 + 显存监控：估算张量显存、模拟 OOM，展示 OOM 时该看什么；
  3. 常见 GPU 问题速查表 + compute-sanitizer / cuda-gdb / coredump 用法讲解。

运行策略（能不回退就不回退）：
  - 有 torch + CUDA：用真实张量做 NaN 钩子与显存监控演示；
  - 无 GPU/torch：用 numpy/纯 Python 等价演示同样的诊断逻辑，稳定退出。
"""

import os

import numpy as np

# ------------------------------------------------------------
# 环境探测（三布尔量）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

SEP = "=" * 64


# ============================================================
#  1. 常见 GPU 问题速查表
# ============================================================
def print_troubleshoot_table():
    print(SEP)
    print("常见 GPU 问题速查：症状 → 定位工具 → 修复思路")
    print(SEP)
    print("""
  问题类型            典型症状                    定位工具/方法              常见修复
  ---------------------------------------------------------------------------------
  OOM(显存不足)       CUDA out of memory          nvidia-smi / torch.cuda    减 batch/激活重计算
                      报错并给出已分配/保留量       .memory_summary()          /梯度检查点/量化
  非法内存访问        illegal memory access       compute-sanitizer          修越界索引/同步
                      (常延迟报错, 栈不准)         (memcheck) / cuda-gdb      /检查 kernel 边界
  hang(卡死)          进程不退、GPU 利用率异常      py-spy / nsys / gdb 抓栈   死锁/通信不匹配
                      (常见于多卡通信不对齐)                                  /collective 对齐
  NaN/Inf             loss=nan、梯度爆炸           NaN 钩子 / autograd        降 lr/加 clip/查
                      精度异常                     detect_anomaly             log(0)、除0、溢出
  精度不对齐          CPU/GPU 结果不一致            逐层对拍 / 双精度参考       非确定性算子/TF32
  性能骤降            吞吐下降、卡顿                nsys/ncu profiling         访存/同步/小 kernel

  关键命令行工具：
    nvidia-smi                    看显存占用、进程、GPU 利用率、温度
    compute-sanitizer ./app       检测非法访存/竞争/未初始化(替代旧 cuda-memcheck)
    cuda-gdb --args python x.py    GPU 断点调试, 崩溃时看 kernel 栈
    ulimit -c unlimited           开启 coredump; CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
    nsys profile / ncu            时间线 / kernel 级性能剖析(见 Day35)
""")


# ============================================================
#  2. NaN/Inf 检测钩子：定位哪一层最先出问题
# ============================================================
def demo_nan_hook_torch():
    print(SEP)
    print("演示 2：NaN/Inf 检测钩子（PyTorch forward hook 定位问题层）")
    print(SEP)

    import torch.nn as nn

    torch.manual_seed(0)

    class BuggyNet(nn.Module):
        """故意在中间制造一个会产生 Inf/NaN 的算子。"""

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 8)
            self.fc3 = nn.Linear(8, 4)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            # 故意制造数值灾难：连乘极大值让 fp32 溢出 → Inf
            x = x * 1e30 * 1e30
            x = self.fc3(x)
            return x

    def make_nan_hook(name):
        def hook(module, inputs, output):
            out = output if isinstance(output, torch.Tensor) else output[0]
            if torch.isnan(out).any() or torch.isinf(out).any():
                n_nan = int(torch.isnan(out).sum())
                n_inf = int(torch.isinf(out).sum())
                print(f"    [NaN钩子] 层 '{name}' 输出异常: "
                      f"NaN={n_nan}, Inf={n_inf}  <== 首个出问题的层")
        return hook

    net = BuggyNet()
    if CUDA_AVAILABLE:
        net = net.cuda()
    handles = [m.register_forward_hook(make_nan_hook(n))
               for n, m in net.named_modules() if len(list(m.children())) == 0]

    x = torch.randn(4, 8)
    if CUDA_AVAILABLE:
        x = x.cuda()
    print("  前向一次，钩子会在第一个产生 NaN/Inf 的层报警：")
    _ = net(x)
    for h in handles:
        h.remove()
    print("  → 定位到问题层后，检查该层是否有 除0 / log(0) / 溢出 / 未归一化。")
    print("  提示：也可用 torch.autograd.set_detect_anomaly(True) 定位反向 NaN。")


def demo_nan_hook_numpy():
    print(SEP)
    print("演示 2：NaN/Inf 检测钩子（NumPy 版，逐层检查）")
    print(SEP)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)

    def layer_linear(a, out_dim, name, buggy=False):
        w = rng.standard_normal((a.shape[1], out_dim)).astype(np.float32)
        y = a @ w
        if buggy:
            y = y * np.float32(1e30) * np.float32(1e30)   # fp32 溢出 → Inf
        if not np.isfinite(y).all():
            n_nan = int(np.isnan(y).sum())
            n_inf = int(np.isinf(y).sum())
            print(f"    [NaN钩子] 层 '{name}' 输出异常: "
                  f"NaN={n_nan}, Inf={n_inf}  <== 首个出问题的层")
        return y

    h1 = np.maximum(layer_linear(x, 8, "fc1"), 0)
    h2 = layer_linear(h1, 8, "fc2", buggy=True)   # 这里会炸
    _ = layer_linear(h2, 4, "fc3")
    print("  → 定位到问题层后，检查该层是否有 除0 / log(0) / 溢出 / 未归一化。")


# ============================================================
#  3. 显存估算 + OOM 模拟
# ============================================================
def estimate_tensor_mb(shape, dtype_bytes=4):
    n = 1
    for s in shape:
        n *= s
    return n * dtype_bytes / (1024 * 1024)


def demo_oom_and_memory():
    print()
    print(SEP)
    print("演示 3：显存估算 + OOM 排查")
    print(SEP)

    # 先做纯计算的显存估算（任何环境都能讲清）
    shapes = [("激活 [B=32,S=2048,H=4096] fp16", (32, 2048, 4096), 2),
              ("权重 [4096,4096] fp32", (4096, 4096), 4),
              ("KV Cache [32,32,2048,128] fp16", (32, 32, 2048, 128), 2)]
    print("  常见张量显存估算（元素数 × 每元素字节）：")
    print(f"    {'张量':<40}{'显存(MB)':>12}")
    print("    " + "-" * 52)
    for name, shape, db in shapes:
        print(f"    {name:<40}{estimate_tensor_mb(shape, db):>12.1f}")
    print()

    if CUDA_AVAILABLE:
        free0, total0 = torch.cuda.mem_get_info()
        print(f"  当前 GPU 显存: 空闲 {free0/1e9:.2f} GB / 共 {total0/1e9:.2f} GB")
        # 分配一个中等张量，观察显存变化
        try:
            a = torch.empty((2048, 2048), device="cuda")  # 16MB fp32
            alloc = torch.cuda.memory_allocated() / 1e6
            print(f"  分配 2048x2048 fp32 后 memory_allocated = {alloc:.1f} MB")
            del a
            torch.cuda.empty_cache()
            print("  已释放并 empty_cache()。")
        except RuntimeError as e:
            print(f"  分配失败：{e}")

        # 演示 OOM 的“正确姿势”：捕获并打印诊断信息（不真的把显存撑爆）
        print()
        print("  模拟 OOM 处理流程（不真正耗尽显存，仅演示排查动作）：")
        huge_gb = (total0 / 1e9) * 100     # 故意远超显存
        need_bytes = int(huge_gb * 1e9)
        try:
            _ = torch.empty(need_bytes // 4, dtype=torch.float32, device="cuda")
            print("    (本机显存意外充足，未触发 OOM)")
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            print(f"    捕获到分配异常: {msg}")
            print("    排查动作: 1) 看 torch.cuda.memory_summary() 找占用大头")
            print("             2) nvidia-smi 看是否有别的进程占显存")
            print("             3) 减 batch / 梯度检查点 / AMP / 量化")
        finally:
            torch.cuda.empty_cache()
    else:
        print("  (无 CUDA GPU) 用上面的估算公式即可在部署前预判显存是否够用：")
        print("    总显存 ≈ 参数 + 梯度 + 优化器状态 + 激活 + KV Cache + 碎片余量")
        print("    OOM 时优先：减 batch / 梯度检查点 / 混合精度 / 权重量化。")


# ============================================================
#  4. coredump / compute-sanitizer 用法讲解
# ============================================================
def print_debug_tools_guide():
    print()
    print(SEP)
    print("coredump / compute-sanitizer / cuda-gdb 实战要点")
    print(SEP)
    print("""
  1) compute-sanitizer（首选，替代旧 cuda-memcheck）
     $ compute-sanitizer --tool memcheck python train.py
       检测：越界读写、未初始化访问、对齐错误、内存泄漏
     $ compute-sanitizer --tool racecheck ./app   # 检测共享内存竞争
     $ compute-sanitizer --tool synccheck ./app   # 检测 __syncthreads 误用
     注意：非法访存常“延迟报错”，报错行号未必是真凶 → 用 sanitizer 精确定位。

  2) GPU coredump（崩溃现场快照）
     $ export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
     $ ulimit -c unlimited
       kernel 异常时会生成 core.* / *.nvcudmp，可用 cuda-gdb 事后分析。

  3) cuda-gdb（GPU 断点调试）
     $ cuda-gdb --args python train.py
       (cuda-gdb) run
       崩溃后:  (cuda-gdb) bt        # 看主机+设备调用栈
                (cuda-gdb) cuda kernel block thread   # 定位到具体线程

  4) hang(卡死) 排查
     - py-spy dump --pid <PID>       # 无侵入抓 Python 栈, 看卡在哪
     - nsys profile                  # 看时间线上是否卡在某次 collective/同步
     - 多卡：确认各 rank 的 all_reduce/broadcast 形状与调用次数一致

  5) NaN 排查（对应上面的钩子演示）
     - torch.autograd.set_detect_anomaly(True)   # 定位反向第一处 NaN
     - 前向 register_forward_hook 逐层查 isnan/isinf
     - 常见根因：lr 过大、无 grad clip、log(0)/除0、fp16 溢出
""")


# ============================================================
#  主流程
# ============================================================
def main():
    print_troubleshoot_table()
    print()
    if TORCH_AVAILABLE:
        demo_nan_hook_torch()
    else:
        demo_nan_hook_numpy()
    demo_oom_and_memory()
    print_debug_tools_guide()

    print()
    print(SEP)
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        print(f"环境：PyTorch + CUDA 就绪（{torch.cuda.get_device_name(0)}）。")
        print("已用真实张量演示 NaN 钩子与显存监控。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU。NaN 钩子在 CPU 上同样有效。")
    else:
        print("环境：未安装 PyTorch。已用 NumPy 演示等价诊断逻辑。")
    print("提示：compute-sanitizer / cuda-gdb / coredump 需 Linux + CUDA 工具链，")
    print("      本天命令行部分为讲解，代码部分在任意环境可跑。")
    print(SEP)


if __name__ == "__main__":
    main()
