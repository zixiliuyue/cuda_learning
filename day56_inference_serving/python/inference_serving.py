#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 56: 推理服务化 — 并发调度与低延迟

一个 LLM 训练完成后，真正对外提供服务时面对的核心矛盾是：
  - 用户希望 **低延迟**（我发一个请求，立刻给我结果）；
  - 平台希望 **高吞吐**（同样的 GPU，尽可能多地服务请求）。

这两者天然冲突：攒一个大 batch 一起算能显著提升吞吐（GPU 利用率高），
但排队攒批会增加每个请求的等待延迟。本天用一个「简易 serving」把这个
权衡量化出来：

  请求队列(Queue) → 批处理调度器(Scheduler) → 模型前向(batched forward) → 统计

为什么用「事件模拟」而不是纯 time.sleep 线程？
  真实前向耗时是微秒级，而 OS 定时精度（尤其 Windows）与 Python GIL 会
  严重污染微秒级的并发计时，让攒批权衡淹没在噪声里。所以本天采用
  **离散事件模拟(DES)**：
    - 前向耗时用真实张量运算「真测」（有 CUDA 用 GPU，无 CUDA 用 CPU）；
    - 请求到达 / 排队 / 攒批调度在一条确定的「事件时间线」上推演。
  这样既是真实计算，又能稳定、可复现地展示 batch 对延迟/吞吐的影响。

运行策略（能不回退就不回退）：
  - 有 PyTorch：真实 batched 前向（CUDA→GPU / 否则 CPU），逻辑一致；
  - 无 PyTorch：用 NumPy 做等价 batched 矩阵运算，原理不打折。
"""

import time
import random

import numpy as np

# ------------------------------------------------------------
# 环境探测（优先真实 GPU，不轻易回退）
# ------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

SEP = "=" * 64

# 玩具「模型」规模：把请求向量 (D,) 过两层全连接，模拟一次前向的计算量
D_MODEL = 512
HIDDEN = 4096


# ============================================================
#  玩具模型：一次 batched 前向（batch 越大，单请求摊薄的固定开销越低）
# ============================================================
class ToyModel:
    """两层 MLP，代表「一次模型前向」。

    关键点：无论 batch 里有几个请求，都是一次矩阵乘搞定。
    一次前向 ≈ 固定开销(kernel 启动/调度) + 与 batch 成正比的计算。
    batch 越大 → 固定开销被摊薄得越狠 → 单请求成本下降 → 吞吐上升。
    """

    def __init__(self):
        rng = np.random.default_rng(0)
        w1 = rng.standard_normal((D_MODEL, HIDDEN)).astype(np.float32) * 0.02
        w2 = rng.standard_normal((HIDDEN, D_MODEL)).astype(np.float32) * 0.02
        if TORCH_AVAILABLE:
            self.w1 = torch.tensor(w1, device=DEVICE)
            self.w2 = torch.tensor(w2, device=DEVICE)
        else:
            self.w1, self.w2 = w1, w2

    def forward(self, batch):
        """batch: (B, D_MODEL) → (B, D_MODEL)。代表处理 B 个请求的一次前向。"""
        if TORCH_AVAILABLE:
            x = torch.relu(batch @ self.w1)
            out = x @ self.w2
            if DEVICE == "cuda":
                torch.cuda.synchronize()  # GPU 异步，计时前必须同步
            return out
        x = np.maximum(batch @ self.w1, 0.0)
        return x @ self.w2


def make_input(batch_size):
    """构造一个 (B, D_MODEL) 的输入 batch。"""
    if TORCH_AVAILABLE:
        return torch.randn(batch_size, D_MODEL, device=DEVICE)
    return np.random.randn(batch_size, D_MODEL).astype(np.float32)


def measure_forward_ms(model, batch_size, warmup=3, iters=10):
    """真实测量某个 batch_size 下一次前向的耗时(ms)。

    正确计时三要素：warmup(触发编译/稳定时钟) + 多次取均值 + 设备同步。
    """
    x = make_input(batch_size)
    for _ in range(warmup):
        model.forward(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model.forward(x)
    return (time.perf_counter() - t0) / iters * 1e3


# ============================================================
#  离散事件模拟的调度器：在事件时间线上贪心攒批
# ============================================================
def simulate(arrivals, forward_ms, max_batch, max_wait_ms):
    """在确定的事件时间线上模拟「队列 + 批处理调度」。

    参数：
      arrivals    : 已排序的请求到达时刻列表(ms)，模拟并发到达
      forward_ms  : 函数 B -> 处理 B 个请求的一次前向耗时(ms)，真测得到
      max_batch   : 攒批上限
      max_wait_ms : 首个请求进入后，最长再等这么久来凑批

    调度逻辑（贪心攒批）：
      1. 服务器空闲时，取队首请求作为一批的第一个；
      2. 在 max_wait 窗口内，把已到达/即将到达的请求尽量收进来，
         直到攒够 max_batch 或超时；
      3. 触发一次 batched 前向，期间到达的请求进入下一批。

    返回：(每个请求的端到端延迟列表 ms, 总完成时刻 ms)
    """
    n = len(arrivals)
    latencies = []
    i = 0                 # 下一个待服务请求下标
    server_free_at = 0.0  # 服务器下次空闲时刻
    last_finish = 0.0

    while i < n:
        # 一批的第一个请求：要么队列里已有，要么等它到达
        batch_start_wait = max(arrivals[i], server_free_at)
        # 攒批截止时刻 = 首个请求被考虑处理的时刻 + max_wait
        deadline = batch_start_wait + max_wait_ms
        j = i
        # 收集这一批：满足 batch 上限 且 到达时刻不超过攒批截止
        while (j < n
               and (j - i) < max_batch
               and arrivals[j] <= deadline):
            j += 1
        batch = j - i
        # 这一批真正开始计算的时刻：服务器空闲 且 批内最后一个已到达
        begin = max(server_free_at, arrivals[j - 1])
        # 若因等待窗口提前触发，也不能早于批内最后请求到达
        begin = max(begin, batch_start_wait)
        dur = forward_ms(batch)
        finish = begin + dur
        for k in range(i, j):
            latencies.append(finish - arrivals[k])  # 端到端 = 完成 - 到达
        server_free_at = finish
        last_finish = finish
        i = j

    return latencies, last_finish


def gen_arrivals(n_requests, arrival_rate_per_ms):
    """生成 n 个请求的到达时刻(ms)：泊松过程 → 指数间隔累加。"""
    t = 0.0
    arrivals = []
    for _ in range(n_requests):
        t += random.expovariate(arrival_rate_per_ms)
        arrivals.append(t)
    return arrivals


def stats(latencies, makespan_ms, n_requests):
    lat = sorted(latencies)
    avg = sum(lat) / len(lat)
    p99 = lat[min(len(lat) - 1, int(len(lat) * 0.99))]
    throughput = n_requests / (makespan_ms / 1e3)  # req/s
    return avg, p99, throughput


# ============================================================
#  原理讲解
# ============================================================
def explain():
    print(SEP)
    print("推理服务化：延迟 vs 吞吐的核心权衡")
    print(SEP)
    print("""
  组件            职责
  ------------------------------------------------------------
  请求队列        缓冲并发到达的请求，削峰填谷
  批处理调度器    从队列攒一批(受 max_batch/max_wait 约束)后统一前向
  batched 前向    一次矩阵乘服务整批，摊薄固定开销 → 提升吞吐
  统计            端到端延迟(avg/P99) 与吞吐(req/s)

  两个旋钮：
    max_batch  攒批上限：↑ 吞吐↑ 但等待↑ 延迟↑
    max_wait   最长攒批等待：低流量时给攒批设超时上限，避免空等

  关键结论：
    - 高吞吐来自「大 batch 摊薄固定开销」，代价是排队等待；
    - 低延迟要求「来了就算」，但小 batch 让 GPU 吃不饱；
    - 真实系统(vLLM 等)用 continuous batching 缓解：不等整批算完，
      新请求可随时插入正在运行的 batch，兼顾两者(见 Day57)。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    random.seed(0)
    np.random.seed(0)
    explain()

    if TORCH_AVAILABLE:
        env = f"PyTorch 就绪，设备 = {DEVICE.upper()}"
    else:
        env = "未安装 PyTorch，使用 NumPy 做等价 batched 运算"
    print(f"[环境] {env}")
    print(f"[模型] 两层 MLP  D_MODEL={D_MODEL}  HIDDEN={HIDDEN}")

    model = ToyModel()

    # ---- 第一步：真实测量各 batch 下的前向耗时，构造 forward_ms 查表 ----
    print()
    print(SEP)
    print("步骤1：真实测量各 batch 的前向耗时（体现固定开销被摊薄）")
    print(SEP)
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    fwd_table = {b: measure_forward_ms(model, b) for b in batch_sizes}
    print(f"{'batch':>6} | {'一次前向(ms)':>12} | {'每请求(ms)':>11}")
    print("-" * 36)
    for b in batch_sizes:
        print(f"{b:>6} | {fwd_table[b]:>12.4f} | {fwd_table[b] / b:>11.4f}")
    print("  ↑ 每请求耗时随 batch 增大而下降 = 固定开销被摊薄，这是吞吐提升的根源。")

    def forward_ms(b):
        # 用最接近的已测 batch 估计，避免每种批量都重测
        key = min(batch_sizes, key=lambda k: abs(k - b))
        return fwd_table[key] / key * b if b not in fwd_table else fwd_table[b]

    # ---- 第二步：固定并发流量，扫 max_batch，看延迟/吞吐权衡 ----
    N = 2000
    # 到达速率设得比单请求处理能力高 → 队列会积压 → 攒批真正生效
    rate_per_ms = 1.0 / (fwd_table[1] * 0.5)  # 约为单请求吞吐的 2 倍负载
    arrivals = gen_arrivals(N, rate_per_ms)

    print()
    print(SEP)
    print(f"步骤2：固定流量(≈{rate_per_ms*1e3:.0f} req/s 到达)下扫描 max_batch")
    print(SEP)
    print(f"{'max_batch':>10} | {'平均延迟(ms)':>12} | {'P99延迟(ms)':>11} | {'吞吐(req/s)':>12}")
    print("-" * 56)
    results = []
    for mb in [1, 4, 16, 64]:
        lat, makespan = simulate(arrivals, forward_ms, max_batch=mb, max_wait_ms=2.0)
        avg, p99, thr = stats(lat, makespan, N)
        results.append((mb, avg, p99, thr))
        print(f"{mb:>10} | {avg:>12.3f} | {p99:>11.3f} | {thr:>12.1f}")

    print()
    print(SEP)
    print("解读")
    print(SEP)
    base = results[0]
    best = max(results, key=lambda r: r[3])
    print(f"  - max_batch=1（不攒批）：吞吐 {base[3]:.0f} req/s，"
          f"队列积压导致平均延迟高达 {base[1]:.1f} ms。")
    print(f"  - max_batch={best[0]}：吞吐升到 {best[3]:.0f} req/s "
          f"(≈{best[3]/base[3]:.1f}x)，队列被快速消化，平均延迟降到 {best[1]:.1f} ms。")
    print("  - 过载(队列积压)时：大 batch 同时改善吞吐与延迟(更快清空队列)。")
    print("  - 轻载(队列常空)时：大 batch 会因等待凑批而抬高延迟——这才是经典权衡。")
    print("  - 工程折中：延迟敏感→小 batch+短 max_wait；吞吐优先→大 batch；")
    print("            两者兼顾→continuous batching(见 Day57)。")

    print()
    print(SEP)
    if TORCH_AVAILABLE and DEVICE == "cuda":
        print("环境：GPU 就绪，前向耗时为真实 GPU 测量，调度在事件时间线上推演。")
    elif TORCH_AVAILABLE:
        print("环境：PyTorch(CPU) 运行。逻辑与 GPU 一致，仅设备不同；")
        print("      绝对数值不代表 GPU，但延迟/吞吐的相对趋势成立。")
    else:
        print("环境：NumPy 演示。前向为真实 CPU 运算，调度逻辑完全一致。")
    print(SEP)


if __name__ == "__main__":
    main()
