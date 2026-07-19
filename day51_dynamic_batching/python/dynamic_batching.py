#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 51: Continuous / Dynamic Batching（连续/动态批处理）

推理服务里，请求随机到达、生成长度各不相同。如何把它们组成 batch 一起算，
是吞吐与延迟的关键。本天用一个离散事件模拟对比两种调度策略：

  1. 静态批处理（Static Batching）：
     - 凑齐一批请求一起开始，一起解码；
     - 整批必须等到「最长的那个」生成完才能返回、才能开始下一批；
     - 短请求被长请求拖累（对头阻塞），GPU 在很多步上算的是已完成请求的 padding。

  2. 连续批处理（Continuous Batching，vLLM/ORCA 的核心）：
     - 以「解码步」为调度粒度；
     - 某请求完成即刻从 batch 中移除并返回（空出的槽位立即让新请求补入）；
     - GPU 每一步都尽量装满有效请求，几乎无 padding 浪费。

模拟中我们统计：每个请求的完成时间(延迟)、整体吞吐(请求/步)、GPU 有效利用率。

本脚本核心是调度逻辑（纯 Python 事件模拟，不依赖 GPU）。torch 仅用于探测环境
并说明真实系统里每一步是一次 batched decode 前向；无 torch 也完全可跑。
"""

import random
from collections import deque

SEP = "=" * 64

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"


# ============================================================
#  请求模型
# ============================================================
class Request:
    """一个推理请求：到达时间 arrival、需要生成 gen_len 个 token。"""

    __slots__ = ("rid", "arrival", "gen_len", "start_step", "finish_step")

    def __init__(self, rid, arrival, gen_len):
        self.rid = rid
        self.arrival = arrival        # 到达的「步」时间
        self.gen_len = gen_len        # 需要解码多少步
        self.start_step = None        # 实际开始解码的步
        self.finish_step = None       # 完成的步

    def latency(self):
        # 端到端延迟 = 完成时间 - 到达时间
        return self.finish_step - self.arrival


def make_requests(n, seed):
    """生成一批随机到达、变长的请求。"""
    rng = random.Random(seed)
    reqs = []
    t = 0
    for i in range(n):
        t += rng.randint(0, 2)                 # 到达间隔 0~2 步
        gen_len = rng.choice([2, 3, 5, 8, 12, 20])  # 生成长度差异很大
        reqs.append(Request(i, t, gen_len))
    return reqs


# ============================================================
#  策略 1：静态批处理
# ============================================================
def simulate_static(requests, batch_size):
    """静态批：按到达顺序凑满 batch_size 一起跑，整批等最长者完成再开下一批。

    返回 (完成的请求列表, 总步数, 有效计算格子数, 总计算格子数)
    """
    pending = deque(sorted(requests, key=lambda r: (r.arrival, r.rid)))
    step = 0
    total_slots = 0   # batch_size * 运行步数 的累计（GPU 花费的算力格子）
    useful_slots = 0  # 其中真正在算「未完成请求」的格子
    finished = []

    while pending:
        # 等到能凑够一批（或取尽剩余），batch 起始步 = max(当前step, 队首到达时间)
        batch = []
        while pending and len(batch) < batch_size:
            batch.append(pending.popleft())
        batch_arrival = max(r.arrival for r in batch)
        step = max(step, batch_arrival)  # 批不能早于最后到达者开始

        batch_max_len = max(r.gen_len for r in batch)
        for r in batch:
            r.start_step = step
            r.finish_step = step + r.gen_len  # 该请求自身完成的步
        # 整批必须跑满 batch_max_len 步（短请求也得陪跑到最长者结束）
        for s in range(batch_max_len):
            # 这一步 batch 占用 batch_size 个格子（静态批槽位固定，空槽也占）
            total_slots += batch_size
            # 有效：该请求还没生成完（s < gen_len）才算有效计算
            useful_slots += sum(1 for r in batch if s < r.gen_len)
        step += batch_max_len
        finished.extend(batch)

    return finished, step, useful_slots, total_slots


# ============================================================
#  策略 2：连续批处理
# ============================================================
def simulate_continuous(requests, max_batch):
    """连续批：以「步」为粒度调度。每步：
       - 把已到达且有空位的请求加入运行集；
       - 运行集里每个请求解码 1 步；
       - 完成的请求立即移除（空出槽位下一步即可补新请求）。
    """
    arrivals = deque(sorted(requests, key=lambda r: (r.arrival, r.rid)))
    running = []       # 当前正在解码的请求
    progress = {}      # rid -> 已解码步数
    step = 0
    total_slots = 0
    useful_slots = 0
    finished = []
    n = len(requests)

    while len(finished) < n:
        # 1) 有空位就把已到达的请求补进来
        while arrivals and arrivals[0].arrival <= step and len(running) < max_batch:
            r = arrivals.popleft()
            r.start_step = step
            running.append(r)
            progress[r.rid] = 0

        # 若当前没有可运行请求（都还没到达），时间快进到下一个到达
        if not running:
            if arrivals:
                step = arrivals[0].arrival
                continue
            else:
                break

        # 2) 运行集里每个请求解码 1 步
        total_slots += max_batch                # GPU 本步有 max_batch 个槽位
        useful_slots += len(running)            # 其中 len(running) 个在做有效计算
        still_running = []
        for r in running:
            progress[r.rid] += 1
            if progress[r.rid] >= r.gen_len:
                r.finish_step = step + 1        # 本步结束即完成
                finished.append(r)
            else:
                still_running.append(r)
        running = still_running
        step += 1

    return finished, step, useful_slots, total_slots


# ============================================================
#  讲解
# ============================================================
def explain_batching():
    print(SEP)
    print("静态批 vs 连续批（Continuous Batching）")
    print(SEP)
    print("""
  时间轴（每列一个解码步，█=有效计算，░=padding/空闲槽位）：

  静态批（batch=4，整批等最长者）：
    reqA(len2): ██░░░░░░   ← 生成完仍占槽陪跑
    reqB(len3): ███░░░░░
    reqC(len8): ████████
    reqD(len5): █████░░░
              └ 整批 8 步后才统一结束、才开下一批 → 大量 ░ 浪费、短请求延迟高

  连续批（完成即出、空位即入）：
    reqA(len2): ██                 ← 完成立即返回并让位
    reqB(len3): ███
    reqC(len8): ████████
    reqD(len5): █████
    新到的 reqE 在 A 完成后立刻补入 A 的槽位 → 每步尽量装满，几乎无 ░

  结论：连续批以「步」为调度粒度，消除对头阻塞与 padding，
        显著提升吞吐、降低平均延迟（vLLM/ORCA 的关键机制）。
""")


# ============================================================
#  主流程
# ============================================================
def report(name, finished, total_steps, useful, total):
    avg_lat = sum(r.latency() for r in finished) / len(finished)
    max_lat = max(r.latency() for r in finished)
    util = useful / total * 100 if total else 0.0
    throughput = len(finished) / total_steps if total_steps else 0.0
    print(f"  [{name}]")
    print(f"     总步数(耗时)   = {total_steps}")
    print(f"     平均延迟       = {avg_lat:.2f} 步 (最大 {max_lat})")
    print(f"     吞吐           = {throughput:.3f} 请求/步")
    print(f"     GPU 有效利用率 = {util:.1f}%  (有效格子 {useful}/{total})")
    return avg_lat, throughput, util


def main():
    N, batch = 12, 4
    print(f"[环境] device = {DEVICE}"
          + ("（真实系统中每步是一次 batched decode 前向）" if TORCH_AVAILABLE else "（纯事件模拟）"))
    print(f"[配置] 请求数 N={N}, 最大并发 batch={batch}")

    explain_batching()

    print(SEP)
    print("模拟对比（相同请求集，随机到达 + 变长）")
    print(SEP)

    reqs_static = make_requests(N, seed=42)
    reqs_cont = make_requests(N, seed=42)  # 同一组请求，保证公平对比
    lens = [r.gen_len for r in reqs_static]
    arr = [r.arrival for r in reqs_static]
    print(f"  生成长度: {lens}")
    print(f"  到达步:   {arr}")
    print()

    fs, ss, us, ts = simulate_static(reqs_static, batch)
    lat_s, thr_s, util_s = report("静态批处理", fs, ss, us, ts)
    print()
    fc, sc, uc, tc = simulate_continuous(reqs_cont, batch)
    lat_c, thr_c, util_c = report("连续批处理", fc, sc, uc, tc)

    print()
    print(SEP)
    print("对比结论")
    print(SEP)
    print(f"  平均延迟:  静态 {lat_s:.2f} → 连续 {lat_c:.2f} 步  "
          f"(降低 {(lat_s - lat_c) / lat_s * 100:.0f}%)")
    print(f"  吞吐:      静态 {thr_s:.3f} → 连续 {thr_c:.3f} 请求/步  "
          f"(提升 {(thr_c - thr_s) / thr_s * 100:.0f}%)")
    print(f"  利用率:    静态 {util_s:.1f}% → 连续 {util_c:.1f}%")
    print("  [结论] 连续批处理靠「完成即出、空位即入」减少 padding 与对头阻塞，"
          "吞吐更高、延迟更低 ✓")

    print()
    print(SEP)
    if TORCH_AVAILABLE and DEVICE == "cuda":
        print("环境：PyTorch + CUDA GPU 就绪（本天为调度模拟，核心逻辑与 GPU 无关）。")
    elif TORCH_AVAILABLE:
        print("环境：有 PyTorch 但无 CUDA GPU。调度模拟逻辑完全一致。")
    else:
        print("环境：未安装 PyTorch。调度模拟为纯 Python，结果一致。")
    print(SEP)


if __name__ == "__main__":
    main()
