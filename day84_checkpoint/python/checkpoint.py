#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 84: Checkpoint 机制与性能优化

训练动辄几天几周，机器随时可能挂。Checkpoint（检查点）就是「定期存档」，
让训练能从最近一次存档恢复。但存 checkpoint 会**阻塞训练**，本天讲怎么优化。

核心目标（单机可跑最小实现 + 原理）：
  1. 同步 checkpoint：保存/加载并验证恢复正确性；
  2. 异步 checkpoint：后台线程写盘，量化「训练阻塞时间」的减少；
  3. 分片 checkpoint：大模型按 rank 切分，各存各的分片；
  4. 增量 checkpoint：只存相对上次「变化的」张量，省 IO。

有 torch 用 torch.save/load；无 torch 用 numpy + pickle。均无需 GPU。
文件写入系统临时目录，运行结束自动清理。
"""

import os
import io
import time
import pickle
import shutil
import tempfile
import threading

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
#  统一的「保存/加载」后端：有 torch 用 torch，否则 numpy+pickle
# ============================================================
def save_state(state, path):
    """把一个 state dict（张量集合）保存到 path。"""
    if TORCH_AVAILABLE:
        torch.save(state, path)
    else:
        with open(path, "wb") as f:
            pickle.dump(state, f)


def load_state(path):
    if TORCH_AVAILABLE:
        return torch.load(path, weights_only=False)
    with open(path, "rb") as f:
        return pickle.load(f)


def make_model_state(num_layers=6, dim=512, seed=0):
    """构造一个「模型」的参数字典，模拟一个中等规模网络。"""
    rng = np.random.default_rng(seed)
    state = {}
    for i in range(num_layers):
        w = rng.standard_normal((dim, dim)).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)
        if TORCH_AVAILABLE:
            state[f"layer{i}.weight"] = torch.from_numpy(w)
            state[f"layer{i}.bias"] = torch.from_numpy(b)
        else:
            state[f"layer{i}.weight"] = w
            state[f"layer{i}.bias"] = b
    return state


def state_equal(a, b):
    """判断两个 state 是否逐张量相等。"""
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if TORCH_AVAILABLE and hasattr(va, "numpy"):
            va, vb = va.numpy(), vb.numpy()
        if not np.array_equal(np.asarray(va), np.asarray(vb)):
            return False
    return True


def _bytes_of(state):
    """估算 state 的字节数（用于展示 checkpoint 大小）。"""
    total = 0
    for v in state.values():
        arr = v.numpy() if (TORCH_AVAILABLE and hasattr(v, "numpy")) else np.asarray(v)
        total += arr.nbytes
    return total


# ============================================================
#  1. 同步 checkpoint（基线）
# ============================================================
def demo_sync_checkpoint(workdir):
    print(SEP)
    print("1) 同步 checkpoint：保存 → 加载 → 验证恢复")
    print(SEP)
    state = make_model_state()
    path = os.path.join(workdir, "ckpt_sync.pt")

    t0 = time.perf_counter()
    save_state(state, path)
    save_ms = (time.perf_counter() - t0) * 1000

    loaded = load_state(path)
    ok = state_equal(state, loaded)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  checkpoint 大小 = {size_mb:.1f} MB，同步保存耗时 = {save_ms:.1f} ms")
    print(f"  恢复正确性：{'✓ 一致' if ok else '✗ 不一致'}")
    print("  注意：同步保存期间，训练主线程被完全阻塞。")
    return ok


# ============================================================
#  2. 异步 checkpoint（后台线程写盘）
# ============================================================
def demo_async_checkpoint(workdir):
    print(SEP)
    print("2) 异步 checkpoint：后台线程写盘，减少训练阻塞")
    print(SEP)
    state = make_model_state(seed=1)

    # --- 关键点：先在主线程把张量「快照」到内存 buffer（快），
    #     再交给后台线程慢慢落盘（慢）。主线程只被 snapshot 阻塞。 ---
    def serialize_to_buffer(st):
        buf = io.BytesIO()
        if TORCH_AVAILABLE:
            torch.save(st, buf)
        else:
            pickle.dump(st, buf)
        return buf

    # 模拟同步：序列化 + 写盘都在主线程
    path_sync = os.path.join(workdir, "ckpt_sync2.pt")
    t0 = time.perf_counter()
    save_state(state, path_sync)
    # 人为加一点「慢盘」模拟，让对比更直观
    time.sleep(0.05)
    sync_block = (time.perf_counter() - t0) * 1000

    # 异步：主线程只做内存快照，落盘丢给后台线程
    path_async = os.path.join(workdir, "ckpt_async.pt")
    t0 = time.perf_counter()
    buf = serialize_to_buffer(state)  # 主线程阻塞仅此步
    main_block = (time.perf_counter() - t0) * 1000

    done = threading.Event()

    def writer():
        with open(path_async, "wb") as f:
            f.write(buf.getvalue())
        time.sleep(0.05)  # 模拟慢盘，与上面同步版对齐
        done.set()

    th = threading.Thread(target=writer, daemon=True)
    th.start()
    # 主线程此刻可以「继续训练」——这里模拟继续算一步
    _ = np.random.default_rng(0).standard_normal((256, 256)) @ \
        np.random.default_rng(1).standard_normal((256, 256))
    th.join()  # 演示末尾等后台写完，真实训练中无需在此等待

    ok = state_equal(state, load_state(path_async))
    print(f"  同步版：主线程阻塞 ≈ {sync_block:.1f} ms（序列化 + 写盘都在前台）")
    print(f"  异步版：主线程阻塞 ≈ {main_block:.1f} ms（仅内存快照，写盘在后台）")
    speedup = sync_block / max(main_block, 1e-6)
    print(f"  阻塞时间缩短约 {speedup:.1f}x，恢复正确性：{'✓' if ok else '✗'}")
    print("  真实框架：torch.distributed.checkpoint / 各类 async-ckpt 用此思路。")
    return ok


# ============================================================
#  3. 分片 checkpoint（大模型按 rank 切分）
# ============================================================
def demo_sharded_checkpoint(workdir, world_size=4):
    print(SEP)
    print(f"3) 分片 checkpoint：模型按 {world_size} 个 rank 切分，各存各的分片")
    print(SEP)
    full = make_model_state(num_layers=8, dim=256, seed=2)
    keys = list(full.keys())

    # 简单按参数名 round-robin 切分到各 rank（真实中按张量切/按层切）
    shards = [dict() for _ in range(world_size)]
    for idx, k in enumerate(keys):
        shards[idx % world_size][k] = full[k]

    # 每个 rank 独立写自己的分片文件
    shard_paths = []
    for r in range(world_size):
        p = os.path.join(workdir, f"ckpt_shard_rank{r}.pt")
        save_state(shards[r], p)
        shard_paths.append(p)
        mb = os.path.getsize(p) / 1024 / 1024
        print(f"  rank{r} 分片：{len(shards[r]):2d} 个张量，{mb:5.1f} MB -> {os.path.basename(p)}")

    # 恢复：把所有分片合并回完整 state
    merged = {}
    for p in shard_paths:
        merged.update(load_state(p))
    ok = state_equal(full, merged)
    print(f"  合并 {world_size} 个分片后与原模型：{'✓ 一致' if ok else '✗ 不一致'}")
    print("  优点：各 rank 并行写，单文件更小；ZeRO/FSDP 就是按分片存优化器状态。")
    return ok


# ============================================================
#  4. 增量 checkpoint（只存变化的张量）
# ============================================================
def demo_incremental_checkpoint(workdir):
    print(SEP)
    print("4) 增量 checkpoint：只保存相对上次「变化」的张量")
    print(SEP)
    base = make_model_state(num_layers=6, dim=256, seed=3)

    # 存一次全量作为 base
    base_path = os.path.join(workdir, "ckpt_full_base.pt")
    save_state(base, base_path)
    base_mb = os.path.getsize(base_path) / 1024 / 1024

    # 模拟又训练了一步：只有部分层参数发生变化
    new_state = {k: (v.clone() if TORCH_AVAILABLE and hasattr(v, "clone")
                     else v.copy()) for k, v in base.items()}
    changed_keys = ["layer0.weight", "layer2.bias", "layer5.weight"]
    for k in changed_keys:
        if TORCH_AVAILABLE and hasattr(new_state[k], "add_"):
            new_state[k] = new_state[k] + 0.01
        else:
            new_state[k] = new_state[k] + 0.01

    # 增量：只保存变化的张量（用 array_equal 检测差异）
    delta = {}
    for k in new_state:
        v0, v1 = base[k], new_state[k]
        a0 = v0.numpy() if (TORCH_AVAILABLE and hasattr(v0, "numpy")) else np.asarray(v0)
        a1 = v1.numpy() if (TORCH_AVAILABLE and hasattr(v1, "numpy")) else np.asarray(v1)
        if not np.array_equal(a0, a1):
            delta[k] = v1
    delta_path = os.path.join(workdir, "ckpt_delta.pt")
    save_state(delta, delta_path)
    delta_mb = os.path.getsize(delta_path) / 1024 / 1024

    # 恢复：base + delta 覆盖
    restored = dict(load_state(base_path))
    restored.update(load_state(delta_path))
    ok = state_equal(new_state, restored)

    print(f"  全量 base：{len(base)} 个张量，{base_mb:.1f} MB")
    print(f"  本次变化：{len(delta)} 个张量 {changed_keys}")
    print(f"  增量文件：{delta_mb:.2f} MB（仅为全量的 {delta_mb / base_mb * 100:.0f}%）")
    print(f"  base + delta 恢复：{'✓ 一致' if ok else '✗ 不一致'}")
    print("  适用：优化器状态大、每步只动一小部分（如 LoRA/微调）时省 IO 明显。")
    return ok


def explain_checkpoint():
    print(SEP)
    print("Checkpoint 为什么重要 & 四种优化思路")
    print(SEP)
    print("""
  为什么：训练几天几周，硬件/抢占随时中断，必须能从最近存档恢复。
          恢复内容通常包括：模型权重 + 优化器状态 + 学习率/step + RNG 状态。

  四种思路（本脚本逐一演示）：
    1. 同步   ：最简单，但保存时训练完全停摆（GPU 空转）。
    2. 异步   ：主线程只做内存快照，落盘丢后台线程 —— 大幅减少阻塞。
    3. 分片   ：大模型/优化器状态按 rank 切分并行写，单文件更小（ZeRO/FSDP）。
    4. 增量   ：只存相对上次变化的张量，省 IO（微调/LoRA 场景收益大）。

  工程要点：写临时文件再 rename（原子替换），避免写到一半崩溃损坏 ckpt。
""")


# ============================================================
#  主流程
# ============================================================
def main():
    explain_checkpoint()
    workdir = tempfile.mkdtemp(prefix="day84_ckpt_")
    print(f"  临时工作目录：{workdir}")
    print(f"  存取后端：{'torch.save/load' if TORCH_AVAILABLE else 'pickle + numpy'}")
    print()

    results = []
    try:
        results.append(demo_sync_checkpoint(workdir)); print()
        results.append(demo_async_checkpoint(workdir)); print()
        results.append(demo_sharded_checkpoint(workdir)); print()
        results.append(demo_incremental_checkpoint(workdir))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    print(SEP)
    all_ok = all(results)
    print("[结果] 四种 checkpoint 恢复" + ("全部正确 ✓" if all_ok else "存在错误 ✗"))
    print(f"  临时文件已清理（{os.path.basename(workdir)}）。")
    if TORCH_AVAILABLE:
        print("环境：使用 torch.save/load。分布式场景见 torch.distributed.checkpoint。")
    else:
        print("环境：未装 torch，用 pickle+numpy 演示，思路与 torch 一致。")
    print(SEP)


if __name__ == "__main__":
    main()
