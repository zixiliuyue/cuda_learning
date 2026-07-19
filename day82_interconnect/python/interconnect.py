#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Day 82: NVLink / RDMA / GPUDirect 原理 — 互联拓扑与带宽

Day81 讲了 AllReduce「算法」，本天讲支撑它的「硬件互联」：
数据到底走哪条物理链路、带宽/延迟差多少、为什么要 GPUDirect。

核心目标（原理为主 + 数据模拟）：
  1. 讲清 PCIe / NVLink / NVSwitch 的拓扑与带宽层级；
  2. 讲清 RDMA（RoCE / InfiniBand）的零拷贝与 kernel-bypass；
  3. 讲清 GPUDirect P2P / GPUDirect RDMA 如何绕开 CPU/系统内存；
  4. 用一组典型带宽/延迟数据，模拟同一份数据走不同链路的传输耗时对比；
  5. 若有 GPU：探测设备数量与 P2P 可达性（torch.cuda）。

本脚本无需 GPU 即可运行；数据为业界典型量级（示意用，非实测）。
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
#  典型互联参数（示意量级，用于对比，不代表某具体型号实测）
#    bw_GBps : 单向带宽 GB/s
#    lat_us  : 单次传输延迟 微秒
# ============================================================
LINKS = [
    #  名称                    带宽GB/s  延迟us   说明
    ("System DRAM (本地内存)",   50.0,   0.1,  "CPU 直连 DRAM，作参照"),
    ("PCIe 4.0 x16",            32.0,   1.5,  "GPU<->CPU 传统路径"),
    ("PCIe 5.0 x16",            64.0,   1.2,  "新一代主机总线"),
    ("NVLink 3.0 (per link)",   50.0,   0.5,  "A100 单链路，多链路叠加"),
    ("NVLink 4.0 聚合",        450.0,   0.4,  "H100 多链路聚合带宽"),
    ("NVSwitch 全互联",        900.0,   0.4,  "DGX 内任意 GPU 对等全带宽"),
    ("InfiniBand NDR (400G)",   50.0,   2.0,  "跨节点 RDMA 网络"),
    ("RoCE v2 (100G)",          12.5,   3.0,  "以太网上的 RDMA"),
]


def explain_topology():
    print(SEP)
    print("互联拓扑层级：从片内到跨节点（带宽递减 / 延迟递增）")
    print(SEP)
    print(r"""
  单机内（Scale-Up，NVLink/NVSwitch 为主）
  ------------------------------------------------------------
        GPU0 ═══NVLink═══ GPU1
         ║  ╲          ╱  ║
      NVSwitch ◀══全互联══▶ NVSwitch      (DGX: 任意 GPU 对全带宽)
         ║  ╱          ╲  ║
        GPU2 ═══NVLink═══ GPU3
         │                │
       PCIe               PCIe
         │                │
        ┌──────── CPU / DRAM ────────┐
         │
       网卡(NIC, IB/RoCE)
         │
  跨节点（Scale-Out，RDMA 网络为主）
  ------------------------------------------------------------
     Node A ─IB/RoCE交换机─ Node B ─...─ Node N

  关键层级（带宽从高到低）：
    NVSwitch 全互联 > NVLink 聚合 > PCIe 5.0 > IB/RoCE 跨节点
  经验法则：能不出节点就不出节点，能走 NVLink 就别走 PCIe。
""")


def explain_rdma_gpudirect():
    print(SEP)
    print("RDMA 与 GPUDirect：数据怎样少拷贝、少经手 CPU")
    print(SEP)
    print(r"""
  1) 传统路径（无 GPUDirect，多次拷贝 + CPU 经手）
     GPU显存 ─copy→ 系统内存(pinned) ─copy→ 网卡 ─网络→ 对端反向
     缺点：多次 memcpy、CPU 参与、延迟高、占用 CPU 与 PCIe。

  2) RDMA（Remote Direct Memory Access）
     网卡直接读写「对端内存」，kernel-bypass（绕过 OS 内核）、零拷贝。
     实现：InfiniBand（原生）或 RoCE（跑在以太网上）。
     好处：CPU 几乎不参与，延迟低、吞吐高。

  3) GPUDirect P2P（单机内）
     同一节点两块 GPU 经 NVLink/PCIe 直接互访显存，不经系统内存。
     对应 CUDA: cudaDeviceEnablePeerAccess / cudaMemcpyPeer。

  4) GPUDirect RDMA（跨节点）
     网卡直接 DMA 读写 GPU 显存，数据不再落地系统内存，
     GPU显存 ──网卡(RDMA)──网络──网卡(RDMA)──▶ 对端GPU显存
     这是 NCCL 跨节点高性能 AllReduce 的物理基础。

  一句话：GPUDirect = 让「网卡/对端 GPU」直接够得着「本地 GPU 显存」，
          省掉 GPU->CPU->NIC 的中转拷贝。
""")


def simulate_transfer(size_mb):
    """模拟一份 size_mb 数据走各链路的传输耗时。

    简单模型：耗时 = 延迟 + 数据量 / 带宽。
      time_us = lat_us + (size_bytes / (bw_GBps * 1e9)) * 1e6
    """
    size_bytes = size_mb * 1024 * 1024
    print(SEP)
    print(f"传输耗时模拟：搬运 {size_mb} MB 数据（模型：延迟 + 量/带宽）")
    print(SEP)
    print(f"  {'链路':26s}{'带宽GB/s':>9s}{'延迟us':>8s}{'总耗时us':>11s}{'相对':>8s}")
    print("  " + "-" * 62)
    times = []
    for name, bw, lat, _ in LINKS:
        t = lat + (size_bytes / (bw * 1e9)) * 1e6
        times.append((name, t))
    best = min(t for _, t in times)
    for (name, bw, lat, _), (_, t) in zip(LINKS, times):
        print(f"  {name:26s}{bw:9.1f}{lat:8.2f}{t:11.1f}{t / best:7.1f}x")
    print("\n  观察：大数据传输由『带宽』主导（NVSwitch 最快）；")
    print("        小数据由『延迟』主导，此时链路差距会缩小。")


def small_vs_large():
    """对比小数据（延迟主导）vs 大数据（带宽主导）下的链路差异。"""
    print(SEP)
    print("延迟主导 vs 带宽主导：不同数据大小下谁说了算")
    print(SEP)
    sizes_kb = [1, 64, 1024, 16 * 1024, 256 * 1024]  # 1KB ~ 256MB
    picks = ["PCIe 4.0 x16", "NVLink 4.0 聚合", "NVSwitch 全互联", "InfiniBand NDR (400G)"]
    link_map = {n: (bw, lat) for n, bw, lat, _ in LINKS}
    header = "  数据大小   " + "".join(f"{p.split()[0]:>14s}" for p in picks)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for kb in sizes_kb:
        b = kb * 1024
        row = f"  {_fmt_size(kb):>8s}  "
        for p in picks:
            bw, lat = link_map[p]
            t = lat + (b / (bw * 1e9)) * 1e6
            row += f"{t:12.2f}us"
        print(row)
    print("\n  1KB 级：各链路耗时接近延迟，差别小 → 小消息用 Tree（少步数）；")
    print("  256MB 级：带宽差异被放大 → 大消息用 Ring + NVLink/NVSwitch。")


def _fmt_size(kb):
    if kb < 1024:
        return f"{kb}KB"
    return f"{kb // 1024}MB"


def probe_gpu_p2p():
    """有 GPU 时探测设备与 P2P 可达性。"""
    print(SEP)
    print("GPU P2P 探测（GPUDirect P2P 能力）")
    print(SEP)
    if not TORCH_CUDA:
        print("  未检测到 CUDA GPU，跳过真实探测。")
        print("  真实环境可用：nvidia-smi topo -m   查看 GPU 间互联矩阵（NV#/PIX/SYS）。")
        return
    n = torch.cuda.device_count()
    print(f"  检测到 {n} 张 GPU：")
    for i in range(n):
        print(f"    GPU{i}: {torch.cuda.get_device_name(i)}")
    if n < 2:
        print("  只有 1 张 GPU，无法测 P2P。多卡时可用 torch.cuda.can_device_access_peer(i,j)。")
        return
    print("  P2P 可达矩阵（行 i 能否直接访问列 j 的显存）：")
    header = "        " + "".join(f"GPU{j:<4d}" for j in range(n))
    print(header)
    for i in range(n):
        row = f"    GPU{i} "
        for j in range(n):
            if i == j:
                row += "  -   "
            else:
                ok = torch.cuda.can_device_access_peer(i, j)
                row += "  Y   " if ok else "  N   "
        print(row)


def main():
    explain_topology()
    print()
    explain_rdma_gpudirect()
    print()
    simulate_transfer(size_mb=256)
    print()
    small_vs_large()
    print()
    probe_gpu_p2p()

    print()
    print(SEP)
    if MULTI_GPU:
        print("环境：检测到多 GPU，已探测 P2P 可达性；建议再跑 `nvidia-smi topo -m` 看物理拓扑。")
    elif TORCH_CUDA:
        print("环境：单 GPU。互联/带宽为原理讲解 + 数据模拟；多卡拓扑需真实集群。")
    else:
        print("环境：无 GPU。已用典型带宽/延迟数据模拟对比，理解互联层级即可。")
    print(SEP)


if __name__ == "__main__":
    main()
