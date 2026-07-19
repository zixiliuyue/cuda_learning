# Day 82: NVLink / RDMA / GPUDirect 原理 — 互联拓扑与带宽

> 🔌 Day81 讲了 AllReduce 的**算法**，本天讲支撑它的**硬件互联**：数据到底走哪条物理链路、带宽/延迟差多少、为什么要 GPUDirect。分布式训练慢，一半是通信慢，通信慢往往是走错了链路。

## 学习目标

1. **描述** PCIe / NVLink / NVSwitch 的拓扑与带宽层级
2. **解释** RDMA（RoCE / InfiniBand）的零拷贝与 kernel-bypass
3. **区分** GPUDirect P2P（单机）与 GPUDirect RDMA（跨节点）
4. **判断** 何时受「延迟」主导、何时受「带宽」主导，并据此选通信算法
5. **探测**（有 GPU 时）设备数量与 P2P 可达性

---

## 前置依赖

- **Day 81 NCCL 算法**：Ring/Tree 的性能取决于底层链路带宽/延迟
- **Day 33 CUDA IPC**：P2P 显存共享的编程接口
- **Day 25 多 GPU**：多卡协作的物理基础

---

## 核心概念

### 互联层级（带宽从高到低）

```
NVSwitch 全互联 > NVLink 聚合 > PCIe 5.0 > InfiniBand/RoCE 跨节点
```

- **PCIe**：GPU↔CPU 传统总线，4.0 x16 ≈ 32 GB/s，5.0 ≈ 64 GB/s。
- **NVLink**：GPU↔GPU 专用高速链路，多链路聚合可达数百 GB/s。
- **NVSwitch**：交换芯片，让 DGX 内**任意 GPU 对**都享全带宽（非菊花链）。
- **InfiniBand / RoCE**：跨节点 RDMA 网络，Scale-Out 的骨干。

经验法则：**能不出节点就不出节点，能走 NVLink 就别走 PCIe。**

### RDMA 与 GPUDirect

| 路径 | 是否经 CPU/系统内存 | 说明 |
|------|---------------------|------|
| 传统 | 是（多次 memcpy） | GPU→pinned→NIC，CPU 经手，慢 |
| RDMA | 否（kernel-bypass） | 网卡直读写对端内存，零拷贝 |
| GPUDirect P2P | 否 | 单机内 GPU 经 NVLink/PCIe 直接互访显存 |
| GPUDirect RDMA | 否 | 网卡直接 DMA GPU 显存，跨节点不落地系统内存 |

一句话：**GPUDirect = 让网卡/对端 GPU 直接够得着本地 GPU 显存**，省掉中转拷贝。这是 NCCL 跨节点高性能 AllReduce 的物理基础。

### 延迟主导 vs 带宽主导

传输耗时模型：`时间 = 延迟 + 数据量 / 带宽`。

- **小消息（KB 级）**：延迟主导 → 选步数少的 **Tree**。
- **大消息（百 MB 级）**：带宽主导 → 选 **Ring** + NVLink/NVSwitch。

这正好解释了 Day81 里 NCCL 为何按消息大小自动切换算法。

---

## 文件结构

```
day82_interconnect/
├── python/interconnect.py   # 拓扑 ASCII 图 + 传输耗时模拟 + P2P 探测
├── build.sh                 # Linux/WSL2 运行脚本
├── build.bat                # Windows 运行脚本
└── README.md                # 本文件
```

> 脚本无需 GPU；带宽/延迟为业界典型量级（示意对比用，非实测）。有 GPU 时会探测 P2P 可达矩阵。

---

## 编译与运行

### 单机（无需 GPU）

```bash
cd day82_interconnect
python python/interconnect.py
```

### 真实拓扑排查（Linux + NVIDIA GPU）

```bash
nvidia-smi topo -m          # 打印 GPU 间互联矩阵：NV#(NVLink) / PIX / SYS
nvidia-smi nvlink -s        # 查看 NVLink 状态与带宽
export NCCL_DEBUG=INFO      # 让 NCCL 打印它实际走的链路
```

### Windows

```powershell
cd day82_interconnect
.\build.bat
```

---

## 知识点总结

1. **带宽层级**：NVSwitch > NVLink > PCIe > 跨节点网络，选链路 = 选性能。
2. **RDMA**：kernel-bypass + 零拷贝，CPU 不参与，跨节点通信首选。
3. **GPUDirect P2P/RDMA**：绕开 GPU→CPU→NIC 中转，NCCL 跨机的关键。
4. **传输模型**：延迟 + 量/带宽；小消息看延迟、大消息看带宽。
5. **拓扑感知**：NCCL 自动感知拓扑；`nvidia-smi topo -m` 是排障第一命令。

---

## 验收标准

- 程序打印拓扑 ASCII 图与 RDMA/GPUDirect 四种路径说明，你能复述其区别。
- 传输耗时表中 NVSwitch 对 256MB 最快、RoCE 最慢，你能解释为何。
- 小/大数据对比表中，你能指出「延迟主导→Tree、带宽主导→Ring」的分界。
- 有多 GPU 时能看到 P2P 可达矩阵。

---

## 动手挑战

1. **改数据大小**：把 `simulate_transfer` 的 `size_mb` 改成 1 / 1024，观察排名是否变化。
2. **加一条链路**：在 `LINKS` 里加入你机器的实测 PCIe 带宽（用 Day2/Day10 的 H2D 测速），对比模型值。
3. **画交叉点**：找出 PCIe 与 NVLink 在多大数据量时耗时相等（延迟差 / 带宽差）。
4. **真实探测**：多卡机器上跑 `nvidia-smi topo -m`，把输出与脚本的 P2P 矩阵对照。

---

## 常见问题

- **Q：RoCE 和 InfiniBand 哪个好？**
  A：IB 原生 RDMA、延迟低但贵；RoCE 复用以太网、成本低但需无损网络（PFC/ECN）调优。两者都支持 GPUDirect RDMA。

- **Q：没有 NVLink 的消费级显卡能多卡训练吗？**
  A：能，走 PCIe P2P（若主板/BIOS 支持）或经系统内存中转，只是带宽低。`nvidia-smi topo -m` 显示 `SYS` 即走系统总线。

---

## 延伸阅读

- [NVIDIA NVLink & NVSwitch 官方介绍](https://www.nvidia.com/en-us/data-center/nvlink/)
- [GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)
