# Day 86: GPUDirect Storage (GDS) 原理 — IO 路径分析

> 🚚 **第十三阶段（通信·存储·工程化）存储专段收官**。大模型训练的数据、权重、checkpoint 动辄几十上百 GB，如何把它们**高效地从存储搬进 GPU 显存**，是 AI Infra 的关键工程问题。本天讲清 GPUDirect Storage 如何绕过 CPU bounce buffer 直达 GPU。

## 学习目标

1. **画出并解释**传统 IO 路径（存储→CPU 内存→GPU）与 GPUDirect Storage 路径（存储→GPU 直达）的区别
2. **量化**两条路径的数据拷贝次数、CPU/内存带宽占用与总线往返次数
3. **用带宽模型模拟**不同数据量下两条路径的耗时、加速比与有效带宽
4. **理解** GDS 的落地条件（Linux + 支持的 NVMe/并行文件系统 + cuFile 驱动）与适用场景

---

## 前置依赖

- **Day 3 内存模型 / Day 10 流与异步**：理解 `cudaMemcpy` H2D 拷贝与 DMA 的概念
- **Day 82 NVLink/RDMA/GPUDirect 原理**：本天是 GPUDirect 家族在「存储」方向的延伸
- **Day 84/85 Checkpoint 与存储瓶颈**：GDS 正是解决大 checkpoint / 数据加载瓶颈的手段之一

---

## 核心概念

### 传统 IO 路径：两跳，多一次拷贝

```
存储 --DMA--> CPU 内存(bounce buffer) --DMA--> GPU 显存
  拷贝1(存储->CPU)          拷贝2(CPU->GPU)
```

数据必须先落到 CPU 内存的 bounce buffer，再由 CPU 发起第二次 DMA 送进 GPU。代价：
- **两次数据面拷贝**，两段 PCIe 往返；
- 占用宝贵的 **CPU 内存带宽**；
- 每次搬运都要 CPU 参与，小 IO 延迟高。

### GPUDirect Storage：一跳，直达 GPU

```
存储 --DMA(直达)--> GPU 显存      (CPU 只下发命令, 不碰数据)
```

存储控制器通过 DMA **直接把数据写入 GPU 显存**，CPU 只负责控制面（下发命令）。收益：
- **一次数据面拷贝**，一段 PCIe；
- 几乎不占用 CPU 内存带宽；
- 降低延迟、提升端到端吞吐，在大批量数据加载 / checkpoint 读写时最明显。

### 对比表

| 维度 | 传统路径 | GPUDirect Storage |
|------|----------|-------------------|
| 数据面拷贝次数 | 2 次（存储→CPU→GPU） | 1 次（存储→GPU 直达） |
| CPU 内存带宽占用 | 占用（bounce buffer） | 几乎不占用 |
| CPU 参与 | 参与数据搬运 | 仅下发命令（控制面） |
| PCIe 往返 | 2 段 | 1 段 |
| 落地条件 | 无（通用） | Linux + 支持的 NVMe/文件系统 + cuFile 驱动 |

---

## 文件结构

```
day86_gpudirect_storage/
├── python/gpudirect_storage.py   # IO 路径图 + 拷贝对比 + 带宽模型模拟
├── build.sh                      # Linux/WSL2 运行脚本
├── build.bat                     # Windows 运行脚本
└── README.md                     # 本文件
```

> 本天以**原理为主**：真实 GDS 需要 Linux + 特定硬件/驱动，脚本在任何环境都能用带宽模型稳定演示，有 GPU 时额外实测「磁盘→CPU→GPU」传统路径作为参考。

---

## 编译与运行

### Linux / WSL2（推荐，可选真实 GDS）

```bash
cd day86_gpudirect_storage
# 可选：pip install kvikio-cu12   # GDS 的 Python 入口（需 NVMe + cuFile）
python python/gpudirect_storage.py
```

### Windows

```powershell
cd day86_gpudirect_storage
.\build.bat
```

> 有 CUDA GPU 时会额外实测传统两跳路径耗时；无 GPU 时用带宽模型演示等价原理（不伪造结果）。

---

## 知识点总结

1. **bounce buffer**：传统路径里 CPU 内存中转数据的缓冲区，是「多一次拷贝」的根源。
2. **数据面 vs 控制面**：GDS 让 CPU 只管控制面（下发命令），数据面完全绕过 CPU。
3. **拷贝次数决定上限**：传统 2 次拷贝 ≈ 2 倍耗时，GDS 1 次直达接近单段 PCIe 带宽上限。
4. **收益随数据量放大**：大权重 / 大 checkpoint 读写时，省掉的那次拷贝收益最稳定。
5. **落地条件**：GDS 依赖 Linux + 支持的 NVMe/并行文件系统 + cuFile 驱动，Python 侧可用 `kvikio`。
6. **与 GPUDirect 家族的关系**：GPUDirect RDMA（网卡→GPU）、P2P（GPU→GPU）、Storage（存储→GPU）同源思想——绕过 CPU 直达显存。

---

## 验收标准

- 程序打印两条 IO 路径的 ASCII 图，你能逐段解释「多一次拷贝」发生在哪里。
- 你能说出传统路径与 GDS 在拷贝次数、CPU 内存带宽占用、PCIe 往返上的差异。
- 带宽模型模拟中，传统路径耗时约为 GDS 的 2 倍，你能解释为什么加速比稳定在≈2x。
- 你能说出 GDS 的三个落地条件与最典型的两个适用场景。

---

## 动手挑战

1. **改带宽参数**：把 `simulate_bandwidth` 里的 `pcie_bw` 改成 Gen5 的带宽，观察耗时如何变化。
2. **加一条 RDMA 路径**：仿照带宽模型，加入「GPUDirect RDMA（远程存储→GPU）」路径并对比。
3. **实测对比**：有 GPU 时，把实测的「磁盘→CPU→GPU」耗时与带宽模型预测值对照，思考差异来源（页缓存、pinned memory 等）。

---

## 常见问题

- **Q：Windows 能用 GPUDirect Storage 吗？**
  A：GDS 官方面向 Linux + 特定 NVMe/文件系统。Windows 上以原理演示为主；本天脚本用带宽模型让你在任意环境理解收益。

- **Q：GDS 一定更快吗？**
  A：小 IO 或数据本就在 CPU 页缓存中时收益有限；GDS 在**大块、直连 NVMe/并行文件系统**的连续读写场景收益最大。

- **Q：pinned memory 和 GDS 是一回事吗？**
  A：不是。pinned memory 只是加速「CPU→GPU」那一跳；GDS 更进一步，直接省掉「存储→CPU」这一跳。

---

## 延伸阅读

- [NVIDIA GPUDirect Storage 文档](https://docs.nvidia.com/gpudirect-storage/)
- [kvikio（cuFile 的 Python 绑定）](https://github.com/rapidsai/kvikio)
- [NVIDIA Magnum IO 概览](https://developer.nvidia.com/magnum-io)
