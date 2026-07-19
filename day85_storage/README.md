# Day 85: 存储 — 并行文件系统 / 小文件 / 数据加载瓶颈

> 📦 GPU 再快，数据喂不上也白搭。本天讲训练的「最后一公里」：数据从磁盘到 GPU 的流水线，以及三大痛点——并行文件系统、小文件问题、数据加载瓶颈。这是 JD 里长期空白、却极影响训练吞吐的一环。

## 学习目标

1. **解释** 并行文件系统（Lustre / GPFS）的条带化与元数据瓶颈
2. **量化** 小文件问题：海量小文件 vs 打包成大文件（tar / WebDataset）
3. **实现** 数据加载流水线：朴素串行 vs prefetch 多 worker，对比吞吐
4. **掌握** DataLoader 的 num_workers / prefetch_factor / pin_memory 作用

---

## 前置依赖

- **Day 10 CUDA 流**：pin_memory + non_blocking 的 H2D 异步拷贝基础
- **Day 25 多 GPU / Day 26 训练**：数据加载是训练吞吐的关键路径

---

## 核心概念

### 并行文件系统

集群里成百上千 GPU 同时读数据，单 NFS 扛不住。Lustre/GPFS 把一个文件**条带化**打散到多个存储节点（OST），带宽随节点近线性扩展。

```
客户端 ── 元数据服务(MDS) 查「文件在哪」
   ├──▶ OST1 ┐
   ├──▶ OST2 ├─ 大文件切条带并行读
   └──▶ OST3 ┘
```

两大瓶颈：**元数据瓶颈**（海量小文件把 MDS 打爆）、**带宽 vs IOPS**（擅长大文件高带宽，怕小文件高 IOPS）。

### 小文件问题

「一图一文件」每次都付打开/元数据/寻道开销。**打包成大文件顺序流式读**（WebDataset `.tar` 分片 / TFRecord）只付一次开销、顺序读打满带宽。脚本模拟显示可快数十倍。

### 数据加载流水线

朴素串行：加载完才计算、计算完才加载（GPU 空等）。**prefetch 多 worker**：后台并行加载 + 提前预取，让加载被计算掩盖。对应 PyTorch DataLoader：

| 参数 | 作用 | 经验值 |
|------|------|--------|
| num_workers | 并行加载子进程数 | 4~8/GPU 起调 |
| prefetch_factor | 每 worker 预取批数 | 2~4 |
| pin_memory | 锁页内存，加速 H2D | 训练 True |
| persistent_workers | 跨 epoch 不重建 | 大数据 True |

诊断：GPU 利用率忽高忽低/偏低 → 多半数据加载没跟上，先加 num_workers、用打包格式、上 pin_memory。

---

## 文件结构

```
day85_storage/
├── python/storage.py   # 并行FS原理 + 小文件对比 + 加载流水线模拟
├── build.sh            # Linux/WSL2 运行脚本
├── build.bat           # Windows 运行脚本
└── README.md           # 本文件
```

> 脚本无需 GPU、无需真实数据集，用 `time.sleep` 模拟 IO/计算耗时。

---

## 编译与运行

### 单机（无需 GPU）

```bash
cd day85_storage
python python/storage.py
```

### 真实数据流水线（PyTorch）

```python
loader = torch.utils.data.DataLoader(
    dataset, batch_size=256, shuffle=True,
    num_workers=8, prefetch_factor=4,
    pin_memory=True, persistent_workers=True)
for x, y in loader:
    x = x.cuda(non_blocking=True)   # 配合 pin_memory 异步拷贝
```

### Windows

```powershell
cd day85_storage
.\build.bat
```

---

## 知识点总结

1. **并行 FS**：条带化提带宽，但元数据（小文件）是瓶颈。
2. **打包格式**：WebDataset/TFRecord 顺序流式读，避开小文件 IOPS 陷阱。
3. **流水线**：num_workers + prefetch 让加载与计算重叠，掩盖 IO。
4. **pin_memory**：锁页内存 + non_blocking 加速 H2D。
5. **诊断**：GPU 利用率低先查数据加载，而非盲目改模型。

---

## 验收标准

- 程序打印并行 FS 拓扑与两大瓶颈，你能解释小文件为何拖慢 MDS。
- 小文件 vs 打包对比显示打包明显更快（加速比数十倍）。
- prefetch 流水线吞吐明显高于朴素串行，且结果一致 ✓。

---

## 动手挑战

1. **改 worker 数**：把 `prefetch_pipeline` 的 `num_workers` 改 1/2/8，观察吞吐拐点。
2. **IO/计算比**：把 `io_us` 调到远大于 `compute_us`（IO 瓶颈），看 prefetch 收益变化。
3. **真打包**：用 Python `tarfile` 把一堆小 `.npy` 打包成一个 `.tar`，对比读取耗时。
4. **上 WebDataset**：`pip install webdataset`，把数据写成 `.tar` 分片并用它训练。

---

## 常见问题

- **Q：num_workers 越多越好吗？**
  A：不是。过多会争抢 CPU/内存、增加进程开销，甚至更慢。从 4~8/GPU 起，看 GPU 利用率调优。

- **Q：小数据集也要打包吗？**
  A：能放进内存的小数据集直接缓存即可。打包主要解决**海量小文件在并行 FS 上的元数据/IOPS 瓶颈**。

---

## 延伸阅读

- [WebDataset 项目](https://github.com/webdataset/webdataset)
- [Lustre 文件系统架构](https://www.lustre.org/)
- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
