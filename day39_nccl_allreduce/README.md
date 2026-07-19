# Day 39: NCCL 多 GPU AllReduce 通信实战

## 学习目标

1. 理解 **NCCL（NVIDIA Collective Communications Library）** 的作用：为多 GPU / 多节点提供高性能的集合通信原语（AllReduce、Broadcast、AllGather、ReduceScatter 等）。
2. 掌握 **AllReduce** 的语义：K 个 GPU 各持一份数据，规约（如求和）后，每个 GPU 都得到全体之和，结果完全一致。它是数据并行分布式训练中同步梯度的核心操作。
3. 掌握 NCCL 两种初始化方式：
   - 单进程多 GPU：`ncclCommInitAll`（一次性初始化所有通信器）
   - 多进程每进程一 GPU：`ncclGetUniqueId` + 广播 + `ncclCommInitRank`
4. 掌握 `ncclAllReduce` 的 API：发送/接收缓冲、元素数、数据类型、规约操作、通信器、CUDA 流。
5. 理解 NCCL 操作与 CUDA 流的绑定关系，以及 `ncclGroupStart/End`、`cudaStreamSynchronize`。

---

## NCCL 是什么

NCCL 是 NVIDIA 提供的多 GPU 通信库，专门优化 GPU 之间的集合通信。相比朴素的“拷回 CPU 求和再分发”，NCCL 能：

- 直接利用 **NVLink / PCIe / InfiniBand / RoCE** 等高速互联，绕过 CPU；
- 使用 **Ring / Tree** 等拓扑感知算法，把带宽和延迟压到接近硬件极限；
- 与 CUDA 流深度集成，通信可与计算重叠。

它是 PyTorch DDP、DeepSpeed、Megatron-LM 等分布式训练框架底层的通信引擎。

---

## AllReduce 的 Ring 与 Tree 算法简介

### Ring AllReduce（环形，带宽最优）

把 K 个 GPU 排成一个环，数据切成 K 份。分两个阶段：

```
阶段一 Reduce-Scatter（K-1 步）:
    每步每个 GPU 把一个分片发给下一个邻居，同时累加收到的分片。
    K-1 步后，每个 GPU 各自持有“某一分片”的全局规约结果。

阶段二 AllGather（K-1 步）:
    再沿环传递 K-1 步，把每个已规约的分片广播给所有 GPU。
```

- 每个 GPU 收发的数据量约为 `2 * (K-1)/K * N`，**与 GPU 数量几乎无关**，带宽利用率最优。
- 适合：单机多卡、大张量（如深度学习梯度）。

### Tree AllReduce（树形，延迟最优）

把 GPU 组织成二叉树：数据先自底向上 **Reduce** 到根，再自顶向下 **Broadcast** 回去。

- 通信步数为 `O(log K)`，**延迟随规模增长慢**。
- 适合：多节点、小张量、GPU 数量很大的场景。

NCCL 会根据张量大小、拓扑自动在 Ring / Tree（及 CollNet 等）之间选择。

---

## 文件结构

```
day39_nccl_allreduce/
├── src/main.cu            # CUDA + NCCL 实现（条件编译，无 NCCL 自动 CPU 模拟）
├── python/nccl_demo.py    # Python 版（torch.distributed 优先，否则 numpy 多进程）
├── CMakeLists.txt         # 版本感知架构 + 条件查找/链接 NCCL
├── build.bat              # Windows 编译脚本
├── build.sh               # Linux/macOS 编译脚本
└── README.md              # 本文件
```

### C++ 程序两条路径

| 条件 | 行为 |
|------|------|
| 找到 `<nccl.h>`（`HAS_NCCL=1`） | 用 `ncclCommInitAll` 初始化，执行真实 GPU AllReduce，验证结果并与 CPU 求和对比计时 |
| 找不到 NCCL（`HAS_NCCL=0`） | 退化为 CPU 模拟，打印 AllReduce 语义与 NCCL 安装方法 |

---

## 安装 NCCL

> NCCL 官方**仅支持 Linux**。Windows 用户请使用 WSL2，或直接运行本项目的 Python / CPU 模拟版本。

### Linux（APT）

```bash
sudo apt update
sudo apt install libnccl2 libnccl-dev
```

### Conda

```bash
conda install -c nvidia nccl
```

### 从源码编译

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j src.build CUDA_HOME=/usr/local/cuda
sudo make install
```

安装后，若头文件/库不在默认路径，可设置环境变量帮助 CMake 查找：

```bash
export NCCL_ROOT=/path/to/nccl
```

---

## 编译与运行

### Windows（PowerShell / CMD）

```powershell
cd day39_nccl_allreduce
.\build.bat
.\build\Release\nccl_demo.exe
```

### Linux / macOS

```bash
cd day39_nccl_allreduce
bash build.sh
./build/nccl_demo
```

### Python 版本

```bash
cd day39_nccl_allreduce/python
python nccl_demo.py
# 真实多 GPU 通信（需要多卡 + PyTorch）:
torchrun --nproc_per_node=4 nccl_demo.py
```

---

## 多 GPU 编程的注意事项

1. **每次操作前用 `cudaSetDevice(g)` 切换设备上下文**：分配显存、创建流、发起 NCCL 调用都必须在正确的当前设备上进行。
2. **NCCL 调用必须绑定 CUDA 流**：`ncclAllReduce(..., stream)` 是异步入队操作，需用 `cudaStreamSynchronize` 等待完成。
3. **单进程多 GPU 用 `ncclGroupStart/End` 包裹多次调用**：避免多个通信器互相等待导致死锁。
4. **多进程场景保证 `ncclUniqueId` 一致**：由 rank 0 生成后广播给所有进程，所有进程用同一 id 调用 `ncclCommInitRank`。
5. **发送与接收缓冲大小、数据类型必须在所有 rank 上一致**，否则行为未定义。
6. **NVLink / 拓扑影响性能**：`nvidia-smi topo -m` 可查看 GPU 互联拓扑；`NCCL_DEBUG=INFO` 可打印 NCCL 选择的算法与通道。
7. **及时释放资源**：`ncclCommDestroy`、`cudaStreamDestroy`、`cudaFree` 配对释放，避免泄漏。

---

## 知识点总结

| 序号 | 知识点 | 说明 |
|------|--------|------|
| 1 | NCCL 作用 | 多 GPU / 多节点高性能集合通信库，分布式训练底座 |
| 2 | AllReduce 语义 | 各 GPU 数据规约后，所有 GPU 得到相同的全体结果 |
| 3 | `ncclCommInitAll` | 单进程多 GPU 一次性初始化所有通信器 |
| 4 | `ncclGetUniqueId` + `ncclCommInitRank` | 多进程初始化，需广播 uniqueId |
| 5 | `ncclAllReduce` | (sendbuf, recvbuf, count, datatype, op, comm, stream) |
| 6 | Ring 算法 | 带宽最优，收发量与 GPU 数几乎无关，大张量首选 |
| 7 | Tree 算法 | 延迟 O(log K)，多节点小张量首选 |
| 8 | 流绑定 | NCCL 操作异步入队，需 `cudaStreamSynchronize` |
| 9 | `ncclGroupStart/End` | 合并多 GPU 调用，避免死锁 |
| 10 | 条件编译 | `__has_include(<nccl.h>)` + CMake `find_library`，无 NCCL 自动降级 |

---

## 常见问题

- **Windows 能跑真实 NCCL 吗？** 不能，NCCL 官方仅支持 Linux。Windows 上本程序会编译为 CPU 模拟版；如需真实通信请用 WSL2 或 Linux。
- **只有 1 个 GPU 也能运行吗？** 可以。单 GPU 的 AllReduce 结果就是它自身的数据，程序仍会完整演示 API 流程。
- **为什么 NCCL 比“拷回 CPU 求和”快很多？** 后者数据要经 PCIe 往返主机、串行累加；NCCL 直接在 GPU 间用 NVLink/PCIe 做拓扑感知的并行规约，避免了 CPU 瓶颈。
- **想看 NCCL 内部选了哪种算法？** 运行前设置 `NCCL_DEBUG=INFO`，日志会打印 Ring/Tree 通道与拓扑信息。

---

## 验收标准

完成本天学习后，运行 `nccl_demo`（或 Windows 的 `nccl_demo.exe`），你应当能够观察到并解释以下现象，才算真正掌握：

1. **程序打印编译路径与设备数**：开头输出 `[系统] 检测到 N 个 CUDA 设备` 与 `[编译] HAS_NCCL = 0/1`，你能据此判断当前是走真实 NCCL 路径还是 CPU 模拟路径，理解 `__has_include(<nccl.h>)` 的条件编译降级机制。
2. **AllReduce 结果一致且等于 1+2+…+K**：无论真实 NCCL 还是 CPU 模拟，每个 GPU 的结果都应相同，且每个元素等于 `1+2+...+num_gpus`（GPU g 初始化为 g+1）；真实路径会打印 `全部 GPU 结果一致，AllReduce 正确！`。
3. **性能对比表体现 NCCL 优势**（有 NCCL 且多卡时）：表格中 `NCCL AllReduce (GPU)` 的耗时应低于 `拷回 CPU + CPU 求和`，加速比大于 1，验证 NCCL 绕过 CPU、走 GPU 间互联的价值。
4. **能读懂多进程初始化流程**：程序打印的多进程说明（`ncclGetUniqueId` → 广播 → `ncclCommInitRank`）你能复述，并能说明它与单进程 `ncclCommInitAll` 的区别。

## 动手挑战

以下任务基于本天 `src/main.cu` 的真实实现，循序渐进地加深理解：

1. **改变规约操作与初值**：把 `ncclSum` 改为 `ncclMax` 或 `ncclProd`，并调整每个 GPU 的初始化值，重新推导「AllReduce 后每个元素应等于多少」，再运行验证你的预测（CPU 参考实现 `cpu_allreduce_reference` 也需同步改成对应语义）。
2. **观察 NCCL 选择的算法**：在 Linux 多卡环境下设置 `NCCL_DEBUG=INFO` 运行，从日志中找出 NCCL 实际选用的是 Ring 还是 Tree，并结合 `nvidia-smi topo -m` 的拓扑输出解释为什么。
3. **改造为多进程版本**：参考 `print_multiprocess_notes` 的流程，把单进程 `ncclCommInitAll` 版本改写为「每进程 1 GPU」的多进程程序：rank 0 用 `ncclGetUniqueId` 生成 id，经 MPI 或 socket 广播后各进程 `ncclCommInitRank`，再各自调用 `ncclAllReduce`。
