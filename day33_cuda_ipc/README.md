# Day 33: CUDA IPC 进程间通信

## 学习目标

1. 理解 CUDA IPC（Inter-Process Communication）机制及其应用价值
2. 掌握核心 API：`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`
3. 理解 IPC 内存句柄的序列化与反序列化本质
4. 掌握 IPC 事件（`cudaIpcGetEventHandle`）实现跨进程 GPU 端同步
5. 了解 `cudaMemcpyPeer` 在多 GPU 之间的点对点拷贝
6. 理解 IPC 在分布式训练、多进程推理服务中的作用

## 核心概念

### 什么是 CUDA IPC

普通情况下，一个进程分配的 GPU 显存对另一个进程是不可见的。若两个进程需要交换
GPU 数据，通常要走 "设备 → 主机 → 设备" 的昂贵拷贝路径。

CUDA IPC 允许**同一台机器上**的不同进程直接共享同一块物理显存，实现真正的
"零拷贝" 数据交换，极大降低多进程协作的开销。

### IPC 共享显存三步流程

```
进程 A（生产者）                       进程 B（消费者）
------------------                     ------------------
cudaMalloc(&ptr)                       (等待接收句柄)
cudaIpcGetMemHandle(&h, ptr)  --句柄-->  cudaIpcOpenMemHandle(&ptr2, h)
(写入数据)                              (读写同一块显存)
                                       cudaIpcCloseMemHandle(ptr2)
```

- `cudaIpcGetMemHandle()`：从设备指针导出一个可共享的句柄 `cudaIpcMemHandle_t`
- `cudaIpcOpenMemHandle()`：另一进程用句柄映射出本进程可用的设备指针
- `cudaIpcCloseMemHandle()`：用完后关闭映射

### IPC 句柄的序列化本质

`cudaIpcMemHandle_t` 本质上是一段**定长字节序列**（通常 64 字节）。它可以被直接按
字节拷贝，通过任意 IPC 通道传递给另一个进程：

- Unix Domain Socket
- 命名管道（Named Pipe）
- 共享内存（如 POSIX `shm_open`）
- 文件

对方进程收到这段字节后，反序列化成 `cudaIpcMemHandle_t` 即可打开共享显存。

### IPC 事件同步

跨进程共享显存后，还需要**同步机制**避免竞态。CUDA 提供 IPC 事件：

- 创建时需带 `cudaEventInterprocess | cudaEventDisableTiming` 标志
- `cudaIpcGetEventHandle()` 导出事件句柄
- `cudaIpcOpenEventHandle()` 另一进程打开
- 配合 `cudaStreamWaitEvent` 实现跨进程 GPU 端同步（无需 CPU 介入）

## 文件说明

| 文件 | 说明 |
|------|------|
| `src/main.cu` | C++/CUDA 主程序，单进程模拟完整 IPC 流程 |
| `python/cuda_ipc.py` | Python 版本（multiprocessing.shared_memory 类比） |
| `CMakeLists.txt` | CMake 构建配置 |
| `build.bat` | Windows 一键编译（sm_120 架构） |
| `build.sh` | macOS/Linux 一键编译 |

## 编译与运行

### Windows (有 NVIDIA GPU)

```powershell
cd day33_cuda_ipc
.\build.bat
.\build\Release\cuda_ipc.exe
```

> 注意：CUDA IPC 在 Windows 上有平台限制（受 WDDM/TCC 驱动模式影响），
> 部分 IPC 功能在 Linux 上才最完整。若遇 IPC 相关错误，建议在 Linux 环境运行。

### macOS / 无 GPU 环境

```bash
cd day33_cuda_ipc/python
python cuda_ipc.py
```

## 预期输出

```
=============================================================
     Day 33: CUDA IPC - 进程间共享 GPU 显存
=============================================================

... (设备信息) ...

[主机] 共享数组元素: 1048576 个 (4.00 MB)

-------------------------------------------------------------
[演示 1] IPC 共享显存 (Get/Open/Close MemHandle)
-------------------------------------------------------------
  [生产者] 已在设备指针 ... 写入数据
  [生产者] 已通过 cudaIpcGetMemHandle 导出句柄:
  IPC 句柄 (前 8 字节): ...
  [消费者] 已通过 cudaIpcOpenMemHandle 映射到设备指针 ...
  [校验] 消费者写入对生产者可见: 通过
  [消费者] 已通过 cudaIpcCloseMemHandle 关闭共享内存映射

-------------------------------------------------------------
[演示 2] 多 GPU 间点对点拷贝 (cudaMemcpyPeer)
-------------------------------------------------------------
  ...

-------------------------------------------------------------
[演示 3] IPC 事件同步 (cudaIpcGetEventHandle)
-------------------------------------------------------------
  ...

=============================================================
                        性能与要点总结
=============================================================
  ...
=============================================================
```

## 关键 API 说明

| API | 作用 |
|-----|------|
| `cudaIpcGetMemHandle` | 从设备指针导出可共享的内存句柄 |
| `cudaIpcOpenMemHandle` | 从句柄映射出本进程可用的设备指针 |
| `cudaIpcCloseMemHandle` | 关闭（取消映射）共享内存 |
| `cudaIpcGetEventHandle` | 导出 IPC 事件句柄 |
| `cudaIpcOpenEventHandle` | 打开 IPC 事件句柄 |
| `cudaMemcpyPeer` | 多 GPU 间点对点直接拷贝 |

## 应用场景

- **分布式训练**：多个训练进程共享梯度/参数所在显存，减少拷贝
- **多进程推理服务**：多个 worker 进程共享同一份模型权重显存
- **零拷贝数据管道**：数据预处理进程与推理进程共享 GPU 缓冲区

## 练习题

1. 将单进程模拟改写为真正的两进程程序（用 `fork` 或独立可执行文件 + socket 传句柄）
2. 用 IPC 事件实现生产者-消费者之间的严格顺序保证
3. 对比 IPC 零拷贝与传统 "设备→主机→设备" 拷贝的耗时差异
4. 研究 NCCL 如何在底层利用 IPC 加速单机多卡通信

## 常见问题

**Q: 运行报错 "operation not supported" 或 IPC 句柄获取失败？**

A: CUDA IPC 对平台和驱动模式有要求：
- Windows 上受 WDDM/TCC 模式限制，功能可能不完整
- 需要 GPU 支持统一虚拟寻址（UVA）
- 建议在 Linux 环境下运行以获得完整 IPC 功能

**Q: IPC 能跨机器共享显存吗？**

A: 不能。CUDA IPC 仅限于**同一台物理机**内的进程。跨机器通信需使用 NCCL、MPI 或 RDMA。

## 进阶阅读

- [CUDA Runtime API — Interprocess Communication](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)
- [CUDA C++ Programming Guide — Interprocess Communication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication)

---

## 验收标准

完成本天学习后，运行 `cuda_ipc.exe`，你应当能够观察到并解释以下现象，才算真正掌握：

1. **共享显存读写校验通过**：演示 1 中生产者写入 `MAGIC_VALUE + i`、消费者原地乘 2，最终 `[校验] 消费者写入对生产者可见` 应显示「通过」——这证明两个指针指向的是同一块物理显存，而非各自的副本。
2. **能读懂 IPC 句柄的字节本质**：程序打印的 `IPC 句柄 (前 8 字节)` 是一串十六进制，且 `IPC 句柄总大小` 为固定值（通常 64 字节）；你应能解释为什么它可以被当作定长字节序列经 socket / 共享内存传给另一进程。
3. **P2P 演示按 GPU 数量正确分支**：演示 2 在单 GPU 环境应打印「跳过 P2P 拷贝演示」，在多 GPU 环境则打印拷贝的数据量与 P2P 直连是否支持——能据此判断当前机器的硬件拓扑。
4. **三项演示状态汇总正确**：结尾总结表格中「IPC 共享显存读写校验 = 通过」「IPC 事件同步 = 已演示」，且程序以 `[主机] 程序正常退出。` 结束，无 `operation not supported` 类错误。

## 动手挑战

以下任务基于本天 `src/main.cu` 的真实实现，循序渐进地加深理解：

1. **打印完整句柄并观察指针差异**：把 `print_ipc_handle` 的字节数从 8 改为 `sizeof(cudaIpcMemHandle_t)` 全量打印；同时对比 `d_producer` 与 `d_consumer` 两个指针值——它们在数值上可能不同，但指向同一块显存，借此理解 IPC 映射的虚拟地址语义。
2. **拆成真正的双进程程序**：以现有 `producer_kernel` / `consumer_kernel` 为基础，拆分出独立的生产者、消费者两个可执行文件，把 `cudaIpcGetMemHandle` 得到的句柄写入文件或命名管道，由消费者读入后 `cudaIpcOpenMemHandle`，验证跨进程共享（Linux 环境更完整）。
3. **加入 IPC 事件做严格同步**：在双进程版本中用 `demo_ipc_event` 里的事件机制，让消费者通过 `cudaStreamWaitEvent` 等待生产者的 `producer_kernel` 完成后再启动 `consumer_kernel`，消除竞态，体会跨进程 GPU 端同步。
