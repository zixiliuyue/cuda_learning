# Day 34: 自定义 GPU 内存池

## 学习目标

1. 理解 `cudaMalloc` / `cudaFree` 的开销来源（驱动 / 内核态交互）
2. 掌握内存池（Memory Pool）的设计思想与实现
3. 实现最简单的 **bump allocator**（线性分配器）
4. 对比“直接分配”与“内存池分配”在高频场景下的性能差异
5. 了解内存池在深度学习框架显存管理中的应用

## 核心概念

### 为什么需要内存池？

`cudaMalloc` 和 `cudaFree` 并不是简单的指针运算，它们涉及：
- 用户态到内核态的切换
- 驱动内部的显存簿记与页表管理
- 可能的隐式同步

在需要**频繁分配 / 释放**的场景（如每帧的临时缓冲、深度学习的中间张量），
这些开销会显著拖慢程序。内存池通过“一次性预分配 + 复用”来摊销这些开销。

### Bump Allocator（线性分配器）

本项目采用最简单的策略：

```
[========|--------------------------------]
         ^offset（水位线）
 已分配区  剩余可用区
```

- **分配**：返回当前 `offset` 处的地址，然后把 `offset` 向前推进（对齐后）
- **释放**：单块释放为 no-op（bump allocator 不记录每块边界）
- **重置**：`offset = 0`，O(1) 一次性回收所有内存
- **销毁**：整块 `cudaFree`

优点：分配极快（仅指针运算）；缺点：不支持随机顺序的单块释放。
适合“生命周期一致、批量回收”的场景。

### 数据结构

```c
typedef struct {
    char*  base_ptr;        // 预分配显存的基地址
    size_t total_size;      // 总大小（字节）
    size_t offset;          // 当前分配偏移（bump allocator）
    int    num_allocations; // 分配次数统计
} MemPool;
```

### 核心 API

| 函数 | 作用 |
|------|------|
| `init_memory_pool(pool, size)` | 一次性预分配一大块显存 |
| `pool_allocate(pool, size)`    | 从池中分配（移动偏移，O(1)） |
| `pool_free(pool, ptr)`         | 归还（bump allocator 下为 no-op） |
| `pool_reset(pool)`             | 一次性回收所有分配（O(1)） |
| `destroy_memory_pool(pool)`    | 释放整块预分配显存 |

## 性能对比

| 方案 | 分配开销 | 说明 |
|------|---------|------|
| 直接 cudaMalloc/cudaFree | 高 | 每次都与驱动交互 |
| 自定义内存池 | 极低 | 一次预分配 + O(1) 偏移移动 |

在 1000 次分配 / 释放的循环中，内存池通常可带来数十倍到上百倍的加速
（具体倍数取决于 GPU、驱动与分配大小）。

## 编译与运行

### Windows (nvcc + sm_120)

```powershell
cd day34_memory_pool
.\build.bat
.\build\memory_pool.exe
```

### Linux / macOS (CMake)

```bash
cd day34_memory_pool
bash build.sh
./build/memory_pool
```

### Python 版本

```bash
python python/memory_pool.py
```

## 关键教学点

1. **摊销分析**：把 N 次昂贵操作变成 1 次昂贵 + N 次廉价操作
2. **对齐**：GPU 访问需满足对齐要求，本例使用 256 字节对齐
3. **取舍**：bump allocator 用“无法随机释放”换取“极致的分配速度”
4. **进阶方向**：free-list 分配器、slab 分配器、CUDA 11.2+ 的 `cudaMallocAsync`
   与内存池 API（`cudaMemPool_t`）

---

## 验收标准

完成本天学习后，运行 `memory_pool.exe`，你应当能够观察到并解释以下现象，才算真正掌握：

1. **内存池初始化与水位线正确**：程序打印 `[内存池] 初始化成功，预分配显存: 8.00 MB`，功能验证阶段从池中分配两个 1MB 缓冲后，`当前池水位线` 应显示约 `2.00 MB / 8.00 MB`，说明 bump allocator 的偏移推进符合预期。
2. **Kernel 计算结果正确**：使用池内显存运行 `vector_scale`（`c[i] = a[i] * 2.5`）后，校验输出 `Kernel 计算结果: 正确 (PASS)`，证明内存池分配出的指针可被 Kernel 正常读写。
3. **内存池显著快于直接分配**：性能对比表格中「自定义内存池」的总耗时与平均每次耗时应远小于「直接 cudaMalloc/cudaFree」，且性能总结的「内存池相对加速比」应明显大于 1（通常数十倍以上）。
4. **资源正确回收**：结尾打印 `[内存池] 已销毁，累计分配次数: N`（N 与实际分配次数一致），并以 `[主机] 程序正常退出。` 结束，无显存泄漏或分配失败提示。

## 动手挑战

以下任务基于本天 `src/main.cu` 的真实实现，循序渐进地加深理解：

1. **触发并观察空间不足**：在功能验证后不调用 `pool_reset`，连续 `pool_allocate` 直到超过 8MB 总容量，观察 `[内存池] 空间不足！` 的错误分支被触发，理解 bump allocator 没有 reset 就无法复用空间的局限。
2. **验证对齐效果**：修改 `pool_allocate` 请求一个非 256 倍数的大小（如 100 字节），打印对齐前后的 `offset`，确认 `align_up` 把每次分配都补齐到 `POOL_ALIGNMENT`（256）的整数倍，理解对齐对 GPU 访问的意义。
3. **升级为 free-list 分配器**：在 `MemPool` 中维护一个已释放块的链表，让 `pool_free` 真正把块归还到空闲链表、`pool_allocate` 优先复用空闲块，把「只能整体 reset」的 bump allocator 改造成支持随机单块释放的分配器，并重新跑性能对比。
