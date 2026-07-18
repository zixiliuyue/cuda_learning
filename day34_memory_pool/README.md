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
