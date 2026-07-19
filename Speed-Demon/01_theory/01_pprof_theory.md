# pprof 工作原理与采样机制

## 概述

pprof 是 Go 内置的性能分析工具，它基于 Google 的 pprof 工具。理解其工作原理是高效进行性能分析的基础。

## pprof 的架构

```
┌─────────────────┐
│  被分析应用程序  │
│  (runtime/pprof)│
└────────┬────────┘
         │ 写入 profile 数据
         ▼
┌─────────────────┐
│  profile 文件   │ ← protobuf 格式
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  go tool pprof  │
└────────┬────────┘
         │
         ▼
┌───────────────────────────────────┐
│  可视化输出: top, web, list, ...  │
└───────────────────────────────────┘
```

## CPU 采样原理

### 采样机制
1. **定时器中断**: Go runtime 使用 `setitimer`（类 Unix）或高分辨率计时器（Windows）定期触发中断
2. **采样频率**: 默认 100Hz（每 10ms 采样一次）
3. **采样内容**: 每个 P（Processor）的当前执行栈

### 采样流程

```
  时间轴 →
    │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼
  [中断] [中断] [中断] [中断] [中断]
    │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼
  [采样] [采样] [采样] [采样] [采样]
    │      │      │      │      │
    └──────┴──────┴──────┴──────┘
           │
           ▼
      [统计汇总]
```

### 为什么是 100Hz？

- **精度与开销的平衡**: 太高会增加性能开销，太低样本不足
- **理论依据**: 根据奈奎斯特定理，100Hz 可以捕捉到 50Hz 以上的周期性行为
- **实际效果**: 对于大多数 Go 应用，这已经足够准确

### 采样偏差问题

**重要**: CPU 采样是统计采样，不是精确计数

- **采样偏差**: 某些函数可能正好在采样间隙执行
- **最小可检测**: 通常认为占比 < 0.1% 的函数不可靠
- **解决方案**: 延长采样时间，或提高采样频率

## 内存采样原理

### 采样机制
内存采样是基于**抽样**的，不是每次分配都采样。

**采样率计算公式**:
```go
采样率 = 1 / (1 + exp(-分配大小 / 512KB))
```

这意味着：
- 小分配（< 1KB）: 采样率 ≈ 0.2%
- 中分配（≈ 128KB）: 采样率 ≈ 50%
- 大分配（> 1MB）: 采样率 ≈ 100%

### 为什么这样设计？

1. **性能考虑**: 每次分配都采样会严重影响性能
2. **统计有效**: 大分配对内存影响更大，更需要被采样
3. **外推估算**: 采样后的数据会通过公式外推，估算真实分配量

### 内存 profile 的两种视图

1. **alloc_space**: 分配的总字节数（包括已释放）
2. **inuse_space**: 当前正在使用的字节数

## Goroutine 采样

Goroutine profile 不是采样，而是**快照**：

- 记录所有 goroutine 的调用栈
- 显示当前 goroutine 数量及其状态
- 可用于检测 goroutine 泄漏

## Block/Mutex 采样

### Block Profile
- 记录 goroutine 因同步原语阻塞的时间
- 默认关闭，需要手动启用

### Mutex Profile
- 记录锁竞争的情况
- 采样可配置的竞争事件

## Profile 数据格式

pprof 使用 Protocol Buffers 存储数据，结构如下：

```protobuf
message Profile {
  repeated Sample sample = 1;        // 采样样本
  repeated ValueType sample_type = 2; // 样本类型（CPU, memory 等）
  repeated Mapping mapping = 3;       // 内存映射
  repeated Location location = 4;     // 代码位置
  repeated Function function = 5;     // 函数信息
  repeated string string_table = 6;   // 字符串表
  int64 time_nanos = 7;              // 采样时间
  int64 duration_nanos = 8;          // 采样持续时间
}
```

## 采样开销分析

| Profile 类型 | 典型开销 | 建议使用场景 |
|------------|---------|------------|
| CPU | 2-5% | 生产环境可用 |
| Heap | < 1% | 生产环境持续开启 |
| Goroutine | 低 | 按需使用 |
| Block/Mutex | 可变 | 问题排查时启用 |
| Trace | 10-20% | 开发/测试环境 |

## 高级技巧

### 1. 自定义采样频率

```go
import "runtime/pprof"

// 修改 CPU 采样频率（Hz）
// 注意：必须在开始 CPU profile 前设置
pprof.StartCPUProfile(w)
```

### 2. 内存采样率控制

```go
import "runtime"

// 设置内存采样率，单位字节
// 1 = 每次分配都采样，0 = 关闭采样
runtime.MemProfileRate = 128 * 1024
```

### 3. 并发安全

**重要**: pprof 的所有操作都是并发安全的，但不要同时启动多个相同类型的 profile。

## 常见误区

1. ❌ **误区**: CPU profile 显示的是精确的时间消耗
   ✅ **事实**: 是统计采样，有误差，需看相对值

2. ❌ **误区**: 内存 profile 显示所有分配
   ✅ **事实**: 是抽样结果，小分配可能被遗漏

3. ❌ **误区**: 生产环境不能开 profile
   ✅ **事实**: CPU/Heap profile 开销很低，生产环境可用

## 进一步阅读

- [Russ Cox - Profiling Go Programs](https://blog.golang.org/profiling-go-programs)
- [Go Runtime Source - pprof.go](https://github.com/golang/go/blob/master/src/runtime/pprof/pprof.go)
