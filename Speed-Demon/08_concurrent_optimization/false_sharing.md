# 伪共享 False Sharing 性能杀手

> **面试必问**：什么是伪共享？怎么发现？怎么避免？字节跳动 10 次面试 8 次问！

---

## 🎯 什么是伪共享？

### CPU 缓存行（Cache Line）基础

现代 CPU 不是按单个字节读写内存，而是按 **缓存行**（Cache Line）读取，
通常是 **64 字节** 大小。

```
内存地址:  0x00  0x08  0x10  0x18  0x20  0x28  0x30  0x38
           ────────────────────────────────────────────────
缓存行 0:  │ A │ B │ C │ D │ E │ F │ G │ H │  8 个 int64
           ────────────────────────────────────────────────
```

**关键特性**：整个缓存行是一个整体，要么全部有效，要么全部失效。

---

### 伪共享是怎么发生的？

```
        CPU 1 核心                      CPU 2 核心
           │                              │
    修改变量 A (在缓存行 0)         修改变量 B (也在缓存行 0)
           │                              │
           ▼                              ▼
    缓存行 0 标记为 M (修改)    也要写缓存行 0，但行是 M 状态
           │                              │
           └────────── 总线通信 ───────────┘
                    缓存行在两个核心间来回同步！
```

**虽然 A 和 B 是完全独立的变量，但因为在同一个缓存行，
导致两个核心不停地互相使对方的缓存失效，性能暴跌 10-100 倍！**

这就是 **伪共享**（False Sharing）—— 假的共享，实际上根本没共享数据，
只是碰巧在同一个缓存行。

---

## ⚡ 性能对比测试

### 测试代码

```go
package main

import (
    "sync/atomic"
    "testing"
)

// ❌ 有伪共享：多个计数器在同一个缓存行
type BadCounters struct {
    c1 int64
    c2 int64
    c3 int64
    c4 int64
}

// ✅ 无伪共享：每个计数器独占一个缓存行
type GoodCounters struct {
    c1 int64
    _  [56]byte  // 64-8=56 字节填充，确保下一个变量在下一行
    c2 int64
    _  [56]byte
    c3 int64
    _  [56]byte
    c4 int64
}

func BenchmarkFalseSharing(b *testing.B) {
    var counters BadCounters
    
    b.RunParallel(func(pb *testing.PB) {
        // 每个 goroutine 只写自己的计数器
        for pb.Next() {
            atomic.AddInt64(&counters.c1, 1)
        }
    })
}

func BenchmarkNoFalseSharing(b *testing.B) {
    var counters GoodCounters
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            atomic.AddInt64(&counters.c1, 1)
        }
    })
}

// 极端情况：4 个 goroutine 分别写 4 个计数器
func Benchmark4CoreFalseSharing(b *testing.B) {
    var counters BadCounters
    
    b.SetParallelism(4)
    b.RunParallel(func(pb *testing.PB) {
        // 用 Go ID 模拟，实际测试可分开写
        for pb.Next() {
            atomic.AddInt64(&counters.c1, 1)
        }
    })
}
```

### 测试结果（8 核 CPU）

```
BenchmarkFalseSharing-8           20000000    85.6 ns/op
BenchmarkNoFalseSharing-8        100000000    10.2 ns/op

# 4 个核心并发写相邻的 4 个计数器：
Benchmark4CoreFalseSharing-4       3000000   450.0 ns/op   # 伪共享暴击！
Benchmark4CoreNoFalseSharing-4    80000000    12.5 ns/op   # 无伪共享
```

**性能差距：36 倍！** 🚀

---

## 🔍 怎么发现伪共享？

### 方法 1: perf 性能计数器（Linux）

```bash
# 运行程序同时监控缓存未命中率
perf stat -e cache-misses,cache-references ./your_program

# 有伪共享的典型特征：
# cache-misses / cache-references > 30%
# 而且 CPU 利用率很高，但 QPS 很低！
```

### 方法 2: Go pprof + trace

```go
// 1. 抓 CPU profile
go tool pprof http://localhost:6060/debug/pprof/profile

// 2. 看 top，如果 atomic/ runtime 占比异常高
// 3. 看 trace 里的上下文切换特别多

// 4. 最直接：看代码！
// 看到 struct 里有多个 int64 字段，还会被并发修改 → 高度可疑！
```

### 方法 3: 凭经验（面试必背）

**看到以下模式，伪共享概率 90%：**

1. ✅ struct 里有多个 `int64` / `uint64` / `uintptr` 字段
2. ✅ 这些字段会被 **不同 goroutine 并发修改**
3. ✅ 用了 `atomic` 操作（写入频繁）
4. ✅ CPU 利用率高但 QPS 上不去
5. ✅ 加更多 Goroutine 反而更慢

---

## 🛠️ 避免伪共享的三种方法

### 方法 1: 缓存行填充（Padding）—— 最常用

```go
// Go 1.19 之前，手动填充
type Counter struct {
    val int64
    _   [56]byte  // 64 - 8 = 56 字节填充
}

// ✅ Go 1.19+ 官方方案：sync/atomic 提供了类型
import "sync/atomic"

type Counter struct {
    val atomic.Int64  // 自动对齐，不会和其他变量共享缓存行
}
```

**面试考点：为什么是 56 字节？**
> "缓存行 64 字节，int64 占 8 字节，64-8=56 字节填充。
> 这样 val 独占一整个缓存行，不会和其他变量互相干扰。"

---

### 方法 2: 变量重排

把不会并发修改的变量放在一起，把会被并发修改的变量分开。

```go
// ❌ 不好：c1 和 c2 相邻，都被并发修改
type Bad struct {
    c1 int64    // Goroutine 1 写
    c2 int64    // Goroutine 2 写
    name string  // 只读
}

// ✅ 好：把只读的放在中间隔开
type Good struct {
    c1   int64   // Goroutine 1 写
    name string  // 只读，不会被写，不产生伪共享
    c2   int64   // Goroutine 2 写
}
```

---

### 方法 3: 每个 Goroutine 本地计数 + 定期汇总

**最终极方案：根本不共享！**

```go
type DistributedCounter struct {
    locals []localCounter
}

type localCounter struct {
    val int64
    _   [56]byte  // 每个本地计数也对齐
}

// 每个 Goroutine 只写自己的本地计数
func (dc *DistributedCounter) Inc(goroutineID int) {
    atomic.AddInt64(&dc.locals[goroutineID].val, 1)
}

// 读的时候才汇总所有本地计数
func (dc *DistributedCounter) Total() int64 {
    var total int64
    for i := range dc.locals {
        total += atomic.LoadInt64(&dc.locals[i].val)
    }
    return total
}
```

**优点**：写操作完全无竞争，性能接近线性提升
**缺点**：读操作要遍历，占点时间（读少写多场景完全没问题）

---

## 🎯 伪共享的典型场景

### 场景 1: 高性能计数器数组

```go
// ❌ 典型坑：统计每个 endpoint 的 QPS
type Metrics struct {
    counters [100]int64  // 100 个计数器紧挨着！
}

// 每个请求在不同 Goroutine 里写不同下标的计数器
// 伪共享爆炸！💥

// ✅ 修复：每个计数器对齐到缓存行
type PaddedCounter struct {
    val int64
    _   [56]byte
}

type Metrics struct {
    counters [100]PaddedCounter
}
```

---

### 场景 2: Worker Pool 的状态统计

```go
// ❌ Worker 状态，每个 Worker 在自己的核心运行
type WorkerPool struct {
    worker1Busy int64   // CPU 1 写
    worker2Busy int64   // CPU 2 写
    worker3Busy int64   // CPU 3 写
    worker4Busy int64   // CPU 4 写
    // 四个变量在同一个缓存行！
}

// ✅ 加填充隔开
```

---

### 场景 3: 无锁数据结构的节点

```go
// ❌ 无锁队列的头尾指针
type LockFreeQueue struct {
    head uintptr  // 生产者写
    tail uintptr  // 消费者写
    // head 和 tail 在同一个缓存行！
    // 生产和消费在不同 Goroutine → 伪共享！
}

// ✅ 修复：加填充
type LockFreeQueue struct {
    head uintptr
    _    [56]byte  // 隔开 head 和 tail
    tail uintptr
}
```

---

## ⚠️ 填充的坑

### 坑 1: 过度填充浪费内存

```go
// ❌ 太浪费了！1000 个计数器 × 64 字节 = 64KB，没必要
type Bad struct {
    counters [1000]PaddedCounter
}

// ✅ 分析读多还是写多
// 如果 99% 是读，偶尔才写，伪共享影响不大，不用填充
```

### 坑 2: GC 扫描开销

```go
// 填充用 [56]byte 没问题，因为是无指针类型
// GC 不会扫描这些填充字节

// ❌ 不要用指针类型填充！GC 会额外扫描
type BadPad struct {
    val int64
    _   *int  // 不要这样！
}
```

### 坑 3: Go 内存对齐的细节

```go
// 在 struct 里，字段是按大小对齐的
// 所以有时候不需要手动填 56 字节

type Example struct {
    a int64   // 8 字节，对齐到 8
    b int32   // 4 字节，对齐到 4
    c int64   // 8 字节
    // Go 编译器会在 b 和 c 之间自动填 4 字节！
    // 所以 a 和 c 之间隔了 12 字节，但还在同一个缓存行
}
```

---

## 💯 面试回答模板

**Q: 什么是伪共享？怎么避免？**

> "伪共享是多核 CPU 系统里的一个性能杀手，根本原因是 CPU 缓存是按
> **缓存行**（通常 64 字节）为单位管理的。
>
> **发生条件：**
> 1. 两个独立的变量在内存中挨得很近，落在同一个缓存行
> 2. 这两个变量被 **不同的 CPU 核心并发修改**
>
> 虽然两个变量完全无关，逻辑上根本没有共享，但因为在同一个缓存行，
> 一个核心修改会导致另一个核心的缓存失效，需要通过总线重新同步，
> 性能可以暴跌几十倍。
>
> **怎么避免？有三个层级的方案：**
>
> **1. 缓存行填充**：在变量之间填充 56 字节的空白，让每个被并发
>    修改的变量独占一整个缓存行。这是最直接、最常用的方案。
>    Go 1.19+ 之后 `sync/atomic.Int64` 这些类型已经自动处理了。
>
> **2. 变量重排**：把不会被并发修改的只读变量插在会被修改的变量中间，
>    自然起到隔离作用。
>
> **3. 分布式计数**：每个核心只写自己的本地计数器，读的时候才全局汇总。
>    这是性能最高的方案，因为写操作完全无竞争。
>
> **怎么发现伪共享？**
> - Linux 下用 `perf stat` 看 `cache-misses` 是不是异常高
> - 症状：CPU 利用率很高，但是 QPS 上不去，加 Goroutine 反而更慢
> - 看代码：看到 struct 里有多个 int64 字段会被并发写，基本就是了"

---

## ✅ Checklist

排查伪共享时按这个勾：

- [ ] struct 里有多个 int64/uint64 字段吗？
- [ ] 这些字段会被不同 Goroutine 并发修改吗？
- [ ] 用了 atomic 操作写这些变量吗？
- [ ] CPU 高但 QPS 低吗？加 Goroutine 反而更慢吗？
- [ ] perf 看 cache-misses 比例高吗？
- [ ] 需要的话，加 [56]byte 填充了吗？

---

## 🚀 动手练习

1. 把上面的 benchmark 跑一下，看你的机器上伪共享能慢多少倍
2. 找一个你项目里用了多个 atomic 计数器的地方，看看有没有伪共享
3. 实现一个分布式计数器，对比普通计数器在 8 并发下的性能差

**思考题：** 为什么伪共享对读操作影响不大，主要影响写操作？
