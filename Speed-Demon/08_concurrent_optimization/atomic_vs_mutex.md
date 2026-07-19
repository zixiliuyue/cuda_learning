# Atomic vs Mutex 原子操作与互斥锁性能对比

> **面试必问**：什么时候用原子操作？什么时候用互斥锁？性能差距有多大？

---

## 🎯 核心概念对比

| 特性 | Atomic 原子操作 | Mutex 互斥锁 |
|-----|----------------|-------------|
| **实现方式** | CPU 指令级 CAS（Compare-And-Swap） | 操作系统内核态信号量 |
| **上下文切换** | 无，纯用户态 | 有，可能陷入内核态 |
| **适用场景** | 单个变量的读写/递增 | 临界区包含复杂逻辑 |
| **性能开销** | ~10ns 级别 | ~50-200ns 级别 |
| **阻塞等待** | 不阻塞，自旋重试 | 阻塞，goroutine 挂起 |
| **公平性** | 无 | 可选（Go 1.9+ 支持饥饿模式） |

---

## ⚡ 性能基准测试

### 测试代码

```go
package main

import (
    "sync"
    "sync/atomic"
    "testing"
)

// 原子操作计数器
func BenchmarkAtomicCounter(b *testing.B) {
    var counter int64
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            atomic.AddInt64(&counter, 1)
        }
    })
}

// 互斥锁计数器
func BenchmarkMutexCounter(b *testing.B) {
    var counter int64
    var mu sync.Mutex
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            counter++
            mu.Unlock()
        }
    })
}

// RWMutex 读多写少场景
func BenchmarkRWMutexReadHeavy(b *testing.B) {
    var counter int64
    var mu sync.RWMutex
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.RLock()
            _ = counter
            mu.RUnlock()
        }
    })
}

// Atomic 读取
func BenchmarkAtomicRead(b *testing.B) {
    var counter int64
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _ = atomic.LoadInt64(&counter)
        }
    })
}
```

### 测试结果（8 核 CPU）

```
BenchmarkAtomicCounter-8        100000000    10.2 ns/op    0 B/op    0 allocs/op
BenchmarkMutexCounter-8          30000000    45.6 ns/op    0 B/op    0 allocs/op
BenchmarkRWMutexReadHeavy-8      60000000    18.3 ns/op    0 B/op    0 allocs/op
BenchmarkAtomicRead-8           100000000     3.1 ns/op    0 B/op    0 allocs/op
```

**结论：**
- Atomic 写入比 Mutex 快 **4.5 倍**
- Atomic 读取比 RWMutex 读快 **6 倍**
- 高并发下差距会进一步拉大

---

## 🎯 选择决策树

```
需要并发保护一个变量？
    │
    ├── 是单个变量的简单操作？（递增、交换、比较交换）
    │       │
    │       ├── 64 位以内？（int32/int64/uintptr/pointer）
    │       │       └── ✅ 用 Atomic
    │       │
    │       └── 超过 64 位？（struct/slice/map）
    │               └── ❌ 用 Mutex
    │
    └── 临界区有复杂逻辑？（多个变量操作、函数调用）
            │
            ├── 读多写少（90%+ 读）？
            │       └── ✅ 用 RWMutex
            │
            └── 读写比例接近？
                    └── ✅ 用 Mutex
```

---

## ⚠️ Atomic 的坑

### 坑 1: 32 位系统上的 64 位对齐问题

```go
// ❌ 错误：在 32 位系统上可能崩溃
type BadStruct struct {
    a int32   // 4 字节
    b int64   // 8 字节，但不对齐！
}

// ✅ 正确：把 64 位字段放在前面
type GoodStruct struct {
    b int64   // 8 字节，放在最前面自然对齐
    a int32   // 4 字节
}
```

**面试考点**：为什么 atomic 操作需要内存对齐？
> "32 位 CPU 一次只能读写 4 字节，如果 64 位变量跨了缓存行边界，需要两次读写，中间可能被其他 CPU 修改，破坏原子性。"

---

### 坑 2: Atomic 只保护单个变量

```go
// ❌ 错误：看起来用了 atomic，但逻辑不是原子的
var a, b int64

func badSwap() {
    temp := atomic.LoadInt64(&a)
    atomic.StoreInt64(&a, atomic.LoadInt64(&b))
    atomic.StoreInt64(&b, temp)
    // 中间可能被打断！
}

// ✅ 正确：多个变量用 Mutex
var mu sync.Mutex

func goodSwap() {
    mu.Lock()
    a, b = b, a
    mu.Unlock()
}
```

---

### 坑 3: Value 的类型安全问题

```go
var v atomic.Value

func main() {
    v.Store(100)        // 存 int
    
    // ❌ 运行时 panic！类型不一致
    v.Store("hello")    // 不同类型不能存同一个 Value
    
    // ✅ 正确：包装成 interface{} 或者统一类型
    type Data struct {
        Kind int
        Val  interface{}
    }
    v.Store(Data{1, 100})
    v.Store(Data{2, "hello"})
}
```

---

## 🚀 Mutex 的优化技巧

### 技巧 1: 减小临界区范围

```go
// ❌ 不好：锁里包含了 IO 操作
func badHandle() {
    mu.Lock()
    defer mu.Unlock()
    
    data := queryDB()  // 慢操作！占着锁不放
    cache[key] = data
}

// ✅ 好：只锁必要的部分
func goodHandle() {
    data := queryDB()  // 锁外面查 DB
    
    mu.Lock()
    cache[key] = data
    mu.Unlock()
}
```

---

### 技巧 2: 分片锁（Sharded Lock）

高并发下单个锁成为瓶颈时，用分片分散锁竞争。

```go
const ShardCount = 32

type ShardedMap struct {
    shards [ShardCount]*shard
}

type shard struct {
    m  map[string]interface{}
    mu sync.RWMutex
}

func (sm *ShardedMap) Get(key string) interface{} {
    shardIdx := fnv32(key) % ShardCount
    s := sm.shards[shardIdx]
    
    s.mu.RLock()
    defer s.mu.RUnlock()
    return s.m[key]
}

// 性能提升：
// 不同 key 落在不同 shard，锁竞争从 1/N 降到 1/(N*32)
```

**面试加分：**
> "分片锁本质是用空间换时间，把一个大锁拆成多个小锁，降低锁碰撞概率。
> 类似的思想在 Redis Cluster、MySQL 分库分表里也有。"

---

### 技巧 3: 避免锁的拷贝

```go
// ❌ 错误：Mutex 是值类型，函数传参会拷贝！
type BadCounter struct {
    mu sync.Mutex
    n  int
}

func (c BadCounter) Inc() {  // 值接收者！拷贝了 mu
    c.mu.Lock()              // 锁的是拷贝！完全没用
    c.n++
    c.mu.Unlock()
}

// ✅ 正确：指针接收者
func (c *GoodCounter) Inc() {
    c.mu.Lock()
    c.n++
    c.mu.Unlock()
}
```

---

## 📊 高并发下的性能对比

| 并发 Goroutine 数 | Atomic | Mutex | RWMutex(90%读) |
|-----------------|--------|-------|----------------|
| 1 | 8ns | 15ns | 12ns |
| 8 | 10ns | 45ns | 18ns |
| 100 | 12ns | 200ns | 25ns |
| 1000 | 15ns | 800ns | 35ns |

**关键观察：**
- Atomic 性能几乎不随并发数下降
- Mutex 性能随并发数增长超线性下降（因为阻塞排队）
- 这就是为什么高并发场景要尽量避免用锁！

---

## 💯 面试回答模板

**Q: 什么时候用 atomic，什么时候用 mutex？**

> "我按这个原则选择：
>
> **能用 atomic 就优先用 atomic**，因为是用户态 CPU 指令，没有上下文切换，
> 性能在高并发下比 mutex 好 5-10 倍。
>
> 但 atomic 的适用场景很有限：
> 1. 只能保护单个基本类型变量（int/int64/uintptr/pointer）
> 2. 只能做简单操作：加减、交换、比较交换、加载、存储
>
> **只要满足以下任一情况，就必须用 mutex：**
> 1. 保护多个变量
> 2. 临界区有复杂逻辑（比如函数调用、IO 操作）
> 3. 需要等待条件（配合 Cond）
>
> 另外读多写少场景（读占 90%+），可以用 RWMutex 进一步优化读性能。
>
> **最后一个判断标准：** 先写 benchmark 测一下！有时候锁竞争不激烈的场景，
> mutex 也够快，过度优化反而增加代码复杂度。"

---

**Q: atomic 是怎么实现的？为什么能保证原子性？**

> "atomic 底层是 CPU 提供的原子指令，关键是两点：
>
> **第一，总线锁定**：CPU 会锁住内存总线，不让其他 CPU 访问这块内存，
> 保证操作中间不会被打断。但总线锁开销大，现代 CPU 尽量不用。
>
> **第二，缓存一致性协议（MESI）**：现代 CPU 用缓存锁定代替总线锁，
> 只有被修改的缓存行被锁住，其他 CPU 不能同时修改这个缓存行。
>
> 还有一个前提：**变量必须内存对齐**，不能跨缓存行边界。
> 如果跨了两个缓存行，需要两次内存访问，中间就可能被打断。
> 所以 Go 在 32 位系统上会特别注意 64 位变量的对齐问题。"

---

## ✅ Checklist

写并发代码前检查一遍：

- [ ] 是不是只需要保护单个变量？→ 试试 atomic
- [ ] 64 位变量在 32 位系统上对齐了吗？
- [ ] 临界区是不是最小了？IO 操作移到锁外面了吗？
- [ ] Mutex 有没有被值拷贝？用指针接收者了吗？
- [ ] 读多写少吗？有没有考虑 RWMutex？
- [ ] 锁竞争太激烈吗？有没有考虑分片锁？

---

## 🚀 动手练习

1. 把上面的 benchmark 跑一遍，看看你的机器上性能差多少倍
2. 实现一个分片锁的 map，对比普通锁 map 的性能
3. 找一个你项目里用 mutex 的地方，看看能不能改成 atomic

**思考题：** atomic 能用来实现一个简单的自旋锁吗？如果能，和 Mutex 比有什么优缺点？
