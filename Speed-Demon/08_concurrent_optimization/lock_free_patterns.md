# 无锁编程模式与最佳实践

> **面试必问**：什么是无锁编程？CAS 是什么？无锁一定比有锁快吗？

---

## 🎯 核心概念

### 什么是无锁（Lock-Free）？

**传统有锁编程：**
- 某个 goroutine 持有锁，其他所有 goroutine 都得等，啥也干不了
- 最坏情况：持有锁的 goroutine 被调度出去，所有人都卡住
- 典型问题：死锁、活锁、优先级反转

**无锁编程：**
- 不用 mutex，只用 CPU 原子指令（CAS）
- 任意时刻，至少有一个 goroutine 能继续执行
- 不会因为某个 goroutine 卡住就所有人都卡住

---

### CAS: Compare-And-Swap 灵魂操作

CAS 是所有无锁数据结构的基础。

```go
// CAS 伪代码
func cas(ptr *int64, old, new int64) bool {
    if *ptr == old {
        *ptr = new
        return true  // 交换成功
    }
    return false  // 中间被别人改了，重试
}

// Go 标准库实现
import "sync/atomic"

atomic.CompareAndSwapInt64(&val, old, new)
```

**面试重点：** CAS 是原子操作，由 CPU 保证原子性！

---

## ⚡ 无锁 vs 有锁性能对比

### 测试代码

```go
package main

import (
    "sync"
    "sync/atomic"
    "testing"
)

// 有锁计数器
func BenchmarkMutexCounter(b *testing.B) {
    var mu sync.Mutex
    var counter int64
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            counter++
            mu.Unlock()
        }
    })
}

// 无锁计数器
func BenchmarkLockFreeCounter(b *testing.B) {
    var counter int64
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            atomic.AddInt64(&counter, 1)
        }
    })
}

// 高竞争场景
func BenchmarkHighContentionMutex(b *testing.B) {
    var mu sync.Mutex
    var counter int64
    
    b.SetParallelism(100)  // 100 个 goroutine 抢
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            counter++
            mu.Unlock()
        }
    })
}

func BenchmarkHighContentionLockFree(b *testing.B) {
    var counter int64
    
    b.SetParallelism(100)
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            atomic.AddInt64(&counter, 1)
        }
    })
}
```

### 测试结果

```
# 8 并发，竞争一般
BenchmarkMutexCounter-8           30000000    45 ns/op
BenchmarkLockFreeCounter-8     100000000   10 ns/op  # 4.5 倍

# 100 并发，高竞争
BenchmarkHighContentionMutex-8   5000000   250 ns/op
BenchmarkHighContentionLockFree-8 50000000  25 ns/op  # 10 倍！
```

**关键结论：** 竞争越激烈，无锁优势越大！

---

## 🔧 常见无锁模式

### 模式 1: 无锁计数器（最简单）

```go
type Counter struct {
    val int64
}

func (c *Counter) Inc() {
    atomic.AddInt64(&c.val, 1)
}

func (c *Counter) Get() int64 {
    return atomic.LoadInt64(&c.val)
}

// ✅ 优点：零额外开销，性能最好
// ⚠️ 局限：只能加减，不能做更复杂的操作
```

---

### 模式 2: 无锁栈（LIFO）

```go
import "unsafe"

type Node struct {
    val  T
    next *Node
}

type LockFreeStack struct {
    head unsafe.Pointer  // *Node
}

func (s *LockFreeStack) Push(val T) {
    newNode := &Node{val: val}
    
    for {
        // 1. 读取当前 head
        oldHead := atomic.LoadPointer(&s.head)
        newNode.next = (*Node)(oldHead)
        
        // 2. CAS 把新节点放到 head
        if atomic.CompareAndSwapPointer(
            &s.head,
            oldHead,
            unsafe.Pointer(newNode),
        ) {
            return  // 成功
        }
        // 失败了说明中间被别人改了，重试
    }
}

func (s *LockFreeStack) Pop() (T, bool) {
    for {
        oldHead := atomic.LoadPointer(&s.head)
        if oldHead == nil {
            var zero T
            return zero, false
        }
        
        next := (*Node)(oldHead).next
        
        if atomic.CompareAndSwapPointer(
            &s.head,
            oldHead,
            unsafe.Pointer(next),
        ) {
            return (*Node)(oldHead).val, true
        }
        // 失败重试
    }
}
```

**ABA 问题！！** ⚠️

```
线程 1: Pop 读到 head=A，准备 CAS
线程 2: Pop A，Push B，Push A
线程 1: CAS 把 head 从 A 换成 A.next
但是 A 已经不是原来的 A 了！
```

**解决方案：** 版本号 + 指针一起 CAS

```go
type TaggedPointer struct {
    ptr unsafe.Pointer
    tag uint64  // 版本号
}

// 每次 Pop/Push tag 都 +1，避免 ABA
```

---

### 模式 3: 无锁队列（FIFO）

经典的 Michael-Scott 无锁队列算法：

```go
type LKQueue struct {
    head unsafe.Pointer
    tail unsafe.Pointer
}

type node struct {
    value T
    next  unsafe.Pointer
}

func (q *LKQueue) Enqueue(val T) {
    newNode := unsafe.Pointer(&node{value: val})
    
    for {
        tail := atomic.LoadPointer(&q.tail)
        next := atomic.LoadPointer(&(*node)(tail).next)
        
        if tail == atomic.LoadPointer(&q.tail) {
            if next == nil {
                // 尝试把 tail.next 指向新节点
                if atomic.CompareAndSwapPointer(
                    &(*node)(tail).next,
                    next,
                    newNode,
                ) {
                    // 尝试移动 tail
                    atomic.CompareAndSwapPointer(
                        &q.tail,
                        tail,
                        newNode,
                    )
                    return
                }
            } else {
                // 帮其他线程移动 tail
                atomic.CompareAndSwapPointer(
                    &q.tail,
                    tail,
                    next,
                )
            }
        }
    }
}
```

---

## ⚠️ 无锁编程的坑

### 坑 1: ABA 问题（最经典）

```go
// ❌ 只比较值，不比较版本
if cas(&ptr, old, new) {
    // 中间可能被改成其他值又改回来！
}

// ✅ 用版本号
type Value struct {
    ptr unsafe.Pointer
    tag uint64  // 每次修改都 +1
}

// 64 位指针 + 64 位版本号 = 128 位 CAS
// Go 1.19+ 有 atomic.Pointer，但是没有 128 位 CAS
// 需要用其他方式避免 ABA
```

---

### 坑 2: 忙等待，CPU 100%

```go
// ❌ 无限重试，不成功不罢休
for !cas(&val, old, new) {
    // 什么都不干，空转！
    // 高竞争下 CPU 100%
}

// ✅ 加个暂停，或者退化成有锁
for i := 0; i < 10; i++ {
    if cas(&val, old, new) {
        return
    }
    runtime.Gosched()  // 让出 CPU
}
// 重试多次还不成功，竞争太激烈了，用 mutex
mu.Lock()
val = new
mu.Unlock()
```

---

### 坑 3: 内存回收问题

```go
// ❌ Pop 出来的节点不能立刻 free！
// 其他 goroutine 可能还拿着指针在访问！

// 解决方案 1: Hazard Pointers（危险指针）
// 每个线程把正在访问的指针记下来，
// 回收前先确认没人在用

// 解决方案 2: Epoch-Based Reclamation（EBR）
// 按 epoch 分批回收

// 解决方案 3: 最简单——直接泄漏！
// （除非是个永不停机的服务）
```

---

### 坑 4: 内存重排 / 内存屏障

```go
// CPU 可能重排指令顺序！
// 看起来顺序写的代码，实际执行顺序可能不一样

// ❌ 没有内存屏障
a = 1
b = 2
// CPU 可能先写 b 再写 a！

// ✅ Go atomic 内部已经有内存屏障了
atomic.StoreInt64(&a, 1)
atomic.StoreInt64(&b, 2)
// 保证 a 的写在 b 的写之前完成
```

---

## 🚫 什么时候不要用无锁？

**别用无锁的场景：

| 场景 | 为什么 |
|-----|-------|
| **竞争不激烈 | 无锁比有锁快不了多少，甚至更慢 |
| **临界区复杂 | 多个变量，CAS 写起来太复杂 |
| **内存回收困难 | 很难处理 ABA 和内存回收 |
| **代码可读性重要 | 无锁代码太难懂，维护成本高 |
| **性能不是瓶颈 | 不要为了优化而优化 |

**现实：** 99% 的业务场景不需要无锁编程！
Mutex 足够快，足够简单！

---

## 📊 无锁 vs 有锁决策树

```
需要并发安全吗？
    │
    ├── 只是单个变量加减？
    │       └── ✅ 用 atomic
    │
    ├── 简单数据结构（栈/队列）
    │   └── 竞争特别激烈？
    │           ├── 是 → 考虑无锁
    │           └── 否 → 用有锁
    │
    └── 复杂逻辑/多个变量
            └── ✅ 老老实实 Mutex
```

---

## 💯 面试回答模板

**Q: 无锁编程一定比有锁快吗？**

> "不一定，要看场景：
>
> **竞争激烈时，无锁确实比有锁快很多**（5-10 倍），
> 因为有锁会导致大量 goroutine 阻塞排队，
> 无锁虽然重试，但总比阻塞好。
>
> **竞争不激烈时，反而可能更慢**，
> 因为 CAS 循环重试本身有开销，
> 而且 CAS 指令本身也不比 mutex 快多少。
>
> **更重要的是：无锁编程复杂度太高了！
> 要处理 ABA 问题、内存屏障、内存回收...
> 99% 的业务场景根本不需要，Mutex 足够快，
> 而且代码简单，不容易错。
>
> **我的原则：**
> 能 atomic 就 atomic，
> 不行就 Mutex，
> 实在不行才考虑无锁数据结构。
> 无锁是最后一步的优化手段，不是上来就用的！"

---

**Q: 什么是 ABA 问题？怎么解决？**

> "ABA 是无锁编程里最经典的坑：
>
> **场景：**
> 1. 线程 1 读到值是 A，准备做 CAS
> 2. 线程 2 把 A 改成 B，再改回 A
> 3. 线程 1 CAS 一看值还是 A，以为没变，成功了
> 4. 但这个 A 已经不是原来的 A 了！
>
> **危害：**
> - 节点释放了又被放回来 → 悬垂指针
> - 状态被改过但没被发现
>
> **解决方案：**
> 1. **版本号**：每次修改都加版本号，CAS 同时比较值和版本
> 2. **Hazard Pointers**：记录正在访问的指针
> 3. **EBR 分代回收**：按 epoch 分批回收内存
>
> **Go 里最简单的方案：**
> 简单场景用 `sync/atomic.Value`，复杂场景直接用成熟的库，
> 别自己写无锁数据结构，坑太多了！"

---

## ✅ 无锁编程 Checklist

- [ ] 真的需要无锁吗？Mutex 真的不够吗？
- [ ] ABA 问题考虑了吗？需要版本号吗？
- [ ] 内存屏障考虑了吗？顺序对吗？
- [ ] 内存回收怎么处理？Hazard Pointers？EBR？
- [ ] 高竞争下的忙等待怎么处理？会把 CPU 跑满吗？
- [ ] 测试过多核场景吗？单线程没问题不代表多线程没问题！
- [ ] 代码注释写清楚了吗？别人能看懂吗？

---

## 🚀 动手练习

1. 把上面的 benchmark 跑一下，看你机器上无锁 vs 有锁在不同并发下的性能差
2. 实现一个带版本号的无锁栈，解决 ABA 问题
3. 找一个你项目里用 mutex 的地方，看看能不能改成无锁

**思考题：** 为什么 Go 标准库没有提供无锁队列？是不会写吗？
