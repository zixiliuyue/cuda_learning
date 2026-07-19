# 锁竞争分析实战

## 概述

锁竞争是并发编程中最常见的性能瓶颈之一。Go 提供了两种 profile 来分析锁竞争：

- **mutex profile**: 分析互斥锁的竞争情况
- **block profile**: 分析 goroutine 阻塞等待的情况

## 运行示例

```bash
go run main.go
```

## 实验步骤

### 步骤 1: 制造锁竞争

```bash
# 使用 hey 进行并发压力测试
hey -n 10000 -c 100 http://localhost:6060/mutex/bad
```

### 步骤 2: 采集 Mutex Profile

```bash
# 注意：默认 mutex profile 采样率很低，需要设置
# 在代码中添加（已经在 main.go 中设置）
runtime.SetMutexProfileFraction(1)

# 采集 mutex profile
go tool pprof http://localhost:6060/debug/pprof/mutex

# 在 pprof 中
(pprof) top
(pprof) list badMutexHandler
(pprof) web
```

### 步骤 3: 采集 Block Profile

```bash
# 设置 block profile 采样率
runtime.SetBlockProfileRate(1)

# 采集 block profile
go tool pprof http://localhost:6060/debug/pprof/block

# 查看阻塞时间
(pprof) top
```

## 常见锁竞争模式与优化

### 1. 热点锁竞争

**BAD**: 所有 goroutine 竞争同一个锁
```go
var mu sync.Mutex
var counter int64

// 1000 个 goroutine 同时操作
func increment() {
    mu.Lock()
    counter++
    mu.Unlock()
}
```

**问题**: 锁成为瓶颈，大量时间花在等待上

**优化 1: 分片锁 (Sharding)**
```go
type ShardedCounter struct {
    shards []shard
}

type shard struct {
    mu sync.Mutex
    count int64
}

// 使用多个锁，减少竞争
func (sc *ShardedCounter) Inc(idx int) {
    sc.shards[idx % len(sc.shards)].mu.Lock()
    sc.shards[idx % len(sc.shards)].count++
    sc.shards[idx % len(sc.shards)].mu.Unlock()
}
```

**优化 2: 使用 atomic**
```go
var counter int64

func increment() {
    atomic.AddInt64(&counter, 1)
}
```

**优化 3: 每个 P 一个本地计数器**
```go
localCounters := make([]int64, runtime.NumCPU())

// 每个 goroutine 操作自己的本地计数器
// 最后汇总
```

### 2. RWMutex 使用不当

**BAD**: 读多写少但用了普通 Mutex
```go
var mu sync.Mutex
var config Config

func GetConfig() Config {
    mu.Lock()
    defer mu.Unlock()
    return config
}
```

**GOOD**: 使用 RWMutex
```go
var mu sync.RWMutex
var config Config

func GetConfig() Config {
    mu.RLock()  // 读锁，可并发
    defer mu.RUnlock()
    return config
}

func UpdateConfig(c Config) {
    mu.Lock()   // 写锁，排他
    defer mu.Unlock()
    config = c
}
```

**注意**: RWMutex 不是银弹！
- 写锁饥饿问题
- 读锁升级写锁问题
- 竞争激烈时，性能可能还不如普通 Mutex

### 3. 锁粒度过大

**BAD**: 一个大锁保护所有资源
```go
var mu sync.Mutex
var users map[int]*User
var orders map[int]*Order

func GetUser(id int) *User {
    mu.Lock()
    defer mu.Unlock()
    return users[id]
}

func GetOrder(id int) *Order {
    mu.Lock()
    defer mu.Unlock()
    return orders[id]
}
```

**GOOD**: 细粒度锁
```go
var (
    userMu sync.RWMutex
    orderMu sync.RWMutex
)

// 不同资源用不同锁，互不影响
```

### 4. 乐观并发控制

**无锁编程**，适用于读多写少场景：
```go
var config atomic.Value

// 读取总是无锁
func GetConfig() Config {
    return config.Load().(Config)
}

// 写入使用 CAS
func UpdateConfig(c Config) {
    config.Store(c)
}
```

## 性能对比测试

运行以下 benchmark 对比各种方案：

```go
package main

import (
    "sync"
    "sync/atomic"
    "testing"
)

func BenchmarkMutex(b *testing.B) {
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

func BenchmarkAtomic(b *testing.B) {
    var counter int64
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            atomic.AddInt64(&counter, 1)
        }
    })
}

func BenchmarkChannel(b *testing.B) {
    ch := make(chan int, 1)
    var counter int64
    go func() {
        for delta := range ch {
            counter += delta
        }
    }()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            ch <- 1
        }
    })
}
```

运行 benchmark：
```bash
go test -bench=. -benchmem -cpuprofile cpu.pprof
```

## 锁竞争诊断指标

在 mutex profile 中关注：

| 指标 | 说明 | 阈值 |
|-----|------|------|
| contention/call | 每次调用的平均等待时间 | > 1µs 需注意 |
| total contention | 总竞争时间 | 占 CPU > 5% |
| hot lock | 热点锁的位置 | 看 list 命令 |

## 高级优化技巧

### 1. 避免锁复制

**BAD**:
```go
type Data struct {
    mu sync.Mutex  // 值类型！
    // ...
}

func (d Data) Process() {  // 值接收者！
    d.mu.Lock()  // 锁被复制了！无效！
    // ...
}
```

**GOOD**:
```go
func (d *Data) Process() {  // 指针接收者
    d.mu.Lock()
    // ...
}
```

### 2. 使用 sync.Map

适用于读多写少且 key 相对稳定的场景：
```go
var m sync.Map

// 读无锁
val, ok := m.Load(key)

// 写使用 CAS
m.Store(key, value)
```

### 3. 单 goroutine + channel

将并发操作序列化，避免锁：
```go
type Service struct {
    cmdCh chan Command
    // ...
}

func (s *Service) Run() {
    for cmd := range s.cmdCh {
        // 单 goroutine 处理，不需要锁
        process(cmd)
    }
}
```

## 常见误区

### ❌ 误区 1: RWMutex 总是比 Mutex 快
**事实**: 读多写少才有用，否则 RWMutex 开销更大

### ❌ 误区 2: 锁越多越安全
**事实**: 锁越多死锁风险越高，性能越差

### ❌ 误区 3: atomic 总是比锁快
**事实**: 简单场景是的，但复杂场景需要权衡

## 延伸阅读

- [Go 内存模型](https://golang.org/ref/mem)
- [Mutex 源码分析](https://github.com/golang/go/blob/master/src/sync/mutex.go)
- [RWMutex 源码分析](https://github.com/golang/go/blob/master/src/sync/rwmutex.go)
- [无锁编程指南](https://preshing.com/)
