# Go 性能优化速查表

## 📋 目录

- [内存优化](#内存优化)
- [并发优化](#并发优化)
- [CPU 优化](#cpu-优化)
- [I/O 优化](#io-优化)
- [诊断工具](#诊断工具)
- [常见性能坑](#常见性能坑)

---

## 内存优化

### 1. 字符串拼接

**❌ BAD** (O(n²) 时间, O(n²) 分配)
```go
result := ""
for _, s := range parts {
    result += s  // 每次都创建新字符串
}
```

**✅ GOOD** (O(n) 时间, 1-2 次分配)
```go
var builder strings.Builder
builder.Grow(totalSize)  // 预分配！
for _, s := range parts {
    builder.WriteString(s)
}
result := builder.String()
```

### 2. 切片预分配

**❌ BAD** (多次扩容, 内存拷贝)
```go
var s []int
for i := 0; i < 10000; i++ {
    s = append(s, i)
}
```

**✅ GOOD** (1 次分配)
```go
s := make([]int, 0, 10000)  // 预分配容量
for i := 0; i < 10000; i++ {
    s = append(s, i)
}
```

### 3. sync.Pool 复用对象

**✅ 适用场景**:
- 频繁分配/释放的临时对象
- 对象大小适中 (< 1MB)
- 生命周期短

```go
var bufPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 4096)  // 4KB buffer
    },
}

// 使用
buf := bufPool.Get().([]byte)
defer bufPool.Put(buf)  // 一定要放回！
```

**❌ 不适用**:
- 状态需要保持的对象
- 超大对象
- 需要被 GC 及时回收的

### 4. 结构体内存对齐

**❌ BAD** (24 bytes 实际只需要 16)
```go
type Bad struct {
    a bool   // 1 byte
    b int64  // 8 bytes   + 7 padding
    c bool   // 1 byte    + 7 padding
    d int32  // 4 bytes
}  // Total: 32 bytes
```

**✅ GOOD** (按大小排序)
```go
type Good struct {
    b int64  // 8 bytes
    d int32  // 4 bytes
    a bool   // 1 byte
    c bool   // 1 byte
}  // Total: 16 bytes
```

**验证工具**:
```bash
go install golang.org/x/tools/go/analysis/passes/fieldalignment/cmd/fieldalignment@latest
fieldalignment ./path/to/package
```

### 5. 减少逃逸

**❌ 导致逃逸的常见操作**:
1. 返回指针
2. 发送指针到 channel
3. 被闭包捕获
4. interface 方法调用
5. 大小超过栈限制

**✅ 减少逃逸的技巧**:
1. 值传递 vs 指针传递（小对象用值）
2. 尽可能使用具体类型而非 interface
3. 避免在循环中分配

**查看逃逸分析**:
```bash
go build -gcflags="-m -m" ./... 2>&1 | grep escape
```

---

## 并发优化

### 1. 锁的选择

| 场景 | 推荐 | 避免 |
|-----|------|------|
| 读多写少 | sync.RWMutex | sync.Mutex |
| 简单计数器 | atomic | 任何锁 |
| 分片数据 | sharded lock | 全局锁 |

**✅ Atomic 操作**:
```go
var counter int64

// 增加
atomic.AddInt64(&counter, 1)

// 读取
current := atomic.LoadInt64(&counter)

// CAS
if atomic.CompareAndSwapInt64(&counter, old, new) {
    // 成功
}
```

### 2. Worker Pool 模式

**❌ BAD** (无限 goroutine)
```go
for task := range tasks {
    go process(task)  // 可能创建百万级 goroutine
}
```

**✅ GOOD** (固定并发数)
```go
const numWorkers = runtime.NumCPU() * 2

// 启动 worker
for i := 0; i < numWorkers; i++ {
    go func() {
        for task := range taskChan {
            process(task)
        }
    }()
}

// 发送任务
for _, task := range tasks {
    taskChan <- task
}
```

### 3. Channel 缓冲

**选择合适的缓冲大小**:
```go
// 无缓冲 - 同步
ch := make(chan Task)  // 发送者阻塞直到接收者就绪

// 有缓冲 - 异步
ch := make(chan Task, 100)  // 缓冲 100 个元素
```

**缓冲大小选择指南**:
- 0: 需要严格同步的场景
- 1-10: 削峰填谷, 平滑突发流量
- 100+: 高吞吐场景

### 4. 避免 Goroutine 泄漏

**确保每个 goroutine 都能退出**:

```go
// ✅ 好的方式: 使用 context
func worker(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return  // 退出
        case task := <-taskChan:
            process(task)
        }
    }
}

// ✅ 好的方式: 关闭 channel 通知退出
func worker(done <-chan struct{}) {
    for {
        select {
        case <-done:
            return
        default:
        }
    }
}
```

---

## CPU 优化

### 1. 热点函数优化

**使用 pprof 定位热点**:
```bash
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=10
(pprof) top  // 看前 10 热点
(pprof) list <function>  // 函数级分析
```

### 2. 循环优化

**✅ 循环展开**:
```go
// 普通循环
for i := 0; i < len(data); i++ {
    process(data[i])
}

// 展开 4 次
i := 0
for ; i+3 < len(data); i += 4 {
    process(data[i])
    process(data[i+1])
    process(data[i+2])
    process(data[i+3])
}
// 处理剩余
for ; i < len(data); i++ {
    process(data[i])
}
```

**✅ 减少循环内的条件判断**:
```go
// ❌ 每次循环都判断
for i := 0; i < len(data); i++ {
    if featureEnabled {
        processA(data[i])
    } else {
        processB(data[i])
    }
}

// ✅ 把判断提到循环外
if featureEnabled {
    for i := 0; i < len(data); i++ {
        processA(data[i])
    }
} else {
    for i := 0; i < len(data); i++ {
        processB(data[i])
    }
}
```

### 3. 函数内联

**编译器决定是否内联，但可以帮助**:
- 函数体小
- 没有闭包
- 没有递归
- 没有 defer

**查看内联决策**:
```bash
go build -gcflags="-m" ./... 2>&1 | grep inline
```

---

## I/O 优化

### 1. 批量操作

**❌ BAD** (每次操作都 I/O)
```go
for _, record := range records {
    // 每次都写入文件
    fmt.Fprintf(file, "%v\n", record)
}
```

**✅ GOOD** (使用 buffer)
```go
writer := bufio.NewWriter(file)
defer writer.Flush()  // 别忘了 flush！

for _, record := range records {
    fmt.Fprintf(writer, "%v\n", record)
}
```

### 2. 异步 I/O

**✅ 使用 goroutine + channel**:
```go
results := make(chan Result, 100)

// 并发获取数据
for _, url := range urls {
    go func(u string) {
        results <- fetch(u)
    }(url)
}

// 收集结果
for i := 0; i < len(urls); i++ {
    result := <-results
    // 处理 result
}
```

**✅ 限制并发**:
```go
sem := make(chan struct{}, 10)  // 最多 10 并发

for _, url := range urls {
    sem <- struct{}{}  // 获取信号量
    go func(u string) {
        defer func() { <-sem }()  // 释放
        results <- fetch(u)
    }(url)
}
```

---

## 诊断工具

### pprof 命令速查

| 命令 | 用途 |
|-----|------|
| `go tool pprof <profile>` | 启动交互式分析 |
| `top` | 显示热点函数 |
| `top -cum` | 按累计时间排序 |
| `list <func>` | 函数源码级分析 |
| `web` | 生成调用图 (需要 graphviz) |
| `svg` | 生成 SVG 格式 |
| `pdf` | 生成 PDF 报告 |
| `traces` | 显示所有调用栈 |

### 常用 profile 端点

```bash
# CPU profile
curl http://localhost:6060/debug/pprof/profile?seconds=30

# Heap profile
curl http://localhost:6060/debug/pprof/heap

# Goroutine profile
curl http://localhost:6060/debug/pprof/goroutine?debug=2

# Mutex profile (需要 runtime.SetMutexProfileFraction)
curl http://localhost:6060/debug/pprof/mutex

# Block profile (需要 runtime.SetBlockProfileRate)
curl http://localhost:6060/debug/pprof/block

# Execution trace
curl http://localhost:6060/debug/pprof/trace?seconds=10
```

### Trace 视图

```bash
go tool trace trace.out
```

| 视图 | 用途 |
|-----|------|
| View trace | 时间轴视图, 最强大 |
| Goroutine analysis | 每个 goroutine 的状态和时间 |
| Network flow | 网络阻塞分析 |
| Synchronization blocking | 同步阻塞分析 |
| Minimum mutator utilization | GC 对应用的影响 |

### Benchmark 命令

```bash
# 运行所有 benchmark
go test -bench=. -benchmem

# 运行特定的
go test -bench="Name" -benchmem

# 带 profile
go test -bench=. -benchmem -cpuprofile=cpu.pprof -memprofile=mem.pprof

# 运行多次, 统计更准确
go test -bench=. -benchmem -count=10

# 使用 benchstat 对比
go install golang.org/x/perf/cmd/benchstat@latest
benchstat old.txt new.txt
```

---

## 常见性能坑

### ❌ 坑 1: 忽视内建函数的复杂度

```go
// ❌ len(s) 在循环中每次都计算
for i := 0; i < len(s); i++ {  // 字符串 len 是 O(1), 但切片也是
    // ...
}

// ✅ 先保存
n := len(s)
for i := 0; i < n; i++ {
    // ...
}
```

### ❌ 坑 2: defer 在循环中

```go
// ❌ BAD: defer 只在函数返回时执行, 循环中 defer 会累积
for i := 0; i < 10000; i++ {
    mu.Lock()
    defer mu.Unlock()  // 这会积累 10000 个 defer!
    // ...
}

// ✅ GOOD: 不用 defer, 或包装成函数
for i := 0; i < 10000; i++ {
    func() {
        mu.Lock()
        defer mu.Unlock()  // 匿名函数返回时执行
        // ...
    }()
}
```

### ❌ 坑 3: 忽略错误, 导致重试风暴

```go
// ❌ 失败立即重试, 可能打垮下游
for {
    err := fetch()
    if err != nil {
        continue  // 立即重试
    }
    break
}

// ✅ 指数退避
for attempt := 0; attempt < 5; attempt++ {
    err := fetch()
    if err == nil {
        break
    }
    time.Sleep(time.Second * time.Duration(1<<attempt))  // 1s, 2s, 4s...
}
```

### ❌ 坑 4: range 循环拷贝大元素

```go
// ❌ BAD: 每次循环拷贝整个 struct
for _, item := range bigStructs {  // item 是拷贝!
    // ...
}

// ✅ GOOD: 使用索引, 或用指针切片
for i := range bigStructs {
    item := &bigStructs[i]  // 只拷贝指针
    // ...
}
```

### ❌ 坑 5: fmt.Sprintf 性能敏感场景

```go
// ❌ 热路径使用
for _, req := range requests {
    log.Printf("processing %s", fmt.Sprintf("%d", req.ID))
}

// ✅ 使用更快的字符串拼接
// 或者预先生成日志需要的字符串
```

---

## 性能优化决策流程

```
发现性能问题
    ↓
使用 pprof/trace 定位瓶颈
    ↓
是 CPU 瓶颈? → 看 CPU profile → 热点函数优化
    ↓
是内存瓶颈? → 看 heap profile → 减少分配
    ↓
是 GC 压力? → 看 alloc profile → 减少对象生命周期
    ↓
是锁竞争? → 看 mutex/block profile → 减少持有时间
    ↓
是调度问题? → 看 trace → 优化 goroutine 模式
    ↓
运行 benchmark 验证
    ↓
用 benchstat 确认提升
    ↓
提交代码 (附性能数据)
```

---

## 性能优化金字塔

```
        ┌─────────────────┐
        │   架构设计优化   │  ← 影响最大 (10x - 1000x)
        │  - 服务拆分      │
        │  - 缓存策略      │
        │  - 异步化        │
        └─────────────────┘
        ┌─────────────────┐
        │   算法优化       │  ← 10x - 100x
        │  - O(n²) → O(n) │
        │  - 空间换时间    │
        └─────────────────┘
        ┌─────────────────┐
        │   并发优化       │  ← 2x - 10x
        │  - 减少锁竞争    │
        │  - 合理并发度    │
        └─────────────────┘
        ┌─────────────────┐
        │   内存优化       │  ← 1.5x - 5x
        │  - 预分配        │
        │  - 对象池        │
        │  - 减少逃逸      │
        └─────────────────┘
        ┌─────────────────┐
        │   微优化         │  ← 1.1x - 2x
        │  - 循环优化      │
        │  - 内联          │
        │  - 内存对齐      │
        └─────────────────┘
```

**记住**: 先优化上层, 再优化下层。架构优化的收益最大, 微优化的收益最小。

---

## 资源链接

- [Go 官方 pprof 博客](https://blog.golang.org/pprof)
- [Go 诊断指南](https://golang.org/doc/diagnostics)
- [Go 语言高性能编程](https://geektutu.com/post/high-performance-go.html)
- [High Performance Go Workshop](https://dave.cheney.net/high-performance-go-workshop)
