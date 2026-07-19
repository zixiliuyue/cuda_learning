# Web Service 性能优化对比 Demo

这个 Demo 展示了常见的 Go Web 服务性能问题和优化方案。

## 🚀 快速开始

### 1. 启动服务

```bash
go run main.go
```

服务会在 `http://localhost:6060` 启动

### 2. 测试接口

提供 4 个对比接口：

| 接口 | 说明 |
|-----|------|
| `/user/bad?id=123` | 有性能问题的实现 |
| `/user/good?id=123` | 优化后的实现 |
| `/user/batch/bad?count=100` | 批量处理差实现 |
| `/user/batch/good?count=100` | 批量处理好实现 |
| `/stats` | 查看系统运行指标 |

### 3. 查看性能指标

```bash
curl http://localhost:6060/stats
```

会输出：
- Goroutine 数量
- 内存使用量（MB）
- GC 次数
- 缓存大小

---

## 📊 压测对比

### 准备工作

安装 hey 压测工具：
```bash
go install github.com/rakyll/hey@latest
```

### 压测单接口

**Bad 版本（有问题）：**
```bash
hey -n 3000 -c 100 http://localhost:6060/user/bad?id=123
```

**Good 版本（优化后）：**
```bash
hey -n 3000 -c 100 http://localhost:6060/user/good?id=123
```

### 压测批量接口

**Bad 版本：**
```bash
hey -n 1000 -c 50 "http://localhost:6060/user/batch/bad?count=100"
```

**Good 版本：**
```bash
hey -n 1000 -c 50 "http://localhost:6060/user/batch/good?count=100"
```

---

## 🔍 优化点详解

### 1. 缓存锁优化

**Bad 实现问题：**
```go
cacheLock.Lock()         // 进来就上锁
time.Sleep(20 * time.Millisecond)  // ❌ 慢操作在锁里面！
cacheLock.Unlock()
```

**Good 优化：**
```go
// 1. 先读缓存（读锁，并发可以进）
cacheLock.RLock()
p, ok := localCache[id]
cacheLock.RUnlock()

// 2. 没命中才去 DB（锁外面！！！）
time.Sleep(20 * time.Millisecond)

// 3. 写回缓存（只锁写操作）
cacheLock.Lock()
localCache[id] = p
cacheLock.Unlock()
```

**优化效果：** 锁等待时间减少 90% 以上

---

### 2. Goroutine 池优化

**Bad 实现问题：**
```go
for i := 0; i < count; i++ {
    go func() {  // 创建 count 个 goroutine！
        // ...
    }()
}
```

**Good 优化：**
```go
const numWorkers = 10  // 固定 worker 数量
work := make(chan int, count)

for w := 0; w < numWorkers; w++ {
    go func() {
        for idx := range work {
            // 处理任务
        }
    }()
}
```

**优化效果：** Goroutine 数量从 100+ 降到 10 个

---

### 3. Channel 缓冲防泄漏

**Bad 实现问题：**
```go
ch := make(chan int)  // ❌ 无缓冲！超时会泄露 goroutine
```

**Good 优化：**
```go
ch := make(chan int, 1)  // ✅ 有缓冲！写完直接退出
```

---

## 📈 预期性能对比

| 指标 | Bad 版本 | Good 版本 | 提升 |
|-----|---------|----------|------|
| QPS | ~300 | ~1500 | **5x** |
| P99 延迟 | 500ms+ | 100ms | **5x** |
| Goroutine 峰值 | 1000+ | < 100 | **10x** |
| 内存分配 | 大量小对象 | 显著减少 | **3-5x** |

---

## 🔧 pprof 性能分析

服务已经内置 pprof，压测时可以同时抓 profile：

### 1. CPU Profile
```bash
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=10
```

### 2. Heap Profile
```bash
go tool pprof http://localhost:6060/debug/pprof/heap
```

### 3. Goroutine Profile
```bash
go tool pprof http://localhost:6060/debug/pprof/goroutine
```

### 4. Trace
```bash
curl http://localhost:6060/debug/pprof/trace?seconds=10 > trace.out
go tool trace trace.out
```

---

## ✅ Checklist 验证

- [ ] Bad 版本压测 QPS 明显低于 Good 版本
- [ ] Bad 版本 Goroutine 数量持续上涨
- [ ] Good 版本 Goroutine 数量稳定在低位
- [ ] Good 版本 P99 延迟明显更低
- [ ] pprof 可以看到 Bad 版本锁等待时间长

---

## 💡 延伸思考

1. 为什么全局大锁会严重影响性能？
2. Goroutine 太多会导致什么问题？
3. 为什么 Channel 无缓冲会导致 goroutine 泄漏？
4. 还有哪些常见的 Web 服务性能优化手段？
