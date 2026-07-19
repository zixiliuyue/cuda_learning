# Goroutine Leak 排查实战

## 实验目标

1. 理解 goroutine 泄漏的常见模式
2. 学习使用 pprof 排查 goroutine 泄漏
3. 掌握 goroutine 生命周期管理

## 快速开始

```bash
go run main.go
```

## 实验步骤

### 步骤 1: 观察基线

1. 启动服务
2. 访问 http://localhost:6060/debug/pprof/goroutine?debug=1
3. 记录初始 goroutine 数量

### 步骤 2: 制造泄漏

```bash
# 多次访问泄漏接口
for i in {1..20}; do curl http://localhost:6060/leak1; done
```

再次检查 goroutine 数量，观察增长。

### 步骤 3: 使用 pprof 分析

```bash
# 采集 goroutine profile
go tool pprof http://localhost:6060/debug/pprof/goroutine

# 在 pprof 中
(pprof) top
(pprof) list leakHandler1
(pprof) web
```

**预期结果**: 应该能看到大量 goroutine 卡在 channel receive 操作

### 步骤 4: 对比正确实现

```bash
# 访问正确实现
for i in {1..20}; do curl http://localhost:6060/good1; done
```

观察 goroutine 数量不会持续增长。

## Goroutine 泄漏常见模式

### 1. Channel 阻塞

**BAD**:
```go
func leak() {
    ch := make(chan int)
    go func() {
        val := <-ch  // 永远阻塞！
        fmt.Println(val)
    }()
    // 忘记往 ch 里发数据
}
```

**GOOD**:
```go
func noLeak() {
    ch := make(chan int)
    go func() {
        select {
        case val := <-ch:
            fmt.Println(val)
        case <-time.After(1 * time.Second):
            fmt.Println("timeout")
        }
    }()
    ch <- 42  // 或者有超时机制
}
```

### 2. 无限循环无退出

**BAD**:
```go
func leak() {
    go func() {
        for {  // 永远循环！
            time.Sleep(1 * time.Second)
        }
    }()
}
```

**GOOD**:
```go
func noLeak() {
    ctx, cancel := context.WithCancel(context.Background())
    go func() {
        for {
            select {
            case <-ctx.Done():
                return
            default:
                time.Sleep(1 * time.Second)
            }
        }
    }()
    // 用完后 cancel()
}
```

### 3. WaitGroup 使用不当

**BAD**:
```go
func leak() {
    var wg sync.WaitGroup
    wg.Add(1)  // Add 了，但 Done 可能永远不调用！
    go func() {
        // 某个分支忘记 wg.Done()
        if someCondition {
            return
        }
        wg.Done()
    }()
    wg.Wait()  // 永远等不到！
}
```

**GOOD**:
```go
func noLeak() {
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {
        defer wg.Done()  // defer 确保一定调用！
        if someCondition {
            return
        }
    }()
    wg.Wait()
}
```

### 4. Timer 泄漏

**BAD**:
```go
func leak(done <-chan struct{}) {
    timer := time.NewTimer(5 * time.Second)
    select {
    case <-timer.C:
        fmt.Println("timeout")
    case <-done:
        // timer 没停止！goroutine 泄漏！
    }
}
```

**GOOD**:
```go
func noLeak(done <-chan struct{}) {
    timer := time.NewTimer(5 * time.Second)
    defer timer.Stop()  // 确保停止！
    select {
    case <-timer.C:
        fmt.Println("timeout")
    case <-done:
    }
}
```

### 5. Goroutine 池无界

**BAD**:
```go
func handle(req Request) {
    go process(req)  // 每个请求一个 goroutine
}
```

**GOOD**:
```go
var pool = make(chan struct{}, 100)  // 限制并发数

func handle(req Request) {
    pool <- struct{}{}
    go func() {
        defer func() { <-pool }()
        process(req)
    }()
}
```

## 排查工具与技巧

### 1. runtime.NumGoroutine()

```go
// 在代码中监控
log.Printf("Goroutines: %d", runtime.NumGoroutine())
```

### 2. pprof goroutine profile

```bash
# debug=1: 摘要信息
curl http://localhost:6060/debug/pprof/goroutine?debug=1

# debug=2: 详细堆栈
curl http://localhost:6060/debug/pprof/goroutine?debug=2
```

### 3. go tool pprof 交互

```bash
go tool pprof http://localhost:6060/debug/pprof/goroutine

(pprof) top           # 看热点
(pprof) list func     # 源码定位
(pprof) traces        # 看调用栈
(pprof) web           # 可视化
```

### 4. 对比快照

```bash
# 时间点 1
curl http://localhost:6060/debug/pprof/goroutine > gor1.pprof

# ... 做一些操作 ...

# 时间点 2
curl http://localhost:6060/debug/pprof/goroutine > gor2.pprof

# 对比差异
go tool pprof -base gor1.pprof gor2.pprof
```

## 高级模式：goroutine 泄露检测器

### 使用 goleak

```go
import "go.uber.org/goleak"

func TestMyFunction(t *testing.T) {
    defer goleak.VerifyTestMain(t)  // 检查测试结束后有无泄露
    
    // 你的测试代码
}
```

## 真实案例分析

### 案例 1: HTTP Handler 泄漏

**问题代码**:
```go
func handler(w http.ResponseWriter, r *http.Request) {
    result := make(chan Result)
    
    go func() {
        result <- longRunningOperation()
    }()
    
    select {
    case res := <-result:
        writeResponse(w, res)
    case <-r.Context().Done():
        // 客户端断开连接，但 goroutine 还在跑！
    }
}
```

**问题**: 客户端断开后，longRunningOperation 还在继续运行

**修复**:
```go
func handler(w http.ResponseWriter, r *http.Request) {
    result := make(chan Result, 1)  // 缓冲 channel
    
    go func() {
        result <- longRunningOperation(r.Context())  // 传递 context
    }()
    
    select {
    case res := <-result:
        writeResponse(w, res)
    case <-r.Context().Done():
    }
}
```

### 案例 2: Channel 作为信号使用

**BAD**:
```go
func worker(stop chan bool) {
    for {
        select {
        case <-stop:
            return
        default:
            work()
        }
    }
}

func main() {
    stop := make(chan bool)
    go worker(stop)
    // 程序退出时没有 close(stop)
}
```

**GOOD**:
```go
func worker(ctx context.Context) {
    for {
        select {
        case <-ctx.Done():
            return
        default:
            work()
        }
    }
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()  // 确保退出时 cancel
    go worker(ctx)
}
```

## Best Practices

### 1. 永远考虑 goroutine 如何退出

**问自己**: 这个 goroutine 在什么条件下会结束？

### 2. 使用 context 管理生命周期

```go
func doWork(ctx context.Context) error {
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            // do work
        }
    }
}
```

### 3. 避免在循环中启动 goroutine

**BAD**:
```go
for item := range items {
    go process(item)  // 可能创建大量 goroutine
}
```

**GOOD**:
```go
// 使用 worker pool
const numWorkers = 10
for i := 0; i < numWorkers; i++ {
    go worker(items)
}
```

### 4. 为 goroutine 添加监控

```go
func monitoredWorker(ctx context.Context, id int) {
    log.Printf("Worker %d started", id)
    defer log.Printf("Worker %d exited", id)
    
    for {
        select {
        case <-ctx.Done():
            return
        default:
        }
    }
}
```

## 延伸阅读

- [Go Concurrency Patterns](https://blog.golang.org/pipelines)
- [Context Package](https://blog.golang.org/context)
- [Goroutine Leaks: The Forgotten Sender](https://www.ardanlabs.com/blog/2018/11/goroutine-leaks-the-forgotten-sender.html)
- [goleak 库](https://github.com/uber-go/goleak)
