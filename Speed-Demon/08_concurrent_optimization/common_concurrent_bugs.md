# 并发编程常见 Bug 与避坑指南

> **面试必问**：你遇到过哪些并发 Bug？怎么排查的？

---

## 🐛 Bug 1: Goroutine 泄漏（最常见）

### 症状
- Goroutine 数量持续上涨，不下降
- 内存占用越来越高
- 最后 OOM

### 场景 1: Channel 没人读，写的人永远卡住

```go
// ❌ 泄漏！
func badLeak() {
    ch := make(chan int)  // 无缓冲
    
    go func() {
        ch <- 123  // 永远卡住！没人读
    }()
    
    // 函数返回了，goroutine 还在等！
}
```

**修复：**

```go
// ✅ 方案 1: 加缓冲，至少能放一个
func goodLeak1() {
    ch := make(chan int, 1)  // 有缓冲
    
    go func() {
        ch <- 123  // 放完就走，不阻塞
    }()
}

// ✅ 方案 2: 用 context 取消
func goodLeak2(ctx context.Context) {
    ch := make(chan int)
    
    go func() {
        select {
        case ch <- 123:
        case <-ctx.Done():  // context 取消了就退出
            return
        }
    }()
}
```

---

### 场景 2: for + select 没有退出条件

```go
// ❌ 永远循环！
func badLoop() {
    go func() {
        for {
            select {
            case item := <-ch:
                process(item)
            // 没有 default，也没有退出通道！
            // ch 关闭了会读到零值，无限循环！
            }
        }
    }()
}
```

**修复：**

```go
// ✅ 检查 channel 是否关闭
func goodLoop() {
    go func() {
        for {
            select {
            case item, ok := <-ch:
                if !ok {  // channel 关闭了
                    return  // goroutine 退出！
                }
                process(item)
            }
        }
    }()
}
```

---

### 场景 3: WaitGroup Add 放错位置

```go
// ❌ Add 在 goroutine 里面！
func badWG() {
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        go func() {
            wg.Add(1)  // ❌ 太晚了！
            defer wg.Done()
            // ...
        }()
    }
    
    wg.Wait()  // 可能 Add 还没执行就 Wait 了！
}
```

**修复：**

```go
// ✅ Add 在 goroutine 外面
func goodWG() {
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        wg.Add(1)  // 启动 goroutine 之前就 Add
        go func() {
            defer wg.Done()
            // ...
        }()
    }
    
    wg.Wait()
}
```

---

## 🐛 Bug 2: 数据竞争 Data Race

### 症状
- 程序偶尔崩溃
- 出现奇怪的数值
- 很难复现，时好时坏

### 场景 1: 多个 goroutine 写同一个变量，没加锁

```go
// ❌ 数据竞争！
func badRace() {
    var count int
    
    for i := 0; i < 1000; i++ {
        go func() {
            count++  // 1000 个 goroutine 同时写！
        }()
    }
}
```

**怎么发现：** 用 Go 竞态检测器

```bash
go run -race main.go
go test -race ./...
```

会输出类似：

```
WARNING: DATA RACE
Read at 0x00c0000b4020 by goroutine 7:
  main.badRace.func1()
      main.go:10 +0x3c

Previous write at 0x00c0000b4020 by goroutine 6:
  main.badRace.func1()
      main.go:10 +0x56
```

**修复：**

```go
// ✅ 方案 1: 加锁
func goodRace1() {
    var count int
    var mu sync.Mutex
    
    for i := 0; i < 1000; i++ {
        go func() {
            mu.Lock()
            count++
            mu.Unlock()
        }()
    }
}

// ✅ 方案 2: atomic
func goodRace2() {
    var count int64
    
    for i := 0; i < 1000; i++ {
        go func() {
            atomic.AddInt64(&count, 1)
        }()
    }
}
```

---

### 场景 2: 闭包捕获循环变量

```go
// ❌ 经典中的经典！90% 的 Go 开发者都踩过
func badClosure() {
    for i := 0; i < 5; i++ {
        go func() {
            fmt.Println(i)  // 打印什么？全是 5！
        }()
    }
}
```

**为什么？** 所有 goroutine 捕获的是同一个变量 `i` 的引用，
等 goroutine 跑起来的时候，循环已经结束了，`i` 已经是 5 了。

**修复：**

```go
// ✅ 方案 1: 参数传进去
func goodClosure1() {
    for i := 0; i < 5; i++ {
        go func(i int) {  // 作为参数传进去！
            fmt.Println(i)  // 0,1,2,3,4
        }(i)  // 每次循环都拷贝一份
    }
}

// ✅ 方案 2: 循环内局部变量
func goodClosure2() {
    for i := 0; i < 5; i++ {
        i := i  // 循环内的局部变量！
        go func() {
            fmt.Println(i)
        }()
    }
}
```

---

## 🐛 Bug 3: 死锁 Deadlock

### 场景 1: 自己等自己

```go
// ❌ 递归调用，锁不可重入！
func badRecursive() {
    var mu sync.Mutex
    
    mu.Lock()
    defer mu.Unlock()
    
    // ... 中间又调用了同一个锁的函数！
    mu.Lock()  // 永远卡住！Go mutex 不可重入！
    defer mu.Unlock()
}
```

**面试考点：** Go 的 Mutex 是不可重入的！和 Java 不一样！

---

### 场景 2: 循环等待（A 等 B，B 等 A）

```go
// ❌ 经典死锁：哲学家吃饭问题
func badDeadlock() {
    var mu1, mu2 sync.Mutex
    
    go func() {
        mu1.Lock()
        time.Sleep(10 * time.Millisecond)  // 拿着锁等
        mu2.Lock()  // 等 mu2，但另一个 goroutine 拿着
        // ...
        mu2.Unlock()
        mu1.Unlock()
    }()
    
    go func() {
        mu2.Lock()
        time.Sleep(10 * time.Millisecond)
        mu1.Lock()  // 等 mu1，但第一个 goroutine 拿着
        // ...
        mu1.Unlock()
        mu2.Unlock()
    }()
}
```

**修复：** 所有 goroutine 按相同顺序加锁！

```go
// ✅ 都先锁 mu1，再锁 mu2，不会循环等待
func goodDeadlock() {
    var mu1, mu2 sync.Mutex
    
    go func() {
        mu1.Lock()
        mu2.Lock()
        // ...
        mu2.Unlock()
        mu1.Unlock()
    }()
    
    go func() {
        mu1.Lock()  // 一样的顺序！
        mu2.Lock()
        // ...
        mu2.Unlock()
        mu1.Unlock()
    }()
}
```

---

## 🐛 Bug 4: Channel 关闭相关

### 场景 1: 关闭已经关闭的 channel

```go
// ❌ panic: close of closed channel
func badDoubleClose() {
    ch := make(chan int)
    close(ch)
    close(ch)  // panic！
}
```

### 场景 2: 往已经关闭的 channel 写

```go
// ❌ panic: send on closed channel
func badSendAfterClose() {
    ch := make(chan int)
    close(ch)
    ch <- 123  // panic！
}
```

**正确的关闭原则：**

```
✅ 谁发送，谁关闭
✅ 多个发送者，用一个专门的 goroutine 负责关闭
✅ 不要让接收者关闭！
```

```go
// ✅ 正确的多发送者关闭模式
func goodMultiSender() {
    ch := make(chan int, 100)
    var wg sync.WaitGroup
    
    // 10 个发送者
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            ch <- id
        }(i)
    }
    
    // 一个专门的 goroutine 等所有发送完再 close
    go func() {
        wg.Wait()
        close(ch)  // 只有这里能 close！
    }()
    
    // 接收者放心 for range
    for v := range ch {
        fmt.Println(v)
    }
}
```

---

## 🔍 并发 Bug 排查工具

### 1. 竞态检测器 -race

```bash
# 必须会用！
go run -race main.go
go test -race ./...
```

- 能发现 99% 的数据竞争问题
- 性能开销约 2-10 倍，测试环境开，生产环境别开

---

### 2. pprof + goroutine profile

```go
import _ "net/http/pprof"

func main() {
    go func() {
        http.ListenAndServe(":6060", nil)
    }()
}
```

```bash
# 看有多少 goroutine
curl http://localhost:6060/debug/pprof/goroutine?debug=2

# 或者用 go tool pprof
go tool pprof http://localhost:6060/debug/pprof/goroutine
```

看输出：

```
goroutine profile: total 1234
# 太多了！正常应该几十到几百
```

---

### 3. deadlock 检测库

```go
import "github.com/sasha-s/go-deadlock"

// 把 sync.Mutex 换成 deadlock.Mutex
var mu deadlock.Mutex

// 死锁时会自动输出：
POTENTIAL DEADLOCK:
Current goroutine 0x1:
  lock 0x123 was taken by goroutine 0x2
  waiting for lock 0x456

goroutine 0x2:
  lock 0x456 was taken by goroutine 0x2
  waiting for lock 0x123
```

---

## 💡 并发编程最佳实践

### 原则 1: 不要通过共享内存通信，通过通信共享内存

```go
// ❌ 不好：共享变量加锁
var result int
var mu sync.Mutex

go func() {
    mu.Lock()
    result = compute()
    mu.Unlock()
}()

// ✅ 好：通过 channel 传结果
ch := make(chan int, 1)

go func() {
    ch <- compute()
}()

result := <-ch
```

---

### 原则 2: 尽量不要让 goroutine 关心其他 goroutine 的状态

每个 goroutine 只关心：
- 自己的输入 channel
- 自己的输出 channel
- 什么时候该退出

```go
// ✅ 好的 worker 模式
func worker(ctx context.Context, in <-chan Job, out chan<- Result) {
    for {
        select {
        case job := <-in:
            out <- process(job)
        case <-ctx.Done():  // 只关心自己该不该退出
            return
        }
    }
}
```

---

### 原则 3: 不要启动不知道怎么停止的 goroutine

```go
// ❌ 不好：goroutine 跑了就不管了
func bad() {
    go func() {
        for {
            // ... 永远跑！想停都停不了！
        }
    }()
}

// ✅ 好：每个 goroutine 都有停止方式
func good() {
    ctx, cancel := context.WithCancel(context.Background())
    
    go func() {
        for {
            select {
            // ...
            case <-ctx.Done():
                return  // 能停！
            }
        }
    }()
    
    // 想停的时候调用 cancel
    cancel()
}
```

---

## 💯 面试回答模板

**Q: 你遇到过哪些并发 Bug？怎么排查的？**

> "我遇到过的并发 Bug 主要有这几类：
>
> **最常见的是 goroutine 泄漏**
> 一开始写代码经常忘记 channel 阻塞的问题，
> 比如无缓冲 channel 没人读，写的 goroutine 就永远卡住了。
> 排查方法是用 pprof 的 goroutine profile，
> 看 goroutine 数量是不是持续上涨，然后看堆栈都堵在哪一行。
>
> **然后是数据竞争**
> 多个 goroutine 写同一个变量没加锁，
> 现象是程序偶尔崩溃，出现奇怪的值，很难复现。
> 排查用 Go 自带的 `-race` 竞态检测器，
> 几乎 100% 能找到问题。
>
> **还有死锁**
> 比如循环等待，A 等 B，B 等 A，
> 或者 Go Mutex 不可重入，递归调用自己锁死。
> 排查可以用 `go-deadlock` 这个库，替换 sync.Mutex，
> 死锁时会自动打印锁依赖关系。
>
> **最经典的坑是闭包捕获循环变量**
> for 循环里启动 goroutine，直接用循环变量，
> 结果所有 goroutine 都拿到最后一个值。
> 这个坑几乎每个 Go 开发者都踩过...
>
> **总结一下我的并发编程原则：**
> 1. 每个 goroutine 都必须有明确的退出条件
> 2. 能不用锁就不用，用 channel 通信
> 3. 用 `-race` 跑测试，数据竞争零容忍
> 4. 不要写你不知道怎么停止的 goroutine"

---

## ✅ 并发编程 Checklist

写并发代码前过一遍：

- [ ] 每个 goroutine 都有退出条件吗？
- [ ] channel 缓冲大小合理吗？会阻塞吗？
- [ ] 闭包捕获循环变量了吗？参数传进去了吗？
- [ ] WaitGroup Add 是在 goroutine 外面调用的吗？
- [ ] 谁发 channel 谁关闭，接收者不关闭吗？
- [ ] 锁的顺序一致吗？不会循环等待吗？
- [ ] 用 `-race` 跑过测试了吗？
- [ ] pprof goroutine 数量正常吗？

---

## 🚀 动手练习

1. 写一个故意有数据竞争的程序，用 `-race` 检测一下
2. 写一个 goroutine 泄漏的程序，用 pprof 定位到泄漏点
3. 写一个死锁程序，用 go-deadlock 库检测

**思考题：** 为什么 Go 不提供可重入锁？是忘了吗？
