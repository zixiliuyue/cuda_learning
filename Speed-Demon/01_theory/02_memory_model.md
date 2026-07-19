# Go 内存模型与逃逸分析

## Go 内存分配器架构

### TCMalloc 继承

Go 的内存分配器基于 Google TCMalloc（Thread-Caching Malloc），但做了专门优化以适应 Go 的并发模型。

```
┌─────────────────────────────────────────────────────┐
│                    应用程序                         │
└───────────┬─────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│  MCache (per-P cache)                              │
│  ├─ span class 0 (tiny objects < 16B)             │
│  ├─ span class 1 (16-32B)                         │
│  ├─ ...                                           │
│  └─ span class 66 (> 32KB)                        │
└───────────┬─────────────────────────────────────────┘
            │ 从 MCache 获取失败时
            ▼
┌─────────────────────────────────────────────────────┐
│  MCentral (per-size-class central cache)           │
│  ┌─────────────┐    ┌─────────────┐                │
│  │  empty list │    │  nonempty   │                │
│  └─────────────┘    └─────────────┘                │
└───────────┬─────────────────────────────────────────┘
            │ 从 MCentral 获取失败时
            ▼
┌─────────────────────────────────────────────────────┐
│  MHeap (global heap)                               │
│  ┌─────────────────────────────────────────────┐   │
│  │                  Arenas                     │   │
│  │  (64-bit: 每个 64MB, 32-bit: 每个 4MB)     │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## 逃逸分析 (Escape Analysis)

### 什么是逃逸？

**逃逸** = 变量的生命周期超出了当前函数栈帧

**不逃逸** = 变量只在函数内使用，函数返回后不再引用

### 为什么重要？

| 分配位置 | 分配开销 | GC 压力 | 访问速度 |
|---------|---------|--------|---------|
| 栈 (Stack) | ~1ns | 无 | L1 Cache |
| 堆 (Heap) | ~100ns | 有 | 可能较慢 |

### 逃逸分析决策树

```
变量定义
    │
    ▼
  是否被返回? ──是──→ 逃逸到堆
    │否
    ▼
  是否被 goroutine 引用? ──是──→ 逃逸到堆
    │否
    ▼
  是否被闭包捕获并在外部使用? ──是──→ 逃逸到堆
    │否
    ▼
  是否通过 interface 调用方法? ──是──→ 可能逃逸
    │否
    ▼
  大小是否超过栈限制? ──是──→ 逃逸到堆
    │否
    ▼
  可以分配在栈上
```

### 经典逃逸场景

#### 1. 返回指针

```go
// BAD: user 逃逸到堆
func NewUser() *User {
    return &User{}  // 逃逸
}

// GOOD: 使用值返回（如果合适）
func NewUser() User {
    return User{}  // 不逃逸
}
```

#### 2. 发送到 channel

```go
// BAD: 发送到 channel 的指针会逃逸
func producer(ch chan<- *Data) {
    data := &Data{}  // 逃逸
    ch <- data
}

// GOOD: 发送值（如果数据小）
func producer(ch chan<- Data) {
    data := Data{}  // 不逃逸
    ch <- data
}
```

#### 3. 闭包捕获

```go
// BAD: 被闭包捕获的变量可能逃逸
func Counter() func() int {
    count := 0  // 逃逸！
    return func() int {
        count++
        return count
    }
}
```

#### 4. interface 调用

```go
// BAD: interface 方法调用可能导致逃逸
func process(v io.Reader) {
    buf := make([]byte, 1024)  // 可能逃逸
    v.Read(buf)
}

// GOOD: 使用具体类型
func processFile(f *os.File) {
    buf := make([]byte, 1024)  // 通常不逃逸
    f.Read(buf)
}
```

### 查看逃逸分析结果

```bash
# 查看详细逃逸分析
go build -gcflags="-m -m" ./...

# 只看逃逸
go build -gcflags="-m" ./... 2>&1 | grep escape
```

### 输出解释

```
./main.go:10:2: moved to heap: x      // x 逃逸到堆
./main.go:11:13: y does not escape   // y 不逃逸
./main.go:12:13: z escapes to heap   // z 逃逸
```

## 内存对齐

### 为什么需要对齐？

1. **性能**: 现代 CPU 按字（word）访问内存，对齐的数据访问更快
2. **原子操作**: 某些原子操作要求内存对齐
3. **硬件要求**: 某些架构（如 ARM）对未对齐访问会崩溃

### 内存对齐规则

| 类型 | 对齐系数（字节） |
|-----|----------------|
| bool, uint8, int8 | 1 |
| uint16, int16 | 2 |
| uint32, int32, float32 | 4 |
| uint64, int64, float64 | 8 |
| 指针、map、chan、func | 8 (64位) / 4 (32位) |
| struct | 最大字段的对齐系数 |

### 结构体优化示例

**BAD - 内存浪费**：
```go
// 实际占用: 32 字节（浪费 13 字节）
type Bad struct {
    a bool    // 1 字节
    b int64   // 8 字节
    c bool    // 1 字节
    d int32   // 4 字节
    e bool    // 1 字节
}
```

**内存布局**：
```
[a][padding x7][b][c][padding x3][d][e][padding x7]
= 1 + 7 + 8 + 1 + 3 + 4 + 1 + 7 = 32 bytes
```

**GOOD - 优化排列**：
```go
// 实际占用: 16 字节（无浪费）
type Good struct {
    b int64   // 8 字节
    d int32   // 4 字节
    a bool    // 1 字节
    c bool    // 1 字节
    e bool    // 1 字节
}
```

**内存布局**：
```
[      b      ][   d   ][a][c][e][padding x5]
= 8 + 4 + 1 + 1 + 1 + 5 = 20 bytes

注意：struct 整体对齐到 8 字节，
所以实际是 24 字节（64位系统）
```

### 自动化工具

```bash
# 使用 fieldalignment 检查和修复
go install golang.org/x/tools/go/analysis/passes/fieldalignment/cmd/fieldalignment@latest

fieldalignment ./path/to/package
```

## 内存管理策略

### 1. 对象池 (sync.Pool)

**适用场景**：
- 频繁分配/释放的临时对象
- 对象大小适中（不要太大或太小）
- 对象生命周期短

**最佳实践**：
```go
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 4096)  // 4KB 缓冲区
    },
}

func handleRequest() {
    buf := bufferPool.Get().([]byte)
    defer bufferPool.Put(buf)
    // ... 使用 buf
}
```

### 2. 预分配

**BAD**：
```go
func process(n int) []int {
    var result []int
    for i := 0; i < n; i++ {
        result = append(result, i)  // 多次扩容
    }
    return result
}
```

**GOOD**：
```go
func process(n int) []int {
    result := make([]int, 0, n)  // 预分配
    for i := 0; i < n; i++ {
        result = append(result, i)
    }
    return result
}
```

### 3. 避免不必要的指针

指针会增加逃逸概率，且：
- 增加 GC 扫描工作量
- 可能导致更多的缓存失效
- 增加间接访问开销

**权衡**：
- 大对象（> 4KB）: 使用指针
- 小对象（struct 几个字段）: 考虑值传递
- 需要修改原对象: 必须用指针

## 内存泄漏的常见模式

### 1. Goroutine 泄漏

```go
// BAD: goroutine 可能永远阻塞
func leak() {
    ch := make(chan int)
    go func() {
        val := <-ch  // 永远阻塞，因为没人发送
        fmt.Println(val)
    }()
    // 函数返回，但 goroutine 没退出
}
```

### 2. 未关闭的资源

```go
// BAD: 忘记关闭
func processFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    // 可能提前返回，f 没关闭
    // ...
    return nil
}

// GOOD: defer 关闭
func processFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()  // 确保关闭
    // ...
    return nil
}
```

### 3. 全局 map 持有引用

```go
// BAD: 全局 map 可能无限增长
var cache = make(map[string]*BigObject)

func addToCache(key string, obj *BigObject) {
    cache[key] = obj  // 永远不会被 GC
}
```

## 内存分析工具

### 1. runtime.ReadMemStats

```go
var stats runtime.MemStats
runtime.ReadMemStats(&stats)
fmt.Printf("HeapAlloc: %d MB\n", stats.HeapAlloc/1024/1024)
fmt.Printf("HeapObjects: %d\n", stats.HeapObjects)
```

### 2. GODEBUG=gctrace=1

```bash
# 运行时查看 GC 追踪
GODEBUG=gctrace=1 ./myapp
```

### 3. pprof heap profile

```bash
# 生成 heap profile
curl http://localhost:6060/debug/pprof/heap > heap.pprof

# 分析
go tool pprof heap.pprof
```

## 深入阅读

- [Go Memory Model](https://golang.org/ref/mem)
- [Go GC Guide](https://blog.golang.org/ismmkeynote)
- [Go 1.5 垃圾回收](https://blog.golang.org/go15gc)
