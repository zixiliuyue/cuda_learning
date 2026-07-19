# Memory Profiling 实战

## 实验目标

1. 学习如何使用 pprof 分析内存分配
2. 理解堆内存和栈内存的区别
3. 发现和修复内存泄漏
4. 掌握内存优化技术

## 运行方式

### 1. Server 模式

```bash
go run main.go
```

### 2. Leak Demo 模式

```bash
go run main.go -mode leak -heapprofile leak.pprof
```

### 3. Benchmark 模式

```bash
go run main.go -mode bench
```

## 内存 Profile 类型

### inuse_space (默认)
- 当前正在使用的内存
- 用于发现内存泄漏

### alloc_space
- 累计分配的内存（包括已释放）
- 用于发现分配热点

```bash
# 查看当前使用的内存
go tool pprof http://localhost:6060/debug/pprof/heap

# 查看累计分配
go tool pprof -alloc_space http://localhost:6060/debug/pprof/heap
```

## 实验步骤

### 步骤 1: 发现分配热点

1. 启动服务: `go run main.go`
2. 制造一些分配:
```bash
while true; do curl http://localhost:6060/alloc; done
```
3. 采集 heap profile:
```bash
go tool pprof http://localhost:6060/debug/pprof/heap
```
4. 查看 top:
```bash
(pprof) top
```

**预期结果**: `naiveConcat` 应该是分配热点

### 步骤 2: 对比两种字符串拼接方式

采集 profile，然后比较:

```bash
(pprof) list naiveConcat
(pprof) list builderConcat
```

**思考**:
- 为什么 naiveConcat 分配这么多？
- strings.Builder 做了什么优化？

### 步骤 3: 观察内存泄漏

1. 制造泄漏:
```bash
for i in {1..100}; do curl http://localhost:6060/leak; done
```

2. 采集 profile:
```bash
go tool pprof http://localhost:6060/debug/pprof/heap
```

3. 查看内存使用:
```bash
(pprof) top
(pprof) list leakHandler
```

**预期结果**: 应该能看到 `leakHandler` 占用了大量内存

### 步骤 4: 使用对象池对比

```bash
# 没有使用 pool
ab -n 1000 -c 10 http://localhost:6060/alloc

# 使用 pool
ab -n 1000 -c 10 http://localhost:6060/pool
```

分别采集 profile 并对比分配数量。

## 内存优化技巧

### 1. 预分配切片

**BAD**:
```go
func appendNoCap(n int) []int {
    var result []int
    for i := 0; i < n; i++ {
        result = append(result, i)
    }
    return result
}
```

**GOOD**:
```go
func appendWithCap(n int) []int {
    result := make([]int, 0, n)
    for i := 0; i < n; i++ {
        result = append(result, i)
    }
    return result
}
```

**性能差异**:
- 无预分配: O(log n) 次扩容，大量内存拷贝
- 预分配: 一次分配，无拷贝

### 2. 使用 strings.Builder

**BAD**:
```go
func naiveConcat(n int) string {
    result := ""
    for i := 0; i < n; i++ {
        result += "hello "  // O(n²) 时间
    }
    return result
}
```

**GOOD**:
```go
func builderConcat(n int) string {
    var builder strings.Builder
    builder.Grow(n * 6)  // 预分配
    for i := 0; i < n; i++ {
        builder.WriteString("hello ")
    }
    return builder.String()
}
```

### 3. sync.Pool 复用对象

```go
var bufPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 4096)
    },
}

func process() {
    buf := bufPool.Get().([]byte)
    defer bufPool.Put(buf)
    // 使用 buf
}
```

**注意事项**:
- Pool 中的对象随时可能被清理
- Get 返回的对象可能是任意状态
- 适合用于无状态的临时对象

### 4. 小对象合并

**BAD**:
```go
type Point struct {
    X int32
    Y int32
    Z int32
}  // 12 bytes

type Particle struct {
    Pos Point
    Vel Point
    Color [3]byte
}
```

**GOOD**: 考虑内存对齐优化

## 内存泄漏常见模式

### 1. 全局缓存无界

```go
// BAD: 无限增长
var cache = make(map[string][]byte)

func add(key string, data []byte) {
    cache[key] = data  // 永远不会被删除
}

// GOOD: 使用带淘汰机制的 cache
import "github.com/hashicorp/golang-lru"
cache, _ := lru.New(1024)  // 最多 1024 项
```

### 2. Goroutine 泄漏（见下一节）

### 3. 未关闭的资源

```go
// BAD
func readFile(path string) ([]byte, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    // 忘记 close！
    return io.ReadAll(f)
}

// GOOD
func readFile(path string) ([]byte, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer f.Close()  // 确保关闭
    return io.ReadAll(f)
}
```

### 4. 停止的 timer 未释放

```go
// BAD
timer := time.NewTimer(5 * time.Second)
select {
case <-timer.C:
    fmt.Println("timeout")
case <-done:
    // timer 没被停止！goroutine 泄漏
}

// GOOD
timer := time.NewTimer(5 * time.Second)
defer timer.Stop()  // 确保停止
select {
case <-timer.C:
    fmt.Println("timeout")
case <-done:
}
```

## 进阶实验

### 1. 逃逸分析

```bash
# 查看逃逸分析报告
go build -gcflags="-m -m" . 2>&1 | grep escape
```

**常见逃逸原因**:
- 发送到 channel
- 被闭包捕获
- 赋值给指针
- interface 方法调用

### 2. GC 追踪

```bash
# 运行时观察 GC 行为
GODEBUG=gctrace=1 go run main.go
```

### 3. 对比优化前后

```bash
# 优化前
go test -bench=. -memprofile=old.mem -benchmem

# 优化后
go test -bench=. -memprofile=new.mem -benchmem

# 对比
go tool pprof -base old.mem new.mem
```

## pprof 内存分析命令

```bash
# 常用命令
top           # 看热点
top -alloc_objects  # 按分配对象数排序
list func     # 源码级分析
web           # 调用图
png           # 生成 PNG
diff base     # 对比两个 profile

# 切换视图
(pprof) heap   # inuse_space
(pprof) alloc  # alloc_space
(pprof) objects # inuse_objects
(pprof) allocs  # alloc_objects
```

## 真实案例分析

### 案例 1: 大量小对象分配

**问题现象**:
- GC 频繁
- 吞吐量上不去
- heap profile 显示大量 runtime.mallocgc

**解决方法**:
- 预分配
- 对象池
- 小对象合并为 struct

### 案例 2: 内存持续增长

**问题现象**:
- 内存每天增长 1GB
- heap profile 显示增长不明显

**排查过程**:
1. 检查 goroutine 数量 → 发现持续增长
2. goroutine profile 看到大量 goroutine 卡在 channel 接收
3. 定位到某段代码忘记关闭 channel

**教训**: 内存泄漏不一定是堆内存泄漏，也可能是 goroutine 泄漏

## 延伸阅读

- [Go Memory Model](https://golang.org/ref/mem)
- [Go GC Guide](https://blog.golang.org/ismmkeynote)
- [Taming the Go Garbage Collector](https://about.sourcegraph.com/go/gophercon-2019-taming-the-go-garbage-collector)
- [sync.Pool 源码分析](https://www.ardanlabs.com/blog/2017/05/language-mechanics-on-storage-class.html)
