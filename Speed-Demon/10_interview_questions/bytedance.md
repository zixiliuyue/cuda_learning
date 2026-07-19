# 字节跳动 Go 面试高频 50 题

> **字节特点：** 重基础、重并发、重性能、喜欢问原理，手写代码多

---

## 🟢 Go 基础（必问）

### 1. Go 的 GMP 调度模型？
```
G: Goroutine，用户态线程
M: Machine，内核线程
P: Processor，调度上下文，本地 G 队列

工作流程：
1. M 绑定 P，从 P 的本地队列取 G 执行
2. 本地队列空了，去其他 P 偷一半 G（工作窃取）
3. G 阻塞时，M 和 P 解绑，P 找其他空闲 M 继续
4. Go 1.14+ 基于信号的异步抢占，解决死循环问题
```

---

### 2. Go GC 工作原理？
```
三色标记清除 + 写屏障：
1. 初始所有对象都是白色
2. 从根对象（栈、全局变量）开始扫描，标记为灰色
3. 遍历灰色对象，把它引用的对象标记为灰色，自己变成黑色
4. 重复直到没有灰色对象
5. 所有白色对象都是垃圾，清除

STW 阶段：
- Mark Termination：标记结束 STW，大概 10-50ms
- Sweep 是并发的，不 STW

Go 1.5+ 三色并发标记，1.8+ 混合写屏障
```

---

### 3. Channel 底层实现？
```go
type hchan struct {
    qcount   uint           // 队列里元素个数
    dataqsiz uint           // 环形队列大小
    buf      unsafe.Pointer // 数据缓冲区
    elemsize uint16         // 单个元素大小
    elemtype *_type         // 元素类型
    sendx    uint           // 发送索引
    recvx    uint           // 接收索引
    
    recvq    waitq          // 等待接收的 goroutine 队列
    sendq    waitq          // 等待发送的 goroutine 队列
    
    lock     mutex          // 锁，保护所有字段
}
```

**发送流程：**
1. 加锁
2. 有接收者直接把数据拷给接收者，唤醒接收者
3. 没接收者但 buf 有空位，把数据放 buf
4. buf 满了，把当前 goroutine 放进 sendq，阻塞等待

---

### 4. make 和 new 的区别？
```
make:
- 只能用于 slice/map/channel
- 返回的是类型本身，不是指针
- 会初始化内部结构

new:
- 可以用于任何类型
- 返回的是指针，*T
- 只是分配内存，置零值，不初始化结构

示例：
make([]int, 0) → []int
new([]int) → *[]int，值是 nil
```

---

### 5. slice 和 array 的区别？
```
数组 array:
- 固定长度，长度是类型的一部分
- 值类型，赋值会拷贝整个数组
- 作为参数传递是值拷贝，函数内改不了外面

切片 slice:
- 动态长度，长度不是类型
- 引用类型，内部是指针+长度+容量
- 作为参数传递是传引用，函数内改了外面能看到

slice 结构：
type slice struct {
    array unsafe.Pointer  // 底层数组指针
    len   int             // 长度
    cap   int             // 容量
}
```

---

### 6. slice 扩容策略？
```
Go 1.18 之前：
- len < 1024，cap × 2
- len >= 1024，cap × 1.25

Go 1.18 之后：
- len < 256，cap × 2
- len >= 256，cap = cap + (cap + 3*256)/4
- 平滑过渡，避免突然从 2 倍变 1.25 倍的拐点

扩容后底层数组是新的，所以 append 可能换地址！
```

---

### 7. map 底层实现？
```go
type hmap struct {
    count     int       // 元素个数
    B         uint8     // 桶数量：2^B 个桶
    hash0     uint32    // 哈希种子
    
    buckets    unsafe.Pointer // 桶数组指针
    oldbuckets unsafe.Pointer // 扩容时的旧桶数组
    
    extra *mapextra // 溢出桶指针
}

type bmap struct {
    tophash [8]uint8  // 8 个哈希值高 8 位
    keys    [8]keyType
    values  [8]valueType
    overflow *bmap     // 溢出桶链表
}
```

**核心点：**
- 每个桶最多存 8 个 key-value
- 哈希冲突了就链到溢出桶
- 扩容是渐进式的，不是一次搬完，每次访问搬一点

---

### 8. map 为什么不是线程安全？
```
map 内部没有锁保护，并发读写直接 panic！

为什么不做成线程安全？
1. 90% 的场景不需要并发 map，加锁影响性能
2. 真需要并发自己用 sync.RWMutex 或者 sync.Map

sync.Map 适用场景：
- 读多写少
- key 稳定，不怎么新增
```

---

### 9. defer 执行顺序？
```
1. 先声明的后执行，后声明的先执行（栈）
2. return 不是原子操作：
   return = 给返回值赋值 + defer + return 指令
   defer 在赋值之后，return 之前执行

经典题：
func f1() int { x := 5; defer func() { x++ }(); return x } → 返回 5
func f2() (x int) { defer func() { x++ }(); return 5 } → 返回 6
```

---

### 10. interface 的内部结构？
```
两种 interface：

1. iface：有方法的 interface
type iface struct {
    tab  *itab          // 方法表指针
    data unsafe.Pointer // 数据指针
}

2. eface：空 interface interface{}
type eface struct {
    _type *_type         // 类型指针
    data  unsafe.Pointer // 数据指针
}

所以 interface != nil 的条件是：_type 和 data 都不为 nil！
经典坑：
var p *int = nil
var i interface{} = p
fmt.Println(i == nil) → false！因为 _type 不是 nil！
```

---

## 🟡 并发编程（字节最看重）

### 11. Goroutine 为什么比线程轻？
```
1. 栈小：初始 2KB，最大 1GB，动态扩容缩
   线程栈默认 1-2MB，固定大小

2. 上下文切换快：
   - 线程切换：内核态，需要保存所有寄存器，开销 ~1000-1500ns
   - Goroutine 切换：用户态，只保存几个寄存器，开销 ~200ns

3. 调度开销小：
   M:N 调度，N 个 Goroutine 映射到 M 个线程
   不用每次都进内核
```

---

### 12. 怎么控制 Goroutine 并发数？
```go
方案 1：有缓冲 channel 做信号量
var sem = make(chan struct{}, 100)  // 最多 100 并发

func task() {
    sem <- struct{}{}        // 占一个坑
    defer func() { <-sem }() // 释放
    
    // 业务逻辑
}

方案 2：worker pool
jobs := make(chan Job, 1000)
for i := 0; i < 100; i++ {
    go worker(jobs)
}
```

---

### 13. Goroutine 泄露的常见场景？
```
1. Channel 没人接收，发送者永远阻塞
2. Channel 没人发送，接收者永远阻塞
3. for + select，没有退出分支
4. WaitGroup Add/Done 不匹配，Wait 永远等
5. Mutex 没解锁，其他人永远拿不到锁
6. time.After 在循环里，Timer 没回收

怎么排查：
pprof 看 goroutine profile，看 stack trace 堵在哪一行
```

---

### 14. atomic 和 mutex 怎么选？
```
能用 atomic 就用 atomic，性能好，不会死锁
但是 atomic 只能保护单个变量，操作有限

atomic 适用场景：
- 计数器
- 标志位
- 单个指针交换

mutex 适用场景：
- 多个变量
- 临界区有复杂逻辑
- 需要条件等待

性能对比（8 核）：
atomic.Add ~10ns
mutex.Lock+Unlock ~45ns
竞争激烈时差更大
```

---

### 15. 什么是伪共享？怎么避免？
```
CPU 缓存是按缓存行（64 字节）为单位的
如果两个变量在同一个缓存行，不同 CPU 核心分别修改这两个变量
会导致缓存行在核心之间来回同步，性能暴跌

怎么避免：
1. 变量之间填充 Padding，让每个变量独占一个缓存行
2. 把会被同一个 CPU 修改的变量放一起
3. 每个核心用本地变量，最后汇总

Go 1.19+ sync/atomic 里的类型已经自动处理了
```

---

### 16. sync.Map 原理？
```
空间换时间，两个 map：read 和 dirty

read map：
- 原子访问，不加锁
- 存的是稳定的 key，不怎么新增的

dirty map：
- 读写都加锁
- 存新增的 key

读取流程：
1. 先看 read 里有没有，有就直接返回，不加锁
2. read 没有，就加锁去 dirty 里找

写入流程：
1. read 里有就原子更新
2. read 没有就去 dirty 里写

删除流程：
read 里有就打个标记，不真删
等 dirty 晋升成 read 的时候一起清理

适用场景：读多写少
```

---

### 17. 手写一个线程安全的单例模式？
```go
// 经典双重检查 + atomic
type Singleton struct {
    // ...
}

var (
    instance *Singleton
    once     sync.Once  // 最推荐！用 sync.Once
)

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{}
    })
    return instance
}

// sync.Once 内部就是 atomic + CAS
// 保证只执行一次，而且并发安全
```

---

### 18. 怎么实现一个简单的协程池？
```go
type Pool struct {
    jobs    chan Job
    results chan Result
    wg      sync.WaitGroup
}

func NewPool(workerCount int) *Pool {
    p := &Pool{
        jobs:    make(chan Job, 1000),
        results: make(chan Result, 1000),
    }
    
    for i := 0; i < workerCount; i++ {
        p.wg.Add(1)
        go func() {
            defer p.wg.Done()
            for job := range p.jobs {
                p.results <- process(job)
            }
        }()
    }
    return p
}
```

---

### 19. Context 的作用？原理？
```
Context 四个作用：
1. 传递取消信号
2. 传递超时时间
3. 传递截止时间
4. 传递 KV 元数据

原理：
- 链式结构，父 Context 取消，所有子 Context 都取消
- 内部用 channel 做信号传递
- Done() 返回一个 channel，这个 channel 关闭的时候就是取消信号

注意：
- KV 只能传不可变的元数据，别传业务数据
- Context 必须作为函数第一个参数
- 不要把 Context 存在 struct 里
```

---

## 🔴 性能优化（字节必问）

### 20. 线上 CPU 高怎么排查？
```
按这个流程：

1. top/htop 看哪个进程 CPU 高
2. 确认是不是这个服务的问题，不是其他进程捣乱
3. go tool pprof 抓 CPU profile
4. top 看哪个函数占比最高
5. list 函数名看哪一行耗 CPU
6. 看是不是 GC 占比高？如果是：
   - 看 GC 频率是不是太高
   - 看 heap profile 哪分配多
7. 看是不是有死循环？
8. 看是不是正则/JSON 序列化太费？

常用命令：
go tool pprof http://ip:port/debug/pprof/profile?seconds=30
go tool trace
```

---

### 21. 内存泄漏怎么排查？
```
1. 看内存是不是持续上涨，不会降
2. 抓 heap profile，看 5 分钟前和现在的对比
3. go tool pprof -base old.prof new.prof
4. top 看哪个函数新分配的多
5. 重点查：
   - Goroutine 泄漏（最常见）
   - Goroutine 栈泄漏
   - 全局 map 只加不删
   - sync.Pool 误用
   - Goroutine 本地变量逃逸到堆，没释放

6. 用 go tool trace 看 GC 次数和时间
```

---

### 22. GC 调优有哪些手段？
```
目标：降低 GC 频率，减少 STW 时间

手段：
1. 减少内存分配：
   - 复用对象，sync.Pool
   - 预分配容量，append 前预估 cap
   - 小对象合并成大对象
   - 避免内存逃逸

2. 调整 GOGC：
   - 默认 GOGC=100，堆增长 1 倍触发 GC
   - 内存充裕可以设 GOGC=200，GC 频率降一半
   - 延迟敏感可以设 GOGC=50，GC 更频繁但单次更短

3. 控制 Goroutine 数量：
   - Goroutine 栈也是 GC 扫描对象
   - 太多 Goroutine 会增加 GC 扫描时间

4. Go 版本升级：
   - 每个版本 GC 都有优化，尽量用新版本
```

---

### 23. 内存逃逸是什么？怎么避免？
```
本应该分配在栈上的变量，因为各种原因跑到堆上了

逃逸场景：
1. 函数返回局部变量的指针
2. 变量太大，超过栈大小
3. 闭包捕获的变量
4. interface{} 调用方法
5. 发送指针到 channel

怎么看有没有逃逸：
go build -gcflags="-m"

怎么避免：
1. 不需要指针就别返回指针，返回值拷贝
2. 小对象直接传值，别传指针
3. 预分配 slice 容量，避免 append 扩容
4. 减少闭包使用
```

---

### 24. Benchmark 怎么写才准确？
```go
// ❌ 错误写法
func BenchmarkBad(b *testing.B) {
    for i := 0; i < b.N; i++ {
        // 编译器把循环优化没了！
    }
}

// ✅ 正确写法
func BenchmarkGood(b *testing.B) {
    b.ResetTimer()  // 重置时间，把初始化时间排除
    
    for i := 0; i < b.N; i++ {
        result := f()
        atomic.AddUint64(&total, result)  // 防止结果被优化掉
    }
}

注意事项：
1. ResetTimer 排除准备工作时间
2. 用 atomic 或者 channel 把结果存起来，防止编译器优化
3. b.N 是动态的，不要写死
4. 多次跑取平均值，用 benchstat 统计显著性
5. 关闭 GC 或者控制 GC 频率
```

---

### 25. 怎么避免 Goroutine 调度开销？
```
1. 控制 Goroutine 数量：
   - 不是越多越快，一般 CPU 核数 × 2-10 倍
   - 太多 Goroutine 上下文切换开销大

2. 绑定 CPU：
   - runtime.LockOSThread() 把 M 绑定到 CPU 核心
   - 适合计算密集型任务

3. 减少阻塞：
   - 用非阻塞 IO
   - 批量操作减少系统调用

4. 避免频繁创建销毁 Goroutine：
   - 用 worker pool 复用 Goroutine
```

---

## 🟣 系统设计（高级岗）

### 26. 怎么设计一个高并发的计数器？
```
需求：支持百万级 QPS 的计数

方案 1：单机版
- 每个 CPU 核心一个本地计数器，原子操作
- 读的时候把所有核心的数加起来
- 避免伪共享，每个计数器占一个缓存行
- 性能：单机能到千万级 QPS

方案 2：分布式版
- Redis INCR，但是单 Redis 瓶颈
- 分片：按用户 ID 哈希到不同 Redis 实例
- 批量：本地攒一批再同步到 Redis
- 最终一致：不要求绝对实时，有延迟没关系
```

---

### 27. 怎么实现限流？有哪些算法？
```
常用四种限流算法：

1. 计数器（固定窗口）：
   - 最简单，每秒允许 N 个请求
   - 问题：边界突刺，0.9-1.0s 100个，1.0-1.1s 100个
     0.95-1.05s 这 100ms 有 200 个请求

2. 滑动窗口：
   - 把 1 秒分成 10 个格子，每个格子 100ms
   - 每次请求把当前格子计数 +1
   - 统计最近 10 个格子的总和
   - 解决了边界问题，但是实现复杂

3. 令牌桶（最常用）：
   - 桶里有 N 个令牌，每秒补充 R 个
   - 每个请求拿一个令牌，拿到就过，拿不到就限流
   - 优点：平滑，允许一定突发流量
   - Go 官方实现：golang.org/x/time/rate

4. 漏桶：
   - 请求来了先进水桶，水桶以固定速率漏水处理
   - 优点：绝对平滑，不允许突刺
   - 缺点：不抗突发，适合下游能力固定的场景
```

---

### 28. 缓存穿透、击穿、雪崩区别？
```
穿透：查不存在的数据，每次都打到 DB
   - 布隆过滤器：把所有存在的 key 放布隆里
   - 空值缓存：DB 查不到也缓存个空值

击穿：某一个热点 key 失效瞬间，大量请求打 DB
   - 互斥锁：只有一个 goroutine 去查 DB 写缓存
   - 永不过期：逻辑过期，后台异步更新

雪崩：大量 key 同时过期，DB 压力陡增
   - 过期时间加随机抖动，不要都整点过期
   - 多级缓存：本地缓存 + 分布式缓存
```

---

### 29. 怎么实现分布式锁？
```
Redis 实现分布式锁：

1. 加锁：SET key value NX EX 30
   - NX：key 不存在才设置成功
   - EX 30：30 秒自动过期，防止死锁
   - value 要是唯一标识，比如 UUID，防止解别人的锁

2. 解锁：Lua 脚本原子操作
   if redis.call('get', KEYS[1]) == ARGV[1] then
       return redis.call('del', KEYS[1])
   else
       return 0
   end
   - 必须原子，不能先 GET 再 DEL！

3. 续期：看门狗机制
   - 业务没执行完，定期把锁的过期时间延长

注意事项：
- 不要用 RedLock，太复杂，99% 场景不需要
- 锁的过期时间要大于业务执行时间
- 一定要有超时，不能无限等
```

---

### 30. 消息队列怎么选型？Kafka vs RocketMQ vs RabbitMQ
```
Kafka：
- 优点：吞吐量最高，百万级 QPS，生态好
- 缺点：消息可能乱序，不支持延迟消息，功能少
- 适用：日志、大数据、高吞吐不在乎延迟的场景

RocketMQ：
- 优点：功能全，事务消息、延迟消息、顺序消息
- 缺点：吞吐量比 Kafka 低一点，十万级 QPS
- 适用：业务系统，需要各种高级特性

RabbitMQ：
- 优点：最灵活，路由功能最强，一致性好
- 缺点：吞吐量最低，万级 QPS，Erlang 难维护
- 适用：路由复杂，吞吐量要求不高的场景

字节内部主要用 Kafka 和自研 MQ
```

---

## 📝 手写代码题（字节必写）

### 31. 用两个 channel 实现交替打印 1-100
```go
func main() {
    ch1, ch2 := make(chan struct{}), make(chan struct{})
    
    go func() {
        for i := 1; i <= 100; i += 2 {
            <-ch1
            fmt.Println(i)
            ch2 <- struct{}{}
        }
    }()
    
    go func() {
        for i := 2; i <= 100; i += 2 {
            <-ch2
            fmt.Println(i)
            ch1 <- struct{}{}
        }
    }()
    
    ch1 <- struct{}{}
    select {}
}
```

---

### 32. 实现一个超时控制的函数
```go
func WithTimeout(f func() error, timeout time.Duration) error {
    done := make(chan error, 1)
    
    go func() {
        done <- f()
    }()
    
    select {
    case err := <-done:
        return err
    case <-time.After(timeout):
        return errors.New("timeout")
    }
}
```

---

### 33. 实现一个简单的 worker pool
```go
type Pool struct {
    jobs chan func()
    wg sync.WaitGroup
}

func NewPool(workerCount int) *Pool {
    p := &Pool{jobs: make(chan func(), 1000)}
    
    for i := 0; i < workerCount; i++ {
        p.wg.Add(1)
        go func() {
            defer p.wg.Done()
            for job := range p.jobs {
                job()
            }
        }()
    }
    return p
}

func (p *Pool) Submit(f func()) {
    p.jobs <- f
}

func (p *Pool) Wait() {
    close(p.jobs)
    p.wg.Wait()
}
```

---

### 34. 实现一个线程安全的 map
```go
type SafeMap struct {
    mu sync.RWMutex
    m  map[string]interface{}
}

func NewSafeMap() *SafeMap {
    return &SafeMap{m: make(map[string]interface{})}
}

func (sm *SafeMap) Get(key string) interface{} {
    sm.mu.RLock()
    defer sm.mu.RUnlock()
    return sm.m[key]
}

func (sm *SafeMap) Set(key string, val interface{}) {
    sm.mu.Lock()
    defer sm.mu.Unlock()
    sm.m[key] = val
}
```

---

### 35. 实现一个单例模式
```go
// 最推荐用 sync.Once
type Singleton struct{}

var (
    instance *Singleton
    once     sync.Once
)

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{}
    })
    return instance
}
```

---

## 🚀 场景题（字节最爱问）

### 36. 100 万请求同时打过来，怎么处理？
```
按这个思路说：

1. 最上层：限流
   - 每个用户每秒限 100 次，超过直接返回
   - 总 QPS 限 10 万，超过直接降级

2. 第二层：缓存
   - 热点数据放本地缓存，不用打 Redis
   - 热点 key 复制多份，避免热 key 问题

3. 第三层：削峰填谷
   - 非核心请求进 MQ，慢慢消费
   - 同步改异步，不用等结果的直接返回

4. 第四层：降级
   - 非核心功能直接关闭，返回默认值
   - 比如评论、推荐先关了，保证下单能用

5. 第五层：扩容
   - K8s 自动扩容，流量大就加机器
   - 数据库加从库，Redis 加集群
```

---

### 37. 接口 P99 延迟太高，怎么排查？
```
排查流程：

1. 先看监控：
   - 是所有接口都慢还是某几个慢？
   - 是什么时候开始变慢的？那个时间点发了什么变更？
   - CPU/内存/DB/Redis 哪个指标先异常？

2. 慢在哪个环节：
   - 打日志看每个环节耗时
   - DB 慢不慢？Redis 慢不慢？RPC 调用慢不慢？
   - 是等待时间长还是处理时间长？

3. 抓 profile：
   - CPU profile 看哪个函数占比高
   - trace 看 Goroutine 调度、GC 时间
   - mutex profile 看锁竞争
   - goroutine profile 看有没有阻塞

4. 常见原因：
   - 没加索引，全表扫描
   - 缓存命中率低，都打到 DB
   - 锁竞争太激烈
   - GC 太频繁
   - 下游依赖慢
   - 网络抖动
```

---

### 38. 怎么设计一个短链接系统？
```
核心点：

1. 发号器：
   - 雪花算法生成唯一 ID
   - 62 进制编码：0-9a-zA-Z
   - 10 位长度就能支持 8 亿个短链

2. 存储：
   - MySQL 存长链接 → 短链接映射
   - 短链接哈希做索引，查的时候快
   - Redis 做缓存，热点短链接不用查 DB

3. 跳转：
   - 302 临时重定向，不要用 301 永久
   - 因为后面可能要改统计数据

4. 统计：
   - 访问日志异步写，不要同步影响跳转
   - 埋点进 Kafka，后台消费统计

5. 防刷：
   - 同一个 IP 1 分钟最多创建 10 个
   - 黑名单域名不让跳转
```

---

### 39. 怎么设计一个秒杀系统？
```
核心思路：把请求层层过滤，最后落到 DB 的越少越好

1. 最上层：
   - 页面静态化，CDN 缓存
   - 按钮置灰，防止用户重复点
   - 前端生成 UUID，同一个用户同一个商品只能点一次

2. Nginx 层：
   - 限流：每个 IP 1 秒 10 次
   - 恶意 IP 直接拉黑

3. 业务层：
   - 库存预扣到 Redis，不用查 DB
   - Lua 脚本原子操作：查库存 → 减库存
   - 一人一单，用户 ID + 商品 ID 做唯一键
   - 减库存成功就发 MQ，直接返回排队中

4. MQ 异步消费：
   - 慢慢消费，真正创建订单，扣 DB 库存
   - 失败了就回补 Redis 库存

5. 结果查询：
   - 用户轮询或者 Websocket 通知结果
   - 不用同步等结果，用户体验好

关键：绝对不能让请求打到 DB！
100 万请求最后能落到 MQ 的就几千个，DB 完全扛得住
```

---

### 40. 线上服务 OOM 了，怎么处理？
```
处理流程：

1. 先止损：
   - 把这个实例从线上摘下来，不要让流量再打进来
   - 还能救就先别重启，留着现场

2. 保存现场：
   - 抓 heap profile：go tool pprof .../heap
   - 抓 goroutine profile
   - 把 core dump 保存下来（如果有）
   - 监控截图保存

3. 重启服务先恢复线上

4. 事后排查：
   - 看 OOM 前内存曲线是慢慢涨还是突然爆
   - 慢慢涨：大概率是 Goroutine 泄漏 / 内存泄漏
   - 突然爆：大概率是某次请求加载了太多数据
   - 看 profile 哪个函数分配内存多
   - 看 Goroutine 数量是不是异常多
   - 看是不是加载了大文件 / 大 SQL 查询没分页

5. 预防：
   - 加内存使用率告警，80% 就告警
   - 业务层做限制，比如一次最多查 1000 条
   - Go 版本升级，新版本内存管理更好
```

---

## 💡 其他高频题

### 41. Go 里的 0 值有用法？
```
零值的好处：不用初始化，直接能用

例子：
var mu sync.Mutex → 零值就可以直接 Lock，不用 New
var wg sync.WaitGroup → 零值就可以 Add/Done
var ch chan struct{} → 零值是 nil，但是不会 panic
var m map[string]int → 读 nil map 不会 panic，写才会

面试加分：这是 Go 的设计哲学，让零值也有用
```

---

### 42. for range slice 是值拷贝还是引用？
```
for range 循环里的 value 是值拷贝！
所以改 value 不会改原 slice 里的值！

❌ 错误：
for _, v := range s {
    v.Name = "new"  // 改的是拷贝，没用！
}

✅ 正确：
for i := range s {
    s[i].Name = "new"  // 直接改原 slice
}
```

---

### 43. nil slice 和空 slice 区别？
```
var s []int → nil slice，指针是 nil，len=0，cap=0
s := make([]int, 0) → 空 slice，指针不是 nil，len=0，cap=0

使用上几乎没区别，都可以 append，len 都是 0
但是 JSON 序列化的时候：
nil slice → null
空 slice → []
```

---

### 44. 为什么要避免闭包捕获循环变量？
```go
❌ 经典坑：
for i := 0; i < 5; i++ {
    go func() {
        fmt.Println(i)  // 打印的全是 5！
    }()
}

原因：所有 Goroutine 捕获的是同一个变量 i 的引用
等 Goroutine 跑起来的时候循环已经结束了，i 已经是 5

✅ 解决 1：参数传进去
for i := 0; i < 5; i++ {
    go func(i int) {
        fmt.Println(i)
    }(i)
}

✅ 解决 2：循环内局部变量
for i := 0; i < 5; i++ {
    i := i
    go func() {
        fmt.Println(i)
    }()
}
```

---

### 45. Go 的错误处理最佳实践？
```
1. 不要忽略错误，每个 error 都要处理
2. 不推荐 panic，除非是初始化失败这种程序启动不了的
3. 错误要 wrap，不要直接 return err，要加上下文
4. 用 errors.Is/As 判断错误类型，不要 ==
5. 自定义错误要包含调用栈，方便排查

例子：
if err != nil {
    return fmt.Errorf("get user %d failed: %w", userID, err)
}
```

---

### 46. Go 版本升级要注意什么？
```
1. 看 Release Notes，每个版本都有不兼容改动
2. 先在测试环境跑，所有单元测试过了再上
3. 灰度发布，先上几台观察
4. 注意 GC 行为变化，每个版本 GC 都有调整
5. 标准库函数有没有废弃或者行为变化
6. Go Modules 依赖有没有冲突
```

---

### 47. Go 怎么做到跨平台编译？
```
Go 编译器自带交叉编译，不需要其他工具：

CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build

原理：
- Go 的 Runtime 自己实现了系统调用，不依赖 libc
- 只要 CGO_ENABLED=0，编译出来的就是静态链接
- 放到对应操作系统直接跑，不需要依赖
```

---

### 48. 为什么 Go 的编译速度快？
```
1. 没有头文件，依赖分析在编译期一次搞定
2. 没有模板，没有 C++ 那种实例化开销
3. 没有重载，没有虚函数，编译期不用做复杂推导
4. 编译目标是机器码，不是字节码，不需要解释执行
5. 并行编译，Go 编译器本身就支持多核心并行
```

---

### 49. Go 和其他语言对比，优势在哪？
```
对比 C++：
- 开发效率高，GC 不用手动管理内存
- 编译速度快，C++ 编译几分钟，Go 几秒
- 没有那么多复杂特性，代码可读性好

对比 Java：
- 启动快，毫秒级，Java 要几秒
- 内存占用小，Go 服务几十 MB，Java 几百 MB
- 没有预热，启动就能跑满性能
- 不需要虚拟机，直接本机跑

对比 Python：
- 性能好 10-50 倍
- 静态类型，编译期发现错误
- 并发支持好，GIL 问题

总结：Go 的定位是云计算时代的 C 语言
简单，高效，适合写基础设施和高并发服务
```

---

### 50. 你做过的最有成就感的性能优化是什么？
```
这是字节必问题！按 STAR 法则说：

S (Situation)：
背景：我们有个服务大促的时候 CPU 80%，P99 500ms，扛不住预期流量

T (Task)：
我负责排查和优化，目标是 P99 降到 200ms 以内，CPU 降到 50%

A (Action)：
1. 先抓 profile，发现 JSON 序列化占了 40% CPU
   - 把标准库 json 换成 sonic/ffjson，性能提升 2 倍
2. 发现有大量小对象分配，GC 占了 20%
   - 用 sync.Pool 复用对象，GC 次数降了一半
3. 发现锁竞争严重，一个全局锁占了 30%
   - 改成分片锁，把锁拆成 32 份，锁竞争降了 90%
4. 还有大量 N+1 查询，循环查 DB
   - 改成 IN 查询批量查，DB 压力降了 80%

R (Result)：
最后 P99 从 500ms 降到 120ms，CPU 从 80% 降到 35%
大促的时候机器扛住了 5 倍流量，没有告警
后来把这次优化的经验沉淀成了团队的最佳实践

💡 关键点：一定要有具体数字！别只说变快了！
```

---

## ✅ 字节面试注意事项

1. **基础一定要牢**：GMP/GC/Map/Slice 这些原理，每一个都要能手写出来画出来
2. **并发一定要会**：字节是 Go 重度用户，并发题是必问的
3. **性能优化要能说清楚**：怎么排查，用了什么工具，优化了多少，有数字
4. **手写代码别怂**：字节很看重编码能力，一般会让你在线写 1-2 题
5. **场景题要思路清晰**：从简到繁，从上层到下层，层层递进
6. **一定要说自己的思考**：不要只背答案，说你为什么这么选，有什么trade-off
