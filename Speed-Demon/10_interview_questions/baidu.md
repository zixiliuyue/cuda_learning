# 百度 Go 面试高频 50 题

> **百度特点：** 重基础原理、重算法、重工程实践、问得深、喜欢追根问底

---

## 🟢 Go 基础（必问）

### 1. Go 的 GMP 调度模型？
```
G: Goroutine，用户态轻量级线程
M: Machine，内核线程，真正执行代码的实体
P: Processor，调度器上下文，M 需要绑定 P 才能调度 G

核心数据结构：
- P 本地队列：每个 P 有自己的本地 G 队列，无锁访问，性能好
- 全局队列：所有 P 共享，需要加锁访问
- 调度器：schedt 结构，维护全局状态

调度流程：
1. M 绑定 P，从 P 本地队列取 G 执行
2. 本地队列空了，去其他 P 偷一半 G（工作窃取）
3. 所有 P 都没 G 了，去全局队列拿
4. G 系统调用阻塞时，M 和 P 解绑，P 找其他空闲 M
5. 系统调用返回时，如果没 P 了，就把 G 放全局队列

Go 1.14+ 基于信号的异步抢占：
- 每 10ms 发一个 SIGURG 信号
- 收到信号把 G 抢占下来，放到全局队列
- 解决死循环 goroutine 占着 CPU 不让的问题
```

---

### 2. Go GC 三色标记原理？写屏障？
```
三色标记清除算法：

1. 初始状态：所有对象都是白色
2. 从根对象（栈、全局变量、寄存器）开始扫描，标记为灰色
3. 遍历灰色对象，把它引用的对象标记为灰色，自己变成黑色
4. 重复步骤 3，直到没有灰色对象
5. 所有白色对象都是垃圾，清除

混合写屏障（Go 1.8+）：
删除屏障 + 插入屏障结合

删除屏障：
- 删除引用时，把被删除的对象标记为灰色
- 保证被删除的对象在这轮 GC 中不会被误删

插入屏障：
- 新插入的引用，把新引用的对象标记为灰色
- 保证新插入的对象不会漏标

STW 阶段：
- Mark Termination：标记结束 STW，重新扫描栈
- 大概 10-50ms，和 goroutine 数量正相关
```

---

### 3. Channel 底层结构？发送/接收流程？
```go
type hchan struct {
    qcount   uint           // 队列中元素个数
    dataqsiz uint           // 环形队列大小
    buf      unsafe.Pointer // 数据缓冲区
    elemsize uint16         // 单个元素大小
    elemtype *_type         // 元素类型
    sendx    uint           // 发送索引
    recvx    uint           // 接收索引
    
    recvq    waitq          // 等待接收的 goroutine 队列
    sendq    waitq          // 等待发送的 goroutine 队列
    
    lock     mutex          // 全局锁
}

发送流程：
1. 加锁
2. recvq 不为空，直接把数据拷给第一个等待的接收者，唤醒接收者
3. recvq 空但 buf 有空位，把数据放 buf，sendx++，qcount++
4. buf 满了，把当前 G 塞进 sendq，解锁，等待被唤醒

接收流程：
1. 加锁
2. sendq 不为空，直接从发送者那拷数据，唤醒发送者
3. sendq 空但 buf 有数据，从 buf 拿，recvx++，qcount--
4. buf 空了，把当前 G 塞进 recvq，解锁，等待被唤醒

关闭 channel：
- 唤醒所有 recvq 等待的接收者，收到零值
- 唤醒所有 sendq 等待的发送者，直接 panic
```

---

### 4. make 和 new 的区别？
```
make:
- 只能用于 slice/map/channel 三种类型
- 返回的是类型本身，不是指针
- 会初始化内部结构，分配底层内存
- 比如 make([]int, 0, 10) 返回 []int

new:
- 可以用于任何类型
- 返回的是指针，*T
- 只是分配内存，置零值，不初始化结构
- 比如 new(int) 返回 *int，值是 0

经典面试题：
var m *map[string]int = new(map[string]int)
*m["a"] = 1  // panic！因为 new 只是分配了指针，没初始化 map 内部结构

正确写法：
m := make(map[string]int)
m["a"] = 1
```

---

### 5. slice 的扩容策略？
```
Go 1.18 之前：
- len < 1024，cap × 2
- len >= 1024，cap × 1.25

Go 1.18 之后，平滑过渡：
- len < 256，cap × 2
- len >= 256，cap = cap + (cap + 3*256)/4

为什么改？
- 原来 1024 那个拐点太突兀
- 1023 个元素扩容到 2048，1024 个元素只扩容到 1280
- 新算法平滑增长，没有明显拐点

注意：
- 扩容后底层数组是新的，所以 append 可能返回新地址
- 多个 slice 共享同一个底层数组时，改一个其他的也会变
```

---

### 6. map 是怎么实现的？
```
hmap 结构：
type hmap struct {
    count     int       // 元素个数
    B         uint8     // 桶数量：2^B 个桶
    hash0     uint32    // 哈希种子，防碰撞攻击
    
    buckets    unsafe.Pointer // 桶数组
    oldbuckets unsafe.Pointer // 扩容时的旧桶数组
    
    nevacuate uintptr        // 搬迁进度
    
    extra *mapextra          // 溢出桶指针
}

bmap 桶结构：
type bmap struct {
    tophash [8]uint8       // 8 个 key 的哈希值高 8 位
    keys    [8]keyType     // 8 个 key
    values  [8]valueType   // 8 个 value
    overflow *bmap         // 溢出桶链表
}

查找流程：
1. key 算哈希值 hash
2. hash & (1<<B -1) 算在哪个桶
3. 遍历桶的 tophash，比较高 8 位
4. 高 8 位一样，再比较完整的 key
5. 找到了返回，没找到去溢出桶继续找

渐进式扩容：
- 不是一次把所有桶都搬完，每次访问 map 搬 1-2 个桶
- oldbuckets 指向旧桶，慢慢搬完才置空
- 这样不会因为扩容卡住用户请求
```

---

### 7. Go 的内存分配器原理？
```
TCMalloc 思路，多级缓存：

层级结构：
1. mcache：每个 P 独享的本地缓存，无锁
   - 每个大小级别都有一个 span 链表
   - 小对象直接从这分配，超快

2. mcentral：全局中心缓存，按大小分
   - 每个大小级别有一个 mcentral
   - mcache 没了去 mcentral 拿一批
   - 需要加锁

3. mheap：堆内存，向操作系统申请的大内存
   - mcentral 也没了，去 mheap 申请新的 span
   - 大对象（>32KB）直接从 mheap 分配

大小规格：
- 67 种大小级别，从 8B 到 32KB
- 比如 8B, 16B, 24B, ..., 32KB
- 每个级别对应一种 span

为什么快：
- 大部分小对象都在 P 的 mcache 分配，无锁
- 按大小分块，减少内存碎片
- 每个 P 自己的缓存，没有锁竞争
```

---

### 8. interface 的内部结构？
```
两种 interface：

1. iface：有方法的 interface
type iface struct {
    tab  *itab          // 方法表指针
    data unsafe.Pointer // 数据指针
}

itab 结构：
- 接口类型指针
- 具体类型指针
- 方法哈希
- 方法函数指针数组

2. eface：空 interface interface{}
type eface struct {
    _type *_type         // 类型指针
    data  unsafe.Pointer // 数据指针
}

经典面试题：interface == nil 的条件
var i interface{}
fmt.Println(i == nil)  // true，_type 和 data 都是 nil

var p *int = nil
var i interface{} = p
fmt.Println(i == nil)  // false！_type 不是 nil！

所以 interface == nil 必须 _type 和 data 都是 nil！
```

---

### 9. defer 的执行顺序？return 和 defer 的关系？
```
defer 执行顺序：
- 先声明的后执行，后声明的先执行
- 相当于栈，先进后出

return 不是原子操作！
return = 给返回值赋值 → 执行所有 defer → 真正 return 指令

经典面试题：
func f1() int {
    x := 5
    defer func() { x++ }()
    return x
} → 返回 5，因为返回值已经复制好了，x++ 不影响

func f2() (x int) {
    defer func() { x++ }()
    return 5
} → 返回 6，因为 x 就是返回值本身，x++ 直接改返回值

func f3() int {
    x := 5
    defer func(p int) { p++ }(x)
    return x
} → 返回 5，defer 参数是值拷贝，x++ 改的是拷贝
```

---

### 10. 什么是内存逃逸？怎么分析？
```
内存逃逸：本该在栈上的变量，因为各种原因跑到堆上了

逃逸 = 堆分配 = GC 要扫 = 性能差

常见逃逸场景：
1. 函数返回局部变量的指针
2. 变量太大，栈装不下（栈一般只有几 MB）
3. 变量大小编译期不确定，比如 make([]int, n)
4. 闭包捕获的变量
5. 发送指针到 channel
6. interface{} 调用方法

怎么分析逃逸：
go build -gcflags="-m -l"
输出 "moved to heap" 就是逃逸了

优化思路：
- 小对象直接传值，别传指针
- 返回值拷贝比返回指针好
- 预分配容量，避免 append 过程中多次分配
- 能不用闭包就不用
- 能不用 interface 就用具体类型
```

---

## 🟡 并发编程（百度重点）

### 11. Goroutine 为什么比线程轻？
```
1. 栈小，动态伸缩
- Goroutine 栈初始只有 2KB
- 不够了自动扩容，最大 1GB
- 用多少占多少，不浪费

2. 上下文切换快
- 线程切换：内核态，需要保存所有寄存器，开销 ~1000-1500ns
- Goroutine 切换：用户态，只保存几个寄存器，开销 ~200ns

3. 调度开销小
- M:N 调度，N 个 Goroutine 映射到 M 个线程
- 不用每次进内核，用户态调度

4. 资源占用小
- 一个 Goroutine 栈 + 元数据几 KB
- 一台机器可以跑几十万甚至几百万 Goroutine
- 线程最多几千个
```

---

### 12. Goroutine 泄露的常见场景？怎么排查？
```
常见泄漏场景：
1. Channel 没人接收，发送者永远阻塞
2. Channel 没人发，接收者永远阻塞
3. for + select 没有退出分支，永远循环
4. Context 没 cancel，子 goroutine 永远等
5. time.After 在循环里，Timer 没被回收
6. WaitGroup Add/Done 不匹配，Wait 永远等
7. Mutex 没解锁，其他人永远拿不到锁

怎么排查：
1. pprof 看 goroutine profile
   go tool pprof http://ip:port/debug/pprof/goroutine
2. 看 goroutine 数量是不是异常多
3. 看 stack trace，都堵在哪个函数哪一行
4. debug.WriteHeapDump 把所有 goroutine 栈打出来

怎么预防：
- 每个 go 启动的 goroutine，必须有明确的退出方式
- 所有 channel 操作都要有超时
- 用 Context 控制生命周期
- 加 goroutine 数量监控
```

---

### 13. atomic 和 mutex 怎么选？
```
能用 atomic 就用 atomic，性能好，不会死锁

性能对比（8 核）：
atomic.AddInt64: ~10ns
mutex.Lock+Unlock: ~45ns

atomic 适用场景：
- 计数器
- 标志位
- 单个指针交换
- 简单的状态机
- 一个变量就能搞定的场景

mutex 适用场景：
- 多个变量需要原子操作
- 临界区有复杂逻辑
- 需要条件等待
- 临界区执行时间长，空转不如等锁

注意：
- atomic 虽然快，但不要滥用
- 复杂场景用 mutex 可读性更好，不容易出错
- sync/atomic 包的函数要小心，顺序一致性问题
```

---

### 14. sync.Map 原理？适用场景？
```
sync.Map 空间换时间，两个 map：read 和 dirty

read map：
- 原子访问，不加锁
- 存稳定的 key，不怎么新增的
- 读命中直接返回，超快

dirty map：
- 读写都加锁
- 存新增的 key
- read 没命中才去 dirty 找

miss 计数：
- miss 超过 len(dirty) 一半，把 dirty 升级成新的 read
- dirty 置空，下次新增再建

删除：
- read 里有就打个标记，不真删
- 等 dirty 升级成 read 的时候一起清理

性能对比：
- 读多写少：sync.Map 比 RWMutex+map 快 5 倍以上
- 写多读少：差不多，甚至可能更慢

适用场景：
- 读多写少
- key 比较稳定，不怎么新增
- 并发高的缓存场景

不适用场景：
- 写很多，key 一直在变
- 需要遍历所有 key
```

---

### 15. Context 原理？怎么用？
```
Context 四个作用：
1. 传递取消信号
2. 传递超时时间
3. 传递截止时间
4. 传递 KV 元数据

Context 链式结构：
context.Background() → 根 Context
    ↓ context.WithCancel()
    cancelCtx
        ↓ context.WithTimeout()
        timerCtx

取消流程：
1. 调用 cancel() 函数
2. cancelCtx 加锁
3. 关闭自己的 done channel
4. 遍历所有 children，一个个 cancel
5. 把自己从父 Context 的 children 里删掉
6. 解锁

最佳实践：
1. Context 必须作为函数第一个参数，名字叫 ctx
2. 不要把 Context 存在 struct 里，要显式传递
3. Value 只能传请求级别的元数据，不要传业务参数
4. 不要传 nil Context，要用 context.TODO() 当占位
5. 调用 cancel 是必须的，不然 goroutine 泄漏！
```

---

### 16. select 的原理？多个 case 同时 ready 怎么选？
```
select 是 Go runtime 实现的，不是系统调用

编译期处理：
- 0 个 case：直接阻塞，永远等
- 1 个 case：等价于直接 channel 操作
- 2 个及以上 case：走 runtime.selectgo

selectgo 流程：
1. 对所有 case 按 channel 地址排序（加锁顺序，防止死锁）
2. 按顺序把所有 channel 都加锁
3. 先快速检查一遍有没有已经 ready 的 case
4. 有就直接处理这个 case，解锁返回
5. 都没 ready，把当前 G 注册到所有 channel 的等待队列
6. 解锁所有 channel，把 G 设为等待状态，调度走
7. 被唤醒的时候处理对应的 case

多个 case 同时 ready 怎么办？
- 伪随机选一个！不是顺序的！
- 这样防止饿死，不会总是先写的 case 总是被选中
- 所以不要依赖 select 的 case 顺序！
```

---

### 17. 怎么实现一个线程安全的单例？
```go
// 最推荐：用 sync.Once
type Singleton struct {
    // ...
}

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

// sync.Once 内部就是 atomic + CAS
// 保证只执行一次，而且并发安全

// 不推荐：双重检查锁（Java 写法）
// Go 里有内存重排问题，可能拿到未初始化的 instance
func GetInstanceBad() *Singleton {
    if instance == nil {
        mu.Lock()
        defer mu.Unlock()
        if instance == nil {
            instance = &Singleton{}
        }
    }
    return instance
}
```

---

### 18. 怎么控制 Goroutine 并发数？
```go
方案 1: 有缓冲 channel 做信号量（最常用）
var sem = make(chan struct{}, 100)  // 最多 100 并发

func task() {
    sem <- struct{}{}        // 占一个坑
    defer func() { <-sem }() // 释放
    
    // 业务逻辑
}

方案 2: Worker Pool
func workerPool(tasks []Task, workerCount int) {
    jobs := make(chan Task, 1000)
    var wg sync.WaitGroup
    
    for i := 0; i < workerCount; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for task := range jobs {
                process(task)
            }
        }()
    }
    
    for _, task := range tasks {
        jobs <- task
    }
    close(jobs)
    wg.Wait()
}

方案 3: errgroup.Group（golang.org/x/sync/errgroup）
// 不仅能控制并发，还能收集错误
```

---

### 19. WaitGroup 有哪些坑？
```
常见坑：

1. Add 要在 goroutine 外面加
❌ 错误：
go func() {
    wg.Add(1)  // goroutine 已经跑了，Wait 可能先执行
    defer wg.Done()
}()

✅ 正确：
wg.Add(1)  // 启动 goroutine 之前加
go func() {
    defer wg.Done()
}()

2. Wait 要等所有 Add 都完成
// 不要边加任务边 Wait

3. Done 不能多调用，会变成负数 panic
// 每个 Add(1) 必须对应一个 Done()

4. WaitGroup 不能拷贝
// 传值的话会拷贝，状态不同步了
// 必须传指针

5. Done 要 defer 调用
// 不然 goroutine panic 了 Done 就没调用，Wait 永远卡
```

---

### 20. 什么是伪共享？怎么避免？
```
伪共享（False Sharing）：
CPU 缓存是按缓存行（Cache Line）为单位的，一般 64 字节
如果两个变量在同一个缓存行，不同 CPU 核心分别修改这两个变量
会导致缓存行在核心之间来回同步，性能暴跌

现象：
- 看似没有锁竞争
- 但性能很差，CPU 占用高
- perf 看 cache_miss 很高

怎么避免：
1. 变量之间填充 Padding，让每个变量独占一个缓存行
type Counter struct {
    c1 int64
    _  [56]byte  // 64 - 8 = 56 字节填充
    c2 int64
    _  [56]byte
    c3 int64
}

2. 把会被同一个 CPU 修改的变量放一起
3. 每个核心用本地变量，最后汇总

Go 1.19+ sync/atomic 里的类型已经自动处理了
```

---

## 🔴 性能优化（百度必问）

### 21. 线上服务 CPU 高怎么排查？
```
排查流程：

Step 1: 确认是不是这个服务的问题
- top/htop 看哪个进程 CPU 高
- 看监控，CPU 是不是真的超过阈值
- 看流量是不是涨了，是不是正常的业务增长

Step 2: 看 CPU 是用户态高还是内核态高
- us 高：大部分是业务代码，或者 GC
- sy 高：系统调用多，或者频繁上下文切换

Step 3: 抓 profile
go tool pprof http://ip:port/debug/pprof/profile?seconds=30

进入 pprof 交互：
top → 看哪个函数占 CPU 最高
list 函数名 → 看函数哪一行占 CPU 多
web → 生成火焰图，看调用链占比
traces → 看调用栈分布

Step 4: 常见原因分析
如果是 GC 占比高：
- go tool pprof heap profile
- 看哪个函数分配内存多
- 是不是小对象太多？sync.Pool 用了吗？
- 是不是有内存泄漏？

如果是某个业务函数占比高：
- 是不是循环太多？O(n²) 算法？
- 是不是正则表达式太复杂？
- 是不是 JSON 序列化/反序列化？换 sonic
- 是不是加解密太费 CPU？

如果是锁竞争高：
- go tool pprof mutex profile
- 看哪个锁竞争最激烈
- 能不能把锁粒度变小？分片锁？

Step 5: trace 更详细
go tool trace http://ip:port/debug/pprof/trace?seconds=10
看：
- GC 执行时间、频率
- Goroutine 调度延迟
- 系统调用耗时
- 锁等待时间
```

---

### 22. 内存泄漏怎么排查？
```
排查流程：

1. 看内存曲线
   - 是慢慢涨还是突然爆
   - 慢慢涨大概率是泄漏
   - 突然爆大概率是某次请求加载了太多数据

2. 抓 heap profile
   go tool pprof http://ip:port/debug/pprof/heap
   top 看哪个函数分配的内存多
   list 看具体哪一行

   对比两个时间点的 profile：
   go tool pprof -base old.prof new.prof
   看哪个函数新增的分配多

3. 看 goroutine 数量
   go tool pprof http://ip:port/debug/pprof/goroutine
   goroutine 数量异常多就是 goroutine 泄漏

Go 常见内存泄漏场景：

1. Goroutine 泄漏（最常见，占 90%）
   - channel 没人收，发送者永远阻塞
   - channel 没人发，接收者永远阻塞
   - for + select 没有退出分支
   - context 没 cancel，子 goroutine 永远等

2. 全局 map 只加不删
   var cache = make(map[Key]Value)
   不断往里面放，永远不删，内存一直涨
   解决：加 LRU，定期淘汰过期 key

3. sync.Pool 误用
   - 把大对象放进 Pool，永远不回收
   - 放进去的对象还在别的地方用

4. Goroutine 栈泄漏
   - Goroutine 栈初始 2KB，最大 1GB
   - 递归太深栈扩容，用完不缩回去
```

---

### 23. GC 调优有哪些手段？
```
GC 调优目标：
- 降低 GC 频率
- 减少 STW 时间
- 降低 GC CPU 占用

手段从易到难：

1. 减少内存分配（最简单，效果最好）
   - 复用对象：sync.Pool 复用临时对象
   - 预分配容量：make 时指定 len 和 cap，避免 append 扩容
   - 小对象合并成大对象
   - 避免内存逃逸
   - 返回值拷贝代替返回指针

2. 调整 GOGC
   - 默认 GOGC=100，heap 翻倍触发 GC
   - 内存充裕可以设 GOGC=200，GC 频率降一半
   - 延迟敏感可以设 GOGC=50，GC 更频繁但单次更短
   - 极限：GOGC=off 完全关闭 GC，特殊场景用

3. Go 版本升级（零成本！）
   - 每个版本 GC 都有优化
   - Go 1.14: 异步抢占，STW 大降
   - Go 1.17: 栈拷贝优化
   - Go 1.19: GC 调度优化，CPU 占用更平滑
   - Go 1.20: 大片内存回收优化

4. 控制 Goroutine 数量
   - Goroutine 栈也是 GC 扫描对象
   - Goroutine 太多，GC 扫描栈时间长
   - 用 worker pool 控制数量，不要无限 go

GC 调优原则：
- 先减少分配，再谈其他
- 能升级版本就升级，性价比最高
- 不要乱调参数，默认值一般是最优的
- 调完一定要有压测验证
```

---

### 24. Benchmark 怎么写才准确？
```go
正确的 Benchmark 写法：

func BenchmarkGood(b *testing.B) {
    // 准备工作：初始化数据、连接等
    data := prepareData()
    
    b.ResetTimer()  // 重置计时器，把准备时间排除！
    
    for i := 0; i < b.N; i++ {
        result := f(data)  // 要测的函数
        
        // 防止编译器优化掉没用的结果！
        atomic.AddUint64(&total, result)
    }
}

常见坑：

1. ❌ 编译器把循环优化没了
// 编译器发现 f() 没副作用，直接把循环删了

✅ 解决：把结果用 atomic 或者 channel 存起来

2. ❌ ResetTimer 忘加了
// 准备工作花了 1s，也被算进 benchmark 时间了

✅ 解决：准备完调用 b.ResetTimer()

3. ❌ b.N 被写死了
for i := 0; i < 1000; i++ { ... }

✅ 解决：必须用 b.N，testing 会动态调整

4. ❌ 并发测试写错了
func BenchmarkBad(b *testing.B) {
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            go f()  // ❌ 每次循环又起一个 goroutine，爆炸！
        }
    })
}

✅ 解决：RunParallel 本身就是多 goroutine 跑，里面直接调用

运行命令：
go test -bench=. -benchmem -count=10 -timeout=30m | tee result.txt
benchstat result.txt  # 统计平均值和方差，benchstat 很重要！
```

---

### 25. 怎么避免内存逃逸？
```
常见逃逸场景和解决办法：

1. 返回局部变量指针 ❌
func NewUser() *User {
    return &User{Name: "bob"}  // 逃逸！
}

✅ 解决 1：返回值拷贝，不传指针
func NewUser() User {
    return User{Name: "bob"}  // 不逃逸！
}

✅ 解决 2：让调用方分配，函数内赋值
func NewUser(u *User) {
    u.Name = "bob"  // u 是调用方传进来的，不逃逸
}

2. 闭包捕获变量 ❌
func f() func() {
    x := 0
    return func() { x++ }  // x 被闭包捕获，逃逸！
}

✅ 解决：尽量不用闭包，或者用参数传进去

3. 变量大小不确定 ❌
n := 100
s := make([]int, n)  // n 是变量，大小不确定，可能很大，逃逸！

✅ 解决：如果大小固定就写死
s := make([]int, 100)  // 编译期知道 100，不大就栈分配

4. 发送指针到 channel ❌
ch <- &u  // u 逃逸！不知道哪个 Goroutine 会读

✅ 解决：发值不发指针
ch <- User{}  // 拷贝，不逃逸

5. interface{} 调用方法 ❌
var fmt.Stringer s = User{}  // User 逃逸！

✅ 解决：尽量用具体类型，少用 interface

逃逸优化原则：
- 小对象（小于 4KB）直接传值，别传指针
- 返回值拷贝比返回指针好，拷贝是栈上，指针要堆分配
- 预分配！make 的时候 len 和 cap 都给好
- 能不用闭包就不用
- 能不用 interface 就不用
```

---

## 🟣 系统设计（百度高级岗）

### 26. 怎么设计一个高并发计数器？
```
需求：支持百万级 QPS 的计数

方案 1: 单机版 - 分片计数器
type Counter struct {
    counters []int64
    pads     [][56]byte  // 每个计数器占一个缓存行，避免伪共享
}

func NewCounter(shards int) *Counter {
    return &Counter{
        counters: make([]int64, shards),
        pads:     make([][56]byte, shards),
    }
}

func (c *Counter) Add(n int64) {
    // 按 goroutine id 或者哈希分片
    shard := getgid() % len(c.counters)
    atomic.AddInt64(&c.counters[shard], n)
}

func (c *Counter) Get() int64 {
    var total int64
    for i := range c.counters {
        total += atomic.LoadInt64(&c.counters[i])
    }
    return total
}

性能：单机能到千万级 QPS

方案 2: 分布式版
- Redis INCR，但是单 Redis 瓶颈大概 10 万 QPS
- 分片：按用户 ID 哈希到不同 Redis 实例
- 批量：本地攒一批再同步到 Redis
- 最终一致：不要求绝对实时，有延迟没关系

方案 3: 读写分离
- 写：写本地内存，异步刷盘
- 读：读本地内存
- 定时同步到数据库
- 适合不需要强一致的计数场景，比如 PV、UV
```

---

### 27. 怎么实现一个限流？有哪些算法？
```
常用四种限流算法：

1. 计数器（固定窗口）
- 最简单，每秒允许 N 个请求
- 问题：边界突刺，0.9-1.0s 100个，1.0-1.1s 100个
  0.95-1.05s 这 100ms 有 200 个请求

2. 滑动窗口
- 把 1 秒分成 10 格，每个格子 100ms
- 每次请求把当前格子计数 +1
- 统计最近 10 个格子的总和
- 解决了边界问题，但是实现复杂

3. 令牌桶（最常用！）
- 桶里有 N 个令牌，每秒补充 R 个
- 每个请求拿一个令牌，拿到就过，拿不到就限流
- 优点：平滑，允许一定突发流量
- Go 官方实现：golang.org/x/time/rate

4. 漏桶
- 请求来了先进水桶，水桶以固定速率漏水处理
- 优点：绝对平滑，不允许突刺
- 缺点：不抗突发，适合下游能力固定的场景

分布式限流：
- Redis + Lua 脚本原子执行
- 或者用 sentinel 集群的限流功能
```

---

### 28. 怎么实现分布式锁？
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
- 锁一定要有唯一标识，不能解别人的锁

ZooKeeper 实现：
- 临时节点，会话断开自动删除
- 顺序节点，最小的那个拿到锁
- 比 Redis 可靠，但性能差一些
- 适合对一致性要求高的场景
```

---

### 29. 缓存穿透、击穿、雪崩区别和解决方案？
```
缓存穿透：
- 查询一个一定不存在的数据，缓存没有，每次都查 DB
- 比如黑客用不存在的 ID 狂打
- 解决方案：
  1. 布隆过滤器：把所有存在的 key 放布隆里
  2. 空值缓存：DB 查不到也缓存个空值，短 TTL

缓存击穿：
- 某一个热点 key 突然失效的瞬间，大量请求同时打 DB
- 比如双 11 爆款商品缓存过期
- 解决方案：
  1. 互斥锁：只有一个 goroutine 去查 DB 写缓存
  2. 永不过期：逻辑过期，后台异步更新
  3. 热点 key 预热

缓存雪崩：
- 大量 key 同时过期，DB 压力陡增
- 比如所有 key 的过期时间都是 1 小时，整点同时失效
- 解决方案：
  1. 过期时间加随机抖动，不要都整点过期
  2. 多级缓存：本地缓存 + 分布式缓存
  3. 服务降级：非核心接口直接返回默认值

一句话总结区别：
穿透：查不存在的
击穿：单个热点 key 过期
雪崩：大量 key 同时过期
```

---

### 30. 怎么设计一个短链接系统？
```
核心点：

1. 发号器：
   - 雪花算法生成唯一 ID
   - 62 进制编码：0-9, a-z, A-Z
   - 10 位长度就能支持 62^10 ≈ 8 亿个短链

2. 存储：
   - MySQL 存长链接 → 短链接映射
   - 短链接哈希做索引，查的时候快
   - Redis 做缓存，热点短链接不用查 DB

3. 跳转：
   - 302 临时重定向，不要用 301 永久
   - 因为后面可能要改统计数据或者屏蔽

4. 统计：
   - 访问日志异步写，不要同步影响跳转
   - 埋点进 Kafka，后台消费统计

5. 防刷：
   - 同一个 IP 1 分钟最多创建 10 个
   - 黑名单域名不让跳转
   - 内容审核，不能是违法网站

6. 高可用：
   - 多机房部署
   - Redis 集群
   - MySQL 主从
   - 短链接生成可以预生成一批放着，不用实时算
```

---

## 📝 手写代码题（百度常考）

### 31. 实现一个线程安全的 LRU 缓存
```go
type LRUCache struct {
    capacity int
    cache    map[int]*DListNode
    head     *DListNode
    tail     *DListNode
    mu       sync.RWMutex
}

type DListNode struct {
    key   int
    value int
    prev  *DListNode
    next  *DListNode
}

func Constructor(capacity int) LRUCache {
    head := &DListNode{}
    tail := &DListNode{}
    head.next = tail
    tail.prev = head
    return LRUCache{
        capacity: capacity,
        cache:    make(map[int]*DListNode),
        head:     head,
        tail:     tail,
    }
}

func (c *LRUCache) Get(key int) int {
    c.mu.RLock()
    node, ok := c.cache[key]
    c.mu.RUnlock()
    
    if !ok {
        return -1
    }
    
    c.mu.Lock()
    c.moveToHead(node)
    c.mu.Unlock()
    
    return node.value
}

func (c *LRUCache) Put(key int, value int) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    if node, ok := c.cache[key]; ok {
        node.value = value
        c.moveToHead(node)
        return
    }
    
    newNode := &DListNode{key: key, value: value}
    c.cache[key] = newNode
    c.addToHead(newNode)
    
    if len(c.cache) > c.capacity {
        tail := c.removeTail()
        delete(c.cache, tail.key)
    }
}

func (c *LRUCache) addToHead(node *DListNode) {
    node.prev = c.head
    node.next = c.head.next
    c.head.next.prev = node
    c.head.next = node
}

func (c *LRUCache) removeNode(node *DListNode) {
    node.prev.next = node.next
    node.next.prev = node.prev
}

func (c *LRUCache) moveToHead(node *DListNode) {
    c.removeNode(node)
    c.addToHead(node)
}

func (c *LRUCache) removeTail() *DListNode {
    node := c.tail.prev
    c.removeNode(node)
    return node
}
```

---

### 32. 实现一个简单的限流器（令牌桶）
```go
type TokenBucket struct {
    rate       int64         // 每秒生成多少令牌
    capacity   int64         // 桶容量
    tokens     int64         // 当前令牌数
    lastUpdate time.Time     // 上次更新时间
    mu         sync.Mutex
}

func NewTokenBucket(rate, capacity int64) *TokenBucket {
    return &TokenBucket{
        rate:       rate,
        capacity:   capacity,
        tokens:     capacity,
        lastUpdate: time.Now(),
    }
}

func (tb *TokenBucket) Allow() bool {
    tb.mu.Lock()
    defer tb.mu.Unlock()
    
    now := time.Now()
    elapsed := now.Sub(tb.lastUpdate)
    
    newTokens := int64(elapsed.Seconds()) * tb.rate
    
    if newTokens > 0 {
        tb.tokens = min(tb.tokens + newTokens, tb.capacity)
        tb.lastUpdate = now
    }
    
    if tb.tokens > 0 {
        tb.tokens--
        return true
    }
    
    return false
}

func min(a, b int64) int64 {
    if a < b {
        return a
    }
    return b
}
```

---

## ✅ 百度面试注意事项

1. **基础一定要非常扎实**：GMP、GC、Map、Slice 这些原理要能手绘
2. **算法不能差**：LeetCode hot100 要刷熟，百度很看重算法
3. **性能优化要能讲清楚**：怎么排查，用了什么工具，优化了多少，要有具体数字
4. **喜欢追根问底**：不要只知道怎么用，要知道为什么，原理是什么
5. **系统设计要有条理**：从需求分析到技术选型，从简到繁，层层递进
