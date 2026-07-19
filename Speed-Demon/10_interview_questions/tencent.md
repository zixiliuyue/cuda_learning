# 腾讯 Go 面试高频 50 题

> **腾讯特点：** 重工程实践、重架构、重微服务、问的深、喜欢结合业务场景

---

## 🟢 Go 基础（必问）

### 1. Go 的 GMP 调度模型详细说明？
```
G: Goroutine，用户态轻量级线程
M: Machine，内核线程，真正执行代码的实体
P: Processor，调度上下文，M 需要绑定 P 才能调度 G

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

### 2. Go GC 的完整流程？
```
Go 1.14+ 三色标记清除 + 混合写屏障 + 异步抢占

完整流程：

1. Mark Setup 阶段（STW ~1ms）
   - 开启写屏障
   - 开启辅助标记
   - 扫描所有根对象：栈、全局变量、寄存器

2. Mark 阶段（并发，用户代码一起跑）
   - 从根对象开始三色标记
   - 白色：未扫描
   - 灰色：已扫描，但引用的对象没扫完
   - 黑色：已扫描完，引用的对象也扫完了
   - 期间用户代码和 GC 一起跑，通过写屏障保证正确性

3. Mark Termination 阶段（STW ~10-50ms）
   - 重新扫描栈（因为栈是并发改的）
   - 结束标记，关闭写屏障

4. Sweep 阶段（并发）
   - 把所有白色对象（垃圾）清除
   - 归还给内存分配器，不还操作系统
   - Sweep 是并发的，不影响用户代码

GC 触发条件：
- 内存达到上次 GC 后 heap_size × (1 + GOGC/100)
- 2 分钟没 GC 了强制 GC
- 手动调用 runtime.GC()

混合写屏障：
- 删除屏障 + 插入屏障结合
- 删除屏障：删除引用时，把被删除的对象标记成灰色
- 插入屏障：新插入的引用，把新引用的对象标记成灰色
- 保证三色不变性，不会漏标对象
```

---

### 3. Channel 发送和接收的完整流程？
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

发送流程 chansend：
1. 加锁
2. 有接收者在等（recvq 不为空）：
   - 直接把数据拷贝给第一个等待的接收者
   - 把接收者的 G 从等待队列移到可运行队列
   - 解锁，唤醒接收者
3. 没接收者但 buf 有空位：
   - 把数据拷贝到 buf 的 sendx 位置
   - sendx++，qcount++
   - 解锁返回
4. buf 满了或者没 buf（无缓冲 channel）：
   - 把当前 G 塞进 sendq 等待队列
   - 解锁，把 G 设为等待状态，调度走
   - 等有人接收了再唤醒

接收流程 chanrecv：
1. 加锁
2. 有发送者在等（sendq 不为空）：
   - 无缓冲：直接从发送者那拷数据
   - 有缓冲：把 buf 第一个拿出来，把发送者的放进去
   - 唤醒发送者
3. 没发送者但 buf 有数据：
   - 从 buf 的 recvx 位置拿数据
   - recvx++，qcount--
   - 解锁返回
4. buf 空了：
   - 把当前 G 塞进 recvq 等待队列
   - 解锁，等待唤醒

关闭 channel：
- 唤醒所有在 recvq 等待的接收者，收到零值
- 唤醒所有在 sendq 等待的发送者，直接 panic
```

---

### 4. Go 的内存分配器原理？
```
Go 的内存分配是 TCMalloc 那套思路，多级缓存：

层级结构：
1. mcache：每个 P 独享的本地缓存，无锁
2. mcentral：全局中心缓存，按大小分，需要锁
3. mheap：堆内存，向操作系统申请的大内存，需要锁

分配流程：
小对象 (< 32KB)：
1. 从当前 P 的 mcache 里找对应大小的 span
2. mcache 有空闲的，直接拿，无锁，超快
3. mcache 没了，去 mcentral 拿一批放到 mcache
4. mcentral 也没了，去 mheap 申请新的 span

大对象 (> 32KB)：
1. 直接从 mheap 申请，不走 mcache/mcentral

内存块大小规格：
- 67 种大小级别，从 8B 到 32KB
- 比如 8B, 16B, 24B, ..., 32KB
- 每个级别对应一种 span

为什么快：
1. 大部分小对象都在 P 的 mcache 分配，无锁
2. 按大小分块，减少内存碎片
3. 每个 P 自己的缓存，没有锁竞争
```

---

### 5. slice 的底层原理？扩容策略？
```
slice 结构：
type slice struct {
    array unsafe.Pointer  // 底层数组指针
    len   int             // 实际使用长度
    cap   int             // 容量，底层数组大小
}

扩容策略（Go 1.18+）：
旧容量 < 256：新容量 = 旧容量 × 2
旧容量 >= 256：新容量 = 旧容量 + (旧容量 + 3×256)/4
这样平滑过渡，避免之前 1024 那个拐点

扩容过程：
1. 计算新容量 newcap
2. 计算需要的内存大小：newcap × elem.size
3. 分配新的底层数组
4. 把旧数组的数据拷贝到新数组
5. 更新 slice 的 array 指针和 cap

重要坑点：
- 扩容后 array 指针变了！所以 append 可能返回新地址
- 多个 slice 共享同一个底层数组时，改一个其他的也会变
- s[:] 这种不会扩容，还是共享原数组
```

---

### 6. map 的底层原理？负载因子？扩容？
```
hmap 结构：
type hmap struct {
    count     int       // 元素个数
    B         uint8     // 桶数量：2^B 个桶
    hash0     uint32    // 哈希种子，防哈希碰撞攻击
    
    buckets    unsafe.Pointer // 桶数组
    oldbuckets unsafe.Pointer // 扩容时的旧桶数组
    
    nevacuate uintptr        // 搬迁进度，小于这个的旧桶已经搬完
    
    extra *mapextra          // 溢出桶指针，大 map 才用
}

bmap 桶结构（每个桶 8 个元素）：
type bmap struct {
    tophash [8]uint8       // 8 个 key 的哈希值高 8 位，快速比较
    keys    [8]keyType     // 8 个 key
    values  [8]valueType   // 8 个 value
    overflow *bmap         // 溢出桶链表，哈希冲突多了就链
}

查找流程：
1. key 算哈希值 hash
2. hash & (1<<B -1) 算在哪个桶
3. 遍历桶的 tophash，比较高 8 位
4. 高 8 位一样，再比较完整的 key
5. 找到了返回，没找到去溢出桶继续找
6. 所有桶都找完没找到，返回零值

负载因子：
count / (2^B × 8) > 6.5 触发扩容
也就是平均每个桶超过 6.5 个元素就扩容

扩容策略：
1. 翻倍扩容：负载因子超过 6.5，桶数 × 2
2. 等量扩容：溢出桶太多了，整理一下，桶数不变

渐进式扩容：
- 不是一次把所有桶都搬完，每次访问 map 搬 1-2 个桶
- oldbuckets 指向旧桶，慢慢搬完才置空
- 这样不会因为扩容卡住用户请求
```

---

### 7. interface 的类型转换？类型断言？
```
两种 interface：
iface：有方法的 interface，比如 io.Reader
  - tab: itab 指针，方法表
  - data: 数据指针

eface：空 interface，interface{}
  - _type: 类型指针
  - data: 数据指针

类型断言：
var i interface{} = 123

// 形式 1：panic 版
x := i.(int)     // 对了就赋值，错了直接 panic

// 形式 2：ok 版，不 panic
x, ok := i.(int) // 对了 ok=true，错了 ok=false，x 是零值

类型转换原理：
1. 比较 interface 的类型指针和目标类型
2. 一样就直接取 data
3. 不一样就转换失败

类型 switch：
switch v := i.(type) {
case int:
    fmt.Println("int", v)
case string:
    fmt.Println("string", v)
default:
    fmt.Println("unknown")
}
```

---

### 8. defer 的原理？执行时机？
```
defer 关键字在编译期处理：
1. 把 defer 函数的参数都求值好
2. 把 defer 函数信息插到当前 Goroutine 的 defer 链表头
3. 函数返回前，从链表头开始一个个执行

defer 结构（runtime._defer）：
- sp: 栈指针
- pc: 返回地址
- fn: 要执行的函数
- link: 链表指针，下一个 defer

执行时机（return 不是原子的！）：
return = 给返回值赋值 → 执行所有 defer → 真正 return 指令

经典面试题：
func f1() int {
    x := 5
    defer func() { x++ }()
    return x  // 返回值已经复制好了，x++ 不影响返回值
} → 返回 5

func f2() (x int) {
    defer func() { x++ }()
    return 5  // x 就是返回值本身，x++ 直接改返回值
} → 返回 6
```

---

### 9. Go 的逃逸分析是什么？
```
逃逸分析：编译期分析变量的生命周期，决定放栈还是堆

不逃逸 → 栈上分配，函数返回直接回收，不用 GC
逃逸了 → 堆上分配，需要 GC 回收

常见逃逸场景：
1. 函数返回局部变量的指针
   func f() *int { x := 1; return &x } → x 逃逸到堆

2. 变量太大，栈装不下
   比如 10MB 的 slice，栈只有几 MB，必须放堆

3. 变量大小编译期不确定
   s := make([]int, n) → n 是运行时才知道的，可能很大

4. 闭包捕获的变量
   f := func() { fmt.Println(x) } → x 被闭包捕获，逃逸

5. 发送指针到 channel
   ch <- &x → x 逃逸，因为不知道哪个 Goroutine 会读

6. interface{} 调用方法
   var i interface{} = x → x 逃逸

怎么看有没有逃逸：
go build -gcflags="-m -l"

优化思路：
- 不需要指针就别返回指针，返回值拷贝
- 小对象直接传值，别传指针
- 预分配容量，避免 append 过程中多次分配
- 减少闭包使用
```

---

### 10. Go 的内联优化？
```
内联：把小函数的代码直接拷贝到调用方，省去函数调用开销

为什么要内联：
1. 省去函数调用开销：参数入栈、PC 跳转、返回
2. 省去 defer/panic/recover 这些 runtime 开销
3. 更多优化机会，内联后可以做常量传播、死代码消除等

内联条件（Go 1.20+）：
1. 函数足够小，一般 80 个节点以内（大概 20-30 行代码）
2. 不能有循环、不能有 defer、不能有 panic、不能有 recover
3. 不能有 go statement、不能有 select、不能有 chan 操作
4. 不能递归、不能太复杂

//go:noinline 强制不内联
//go:inline 尽量内联（Go 1.16+）

内联的代价：
- 代码体积变大，同一个函数被多个地方内联拷贝多份
- 栈回溯变复杂，调试的时候堆栈信息没那么清晰

可以看内联决策：
go build -gcflags="-m=2"
```

---

## 🟡 并发编程（腾讯重点）

### 11. Goroutine 调度的完整流程？
```
调度触发时机：
1. 主动：go 创建新 G、time.Sleep、channel 阻塞、锁阻塞
2. 被动：异步抢占（每 10ms 信号）、同步抢占（sysmon 检测）
3. 返回：系统调用返回、函数返回

M 调度循环：
for {
    1. 找一个可运行的 G
       - 先看当前 P 本地队列
       - 本地没了，去其他 P 偷一半（工作窃取）
       - 其他 P 也没，去全局队列拿
       - 都没，去网络轮询器拿（netpoller）
       - 还没，M 和 P 解绑，进入休眠
    
    2. 执行 G
       - 把 G 的上下文放到 M 上
       - 跳转到 G 的 PC 开始执行
    
    3. G 执行完或者被抢占
       - 保存 G 的上下文
       - 把 G 放回可运行队列
       - 回到步骤 1 继续找下一个 G
}

工作窃取细节：
- 随机选一个 P，从它的队列尾部偷一半
- 这样分散负载，不会所有 P 都抢第一个
- 偷不到就多试几次，实在不行去全局队列

抢占细节：
- 同步抢占：G 运行超过 10ms，sysmon 给它发抢占标记
  G 下次进入函数序言的时候检查标记，让出 CPU
- 异步抢占：Go 1.14+，基于信号
  sysmon 发 SIGURG 信号，信号处理函数直接把 G 抢占下来
  这样死循环也能抢下来！
```

---

### 12. select 的原理？多个 case 同时 ready 怎么选？
```
select 是 Go 实现的，不是系统调用

编译期处理：
- 0 个 case：直接阻塞，永远等
- 1 个 case：等价于直接 channel 操作
- 2 个及以上 case：走 runtime.selectgo

selectgo 流程：
1. 对所有 case 按 channel 地址排序（加锁顺序，防止死锁）
2. 按顺序把所有 channel 都加锁
3. 先快速检查一遍有没有已经 ready 的 case
   - 能发的、能收的
   - 有就直接处理这个 case，解锁返回
4. 都没 ready，把当前 G 注册到所有 channel 的等待队列
5. 解锁所有 channel，把 G 设为等待状态，调度走
6. 被唤醒的时候：
   - 找到是哪个 channel 唤醒的
   - 处理这个 case
   - 把 G 从其他所有 channel 等待队列删掉
   - 返回

多个 case 同时 ready 怎么办？
- 伪随机选一个！不是顺序的！
- 这样防止饿死，不会总是先写的 case 总是被选中
- 所以不要依赖 select 的 case 顺序！

nil channel 永远阻塞：
var ch chan int
select {
case <-ch:  // nil channel，永远不会触发
    ...
}
可以用这个特性动态开关 case
```

---

### 13. sync.Mutex 原理？
```
type Mutex struct {
    state int32  // 锁状态
    sema  uint32 // 信号量
}

state 字段各位含义：
bit 0: 锁是否被持有，1=持有
bit 1: 是否有唤醒的 goroutine
bit 2: 是否饥饿模式
bit 3~: 等待者数量

两种模式：
1. 正常模式
   - 新来的 goroutine 先自旋几次试试拿锁
   - 自旋拿不到就进等待队列，FIFO
   - 锁释放了唤醒队列头的 goroutine
   - 但是唤醒的和新来的竞争，新来的可能抢赢
   - 因为新来的已经在 CPU 上跑了，有优势

2. 饥饿模式（Go 1.9+，解决公平性问题）
   - 某个 goroutine 等超过 1ms 没拿到锁，就进入饥饿模式
   - 饥饿模式下，新来的不允许自旋，直接进等待队列尾
   - 锁释放了直接给队列头的，新来的抢不走
   - 等拿到锁的 goroutine 发现自己是最后一个等待者，退出饥饿模式

Lock 流程：
1. CAS 试试直接拿锁，成功直接返回
2. 失败，进入 slow path
3. 自旋几次（如果是多核且等待者少）
4. 自旋还没拿到，把自己加到等待队列
5. 信号量等待，被唤醒
6. 看是不是饥饿模式，是就直接拿锁
7. 不是就和新来的竞争

UnLock 流程：
1. 把持有位清 0
2. 如果有等待者，发信号量唤醒一个
3. 饥饿模式下唤醒队列头的
4. 正常模式下唤醒第一个，但不一定能抢到

饥饿模式解决了什么问题？
新来的 goroutine 总是在 CPU 上，自旋总是赢
等待队列里的 goroutine 永远抢不到，饿死
```

---

### 14. sync.RWMutex 原理？读多写少场景比 Mutex 快多少？
```
type RWMutex struct {
    w           Mutex  // 互斥锁，写之间互斥
    writerSem   uint32 // 写者等待信号量
    readerSem   uint32 // 读者等待信号量
    readerCount int32  // 读者数量，负数表示有写者在等
    readerWait  int32  // 写者等待所有读者完成的数量
}

读锁 RLock：
1. readerCount++
2. 如果 readerCount >= 0，说明没有写者在等，直接拿读锁成功
3. 如果 readerCount < 0，说明有写者在等，不能拿读锁
   - 去 readerSem 等待，等写者走了再拿

读锁 RUnlock：
1. readerCount--
2. 如果 readerCount >= 0，说明没有写者在等，直接返回
3. 如果 readerCount < 0，说明有写者在等
   - readerWait--，如果减到 0 了说明所有读者都走了
   - 给 writerSem 发信号，唤醒写者

写锁 Lock：
1. 先拿 w Mutex，保证写和写之间互斥
2. 把 readerCount 变成负数（减 1<<30），表示有写者来了
   - 这样新来的读者看到 readerCount 负的就不会拿读锁了
3. 如果还有读者在读（readerWait > 0），就去 writerSem 等
4. 等所有读者都读完了，拿到写锁

写锁 Unlock：
1. 把 readerCount 加回去，变回正数
2. 唤醒所有在 readerSem 等待的读者
3. 释放 w Mutex

性能对比（8 核，99% 读，1% 写）：
sync.Mutex: ~400ns 一次操作
sync.RWMutex: ~50ns 一次读锁
RWMutex 读比 Mutex 快 8 倍左右！

注意事项：
1. RWMutex 不是万能的，写多了反而比 Mutex 慢
2. 读锁不要升级成写锁，会死锁
3. 写锁期间不要拿读锁，也会死锁
4. 一般读占比超过 90% 才考虑用 RWMutex
```

---

### 15. sync.Cond 原理？适用场景？
```
type Cond struct {
    L Locker      // 关联的锁
    notifyList    // 等待通知的 goroutine 列表
    checker copyChecker
}

Wait 操作：
1. 把当前 G 加入等待列表
2. 解锁 L（重要！不然其他人没法拿锁改条件）
3. 等待被唤醒
4. 被唤醒后重新加锁 L
5. 返回

Signal 操作：
1. 唤醒等待列表里的第一个 goroutine

Broadcast 操作：
1. 唤醒等待列表里的所有 goroutine

经典使用模式：
// 消费者
c.L.Lock()
for 条件不满足 {  // 注意是 for 不是 if！
    c.Wait()      // 被唤醒后要重新检查条件
}
消费
c.L.Unlock()

// 生产者
c.L.Lock()
生产，修改条件
c.Signal()  // 或者 Broadcast
c.L.Unlock()

为什么是 for 不是 if？
- 可能多个 goroutine 被 Broadcast 唤醒
- 第一个抢到锁的把条件改了，第二个进来条件又不满足了
- 所以必须循环检查！

适用场景：
- 生产者消费者模型
- 等待某个条件成立，不是简单的互斥
- 一个事件发生，通知多个等待者

性能注意：
- 不要用 Broadcast 代替 Signal，惊群效应
- 条件变了才发通知，不要没事乱发
- 一定要和锁配合用
```

---

### 16. sync.Once 原理？怎么保证只执行一次？
```
type Once struct {
    done uint32  // 0=未执行，1=已执行
    m    Mutex
}

func (o *Once) Do(f func()) {
    // 快速路径：原子读，看看是不是已经执行过了
    if atomic.LoadUint32(&o.done) == 0 {
        o.doSlow(f)
    }
}

func (o *Once) doSlow(f func()) {
    o.m.Lock()
    defer o.m.Unlock()
    
    // 加锁后再检查一遍，双重检查
    if o.done == 0 {
        // 注意：先执行 f，再把 done 设为 1
        // 所以如果 f panic 了，done 还是 0，下次还会执行
        defer atomic.StoreUint32(&o.done, 1)
        f()
    }
}

核心点：
1. 第一个原子 Load 是快速路径，大部分时候 done=1，直接返回，无锁
2. 没执行过才进 slow path，加锁
3. 加锁后还要再检查一次 done，防止多个 goroutine 同时进来
4. 用 defer 保证 f 执行完才把 done 设为 1
5. 如果 f 里又调用了同一个 Once.Do，会死锁！

常见坑：
func f() {
    once.Do(f)  // 死锁！Do 里已经加锁了，递归加锁
}

单例模式最佳实践就是用 sync.Once，比自己写双重检查好
```

---

### 17. sync.Pool 原理？适用场景？为什么能减少 GC？
```
type Pool struct {
    local unsafe.Pointer // 每个 P 的本地池，per-P 无锁访问
    New func() interface{} // 对象创建函数
}

每个 P 的本地池结构：
type poolLocal struct {
    private interface{}   // 私有对象，只有当前 P 能拿，不用锁
    shared  []interface{} // 共享列表，其他 P 可以偷
    pad     [128]byte     // 防止伪共享，占一个缓存行
}

Get 流程：
1. 拿当前 P 的 poolLocal
2. 先看 private 有没有，有直接拿，无锁
3. private 没了，加锁从 shared 拿
4. shared 也没了，去其他 P 的 shared 偷一半
5. 都没了，调用 New() 新建一个

Put 流程：
1. 拿当前 P 的 poolLocal
2. private 空的话直接放 private，无锁
3. private 有了，加锁放 shared

GC 时的行为：
每次 GC 开始前，把所有 Pool 里的对象都清掉！
- 所以 Pool 里的对象最多存活一轮 GC
- 不会内存泄漏，GC 会回收

为什么能减少 GC：
- 对象复用，不用每次都新建，减少分配次数
- 分配少了，GC 频率自然就降了
- 而且是 per-P 设计，大部分操作无锁，性能好

适用场景：
1. 大量创建销毁的临时对象，比如 buffer
2. 对象创建开销大，复用划算
3. 对象生命周期短，用完就可以还回去
4. stateless 的对象，不需要保存状态

不适用场景：
1. 长生命周期对象，Pool 会在 GC 时清掉
2. 需要保存状态的对象，下次拿可能不是同一个
3. 大对象，占内存太多

最佳实践：
- 取出来用完 defer Put 回去
- Put 之前把对象状态重置，别留脏数据
- 不要存指针指针，不然反而有逃逸开销
```

---

### 18. sync.Map 原理？比 RWMutex+map 快多少？
```
type Map struct {
    mu Mutex
    
    // read map，原子访问，不加锁
    // read 里的 key 不会被删，只是标记删除
    read atomic.Value // 实际是 readOnly 结构
    
    // dirty map，读写都加锁
    // 包含所有 key，read 里没有的新 key 都在这
    dirty map[interface{}]*entry
    
    // misses 计数，read 没命中多少次后把 dirty 升级成 read
    misses int
}

type readOnly struct {
    m       map[interface{}]*entry
    amended bool // true 表示 dirty 里有 read 没有的 key
}

type entry struct {
    p unsafe.Pointer // 指向 value
    // p == nil: key 已被删除
    // p == expunged: key 已被标记清除，dirty 里没有这个 key
    // 其他: 正常 value 指针
}

Load 流程：
1. 原子读 read map，看 key 在不在
2. 在，直接返回 value，全程无锁！
3. 不在且 amended=false，说明 dirty 也没有，返回 nil
4. amended=true，加锁
5. 再检查一遍 read（可能中间升级了）
6. read 还是没有，去 dirty 找
7. misses++，如果 misses 超过 len(dirty) 一半
   - 把 dirty 升级成新的 read，dirty 置空
8. 解锁返回

Store 流程：
1. 原子读 read，key 在，直接 CAS 更新 value，无锁
2. key 不在，加锁
3. 再检查 read，可能中间升级了
4. read 有了，CAS 更新
5. read 没有，去 dirty 里更新或者新增
6. 解锁返回

Delete 流程：
1. read 里有，就把 entry.p 设为 nil，标记删除，不真删
2. read 里没有，就去 dirty 里真删

性能对比：
场景：90% 读，9% 更新，1% 新增
sync.RWMutex + map: ~100ns 一次 Load
sync.Map: ~20ns 一次 Load

读性能是 RWMutex 的 5 倍！
但是新增和更新因为要加锁，性能和 RWMutex 差不多

所以 sync.Map 最适合读多写少，key 比较稳定的场景
```

---

### 19. Context 原理？怎么传递取消信号？
```
type Context interface {
    Deadline() (deadline time.Time, ok bool)
    Done() <-chan struct{}
    Err() error
    Value(key interface{}) interface{}
}

Context 是链式结构，父 Context 派生个子 Context：
context.Background() → 根 Context
    ↓ context.WithCancel()
    cancelCtx
        ↓ context.WithTimeout()
        timerCtx

cancelCtx 结构：
type cancelCtx struct {
    Context
    mu       sync.Mutex
    done     chan struct{} // 懒创建，第一次 cancel 才建
    children map[canceler]struct{}
    err      error
}

取消流程：
1. 调用 cancel() 函数
2. cancelCtx 加锁
3. 把自己的 done channel 关闭
4. 遍历所有 children，一个个 cancel
5. 把自己从父 Context 的 children 里删掉
6. 解锁

Done() 原理：
返回那个 chan struct{}，channel 关闭的时候所有读的都会收到
所以子 Context 里的代码只要监听 <-ctx.Done() 就知道要取消了

为什么能级联取消？
因为每个 cancelCtx 都保存了自己的 children
取消的时候递归取消所有子子孙孙

Value 传递原理：
type valueCtx struct {
    Context
    key, val interface{}
}

Value(key) 会沿着链往上找，直到找到 key 或者到根
所以是 O(n) 的，别放太多 Value

Context 最佳实践：
1. Context 必须作为函数第一个参数，名字叫 ctx
2. 不要把 Context 存在 struct 里，要显式传递
3. Value 只能传请求级别的元数据，不要传业务参数
4. 不要传 nil Context，要用 context.TODO() 当占位
5. 调用 cancel 是必须的，不然 goroutine 泄漏！
```

---

### 20. Atomic 和 Mutex 怎么选？CAS 的 ABA 问题？
```
能用 Atomic 就用 Atomic，性能好，不会死锁

性能对比（8 核）：
atomic.AddInt64: ~10ns
mutex.Lock+Unlock: ~45ns

Atomic 适用场景：
- 计数器
- 标志位
- 指针交换
- 简单的状态机
- 一个变量就能搞定的场景

Mutex 适用场景：
- 多个变量需要原子操作
- 临界区有复杂逻辑
- 需要条件等待
- 临界区执行时间长，空转不如等锁

CAS 的 ABA 问题：
1. 线程 1 读 A
2. 线程 2 把 A 改成 B，又改回 A
3. 线程 1 CAS，发现还是 A，以为没变，成功
4. 但其实中间已经变过了！

Go 的 atomic 怎么解决 ABA？
大部分场景其实不需要解决，计数器这种 ABA 不影响
真需要解决可以用：
1. 加版本号，每次修改版本号+1，CAS 同时比较版本和值
2. 用 unsafe.Pointer 的时候小心，指针回收复用问题
3. 一般业务场景遇不到 ABA，不用太担心
```

---

## 🔴 性能优化（腾讯高级岗必问）

### 21. 线上服务 CPU 高，完整排查流程？
```
按这个流程，一步一步来：

Step 1: 确认是不是这个服务的问题
- top/htop 看哪个进程 CPU 高
- 看监控，CPU 是不是真的超过阈值
- 看流量是不是涨了，是不是正常的业务增长

Step 2: 看 CPU 是用户态高还是内核态高
- top 看 us 和 sy
- us 高：大部分是业务代码，或者 GC
- sy 高：系统调用多，或者频繁上下文切换

Step 3: 抓 profile
# 抓 30 秒 CPU profile
go tool pprof http://ip:port/debug/pprof/profile?seconds=30

进入 pprof 交互：
top → 看哪个函数占 CPU 最高，列出来前 10
list 函数名 → 看函数哪一行占 CPU 多
web → 生成火焰图，看调用链占比

Step 4: 常见原因分析
如果是 GC 占比高：
- go tool pprof heap profile
- 看哪个函数分配内存多
- 是不是小对象太多？sync.Pool 用了吗？
- 是不是有内存泄漏？
- GOGC 是不是设太小了？

如果是某个业务函数占比高：
- 是不是循环太多？O(n²) 算法？
- 是不是正则表达式太复杂？
- 是不是 JSON 序列化/反序列化？换 fastjson/sonic
- 是不是加解密太费 CPU？

如果是锁竞争高：
- go tool pprof mutex profile
- 看哪个锁竞争最激烈
- 能不能把锁粒度变小？分片锁？

Step 5: 更详细的 trace
go tool trace http://ip:port/debug/pprof/trace?seconds=10
看：
- GC 执行时间、频率
- Goroutine 调度延迟
- 系统调用耗时
- 锁等待时间

Step 6: 验证优化
- 优化后再抓 profile 对比
- 看 P99 延迟有没有降
- 看 GC 次数有没有减少

⚠️ 重要原则：
- 不要猜！profile 说哪里慢就优化哪里
- 每次只改一个变量，不然不知道哪个起作用
- 优化完一定要有 benchmark 验证
```

---

### 22. 内存泄漏怎么排查？Go 常见内存泄漏场景？
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
   看 stack trace，都堵在哪个函数哪一行

Go 常见内存泄漏场景：

1. Goroutine 泄漏（最常见，占 90%）
   - channel 没人收，发送者永远阻塞
   - channel 没人发，接收者永远阻塞
   - for + select 没有退出分支
   - context 没 cancel，子 goroutine 永远等
   - time.After 在循环里，Timer 没被回收
   - http 请求没设超时，客户端 goroutine 卡住

2. 全局 map 只加不删
   var cache = make(map[Key]Value)
   不断往里面放，永远不删，内存一直涨
   解决：加 LRU，定期淘汰过期 key

3. sync.Pool 误用
   - 把大对象放进 Pool，永远不回收
   - 放进去的对象还在别的地方用
   解决：Pool 只放临时小对象，用完就还

4. Goroutine 栈泄漏
   - Goroutine 栈初始 2KB，最大 1GB
   - 递归太深栈扩容，用完不缩回去
   解决：控制递归深度，避免无限递归

5. 未关闭的资源
   - 文件句柄没关
   - DB 连接没关
   - HTTP body 没关
   解决：defer 关闭

6. cgo 内存泄漏
   - C 语言 malloc 的内存 Go GC 管不到
   解决：C 里 malloc 的必须 free

怎么预防：
- 每个 go 启动的 goroutine，必须有明确的退出方式
- 所有的资源打开都 defer 关闭
- 所有的 context 都要 cancel
- 全局 map 一定要有淘汰机制
- 加内存使用率告警，80% 就告警
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
   - 小对象合并：把多个小字段合并成一个 struct
   - 避免内存逃逸：尽量栈分配，不要逃逸到堆
   - 返回值拷贝代替返回指针
   - 大 slice 不用了就置 nil，帮助 GC

2. 调整 GOGC（效果明显）
   - 默认 GOGC=100：heap 翻倍触发 GC
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

5. 避免大对象
   - 大对象直接分配在堆上，GC 扫描慢
   - 大 slice 不用了置 nil，断开引用
   - 大对象可以考虑内存池自己管理

6. GC 调优（高级，不推荐）
   - runtime/debug.SetGCPercent()
   - runtime/debug.FreeOSMemory() 手动归还内存给 OS
   - 一般不用，让 Go 自己管理就好

GC 调优原则：
- 先减少分配，再谈其他
- 能升级版本就升级，性价比最高
- 不要乱调参数，默认值一般是最优的
- 调完一定要有压测验证
```

---

### 24. 怎么减少内存逃逸？
```
什么是逃逸：本该在栈上的变量跑到堆上了
逃逸 = 堆分配 = GC 要扫 = 性能差

怎么看有没有逃逸：
go build -gcflags="-m -l"
输出 "moved to heap" 就是逃逸了

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
func f(x *int) func() {
    return func() { *x++ }  // x 是参数，不逃逸
}

3. 变量大小不确定 ❌
n := 100
s := make([]int, n)  // n 是变量，大小不确定，可能很大，逃逸！

✅ 解决：如果大小固定就写死
s := make([]int, 100)  // 编译期知道 100，不大就栈分配

4. 发送指针到 channel ❌
ch := make(chan *User)
u := &User{}
ch <- u  // u 逃逸！不知道哪个 Goroutine 会读

✅ 解决：发值不发指针
ch := make(chan User)
ch <- User{}  // 拷贝，不逃逸

5. interface{} 调用方法 ❌
var fmt.Stringer s = User{}  // User 逃逸！
s.String()

✅ 解决：尽量用具体类型，少用 interface

6. 全局变量 ❌
var u *User
func f() {
    u = &User{}  // 全局变量，肯定逃逸！
}

✅ 解决：尽量用局部变量，少用全局变量

逃逸优化原则：
- 小对象（小于 4KB）直接传值，别传指针
- 返回值拷贝比返回指针好，拷贝是栈上，指针要堆分配
- 预分配！make 的时候 len 和 cap 都给好
- 能不用闭包就不用
- 能不用 interface 就不用
```

---

### 25. Benchmark 怎么写才准确？有哪些坑？
```
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

常见坑和避坑：

1. ❌ 编译器把循环优化没了
func BenchmarkBad(b *testing.B) {
    for i := 0; i < b.N; i++ {
        f()  // 编译器发现 f() 没副作用，直接把循环删了！
    }
}

✅ 解决：把结果用 atomic 或者 channel 存起来
var total uint64
func BenchmarkGood(b *testing.B) {
    for i := 0; i < b.N; i++ {
        r := f()
        atomic.AddUint64(&total, r)  // 编译器不敢删
    }
}

2. ❌ ResetTimer 忘加了
准备工作花了 1s，也被算进 benchmark 时间了
✅ 解决：准备完调用 b.ResetTimer()

3. ❌ b.N 被写死了
for i := 0; i < 1000; i++ { ... }
✅ 解决：必须用 b.N，testing 会动态调整

4. ❌ 每次循环都有很大的固定开销
func BenchmarkBad(b *testing.B) {
    for i := 0; i < b.N; i++ {
        time.Sleep(1 * time.Millisecond)  // 固定开销掩盖了函数本身
        f()
    }
}
✅ 解决：b.N 大一点，或者把固定开销挪外面

5. ❌ 并发测试写错了
func BenchmarkBad(b *testing.B) {
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            go f()  // ❌ 每次循环又起一个 goroutine，爆炸！
        }
    })
}
✅ 解决：RunParallel 本身就是多 goroutine 跑，里面直接调用

6. ❌ 结果受 GC 影响
✅ 解决：
- b.StopTimer() 暂停计时
- 或者用 testing.B 的 ReportAllocs() 看分配情况
- 多次跑取平均值，用 benchstat 统计显著性

7. ❌ benchmark 之间互相影响
前一个 benchmark 分配了很多内存，影响后一个
✅ 解决：每个 benchmark 跑之前做清理，或者单独跑

运行命令：
go test -bench=. -benchmem -count=10 -timeout=30m | tee result.txt
benchstat result.txt  # 统计平均值和方差

benchstat 很重要，单次跑的结果可能不准，多次跑统计显著性
```

---

## 🟣 微服务 & 系统设计（腾讯核心）

### 26. 微服务怎么拆分？DDD 怎么落地？
```
微服务拆分原则：

1. 单一职责 / 高内聚低耦合
   - 一个服务只做一件事
   - 服务内部高内聚，服务之间低耦合
   - 改一个服务不用动其他服务

2. 按业务领域拆分（DDD）
   - 不是按表拆，不是按层拆，按业务领域边界拆
   - 比如电商拆成：用户、商品、订单、支付、库存、物流
   - 每个领域有自己的数据库，不跨库直连

3. 服务大小合适
   - 不要太粗：一个服务几万行代码，啥都干
   - 不要太细：一个服务几个接口，几百行，运维成本高
   - 大概 2-20 个人维护一个服务比较合适

4. 考虑团队组织结构
   - 康威定律：系统设计等于组织沟通结构
   - 一个团队维护的服务不要跨太多领域
   - 不然沟通成本太高

DDD 落地步骤：

Step 1: 事件风暴（Event Storming）
- 把产品、开发、测试、运维拉到一起
- 把业务流程用领域事件列出来
- 比如：用户注册 → 发送欢迎邮件 → 初始化用户权益

Step 2: 找出聚合根（Aggregate Root）
- 聚合根是一致性边界
- 比如订单是聚合根，订单项是订单的一部分
- 修改订单项必须通过订单，不能直接改

Step 3: 划分限界上下文（Bounded Context）
- 语义边界，同一个词在不同上下文意思不一样
- 比如 "商品" 在商品中心是完整信息
- "商品" 在订单中心只是 ID + 名称 + 价格

Step 4: 上下文映射（Context Map）
- 上下游关系：比如订单是下游，商品是上游
- 防腐层（ACL）：上游变了，下游不受影响
- 开放主机服务（OHS）：上游提供标准接口

Step 5: 代码落地
- 接口层（Controller）
- 应用层（Service）：编排业务逻辑
- 领域层（Domain）：核心业务逻辑，实体、值对象、领域服务
- 基础设施层（Infra）：DB、RPC、缓存等

拆分常见坑：
1. 拆太细，服务间调用太多，性能差，分布式事务复杂
2. 循环依赖：A 调 B，B 调 A，死循环
3. 分布式事务没处理好，数据不一致
4. 没有灰度，一上线全挂

腾讯内部经验：
- 先粗后细，不要一开始就拆太碎
- 能用最终一致就不用强一致
- 服务治理要跟上：服务发现、熔断、限流、监控
```

---

### 27. 服务雪崩怎么应对？熔断、降级、限流？
```
服务雪崩：一个服务挂了，调用它的所有服务都跟着挂，级联故障

应对三板斧：限流、熔断、降级

1. 限流：保护自己，不让太多请求进来
   - 令牌桶：每秒放 N 个令牌，请求拿不到就拒绝
     优点：平滑，允许一定突发流量
   - 滑动窗口：1 秒分成 10 格，统计最近 1 秒请求数
     优点：精确，没有边界突刺问题
   - 漏桶：绝对平滑，不允许突刺
     优点：流量整形，适合下游能力固定

   腾讯实践：
   - 入口层（Nginx/API 网关）总限流
   - 服务层按接口限流
   - 用户维度限流，防止单个用户打满整个服务

2. 熔断：保护下游，下游挂了不要再打了
   熔断器状态机：
   Closed → 正常放行，统计失败率
     ↓ 失败率超过阈值
   Open → 直接拒绝请求，不调用下游
     ↓ 熔断时间到，放一个请求探测
   Half-Open → 探测成功就回 Closed，失败继续 Open

   腾讯实践：
   - 每个 RPC 客户端都有熔断器
   - 失败率阈值一般设 50%
   - 熔断时间 5-30 秒，指数退避
   - 半开状态放少量请求探测

3. 降级：宁可部分可用，不可完全不可用
   降级策略：
   - 开关降级：配置中心开关，非核心功能直接关
   - 降级返回默认值：比如推荐接口挂了，返回热门商品
   - 降级返回缓存值：返回 5 分钟前的旧数据，总比报错好
   - 简化流程：大促的时候跳过风控、埋点等非必要步骤

   降级分级：
   L1 轻度：关闭不重要的功能，比如评论、收藏
   L2 中度：关闭部分核心功能，比如推荐、搜索
   L3 重度：只保留最核心功能，比如下单、支付

雪崩应对完整流程：

1. 事前预防：
   - 所有服务都加限流、熔断、降级
   - 核心链路压测，知道每个服务的瓶颈
   - 混沌工程，定期注入故障演练

2. 事中处理：
   - 监控告警，发现异常第一时间知道
   - 先摘流量，把故障实例下线
   - 降级非核心功能，保住核心
   - 扩容能扛就扩容，不能扛就限流

3. 事后复盘：
   - 为什么会雪崩？根因是什么？
   - 为什么限流/熔断没起作用？阈值设太高？
   - 监控有没有覆盖到？为什么没提前告警？
   - 怎么避免下次再发生？

腾讯内部经验：
- 降级开关要预置，别等出事了才加
- 限流阈值要留余量，不要压到极限才限流
- 熔断不要太敏感，不然容易误杀
- 核心链路和非核心链路要隔离，别互相影响
```

---

### 28. 分布式事务怎么处理？TCC、Saga、本地消息表对比？
```
分布式事务场景：跨多个服务、跨多个数据库，要保证原子性

常用方案对比：

1. 两阶段提交 2PC / XA
   原理：
   - Prepare 阶段：所有参与者都就绪，回答 Yes/No
   - Commit 阶段：都 Yes 就都提交，有一个 No 就都回滚

   优点：强一致
   缺点：
   - 协调者单点故障
   - 同步阻塞，所有参与者都要等最慢的那个
   - 性能差，吞吐低

   适用：银行、支付等对一致性要求极高的场景
   腾讯内部用得不多，性能太差

2. TCC（Try-Confirm-Cancel）
   原理：
   - Try：尝试执行业务，预留资源
   - Confirm：真正执行，用预留的资源
   - Cancel：回滚，释放预留的资源

   优点：
   - 性能比 2PC 好
   - 最终一致
   - 失败可以重试

   缺点：
   - 业务侵入性大，每个接口都要写 Try/Confirm/Cancel 三个
   - 幂等性要自己保证
   - 空回滚、悬挂等问题要处理

   适用：需要可控，对性能要求较高的场景
   腾讯支付、订单系统有用

3. Saga
   原理：
   - 把大事务拆成多个本地事务
   - 每个本地事务对应一个补偿事务
   - 顺序执行，失败了就反向执行补偿事务
   - 正向：T1 → T2 → T3 → ...
   - 反向：Ci → ... → C3 → C2 → C1

   两种实现：
   - 协同式：每个服务调用下一个，自己失败了调用前一个的补偿
   - 编排式：有个协调者统一调度

   优点：
   - 一阶段提交，没有锁，性能好
   - 业务侵入比 TCC 小，只要写补偿
   - 长事务也能处理

   缺点：
   - 不是隔离的，中间状态别人能看到
   - 补偿逻辑要写好，不然补偿失败更麻烦

   适用：业务流程长，参与方多的场景
   腾讯电商、物流系统用得多

4. 本地消息表（最常用！）
   原理：
   1. 本地事务把业务数据和消息写进同一个数据库
      因为是同一个库，本地事务保证原子性
   2. 后台任务定时把消息发去 MQ
   3. 消费者消费消息，执行本地事务
   4. 消费成功删除消息，失败了重试

   优点：
   - 最简单，业务侵入最小
   - 性能好，没有分布式锁
   - 可靠，消息不丢

   缺点：
   - 消息表要自己维护
   - 只能最终一致

   适用：90% 的业务场景都能用！
   腾讯内部 80% 的分布式事务都是这个方案！

方案选择原则：
1. 能不用分布式事务就不用！能拆就拆，能最终一致就不用强一致
2. 一定要用的话，优先选本地消息表，最简单
3. 流程复杂、参与方多的用 Saga
4. 对一致性要求极高的才用 TCC 或 2PC

腾讯内部经验：
- 最终一致就够了，强一致需求没那么多
- 幂等性是基础，所有接口都要保证幂等
- 补偿逻辑要写好，还要有兜底人工处理
- 消息至少要保证 at-least-once
```

---

### 29. 消息积压怎么处理？百万级消息怎么快速消费？
```
消息积压处理步骤：

Step 1: 先止血，防止情况恶化
- 看消费者是不是挂了？能重启先重启
- 是不是消费逻辑有 bug？比如死循环？先回滚
- 是不是下游 DB/Redis 挂了？先把下游恢复了

Step 2: 加速消费
方案 1: 扩容消费者数量
- 只要 MQ 扛得住，消费端机器够，直接扩容
- 消费速度线性提升，10 倍消费就扩 10 倍
- 注意：下游也要能扛住！别把下游打挂了

方案 2: 批量消费
- 原来一条一条消费，现在攒 100 条批量处理
- 比如批量入库，批量 RPC 调用
- 均摊 IO 开销，速度提升 5-10 倍

方案 3: 并发消费
- 同一个消费线程里，多 goroutine 并发处理
- 处理 100 条可以同时开 10 个 goroutine 处理
- 注意控制并发度，别把下游打挂

Step 3: 极端情况，先切流
- 积压太多，几天都消费不完
- 把新消息切到新的 Topic，新消息正常消费
- 旧消息后台慢慢消费，不影响新业务

Step 4: 事后排查根因
- 为什么会积压？是消费太慢还是生产突然放量？
- 消费为什么慢？是逻辑慢还是下游慢？
- 为什么监控没告警？消息 lag 指标有吗？

百万级消息快速消费最佳实践：

1. 消费端优化：
- 批量消费，减少网络 round trip
- 并发消费，利用多核
- 消息过滤，没用的消息直接丢，不用处理
- 消费逻辑尽量轻，重逻辑异步处理

2. MQ 本身优化：
- Topic 分区数要够，分区数决定最大并发消费数
- Kafka 分区数一般设成 32 或 64，别太少
- 消息体尽量小，只传必要的 ID，别把整个对象塞进去
- 消费位点批量提交，别每条都 commit

3. 下游优化：
- 下游 DB 加缓存，减少查库
- 下游接口也支持批量
- 下游不行就先把消息存本地，慢慢处理

预防措施：
- 消息 lag 监控告警，超过阈值立刻告警
- 消费速度监控，和生产速度对比
- 消费者自动扩容，lag 大了自动加机器
- 消费失败要有死信队列，别卡主所有消息

腾讯内部经验：
- 积压了第一反应是扩容，最简单有效
- 批量消费是性价比最高的优化
- 千万不要为了快跳过幂等校验
- 一定要有死信队列，脏消息不能卡主所有消息
```

---

### 30. 服务发现原理？健康检查怎么实现？
```
服务发现完整流程：

1. 服务注册：
   - 服务启动时，把自己的 IP、Port、服务名注册到注册中心
   - 上报元数据：版本号、机房、权重、标签等
   - 定期发心跳，证明自己还活着

2. 服务发现：
   - 消费方启动时，从注册中心拉取目标服务的实例列表
   - 本地缓存实例列表，不用每次都去注册中心拉
   - 注册中心有变化时，推送更新给消费方

3. 负载均衡：
   - 消费方从本地缓存的实例列表里选一个调用
   - 常用算法：轮询、加权轮询、随机、一致性哈希、最少连接

4. 健康检查：
   - 实例不健康了就从列表里摘掉
   - 恢复了再加回来

注册中心对比：
| 方案 | CAP | 一致性 | 可用性 | 性能 | 特点 |
|-----|-----|-------|-------|-----|-----|
| ZooKeeper | CP | 强一致 | 一般 | 一般 | 重，功能多 |
| Consul | CP | 强一致 | 一般 | 一般 | 多数据中心 |
| Eureka | AP | 最终一致 | 高 | 好 | 简单，适合服务发现 |
| Nacos | AP+CP | 可配置 | 高 | 好 | 国内主流 |

腾讯内部用自研的命名服务，类似 Nacos

健康检查实现方式：

1. 心跳模式（最常用）
   - 服务端定期发心跳给注册中心
   - 注册中心一定时间没收到心跳就把实例摘掉
   - 优点：简单，不用注册中心主动连
   - 缺点：心跳间隔要设好，太长故障发现慢，太频繁注册中心压力大
   - 腾讯内部一般 5 秒一次心跳，15 秒没收到就摘

2. 主动探测模式
   - 注册中心主动探测服务的健康检查接口
   - 比如 HTTP GET /health，返回 200 就是健康
   - 优点：准确，能发现服务是不是真的能处理请求
   - 缺点：注册中心压力大，要维护很多连接

3. 客户端探测模式
   - 注册中心不检查，客户端自己探测
   - 客户端调用失败几次就把那个实例标记为不健康
   - 一段时间后再尝试
   - 优点：注册中心简单，客户端更准
   - 缺点：每个客户端都要维护状态

健康检查分级：
L1 进程检查：进程在不在，端口通不通 → 最基础
L2 业务检查：能不能正常处理请求，DB/Redis 连不连得上 → 业务可用
L3 负载检查：CPU/内存是不是太高，负载太高就别发请求了 → 负载保护

常见坑：
1. 健康检查太严格，偶发网络波动就把实例摘了
2. 健康检查太松，服务已经死了还在发请求
3. 心跳风暴，几万实例同时发心跳，注册中心扛不住
4. 注册中心挂了，整个服务发现都挂了 → 客户端缓存很重要

腾讯内部实践：
- 客户端缓存很重要，注册中心挂了不影响已有服务调用
- 心跳 + 客户端失败探测结合
- 实例摘除要留余量，别全摘完了
- 注册中心多机房部署，高可用
```

---

## 📝 手写代码题（腾讯常考）

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
    c.moveToHead(node)  // 最近访问过，移到头部
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
    
    // 新节点，加到头部
    newNode := &DListNode{key: key, value: value}
    c.cache[key] = newNode
    c.addToHead(newNode)
    
    // 超过容量，删除尾部节点（最久没访问的）
    if len(c.cache) > c.capacity {
        tail := c.removeTail()
        delete(c.cache, tail.key)
    }
}

// 辅助函数，都在锁内调用，不用再加锁
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

### 32. 实现一个简单的 Worker Pool
```go
type Pool struct {
    jobs    chan Job
    results chan Result
    wg      sync.WaitGroup
    quit    chan struct{}
}

type Job func() Result
type Result interface{}

func NewPool(workerCount int) *Pool {
    p := &Pool{
        jobs:    make(chan Job, 1000),
        results: make(chan Result, 1000),
        quit:    make(chan struct{}),
    }
    
    for i := 0; i < workerCount; i++ {
        p.wg.Add(1)
        go p.worker()
    }
    
    return p
}

func (p *Pool) worker() {
    defer p.wg.Done()
    
    for {
        select {
        case job := <-p.jobs:
            result := job()
            p.results <- result
        case <-p.quit:
            return
        }
    }
}

func (p *Pool) Submit(job Job) {
    p.jobs <- job
}

func (p *Pool) Results() <-chan Result {
    return p.results
}

func (p *Pool) Stop() {
    close(p.quit)
    p.wg.Wait()
    close(p.jobs)
    close(p.results)
}

// 使用示例：
func main() {
    pool := NewPool(100)  // 100 个 worker
    defer pool.Stop()
    
    // 提交 10000 个任务
    for i := 0; i < 10000; i++ {
        n := i
        pool.Submit(func() Result {
            time.Sleep(10 * time.Millisecond)
            return n * n
        })
    }
    
    // 收结果
    for i := 0; i < 10000; i++ {
        <-pool.Results()
    }
}
```

---

### 33. 实现一个简单的限流器（令牌桶）
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
        tokens:     capacity,  // 初始满桶
        lastUpdate: time.Now(),
    }
}

func (tb *TokenBucket) Allow() bool {
    tb.mu.Lock()
    defer tb.mu.Unlock()
    
    now := time.Now()
    elapsed := now.Sub(tb.lastUpdate)
    
    // 计算这段时间生成了多少令牌
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

// 也可以一次性拿 N 个令牌
func (tb *TokenBucket) AllowN(n int64) bool {
    tb.mu.Lock()
    defer tb.mu.Unlock()
    
    now := time.Now()
    elapsed := now.Sub(tb.lastUpdate)
    newTokens := int64(elapsed.Seconds()) * tb.rate
    
    if newTokens > 0 {
        tb.tokens = min(tb.tokens + newTokens, tb.capacity)
        tb.lastUpdate = now
    }
    
    if tb.tokens >= n {
        tb.tokens -= n
        return true
    }
    
    return false
}

func min(a, b int64) int64 {
    if a < b {