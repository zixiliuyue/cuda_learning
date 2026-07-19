# 🚀 Speed-Demon - Go 性能分析与调优全指南

> 打造求职/面试导向的 Go 性能优化技能树

## 📚 教程结构

### 第一部分：核心理论 (01_theory/)

| 文件 | 内容 | 面试关键词 |
|-----|------|-----------|
| **01_pprof_theory.md** | pprof 采样原理、CPU/Heap/Goroutine/Mutex profile 机制 | 采样原理、采样偏差 |
| **02_memory_model.md** | Go 内存模型、逃逸分析、T cmalloc 继承、内存对齐 | 逃逸分析、内存分配、T cmalloc |
| **03_scheduler_theory.md** | G-M-P 模型、work stealing、抢占式调度、调度延迟 | GMP、work stealing、抢占 |
| **04_gc_theory.md** | 三色标记、写屏障、GC 调优、pacing 算法 | 三色标记、写屏障、GC 调优 |

---

### 第二部分：实操案例 (02_practice/)

| 目录 | 内容 | 面试关键词 |
|-----|------|-----------|
| **01_cpu_profiling** | CPU 热点定位、火焰图、benchmark、fibonacci 优化案例 | top、list、web、benchstat |
| **02_memory_profiling** | 内存泄漏、分配热点、sync.Pool 实战 | alloc_space、inuse_space、逃逸 |
| **03_goroutine_leak** | goroutine 泄漏 4 种模式、排查实战、context 取消 | chan 阻塞、for+select、泄露排查 |
| **04_lock_contention** | mutex/block profile、锁分片、singleflight、读写锁权衡 | 锁竞争、分片、atomic |
| **05_execution_trace** | trace 视图解析、调度延迟分析、GC Mark Assist、syscall 阻塞 | View trace、scheduler latency |
| **06_benchmark** | benchmark 最佳实践、避免编译器优化、对比测试 | b.ResetTimer、RunParallel、benchstat |

---

### 第三部分：业务场景优化 (03_business/)

| 目录 | 内容 | 面试关键词 |
|-----|------|-----------|
| **01_database** | N+1 查询、事务反模式、全表扫描、索引优化、慢查询分析 | 索引、事务、explain、慢查询 |
| **02_cache** | 缓存穿透/击穿/雪崩、大 Key 问题、热点 Key、本地缓存 + Redis 二级缓存 | 三个缓存问题、bigkey、singleflight |
| **03_grpc** | 连接池配置、并发调用、超时控制、重试退避、长连接保活 | HTTP/2、连接池、超时策略 |
| **04_concurrent** | 并发模式对比：Worker Pool、Pipeline、Fan-in/Fan-out、Semaphore | Worker Pool、背压、并发模式 |

---

### 第四部分：面试准备 (04_interview/)

| 文件 | 内容 | 关键词 |
|-----|------|-------|
| **01_methodology.md** | 性能问题排查方法论、假设驱动、工具使用优先级、优化金字塔 | STAR 法则、四个黄金指标、排查流程 |
| **02_anti_patterns.md** | 10 个经典性能反模式 + 真实故障案例 + 面试讲述模板 | goroutine 泄漏、time after、事务反模式 |
| **03_business_patterns.md** | DB / 缓存 / gRPC 业务场景优化、面试问题回答模板 | N+1、缓存三问题、连接池、全链路压测 |

---

## 🎯 性能问题排查流程（面试必背）

```
发现性能问题
    ↓
1️⃣ 看监控，缩小范围
    - 四个黄金指标：延迟、流量、错误、饱和度
    - 是所有接口都慢？还是某个接口？
    - CPU 高？内存高？GC 频繁？
    ↓
2️⃣ 看链路追踪，定位具体慢在哪
    - 找 trace example，看时间分布
    - DB？Redis？下游服务？还是代码本身？
    ↓
3️⃣ 用工具深入分析
    - CPU 高 → pprof cpu profile
    - 内存高 → pprof heap profile
    - 锁竞争 → mutex profile
    - goroutine 多 → goroutine profile
    - 调度/GC 问题 → execution trace
    ↓
4️⃣ 实施优化，按影响排序
    架构优化（影响最大）
        → 加缓存、批量操作、同步改异步
    应用优化
        → 算法优化、减少分配、并发优化
    微优化（影响最小）
        → 循环优化、内存对齐、内联
    ↓
5️⃣ 验证 + 复盘
    - 压测验证优化效果
    - 加监控告警，避免再犯
    - 写入团队故障手册
```

---

## 🛠️ 工具速查表

| 问题现象 | 优先使用的工具 | 关键命令 |
|---------|---------------|---------|
| **CPU 高** | pprof cpu | `go tool pprof http://.../profile` → `top` → `list` |
| **内存高/泄漏** | pprof heap | `go tool pprof http://.../heap` → `inuse_space` |
| **分配多/GC 频繁** | pprof alloc | `go tool pprof -alloc_space` |
| **goroutine 泄漏** | pprof goroutine | `curl http://.../goroutine?debug=2` |
| **锁竞争严重** | pprof mutex | `runtime.SetMutexProfileFraction(1)` |
| **调度/GC/阻塞** | trace | `curl http://.../trace?seconds=10` → `go tool trace` |
| **基准测试** | benchmark | `go test -bench=. -benchmem -count=5` |
| **对比优化效果** | benchstat | `benchstat old.txt new.txt` |

---

## 💯 面试高频问题清单

### Go 基础

- [ ] 说说 Go 的 GMP 调度模型？
- [ ] Go 的 GC 原理是什么？有什么特点？
- [ ] 什么是逃逸分析？什么情况下变量会逃到堆上？
- [ ] channel 有什么用？无缓冲和有缓冲的区别？
- [ ] context 有什么用？怎么传递取消信号？
- [ ] sync.Pool 的原理？适合什么场景？不适合什么场景？

### 性能分析

- [ ] 怎么排查 CPU 高的问题？
- [ ] 怎么排查内存泄漏？
- [ ] 怎么发现 goroutine 泄漏？有哪些常见原因？
- [ ] 怎么定位锁竞争？
- [ ] execution trace 能看到什么？怎么用？
- [ ] 怎么做基准测试？要注意什么坑？

### 业务场景

- [ ] 缓存穿透、击穿、雪崩的区别和解决方案？
- [ ] N+1 查询是什么？怎么发现？怎么优化？
- [ ] 怎么做接口超时控制？
- [ ] 怎么保证高并发下服务不雪崩？
- [ ] 限流的算法有哪些？各有什么优缺点？
- [ ] 大促前你们会做什么准备？
- [ ] 全链路压测怎么做？

### 真实案例

- [ ] 说一个你印象最深的性能问题排查经历？
- [ ] 你做过的最成功的性能优化是什么？带来了多大提升？
- [ ] 有没有因为性能问题导致的线上故障？怎么处理的？
- [ ] 你们线上服务的 P99 延迟一般是多少？怎么做到的？

---

## 🚀 快速开始

### 第一步：看理论（2 天）

```bash
cd 01_theory
# 按顺序看 4 个 md 文件，理解核心原理
```

### 第二步：跑实操（3 天）

```bash
# 每个目录都跑一下，抓 profile 分析
cd 02_practice/01_cpu_profiling
go run main.go

# 另一个终端
curl http://localhost:6060/debug/pprof/profile?seconds=10 > cpu.pprof
go tool pprof cpu.pprof
```

### 第三步：业务场景（2 天）

```bash
cd 03_business
# 理解 DB/缓存/gRPC 常见性能坑和优化方案
```

### 第四步：面试准备（1 天）

```bash
cd 04_interview
# 看方法论和反模式
# 结合自己的经历，准备 2-3 个性能问题故事
```

**总时间：约 1 周 → 从性能优化小白到面试有料！**

---

## 📊 性能优化金字塔

```
        ┌───────────────────────────────────┐
        │       架构优化（10x-100x）          │ ← 影响最大
        │  - 加缓存                            │
        │  - 同步改异步                         │
        │  - 批量操作                           │
        │  - 减少依赖调用                       │
        └───────────────────────────────────┘
        ┌───────────────────────────────────┐
        │       应用优化（2x-10x）            │
        │  - 算法优化（O(n²)→O(n)）            │
        │  - 减少内存分配                       │
        │  - 并发优化（Worker Pool）           │
        │  - 锁优化（缩小范围、分片）           │
        └───────────────────────────────────┘
        ┌───────────────────────────────────┐
        │       微优化（1.1x-2x）             │ ← 影响最小
        │  - 循环优化                           │
        │  - 内存对齐                           │
        │  - 内联                               │
        │  - 避免反射                           │
        └───────────────────────────────────┘
```

**面试加分回答：**
> "我觉得性能优化优先级很重要。首先看架构层面能不能优化，这往往能带来数量级的提升；
> 然后看应用层面，比如算法、并发、锁；最后才是微优化。
> 微优化不是不重要，而是应该放在最后，先解决主要矛盾。"

---

## 📚 延伸阅读

### 官方文档
- [Go Diagnostics](https://golang.org/doc/diagnostics)
- [pprof 博客](https://blog.golang.org/pprof)
- [Go 语言设计与实现](https://draveness.me/golang/)

### 经典书籍
- **《Go 语言高性能编程》** - 实战向
- **《Go 专家编程》** - 底层原理
- **《SRE: Google 运维解密》** - 稳定性、监控、排障

### 推荐博客
- [Dave Cheney - High Performance Go](https://dave.cheney.net/high-performance-go-workshop)
- [Go 101](https://go101.org/article/table-of-content.html)
- [脑壳 Go 语言专栏](https://www.kancloud.cn/aceld/golang)

---

## 🎓 学习建议

1. **动手是关键** - 不要只看文档，一定要自己跑 demo，抓 profile 看
2. **对比思考** - 优化前后一定要对比数据，用数据说话
3. **系统思维** - 性能问题往往是系统级的，不要只盯着一行代码
4. **准备故事** - 面试时，一个好的真实案例胜过 10 个理论知识点

---

## ⚠️ 免责声明

本教程的代码示例旨在演示**问题模式**和**优化思路**，生产环境使用请结合实际场景充分测试。
性能优化没有银弹，没有绝对正确的方案，只有适合当前场景的权衡。

---

**最后更新：2024 年**

**如果这个教程帮到了你，欢迎给个 Star ⭐️**
