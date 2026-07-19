# 👁️ 可观测性实战 - 性能问题从发现到定位

> **面试必问**："线上接口突然变慢了，你怎么排查？"

---

## 🚀 快速开始：5 分钟搭好监控

### 1. 启动示例服务

```bash
cd 07_observability
go run metrics_example.go
```

服务启动后，访问：
- **API 接口**: http://localhost:8080/api/product?id=123
- **Metrics 端点**: http://localhost:8080/metrics
- **Stats**: http://localhost:8080/stats

### 2. 启动 Prometheus（Docker 方式）

创建 `prometheus.yml` 配置文件：
```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'go-app'
    static_configs:
      - targets: ['host.docker.internal:8080']
```

启动 Prometheus：
```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

访问 Prometheus: http://localhost:9090

### 3. 启动 Grafana（Docker 方式）

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

访问 Grafana: http://localhost:3000 (默认账号密码: admin/admin)

### 4. 导入仪表盘

1. 登录 Grafana
2. 点击左侧菜单：**Dashboards** → **New** → **Import**
3. 上传 `example_grafana_dashboard.json` 文件，或粘贴 JSON 内容
4. 选择 Prometheus 数据源，点击 Import

### 5. 生成测试流量

新开一个终端，压测接口：
```bash
hey -n 10000 -c 100 http://localhost:8080/api/product?id=123
```

然后去 Grafana 看实时监控效果！

---

## 📊 Metrics 示例代码说明

`metrics_example.go` 包含以下核心功能：

### 1. HTTP 中间件 Metrics

```go
// 请求总数
http_requests_total{path, method, status}

// 请求延迟直方图
http_request_duration_seconds{path, method}

// 正在处理的请求数
http_request_in_flight
```

### 2. 业务 Metrics

```go
// 缓存命中/未命中
cache_hit_total
cache_miss_total

// DB 查询延迟
db_query_duration_seconds{operation}
```

### 3. Go Runtime Metrics

自动暴露的 Go 运行时指标：
- `go_goroutines` - Goroutine 数量
- `go_memstats_alloc_bytes` - 内存分配
- `go_gc_duration_seconds` - GC 耗时
- 等等...

---

## ⭐ 四个黄金指标 Four Golden Signals

**面试直接说这 4 个，直接亮瞎面试官！**

| 指标 | 说明 | 看什么 |
|-----|------|--------|
| **1. 延迟 (Latency)** | 请求处理时间 | P50/P95/P99/P999，不要只看平均！ |
| **2. 流量 (Traffic)** | 每秒请求数 | QPS，看是业务高峰还是异常？ |
| **3. 错误率 (Errors)** | 失败请求比例 | 5xx/4xx/业务错误码占比 |
| **4. 饱和度 (Saturation)** | 系统多满？ | CPU/内存/连接池/磁盘使用率 |

---

## 🚨 性能排查的标准流程

### Step 1: 缩小范围（1 分钟搞定）

```
问题："用户反馈接口慢"

  先看这 4 个问题：
    ├── 是所有接口还是某几个？
    │       ↓
    ├── 什么时候开始的？有发布？有流量高峰？
    │       ↓
    ├── 错误率涨了没？还是只是慢？
    │       ↓
    └── 哪个指标先异常？CPU？内存？网络？DB？
```

**面试加分回答**：
> "我一般先从全局监控大盘入手，先看四个黄金指标，把问题范围缩小到：
> 是 CPU 问题？内存问题？还是下游 DB/Redis 问题？
> 而不是上来就抓 profile 盲猜。"

---

### Step 2: 逐层深入定位

```
监控发现异常
    ↓
┌─────────────────────────────────────────────┐
│ 先看应用层指标                             │
│ ─────────────────────────────────────────   │
│ CPU 高不高？ → 抓 CPU profile              │
│ 内存涨不涨？→ 抓 heap profile               │
│ Goroutine 暴涨？ → goroutine profile       │
└───────────────┬─────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│ 再看 Go 运行时                             │
│ ─────────────────────────────────────────   │
│ GC 次数？停顿时间？→ trace + gc trace     │
│ 线程数暴涨？ → 看上下文切换次数？            │
└───────────────┬─────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│ 最后看下游依赖                              │
│ ─────────────────────────────────────────   │
│ DB CPU/连接池/慢查询？                      │
│ Redis 命中率/QPS？                         │
│ 其他 RPC 调用延迟？                        │
└─────────────────────────────────────────────┘
```

---

## 📊 Prometheus 关键指标大盘

### 1. 必须有的 Go Metrics

先加 Prometheus 客户端：
```go
import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
    "github.com/prometheus/client_golang/prometheus/collectors/go"
)

func main() {
    // 自动暴露 Go Runtime 指标！
    prometheus.MustRegister(collectors.NewGoCollector())
    
    http.Handle("/metrics", promhttp.Handler())
}
```

### 2. 关键的 Go Runtime 指标

| PromQL | 说明 | 告警阈值 |
|--------|------|---------|
| `go_goroutines` | Goroutine 数量 | > 10000 或者持续上涨 → 泄露！ |
| `go_memstats_alloc_bytes` | 当前分配内存 | 持续上涨 OOM 风险 |
| `go_memstats_heap_objects` | 堆对象数 | 持续上涨 |
| `rate(go_gc_duration_seconds_sum[5m])` | GC 频率 | 太高说明分配多 |
| `go_threads` | OS 线程数 | > 500 检查上下文切换 |

---

### 3. 业务接口指标（RED 方法）

```go
// 三个核心指标：Rate（QPS)、Errors（错误率）、Duration（延迟）
var (
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total HTTP requests",
        },
        []string{"path", "method", "status"},
    )
    
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds",
            Help: "HTTP request duration",
            Buckets: []float64{0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
        },
        []string{"path", "method"},
    )
)
```

**怎么用这三个指标排查？**

1. **Rate**：`rate(http_requests_total[5m]) by path`
   - 是不是某个接口流量突增？

2. **Errors**：`rate(http_requests_total{status=~"5.."}[5m])`
   - 错误率是不是涨了？

3. **Duration**：`histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))`
   - P99 延迟是不是暴涨了？

---

## 🚨 常见异常模式识别

### 模式 1: CPU 100%，QPS 下降

```
症状：
├── CPU 从 40% → 100%
├── 但 QPS 反而降了
└── GC 次数暴涨

可能原因：
├── 死循环？ → trace 看用户态时间
├── 正则表达式写得差？ → CPU profile 看 regexp 占比高
├── 大量小对象分配导致 GC 忙？ → alloc profile
└── JSON 序列化/反序列化爆炸？
```

### 模式 2: 内存持续上涨，OOM

```
症状：
├── go_memstats_alloc_bytes 线性上涨
├── Goroutine 数量也线性上涨
└── 最后被 Linux OOM Killer 杀了

可能原因：
├── Goroutine 泄漏！ → goroutine profile 看都堵在哪？
│   ├── channel 永远读不到
│   ├── for+select 没退出条件
│   └── context 没取消
├── 全局 map 只加不删？
├── sync.Pool 被误用？
└── 内存泄漏？ heap profile 对比
```

### 模式 3: 延迟周期性尖刺

```
症状：
├── P99 每隔 10 秒尖刺一次
└── 正好和 GC 周期吻合

可能原因：
├── GC STW 停顿太长
├── 大对象太多，标记扫描慢
├── 分配速率太快，GC 跟不上
└── GOGC 设得太低，GC 太频繁
```

### 模式 4: 锁竞争，CPU 不高但 QPS 上不去

```
症状：
├── CPU 只用了 30%，但 QPS 上不去
├── 延迟却很高
└── Goroutine 数量也很高

可能原因：
├── 全局大锁！ → mutex profile 看锁等待时间
├── DB 连接池满了，都在等连接
├── 下游服务太慢，都在等响应
└── channel 无缓冲，都阻塞在 send/recv
```

---

## 💯 面试回答模板：线上接口突然慢了怎么排查？

> **面试官问：线上有个接口 P99 从 50ms 涨到 500ms，怎么排查？**

> **你回答：**

> "我一般按这几步排查：

> **第一步：先看四个黄金指标**
> 先看监控大盘，先搞清楚：
> - 是所有接口都慢，还是只有这一个？
> - 是什么时候开始的？那个时间点有没有发布？流量有没有涨？
> - 错误率有没有涨？是变慢还是报错了？
> - CPU/内存/DB/Redis 哪个指标先异常？

> **第二步：看链路追踪找瓶颈点**
> 捞一个慢请求的 trace，看时间都花在哪一步了：
> - 是 DB 慢？是哪个 SQL？有没有全表扫描？
> - 是 Redis 慢？命中率低？还是连接池满了？
> - 是调用其他下游服务慢？
> - 还是业务代码本身慢？

> **第三步：用 profile 深入定位**
> - 如果 CPU 高：抓 CPU profile 看热点函数在哪？
> - 如果内存涨：抓 heap profile 看哪分配多？
> - 如果锁看起来有问题：抓 mutex profile 看锁等待时间？
> - 如果怀疑 GC：抓 trace 看 GC 停顿和 Mark Assist？
> - Goroutine 涨：抓 goroutine profile 看都堵在哪？

> **第四步：定位根因，修复验证**
> 找到根因后修复，然后：
> - 先在测试环境压测确认优化有效
> - 灰度发布观察核心指标有没有回去
> - 最后加入 checklist/code review 避免再犯

> **最后复盘总结**：
> 有没有监控告警能不能更早发现？要不要加什么新的告警？

---

## 🚨 告警配置建议

这些告警有了，性能问题你比用户先知道！

| 告警项 | 触发条件 | 级别 |
|--------|---------|------|
| Goroutine 数量突增 | `increase(go_goroutines[10m]) > 5000` | P1 |
| P99 延迟上涨 | `P99 > 500ms` AND `P99 > 2 * 7天均值` | P1 |
| 错误率 | `5xx错误率 > 5%` | P1 |
| GC 停顿太久 | `go_gc_duration_seconds{quantile="0.99"} > 100ms` | P2 |
| 内存使用率 | `内存 > 80%` 且持续上涨 | P2 |
| CPU 使用率 | `CPU > 80%` 持续 5 分钟 | P2 |

---

## ✅ 排查 Checklist

线上出问题时，按这个勾：

- [ ] **先看时间线**：问题开始时间和什么吻合？发布？配置变更？流量高峰？
- [ ] **四个黄金指标都看了？** 延迟/流量/错误/饱和度？
- [ ] **所有接口还是单个？** 缩小范围！
- [ ] **错误码是什么？** 5xx？4xx？还是业务错误？
- [ ] **CPU 高不高？用户态还是内核态？**
- [ ] **GC 正常吗？次数/停顿时间有没有涨？**
- [ ] **Goroutine 有没有涨？都卡在什么调用栈？**
- [ ] **DB 慢查询有没有涨？连接池满了没？**
- [ ] **Redis 命中率有没有降？慢命令？**
- [ ] **有没有 trace 可以看慢请求全链路？**

---

## 🎯 面试官最爱问的几个问题

**Q: 怎么判断是应用的问题，还是 DB 的问题？**
> "看 DB 本身的监控：CPU、慢查询数、连接池使用率。
> 再看应用端的 DB 调用延迟指标。
> 如果 DB 本身 CPU 高有慢查：是 DB 的问题。
> 如果 DB 正常但应用看 DB 调用延迟高：是网络或者连接池问题。"

**Q: 怎么判断是不是 GC 导致的尖刺？**
> "看两个指标：1. GC 时间和延迟尖刺的时间点是否吻合？
> 2. trace 里看 Mark Assist 时间占比高不高？
> 如果用户 goroutine 大量时间在帮 GC 标记，就是 GC 影响。"

**Q: 线上能直接抓 profile 吗？会不会影响性能？**
> "CPU profile 10 秒采样大概 2-5% 性能影响，一般可以接受。
> heap profile 影响更小，因为是统计抽样。
> mutex/block profile 影响不大，但默认关闭，需要 runtime 设置。
> trace 影响稍大 5-10%，生产环境谨慎用。"

---

## 🚀 总结：性能排查心法

1. **先宏观，后微观**
   - 先看监控大盘，别上来就抓 profile 瞎看！

2. **找变化点**
   - 问题开始时发生了什么？发布？流量涨？配置变了？

3. **大胆假设，小心求证**
   - 先猜可能原因，再用数据验证，不要反过来！

4. **一次只改一个变量**
   - 验证优化效果时，一次改多点你就不知道哪个生效了！

5. **能复现就不是问题！**
   - 能在测试环境稳定复现 = 离解决不远了！

---

## 📁 本目录文件说明

| 文件 | 说明 |
|------|------|
| `metrics_example.go` | Prometheus Metrics 完整示例代码 |
| `example_grafana_dashboard.json` | Grafana 仪表盘配置 |
| `README.md` | 本文档 |
