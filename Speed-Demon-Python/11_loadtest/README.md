# Python 服务负载测试指南

本目录提供完整的负载测试工具和方法，用于对 10_full_case 中的 Flask/FastAPI 服务进行性能压测。

## 目录结构

```
11_loadtest/
├── README.md              # 本文档 - 负载测试指南
├── locustfile.py          # Locust 分布式压测脚本
├── simple_loadtest.py     # 简单 requests 并发压测脚本
└── benchmark_plot.py      # 压测结果可视化脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install locust requests matplotlib numpy pandas
```

### 2. 启动待测试服务

在两个终端分别启动两个版本的服务：

```bash
# 终端1: 启动 v1_bad (有性能问题版本)
cd ../10_full_case/v1_bad
python app.py  # 运行在 http://localhost:5000

# 终端2: 启动 v2_good (优化版本)
cd ../10_full_case/v2_good
python app.py  # 运行在 http://localhost:5001
```

### 3. 运行简单压测

```bash
# 测试 v1_bad
python simple_loadtest.py --url http://localhost:5000 --concurrency 10 --requests 100

# 测试 v2_good
python simple_loadtest.py --url http://localhost:5001 --concurrency 10 --requests 100
```

### 4. 运行 Locust 压测（Web界面）

```bash
# 启动 Locust Web UI
locust -f locustfile.py --host=http://localhost:5000

# 浏览器访问: http://localhost:8089
```

### 5. 运行 Locust 压测（无头模式）

```bash
# 压测 v1_bad: 10并发，持续30秒
locust -f locustfile.py --host=http://localhost:5000 --headless -u 10 -r 2 --run-time 30s --csv=results_v1

# 压测 v2_good: 10并发，持续30秒
locust -f locustfile.py --host=http://localhost:5001 --headless -u 10 -r 2 --run-time 30s --csv=results_v2
```

### 6. 可视化压测结果

```bash
# 生成性能对比图表
python benchmark_plot.py --csv results_v1_stats.csv results_v2_stats.csv --output comparison.html
```

---

## 性能指标说明

### 核心性能指标

| 指标 | 说明 | 计算公式 |
|------|------|----------|
| **QPS** | 每秒请求数 | 总请求数 / 总耗时 |
| **RPS** | 每秒请求数（同QPS） | 同上 |
| **平均响应时间** | 所有请求的平均耗时 | 总耗时 / 总请求数 |
| **TP50** | 50%分位响应时间 | 排序后第50%位置的耗时 |
| **TP90** | 90%分位响应时间 | 排序后第90%位置的耗时 |
| **TP95** | 95%分位响应时间 | 排序后第95%位置的耗时 |
| **TP99** | 99%分位响应时间 | 排序后第99%位置的耗时 |
| **TP999** | 99.9%分位响应时间 | 排序后第99.9%位置的耗时 |
| **成功率** | 请求成功比例 | 成功请求数 / 总请求数 |
| **并发用户数** | 同时发送请求的用户数 | 配置的并发数 |
| **错误率** | 请求失败比例 | 失败请求数 / 总请求数 |

### 分位数（TP值）详解

**为什么需要 TP 值？**

平均响应时间会掩盖极端情况，TP 值能更好地反映真实用户体验：

- **TP50**：普通用户体验，大部分用户能达到的响应时间
- **TP90**：较好的用户体验，90%用户能达到
- **TP95**：优秀的用户体验，95%用户能达到
- **TP99**：极致用户体验，99%用户能达到
- **TP999**：企业级标准，99.9%用户能达到

**示例说明**：

假设1000个请求的响应时间排序后：
- TP50 = 100ms：500个请求 ≤ 100ms
- TP90 = 200ms：900个请求 ≤ 200ms
- TP99 = 500ms：990个请求 ≤ 500ms
- TP999 = 1000ms：999个请求 ≤ 1000ms

---

## 负载测试策略

### 1. 基准测试（Benchmark）

**目的**：建立性能基线

```bash
# 低并发下的基准测试
python simple_loadtest.py --url http://localhost:5000 --concurrency 5 --requests 50
```

**关注点**：
- 单线程/低并发下的最大QPS
- 基础响应时间
- CPU/内存使用率

### 2. 负载测试（Load Test）

**目的**：测试系统在预期负载下的表现

```bash
# 逐步增加并发
for concurrency in 10 20 50 100; do
    python simple_loadtest.py --url http://localhost:5000 --concurrency $concurrency --requests 200
done
```

**关注点**：
- 不同并发下的QPS变化
- 响应时间的变化趋势
- 系统资源使用情况

### 3. 压力测试（Stress Test）

**目的**：找到系统的极限

```bash
# 使用 Locust 进行高并发压测
locust -f locustfile.py --host=http://localhost:5000 --headless -u 200 -r 10 --run-time 2m
```

**关注点**：
- 系统的极限QPS
- 极限情况下的错误率
- 系统是否会崩溃

### 4. 耐久性测试（Soak Test）

**目的**：测试长时间运行的稳定性

```bash
# 长时间运行（例如1小时）
locust -f locustfile.py --host=http://localhost:5000 --headless -u 20 -r 1 --run-time 1h
```

**关注点**：
- 内存泄漏（v1_bad版本尤其需要关注）
- 性能是否随时间下降
- 资源是否持续增长

### 5. 对比测试（Comparison）

**目的**：对比优化前后的性能差异

```bash
# 压测 v1_bad
locust -f locustfile.py --host=http://localhost:5000 --headless -u 10 -r 2 --run-time 30s --csv=v1_bad

# 压测 v2_good（先清空缓存）
curl http://localhost:5001/cache/clear
locust -f locustfile.py --host=http://localhost:5001 --headless -u 10 -r 2 --run-time 30s --csv=v2_good

# 生成对比图表
python benchmark_plot.py --csv v1_bad_stats.csv v2_good_stats.csv --output comparison.html
```

---

## simple_loadtest.py 使用说明

### 命令行参数

```bash
python simple_loadtest.py [参数]

参数说明:
  --url URL              目标URL (默认: http://localhost:5000)
  --endpoint ENDPOINT    测试端点 (默认: /product/1)
  --concurrency N        并发数 (默认: 10)
  --requests N           总请求数 (默认: 100)
  --timeout SEC          超时时间秒 (默认: 30)
  --output FILE          输出JSON结果文件
  --compare              对比两个服务的性能
  --url1 URL             第一个服务URL
  --url2 URL             第二个服务URL
```

### 使用示例

```bash
# 基本压测
python simple_loadtest.py --url http://localhost:5000 --concurrency 20 --requests 200

# 保存结果到文件
python simple_loadtest.py --url http://localhost:5000 --output result_v1.json

# 对比两个服务
python simple_loadtest.py --compare --url1 http://localhost:5000 --url2 http://localhost:5001
```

### 输出示例

```
======================================================================
负载测试结果
======================================================================

基本信息:
  并发用户数:          10
  总请求数:            100
  成功请求数:          100
  失败请求数:          0
  成功率:              100.00%

时间统计:
  总耗时:              15.23s
  QPS:                 6.57
  平均响应时间:        1523.45ms
  最小响应时间:        1450.23ms
  最大响应时间:        2100.56ms

分位数响应时间:
  TP50:                1500.12ms
  TP90:                1850.34ms
  TP95:                1950.45ms
  TP99:                2050.50ms
  TP999:               2100.56ms

状态码统计:
  200:                 100
```

---

## locustfile.py 使用说明

### 启动 Web 界面

```bash
# 基本启动
locust -f locustfile.py

# 指定目标主机
locust -f locustfile.py --host=http://localhost:5000

# 自定义端口
locust -f locustfile.py --host=http://localhost:5000 --web-port 8088
```

### 无头模式运行

```bash
# 10并发，每秒启动2个用户，运行30秒
locust -f locustfile.py --host=http://localhost:5000 --headless -u 10 -r 2 --run-time 30s

# 50并发，每秒启动5个用户，运行2分钟，保存CSV结果
locust -f locustfile.py --host=http://localhost:5000 --headless -u 50 -r 5 --run-time 2m --csv=loadtest_result

# 100并发，运行5分钟，每10秒输出一次统计
locust -f locustfile.py --host=http://localhost:5000 --headless -u 100 -r 10 --run-time 5m --print-stats-every 10
```

### 测试场景

脚本包含三个测试场景：

1. **商品详情查询**（70%权重）：主要业务场景
2. **健康检查**（20%权重）：轻量级请求
3. **缓存操作**（10%权重）：清空缓存（仅v2_good）

### 自定义测试

编辑 `locustfile.py` 中的 `UserBehavior` 类来自定义测试场景：

```python
class UserBehavior(TaskSet):
    @task(7)  # 权重越高，执行频率越高
    def get_product(self):
        product_id = random.randint(1, 5)
        self.client.get(f"/product/{product_id}")
```

---

## benchmark_plot.py 使用说明

### 功能特性

1. **响应时间分布图**：直方图 + 核密度估计
2. **QPS 曲线图**：QPS 随时间的变化
3. **分位数对比图**：TP50/TP90/TP99/TP999 对比
4. **并发性能图**：不同并发下的性能表现
5. **交互式图表**：使用 Plotly 生成可交互的 HTML 图表

### 命令行参数

```bash
python benchmark_plot.py [参数]

参数说明:
  --csv FILES       一个或多个CSV结果文件（空格分隔）
  --json FILES      一个或多个JSON结果文件（空格分隔）
  --output FILE     输出HTML文件路径 (默认: benchmark_report.html)
  --title TITLE     图表标题 (默认: 负载测试性能报告)
```

### 使用示例

```bash
# 从CSV生成报告
python benchmark_plot.py --csv results_v1_stats.csv results_v2_stats.csv --output comparison.html

# 从JSON生成报告
python benchmark_plot.py --json result_v1.json result_v2.json --output comparison.html

# 混合使用
python benchmark_plot.py --csv results_v1_stats.csv --json result_v2.json --output comparison.html
```

### 图表说明

#### 1. 响应时间分布

- **X轴**：响应时间（毫秒）
- **Y轴**：请求数量
- **曲线**：核密度估计，显示响应时间的概率分布
- **用途**：判断响应时间是否稳定，是否有长尾现象

#### 2. QPS 变化曲线

- **X轴**：时间（秒）
- **Y轴**：每秒请求数
- **用途**：观察系统吞吐量随时间的变化，是否有性能波动

#### 3. 分位数对比

- **X轴**：分位指标（TP50/TP90/TP95/TP99/TP999）
- **Y轴**：响应时间（毫秒）
- **用途**：对比不同版本在各个分位上的表现

#### 4. 并发性能对比

- **X轴**：并发用户数
- **Y轴**：QPS 或 响应时间
- **用途**：观察系统在不同并发下的性能表现

---

## 结果分析方法

### 1. 性能瓶颈识别

#### CPU 瓶颈

**特征**：
- QPS 随并发增加而达到平台期
- CPU 使用率接近 100%
- 响应时间持续上升

**排查方法**：
```bash
# 使用 py-spy
py-spy top --pid <进程ID>

# 使用 cProfile
python -m cProfile -s cumulative app.py
```

#### 内存瓶颈

**特征**：
- 内存持续增长（特别是 v1_bad 版本）
- GC 频率增加
- 服务最终 OOM

**排查方法**：
```bash
# 使用 tracemalloc
import tracemalloc
tracemalloc.start()

# 使用 objgraph
import objgraph
objgraph.show_growth()
```

#### I/O 瓶颈

**特征**：
- CPU 使用率低，但响应时间高
- QPS 上不去
- 数据库/网络 I/O 等待时间长

**排查方法**：
- 检查数据库查询时间
- 检查外部HTTP调用耗时
- 检查网络延迟

### 2. 性能对比分析

#### 对比表格模板

| 指标 | v1_bad | v2_good | 提升幅度 |
|------|--------|---------|----------|
| QPS | X | Y | (Y-X)/X |
| 平均响应时间 | A | B | (A-B)/A |
| TP90 | C | D | (C-D)/C |
| TP99 | E | F | (E-F)/F |
| 成功率 | G | H | H-G |

#### 性能提升计算公式

```
QPS提升率 = (QPS_new - QPS_old) / QPS_old * 100%

响应时间减少率 = (RT_old - RT_new) / RT_old * 100%
```

### 3. 拐点分析

寻找系统的**性能拐点**：

1. **线性增长区**：并发增加，QPS线性增加
2. **亚线性增长区**：并发增加，QPS增加放缓
3. **平台区**：并发增加，QPS不再增加
4. **下降区**：并发增加，QPS开始下降

**拐点位置**就是系统的最优并发数。

### 4. 稳定性分析

长时间压测后观察：

- **内存趋势**：是否持续增长？
- **QPS趋势**：是否下降？
- **错误率**：是否增加？
- **响应时间**：是否变长？

---

## 压测最佳实践

### 1. 压测前准备

- [ ] 确保测试环境稳定
- [ ] 关闭其他干扰进程
- [ ] 监控系统资源（CPU/内存/网络/磁盘）
- [ ] 准备好监控工具
- [ ] 设定明确的测试目标

### 2. 压测过程中

- [ ] 从低并发开始，逐步增加
- [ ] 记录每次压测的配置和结果
- [ ] 观察系统日志，排查错误
- [ ] 监控服务状态，避免服务崩溃
- [ ] 注意压测机本身的性能瓶颈

### 3. 压测后分析

- [ ] 整理所有测试数据
- [ ] 生成可视化图表
- [ ] 识别性能瓶颈
- [ ] 对比优化前后的差异
- [ ] 撰写测试报告

### 4. 注意事项

⚠️ **重要提醒**：

1. **不要在生产环境压测**：除非你知道自己在做什么
2. **控制压测强度**：避免压垮服务或网络
3. **多次测试取平均值**：单次测试可能有偶然性
4. **预热系统**：压测前先发送一些请求让系统热身
5. **记录环境信息**：Python版本、系统配置、依赖版本等
6. **清理缓存**：对比测试时确保缓存状态一致
7. **注意内存泄漏**：长时间压测时监控内存使用（特别是 v1_bad）

---

## 常见问题

### Q: 为什么压测结果不稳定？

**可能原因**：
1. 系统资源波动
2. 网络不稳定
3. 缓存效应
4. GC 暂停
5. 其他进程干扰

**解决方法**：
- 多次测试取平均值
- 控制测试环境
- 延长测试时间
- 排除缓存影响

### Q: 如何判断性能优化是否有效？

**判断标准**：
1. 相同并发下 QPS 提升 > 20%
2. 相同 QPS 下响应时间下降 > 20%
3. TP90/TP99 有明显改善
4. 系统能够支撑更高的并发

### Q: 压测时服务报错怎么办？

**排查步骤**：
1. 检查服务日志
2. 降低并发数
3. 检查超时设置
4. 检查资源限制
5. 检查依赖服务状态

### Q: 如何模拟真实用户行为？

**建议**：
1. 使用不同的商品ID，避免完全相同的请求
2. 设置合理的思考时间（think time）
3. 混合不同类型的请求
4. 模拟用户的访问模式

---

## 工具对比

| 工具 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **simple_loadtest** | 简单易用，无需额外依赖 | 功能简单，分布式支持弱 | 快速验证，简单对比 |
| **Locust** | 功能强大，Web界面，分布式 | 需要安装，学习曲线 | 正式压测，复杂场景 |
| **wrk** | 高性能，C编写 | 无Python集成，Lua脚本 | 极限性能测试 |
| **ab** | Apache自带，简单 | 功能有限，不支持并发 | 快速基准测试 |

---

## 参考资料

- [Locust 官方文档](https://docs.locust.io/)
- [HTTP 负载测试最佳实践](https://www.brendangregg.com/blog/2017-04-27/linux-performance-analysis.html)
- [性能测试入门指南](https://www.softwaretestinghelp.com/performance-testing-guide/)
- [Python 性能优化](https://docs.python.org/3/faq/design.html)

---

**Happy Testing!** 🚀
