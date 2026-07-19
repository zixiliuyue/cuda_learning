# Python服务可观测性指南

## 概述

可观测性（Observability）是现代微服务架构中的关键概念，包含三个核心支柱：
- **指标（Metrics）** - 聚合的数值数据
- **日志（Logs）** - 离散的事件记录
- **追踪（Traces）** - 请求的端到端链路

本目录提供Python服务可观测性的完整实现示例。

## 目录结构

```
12_observability/
├── README.md              # 本文档
├── app_with_metrics.py    # FastAPI + Prometheus指标示例
├── prometheus.yml         # Prometheus配置
├── grafana_dashboard.json # Grafana仪表盘配置
└── tracing_example.py     # OpenTelemetry分布式追踪示例
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn prometheus-client opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi
```

### 2. 启动应用

```bash
# 启动带指标的FastAPI应用
python app_with_metrics.py

# 启动分布式追踪示例
python tracing_example.py
```

### 3. 启动Prometheus和Grafana

使用Docker Compose启动监控栈：

```yaml
version: '3'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 关键指标定义

### 请求相关指标

| 指标名称 | 类型 | 描述 | 用途 |
|---------|------|------|------|
| `http_requests_total` | Counter | 请求总数 | 计算QPS、请求量趋势 |
| `http_request_duration_seconds` | Histogram | 请求耗时 | P50/P95/P99延迟分析 |
| `http_requests_in_progress` | Gauge | 正在处理的请求数 | 并发量监控 |
| `http_errors_total` | Counter | 错误请求数 | 错误率计算 |

### 业务指标

| 指标名称 | 类型 | 描述 |
|---------|------|------|
| `user_registrations_total` | Counter | 用户注册数 |
| `order_processing_duration_seconds` | Histogram | 订单处理耗时 |
| `active_users` | Gauge | 活跃用户数 |

## 告警规则

### Prometheus告警规则示例

```yaml
groups:
  - name: python_service_alerts
    rules:
      # 高错误率告警
      - alert: HighErrorRate
        expr: rate(http_errors_total[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "高错误率检测到"
          description: "错误率超过5%，当前值: {{ $value }}"

      # 高延迟告警
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "P95延迟过高"
          description: "P95延迟超过1秒"

      # 服务宕机告警
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "服务不可用"
```

## 可视化配置

### Grafana仪表盘

导入 `grafana_dashboard.json` 到Grafana，包含以下面板：

1. **概览面板**
   - QPS（每秒请求数）
   - 错误率
   - P95延迟
   - 并发请求数

2. **延迟分析面板**
   - 延迟分布直方图
   - P50/P95/P99延迟趋势
   - 按端点分组的延迟对比

3. **错误分析面板**
   - 错误率趋势
   - 按状态码分组的错误分布
   - 按端点分组的错误率

4. **资源使用面板**
   - CPU使用率
   - 内存使用率
   - GC次数和耗时

## 分布式追踪

### OpenTelemetry配置

`tracing_example.py` 展示了如何：
1. 初始化Tracer Provider
2. 配置Span Processor和Exporter
3. 创建自定义Span
4. 添加属性和事件
5. 上下文传播

### 追踪可视化

使用Jaeger或Zipkin查看追踪数据：

```bash
# 启动Jaeger
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest
```

访问 http://localhost:16686 查看追踪UI。

## 最佳实践

### 指标命名规范

1. 使用小写字母和下划线
2. 使用名词+动词结构
3. 包含单位后缀（如 `_seconds`, `_bytes`）
4. 示例：`http_request_duration_seconds`

### Label使用原则

1. Label基数不宜过大（<100）
2. 避免使用高基数维度（如user_id）
3. 常用Label：`endpoint`, `method`, `status_code`, `service`

### 性能考虑

1. Histogram bucket选择要合理
2. 避免在热点路径创建过多指标
3. 使用批量导出减少网络开销

## 故障排查

### 常见问题

1. **指标不显示**
   - 检查 `/metrics` 端点是否可访问
   - 验证Prometheus配置中的targets

2. **Grafana无数据**
   - 确认Prometheus数据源配置正确
   - 检查指标名称是否匹配

3. **追踪不完整**
   - 确认上下文传播正确
   - 检查采样率设置

## 进阶主题

### 自定义指标扩展

```python
from prometheus_client import Counter, Histogram

# 自定义业务指标
order_counter = Counter('orders_total', '订单总数', ['region', 'payment_method'])
order_duration = Histogram('order_processing_seconds', '订单处理耗时')

# 使用指标
with order_duration.time():
    process_order()
order_counter.labels(region='cn', payment_method='alipay').inc()
```

### 告警集成

- Alertmanager + 钉钉/企业微信机器人
- PagerDuty/OpsGenie
- 自定义Webhook

## 参考资源

- [Prometheus官方文档](https://prometheus.io/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Grafana Documentation](https://grafana.com/docs/)
