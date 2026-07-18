# QingTian（擎天）算力调度平台

一个面向多 GPU 节点集群的算力纳管、任务调度、资源隔离与可观测平台。

- 前端：Vue3 + Arco Design
- 后端 & Agent：Go
- 存储：MongoDB（元数据）+ Redis（缓存/队列/锁/心跳）
- 消息队列：NSQ
- 对象存储：MinIO

> 完整设计见 [plan.md](./plan.md)。

## 目录结构

```
QingTian/
├── plan.md                  # 项目计划与设计文档
├── go.mod                   # Go module: qingtian
├── configs/                 # 各服务 YAML 配置示例
├── cmd/                     # 可执行入口
│   ├── apiserver/           # 控制面 API 服务
│   ├── scheduler/           # 调度器服务
│   └── agent/               # 节点 Agent
├── internal/
│   ├── config/              # 配置加载
│   ├── logger/              # 结构化日志
│   ├── domain/              # 领域模型 + 任务状态机
│   ├── auth/                # JWT + RBAC
│   ├── ledger/              # GPU 资源账本（防超卖）
│   ├── scheduler/           # 两阶段调度算法（Filter + Score）
│   ├── repository/          # 仓储接口 + 内存实现 + MongoDB 实现
│   ├── cache/               # 缓存/队列/锁接口 + 内存实现 + Redis 实现
│   ├── mq/                  # 消息队列接口 + 内存实现 + NSQ 实现
│   ├── storage/             # 对象存储接口 + 内存实现 + MinIO 实现
│   ├── service/             # 业务服务（user/node/task）
│   ├── api/                 # HTTP handler / 路由 / 中间件
│   ├── agent/               # Agent 各子模块
│   └── bootstrap/           # 外部中间件连接装配
├── pkg/                     # 通用工具（idgen / apperr / httpx）
└── web/                     # Vue3 + Arco 前端控制台
```

## 架构设计要点

- **控制面 / 数据面分离**：Go 微服务（无状态）负责调度决策，Agent（数据面）负责实际执行。
- **接口抽象 + 依赖注入**：存储/缓存/消息/对象存储均通过接口抽象，提供内存实现（单测）与真实实现（生产）。
- **防超卖**：资源账本原子扣减 + 调度失败回滚。
- **幂等**：消息消费者对已终态任务忽略更新，以 MongoDB 为真相源。

## 构建与测试

### 后端

```bash
# 编译全部
go build ./...

# 运行单元测试（无需外部中间件，使用内存实现）
go test ./...
```

### 前端

```bash
cd web
npm install
npm run dev      # 开发模式（默认代理 /api 到 http://localhost:8080）
npm run build    # 生产构建
```

## 运行（需先启动 MongoDB / Redis / NSQ / MinIO）

```bash
# 1. 启动控制面 API 服务
go run ./cmd/apiserver -config configs/apiserver.yaml

# 2. 启动调度器
go run ./cmd/scheduler -config configs/scheduler.yaml

# 3. 在 GPU 节点启动 Agent
go run ./cmd/agent -config configs/agent.yaml
```

配置中的连接信息可通过环境变量覆盖：
`QT_MONGO_URI`、`QT_REDIS_ADDR`、`QT_NSQD_ADDR`、`QT_MINIO_ENDPOINT`、`QT_JWT_SECRET`。

## 核心闭环

用户提交任务 → 落库 & 入优先级队列 → 调度器 Filter+Score 选节点 → 账本扣减 →
NSQ 下发到目标节点 → Agent 执行（注入 CUDA_VISIBLE_DEVICES）→ 状态经 NSQ 回传 →
控制面更新任务状态 → 前端展示。

该闭环在 `internal/service/service_test.go` 的 `TestClosedLoop` 中以内存实现完整验证。
