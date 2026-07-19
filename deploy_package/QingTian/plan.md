# QingTian（擎天）算力调度平台 —— 项目计划与设计文档

> 一个面向多 GPU 节点集群的算力纳管、任务调度、资源隔离与可观测平台。
> 技术栈：Vue3 + Arco Design（前端）、Go（后端 & Agent）、MongoDB + Redis（存储）、NSQ（消息队列）、MinIO（对象存储）。

---

## 1. 目标与范围

- 打通核心闭环：**用户提交任务 → 调度器选节点 → NSQ 下发 → Agent 执行 → 状态回传 → 前端展示**。
- 控制面（Go 微服务，无状态）与数据面（Agent，常驻 GPU 节点）分离。
- 所有代码收敛在 `F:\code\cuda_learning\QingTian` 单一 Go module（module 名 `qingtian`），不引用目录外任何代码。
- 存储/消息层使用真实客户端库（mongo-driver、go-redis、go-nsq、minio-go）；业务通过 Repository/接口抽象，单元测试用内存 fake 实现，无需真实中间件即可运行。

## 2. 系统架构

```
Vue3 + Arco 控制台
        │ HTTP/JSON (JWT)
        ▼
┌──────────────────────────────────────────────┐
│ API-Server (BFF: 鉴权/编排/校验)               │
├──────────────────────────────────────────────┤
│ Scheduler (Filter+Score 两阶段调度)            │
│ ResourceManager (GPU 资源账本/配额)            │
│ TaskManager (任务状态机/生命周期)              │
│ MonitorService (指标/事件聚合)                 │
└──────────────────────────────────────────────┘
   │ Repo 接口        │ NSQ 消息
   ▼                  ▼
MongoDB(元数据)  Redis(账本/队列/心跳/锁)  MinIO(制品)
                        │
                        ▼ task.dispatch.{node}
                     Agent (注册/心跳/NVML采集/执行/上报)
```

## 3. 目录结构

```
QingTian/
├── plan.md
├── go.mod
├── README.md
├── configs/                 # 各服务 YAML 配置样例
├── cmd/                     # 各可执行入口
│   ├── apiserver/
│   ├── scheduler/
│   └── agent/
├── internal/
│   ├── config/              # 配置加载
│   ├── logger/              # 日志封装
│   ├── domain/              # 领域模型（Task/Node/GPU/User 等）+ 状态机
│   ├── auth/                # JWT + RBAC
│   ├── ledger/              # GPU 资源账本（内存 + Redis 语义）
│   ├── scheduler/           # 调度算法（Filter/Score/策略）
│   ├── repository/          # 仓储接口 + 内存实现 + Mongo 实现
│   ├── mq/                  # 消息队列接口 + 内存实现 + NSQ 实现
│   ├── storage/             # 对象存储接口 + 内存实现 + MinIO 实现
│   ├── cache/               # 缓存接口 + 内存实现 + Redis 实现
│   ├── api/                 # HTTP handler / 路由 / 中间件
│   └── agent/               # Agent 各子模块
├── pkg/                     # 可复用工具（id 生成、错误、响应）
└── web/                     # Vue3 + Arco 前端
```

## 4. 核心数据模型

- **User**：ID、用户名、密码哈希、角色（admin/developer/viewer）。
- **Node**：ID、主机名、状态（online/offline）、GPU 列表、心跳时间。
- **GPU**：UUID、型号、总显存/已用显存、利用率、是否可分配。
- **Task**：ID、名称、提交者、镜像、命令、请求资源（GPU 数/显存）、优先级、状态、目标节点、时间戳、结果制品路径。
- **任务状态机**：`Pending → Scheduled → Running → Succeeded/Failed/Canceled`，非法转移拒绝。

## 5. 调度器设计

- **Filter**：过滤满足 GPU 数量、显存、型号约束且在线的节点。
- **Score**：对候选节点打分（Bin-Packing 装箱率 + 负载均衡），取最高分。
- **一致性**：资源账本 CAS 原子扣减，防止超卖；调度失败回滚。

## 6. 安全策略

- JWT 认证 + RBAC 授权（角色→权限映射）。
- 密码 bcrypt 哈希。
- HTTP 中间件：鉴权、请求日志、恢复 panic、CORS。

## 7. 消息流

- `task.dispatch.{node_id}`：控制面 → Agent 下发任务。
- `task.status`：Agent → 控制面回传状态。
- `node.event`：节点上下线事件。

## 8. 测试策略

- 领域层（状态机、账本、调度算法、auth、id/response 工具）：纯逻辑单测，覆盖率优先。
- 仓储/缓存/mq/storage：内存实现单测；真实实现（Mongo/Redis/NSQ/MinIO）编译期校验接口，运行期集成测试可选。
- 命令：`go test ./...`（无需外部中间件即可通过）。

## 9. 里程碑

1. P1 基础框架：module、config、logger、domain、pkg 工具 + 单测。
2. P2 存储与消息层：接口 + 内存实现 + 真实客户端实现。
3. P3 领域逻辑：auth、ledger、状态机、scheduler + 单测。
4. P4 服务与 Agent：API-Server、Scheduler 服务、Agent，打通闭环。
5. P5 前端：Vue3 + Arco 最小控制台（登录/节点/任务）。
6. P6 验证：全量编译 + 单测通过。

## 10. 风险与对策

| 风险 | 对策 |
|------|------|
| GPU 超卖 | 账本 CAS 原子扣减 + 调度失败回滚 |
| 消息重复/丢失 | 消费者幂等 + MongoDB 为真相源 |
| Agent 与真实 GPU 不同步 | 心跳 + 执行前二次校验 |
| 外部中间件不可用致无法测试 | 接口抽象 + 内存 fake，单测不依赖真实服务 |
