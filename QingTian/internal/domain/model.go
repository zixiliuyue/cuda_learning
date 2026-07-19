// Package domain 定义算力调度平台的核心领域模型。
//
// 领域模型是整个平台的通用语言，被存储层、调度层、API 层、Agent 共同引用。
// 这里只包含数据结构与不依赖外部服务的领域规则（如任务状态机）。
package domain

import "time"

// Role 表示用户角色，用于 RBAC 授权。
type Role string

const (
	RoleAdmin     Role = "admin"     // 管理员：拥有全部权限
	RoleDeveloper Role = "developer" // 开发者：可提交/查看自己的任务
	RoleViewer    Role = "viewer"    // 只读用户：仅可查看
)

// User 表示平台用户。
//
// 字段：
//   - ID:           用户唯一标识。
//   - Username:     登录用户名，全局唯一。
//   - PasswordHash: bcrypt 加密后的密码哈希，切勿存储明文。
//   - Role:         用户角色，决定其权限。
//   - CreatedAt:    创建时间。
type User struct {
	ID           string    `json:"id" bson:"_id"`
	Username     string    `json:"username" bson:"username"`
	PasswordHash string    `json:"-" bson:"password_hash"`
	Role         Role      `json:"role" bson:"role"`
	CreatedAt    time.Time `json:"created_at" bson:"created_at"`
}

// Vendor 表示 GPU/加速卡的厂商，用于异构算力纳管。
//
// 平台在采集层将不同厂商的加速卡抽象为统一的 GPU 结构，
// 通过 Vendor 字段区分 NVIDIA、华为昇腾、寒武纪等，契合信创需求。
type Vendor string

const (
	VendorNVIDIA    Vendor = "nvidia"    // 英伟达 GPU
	VendorAscend    Vendor = "ascend"    // 华为昇腾 NPU
	VendorCambricon Vendor = "cambricon" // 寒武纪 MLU
	VendorUnknown   Vendor = "unknown"   // 未知/未识别
)

// ShareMode 表示 GPU 的共享（切分）模式。
//
// 这是平台的核心差异化能力：通过细粒度切分把整卡拆成多份，
// 让多个任务共享同一块 GPU，从而把利用率从"整卡独占"提升到更高水平。
type ShareMode string

const (
	// ShareExclusive 整卡独占：任务独享整块 GPU，其它任务不可再分配。
	ShareExclusive ShareMode = "exclusive"
	// ShareMemory 显存切片（软共享，近似 MPS）：多个任务按显存额度共享同一块卡。
	ShareMemory ShareMode = "memory"
	// ShareMIG MIG 实例：将物理 GPU 切分为若干硬件隔离的 MIG 实例，任务占用一个实例。
	ShareMIG ShareMode = "mig"
)

// MIGInstance 表示一个 MIG（Multi-Instance GPU）实例。
//
// 字段：
//   - ID:          MIG 实例标识（对应 NVML MIG device UUID）。
//   - Profile:     实例规格名，例如 "1g.5gb"、"2g.10gb"。
//   - MemoryMB:    该实例的显存额度（MB）。
//   - Allocatable: 是否可被分配。
type MIGInstance struct {
	ID          string `json:"id" bson:"id"`
	Profile     string `json:"profile" bson:"profile"`
	MemoryMB    int64  `json:"memory_mb" bson:"memory_mb"`
	Allocatable bool   `json:"allocatable" bson:"allocatable"`
}

// GPU 表示节点上的单块 GPU 设备。
//
// 字段：
//   - UUID:        GPU 全局唯一标识（对应 NVML 的 GPU UUID）。
//   - Vendor:      厂商，用于异构算力纳管（默认视为 nvidia）。
//   - Model:       GPU 型号，例如 "RTX 5060 Ti"。
//   - MemoryTotal: 显存总量（MB）。
//   - MemoryUsed:  已分配显存（MB），由资源账本维护。
//   - Utilization: 实时利用率（0-100），由 Agent 采集上报。
//   - Allocatable: 是否可被调度分配（异常/维护中的卡置为 false）。
//   - MIGEnabled:  是否开启 MIG 切分模式。
//   - MIGInstances: 开启 MIG 时的实例列表。
type GPU struct {
	UUID         string        `json:"uuid" bson:"uuid"`
	Vendor       Vendor        `json:"vendor" bson:"vendor"`
	Model        string        `json:"model" bson:"model"`
	MemoryTotal  int64         `json:"memory_total" bson:"memory_total"`
	MemoryUsed   int64         `json:"memory_used" bson:"memory_used"`
	Utilization  int           `json:"utilization" bson:"utilization"`
	Allocatable  bool          `json:"allocatable" bson:"allocatable"`
	MIGEnabled   bool          `json:"mig_enabled" bson:"mig_enabled"`
	MIGInstances []MIGInstance `json:"mig_instances,omitempty" bson:"mig_instances,omitempty"`
}

// NodeStatus 表示节点在线状态。
type NodeStatus string

const (
	NodeOnline  NodeStatus = "online"  // 在线：心跳正常，可调度
	NodeOffline NodeStatus = "offline" // 离线：心跳超时，不可调度
)

// Node 表示一个 GPU 计算节点。
//
// 字段：
//   - ID:            节点唯一标识。
//   - Hostname:      主机名。
//   - IP:            节点 IP 地址。
//   - Status:        在线状态。
//   - GPUs:          该节点上的 GPU 列表。
//   - DriverVersion: NVIDIA 驱动版本。
//   - CUDAVersion:   CUDA 版本。
//   - LastHeartbeat: 最近一次心跳时间，用于判断是否离线。
//   - RegisteredAt:  首次注册时间。
type Node struct {
	ID            string     `json:"id" bson:"_id"`
	Hostname      string     `json:"hostname" bson:"hostname"`
	IP            string     `json:"ip" bson:"ip"`
	Status        NodeStatus `json:"status" bson:"status"`
	GPUs          []GPU      `json:"gpus" bson:"gpus"`
	DriverVersion string     `json:"driver_version" bson:"driver_version"`
	CUDAVersion   string     `json:"cuda_version" bson:"cuda_version"`
	LastHeartbeat time.Time  `json:"last_heartbeat" bson:"last_heartbeat"`
	RegisteredAt  time.Time  `json:"registered_at" bson:"registered_at"`
}

// TaskStatus 表示任务生命周期中的状态。
type TaskStatus string

const (
	TaskPending   TaskStatus = "pending"   // 待调度：已提交，等待调度器分配节点
	TaskScheduled TaskStatus = "scheduled" // 已调度：已分配节点，等待 Agent 拉起
	TaskRunning   TaskStatus = "running"   // 运行中：Agent 正在执行
	TaskSucceeded TaskStatus = "succeeded" // 成功：正常结束，退出码为 0
	TaskFailed    TaskStatus = "failed"    // 失败：执行出错或退出码非 0
	TaskCanceled  TaskStatus = "canceled"  // 已取消：用户主动取消
)

// ResourceRequest 表示任务对算力资源的需求。
//
// 字段：
//   - GPUCount:  需要的 GPU 数量。
//   - MemoryMB:  单卡最低显存需求（MB）。
//   - GPUModel:  期望的 GPU 型号，空字符串表示不限型号。
//   - Vendor:    期望的厂商，空表示不限（异构算力选型）。
//   - ShareMode: 共享模式；空视为整卡独占（exclusive）。
//   - GangSize:  成组调度所需的节点/实例份数，0 或 1 表示无需成组。
type ResourceRequest struct {
	GPUCount  int       `json:"gpu_count" bson:"gpu_count"`
	MemoryMB  int64     `json:"memory_mb" bson:"memory_mb"`
	GPUModel  string    `json:"gpu_model" bson:"gpu_model"`
	Vendor    Vendor    `json:"vendor,omitempty" bson:"vendor,omitempty"`
	ShareMode ShareMode `json:"share_mode,omitempty" bson:"share_mode,omitempty"`
	GangSize  int       `json:"gang_size,omitempty" bson:"gang_size,omitempty"`
}

// EffectiveShareMode 返回请求实际生效的共享模式，空值归一为整卡独占。
func (r ResourceRequest) EffectiveShareMode() ShareMode {
	if r.ShareMode == "" {
		return ShareExclusive
	}
	return r.ShareMode
}

// FailureCategory 表示任务失败的结构化归类，用于失败诊断。
//
// 相比一段自由文本，结构化归类能让前端给出可操作的排查建议，
// 降低用户挫败感与支持工单量。
type FailureCategory string

const (
	FailureNone         FailureCategory = ""              // 无失败
	FailureOOM          FailureCategory = "oom"           // 显存不足（Out Of Memory）
	FailureImagePull    FailureCategory = "image_pull"    // 镜像拉取失败
	FailureNonZeroExit  FailureCategory = "non_zero_exit" // 进程非零退出
	FailureNodeLost     FailureCategory = "node_lost"     // 节点失联/宕机
	FailureCanceled     FailureCategory = "canceled"      // 被主动取消
	FailureUnknown      FailureCategory = "unknown"       // 未能识别的失败
	FailureStartFailure FailureCategory = "start_failure" // 进程无法启动
	FailureTimeout      FailureCategory = "timeout"       // 运行超时被强制终止
	FailurePreempted    FailureCategory = "preempted"     // 被高优先级任务抢占
)

// Task 表示一个算力任务。
//
// 字段：
//   - ID:            任务唯一标识。
//   - Name:          任务名称。
//   - Owner:         提交者用户 ID。
//   - ProjectID:     所属项目/团队工作区 ID（个人任务可为空）。
//   - Image:         执行使用的容器镜像。
//   - Command:       容器内执行的命令。
//   - Resource:      资源需求。
//   - Priority:      优先级，数值越大越优先调度。
//   - Status:        当前状态。
//   - NodeID:        被调度到的节点 ID（未调度时为空）。
//   - AllocatedGPUs: 分配到的 GPU UUID 列表。
//   - DatasetPath:   输入数据集在 MinIO 中的对象路径。
//   - ResultPath:    结果制品在 MinIO 中的对象路径。
//   - LogPath:       任务日志在 MinIO 中的归档对象路径。
//   - CheckpointPath: 最近一次 Checkpoint 在 MinIO 中的对象路径（断点续跑用）。
//   - GangID:        Gang（成组调度）标识，同组任务共享该 ID。
//   - TimeoutSeconds: 运行超时秒数；>0 时超过该时长仍未结束将被强制终止（0 表示不限）。
//   - Preemptible:   是否可被高优先级任务抢占（竞价/Spot 语义）。
//   - ExitCode:      进程退出码（结束后有效）。
//   - Message:       附加信息（如失败原因）。
//   - FailureCategory: 结构化失败归类，供失败诊断展示。
//   - CreatedAt/UpdatedAt/StartedAt/FinishedAt: 各生命周期时间戳。
type Task struct {
	ID              string          `json:"id" bson:"_id"`
	Name            string          `json:"name" bson:"name"`
	Owner           string          `json:"owner" bson:"owner"`
	ProjectID       string          `json:"project_id,omitempty" bson:"project_id,omitempty"`
	Image           string          `json:"image" bson:"image"`
	Command         string          `json:"command" bson:"command"`
	Resource        ResourceRequest `json:"resource" bson:"resource"`
	Priority        int             `json:"priority" bson:"priority"`
	Status          TaskStatus      `json:"status" bson:"status"`
	NodeID          string          `json:"node_id" bson:"node_id"`
	AllocatedGPUs   []string        `json:"allocated_gpus" bson:"allocated_gpus"`
	DatasetPath     string          `json:"dataset_path" bson:"dataset_path"`
	ResultPath      string          `json:"result_path" bson:"result_path"`
	LogPath         string          `json:"log_path,omitempty" bson:"log_path,omitempty"`
	CheckpointPath  string          `json:"checkpoint_path,omitempty" bson:"checkpoint_path,omitempty"`
	GangID          string          `json:"gang_id,omitempty" bson:"gang_id,omitempty"`
	TimeoutSeconds  int64           `json:"timeout_seconds,omitempty" bson:"timeout_seconds,omitempty"`
	Preemptible     bool            `json:"preemptible,omitempty" bson:"preemptible,omitempty"`
	ExitCode        int             `json:"exit_code" bson:"exit_code"`
	Message         string          `json:"message" bson:"message"`
	FailureCategory FailureCategory `json:"failure_category,omitempty" bson:"failure_category,omitempty"`
	CreatedAt       time.Time       `json:"created_at" bson:"created_at"`
	UpdatedAt       time.Time       `json:"updated_at" bson:"updated_at"`
	StartedAt       *time.Time      `json:"started_at,omitempty" bson:"started_at,omitempty"`
	FinishedAt      *time.Time      `json:"finished_at,omitempty" bson:"finished_at,omitempty"`
}

// IsActive 判断任务是否处于占用资源的活动状态（已调度或运行中）。
func (t *Task) IsActive() bool {
	return t.Status == TaskScheduled || t.Status == TaskRunning
}

// IsTimedOut 判断运行中的任务是否已超过其允许的最大运行时长。
//
// 仅当任务设置了正的 TimeoutSeconds、已进入运行态（StartedAt 非空）时才可能返回 true。
// 该方法是纯函数（不修改状态），便于在守护循环中安全调用与单元测试。
//
// 参数：
//   - now: 当前时间，由调用方注入以便测试。
//
// 返回值：
//   - bool: true 表示任务运行时长已超限，应被强制终止。
func (t *Task) IsTimedOut(now time.Time) bool {
	if t.TimeoutSeconds <= 0 || t.StartedAt == nil {
		return false
	}
	if t.Status != TaskRunning {
		return false
	}
	deadline := t.StartedAt.Add(time.Duration(t.TimeoutSeconds) * time.Second)
	return now.After(deadline)
}

// Project 表示一个团队工作区/项目空间。
//
// 项目让任务、数据集与配额从"个人独享"升级为"团队共享"，
// 是平台从个人工具走向团队协作平台的关键抽象。
//
// 字段：
//   - ID:        项目唯一标识。
//   - Name:      项目名称。
//   - OwnerID:   项目所有者用户 ID。
//   - Members:   成员用户 ID 列表（包含所有者）。
//   - CreatedAt: 创建时间。
type Project struct {
	ID        string    `json:"id" bson:"_id"`
	Name      string    `json:"name" bson:"name"`
	OwnerID   string    `json:"owner_id" bson:"owner_id"`
	Members   []string  `json:"members" bson:"members"`
	CreatedAt time.Time `json:"created_at" bson:"created_at"`
}

// HasMember 判断用户是否为项目成员。
func (p *Project) HasMember(userID string) bool {
	for _, m := range p.Members {
		if m == userID {
			return true
		}
	}
	return false
}

// Quota 表示某个主体（用户或项目）的算力配额与已用量。
//
// 配额是计费与成本控制的基础：提交任务前校验剩余配额，
// 任务结束后按卡时累加已用量，超额则拒绝提交。
//
// 字段：
//   - Subject:       配额主体标识（用户 ID 或项目 ID）。
//   - MaxGPUHours:   卡时上限（GPU·小时）；0 表示不限。
//   - UsedGPUHours:  已用卡时。
//   - MaxConcurrent: 最大并发运行任务数；0 表示不限。
type Quota struct {
	Subject       string  `json:"subject" bson:"_id"`
	MaxGPUHours   float64 `json:"max_gpu_hours" bson:"max_gpu_hours"`
	UsedGPUHours  float64 `json:"used_gpu_hours" bson:"used_gpu_hours"`
	MaxConcurrent int     `json:"max_concurrent" bson:"max_concurrent"`
}

// RemainingGPUHours 返回剩余卡时；MaxGPUHours 为 0（不限）时返回一个很大的值。
func (q *Quota) RemainingGPUHours() float64 {
	if q.MaxGPUHours <= 0 {
		return 1<<62 - 1
	}
	return q.MaxGPUHours - q.UsedGPUHours
}

// UsageRecord 表示一次任务运行产生的算力用量与成本记录。
//
// 用量记录是成本可视化与计费的原始凭证：以卡时为计量单位，
// 结合单价聚合出任务/用户/项目维度的成本。
//
// 字段：
//   - TaskID:    任务 ID。
//   - Owner:     任务所有者用户 ID。
//   - ProjectID: 所属项目 ID（可为空）。
//   - GPUCount:  占用的 GPU 份数。
//   - GPUHours:  消耗的卡时（GPU·小时）。
//   - Cost:      折算成本（卡时 × 单价）。
//   - StartedAt/FinishedAt: 运行起止时间。
type UsageRecord struct {
	ID         string    `json:"id" bson:"_id"`
	TaskID     string    `json:"task_id" bson:"task_id"`
	Owner      string    `json:"owner" bson:"owner"`
	ProjectID  string    `json:"project_id,omitempty" bson:"project_id,omitempty"`
	GPUCount   int       `json:"gpu_count" bson:"gpu_count"`
	GPUHours   float64   `json:"gpu_hours" bson:"gpu_hours"`
	Cost       float64   `json:"cost" bson:"cost"`
	StartedAt  time.Time `json:"started_at" bson:"started_at"`
	FinishedAt time.Time `json:"finished_at" bson:"finished_at"`
}

// Edition 表示平台的功能版本（开源版/企业版），用于功能门控。
//
// 采用"开源核心引流 + 企业版付费"的 PLG 商业化打法：
// 共享调度、多租户、审计、SSO 等高级能力仅在企业版开放。
type Edition string

const (
	EditionCommunity  Edition = "community"  // 开源社区版
	EditionEnterprise Edition = "enterprise" // 企业版
)
