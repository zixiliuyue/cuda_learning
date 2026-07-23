// Package domain 定义算力调度平台的核心领域模型。
//
// 领域模型是整个平台的通用语言，被存储层、调度层、API 层、Agent 共同引用。
// 这里只包含数据结构与不依赖外部服务的领域规则（如任务状态机）。
package domain

import (
	"encoding/json"
	"time"
)

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
	ExternalID   string    `json:"external_id,omitempty" bson:"external_id,omitempty"`
	Email        string    `json:"email,omitempty" bson:"email,omitempty"`
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
//   - Labels:        节点标签，key-value 键值对，用于节点选择与分组。
//   - Taints:        节点污点，只有容忍对应污点的任务才能调度上来。
//   - LastHeartbeat: 最近一次心跳时间，用于判断是否离线。
//   - RegisteredAt:  首次注册时间。
type Node struct {
	ID             string            `json:"id" bson:"_id"`
	Hostname       string            `json:"hostname" bson:"hostname"`
	IP             string            `json:"ip" bson:"ip"`
	Status         NodeStatus        `json:"status" bson:"status"`
	GPUs           []GPU             `json:"gpus" bson:"gpus"`
	DriverVersion  string            `json:"driver_version" bson:"driver_version"`
	CUDAVersion    string            `json:"cuda_version" bson:"cuda_version"`
	Labels         map[string]string `json:"labels,omitempty" bson:"labels,omitempty"`
	Taints         []Taint           `json:"taints,omitempty" bson:"taints,omitempty"`
	LocalDatasets  []string          `json:"local_datasets,omitempty" bson:"local_datasets,omitempty"`
	LastHeartbeat  time.Time         `json:"last_heartbeat" bson:"last_heartbeat"`
	RegisteredAt   time.Time         `json:"registered_at" bson:"registered_at"`
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

// NodeSelector 表示任务对节点标签的选择约束。
//
// 所有指定在 MatchLabels 中的键值对都必须匹配节点标签才能被选中：
//   - key 必须存在于节点标签中
//   - key 对应的值必须完全相等
type NodeSelector struct {
	// MatchLabels 键值对匹配约束。
	MatchLabels map[string]string `json:"match_labels,omitempty" bson:"match_labels,omitempty"`
}

// ResourceRequest 表示任务对算力资源的需求。
//
// 字段：
//   - GPUCount:     需要的 GPU 数量。
//   - MemoryMB:     单卡最低显存需求（MB）。
//   - GPUModel:     期望的 GPU 型号，空字符串表示不限型号。
//   - Vendor:       期望的厂商，空表示不限（异构算力选型）。
//   - ShareMode:    共享模式；空视为整卡独占（exclusive）。
//   - GangSize:     成组调度所需的节点/实例份数，0 或 1 表示无需成组。
//   - NodeSelector: 节点标签选择约束，空表示不限制。
//   - Tolerations:   节点污点容忍列表，匹配污点才能调度。
type ResourceRequest struct {
	GPUCount      int             `json:"gpu_count" bson:"gpu_count"`
	MemoryMB      int64           `json:"memory_mb" bson:"memory_mb"`
	GPUModel      string          `json:"gpu_model" bson:"gpu_model"`
	Vendor        Vendor          `json:"vendor,omitempty" bson:"vendor,omitempty"`
	ShareMode     ShareMode        `json:"share_mode,omitempty" bson:"share_mode,omitempty"`
	GangSize      int             `json:"gang_size,omitempty" bson:"gang_size,omitempty"`
	NodeSelector  *NodeSelector   `json:"node_selector,omitempty" bson:"node_selector,omitempty"`
	Tolerations   []Toleration   `json:"tolerations,omitempty" bson:"tolerations,omitempty"`
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
//   - MaxRetries:    最大自动重试次数；>0 时失败后会自动重试直到达到此次数。
//   - Retries:       已重试次数。
//   - LastRetryAt:   最后一次重试时间。
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
	Env             []EnvVar        `json:"env,omitempty" bson:"env,omitempty"`
	Resource        ResourceRequest `json:"resource" bson:"resource"`
	Priority        int             `json:"priority" bson:"priority"`
	Status          TaskStatus      `json:"status" bson:"status"`
	NodeID          string          `json:"node_id" bson:"node_id"`
	AllocatedGPUs   []string        `json:"allocated_gpus" bson:"allocated_gpus"`
	DatasetPath          string          `json:"dataset_path" bson:"dataset_path"`
	ResultPath           string          `json:"result_path" bson:"result_path"`
	LogPath              string          `json:"log_path,omitempty" bson:"log_path,omitempty"`
	ArtifactsPath        string          `json:"artifacts_path,omitempty" bson:"artifacts_path,omitempty"`
	ArtifactsArchivedPath string        `json:"artifacts_archived_path,omitempty" bson:"artifacts_archived_path,omitempty"`
	CheckpointPath       string          `json:"checkpoint_path,omitempty" bson:"checkpoint_path,omitempty"`
	GangID               string          `json:"gang_id,omitempty" bson:"gang_id,omitempty"`
	TimeoutSeconds  int64           `json:"timeout_seconds,omitempty" bson:"timeout_seconds,omitempty"`
	Preemptible     bool            `json:"preemptible,omitempty" bson:"preemptible,omitempty"`
	MaxRetries      int             `json:"max_retries,omitempty" bson:"max_retries,omitempty"`
	Retries         int             `json:"retries" bson:"retries"`
	LastRetryAt     *time.Time      `json:"last_retry_at,omitempty" bson:"last_retry_at,omitempty"`
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

// TaskTemplate 表示任务模板，保存常用任务配置方便快速提交。
//
// 用户可以把经常提交的训练任务保存为模板，下次提交只需修改少量参数，
// 提升日常提交效率。模板归属于用户或项目，支持私有共享。
//
// 字段：
//   - ID:           模板唯一 ID。
//   - Name:         模板名称。
//   - Description:  模板描述（说明用途）。
//   - OwnerID:      所有者用户 ID。
//   - ProjectID:    所属项目 ID（可为空）。
//   - Config:       任务配置（复用 ResourceRequest 结构）。
//   - Image:        默认镜像。
//   - Command:      默认启动命令。
//   - DatasetPath:  默认数据集路径。
//   - Priority:     默认优先级。
//   - TimeoutSeconds:  默认超时秒数。
//   - Preemptible:  是否可抢占。
//   - CreatedAt:    创建时间。
//   - UpdatedAt:    更新时间。
type TaskTemplate struct {
	ID             string                 `json:"id" bson:"_id"`
	Name           string                 `json:"name" bson:"name"`
	Description    string                 `json:"description,omitempty" bson:"description,omitempty"`
	OwnerID        string                 `json:"owner_id" bson:"owner_id"`
	ProjectID      string                 `json:"project_id,omitempty" bson:"project_id,omitempty"`
	Config         ResourceRequest        `json:"config" bson:"config"`
	Image          string                 `json:"image" bson:"image"`
	Command        string                 `json:"command,omitempty" bson:"command,omitempty"`
	Env            []EnvVar              `json:"env,omitempty" bson:"env,omitempty"`
	DatasetPath    string                 `json:"dataset_path,omitempty" bson:"dataset_path,omitempty"`
	Priority       int                    `json:"priority" bson:"priority"`
	TimeoutSeconds int64                 `json:"timeout_seconds,omitempty" bson:"timeout_seconds,omitempty"`
	Preemptible    bool                   `json:"preemptible" bson:"preemptible"`
	CreatedAt      time.Time              `json:"created_at" bson:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at" bson:"updated_at"`
}

// TaintEffect 表示污点效果，当任务不容忍该污点时采取的行动。
type TaintEffect string

const (
	// TaintEffectNoSchedule 不允许新任务调度到该节点
	TaintEffectNoSchedule TaintEffect = "NoSchedule"
	// TaintEffectPreferNoSchedule 尽量不要调度到该节点（软约束），打分降低优先级
	TaintEffectPreferNoSchedule TaintEffect = "PreferNoSchedule"
)

// Taint 表示节点污点
//
// 污点标记节点具有某种特殊属性，普通任务不应该被调度到这里，
// 只有明确容忍该污点的任务才能被分配过来，常用于：
//   - 节点GPU故障待维修
//   - 节点被标记为专用/独占
//   - 节点资源即将耗尽，不再接收新任务
//   - 节点正在执行批量维护作业
type Taint struct {
	Key    string       `json:"key" bson:"key"`
	Value  string       `json:"value" bson:"value"`
	Effect TaintEffect `json:"effect" bson:"effect"`
}

// Toleration 表示任务对污点的容忍
//
// 任务容忍节点污点，则可以调度到该节点。
// 匹配规则：
//  - 当 Operator 是 Equal 时，污点键相等且值相等才匹配
//  - 当 Operator 是 Exists 时，只要污点键存在就匹配（不需要匹配值）
type Toleration struct {
	Key      string               `json:"key" bson:"key"`
	Value    string               `json:"value,omitempty" bson:"value,omitempty"`
	Operator TolerationOperator `json:"operator" bson:"operator"`
	Effect   TaintEffect         `json:"effect" bson:"effect"`
}

// TolerationOperator 容忍匹配操作符
type TolerationOperator string

const (
	// TolerationOpEqual 键和值都匹配
	TolerationOpEqual TolerationOperator = "Equal"
	// TolerationOpExists 键存在就匹配，忽略值
	TolerationOpExists TolerationOperator = "Exists"
)

// TerminalSession 表示一个活跃的 Web 终端会话。
//
// 终端会话绑定到正在运行的任务，用户可以通过 Web 浏览器连接
// 进行交互式调试，查看实时输出，输入命令。
type TerminalSession struct {
	ID         string    `json:"id" bson:"_id"`
	TaskID     string    `json:"task_id" bson:"task_id"`
	NodeID     string    `json:"node_id" bson:"node_id"`
	OwnerID    string    `json:"owner_id" bson:"owner_id"`
	ContainerID string  `json:"container_id" bson:"container_id"`
	CreatedAt  time.Time `json:"created_at" bson:"created_at"`
	ExpiresAt  time.Time `json:"expires_at" bson:"expires_at"`
	Active     bool      `json:"active" bson:"active"`
}

// EnvVar 表示环境变量键值对。
//
// 用于任务模板和任务定义中传递自定义环境变量。
type EnvVar struct {
	Name  string `json:"name" bson:"name"`
	Value string `json:"value" bson:"value"`
}

// TerminalMessage 是 Web terminal 会话中传输的数据帧。
//
// 用于在前端<->后端<->节点之间传输数据：
//  - Data: 终端输出的字节（base64 编码
//  - Input: 用户输入的字符
//  - Resize: 窗口大小改变事件
//  - Close: 会话关闭
type TerminalMessage struct {
	Type    string `json:"type"` // "data" | "input" | "resize" | "close"
	Data    string `json:"data,omitempty"`
	Rows    uint16 `json:"rows,omitempty"`
	Cols    uint16 `json:"cols,omitempty"`
}

// ==== 工作流编排相关领域模型 ====

// ExecutorType 执行器类型
// 定义工作流节点支持的不同执行方式
type ExecutorType string

const (
	ExecutorTypeContainer ExecutorType = "container" // 容器镜像执行（兼容原有）
	ExecutorTypeShell      ExecutorType = "shell"      // Shell 脚本直接执行
	ExecutorTypePython     ExecutorType = "python"     // Python 脚本执行
	ExecutorTypeAPI        ExecutorType = "api"        // 自定义 API 调用
)

// WorkflowStatus 工作流定义状态
type WorkflowStatus string

const (
	WorkflowStatusDraft     WorkflowStatus = "draft"     // 草稿，不可执行
	WorkflowStatusActive    WorkflowStatus = "active"    // 活跃，可以执行
	WorkflowStatusArchived WorkflowStatus = "archived" // 已归档，不再执行
)

// ContainerExecutorConfig 容器执行配置
// 兼容原有的容器执行模式，使用用户自定义镜像
type ContainerExecutorConfig struct {
	Image   string            `json:"image" bson:"image"`             // 容器镜像
	Command []string          `json:"command" bson:"command"`         // 启动命令
	Args    []string          `json:"args" bson:"args"`               // 命令参数
	Env     map[string]string `json:"env" bson:"env"`                 // 环境变量
}

// ShellExecutorConfig Shell 脚本执行配置
// 使用预构建的 shell-runner 镜像执行用户脚本
type ShellExecutorConfig struct {
	Script string            `json:"script" bson:"script"`           // Shell 脚本内容
	Env    map[string]string `json:"env" bson:"env"`                 // 环境变量
}

// PythonExecutorConfig Python 脚本执行配置
// 使用预构建的 python-runner 镜像执行用户脚本
type PythonExecutorConfig struct {
	Script       string            `json:"script" bson:"script"`           // Python 脚本内容
	Requirements []string          `json:"requirements,omitempty" bson:"requirements,omitempty"` // pip 依赖列表
	Env          map[string]string `json:"env" bson:"env"`                 // 环境变量
}

// APIExecutorConfig 自定义 API 调用配置
// 使用预构建的 api-runner 镜像发起 HTTP 请求
type APIExecutorConfig struct {
	URL         string            `json:"url" bson:"url"`                   // API 地址
	Method      string            `json:"method" bson:"method"`             // HTTP 方法 (GET/POST/PUT/DELETE)
	Headers     map[string]string `json:"headers" bson:"headers"`           // 请求头
	Body        string            `json:"body" bson:"body"`                 // 请求体模板（支持 ${param} 占位符替换）
	Timeout     int               `json:"timeout" bson:"timeout"`           // 超时秒数
	SuccessCodes []int             `json:"successCodes" bson:"success_codes"` // 视为成功的 HTTP 状态码
}

// Workflow 工作流定义
// 由多个任务节点组成 DAG，定义任务执行顺序和依赖关系
type Workflow struct {
	ID          string              `json:"id" bson:"_id"`
	Name        string              `json:"name" bson:"name"`
	Description string              `json:"description" bson:"description"`
	Version     string              `json:"version" bson:"version"`

	// 工作流定义
	Tasks       []WorkflowTask      `json:"tasks" bson:"tasks"`       // 任务列表
	Edges       []Edge              `json:"edges" bson:"edges"`       // 依赖边

	// 参数定义
	Parameters  []Parameter         `json:"parameters" bson:"parameters"` // 可传入参数

	// 元数据
	Labels      map[string]string   `json:"labels" bson:"labels"`
	CreatedBy   string              `json:"createdBy" bson:"createdBy"`
	CreatedAt   time.Time           `json:"createdAt" bson:"createdAt"`
	UpdatedAt   time.Time           `json:"updatedAt" bson:"updatedAt"`

	// 状态
	Status      WorkflowStatus      `json:"status" bson:"status"` // draft/active/archived
}

// WorkflowTask 工作流中的单个任务节点
// 支持多种执行类型：容器、Shell、Python、API
type WorkflowTask struct {
	ID          string            `json:"id" bson:"id"`
	Name        string            `json:"name" bson:"name"`
	Description string            `json:"description" bson:"description"`

	// === 执行器类型与配置 ===
	ExecutorType ExecutorType     `json:"executorType" bson:"executor_type"`
	// 不同执行器类型使用对应的配置结构体，其他为 nil
	ContainerConfig *ContainerExecutorConfig `json:"containerConfig,omitempty" bson:"container_config,omitempty"`
	ShellConfig     *ShellExecutorConfig     `json:"shellConfig,omitempty" bson:"shell_config,omitempty"`
	PythonConfig    *PythonExecutorConfig    `json:"pythonConfig,omitempty" bson:"python_config,omitempty"`
	APIConfig       *APIExecutorConfig       `json:"apiConfig,omitempty" bson:"api_config,omitempty"`

	// 兼容旧数据：保留原有字段，新代码优先使用新配置
	// Deprecated: 使用 ContainerConfig 替代
	Image       string            `json:"image,omitempty" bson:"image"`
	// Deprecated: 使用 ContainerConfig 替代
	Command     []string          `json:"command,omitempty" bson:"command"`
	// Deprecated: 使用 ContainerConfig 替代
	Args        []string          `json:"args,omitempty" bson:"args"`
	// Deprecated: 使用 ContainerConfig 替代
	Env         map[string]string `json:"env,omitempty" bson:"env"`

	// 资源需求
	Resources   ResourceSpec       `json:"resources" bson:"resources"`

	// 执行配置
	Timeout     Duration          `json:"timeout" bson:"timeout"`
	Retries     int               `json:"retries" bson:"retries"`
	RetryDelay  Duration          `json:"retryDelay" bson:"retryDelay"`

	// 条件执行
	Condition   string            `json:"condition" bson:"condition"`

	// 输入输出
	Inputs      []Input           `json:"inputs" bson:"inputs"`
	Outputs     []Output          `json:"outputs" bson:"outputs"`
}

// Edge 依赖边，表示任务之间的依赖关系
type Edge struct {
	From      string `json:"from" bson:"from"`
	To        string `json:"to" bson:"to"`
	Condition string `json:"condition,omitempty" bson:"condition,omitempty"`
}

// Parameter 参数定义
type Parameter struct {
	Name        string      `json:"name" bson:"name"`
	Description string      `json:"description" bson:"description"`
	Type        string      `json:"type" bson:"type"` // string/int/float/bool
	Default     interface{} `json:"default" bson:"default"`
	Required    bool        `json:"required" bson:"required"`
}

// ResourceSpec 资源规格
type ResourceSpec struct {
	GPUCount  int    `json:"gpuCount" bson:"gpuCount"`
	GPUMemory string `json:"gpuMemory" bson:"gpuMemory"`
	GPUModel  string `json:"gpuModel" bson:"gpuModel"`
	CPU       string `json:"cpu" bson:"cpu"`
	Memory    string `json:"memory" bson:"memory"`
}

// Input 任务输入
type Input struct {
	Name     string `json:"name" bson:"name"`
	From     string `json:"from" bson:"from"`       // 来自哪个上游任务的哪个输出
	Default  string `json:"default" bson:"default"` // 默认值（参数）
	Required bool   `json:"required" bson:"required"`
}

// Output 任务输出
type Output struct {
	Name string `json:"name" bson:"name"`
	Path string `json:"path" bson:"path"` // 输出路径（对象存储位置）
	Type string `json:"type" bson:"type"`   // 类型：file/parameter/result
}

// Duration 可序列化 duration 包装
type Duration struct {
	Duration time.Duration `json:"duration"`
}

// UnmarshalJSON 反序列化 JSON
func (d *Duration) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	dur, err := time.ParseDuration(s)
	if err != nil {
		return err
	}
	d.Duration = dur
	return nil
}

// MarshalJSON 序列化 JSON
func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.Duration.String())
}

// WorkflowRunStatus 工作流执行实例状态
type WorkflowRunStatus string

const (
	WorkflowRunStatusPending   WorkflowRunStatus = "Pending"   // 等待执行
	WorkflowRunStatusRunning   WorkflowRunStatus = "Running"   // 正在运行
	WorkflowRunStatusSucceeded WorkflowRunStatus = "Succeeded" // 执行成功
	WorkflowRunStatusFailed    WorkflowRunStatus = "Failed"    // 执行失败
	WorkflowRunStatusCanceled  WorkflowRunStatus = "Canceled"  // 已取消
)

// TaskRunStatus 任务执行实例状态
type TaskRunStatus string

const (
	TaskRunStatusPending   TaskRunStatus = "Pending"
	TaskRunStatusRunning   TaskRunStatus = "Running"
	TaskRunStatusSucceeded TaskRunStatus = "Succeeded"
	TaskRunStatusFailed    TaskRunStatus = "Failed"
	TaskRunStatusCanceled  TaskRunStatus = "Canceled"
	TaskRunStatusSkipped  TaskRunStatus = "Skipped"
)

// WorkflowRun 工作流执行实例
type WorkflowRun struct {
	ID          string                  `json:"id" bson:"_id"`
	WorkflowID  string                  `json:"workflowId" bson:"workflowId"`
	Name        string                  `json:"name" bson:"name"`

	// 参数值
	Parameters map[string]interface{} `json:"parameters" bson:"parameters"`

	// 状态
	Status      WorkflowRunStatus      `json:"status" bson:"status"`

	// 执行信息
	StartedAt  *time.Time            `json:"startedAt" bson:"startedAt"`
	FinishedAt *time.Time            `json:"finishedAt" bson:"finishedAt"`
	Duration   time.Duration         `json:"duration" bson:"duration"`

	// 任务执行实例
	TaskRuns []TaskRun              `json:"taskRuns" bson:"taskRuns"`

	// 触发信息
	Trigger TriggerInfo             `json:"trigger" bson:"trigger"`

	// 元数据
	CreatedBy string                  `json:"createdBy" bson:"createdBy"`
	CreatedAt time.Time             `json:"createdAt" bson:"createdAt"`

	// 结果
	Outputs map[string]interface{} `json:"outputs" bson:"outputs"`
	Message string                  `json:"message" bson:"message"`
}

// TaskRun 单个任务的执行实例
type TaskRun struct {
	ID           string          `json:"id" bson:"id"`
	TaskID       string          `json:"taskId" bson:"taskId"`
	TaskName     string          `json:"taskName" bson:"taskName"`

	// 关联的原生 QingTian 任务 ID
	NativeTaskID string          `json:"nativeTaskId" bson:"nativeTaskId"`

	// 状态
	Status       TaskRunStatus     `json:"status" bson:"status"`

	// 执行信息
	StartedAt  *time.Time        `json:"startedAt" bson:"startedAt"`
	FinishedAt *time.Time        `json:"finishedAt" bson:"finishedAt"`
	Duration   time.Duration     `json:"duration" bson:"duration"`

	// 重试信息
	RetryCount int               `json:"retryCount" bson:"retryCount"`

	// 节点信息
	NodeID string                `json:"nodeId" bson:"nodeId"`

	// 输入输出
	Inputs  map[string]interface{} `json:"inputs" bson:"inputs"`
	Outputs map[string]interface{} `json:"outputs" bson:"outputs"`

	// 日志和错误
	LogPath string                `json:"logPath" bson:"logPath"`
	Error  string                `json:"error" bson:"error"`
}

// ConcurrencyPolicy 定时调度并发策略
type ConcurrencyPolicy string

const (
	ConcurrencyPolicyAllow   ConcurrencyPolicy = "Allow"   // 允许并发，多个实例同时运行
	ConcurrencyPolicyForbid  ConcurrencyPolicy = "Forbid"  // 禁止并发，如果已有实例在运行，跳过新执行
	ConcurrencyPolicyReplace ConcurrencyPolicy = "Replace" // 替换，停止正在运行的实例，启动新的
)

// ScheduleStatus 定时计划状态
type ScheduleStatus string

const (
	ScheduleStatusActive  ScheduleStatus = "active"
	ScheduleStatusPaused ScheduleStatus = "paused"
	ScheduleStatusError  ScheduleStatus = "error"
)

// Schedule 定时计划
type Schedule struct {
	ID             string                `json:"id" bson:"_id"`
	Name           string                `json:"name" bson:"name"`
	Description    string                `json:"description" bson:"description"`

	// 关联的工作流
	WorkflowID     string                `json:"workflowId" bson:"workflowId"`

	// Cron 表达式
	CronExpr string              `json:"cronExpr" bson:"cronExpr"`

	// 时区
	Timezone string              `json:"timezone" bson:"timezone"`

	// 参数
	Parameters map[string]interface{} `json:"parameters" bson:"parameters"`

	// 执行配置
	ConcurrencyPolicy       ConcurrencyPolicy `json:"concurrencyPolicy" bson:"concurrencyPolicy"`
	SuccessfulJobsHistoryLimit int               `json:"successfulJobsHistoryLimit" bson:"successfulJobsHistoryLimit"`
	FailedJobsHistoryLimit    int               `json:"failedJobsHistoryLimit" bson:"failedJobsHistoryLimit"`

	// 状态
	Enabled bool          `json:"enabled" bson:"enabled"`
	Status  ScheduleStatus `json:"status" bson:"status"`

	// 上次执行
	LastRunAt   *time.Time `json:"lastRunAt" bson:"lastRunAt"`
	LastRunID   string    `json:"lastRunId" bson:"lastRunId"`

	// 下次执行
	NextRunAt   *time.Time `json:"nextRunAt" bson:"nextRunAt"`

	// 元数据
	CreatedBy string        `json:"createdBy" bson:"createdBy"`
	CreatedAt time.Time     `json:"createdAt" bson:"createdAt"`
	UpdatedAt time.Time     `json:"updatedAt" bson:"updatedAt"`
}

// TriggerInfo 触发信息，记录是谁/为什么触发了这次执行
type TriggerInfo struct {
	Type        string    `json:"type" bson:"type"` // manual/schedule/webhook
	ScheduleID  string    `json:"scheduleId,omitempty" bson:"scheduleId,omitempty"`
	TriggeredBy string    `json:"triggeredBy" bson:"triggeredBy"`
	TriggeredAt time.Time `json:"triggeredAt" bson:"triggeredAt"`
}

// Feature 功能开关，用于企业版功能门控
type Feature string

const (
	FeatureMultiTenancy  Feature = "multi_tenancy"  // 多租户/项目管理
	FeatureMetering      Feature = "metering"        // 计量计费
	FeatureTemplate      Feature = "template"       // 任务模板
	FeatureTerminal      Feature = "terminal"       // Web 终端
	FeatureAutoScaling   Feature = "auto_scaling"    // 自动扩缩容
	FeatureIdleReclaim  Feature = "idle_reclaim"  // 空闲节点回收
	FeatureSharding     Feature = "sharding"       // 调度分片水平扩展
	FeatureOIDC         Feature = "oidc"           // OIDC SSO 单点登录
	// 工作流编排默认就是企业版功能
	FeatureWorkflow     Feature = "workflow"       // DAG 工作流编排
	// 定时调度默认就是企业版功能
	FeatureScheduling   Feature = "scheduling"     // Cron 定时任务调度
)

// FeatureEnabled 检查给定 edition 是否启用了该功能
func FeatureEnabled(edition Edition, feature Feature) bool {
	if edition == EditionEnterprise {
		return true
	}
	// 社区版默认不启用高级功能
	return false
}
