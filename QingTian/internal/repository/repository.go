// Package repository 定义领域实体的持久化接口（仓储模式）。
//
// 通过接口隔离业务逻辑与具体存储实现，使业务代码不直接依赖 MongoDB。
// 平台同时提供内存实现（用于单元测试）与 MongoDB 实现（用于生产）。
package repository

import (
	"context"

	"qingtian/internal/domain"
)

// UserRepository 定义用户持久化能力。
type UserRepository interface {
	// Create 创建用户。用户名冲突时应返回 CodeConflict 业务错误。
	Create(ctx context.Context, u *domain.User) error
	// GetByID 按 ID 查询用户，不存在时返回 CodeNotFound。
	GetByID(ctx context.Context, id string) (*domain.User, error)
	// GetByUsername 按用户名查询用户，不存在时返回 CodeNotFound。
	GetByUsername(ctx context.Context, username string) (*domain.User, error)
	// FindByExternalID 按第三方 OIDC 外部 ID 查询用户，不存在时返回 CodeNotFound。
	FindByExternalID(ctx context.Context, externalID string) (*domain.User, error)
	// Update 更新用户信息。
	Update(ctx context.Context, u *domain.User) error
	// List 返回全部用户。
	List(ctx context.Context) ([]*domain.User, error)
}

// NodeRepository 定义节点持久化能力。
type NodeRepository interface {
	// Upsert 创建或更新节点（按 ID）。节点重复注册时覆盖旧记录。
	Upsert(ctx context.Context, n *domain.Node) error
	// GetByID 按 ID 查询节点，不存在时返回 CodeNotFound。
	GetByID(ctx context.Context, id string) (*domain.Node, error)
	// List 返回全部节点。
	List(ctx context.Context) ([]*domain.Node, error)
	// UpdateStatus 更新节点在线状态。
	UpdateStatus(ctx context.Context, id string, status domain.NodeStatus) error
	// UpdateHeartbeat 更新节点心跳时间与实时 GPU 信息。
	UpdateHeartbeat(ctx context.Context, id string, gpus []domain.GPU) error
}

// TaskRepository 定义任务持久化能力。
type TaskRepository interface {
	// Create 创建任务。
	Create(ctx context.Context, t *domain.Task) error
	// GetByID 按 ID 查询任务，不存在时返回 CodeNotFound。
	GetByID(ctx context.Context, id string) (*domain.Task, error)
	// Update 全量更新任务。
	Update(ctx context.Context, t *domain.Task) error
	// List 按可选状态过滤返回任务列表；status 为空表示不过滤。
	List(ctx context.Context, status domain.TaskStatus) ([]*domain.Task, error)
	// ListByOwner 返回指定用户的任务列表。
	ListByOwner(ctx context.Context, owner string) ([]*domain.Task, error)
	// ListByNode 返回指定节点上处于指定状态的任务列表。
	// 用于节点离线故障时，找出未完成任务重新排队。
	ListByNode(ctx context.Context, nodeID string, statuses ...domain.TaskStatus) ([]*domain.Task, error)
	// CountActive 统计某主体当前处于活动态（scheduled/running）的任务数，用于并发配额校验。
	// projectID 非空时按项目统计，否则按 owner 统计。
	CountActive(ctx context.Context, owner, projectID string) (int, error)
}

// UsageRepository 定义算力用量记录的持久化能力，是成本计量与计费的凭证库。
type UsageRepository interface {
	// Create 追加一条用量记录。
	Create(ctx context.Context, u *domain.UsageRecord) error
	// List 返回全部用量记录。
	List(ctx context.Context) ([]*domain.UsageRecord, error)
	// ListByOwner 返回指定用户的用量记录。
	ListByOwner(ctx context.Context, owner string) ([]*domain.UsageRecord, error)
}

// QuotaRepository 定义配额的持久化能力。
type QuotaRepository interface {
	// Get 按主体（用户/项目）查询配额，不存在时返回 CodeNotFound。
	Get(ctx context.Context, subject string) (*domain.Quota, error)
	// Upsert 创建或更新配额。
	Upsert(ctx context.Context, q *domain.Quota) error
	// AddUsage 原子累加主体的已用卡时，不存在配额时视为无限并忽略。
	AddUsage(ctx context.Context, subject string, gpuHours float64) error
}

// ProjectRepository 定义项目/团队工作区的持久化能力。
type ProjectRepository interface {
	// Create 创建项目。
	Create(ctx context.Context, p *domain.Project) error
	// GetByID 按 ID 查询项目，不存在时返回 CodeNotFound。
	GetByID(ctx context.Context, id string) (*domain.Project, error)
	// Update 全量更新项目。
	Update(ctx context.Context, p *domain.Project) error
	// Delete 删除项目（软删，仅项目所有者可操作）。
	Delete(ctx context.Context, id string) error
	// ListByMember 返回某用户所属的全部项目。
	ListByMember(ctx context.Context, userID string) ([]*domain.Project, error)
}

// TaskTemplateRepository 定义任务模板持久化能力。
type TaskTemplateRepository interface {
	// Create 创建模板。
	Create(ctx context.Context, t *domain.TaskTemplate) error
	// GetByID 按 ID 查询模板，不存在时返回 CodeNotFound。
	GetByID(ctx context.Context, id string) (*domain.TaskTemplate, error)
	// Update 更新模板。
	Update(ctx context.Context, t *domain.TaskTemplate) error
	// Delete 删除模板。
	Delete(ctx context.Context, id string) error
	// ListByOwner 返回指定用户的所有模板。
	ListByOwner(ctx context.Context, ownerID string) ([]*domain.TaskTemplate, error)
	// ListByProject 返回指定项目的所有模板。
	ListByProject(ctx context.Context, projectID string) ([]*domain.TaskTemplate, error)
}
