// 本文件提供 UserRepository / NodeRepository / TaskRepository 的内存实现。
//
// 内存实现用于单元测试与无外部依赖的本地演示，所有数据保存在进程内存中，
// 通过读写锁保证并发安全。生产环境应使用 MongoDB 实现。
package repository

import (
	"context"
	"sync"

	"qingtian/internal/domain"
	"qingtian/pkg/apperr"
)

// MemoryUserRepo 是 UserRepository 的内存实现。
type MemoryUserRepo struct {
	mu    sync.RWMutex
	byID  map[string]*domain.User
	names map[string]string // username -> id 索引，用于唯一性校验与按名查询
}

// NewMemoryUserRepo 创建内存用户仓储。
func NewMemoryUserRepo() *MemoryUserRepo {
	return &MemoryUserRepo{
		byID:  make(map[string]*domain.User),
		names: make(map[string]string),
	}
}

// Create 实现 UserRepository.Create。
func (r *MemoryUserRepo) Create(_ context.Context, u *domain.User) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.names[u.Username]; ok {
		return apperr.Conflict("用户名已存在: " + u.Username)
	}
	cp := *u // 拷贝存储，避免外部修改影响内部状态
	r.byID[u.ID] = &cp
	r.names[u.Username] = u.ID
	return nil
}

// GetByID 实现 UserRepository.GetByID。
func (r *MemoryUserRepo) GetByID(_ context.Context, id string) (*domain.User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	u, ok := r.byID[id]
	if !ok {
		return nil, apperr.NotFound("用户不存在: " + id)
	}
	cp := *u
	return &cp, nil
}

// GetByUsername 实现 UserRepository.GetByUsername。
func (r *MemoryUserRepo) GetByUsername(_ context.Context, username string) (*domain.User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	id, ok := r.names[username]
	if !ok {
		return nil, apperr.NotFound("用户不存在: " + username)
	}
	u := r.byID[id]
	cp := *u
	return &cp, nil
}

// List 实现 UserRepository.List。
func (r *MemoryUserRepo) List(_ context.Context) ([]*domain.User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.User, 0, len(r.byID))
	for _, u := range r.byID {
		cp := *u
		out = append(out, &cp)
	}
	return out, nil
}

// FindByExternalID 实现 UserRepository.FindByExternalID。
func (r *MemoryUserRepo) FindByExternalID(_ context.Context, externalID string) (*domain.User, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, u := range r.byID {
		if u.ExternalID == externalID {
			cp := *u
			return &cp, nil
		}
	}
	return nil, apperr.NotFound("用户不存在: external_id=" + externalID)
}

// Update 实现 UserRepository.Update。
func (r *MemoryUserRepo) Update(_ context.Context, u *domain.User) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.byID[u.ID]; !ok {
		return apperr.NotFound("用户不存在: " + u.ID)
	}
	// 如果用户名改变，需要更新名称索引
	old, ok := r.byID[u.ID]
	if ok && old.Username != u.Username {
		delete(r.names, old.Username)
		r.names[u.Username] = u.ID
	}
	cp := *u
	r.byID[u.ID] = &cp
	return nil
}

// MemoryNodeRepo 是 NodeRepository 的内存实现。
type MemoryNodeRepo struct {
	mu    sync.RWMutex
	nodes map[string]*domain.Node
}

// NewMemoryNodeRepo 创建内存节点仓储。
func NewMemoryNodeRepo() *MemoryNodeRepo {
	return &MemoryNodeRepo{nodes: make(map[string]*domain.Node)}
}

// Upsert 实现 NodeRepository.Upsert。
func (r *MemoryNodeRepo) Upsert(_ context.Context, n *domain.Node) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	cp := *n
	r.nodes[n.ID] = &cp
	return nil
}

// GetByID 实现 NodeRepository.GetByID。
func (r *MemoryNodeRepo) GetByID(_ context.Context, id string) (*domain.Node, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	n, ok := r.nodes[id]
	if !ok {
		return nil, apperr.NotFound("节点不存在: " + id)
	}
	cp := *n
	return &cp, nil
}

// List 实现 NodeRepository.List。
func (r *MemoryNodeRepo) List(_ context.Context) ([]*domain.Node, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.Node, 0, len(r.nodes))
	for _, n := range r.nodes {
		cp := *n
		out = append(out, &cp)
	}
	return out, nil
}

// UpdateStatus 实现 NodeRepository.UpdateStatus。
func (r *MemoryNodeRepo) UpdateStatus(_ context.Context, id string, status domain.NodeStatus) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	n, ok := r.nodes[id]
	if !ok {
		return apperr.NotFound("节点不存在: " + id)
	}
	n.Status = status
	return nil
}

// UpdateHeartbeat 实现 NodeRepository.UpdateHeartbeat。
func (r *MemoryNodeRepo) UpdateHeartbeat(_ context.Context, id string, gpus []domain.GPU) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	n, ok := r.nodes[id]
	if !ok {
		return apperr.NotFound("节点不存在: " + id)
	}
	n.Status = domain.NodeOnline
	if gpus != nil {
		n.GPUs = gpus
	}
	return nil
}

// MemoryTaskRepo 是 TaskRepository 的内存实现。
type MemoryTaskRepo struct {
	mu    sync.RWMutex
	tasks map[string]*domain.Task
}

// NewMemoryTaskRepo 创建内存任务仓储。
func NewMemoryTaskRepo() *MemoryTaskRepo {
	return &MemoryTaskRepo{tasks: make(map[string]*domain.Task)}
}

// Create 实现 TaskRepository.Create。
func (r *MemoryTaskRepo) Create(_ context.Context, t *domain.Task) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.tasks[t.ID]; ok {
		return apperr.Conflict("任务已存在: " + t.ID)
	}
	cp := *t
	r.tasks[t.ID] = &cp
	return nil
}

// GetByID 实现 TaskRepository.GetByID。
func (r *MemoryTaskRepo) GetByID(_ context.Context, id string) (*domain.Task, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tasks[id]
	if !ok {
		return nil, apperr.NotFound("任务不存在: " + id)
	}
	cp := *t
	return &cp, nil
}

// Update 实现 TaskRepository.Update。
func (r *MemoryTaskRepo) Update(_ context.Context, t *domain.Task) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.tasks[t.ID]; !ok {
		return apperr.NotFound("任务不存在: " + t.ID)
	}
	cp := *t
	r.tasks[t.ID] = &cp
	return nil
}

// List 实现 TaskRepository.List。
func (r *MemoryTaskRepo) List(_ context.Context, status domain.TaskStatus) ([]*domain.Task, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.Task, 0, len(r.tasks))
	for _, t := range r.tasks {
		if status != "" && t.Status != status {
			continue
		}
		cp := *t
		out = append(out, &cp)
	}
	return out, nil
}

// ListByOwner 实现 TaskRepository.ListByOwner。
func (r *MemoryTaskRepo) ListByOwner(_ context.Context, owner string) ([]*domain.Task, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.Task, 0)
	for _, t := range r.tasks {
		if t.Owner == owner {
			cp := *t
			out = append(out, &cp)
		}
	}
	return out, nil
}

// CountActive 实现 TaskRepository.CountActive。
func (r *MemoryTaskRepo) CountActive(_ context.Context, owner, projectID string) (int, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	n := 0
	for _, t := range r.tasks {
		if t.Status != domain.TaskScheduled && t.Status != domain.TaskRunning {
			continue
		}
		if projectID != "" {
			if t.ProjectID == projectID {
				n++
			}
			continue
		}
		if t.Owner == owner {
			n++
		}
	}
	return n, nil
}

// ListByNode 实现 TaskRepository.ListByNode。
func (r *MemoryTaskRepo) ListByNode(_ context.Context, nodeID string, statuses ...domain.TaskStatus) ([]*domain.Task, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.Task, 0)
	for _, t := range r.tasks {
		if t.NodeID != nodeID {
			continue
		}
		// 如果没有指定状态，全部返回；否则只返回匹配状态的
		if len(statuses) == 0 {
			cp := *t
			out = append(out, &cp)
			continue
		}
		for _, s := range statuses {
			if t.Status == s {
				cp := *t
				out = append(out, &cp)
				break
			}
		}
	}
	return out, nil
}

// MemoryUsageRepo 是 UsageRepository 的内存实现。
type MemoryUsageRepo struct {
	mu      sync.RWMutex
	records []*domain.UsageRecord
}

// NewMemoryUsageRepo 创建内存用量仓储。
func NewMemoryUsageRepo() *MemoryUsageRepo {
	return &MemoryUsageRepo{}
}

// Create 实现 UsageRepository.Create。
func (r *MemoryUsageRepo) Create(_ context.Context, u *domain.UsageRecord) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	cp := *u
	r.records = append(r.records, &cp)
	return nil
}

// List 实现 UsageRepository.List。
func (r *MemoryUsageRepo) List(_ context.Context) ([]*domain.UsageRecord, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.UsageRecord, 0, len(r.records))
	for _, u := range r.records {
		cp := *u
		out = append(out, &cp)
	}
	return out, nil
}

// ListByOwner 实现 UsageRepository.ListByOwner。
func (r *MemoryUsageRepo) ListByOwner(_ context.Context, owner string) ([]*domain.UsageRecord, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.UsageRecord, 0)
	for _, u := range r.records {
		if u.Owner == owner {
			cp := *u
			out = append(out, &cp)
		}
	}
	return out, nil
}

// MemoryQuotaRepo 是 QuotaRepository 的内存实现。
type MemoryQuotaRepo struct {
	mu     sync.RWMutex
	quotas map[string]*domain.Quota
}

// NewMemoryQuotaRepo 创建内存配额仓储。
func NewMemoryQuotaRepo() *MemoryQuotaRepo {
	return &MemoryQuotaRepo{quotas: make(map[string]*domain.Quota)}
}

// Get 实现 QuotaRepository.Get。
func (r *MemoryQuotaRepo) Get(_ context.Context, subject string) (*domain.Quota, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	q, ok := r.quotas[subject]
	if !ok {
		return nil, apperr.NotFound("配额不存在: " + subject)
	}
	cp := *q
	return &cp, nil
}

// Upsert 实现 QuotaRepository.Upsert。
func (r *MemoryQuotaRepo) Upsert(_ context.Context, q *domain.Quota) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	cp := *q
	r.quotas[q.Subject] = &cp
	return nil
}

// AddUsage 实现 QuotaRepository.AddUsage。
func (r *MemoryQuotaRepo) AddUsage(_ context.Context, subject string, gpuHours float64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	q, ok := r.quotas[subject]
	if !ok {
		return nil // 无配额视为不限，忽略累加
	}
	q.UsedGPUHours += gpuHours
	return nil
}

// MemoryProjectRepo 是 ProjectRepository 的内存实现。
type MemoryProjectRepo struct {
	mu       sync.RWMutex
	projects map[string]*domain.Project
}

// NewMemoryProjectRepo 创建内存项目仓储。
func NewMemoryProjectRepo() *MemoryProjectRepo {
	return &MemoryProjectRepo{projects: make(map[string]*domain.Project)}
}

// Create 实现 ProjectRepository.Create。
func (r *MemoryProjectRepo) Create(_ context.Context, p *domain.Project) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.projects[p.ID]; ok {
		return apperr.Conflict("项目已存在: " + p.ID)
	}
	cp := *p
	r.projects[p.ID] = &cp
	return nil
}

// GetByID 实现 ProjectRepository.GetByID。
func (r *MemoryProjectRepo) GetByID(_ context.Context, id string) (*domain.Project, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.projects[id]
	if !ok {
		return nil, apperr.NotFound("项目不存在: " + id)
	}
	cp := *p
	return &cp, nil
}

// Update 实现 ProjectRepository.Update。
func (r *MemoryProjectRepo) Update(_ context.Context, p *domain.Project) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.projects[p.ID]; !ok {
		return apperr.NotFound("项目不存在: " + p.ID)
	}
	cp := *p
	r.projects[p.ID] = &cp
	return nil
}

// ListByMember 实现 ProjectRepository.ListByMember。
func (r *MemoryProjectRepo) ListByMember(_ context.Context, userID string) ([]*domain.Project, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.Project, 0)
	for _, p := range r.projects {
		if p.HasMember(userID) {
			cp := *p
			out = append(out, &cp)
		}
	}
	return out, nil
}


// MemoryTaskTemplateRepo 是 TaskTemplateRepository 的内存实现。
type MemoryTaskTemplateRepo struct {
	mu         sync.RWMutex
	templates map[string]*domain.TaskTemplate
}

// NewMemoryTaskTemplateRepo 创建内存任务模板仓储。
func NewMemoryTaskTemplateRepo() *MemoryTaskTemplateRepo {
	return &MemoryTaskTemplateRepo{templates: make(map[string]*domain.TaskTemplate)}
}

// Create 实现 TaskTemplateRepository.Create。
func (r *MemoryTaskTemplateRepo) Create(_ context.Context, t *domain.TaskTemplate) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.templates[t.ID]; ok {
		return apperr.Conflict("模板已存在: " + t.ID)
	}
	cp := *t
	r.templates[t.ID] = &cp
	return nil
}

// GetByID 实现 TaskTemplateRepository.GetByID。
func (r *MemoryTaskTemplateRepo) GetByID(_ context.Context, id string) (*domain.TaskTemplate, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.templates[id]
	if !ok {
		return nil, apperr.NotFound("模板不存在: " + id)
	}
	cp := *t
	return &cp, nil
}

// Update 实现 TaskTemplateRepository.Update。
func (r *MemoryTaskTemplateRepo) Update(_ context.Context, t *domain.TaskTemplate) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.templates[t.ID]; !ok {
		return apperr.NotFound("模板不存在: " + t.ID)
	}
	cp := *t
	r.templates[t.ID] = &cp
	return nil
}

// Delete 实现 TaskTemplateRepository.Delete。
func (r *MemoryTaskTemplateRepo) Delete(_ context.Context, id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.templates, id)
	return nil
}

// ListByOwner 实现 TaskTemplateRepository.ListByOwner。
func (r *MemoryTaskTemplateRepo) ListByOwner(_ context.Context, ownerID string) ([]*domain.TaskTemplate, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]*domain.TaskTemplate, 0)
	for _, t := range r.templates {
		if t.OwnerID == ownerID {
			cp := *t
			out = append(out, &cp)
		}
	}
	return out, nil
}

// ListByProject 实现 TaskTemplateRepository.ListByProject。
func (r *MemoryTaskTemplateRepo) ListByProject(_ context.Context, projectID string) ([]*domain.TaskTemplate, error) {
	r.mu.RLock()
	defer r.mu.Unlock()
	out := make([]*domain.TaskTemplate, 0)
	for _, t := range r.templates {
		if t.ProjectID == projectID {
			cp := *t
			out = append(out, &cp)
		}
	}
	return out, nil
}
