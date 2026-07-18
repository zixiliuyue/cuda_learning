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
