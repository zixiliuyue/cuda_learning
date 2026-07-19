// 本文件实现项目服务：团队工作区管理，支撑任务/数据集/配额的团队共享。
//
// 项目服务把平台从"个人工具"升级为"团队协作平台"：用户可创建项目、
// 邀请成员，团队内任务与资源在同一空间协作，配额也可按项目统一管理。
package service

import (
	"context"
	"time"

	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// ProjectService 提供项目创建、成员管理与查询能力。
type ProjectService struct {
	repo repository.ProjectRepository
	now  func() time.Time
}

// NewProjectService 创建项目服务。
func NewProjectService(repo repository.ProjectRepository) *ProjectService {
	return &ProjectService{repo: repo, now: time.Now}
}

// Create 创建一个项目，创建者自动成为所有者与成员。
//
// 参数：
//   - ctx:     上下文。
//   - name:    项目名称。
//   - ownerID: 创建者用户 ID。
//
// 返回值：
//   - *domain.Project: 创建成功的项目。
//   - error:           参数非法或持久化失败时返回。
func (s *ProjectService) Create(ctx context.Context, name, ownerID string) (*domain.Project, error) {
	if name == "" || ownerID == "" {
		return nil, apperr.InvalidArgument("项目名称与所有者不能为空")
	}
	p := &domain.Project{
		ID:        idgen.New("proj"),
		Name:      name,
		OwnerID:   ownerID,
		Members:   []string{ownerID},
		CreatedAt: s.now(),
	}
	if err := s.repo.Create(ctx, p); err != nil {
		return nil, err
	}
	return p, nil
}

// AddMember 向项目添加成员（仅所有者可操作）。
//
// 参数：
//   - ctx:       上下文。
//   - projectID: 项目 ID。
//   - operator:  操作者用户 ID，必须为项目所有者。
//   - userID:    待加入成员用户 ID。
//
// 返回值：
//   - *domain.Project: 更新后的项目。
//   - error:           无权限、项目不存在或持久化失败时返回。
func (s *ProjectService) AddMember(ctx context.Context, projectID, operator, userID string) (*domain.Project, error) {
	p, err := s.repo.GetByID(ctx, projectID)
	if err != nil {
		return nil, err
	}
	if p.OwnerID != operator {
		return nil, apperr.Forbidden("仅项目所有者可添加成员")
	}
	if p.HasMember(userID) {
		return p, nil // 已是成员，幂等返回
	}
	p.Members = append(p.Members, userID)
	if err := s.repo.Update(ctx, p); err != nil {
		return nil, err
	}
	return p, nil
}

// Get 查询项目详情，要求 requester 为项目成员，防止越权查看。
func (s *ProjectService) Get(ctx context.Context, projectID, requester string) (*domain.Project, error) {
	p, err := s.repo.GetByID(ctx, projectID)
	if err != nil {
		return nil, err
	}
	if !p.HasMember(requester) {
		return nil, apperr.Forbidden("非项目成员无权查看")
	}
	return p, nil
}

// ListMine 返回请求者所属的全部项目。
func (s *ProjectService) ListMine(ctx context.Context, userID string) ([]*domain.Project, error) {
	return s.repo.ListByMember(ctx, userID)
}
