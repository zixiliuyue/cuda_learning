// 本文件实现任务模板服务：模板创建、更新、删除、查询，提升日常提交效率。
//
// 用户把常用的训练任务配置保存为模板，下次提交时可以基于模板快速创建，
// 只需要修改少数参数，减少重复输入，提升效率。
package service

import (
	"context"
	"time"

	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// TaskTemplateService 提供任务模板生命周期管理能力。
//
// 能力：
//   - 创建模板：保存常用任务配置
//   - 更新模板：修改配置
//   - 删除模板：不再需要的模板删除
//   - 查询模板：按用户/项目列出模板
type TaskTemplateService struct {
	repo repository.TaskTemplateRepository
}

// NewTaskTemplateService 创建任务模板服务。
func NewTaskTemplateService(repo repository.TaskTemplateRepository) *TaskTemplateService {
	return &TaskTemplateService{repo: repo}
}

// CreateRequest 是创建模板请求。
type CreateRequest struct {
	Name           string                 `json:"name"`
	Description    string                 `json:"description,omitempty"`
	ProjectID      string                 `json:"project_id,omitempty"`
	Config         domain.ResourceRequest `json:"config"`
	Image          string                 `json:"image"`
	Command        string                 `json:"command,omitempty"`
	Env            []domain.EnvVar        `json:"env,omitempty"`
	DatasetPath    string                 `json:"dataset_path,omitempty"`
	Priority       int                    `json:"priority"`
	TimeoutSeconds int64                  `json:"timeout_seconds,omitempty"`
	Preemptible    bool                   `json:"preemptible"`
}

// Create 创建一个新的任务模板。
func (s *TaskTemplateService) Create(ctx context.Context, ownerID string, req CreateRequest) (*domain.TaskTemplate, error) {
	if req.Name == "" {
		return nil, apperr.InvalidArgument("模板名称不能为空")
	}
	if req.Image == "" {
		return nil, apperr.InvalidArgument("镜像名称不能为空")
	}
	now := time.Now()
	tpl := &domain.TaskTemplate{
		ID:             idgen.New("tpl"),
		Name:           req.Name,
		Description:    req.Description,
		OwnerID:        ownerID,
		ProjectID:      req.ProjectID,
		Config:         req.Config,
		Image:          req.Image,
		Command:        req.Command,
		Env:            req.Env,
		DatasetPath:    req.DatasetPath,
		Priority:       req.Priority,
		TimeoutSeconds: req.TimeoutSeconds,
		Preemptible:    req.Preemptible,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	if err := s.repo.Create(ctx, tpl); err != nil {
		return nil, err
	}
	return tpl, nil
}

// Get 获取模板详情。
func (s *TaskTemplateService) Get(ctx context.Context, id string) (*domain.TaskTemplate, error) {
	return s.repo.GetByID(ctx, id)
}

// UpdateRequest 是更新模板请求。
type UpdateRequest struct {
	Name           *string                 `json:"name"`
	Description    *string                 `json:"description"`
	Config         *domain.ResourceRequest `json:"config"`
	Image          *string                 `json:"image"`
	Command        *string                 `json:"command"`
	Env            *[]domain.EnvVar       `json:"env,omitempty"`
	DatasetPath    *string                 `json:"dataset_path"`
	Priority       *int                    `json:"priority"`
	TimeoutSeconds *int64                  `json:"timeout_seconds"`
	Preemptible    *bool                   `json:"preemptible"`
}

// Update 更新模板信息（部分更新）。
func (s *TaskTemplateService) Update(ctx context.Context, id string, ownerID string, req UpdateRequest) (*domain.TaskTemplate, error) {
	tpl, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return nil, err
	}
	// 权限检查：只有所有者可以修改
	if tpl.OwnerID != ownerID {
		return nil, apperr.Forbidden("无权限修改此模板")
	}
	// 按需更新字段
	if req.Name != nil {
		tpl.Name = *req.Name
	}
	if req.Description != nil {
		tpl.Description = *req.Description
	}
	if req.Config != nil {
		tpl.Config = *req.Config
	}
	if req.Image != nil {
		tpl.Image = *req.Image
	}
	if req.Command != nil {
		tpl.Command = *req.Command
	}
	if req.Env != nil {
		tpl.Env = *req.Env
	}
	if req.DatasetPath != nil {
		tpl.DatasetPath = *req.DatasetPath
	}
	if req.Priority != nil {
		tpl.Priority = *req.Priority
	}
	if req.TimeoutSeconds != nil {
		tpl.TimeoutSeconds = *req.TimeoutSeconds
	}
	if req.Preemptible != nil {
		tpl.Preemptible = *req.Preemptible
	}
	tpl.UpdatedAt = time.Now()
	if err := s.repo.Update(ctx, tpl); err != nil {
		return nil, err
	}
	return tpl, nil
}

// Delete 删除一个模板。
func (s *TaskTemplateService) Delete(ctx context.Context, id string, ownerID string) error {
	tpl, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return err
	}
	if tpl.OwnerID != ownerID {
		return apperr.Forbidden("无权限删除此模板")
	}
	return s.repo.Delete(ctx, id)
}

// ListByOwner 列出指定用户的所有模板。
func (s *TaskTemplateService) ListByOwner(ctx context.Context, ownerID string) ([]*domain.TaskTemplate, error) {
	return s.repo.ListByOwner(ctx, ownerID)
}

// ListByProject 列出指定项目的所有模板。
func (s *TaskTemplateService) ListByProject(ctx context.Context, projectID string) ([]*domain.TaskTemplate, error) {
	return s.repo.ListByProject(ctx, projectID)
}

// Instantiate 基于模板创建一个提交请求，用于快速提交。
//
// 用模板配置填充大部分字段，允许调用方覆盖部分字段。
func (s *TaskTemplateService) Instantiate(tpl *domain.TaskTemplate, overrides SubmitRequest) SubmitRequest {
	req := SubmitRequest{
		Name:           overrides.Name,
		Image:          tpl.Image,
		Command:        tpl.Command,
		Env:            tpl.Env,
		Priority:       tpl.Priority,
		GPUCount:       tpl.Config.GPUCount,
		MemoryMB:       tpl.Config.MemoryMB,
		GPUModel:       tpl.Config.GPUModel,
		Vendor:         tpl.Config.Vendor,
		ShareMode:      tpl.Config.ShareMode,
		TimeoutSeconds: tpl.TimeoutSeconds,
		Preemptible:    tpl.Preemptible,
		DatasetPath:    tpl.DatasetPath,
		ProjectID:      tpl.ProjectID,
		GangSize:       tpl.Config.GangSize,
		NodeSelector:   tpl.Config.NodeSelector,
	}
	// 允许调用方覆盖
	if overrides.Image != "" {
		req.Image = overrides.Image
	}
	if overrides.Command != "" {
		req.Command = overrides.Command
	}
	if len(overrides.Env) > 0 {
		req.Env = overrides.Env
	}
	if overrides.Priority != 0 || overrides.Priority != req.Priority {
		req.Priority = overrides.Priority
	}
	if overrides.GPUCount != 0 {
		req.GPUCount = overrides.GPUCount
	}
	if overrides.MemoryMB != 0 {
		req.MemoryMB = overrides.MemoryMB
	}
	if overrides.GPUModel != "" {
		req.GPUModel = overrides.GPUModel
	}
	if overrides.Vendor != "" {
		req.Vendor = overrides.Vendor
	}
	if overrides.ShareMode != "" {
		req.ShareMode = overrides.ShareMode
	}
	if overrides.TimeoutSeconds != 0 {
		req.TimeoutSeconds = overrides.TimeoutSeconds
	}
	// Preemptible 默认 false，所以总是覆盖
	req.Preemptible = overrides.Preemptible
	if overrides.DatasetPath != "" {
		req.DatasetPath = overrides.DatasetPath
	}
	if overrides.ProjectID != "" {
		req.ProjectID = overrides.ProjectID
	}
	if overrides.GangSize != 0 {
		req.GangSize = overrides.GangSize
	}
	if overrides.NodeSelector != nil {
		req.NodeSelector = overrides.NodeSelector
	}
	return req
}
