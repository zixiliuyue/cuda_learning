// 本文件实现配额服务：算力配额校验与用量计费，是商业化变现的基础。
//
// 配额服务在任务提交前校验主体（用户/项目）剩余卡时是否充足，
// 任务结束后累加已用卡时，从而支撑"按卡时计费 + 配额套餐"的收入模型。
package service

import (
	"context"

	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
)

// QuotaService 提供配额设置、校验与用量累加能力。
type QuotaService struct {
	repo repository.QuotaRepository
}

// NewQuotaService 创建配额服务。
func NewQuotaService(repo repository.QuotaRepository) *QuotaService {
	return &QuotaService{repo: repo}
}

// SetQuota 为主体设置配额（创建或覆盖）。
//
// 参数：
//   - ctx:           上下文。
//   - subject:       主体标识（用户 ID 或项目 ID）。
//   - maxGPUHours:   卡时上限；0 表示不限。
//   - maxConcurrent: 最大并发任务数；0 表示不限。
func (s *QuotaService) SetQuota(ctx context.Context, subject string, maxGPUHours float64, maxConcurrent int) (*domain.Quota, error) {
	if subject == "" {
		return nil, apperr.InvalidArgument("配额主体不能为空")
	}
	// 保留已用量：若已存在配额，仅更新上限。
	used := 0.0
	if existing, err := s.repo.Get(ctx, subject); err == nil {
		used = existing.UsedGPUHours
	}
	q := &domain.Quota{
		Subject:       subject,
		MaxGPUHours:   maxGPUHours,
		UsedGPUHours:  used,
		MaxConcurrent: maxConcurrent,
	}
	if err := s.repo.Upsert(ctx, q); err != nil {
		return nil, err
	}
	return q, nil
}

// Get 查询主体配额；不存在时返回一个"不限"的默认配额。
func (s *QuotaService) Get(ctx context.Context, subject string) (*domain.Quota, error) {
	q, err := s.repo.Get(ctx, subject)
	if err != nil {
		// 未设置配额视为不限。
		return &domain.Quota{Subject: subject}, nil
	}
	return q, nil
}

// CheckAdmit 校验主体在提交新任务时是否满足配额约束。
//
// 校验项：
//   - 剩余卡时是否足够本次预计消耗（estGPUHours）。
//   - 当前并发运行任务数是否已达上限（runningCount 由调用方统计传入）。
//
// 参数：
//   - ctx:          上下文。
//   - subject:      主体标识。
//   - estGPUHours:  本次任务预计消耗卡时。
//   - runningCount: 主体当前正在运行的任务数。
//
// 返回值：
//   - error: 超额时返回 CodeExhausted；未设置配额（不限）时始终通过。
func (s *QuotaService) CheckAdmit(ctx context.Context, subject string, estGPUHours float64, runningCount int) error {
	q, err := s.repo.Get(ctx, subject)
	if err != nil {
		return nil // 未设置配额，视为不限
	}
	if q.MaxGPUHours > 0 && q.RemainingGPUHours() < estGPUHours {
		return apperr.Exhausted("配额不足：剩余卡时不够本次任务")
	}
	if q.MaxConcurrent > 0 && runningCount >= q.MaxConcurrent {
		return apperr.Exhausted("并发任务数已达配额上限")
	}
	return nil
}

// Charge 在任务结束后按实际卡时对主体计费（累加已用量）。
//
// 参数：
//   - ctx:      上下文。
//   - subject:  主体标识。
//   - gpuHours: 实际消耗卡时。
func (s *QuotaService) Charge(ctx context.Context, subject string, gpuHours float64) error {
	if subject == "" || gpuHours <= 0 {
		return nil
	}
	return s.repo.AddUsage(ctx, subject, gpuHours)
}
