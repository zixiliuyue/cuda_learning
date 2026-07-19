// 本文件实现计量服务：算力用量采集与成本聚合，支撑成本可视化看板。
//
// 计量服务把"任务运行"翻译成"卡时与成本"，让"省钱"可量化：
//   - 任务结束时按运行时长 × GPU 份数记一条用量记录，并折算成本。
//   - 看板按用户/项目维度聚合总卡时、总成本与任务数。
package service

import (
	"context"
	"sort"
	"time"

	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/idgen"
)

// MeteringService 提供用量记录与成本聚合能力。
//
// 字段：
//   - repo:          用量记录仓储。
//   - pricePerGPUHour: 单位卡时价格（元/GPU·小时），用于成本折算。
type MeteringService struct {
	repo            repository.UsageRepository
	pricePerGPUHour float64
}

// NewMeteringService 创建计量服务。
//
// 参数：
//   - repo:  用量记录仓储。
//   - price: 单位卡时价格；<0 时归零。
func NewMeteringService(repo repository.UsageRepository, price float64) *MeteringService {
	if price < 0 {
		price = 0
	}
	return &MeteringService{repo: repo, pricePerGPUHour: price}
}

// RecordTaskUsage 依据任务的运行起止时间记录一条用量记录。
//
// 该方法在任务进入终态时调用。若任务未运行过（无 StartedAt/FinishedAt），
// 则视为零用量不记录。卡时 = 运行时长(小时) × GPU 份数。
//
// 参数：
//   - ctx:  上下文。
//   - task: 已结束的任务。
//
// 返回值：
//   - *domain.UsageRecord: 生成的用量记录；无有效运行区间时返回 nil。
//   - error:               持久化失败时返回。
func (s *MeteringService) RecordTaskUsage(ctx context.Context, task *domain.Task) (*domain.UsageRecord, error) {
	if task.StartedAt == nil || task.FinishedAt == nil {
		return nil, nil
	}
	dur := task.FinishedAt.Sub(*task.StartedAt)
	if dur <= 0 {
		return nil, nil
	}
	gpuCount := len(task.AllocatedGPUs)
	if gpuCount == 0 {
		gpuCount = task.Resource.GPUCount
	}
	gpuHours := dur.Hours() * float64(gpuCount)
	rec := &domain.UsageRecord{
		ID:         idgen.New("usage"),
		TaskID:     task.ID,
		Owner:      task.Owner,
		ProjectID:  task.ProjectID,
		GPUCount:   gpuCount,
		GPUHours:   gpuHours,
		Cost:       gpuHours * s.pricePerGPUHour,
		StartedAt:  *task.StartedAt,
		FinishedAt: *task.FinishedAt,
	}
	if err := s.repo.Create(ctx, rec); err != nil {
		return nil, err
	}
	return rec, nil
}

// CostSummary 表示某一维度（用户/项目）的成本汇总。
type CostSummary struct {
	Subject   string  `json:"subject"`    // 主体标识（用户 ID 或项目 ID）
	TaskCount int     `json:"task_count"` // 任务数
	GPUHours  float64 `json:"gpu_hours"`  // 总卡时
	Cost      float64 `json:"cost"`       // 总成本
}

// Dashboard 表示成本看板的聚合结果。
type Dashboard struct {
	TotalGPUHours float64        `json:"total_gpu_hours"` // 全局总卡时
	TotalCost     float64        `json:"total_cost"`      // 全局总成本
	TotalTasks    int            `json:"total_tasks"`     // 全局任务数
	ByOwner       []*CostSummary `json:"by_owner"`        // 按用户聚合
	GeneratedAt   time.Time      `json:"generated_at"`    // 生成时间
}

// BuildDashboard 聚合全部用量记录，生成成本看板。
//
// 参数：
//   - ctx: 上下文。
//
// 返回值：
//   - *Dashboard: 看板数据；按用户维度的汇总按成本降序排列。
//   - error:      读取用量记录失败时返回。
func (s *MeteringService) BuildDashboard(ctx context.Context) (*Dashboard, error) {
	records, err := s.repo.List(ctx)
	if err != nil {
		return nil, err
	}
	byOwner := make(map[string]*CostSummary)
	dash := &Dashboard{GeneratedAt: time.Now()}
	for _, r := range records {
		dash.TotalGPUHours += r.GPUHours
		dash.TotalCost += r.Cost
		dash.TotalTasks++
		sum, ok := byOwner[r.Owner]
		if !ok {
			sum = &CostSummary{Subject: r.Owner}
			byOwner[r.Owner] = sum
		}
		sum.TaskCount++
		sum.GPUHours += r.GPUHours
		sum.Cost += r.Cost
	}
	dash.ByOwner = make([]*CostSummary, 0, len(byOwner))
	for _, sum := range byOwner {
		dash.ByOwner = append(dash.ByOwner, sum)
	}
	sort.Slice(dash.ByOwner, func(i, j int) bool {
		return dash.ByOwner[i].Cost > dash.ByOwner[j].Cost
	})
	return dash, nil
}
