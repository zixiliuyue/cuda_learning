// Package service 定时调度服务
//
// 负责定时计划的管理、Cron 解析、下次执行时间计算、并发控制，按计划触发工作流执行。
package service

import (
	"context"
	"sort"
	"time"

	"github.com/robfig/cron/v3"
	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// ScheduleService 定时调度服务
type ScheduleService struct {
	cron     *cron.Cron
	repo     repository.ScheduleRepository
	workflowEngine *WorkflowEngine
}

// NewScheduleService 创建定时调度服务
func NewScheduleService(
	repo repository.ScheduleRepository,
	workflowEngine *WorkflowEngine,
) *ScheduleService {
	// 使用 second 精度
	c := cron.New(cron.WithSeconds())
	s := &ScheduleService{
		cron:     c,
		repo:     repo,
		workflowEngine: workflowEngine,
	}

	// 启动调度器协程
	c.Start()

	// 加载所有激活的计划开始调度
	go s.start(context.Background())
	return s
}

// GetByID 获取定时计划详情
func (s *ScheduleService) GetByID(ctx context.Context, id string) (*domain.Schedule, error) {
	return s.repo.GetByID(ctx, id)
}

// List 列出所有定时计划
func (s *ScheduleService) List(ctx context.Context, skip, limit int) ([]*domain.Schedule, int64, error) {
	return s.repo.List(ctx, skip, limit)
}

// Create 创建定时计划
func (s *ScheduleService) Create(ctx context.Context, sch *domain.Schedule) error {
	if sch.CronExpr == "" {
		return apperr.InvalidArgument("cron expression is required")
	}
	// 校验 Cron 表达式语法（6 字段：秒 分 时 日 月 周）
	parser := cron.NewParser(cron.Second | cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow)
	_, err := parser.Parse(sch.CronExpr)
	if err != nil {
		return apperr.Wrap("InvalidArgument", "invalid cron expression", err)
	}

	// 计算下次执行时间
	now := time.Now()
	if sch.ID == "" {
		sch.ID = idgen.New("sch")
	}
	sch.CreatedAt = now
	sch.UpdatedAt = now
	sch.Status = domain.ScheduleStatusActive
	sch.Enabled = true

	if err := s.repo.Create(ctx, sch); err != nil {
		return err
	}

	// 添加到 cron 调度器
	s.addToCron(ctx, sch)

	return nil
}

// Update 更新定时计划
func (s *ScheduleService) Update(ctx context.Context, sch *domain.Schedule) error {
	if sch.CronExpr != "" {
		parser := cron.NewParser(cron.Second | cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow)
		_, err := parser.Parse(sch.CronExpr)
		if err != nil {
			return apperr.Wrap("InvalidArgument", "invalid cron expression", err)
		}
	}
	sch.UpdatedAt = time.Now()
	if err := s.repo.Update(ctx, sch); err != nil {
		return err
	}

	// 重新添加到 cron 调度器
	s.addToCron(ctx, sch)

	return nil
}

// Delete 删除定时计划
func (s *ScheduleService) Delete(ctx context.Context, id string) error {
	return s.repo.Delete(ctx, id)
}

// addToCron 将计划添加到 cron 调度器
func (s *ScheduleService) addToCron(ctx context.Context, sch *domain.Schedule) {
	// 解析表达式（6 字段：秒 分 时 日 月 周）
	parser := cron.NewParser(cron.Second | cron.Minute | cron.Hour | cron.Dom | cron.Month | cron.Dow)
	spec, _ := parser.Parse(sch.CronExpr)

	// 根据时区计算下次执行时间
	// 如果没有指定时区，使用系统时区
	var next time.Time
	tz := sch.Timezone
	if tz == "" {
		next = spec.Next(time.Now())
	} else {
		loc, err := time.LoadLocation(tz)
		if err != nil {
			next = spec.Next(time.Now())
		} else {
			next = spec.Next(time.Now().In(loc))
		}
	}

	sch.NextRunAt = &next
	_ = s.repo.UpdateNextRunTime(ctx, sch.ID, &next)

	// 触发函数：工作流执行
	cronEntry := func() {
		// 创建触发信息
		trigger := domain.TriggerInfo{
			Type:        "schedule",
			ScheduleID:  sch.ID,
			TriggeredBy: "system",
			TriggeredAt: time.Now(),
		}

		// 执行工作流：参数使用计划默认参数
		_, _ = s.workflowEngine.Run(context.Background(), sch.WorkflowID, sch.Parameters, trigger)
	}

	// AddJob 返回 EntryID，我们不需要保存它因为数据库已经有 sch.ID
	_, _ = s.cron.AddJob(sch.CronExpr, cron.FuncJob(cronEntry))
}

// start 启动时加载所有激活的计划开始调度
func (s *ScheduleService) start(ctx context.Context) {
	// 找出所有需要执行的计划
	list, err := s.repo.ListToRun(ctx, time.Now(), 100)
	if err != nil {
		return
	}

	// 按下次执行时间排序，先执行最早的
	sort.Slice(list, func(i, j int) bool {
		if list[i].NextRunAt == nil {
			return false
		}
		if list[j].NextRunAt == nil {
			return false
		}
		return list[i].NextRunAt.Before(*list[j].NextRunAt)
	})

	// 逐个添加到 cron
	for _, sch := range list {
		if sch.Enabled {
			s.addToCron(ctx, sch)
		}
	}
}

// TriggerRun 手动触发一次执行
func (s *ScheduleService) TriggerRun(ctx context.Context, id string, params map[string]interface{}) error {
	sch, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return err
	}

	trigger := domain.TriggerInfo{
		Type:        "manual",
		ScheduleID:  id,
		TriggeredBy: "user",
		TriggeredAt: time.Now(),
	}

	_, err = s.workflowEngine.Run(ctx, sch.WorkflowID, params, trigger)
	return err
}

// ListToRun 获取需要在这个时间窗口内执行的计划（给调度器循环扫描用）
func (s *ScheduleService) ListToRun(ctx context.Context, now time.Time, limit int) ([]*domain.Schedule, error) {
	return s.repo.ListToRun(ctx, now, limit)
}
