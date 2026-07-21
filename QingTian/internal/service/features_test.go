package service

import (
	"context"
	"testing"
	"time"

	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/logstream"
	"qingtian/internal/mq"
	"qingtian/internal/notify"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
	"qingtian/internal/storage"
)

// TestMeteringRecordAndDashboard 验证用量记录与成本看板聚合。
func TestMeteringRecordAndDashboard(t *testing.T) {
	repo := repository.NewMemoryUsageRepo()
	svc := NewMeteringService(repo, 2.0) // 单价 2 元/卡时
	ctx := context.Background()

	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	end := start.Add(2 * time.Hour)
	task := &domain.Task{
		ID: "t1", Owner: "u1", AllocatedGPUs: []string{"g0", "g1"},
		Resource: domain.ResourceRequest{GPUCount: 2},
		StartedAt: &start, FinishedAt: &end,
	}
	rec, err := svc.RecordTaskUsage(ctx, task)
	if err != nil || rec == nil {
		t.Fatalf("记录用量应成功: %v", err)
	}
	// 卡时 = 2 小时 × 2 卡 = 4；成本 = 4 × 2 = 8。
	if rec.GPUHours != 4 || rec.Cost != 8 {
		t.Errorf("卡时/成本计算错误: %+v", rec)
	}

	dash, _ := svc.BuildDashboard(ctx)
	if dash.TotalGPUHours != 4 || dash.TotalCost != 8 || dash.TotalTasks != 1 {
		t.Errorf("看板聚合错误: %+v", dash)
	}
	if len(dash.ByOwner) != 1 || dash.ByOwner[0].Subject != "u1" {
		t.Errorf("按用户聚合错误: %+v", dash.ByOwner)
	}
}

// TestMeteringSkipsUnrunTask 验证未运行任务不产生用量。
func TestMeteringSkipsUnrunTask(t *testing.T) {
	svc := NewMeteringService(repository.NewMemoryUsageRepo(), 1)
	rec, err := svc.RecordTaskUsage(context.Background(), &domain.Task{ID: "t1"})
	if err != nil || rec != nil {
		t.Errorf("未运行任务不应产生用量记录")
	}
}

// TestQuotaCheckAndCharge 验证配额校验与计费。
func TestQuotaCheckAndCharge(t *testing.T) {
	repo := repository.NewMemoryQuotaRepo()
	svc := NewQuotaService(repo)
	ctx := context.Background()

	_, _ = svc.SetQuota(ctx, "u1", 10, 2) // 上限 10 卡时、并发 2

	// 剩余充足应通过。
	if err := svc.CheckAdmit(ctx, "u1", 3, 0); err != nil {
		t.Errorf("剩余充足应通过: %v", err)
	}
	// 并发已达上限应拒绝。
	if err := svc.CheckAdmit(ctx, "u1", 1, 2); err == nil {
		t.Error("并发达上限应拒绝")
	}
	// 计费后剩余减少，超额应拒绝。
	_ = svc.Charge(ctx, "u1", 8)
	if err := svc.CheckAdmit(ctx, "u1", 5, 0); err == nil {
		t.Error("超出剩余卡时应拒绝")
	}
	// 未设置配额的主体视为不限。
	if err := svc.CheckAdmit(ctx, "unknown", 9999, 100); err != nil {
		t.Error("无配额主体应不受限")
	}
}

// TestProjectCreateAndMember 验证项目创建与成员管理。
func TestProjectCreateAndMember(t *testing.T) {
	svc := NewProjectService(repository.NewMemoryProjectRepo())
	ctx := context.Background()

	p, err := svc.Create(ctx, "team-a", "owner1")
	if err != nil {
		t.Fatalf("创建项目应成功: %v", err)
	}
	if !p.HasMember("owner1") {
		t.Error("创建者应为成员")
	}
	// 非所有者添加成员应被拒绝。
	if _, err := svc.AddMember(ctx, p.ID, "intruder", "u2"); err == nil {
		t.Error("非所有者添加成员应被拒绝")
	}
	// 所有者添加成员成功。
	p2, err := svc.AddMember(ctx, p.ID, "owner1", "u2")
	if err != nil || !p2.HasMember("u2") {
		t.Fatalf("所有者添加成员应成功: %v", err)
	}
	// 非成员查看应被拒绝。
	if _, err := svc.Get(ctx, p.ID, "outsider"); err == nil {
		t.Error("非成员查看应被拒绝")
	}
	mine, _ := svc.ListMine(ctx, "u2")
	if len(mine) != 1 {
		t.Errorf("u2 应属于 1 个项目，实际 %d", len(mine))
	}
}

// buildTaskSvcWithDeps 构造注入了全部可选依赖的 TaskService，供编排测试使用。
func buildTaskSvcWithDeps() (*TaskService, *notify.MemoryNotifier, *MeteringService, repository.TaskRepository) {
	taskRepo := repository.NewMemoryTaskRepo()
	nodeRepo := repository.NewMemoryNodeRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	sched := scheduler.New(ledger.New(), "binpack")
	notifier := notify.NewMemoryNotifier()
	metering := NewMeteringService(repository.NewMemoryUsageRepo(), 1)
	quota := NewQuotaService(repository.NewMemoryQuotaRepo())
	logs := NewLogService(logstream.NewMemoryStreamer(), storage.NewMemoryStore())

	svc := NewTaskService(taskRepo, nodeRepo, queue, bus, sched).
		WithMetering(metering).WithNotifier(notifier).WithQuota(quota).WithLogService(logs)
	return svc, notifier, metering, taskRepo
}

// TestTaskFinishTriggersSideEffects 验证任务终态触发计量与通知。
func TestTaskFinishTriggersSideEffects(t *testing.T) {
	svc, notifier, metering, taskRepo := buildTaskSvcWithDeps()
	ctx := context.Background()

	// 造一个 running 中的任务。
	start := time.Now().Add(-time.Hour)
	task := &domain.Task{
		ID: "t1", Name: "job", Owner: "u1", Status: domain.TaskRunning,
		AllocatedGPUs: []string{"g0"}, LogPath: "logs/job.log",
		StartedAt: &start,
	}
	_ = taskRepo.Create(ctx, task)

	// 回传 succeeded。
	status := mq.StatusMessage{TaskID: "t1", Status: string(domain.TaskSucceeded), ExitCode: 0}
	if err := svc.HandleStatus(ctx, jsonMarshal(status)); err != nil {
		t.Fatalf("处理状态应成功: %v", err)
	}

	// 应产生通知。
	if len(notifier.Sent) != 1 {
		t.Errorf("成功应触发 1 条通知，实际 %d", len(notifier.Sent))
	}
	// 应产生用量记录。
	dash, _ := metering.BuildDashboard(ctx)
	if dash.TotalTasks != 1 {
		t.Errorf("应产生 1 条用量记录，实际 %d", dash.TotalTasks)
	}
}

// TestTaskFailureDiagnosis 验证失败任务被结构化诊断。
func TestTaskFailureDiagnosis(t *testing.T) {
	svc, _, _, taskRepo := buildTaskSvcWithDeps()
	ctx := context.Background()
	start := time.Now().Add(-time.Minute)
	task := &domain.Task{ID: "t1", Name: "job", Owner: "u1", Status: domain.TaskRunning, StartedAt: &start}
	_ = taskRepo.Create(ctx, task)

	status := mq.StatusMessage{
		TaskID: "t1", Status: string(domain.TaskFailed), ExitCode: 1,
		Output: "RuntimeError: CUDA out of memory",
	}
	_ = svc.HandleStatus(ctx, jsonMarshal(status))

	got, _ := svc.Get(ctx, "t1")
	if got.FailureCategory != domain.FailureOOM {
		t.Errorf("失败应被诊断为 OOM，实际 %s", got.FailureCategory)
	}
}

// TestResumeFromCheckpoint 验证失败任务可断点续跑。
func TestResumeFromCheckpoint(t *testing.T) {
	svc, _, _, taskRepo := buildTaskSvcWithDeps()
	ctx := context.Background()
	task := &domain.Task{
		ID: "t1", Name: "job", Owner: "u1", Status: domain.TaskFailed,
		CheckpointPath: "ckpt/t1/latest", Priority: 3,
	}
	_ = taskRepo.Create(ctx, task)

	if err := svc.Resume(ctx, "t1"); err != nil {
		t.Fatalf("断点续跑应成功: %v", err)
	}
	got, _ := svc.Get(ctx, "t1")
	if got.Status != domain.TaskPending {
		t.Errorf("恢复后应回到 pending，实际 %s", got.Status)
	}
	if got.CheckpointPath == "" {
		t.Error("恢复应保留 CheckpointPath")
	}

	// 无 Checkpoint 的失败任务不可恢复。
	task2 := &domain.Task{ID: "t2", Name: "j2", Owner: "u1", Status: domain.TaskFailed}
	_ = taskRepo.Create(ctx, task2)
	if err := svc.Resume(ctx, "t2"); err == nil {
		t.Error("无 Checkpoint 应无法断点续跑")
	}
}

// TestQuotaBlocksSubmit 验证配额不足时拒绝提交。
func TestQuotaBlocksSubmit(t *testing.T) {
	taskRepo := repository.NewMemoryTaskRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	sched := scheduler.New(ledger.New(), "binpack")
	quotaRepo := repository.NewMemoryQuotaRepo()
	quota := NewQuotaService(quotaRepo)
	ctx := context.Background()
	_, _ = quota.SetQuota(ctx, "u1", 1, 0) // 仅 1 卡时

	svc := NewTaskService(taskRepo, repository.NewMemoryNodeRepo(), queue, bus, sched).WithQuota(quota)
	// 已用满配额。
	_ = quota.Charge(ctx, "u1", 1)

	_, err := svc.Submit(ctx, "u1", SubmitRequest{
		Name: "job", Image: "img", GPUCount: 1, MemoryMB: 1024,
	})
	if err == nil {
		t.Error("配额耗尽时提交应被拒绝")
	}
}

// TestScheduleBatch 验证批量调度处理队列中所有任务。
func TestScheduleBatch(t *testing.T) {
	nodeRepo := repository.NewMemoryNodeRepo()
	taskRepo := repository.NewMemoryTaskRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	resLedger := ledger.New()
	sched := scheduler.New(resLedger, "binpack")
	memCache := cache.NewMemoryCache()
	ctx := context.Background()

	nodeSvc := NewNodeService(nodeRepo, memCache, resLedger, 30*time.Second)
	_, _ = nodeSvc.Register(ctx, RegisterRequest{
		Hostname: "gpu-01",
		GPUs: []domain.GPU{
			{UUID: "g0", Model: "A100", MemoryTotal: 40960, Allocatable: true},
		},
	})
	taskSvc := NewTaskService(taskRepo, nodeRepo, queue, bus, sched).WithNodeService(nodeSvc)

	// 提交 3 个显存共享任务。
	for i := 0; i < 3; i++ {
		_, _ = taskSvc.Submit(ctx, "u1", SubmitRequest{
			Name: "job", Image: "img", GPUCount: 1, MemoryMB: 8000, ShareMode: domain.ShareMemory,
		})
	}
	// 处理队列中所有任务，现在 ScheduleBatch 不返回处理数量
	taskSvc.ScheduleBatch(ctx)

	// 检查所有任务都已被调度
	pending, err := taskRepo.List(ctx, domain.TaskPending)
	if err != nil {
		t.Fatalf("列表查询失败: %v", err)
	}
	if len(pending) != 0 {
		t.Errorf("应处理完所有 3 个任务，剩余 %d 个仍在 pending", len(pending))
	}
}

// TestDetectIdleNodes 验证空闲节点检测。
func TestDetectIdleNodes(t *testing.T) {
	nodeRepo := repository.NewMemoryNodeRepo()
	memCache := cache.NewMemoryCache()
	resLedger := ledger.New()
	ctx := context.Background()
	svc := NewNodeService(nodeRepo, memCache, resLedger, 30*time.Second)

	_, _ = svc.Register(ctx, RegisterRequest{
		Hostname: "idle-node",
		GPUs:     []domain.GPU{{UUID: "g0", Model: "A100", MemoryTotal: 40960, Allocatable: true, Utilization: 0}},
	})
	idle, err := svc.DetectIdleNodes(ctx, 10)
	if err != nil {
		t.Fatalf("检测空闲节点应成功: %v", err)
	}
	if len(idle) != 1 {
		t.Fatalf("应检测到 1 个空闲节点，实际 %d", len(idle))
	}

	// 占用后不再空闲。
	_, _ = resLedger.TryAllocate("node-idle-node", domain.ResourceRequest{GPUCount: 1, MemoryMB: 1024})
	idle2, _ := svc.DetectIdleNodes(ctx, 10)
	if len(idle2) != 0 {
		t.Errorf("占用后不应判为空闲，实际 %d", len(idle2))
	}
}
