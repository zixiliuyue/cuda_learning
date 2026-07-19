package service

import (
	"context"
	"testing"
	"time"

	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/mq"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
)

// TestEnforceTimeoutsKillsOverrunTask 验证超时防护会终止运行超时的任务并置为失败。
func TestEnforceTimeoutsKillsOverrunTask(t *testing.T) {
	svc, _, _, taskRepo := buildTaskSvcWithDeps()
	ctx := context.Background()

	// 造一个已运行 2 小时、超时上限仅 60 秒的运行中任务。
	start := time.Now().Add(-2 * time.Hour)
	task := &domain.Task{
		ID: "t-timeout", Name: "long-job", Owner: "u1", Status: domain.TaskRunning,
		AllocatedGPUs: []string{"g0"}, TimeoutSeconds: 60, StartedAt: &start,
	}
	_ = taskRepo.Create(ctx, task)

	killed, err := svc.EnforceTimeouts(ctx)
	if err != nil {
		t.Fatalf("超时巡检应成功: %v", err)
	}
	if killed != 1 {
		t.Fatalf("应终止 1 个超时任务，实际 %d", killed)
	}
	got, _ := svc.Get(ctx, "t-timeout")
	if got.Status != domain.TaskFailed {
		t.Errorf("超时任务应被置为 failed，实际 %s", got.Status)
	}
	if got.FailureCategory != domain.FailureTimeout {
		t.Errorf("失败归类应为 timeout，实际 %s", got.FailureCategory)
	}
}

// TestEnforceTimeoutsSkipsWithinLimit 验证未超时任务不会被误杀。
func TestEnforceTimeoutsSkipsWithinLimit(t *testing.T) {
	svc, _, _, taskRepo := buildTaskSvcWithDeps()
	ctx := context.Background()

	start := time.Now().Add(-30 * time.Second)
	task := &domain.Task{
		ID: "t-ok", Name: "job", Owner: "u1", Status: domain.TaskRunning,
		TimeoutSeconds: 3600, StartedAt: &start,
	}
	_ = taskRepo.Create(ctx, task)

	killed, _ := svc.EnforceTimeouts(ctx)
	if killed != 0 {
		t.Fatalf("未超时任务不应被终止，实际终止 %d", killed)
	}
	got, _ := svc.Get(ctx, "t-ok")
	if got.Status != domain.TaskRunning {
		t.Errorf("任务状态应保持 running，实际 %s", got.Status)
	}
}

// TestEnforceTimeoutsIgnoresNoTimeout 验证未设置超时的任务永不被终止。
func TestEnforceTimeoutsIgnoresNoTimeout(t *testing.T) {
	svc, _, _, taskRepo := buildTaskSvcWithDeps()
	ctx := context.Background()

	start := time.Now().Add(-100 * time.Hour)
	task := &domain.Task{
		ID: "t-forever", Name: "job", Owner: "u1", Status: domain.TaskRunning,
		TimeoutSeconds: 0, StartedAt: &start,
	}
	_ = taskRepo.Create(ctx, task)

	killed, _ := svc.EnforceTimeouts(ctx)
	if killed != 0 {
		t.Errorf("未设置超时的任务不应被终止，实际终止 %d", killed)
	}
}

// TestPreemptionSchedulesHighPriorityTask 验证无空闲资源时高优先级任务可抢占低优先级可抢占任务。
func TestPreemptionSchedulesHighPriorityTask(t *testing.T) {
	taskRepo := repository.NewMemoryTaskRepo()
	nodeRepo := repository.NewMemoryNodeRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	led := ledger.New()
	sched := scheduler.New(led, "binpack")
	svc := NewTaskService(taskRepo, nodeRepo, queue, bus, sched)
	ctx := context.Background()

	// 单卡节点：显存 8G，仅够一个任务。
	node := &domain.Node{
		ID: "node-1", Status: domain.NodeOnline,
		GPUs: []domain.GPU{{UUID: "gpu-0", MemoryTotal: 8192, Allocatable: true}},
	}
	_ = nodeRepo.Upsert(ctx, node)
	led.SyncNode(node)

	// 已有一个低优先级、可抢占的运行中任务，占满整卡。
	lowAlloc, err := led.TryAllocate("node-1", domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192})
	if err != nil {
		t.Fatalf("低优先级任务应能占满整卡: %v", err)
	}
	low := &domain.Task{
		ID: "low", Name: "low-job", Owner: "u1", Status: domain.TaskRunning,
		Priority: 1, Preemptible: true, NodeID: "node-1",
		AllocatedGPUs: lowAlloc.AllocatedIDs(),
		Resource:      domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}
	_ = taskRepo.Create(ctx, low)

	// 提交高优先级任务并入队。
	high := &domain.Task{
		ID: "high", Name: "high-job", Owner: "u1", Status: domain.TaskPending,
		Priority: 10, Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}
	_ = taskRepo.Create(ctx, high)
	_ = queue.Push(ctx, "high", 10)

	// 触发调度：应抢占 low，随后成功调度 high。
	processed, err := svc.ScheduleOnce(ctx)
	if err != nil || !processed {
		t.Fatalf("调度应处理高优先级任务: processed=%v err=%v", processed, err)
	}

	gotLow, _ := svc.Get(ctx, "low")
	if gotLow.Status != domain.TaskFailed || gotLow.FailureCategory != domain.FailurePreempted {
		t.Errorf("低优先级任务应被抢占（failed/preempted），实际 %s/%s", gotLow.Status, gotLow.FailureCategory)
	}
	gotHigh, _ := svc.Get(ctx, "high")
	if gotHigh.Status != domain.TaskScheduled {
		t.Errorf("高优先级任务应完成调度（scheduled），实际 %s", gotHigh.Status)
	}
}

// TestPreemptionSkipsNonPreemptible 验证不可抢占任务不会被抢占。
func TestPreemptionSkipsNonPreemptible(t *testing.T) {
	taskRepo := repository.NewMemoryTaskRepo()
	nodeRepo := repository.NewMemoryNodeRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	led := ledger.New()
	sched := scheduler.New(led, "binpack")
	svc := NewTaskService(taskRepo, nodeRepo, queue, bus, sched)
	ctx := context.Background()

	node := &domain.Node{
		ID: "node-1", Status: domain.NodeOnline,
		GPUs: []domain.GPU{{UUID: "gpu-0", MemoryTotal: 8192, Allocatable: true}},
	}
	_ = nodeRepo.Upsert(ctx, node)
	led.SyncNode(node)

	alloc, _ := led.TryAllocate("node-1", domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192})
	low := &domain.Task{
		ID: "low", Name: "low-job", Owner: "u1", Status: domain.TaskRunning,
		Priority: 1, Preemptible: false, NodeID: "node-1",
		AllocatedGPUs: alloc.AllocatedIDs(),
		Resource:      domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}
	_ = taskRepo.Create(ctx, low)

	high := &domain.Task{
		ID: "high", Name: "high-job", Owner: "u1", Status: domain.TaskPending,
		Priority: 10, Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}
	_ = taskRepo.Create(ctx, high)
	_ = queue.Push(ctx, "high", 10)

	_, _ = svc.ScheduleOnce(ctx)

	gotLow, _ := svc.Get(ctx, "low")
	if gotLow.Status != domain.TaskRunning {
		t.Errorf("不可抢占任务应保持 running，实际 %s", gotLow.Status)
	}
	gotHigh, _ := svc.Get(ctx, "high")
	if gotHigh.Status != domain.TaskPending {
		t.Errorf("无资源且不可抢占时高优先级任务应仍为 pending，实际 %s", gotHigh.Status)
	}
}
