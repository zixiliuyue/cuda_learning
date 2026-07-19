package service

import (
	"context"
	"testing"

	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/mq"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
)

// TestSharedLedgerPersistsAcrossRestart 验证账本快照经共享存储在实例重启后仍生效。
//
// 场景：实例 A 调度一个任务并把账本写回共享存储；模拟实例 A 崩溃（全新 ledger/
// scheduler/service，仅共享同一个 cache）。新实例 SyncLedger 应从共享快照恢复占用，
// 使得同一块卡不会被重复分配（防止跨实例/重启超卖）。
func TestSharedLedgerPersistsAcrossRestart(t *testing.T) {
	ctx := context.Background()
	shared := cache.NewMemoryCache()
	taskRepo := repository.NewMemoryTaskRepo()
	nodeRepo := repository.NewMemoryNodeRepo()

	node := &domain.Node{
		ID: "node-1", Status: domain.NodeOnline,
		GPUs: []domain.GPU{{UUID: "gpu-0", MemoryTotal: 8192, Allocatable: true}},
	}
	_ = nodeRepo.Upsert(ctx, node)

	// —— 实例 A —— 调度一个占满整卡的任务，并写回共享账本。
	queueA := cache.NewMemoryPriorityQueue()
	busA := mq.NewMemoryBus()
	ledA := ledger.New()
	schedA := scheduler.New(ledA, "binpack")
	svcA := NewTaskService(taskRepo, nodeRepo, queueA, busA, schedA).WithLedgerStore(shared)

	taskA := &domain.Task{
		ID: "a", Name: "job-a", Owner: "u1", Status: domain.TaskPending,
		Priority: 5, Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}
	_ = taskRepo.Create(ctx, taskA)
	_ = queueA.Push(ctx, "a", 5)

	if err := svcA.SyncLedger(ctx); err != nil {
		t.Fatalf("实例A 准备账本应成功: %v", err)
	}
	if _, err := svcA.ScheduleOnce(ctx); err != nil {
		t.Fatalf("实例A 调度应成功: %v", err)
	}
	if err := svcA.PersistLedger(ctx); err != nil {
		t.Fatalf("实例A 写回账本应成功: %v", err)
	}
	gotA, _ := svcA.Get(ctx, "a")
	if gotA.Status != domain.TaskScheduled {
		t.Fatalf("任务 a 应被调度，实际 %s", gotA.Status)
	}

	// —— 实例 B（模拟重启）—— 全新账本，仅共享同一个 cache。
	queueB := cache.NewMemoryPriorityQueue()
	busB := mq.NewMemoryBus()
	ledB := ledger.New()
	schedB := scheduler.New(ledB, "binpack")
	svcB := NewTaskService(taskRepo, nodeRepo, queueB, busB, schedB).WithLedgerStore(shared)

	taskB := &domain.Task{
		ID: "b", Name: "job-b", Owner: "u1", Status: domain.TaskPending,
		Priority: 5, Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}
	_ = taskRepo.Create(ctx, taskB)
	_ = queueB.Push(ctx, "b", 5)

	// 新实例从共享快照恢复账本：整卡已被任务 a 占用。
	if err := svcB.SyncLedger(ctx); err != nil {
		t.Fatalf("实例B 准备账本应成功: %v", err)
	}
	if _, err := svcB.ScheduleOnce(ctx); err != nil {
		t.Fatalf("实例B 调度处理应成功: %v", err)
	}

	// 任务 b 应因整卡已被占用而无法调度（防止超卖），保持 pending。
	gotB, _ := svcB.Get(ctx, "b")
	if gotB.Status != domain.TaskPending {
		t.Errorf("整卡已被占用，任务 b 应保持 pending（防超卖），实际 %s", gotB.Status)
	}
}
