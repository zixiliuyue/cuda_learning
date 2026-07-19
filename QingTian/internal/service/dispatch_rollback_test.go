package service

import (
	"context"
	"errors"
	"testing"

	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
)

// failingProducer 是一个 Publish 永远失败的生产者，用于验证下发失败补偿。
type failingProducer struct{}

func (failingProducer) Publish(_ context.Context, _ string, _ []byte) error {
	return errors.New("模拟 NSQ 下发失败")
}
func (failingProducer) Close() error { return nil }

// TestScheduleOnceRollsBackOnDispatchFailure 验证下发失败时回滚账本、回退状态并重新入队。
//
// 若下发失败后不回滚，会出现"任务已扣减资源且状态为 scheduled，但 Agent 收不到"
// 的悬挂任务与 GPU 泄漏。补偿逻辑应把任务退回 pending、释放账本并重新入队。
func TestScheduleOnceRollsBackOnDispatchFailure(t *testing.T) {
	taskRepo := repository.NewMemoryTaskRepo()
	nodeRepo := repository.NewMemoryNodeRepo()
	queue := cache.NewMemoryPriorityQueue()
	led := ledger.New()
	sched := scheduler.New(led, "binpack")
	svc := NewTaskService(taskRepo, nodeRepo, queue, failingProducer{}, sched)
	ctx := context.Background()

	node := &domain.Node{
		ID: "node-1", Status: domain.NodeOnline,
		GPUs: []domain.GPU{{UUID: "gpu-0", MemoryTotal: 8192, Allocatable: true}},
	}
	_ = nodeRepo.Upsert(ctx, node)
	led.SyncNode(node)

	task := &domain.Task{
		ID: "t1", Name: "job", Owner: "u1", Status: domain.TaskPending,
		Priority: 5, Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}
	_ = taskRepo.Create(ctx, task)
	_ = queue.Push(ctx, "t1", 5)

	if _, err := svc.ScheduleOnce(ctx); err != nil {
		t.Fatalf("下发失败应被补偿而非返回错误: %v", err)
	}

	// 任务应回退为 pending。
	got, _ := svc.Get(ctx, "t1")
	if got.Status != domain.TaskPending {
		t.Errorf("下发失败后任务应回退为 pending，实际 %s", got.Status)
	}
	if got.NodeID != "" || len(got.AllocatedGPUs) != 0 {
		t.Errorf("回退应清空调度信息，实际 node=%q gpus=%v", got.NodeID, got.AllocatedGPUs)
	}
	// 账本应已释放：整卡重新可用。
	if free := led.NodeFreeMemory("node-1"); free != 8192 {
		t.Errorf("下发失败应释放账本，剩余期望 8192，实际 %d", free)
	}
	// 任务应重新入队等待重试。
	if n, _ := queue.Len(ctx); n != 1 {
		t.Errorf("下发失败任务应重新入队，队列长度期望 1，实际 %d", n)
	}
}
