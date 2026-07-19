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

// TestPreemptionEvictsMultipleVictimsUntilFits 验证需要多份资源时，抢占会持续
// 终止低优先级任务直到目标节点能容纳本任务，而非"抢一个就停"。
//
// 场景：单卡 8G 节点被两个各占半卡（4G）的低优先级可抢占任务占满；高优先级
// 任务需要整卡独占（8G），必须抢占两个 victim 才能腾出足够资源。
func TestPreemptionEvictsMultipleVictimsUntilFits(t *testing.T) {
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

	// 两个各占 4G 的低优先级、可抢占的显存共享任务，占满整卡。
	for _, id := range []string{"low-1", "low-2"} {
		alloc, err := led.TryAllocate("node-1", domain.ResourceRequest{
			GPUCount: 1, MemoryMB: 4096, ShareMode: domain.ShareMemory,
		})
		if err != nil {
			t.Fatalf("低优先级任务 %s 应能占用半卡: %v", id, err)
		}
		low := &domain.Task{
			ID: id, Name: id, Owner: "u1", Status: domain.TaskRunning,
			Priority: 1, Preemptible: true, NodeID: "node-1",
			AllocatedGPUs: alloc.AllocatedIDs(),
			Resource:      domain.ResourceRequest{GPUCount: 1, MemoryMB: 4096, ShareMode: domain.ShareMemory},
		}
		_ = taskRepo.Create(ctx, low)
	}

	// 高优先级任务需要整卡独占（8G），需抢占两个 victim。
	high := &domain.Task{
		ID: "high", Name: "high-job", Owner: "u1", Status: domain.TaskPending,
		Priority: 10, Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192, ShareMode: domain.ShareExclusive},
	}
	_ = taskRepo.Create(ctx, high)
	_ = queue.Push(ctx, "high", 10)

	processed, err := svc.ScheduleOnce(ctx)
	if err != nil || !processed {
		t.Fatalf("调度应处理高优先级任务: processed=%v err=%v", processed, err)
	}

	// 两个低优先级任务都应被抢占。
	for _, id := range []string{"low-1", "low-2"} {
		got, _ := svc.Get(ctx, id)
		if got.Status != domain.TaskFailed || got.FailureCategory != domain.FailurePreempted {
			t.Errorf("任务 %s 应被抢占（failed/preempted），实际 %s/%s", id, got.Status, got.FailureCategory)
		}
	}
	gotHigh, _ := svc.Get(ctx, "high")
	if gotHigh.Status != domain.TaskScheduled {
		t.Errorf("腾出整卡后高优先级任务应完成调度（scheduled），实际 %s", gotHigh.Status)
	}
	if gotHigh.NodeID != "node-1" {
		t.Errorf("高优先级任务应被调度到 node-1，实际 %s", gotHigh.NodeID)
	}
}

// TestPreemptionStopsWhenEnoughFreed 验证一旦腾出足够资源即停止抢占，不误伤多余任务。
//
// 场景：单卡 12G 节点上有两个各占 4G 的低优先级可抢占任务（共占 8G，剩 4G）；
// 高优先级任务只需 4G 显存共享，抢占一个即够，另一个应被保留。
func TestPreemptionStopsWhenEnoughFreed(t *testing.T) {
	taskRepo := repository.NewMemoryTaskRepo()
	nodeRepo := repository.NewMemoryNodeRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	led := ledger.New()
	sched := scheduler.New(led, "binpack")
	svc := NewTaskService(taskRepo, nodeRepo, queue, bus, sched)
	ctx := context.Background()

	// 12G 单卡：两个 4G 任务占用 8G 后剩 4G；再要 4G 独占会失败（非空闲），触发抢占。
	node := &domain.Node{
		ID: "node-1", Status: domain.NodeOnline,
		GPUs: []domain.GPU{{UUID: "gpu-0", MemoryTotal: 12288, Allocatable: true}},
	}
	_ = nodeRepo.Upsert(ctx, node)
	led.SyncNode(node)

	for i, id := range []string{"low-1", "low-2"} {
		alloc, err := led.TryAllocate("node-1", domain.ResourceRequest{
			GPUCount: 1, MemoryMB: 4096, ShareMode: domain.ShareMemory,
		})
		if err != nil {
			t.Fatalf("低优先级任务 %s 应能分配: %v", id, err)
		}
		low := &domain.Task{
			ID: id, Name: id, Owner: "u1", Status: domain.TaskRunning,
			Priority: i + 1, Preemptible: true, NodeID: "node-1",
			AllocatedGPUs: alloc.AllocatedIDs(),
			Resource:      domain.ResourceRequest{GPUCount: 1, MemoryMB: 4096, ShareMode: domain.ShareMemory},
		}
		_ = taskRepo.Create(ctx, low)
	}

	// 高优先级任务只需 4G 显存共享，节点剩余 4G 恰好够用，不应触发抢占。
	high := &domain.Task{
		ID: "high", Name: "high-job", Owner: "u1", Status: domain.TaskPending,
		Priority: 10, Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 4096, ShareMode: domain.ShareMemory},
	}
	_ = taskRepo.Create(ctx, high)
	_ = queue.Push(ctx, "high", 10)

	processed, err := svc.ScheduleOnce(ctx)
	if err != nil || !processed {
		t.Fatalf("调度应处理高优先级任务: processed=%v err=%v", processed, err)
	}

	// 高优任务只需 4G 显存共享，节点剩余 4G 本就够用——不应发生抢占。
	survived := 0
	for _, id := range []string{"low-1", "low-2"} {
		got, _ := svc.Get(ctx, id)
		if got.Status == domain.TaskRunning {
			survived++
		}
	}
	if survived != 2 {
		t.Errorf("资源本就充足时不应抢占任何任务，存活期望 2，实际 %d", survived)
	}
	gotHigh, _ := svc.Get(ctx, "high")
	if gotHigh.Status != domain.TaskScheduled {
		t.Errorf("高优先级任务应完成调度，实际 %s", gotHigh.Status)
	}
}
