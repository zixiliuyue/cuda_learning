package scheduler

import (
	"context"
	"testing"

	"qingtian/internal/domain"
	"qingtian/internal/ledger"
)

// buildMultiGPUNode 构造含指定 GPU 数量、支持显存共享的在线节点。
func buildMultiGPUNode(id string, gpuCount int, memPerGPU int64) *domain.Node {
	gpus := make([]domain.GPU, gpuCount)
	for i := 0; i < gpuCount; i++ {
		gpus[i] = domain.GPU{
			UUID: id + "-gpu" + string(rune('0'+i)), Model: "A100",
			Vendor: domain.VendorNVIDIA, MemoryTotal: memPerGPU, Allocatable: true,
		}
	}
	return &domain.Node{ID: id, Status: domain.NodeOnline, GPUs: gpus}
}

// TestScheduleGangSuccess 验证成组调度能一次性拿到全部份数。
func TestScheduleGangSuccess(t *testing.T) {
	l := ledger.New()
	node := buildMultiGPUNode("node1", 1, 40960) // 大显存单卡，支持显存共享
	l.SyncNode(node)
	s := New(l, "binpack")

	task := &domain.Task{Resource: domain.ResourceRequest{
		GPUCount: 1, MemoryMB: 8000, ShareMode: domain.ShareMemory, GangSize: 3,
	}}
	decisions, err := s.ScheduleGang(context.Background(), task, []*domain.Node{node})
	if err != nil {
		t.Fatalf("成组调度应成功: %v", err)
	}
	if len(decisions) != 3 {
		t.Fatalf("应返回 3 份决策，实际 %d", len(decisions))
	}
}

// TestScheduleGangAllOrNothing 验证份数不足时回滚全部（All-or-Nothing）。
func TestScheduleGangAllOrNothing(t *testing.T) {
	l := ledger.New()
	// 单卡仅够 2 份 8000MB 的显存共享，请求 3 份必失败。
	node := buildMultiGPUNode("node1", 1, 18000)
	l.SyncNode(node)
	s := New(l, "binpack")

	task := &domain.Task{Resource: domain.ResourceRequest{
		GPUCount: 1, MemoryMB: 8000, ShareMode: domain.ShareMemory, GangSize: 3,
	}}
	if _, err := s.ScheduleGang(context.Background(), task, []*domain.Node{node}); err == nil {
		t.Fatal("份数不足时成组调度应失败")
	}
	// All-or-Nothing：失败后资源应全部释放。
	if free := l.NodeFreeMemory("node1"); free != 18000 {
		t.Errorf("成组失败应回滚全部资源，剩余期望 18000，实际 %d", free)
	}
}

// TestScheduleGangDegradesToSingle 验证 GangSize<=1 时退化为单次调度。
func TestScheduleGangDegradesToSingle(t *testing.T) {
	l := ledger.New()
	node := buildMultiGPUNode("node1", 1, 16384)
	l.SyncNode(node)
	s := New(l, "binpack")

	task := &domain.Task{Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 4096}}
	decisions, err := s.ScheduleGang(context.Background(), task, []*domain.Node{node})
	if err != nil {
		t.Fatalf("调度应成功: %v", err)
	}
	if len(decisions) != 1 {
		t.Errorf("GangSize<=1 应返回 1 份决策，实际 %d", len(decisions))
	}
}
