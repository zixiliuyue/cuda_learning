package scheduler

import (
	"context"
	"testing"

	"qingtian/internal/domain"
	"qingtian/internal/ledger"
)

// buildNode 构造一块 GPU、指定显存的在线节点。
func buildNode(id string, mem int64) *domain.Node {
	return &domain.Node{
		ID:     id,
		Status: domain.NodeOnline,
		GPUs: []domain.GPU{
			{UUID: id + "-gpu0", Model: "RTX 5060 Ti", MemoryTotal: mem, Allocatable: true},
		},
	}
}

// TestScheduleBinPack 验证装箱策略优先选择剩余资源最少的节点。
func TestScheduleBinPack(t *testing.T) {
	l := ledger.New()
	// node-small 剩余 8192，node-big 剩余 16384。
	small := buildNode("node-small", 8192)
	big := buildNode("node-big", 16384)
	l.SyncNode(small)
	l.SyncNode(big)

	s := New(l, "binpack")
	task := &domain.Task{Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 4096}}
	dec, err := s.Schedule(context.Background(), task, []*domain.Node{small, big})
	if err != nil {
		t.Fatalf("调度应成功: %v", err)
	}
	// 装箱：分配后剩余更少的 node-small（8192-4096=4096）优于 node-big（16384-4096=12288）。
	if dec.NodeID != "node-small" {
		t.Errorf("装箱策略应选择 node-small，实际 %s", dec.NodeID)
	}
}

// TestScheduleSpread 验证分散策略优先选择剩余资源最多的节点。
func TestScheduleSpread(t *testing.T) {
	l := ledger.New()
	small := buildNode("node-small", 8192)
	big := buildNode("node-big", 16384)
	l.SyncNode(small)
	l.SyncNode(big)

	s := New(l, "spread")
	task := &domain.Task{Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 4096}}
	dec, err := s.Schedule(context.Background(), task, []*domain.Node{small, big})
	if err != nil {
		t.Fatalf("调度应成功: %v", err)
	}
	if dec.NodeID != "node-big" {
		t.Errorf("分散策略应选择 node-big，实际 %s", dec.NodeID)
	}
}

// TestScheduleNoAvailableNode 验证无可用节点时返回错误。
func TestScheduleNoAvailableNode(t *testing.T) {
	l := ledger.New()
	node := buildNode("node1", 2048)
	l.SyncNode(node)
	s := New(l, "binpack")
	task := &domain.Task{Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192}} // 需求超过容量
	if _, err := s.Schedule(context.Background(), task, []*domain.Node{node}); err == nil {
		t.Fatal("无满足节点应返回错误")
	}
}

// TestScheduleSkipsOfflineNode 验证离线节点被跳过。
func TestScheduleSkipsOfflineNode(t *testing.T) {
	l := ledger.New()
	node := buildNode("node1", 16384)
	node.Status = domain.NodeOffline
	l.SyncNode(node)
	s := New(l, "binpack")
	task := &domain.Task{Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 1024}}
	if _, err := s.Schedule(context.Background(), task, []*domain.Node{node}); err == nil {
		t.Fatal("离线节点应被跳过，导致调度失败")
	}
}

// TestScheduleRollsBackUnchosen 验证未选中候选节点的试分配被回滚。
func TestScheduleRollsBackUnchosen(t *testing.T) {
	l := ledger.New()
	small := buildNode("node-small", 8192)
	big := buildNode("node-big", 16384)
	l.SyncNode(small)
	l.SyncNode(big)
	s := New(l, "binpack")
	task := &domain.Task{Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 4096}}

	_, err := s.Schedule(context.Background(), task, []*domain.Node{small, big})
	if err != nil {
		t.Fatalf("调度应成功: %v", err)
	}
	// 未被选中的 node-big 应恢复到满额 16384。
	if free := l.NodeFreeMemory("node-big"); free != 16384 {
		t.Errorf("未选中节点应被回滚为 16384，实际 %d", free)
	}
}
