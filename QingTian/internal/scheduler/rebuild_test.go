package scheduler

import (
	"context"
	"testing"

	"qingtian/internal/domain"
	"qingtian/internal/ledger"
)

// TestRebuildLedgerPreventsOversell 验证账本重建后会重放活动任务占用，避免超卖。
//
// 场景：单卡 8G 节点上有一个 running 任务已占满整卡。若调度器重启后账本为空，
// 会误认为整卡空闲而超卖。重建账本应重放该占用，使新任务无处可调度。
func TestRebuildLedgerPreventsOversell(t *testing.T) {
	l := ledger.New()
	s := New(l, "binpack")
	node := buildNode("node1", 8192)

	running := &domain.Task{
		ID: "t-running", Status: domain.TaskRunning, NodeID: "node1",
		AllocatedGPUs: []string{"node1-gpu0"},
		Resource:      domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192},
	}

	// 重建账本：登记节点容量 + 重放活动任务占用。
	s.RebuildLedger([]*domain.Node{node}, []*domain.Task{running})

	// 新任务需要整卡，但卡已被重放占用，应无法调度（防止超卖）。
	newTask := &domain.Task{Resource: domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192}}
	if _, err := s.Schedule(context.Background(), newTask, []*domain.Node{node}); err == nil {
		t.Fatal("重建账本后整卡已被占用，新任务不应被调度（否则超卖）")
	}
}

// TestRebuildLedgerResetsStaleState 验证重建会先清空旧状态再登记。
func TestRebuildLedgerResetsStaleState(t *testing.T) {
	l := ledger.New()
	s := New(l, "binpack")
	node := buildNode("node1", 8192)

	// 首次重建：无活动任务，整卡空闲。
	s.RebuildLedger([]*domain.Node{node}, nil)
	if free := l.NodeFreeMemory("node1"); free != 8192 {
		t.Fatalf("重建后整卡应空闲，剩余期望 8192，实际 %d", free)
	}

	// 再次重建仍无活动任务：应重置为满额，而非在旧账本上叠加。
	s.RebuildLedger([]*domain.Node{node}, nil)
	if free := l.NodeFreeMemory("node1"); free != 8192 {
		t.Errorf("重复重建不应叠加占用，剩余期望 8192，实际 %d", free)
	}
}

// TestScheduleGangSpreadsAcrossNodes 验证成组调度优先把份数分散到不同节点。
func TestScheduleGangSpreadsAcrossNodes(t *testing.T) {
	l := ledger.New()
	// 两个整卡节点，各自足够承载多份，但拓扑分散应让两份落到不同节点。
	n1 := buildNode("node1", 16384)
	n2 := buildNode("node2", 16384)
	l.SyncNode(n1)
	l.SyncNode(n2)
	s := New(l, "binpack")

	task := &domain.Task{Resource: domain.ResourceRequest{
		GPUCount: 1, MemoryMB: 4096, ShareMode: domain.ShareMemory, GangSize: 2,
	}}
	decisions, err := s.ScheduleGang(context.Background(), task, []*domain.Node{n1, n2})
	if err != nil {
		t.Fatalf("成组调度应成功: %v", err)
	}
	if len(decisions) != 2 {
		t.Fatalf("应返回 2 份决策，实际 %d", len(decisions))
	}
	if decisions[0].NodeID == decisions[1].NodeID {
		t.Errorf("拓扑分散：两份应落在不同节点，实际都在 %s", decisions[0].NodeID)
	}
}
