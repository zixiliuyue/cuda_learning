package ledger

import (
	"testing"

	"qingtian/internal/domain"
)

// TestSnapshotRoundTrip 验证账本快照导出后重新导入，占用状态完全一致。
func TestSnapshotRoundTrip(t *testing.T) {
	l := New()
	node := newTestNode("node1", 1, 16384)
	l.SyncNode(node)

	// 分配整卡独占。
	if _, err := l.TryAllocate("node1", domain.ResourceRequest{
		GPUCount: 1, MemoryMB: 8192, ShareMode: domain.ShareExclusive,
	}); err != nil {
		t.Fatalf("分配应成功: %v", err)
	}
	freeBefore := l.NodeFreeMemory("node1")

	// 序列化再反序列化到新账本。
	data, err := l.MarshalSnapshot()
	if err != nil {
		t.Fatalf("序列化快照应成功: %v", err)
	}
	l2 := New()
	if err := l2.LoadSnapshot(data); err != nil {
		t.Fatalf("加载快照应成功: %v", err)
	}

	if free := l2.NodeFreeMemory("node1"); free != freeBefore {
		t.Errorf("快照恢复后剩余显存应一致，期望 %d，实际 %d", freeBefore, free)
	}
	// 恢复后独占标记应保留：该卡不能再被独占分配。
	if l2.CanAllocate("node1", domain.ResourceRequest{
		GPUCount: 1, MemoryMB: 8192, ShareMode: domain.ShareExclusive,
	}) {
		t.Error("恢复后被独占的卡不应再可分配整卡独占")
	}
}

// TestLoadSnapshotEmpty 验证空数据视为空账本。
func TestLoadSnapshotEmpty(t *testing.T) {
	l := New()
	node := newTestNode("node1", 1, 8192)
	l.SyncNode(node)
	if err := l.LoadSnapshot(nil); err != nil {
		t.Fatalf("加载空快照应成功: %v", err)
	}
	if free := l.NodeFreeMemory("node1"); free != 0 {
		t.Errorf("空快照应清空账本，节点不再登记，剩余期望 0，实际 %d", free)
	}
}
