package ledger

import (
	"testing"

	"qingtian/internal/domain"
)

// newTestNode 构造一个带指定 GPU 数量与显存的测试节点。
func newTestNode(id string, gpuCount int, memPerGPU int64) *domain.Node {
	gpus := make([]domain.GPU, gpuCount)
	for i := 0; i < gpuCount; i++ {
		gpus[i] = domain.GPU{
			UUID:        id + "-gpu-" + string(rune('a'+i)),
			Model:       "RTX 5060 Ti",
			MemoryTotal: memPerGPU,
			Allocatable: true,
		}
	}
	return &domain.Node{ID: id, Status: domain.NodeOnline, GPUs: gpus}
}

// TestTryAllocateSuccess 验证资源充足时能成功分配并扣减显存。
func TestTryAllocateSuccess(t *testing.T) {
	l := New()
	node := newTestNode("node1", 2, 16384)
	l.SyncNode(node)

	req := domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192}
	cand, err := l.TryAllocate("node1", req)
	if err != nil {
		t.Fatalf("分配应成功: %v", err)
	}
	if len(cand.GPUUUIDs) != 1 {
		t.Fatalf("应分配 1 块 GPU，实际 %d", len(cand.GPUUUIDs))
	}
	// 分配后剩余显存应为 2*16384 - 8192 = 24576。
	if free := l.NodeFreeMemory("node1"); free != 24576 {
		t.Errorf("剩余显存应为 24576，实际 %d", free)
	}
}

// TestTryAllocateInsufficient 验证资源不足时分配失败且不扣减。
func TestTryAllocateInsufficient(t *testing.T) {
	l := New()
	l.SyncNode(newTestNode("node1", 1, 4096))

	req := domain.ResourceRequest{GPUCount: 2, MemoryMB: 1024} // 需要 2 卡但只有 1 卡
	if _, err := l.TryAllocate("node1", req); err == nil {
		t.Fatal("资源不足应返回错误")
	}
	// 失败后显存不应被扣减。
	if free := l.NodeFreeMemory("node1"); free != 4096 {
		t.Errorf("失败后剩余显存应保持 4096，实际 %d", free)
	}
}

// TestReleaseRestoresMemory 验证释放资源后显存归还。
func TestReleaseRestoresMemory(t *testing.T) {
	l := New()
	l.SyncNode(newTestNode("node1", 1, 16384))
	req := domain.ResourceRequest{GPUCount: 1, MemoryMB: 8192}

	cand, _ := l.TryAllocate("node1", req)
	if free := l.NodeFreeMemory("node1"); free != 8192 {
		t.Fatalf("分配后剩余应为 8192，实际 %d", free)
	}
	l.Release(cand.GPUUUIDs, req.MemoryMB)
	if free := l.NodeFreeMemory("node1"); free != 16384 {
		t.Errorf("释放后剩余应恢复为 16384，实际 %d", free)
	}
}

// TestNoOversell 验证并发/重复分配不会超卖。
func TestNoOversell(t *testing.T) {
	l := New()
	l.SyncNode(newTestNode("node1", 1, 16384))
	req := domain.ResourceRequest{GPUCount: 1, MemoryMB: 10000}

	// 第一次分配成功。
	if _, err := l.TryAllocate("node1", req); err != nil {
		t.Fatalf("首次分配应成功: %v", err)
	}
	// 第二次因剩余不足（6384 < 10000）应失败。
	if _, err := l.TryAllocate("node1", req); err == nil {
		t.Fatal("第二次分配应失败以防止超卖")
	}
}

// TestModelConstraint 验证型号约束生效。
func TestModelConstraint(t *testing.T) {
	l := New()
	l.SyncNode(newTestNode("node1", 1, 16384))
	req := domain.ResourceRequest{GPUCount: 1, MemoryMB: 1024, GPUModel: "A100"}
	if _, err := l.TryAllocate("node1", req); err == nil {
		t.Fatal("型号不匹配应分配失败")
	}
}
