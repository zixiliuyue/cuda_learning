package ledger

import (
	"testing"

	"qingtian/internal/domain"
)

// newMemShareNode 构造一块支持显存共享的大显存 GPU 节点。
func newMemShareNode(id string, memTotal int64) *domain.Node {
	return &domain.Node{
		ID:     id,
		Status: domain.NodeOnline,
		GPUs: []domain.GPU{
			{UUID: id + "-gpu-0", Model: "A100", Vendor: domain.VendorNVIDIA, MemoryTotal: memTotal, Allocatable: true},
		},
	}
}

// TestMemoryShareAllowsMultipleTasks 验证显存切片模式下多个任务可共享同一块卡。
func TestMemoryShareAllowsMultipleTasks(t *testing.T) {
	l := New()
	l.SyncNode(newMemShareNode("node1", 40960)) // 40G
	req := domain.ResourceRequest{GPUCount: 1, MemoryMB: 10000, ShareMode: domain.ShareMemory}

	c1, err := l.TryAllocate("node1", req)
	if err != nil {
		t.Fatalf("第一次显存共享分配应成功: %v", err)
	}
	c2, err := l.TryAllocate("node1", req)
	if err != nil {
		t.Fatalf("第二次显存共享分配应成功（同卡共享）: %v", err)
	}
	if c1.GPUUUIDs[0] != c2.GPUUUIDs[0] {
		t.Error("两次显存共享应落在同一块卡上")
	}
	// 已用 20000，剩余 20960。
	if free := l.NodeFreeMemory("node1"); free != 20960 {
		t.Errorf("剩余显存应为 20960，实际 %d", free)
	}
}

// TestExclusiveBlocksSharing 验证整卡独占后其它任务无法再共享该卡。
func TestExclusiveBlocksSharing(t *testing.T) {
	l := New()
	l.SyncNode(newMemShareNode("node1", 40960))

	excl := domain.ResourceRequest{GPUCount: 1, MemoryMB: 1000, ShareMode: domain.ShareExclusive}
	if _, err := l.TryAllocate("node1", excl); err != nil {
		t.Fatalf("独占分配应成功: %v", err)
	}
	// 独占后即使显存充足，共享请求也应失败。
	share := domain.ResourceRequest{GPUCount: 1, MemoryMB: 1000, ShareMode: domain.ShareMemory}
	if _, err := l.TryAllocate("node1", share); err == nil {
		t.Fatal("整卡被独占后，共享请求应失败")
	}
}

// TestShareBlocksExclusive 验证已被共享占用的卡无法再被独占。
func TestShareBlocksExclusive(t *testing.T) {
	l := New()
	l.SyncNode(newMemShareNode("node1", 40960))

	share := domain.ResourceRequest{GPUCount: 1, MemoryMB: 1000, ShareMode: domain.ShareMemory}
	if _, err := l.TryAllocate("node1", share); err != nil {
		t.Fatalf("共享分配应成功: %v", err)
	}
	excl := domain.ResourceRequest{GPUCount: 1, MemoryMB: 1000, ShareMode: domain.ShareExclusive}
	if _, err := l.TryAllocate("node1", excl); err == nil {
		t.Fatal("卡已有共享占用时，独占请求应失败")
	}
}

// newMIGNode 构造一块开启 MIG、含两个实例的 GPU 节点。
func newMIGNode(id string) *domain.Node {
	return &domain.Node{
		ID:     id,
		Status: domain.NodeOnline,
		GPUs: []domain.GPU{
			{
				UUID: id + "-gpu-0", Model: "A100", Vendor: domain.VendorNVIDIA,
				MemoryTotal: 40960, Allocatable: true, MIGEnabled: true,
				MIGInstances: []domain.MIGInstance{
					{ID: "mig-0", Profile: "2g.10gb", MemoryMB: 10240, Allocatable: true},
					{ID: "mig-1", Profile: "2g.10gb", MemoryMB: 10240, Allocatable: true},
				},
			},
		},
	}
}

// TestMIGAllocateAndRelease 验证 MIG 实例的分配、耗尽与释放。
func TestMIGAllocateAndRelease(t *testing.T) {
	l := New()
	l.SyncNode(newMIGNode("node1"))
	req := domain.ResourceRequest{GPUCount: 1, MemoryMB: 8000, ShareMode: domain.ShareMIG}

	c1, err := l.TryAllocate("node1", req)
	if err != nil {
		t.Fatalf("第一个 MIG 实例分配应成功: %v", err)
	}
	if len(c1.MIGIDs) != 1 || c1.ShareMode != domain.ShareMIG {
		t.Fatalf("应返回一个 MIG 实例，实际 %+v", c1)
	}
	c2, err := l.TryAllocate("node1", req)
	if err != nil {
		t.Fatalf("第二个 MIG 实例分配应成功: %v", err)
	}
	// 两个实例都被占用，第三次应失败。
	if _, err := l.TryAllocate("node1", req); err == nil {
		t.Fatal("MIG 实例耗尽后应分配失败")
	}
	// 释放一个后可再次分配。
	l.Release(c1.AllocatedIDs(), req.MemoryMB)
	if _, err := l.TryAllocate("node1", req); err != nil {
		t.Fatalf("释放 MIG 实例后应可再次分配: %v", err)
	}
	_ = c2
}

// TestVendorConstraint 验证厂商约束在异构算力场景下生效。
func TestVendorConstraint(t *testing.T) {
	l := New()
	node := &domain.Node{
		ID: "node1", Status: domain.NodeOnline,
		GPUs: []domain.GPU{{UUID: "npu-0", Vendor: domain.VendorAscend, MemoryTotal: 32768, Allocatable: true}},
	}
	l.SyncNode(node)
	// 请求 NVIDIA 应失败（该节点只有昇腾 NPU）。
	req := domain.ResourceRequest{GPUCount: 1, MemoryMB: 1024, Vendor: domain.VendorNVIDIA}
	if _, err := l.TryAllocate("node1", req); err == nil {
		t.Fatal("厂商不匹配应分配失败")
	}
	// 请求昇腾应成功。
	req.Vendor = domain.VendorAscend
	if _, err := l.TryAllocate("node1", req); err != nil {
		t.Fatalf("厂商匹配应分配成功: %v", err)
	}
}

// TestNodeUtilization 验证节点显存分配率计算。
func TestNodeUtilization(t *testing.T) {
	l := New()
	l.SyncNode(newMemShareNode("node1", 10000))
	if u := l.NodeUtilization("node1"); u != 0 {
		t.Errorf("初始分配率应为 0，实际 %v", u)
	}
	_, _ = l.TryAllocate("node1", domain.ResourceRequest{GPUCount: 1, MemoryMB: 5000, ShareMode: domain.ShareMemory})
	if u := l.NodeUtilization("node1"); u != 0.5 {
		t.Errorf("分配一半后利用率应为 0.5，实际 %v", u)
	}
}
