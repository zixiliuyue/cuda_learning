package metrics

import (
	"context"
	"strings"
	"testing"

	"qingtian/internal/domain"
)

// fakeNodeLister 是 NodeLister 的测试桩，返回预置节点列表。
type fakeNodeLister struct {
	nodes []*domain.Node
}

func (f *fakeNodeLister) List(_ context.Context) ([]*domain.Node, error) {
	return f.nodes, nil
}

// TestExporterRendersPrometheusFormat 验证导出器输出符合 Prometheus 文本格式的关键指标。
func TestExporterRendersPrometheusFormat(t *testing.T) {
	lister := &fakeNodeLister{nodes: []*domain.Node{
		{
			ID: "node-1", Hostname: "gpu-host", Status: domain.NodeOnline,
			GPUs: []domain.GPU{
				{UUID: "gpu-0", Model: "RTX 5060 Ti", Vendor: domain.VendorNVIDIA,
					MemoryTotal: 16384, MemoryUsed: 4096, Utilization: 50},
			},
		},
	}}
	exp := NewExporter(lister)

	out, err := exp.Render(context.Background())
	if err != nil {
		t.Fatalf("渲染指标应成功: %v", err)
	}

	// 校验包含 HELP/TYPE 头与各指标样本。
	checks := []string{
		"# TYPE qingtian_node_up gauge",
		`qingtian_node_up{node="node-1",hostname="gpu-host"} 1`,
		`qingtian_gpu_memory_total_mb{node="node-1",gpu="gpu-0",model="RTX 5060 Ti",vendor="nvidia"} 16384`,
		`qingtian_gpu_memory_used_mb{node="node-1",gpu="gpu-0",model="RTX 5060 Ti",vendor="nvidia"} 4096`,
		"qingtian_gpu_utilization_ratio",
	}
	for _, c := range checks {
		if !strings.Contains(out, c) {
			t.Errorf("输出应包含 %q，实际输出:\n%s", c, out)
		}
	}
}

// TestExporterOfflineNode 验证离线节点的 up 指标为 0。
func TestExporterOfflineNode(t *testing.T) {
	lister := &fakeNodeLister{nodes: []*domain.Node{
		{ID: "node-2", Hostname: "down-host", Status: domain.NodeOffline},
	}}
	exp := NewExporter(lister)

	out, err := exp.Render(context.Background())
	if err != nil {
		t.Fatalf("渲染指标应成功: %v", err)
	}
	if !strings.Contains(out, `qingtian_node_up{node="node-2",hostname="down-host"} 0`) {
		t.Errorf("离线节点 up 指标应为 0，实际输出:\n%s", out)
	}
}
