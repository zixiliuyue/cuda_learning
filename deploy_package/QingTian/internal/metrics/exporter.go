// Package metrics 实现 Prometheus 文本格式的指标导出。
//
// 平台此前仅在数据库中记录 GPU 分配与用量，缺少对接监控系统的标准出口。
// 本包把节点与 GPU 的实时状态渲染为 Prometheus 文本曝光格式（exposition format），
// 通过 /metrics 端点暴露，可直接被 Prometheus 抓取、在 Grafana 中出图，
// 从而获得集群负载、显存占用、卡利用率、功耗等全维度可观测性。
//
// 为保持零第三方依赖，本包不引入 prometheus/client_golang，
// 而是用标准库手写符合规范的文本输出（GAUGE 类型 + 标签）。
package metrics

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"qingtian/internal/domain"
)

// NodeLister 抽象节点列表来源，便于注入仓储实现与测试桩。
type NodeLister interface {
	List(ctx context.Context) ([]*domain.Node, error)
}

// Exporter 负责把节点/GPU 状态渲染为 Prometheus 文本格式。
//
// 字段：
//   - nodes: 节点列表来源。
type Exporter struct {
	nodes NodeLister
}

// NewExporter 创建指标导出器。
func NewExporter(nodes NodeLister) *Exporter {
	return &Exporter{nodes: nodes}
}

// metric 表示一个待渲染的指标族（同名指标的多条样本）。
type metric struct {
	name    string
	help    string
	samples []sample
}

// sample 表示一条指标样本：一组标签 + 一个数值。
type sample struct {
	labels [][2]string
	value  float64
}

// Render 生成全部指标的 Prometheus 文本格式输出。
//
// 输出包含以下指标（均为 gauge）：
//   - qingtian_node_up:              节点是否在线（1/0）。
//   - qingtian_node_gpu_total:       节点 GPU 总数。
//   - qingtian_gpu_memory_total_mb:  单卡显存总量（MB）。
//   - qingtian_gpu_memory_used_mb:   单卡已用显存（MB）。
//   - qingtian_gpu_utilization_ratio:单卡利用率（0-1）。
//
// 参数：
//   - ctx: 上下文。
//
// 返回值：
//   - string: 符合 Prometheus 规范的文本；每个指标族含 # HELP/# TYPE 头。
//   - error:  读取节点列表失败时返回。
func (e *Exporter) Render(ctx context.Context) (string, error) {
	nodes, err := e.nodes.List(ctx)
	if err != nil {
		return "", err
	}
	// 节点维度指标按节点名排序，保证输出稳定（便于测试与 diff）。
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].ID < nodes[j].ID })

	nodeUp := metric{name: "qingtian_node_up", help: "节点是否在线：1=在线，0=离线。"}
	nodeGPUTotal := metric{name: "qingtian_node_gpu_total", help: "节点上的 GPU 总数。"}
	gpuMemTotal := metric{name: "qingtian_gpu_memory_total_mb", help: "单块 GPU 的显存总量（MB）。"}
	gpuMemUsed := metric{name: "qingtian_gpu_memory_used_mb", help: "单块 GPU 的已用显存（MB）。"}
	gpuUtil := metric{name: "qingtian_gpu_utilization_ratio", help: "单块 GPU 的实时利用率（0-1）。"}

	for _, n := range nodes {
		nodeLabels := [][2]string{{"node", n.ID}, {"hostname", n.Hostname}}
		up := 0.0
		if n.Status == domain.NodeOnline {
			up = 1.0
		}
		nodeUp.samples = append(nodeUp.samples, sample{labels: nodeLabels, value: up})
		nodeGPUTotal.samples = append(nodeGPUTotal.samples, sample{labels: nodeLabels, value: float64(len(n.GPUs))})

		for _, g := range n.GPUs {
			gpuLabels := [][2]string{
				{"node", n.ID},
				{"gpu", g.UUID},
				{"model", g.Model},
				{"vendor", string(g.Vendor)},
			}
			gpuMemTotal.samples = append(gpuMemTotal.samples, sample{labels: gpuLabels, value: float64(g.MemoryTotal)})
			gpuMemUsed.samples = append(gpuMemUsed.samples, sample{labels: gpuLabels, value: float64(g.MemoryUsed)})
			gpuUtil.samples = append(gpuUtil.samples, sample{labels: gpuLabels, value: float64(g.Utilization) / 100.0})
		}
	}

	var sb strings.Builder
	for _, m := range []metric{nodeUp, nodeGPUTotal, gpuMemTotal, gpuMemUsed, gpuUtil} {
		writeMetric(&sb, m)
	}
	return sb.String(), nil
}

// writeMetric 把单个指标族渲染为文本（含 HELP/TYPE 头与全部样本）。
func writeMetric(sb *strings.Builder, m metric) {
	fmt.Fprintf(sb, "# HELP %s %s\n", m.name, m.help)
	fmt.Fprintf(sb, "# TYPE %s gauge\n", m.name)
	for _, s := range m.samples {
		sb.WriteString(m.name)
		if len(s.labels) > 0 {
			sb.WriteByte('{')
			for i, kv := range s.labels {
				if i > 0 {
					sb.WriteByte(',')
				}
				fmt.Fprintf(sb, "%s=\"%s\"", kv[0], escapeLabelValue(kv[1]))
			}
			sb.WriteByte('}')
		}
		fmt.Fprintf(sb, " %s\n", formatFloat(s.value))
	}
}

// escapeLabelValue 按 Prometheus 规范转义标签值中的反斜杠、双引号与换行。
func escapeLabelValue(v string) string {
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`)
	return r.Replace(v)
}

// formatFloat 将数值格式化为紧凑文本：整数不带小数点，其余保留必要精度。
func formatFloat(f float64) string {
	if f == float64(int64(f)) {
		return fmt.Sprintf("%d", int64(f))
	}
	return fmt.Sprintf("%g", f)
}
