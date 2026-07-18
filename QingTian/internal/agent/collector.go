// Package agent 实现部署在 GPU 节点上的 Agent 组件。
//
// Agent 是数据面核心，负责：节点注册、心跳上报、GPU 指标采集、任务执行与状态回传。
// 为保证跨平台可编译（Windows/macOS 无 NVML），采集层通过接口抽象，
// 默认提供可配置的模拟采集器；生产环境可替换为真实 NVML 实现。
package agent

import "qingtian/internal/domain"

// Collector 定义 GPU 指标采集能力。
//
// 生产环境应实现基于 NVML 的采集；测试与无 GPU 环境使用 MockCollector。
type Collector interface {
	// Collect 返回当前节点上所有 GPU 的实时信息。
	Collect() ([]domain.GPU, error)
	// DriverVersion 返回 NVIDIA 驱动版本。
	DriverVersion() string
	// CUDAVersion 返回 CUDA 版本。
	CUDAVersion() string
}

// MockCollector 是 Collector 的模拟实现，用于无 GPU 环境与测试。
//
// 它持有一组固定的 GPU 描述，并在每次 Collect 时返回（可模拟利用率波动）。
type MockCollector struct {
	gpus   []domain.GPU
	driver string
	cuda   string
	tick   int
}

// NewMockCollector 创建一个模拟采集器。
//
// 参数：
//   - gpus:   模拟的 GPU 列表。
//   - driver: 模拟驱动版本。
//   - cuda:   模拟 CUDA 版本。
func NewMockCollector(gpus []domain.GPU, driver, cuda string) *MockCollector {
	return &MockCollector{gpus: gpus, driver: driver, cuda: cuda}
}

// DefaultMockCollector 返回一个带单块 RTX 5060 Ti（16G）的模拟采集器，
// 便于本地演示（对应用户的目标硬件）。
func DefaultMockCollector() *MockCollector {
	return NewMockCollector([]domain.GPU{
		{
			UUID:        "GPU-MOCK-0001",
			Model:       "RTX 5060 Ti",
			MemoryTotal: 16384,
			MemoryUsed:  0,
			Utilization: 0,
			Allocatable: true,
		},
	}, "580.00", "13.1")
}

// Collect 实现 Collector.Collect。
//
// 模拟实现会在每次调用时让利用率在 0-100 之间循环变化，便于观察监控效果。
func (c *MockCollector) Collect() ([]domain.GPU, error) {
	c.tick = (c.tick + 10) % 101
	out := make([]domain.GPU, len(c.gpus))
	for i, g := range c.gpus {
		g.Utilization = c.tick
		out[i] = g
	}
	return out, nil
}

// DriverVersion 实现 Collector.DriverVersion。
func (c *MockCollector) DriverVersion() string { return c.driver }

// CUDAVersion 实现 Collector.CUDAVersion。
func (c *MockCollector) CUDAVersion() string { return c.cuda }
