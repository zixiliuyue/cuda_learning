// 本文件提供异构算力（国产 GPU/NPU）的采集器实现。
//
// 契合信创需求：Agent 采集层通过 Collector 接口抽象，屏蔽不同厂商
// 加速卡的差异。除英伟达外，这里提供华为昇腾、寒武纪的模拟采集器，
// 生产环境可分别替换为基于 CANN(昇腾)/CNTOOLKIT(寒武纪) 的真实实现。
package agent

import "qingtian/internal/domain"

// NewAscendMockCollector 返回一个模拟华为昇腾 NPU 节点的采集器。
//
// 参数：
//   - count:    NPU 数量。
//   - memoryMB: 单卡显存（MB）。
func NewAscendMockCollector(count int, memoryMB int64) *MockCollector {
	return NewVendorMockCollector(domain.VendorAscend, "Ascend 910B", count, memoryMB, "CANN-7.0", "7.0")
}

// NewCambriconMockCollector 返回一个模拟寒武纪 MLU 节点的采集器。
//
// 参数：
//   - count:    MLU 数量。
//   - memoryMB: 单卡显存（MB）。
func NewCambriconMockCollector(count int, memoryMB int64) *MockCollector {
	return NewVendorMockCollector(domain.VendorCambricon, "MLU370-X8", count, memoryMB, "CNToolkit-3.0", "3.0")
}

// NewVendorMockCollector 按指定厂商/型号构造一个多卡模拟采集器。
//
// 参数：
//   - vendor:   厂商。
//   - model:    型号。
//   - count:    卡数量。
//   - memoryMB: 单卡显存（MB）。
//   - driver:   驱动版本。
//   - runtime:  运行时/工具链版本（映射到 CUDAVersion 字段以复用结构）。
func NewVendorMockCollector(vendor domain.Vendor, model string, count int, memoryMB int64, driver, runtime string) *MockCollector {
	gpus := make([]domain.GPU, count)
	for i := 0; i < count; i++ {
		gpus[i] = domain.GPU{
			UUID:        string(vendor) + "-MOCK-" + string(rune('0'+i)),
			Vendor:      vendor,
			Model:       model,
			MemoryTotal: memoryMB,
			MemoryUsed:  0,
			Utilization: 0,
			Allocatable: true,
		}
	}
	return NewMockCollector(gpus, driver, runtime)
}
