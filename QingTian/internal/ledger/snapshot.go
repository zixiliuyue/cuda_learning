// 本文件为资源账本提供可序列化快照能力（Export/Import）。
//
// 快照让账本状态可以持久化到共享存储（如 Redis），从而在多调度器实例之间
// 共享同一份占用视图、并在进程重启后恢复占用，避免"重启即认为所有卡空闲"。
// 快照与内存账本使用完全相同的分配/释放逻辑，不做任何语义重写，杜绝漂移。
package ledger

import (
	"encoding/json"

	"qingtian/internal/domain"
)

// migSnapshot 是单个 MIG 实例的可序列化快照。
type migSnapshot struct {
	ID          string `json:"id"`
	ParentUUID  string `json:"parent_uuid"`
	MemoryMB    int64  `json:"memory_mb"`
	Allocatable bool   `json:"allocatable"`
	Used        bool   `json:"used"`
}

// gpuSnapshot 是单块 GPU 账户的可序列化快照。
type gpuSnapshot struct {
	UUID        string        `json:"uuid"`
	Vendor      domain.Vendor `json:"vendor"`
	Model       string        `json:"model"`
	MemoryTotal int64         `json:"memory_total"`
	MemoryUsed  int64         `json:"memory_used"`
	Allocated   int           `json:"allocated"`
	Allocatable bool          `json:"allocatable"`
	Exclusive   bool          `json:"exclusive"`
	MIGEnabled  bool          `json:"mig_enabled"`
	MIGs        []migSnapshot `json:"migs,omitempty"`
}

// Snapshot 是整个账本的可序列化快照。
type Snapshot struct {
	GPUs     []gpuSnapshot       `json:"gpus"`
	NodeGPUs map[string][]string `json:"node_gpus"`
}

// Export 导出账本当前状态为可序列化快照。
func (l *Ledger) Export() Snapshot {
	l.mu.Lock()
	defer l.mu.Unlock()
	snap := Snapshot{
		GPUs:     make([]gpuSnapshot, 0, len(l.gpus)),
		NodeGPUs: make(map[string][]string, len(l.nodeGPUs)),
	}
	for _, acc := range l.gpus {
		gs := gpuSnapshot{
			UUID:        acc.uuid,
			Vendor:      acc.vendor,
			Model:       acc.model,
			MemoryTotal: acc.memoryTotal,
			MemoryUsed:  acc.memoryUsed,
			Allocated:   acc.allocated,
			Allocatable: acc.allocatable,
			Exclusive:   acc.exclusive,
			MIGEnabled:  acc.migEnabled,
		}
		for _, m := range acc.migs {
			gs.MIGs = append(gs.MIGs, migSnapshot{
				ID:          m.id,
				ParentUUID:  m.parentUUID,
				MemoryMB:    m.memoryMB,
				Allocatable: m.allocatable,
				Used:        m.used,
			})
		}
		snap.GPUs = append(snap.GPUs, gs)
	}
	for node, uuids := range l.nodeGPUs {
		cp := make([]string, len(uuids))
		copy(cp, uuids)
		snap.NodeGPUs[node] = cp
	}
	return snap
}

// Import 用快照覆盖账本当前状态（重建索引）。
func (l *Ledger) Import(snap Snapshot) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.gpus = make(map[string]*gpuAccount, len(snap.GPUs))
	l.migIndex = make(map[string]*migAccount)
	l.nodeGPUs = make(map[string][]string, len(snap.NodeGPUs))
	for _, gs := range snap.GPUs {
		acc := &gpuAccount{
			uuid:        gs.UUID,
			vendor:      gs.Vendor,
			model:       gs.Model,
			memoryTotal: gs.MemoryTotal,
			memoryUsed:  gs.MemoryUsed,
			allocated:   gs.Allocated,
			allocatable: gs.Allocatable,
			exclusive:   gs.Exclusive,
			migEnabled:  gs.MIGEnabled,
		}
		for _, ms := range gs.MIGs {
			m := &migAccount{
				id:          ms.ID,
				parentUUID:  ms.ParentUUID,
				memoryMB:    ms.MemoryMB,
				allocatable: ms.Allocatable,
				used:        ms.Used,
			}
			acc.migs = append(acc.migs, m)
			l.migIndex[m.id] = m
		}
		l.gpus[acc.uuid] = acc
	}
	for node, uuids := range snap.NodeGPUs {
		cp := make([]string, len(uuids))
		copy(cp, uuids)
		l.nodeGPUs[node] = cp
	}
}

// MarshalSnapshot 将当前账本状态序列化为 JSON 字节，供写入共享存储。
func (l *Ledger) MarshalSnapshot() ([]byte, error) {
	return json.Marshal(l.Export())
}

// LoadSnapshot 从 JSON 字节反序列化并覆盖账本状态。
//
// 空字节视为空账本（无占用），用于共享存储中尚无快照时的首次加载。
func (l *Ledger) LoadSnapshot(data []byte) error {
	if len(data) == 0 {
		l.Reset()
		return nil
	}
	var snap Snapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return err
	}
	l.Import(snap)
	return nil
}
