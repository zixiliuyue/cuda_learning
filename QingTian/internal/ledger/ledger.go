// Package ledger 实现 GPU 资源账本，负责算力资源的分配与释放。
//
// 账本是防止 GPU 超卖的关键：调度器在决定将任务放到某些 GPU 上时，
// 必须先通过账本原子地扣减资源；任务结束或调度失败时释放资源。
//
// 账本支持三种共享（切分）模式，是平台"提升 GPU 利用率"的核心能力：
//   - exclusive（整卡独占）：一块 GPU 同时只服务一个任务。
//   - memory（显存切片，近似 MPS）：多个任务按显存额度共享同一块 GPU。
//   - mig：将物理 GPU 切分为若干硬件隔离的 MIG 实例，任务占用一个实例。
package ledger

import (
	"sync"

	"qingtian/internal/domain"
	"qingtian/pkg/apperr"
)

// migAccount 记录单个 MIG 实例的占用情况。
type migAccount struct {
	id          string
	parentUUID  string
	memoryMB    int64
	allocatable bool
	used        bool // 是否已被占用
}

// gpuAccount 记录单块 GPU 的资源占用情况。
type gpuAccount struct {
	uuid        string
	vendor      domain.Vendor
	model       string
	memoryTotal int64
	memoryUsed  int64 // 已分配显存
	allocated   int   // 已分配任务数（用于判断整卡是否被占用）
	allocatable bool
	exclusive   bool          // 是否已被某任务整卡独占（独占期间禁止其它任务共享）
	migEnabled  bool          // 是否处于 MIG 切分模式
	migs        []*migAccount // MIG 实例列表
}

// available 返回该 GPU 当前可用显存。
func (a *gpuAccount) available() int64 {
	return a.memoryTotal - a.memoryUsed
}

// Ledger 是内存资源账本，维护所有节点上所有 GPU 的占用情况。
//
// 通过互斥锁保证分配/释放的原子性，从而在并发调度下避免超卖。
type Ledger struct {
	mu sync.Mutex
	// gpus: gpuUUID -> 账户
	gpus map[string]*gpuAccount
	// nodeGPUs: nodeID -> 该节点的 gpuUUID 列表
	nodeGPUs map[string][]string
	// migIndex: migInstanceID -> MIG 实例账户，便于按实例 ID 释放
	migIndex map[string]*migAccount
}

// New 创建一个空的资源账本。
func New() *Ledger {
	return &Ledger{
		gpus:     make(map[string]*gpuAccount),
		nodeGPUs: make(map[string][]string),
		migIndex: make(map[string]*migAccount),
	}
}

// SyncNode 将节点的 GPU 信息同步到账本。
//
// 该方法在节点注册或心跳时调用，用于登记 GPU 容量。
// 若 GPU 已存在，仅更新其容量与可分配标记，保留已分配用量（避免心跳覆盖导致账目丢失）。
//
// 参数：
//   - node: 节点对象，其 GPUs 字段提供 GPU 列表。
func (l *Ledger) SyncNode(node *domain.Node) {
	l.mu.Lock()
	defer l.mu.Unlock()
	uuids := make([]string, 0, len(node.GPUs))
	for _, g := range node.GPUs {
		uuids = append(uuids, g.UUID)
		if acc, ok := l.gpus[g.UUID]; ok {
			acc.memoryTotal = g.MemoryTotal
			acc.allocatable = g.Allocatable
			acc.model = g.Model
			acc.vendor = normalizeVendor(g.Vendor)
			l.syncMIG(acc, g)
		} else {
			acc := &gpuAccount{
				uuid:        g.UUID,
				vendor:      normalizeVendor(g.Vendor),
				model:       g.Model,
				memoryTotal: g.MemoryTotal,
				memoryUsed:  g.MemoryUsed,
				allocatable: g.Allocatable,
				migEnabled:  g.MIGEnabled,
			}
			l.syncMIG(acc, g)
			l.gpus[g.UUID] = acc
		}
	}
	l.nodeGPUs[node.ID] = uuids
}

// syncMIG 同步 GPU 的 MIG 实例列表到账本，保留已占用实例的占用状态。
func (l *Ledger) syncMIG(acc *gpuAccount, g domain.GPU) {
	acc.migEnabled = g.MIGEnabled
	if !g.MIGEnabled {
		return
	}
	// 保留既有实例的 used 状态，避免心跳覆盖导致账目丢失。
	prev := make(map[string]bool, len(acc.migs))
	for _, m := range acc.migs {
		prev[m.id] = m.used
	}
	acc.migs = acc.migs[:0]
	for _, mi := range g.MIGInstances {
		m := &migAccount{
			id:          mi.ID,
			parentUUID:  g.UUID,
			memoryMB:    mi.MemoryMB,
			allocatable: mi.Allocatable,
			used:        prev[mi.ID],
		}
		acc.migs = append(acc.migs, m)
		l.migIndex[mi.ID] = m
	}
}

// normalizeVendor 将空厂商归一为 NVIDIA（历史节点未上报厂商时的默认）。
func normalizeVendor(v domain.Vendor) domain.Vendor {
	if v == "" {
		return domain.VendorNVIDIA
	}
	return v
}

// Reset 清空账本的全部占用与登记信息，用于账本重建。
//
// 账本是进程内内存态的派生视图，调度器进程重启或周期性巡检时需要以
// 存储层（MongoDB 中的节点与活动任务）为真相源重建，避免"重启即认为
// 所有卡空闲"导致的超卖。重建流程为：Reset -> SyncNode 全量节点登记容量
// -> Occupy 重放全部活动任务的占用。
func (l *Ledger) Reset() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.gpus = make(map[string]*gpuAccount)
	l.nodeGPUs = make(map[string][]string)
	l.migIndex = make(map[string]*migAccount)
}

// Occupy 按已知的资源标识重放一次占用（账本重建时使用）。
//
// 与 TryAllocate 不同，Occupy 不做择优挑选，而是精确占用调用方给定的
// 资源标识（任务此前已分配到的 GPU UUID 或 MIG 实例 ID），从而把存储层
// 中活动任务的占用如实还原到账本，保证重建后账目与真实运行状态一致。
//
// 参数：
//   - ids:      任务已占用的资源标识列表。
//   - memoryMB: 每块 GPU 上占用的显存（MIG 模式忽略）。
//   - mode:     该任务的共享模式，用于正确设置独占标记。
func (l *Ledger) Occupy(ids []string, memoryMB int64, mode domain.ShareMode) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, id := range ids {
		// 优先按 MIG 实例占用。
		if m := l.migIndex[id]; m != nil {
			m.used = true
			continue
		}
		acc := l.gpus[id]
		if acc == nil {
			continue
		}
		acc.memoryUsed += memoryMB
		acc.allocated++
		if mode == domain.ShareExclusive {
			acc.exclusive = true
		}
	}
}

// CanAllocate 只读地判断某节点是否能满足任务资源需求（不修改账本）。
//
// 供调度器 Filter/Score 阶段使用：先只读判断可行性并打分，最终仅对选中的
// 节点做一次真实扣减，避免"对每个候选节点都试分配再回滚"带来的多次写与
// 全局锁竞争。
//
// 参数：
//   - nodeID: 目标节点 ID。
//   - req:    资源需求。
//
// 返回值：
//   - bool: true 表示该节点当前可满足需求。
func (l *Ledger) CanAllocate(nodeID string, req domain.ResourceRequest) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	uuids, ok := l.nodeGPUs[nodeID]
	if !ok {
		return false
	}
	if req.EffectiveShareMode() == domain.ShareMIG {
		fit := 0
		for _, u := range uuids {
			acc := l.gpus[u]
			if acc == nil || !acc.allocatable || !acc.migEnabled {
				continue
			}
			if req.GPUModel != "" && acc.model != req.GPUModel {
				continue
			}
			if req.Vendor != "" && acc.vendor != req.Vendor {
				continue
			}
			for _, m := range acc.migs {
				if !m.used && m.allocatable && m.memoryMB >= req.MemoryMB {
					fit++
				}
			}
		}
		return fit >= req.GPUCount
	}
	mode := req.EffectiveShareMode()
	fit := 0
	for _, u := range uuids {
		if l.gpuFits(l.gpus[u], req, mode) {
			fit++
		}
	}
	return fit >= req.GPUCount
}

// Candidate 表示某节点上满足需求的候选 GPU 分配方案。
//
// 字段：
//   - NodeID:    节点 ID。
//   - GPUUUIDs:  选中的 GPU UUID 列表（显存切片/整卡模式）。
//   - MIGIDs:    选中的 MIG 实例 ID 列表（MIG 模式）。
//   - ShareMode: 实际生效的共享模式。
type Candidate struct {
	NodeID    string
	GPUUUIDs  []string
	MIGIDs    []string
	ShareMode domain.ShareMode
}

// AllocatedIDs 返回本次分配占用的资源标识（供任务记录与释放使用）。
//
// MIG 模式返回 MIG 实例 ID，其余模式返回 GPU UUID。
func (c *Candidate) AllocatedIDs() []string {
	if c.ShareMode == domain.ShareMIG {
		return c.MIGIDs
	}
	return c.GPUUUIDs
}

// TryAllocate 尝试在指定节点上为任务分配资源。
//
// 分配逻辑按共享模式区分：
//   - exclusive：选出当前完全空闲（无任何占用）的整卡，标记独占。
//   - memory：   选出未被独占且剩余显存达标的卡，按显存额度共享。
//   - mig：      选出满足显存的空闲 MIG 实例。
//
// 型号与厂商约束在所有模式下均生效。全部满足则原子扣减并返回分配方案；
// 否则不做任何修改，保证"要么完整成功、要么完全不分配"的原子性。
//
// 参数：
//   - nodeID: 目标节点 ID。
//   - req:    资源需求。
//
// 返回值：
//   - *Candidate: 分配方案；分配失败时为 nil。
//   - error:      资源不足时返回 CodeExhausted 业务错误。
func (l *Ledger) TryAllocate(nodeID string, req domain.ResourceRequest) (*Candidate, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	uuids, ok := l.nodeGPUs[nodeID]
	if !ok {
		return nil, apperr.NotFound("节点不在账本中: " + nodeID)
	}

	if req.EffectiveShareMode() == domain.ShareMIG {
		return l.tryAllocateMIG(nodeID, uuids, req)
	}
	return l.tryAllocateGPU(nodeID, uuids, req)
}

// tryAllocateGPU 处理 exclusive 与 memory 两种整卡/显存切片模式的分配。
func (l *Ledger) tryAllocateGPU(nodeID string, uuids []string, req domain.ResourceRequest) (*Candidate, error) {
	mode := req.EffectiveShareMode()
	chosen := make([]string, 0, req.GPUCount)
	for _, u := range uuids {
		acc := l.gpus[u]
		if !l.gpuFits(acc, req, mode) {
			continue
		}
		chosen = append(chosen, u)
		if len(chosen) == req.GPUCount {
			break
		}
	}

	if len(chosen) < req.GPUCount {
		return nil, apperr.Exhausted("节点资源不足: " + nodeID)
	}

	// 全部满足，原子扣减显存与占用计数。
	for _, u := range chosen {
		acc := l.gpus[u]
		acc.memoryUsed += req.MemoryMB
		acc.allocated++
		if mode == domain.ShareExclusive {
			acc.exclusive = true
		}
	}
	return &Candidate{NodeID: nodeID, GPUUUIDs: chosen, ShareMode: mode}, nil
}

// gpuFits 判断某 GPU 是否满足在指定模式下的分配条件。
func (l *Ledger) gpuFits(acc *gpuAccount, req domain.ResourceRequest, mode domain.ShareMode) bool {
	if acc == nil || !acc.allocatable {
		return false
	}
	if req.GPUModel != "" && acc.model != req.GPUModel {
		return false
	}
	if req.Vendor != "" && acc.vendor != req.Vendor {
		return false
	}
	if acc.exclusive {
		return false // 已被独占，任何模式都不能再用
	}
	switch mode {
	case domain.ShareExclusive:
		// 整卡独占要求该卡当前完全空闲。
		return acc.allocated == 0 && acc.available() >= req.MemoryMB
	default: // ShareMemory
		return acc.available() >= req.MemoryMB
	}
}

// tryAllocateMIG 处理 MIG 模式的分配：为每个所需份数选出一个空闲 MIG 实例。
func (l *Ledger) tryAllocateMIG(nodeID string, uuids []string, req domain.ResourceRequest) (*Candidate, error) {
	chosen := make([]string, 0, req.GPUCount)
	for _, u := range uuids {
		acc := l.gpus[u]
		if acc == nil || !acc.allocatable || !acc.migEnabled {
			continue
		}
		if req.GPUModel != "" && acc.model != req.GPUModel {
			continue
		}
		if req.Vendor != "" && acc.vendor != req.Vendor {
			continue
		}
		for _, m := range acc.migs {
			if !m.used && m.allocatable && m.memoryMB >= req.MemoryMB {
				chosen = append(chosen, m.id)
				m.used = true // 先占用，若最终数量不足会回滚
				if len(chosen) == req.GPUCount {
					return &Candidate{NodeID: nodeID, MIGIDs: chosen, ShareMode: domain.ShareMIG}, nil
				}
			}
		}
	}
	// 数量不足，回滚已试占的实例。
	for _, id := range chosen {
		if m := l.migIndex[id]; m != nil {
			m.used = false
		}
	}
	return nil, apperr.Exhausted("节点 MIG 实例不足: " + nodeID)
}

// Release 释放此前分配的资源。
//
// 该方法在任务结束、取消或调度失败回滚时调用，将显存/实例占用归还。
// 兼容三种模式：传入的 ids 既可以是 GPU UUID，也可以是 MIG 实例 ID。
//
// 参数：
//   - ids:      待释放的资源标识列表（GPU UUID 或 MIG 实例 ID）。
//   - memoryMB: 每块 GPU 上占用的显存（应与分配时一致；MIG 模式忽略）。
func (l *Ledger) Release(ids []string, memoryMB int64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, id := range ids {
		// 优先按 MIG 实例释放。
		if m := l.migIndex[id]; m != nil {
			m.used = false
			continue
		}
		acc := l.gpus[id]
		if acc == nil {
			continue
		}
		acc.memoryUsed -= memoryMB
		if acc.memoryUsed < 0 {
			acc.memoryUsed = 0
		}
		if acc.allocated > 0 {
			acc.allocated--
		}
		if acc.allocated == 0 {
			acc.exclusive = false // 最后一个占用释放后解除独占
		}
	}
}

// NodeFreeMemory 返回指定节点当前可用显存总和，供调度器打分使用。
//
// 参数：
//   - nodeID: 节点 ID。
//
// 返回值：
//   - int64: 该节点所有可分配 GPU 的剩余显存之和。
func (l *Ledger) NodeFreeMemory(nodeID string) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	var sum int64
	for _, u := range l.nodeGPUs[nodeID] {
		if acc := l.gpus[u]; acc != nil && acc.allocatable {
			sum += acc.available()
		}
	}
	return sum
}

// NodeUtilization 返回指定节点的显存分配率（0-1），用于空闲检测与看板。
//
// 参数：
//   - nodeID: 节点 ID。
//
// 返回值：
//   - float64: 已分配显存 / 总显存；无可分配卡时返回 0。
func (l *Ledger) NodeUtilization(nodeID string) float64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	var total, used int64
	for _, u := range l.nodeGPUs[nodeID] {
		if acc := l.gpus[u]; acc != nil && acc.allocatable {
			total += acc.memoryTotal
			used += acc.memoryUsed
		}
	}
	if total == 0 {
		return 0
	}
	return float64(used) / float64(total)
}
