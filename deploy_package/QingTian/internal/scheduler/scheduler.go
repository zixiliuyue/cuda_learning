// Package scheduler 实现任务调度的核心算法。
//
// 调度采用两阶段模型：
//   - Filter（过滤）：筛选出满足任务资源约束且在线的节点。
//   - Score（打分）：对候选节点打分，选出最优节点。
//
// 调度器不直接依赖存储/消息实现，而是通过注入的账本与仓储接口工作，
// 便于单元测试。
package scheduler

import (
	"context"

	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/pkg/apperr"
)

// Strategy 表示打分策略。
type Strategy string

const (
	// StrategyBinPack 装箱策略：优先选择剩余资源少的节点，减少碎片，提高整卡利用率。
	StrategyBinPack Strategy = "binpack"
	// StrategySpread 分散策略：优先选择剩余资源多的节点，均衡负载，降低单点风险。
	StrategySpread Strategy = "spread"
)

// Decision 表示一次调度决策的结果。
type Decision struct {
	NodeID        string           // 选中的节点 ID
	AllocatedGPUs []string         // 分配到的资源标识（GPU UUID 或 MIG 实例 ID）
	ShareMode     domain.ShareMode // 实际生效的共享模式
}

// Scheduler 是调度器核心。
//
// 字段：
//   - ledger:   资源账本，用于试分配与打分。
//   - strategy: 打分策略。
type Scheduler struct {
	ledger   *ledger.Ledger
	strategy Strategy
}

// New 创建调度器。
//
// 参数：
//   - l:        资源账本。
//   - strategy: 打分策略字符串；无法识别时默认为 binpack。
func New(l *ledger.Ledger, strategy string) *Scheduler {
	s := Strategy(strategy)
	if s != StrategyBinPack && s != StrategySpread {
		s = StrategyBinPack
	}
	return &Scheduler{ledger: l, strategy: s}
}

// Schedule 为任务选择最优节点并完成资源分配。
//
// 算法步骤：
//  1. Filter：遍历在线节点，尝试通过账本试分配，能分配的即为候选。
//  2. Score：对候选节点按策略打分。
//  3. 选出最高分节点，其余候选的试分配结果回滚（释放）。
//
// 参数：
//   - ctx:   上下文。
//   - task:  待调度任务。
//   - nodes: 当前所有节点列表（由调用方从仓储获取）。
//
// 返回值：
//   - *Decision: 调度决策；无可用节点时为 nil。
//   - error:     无满足条件的节点时返回 CodeExhausted。
//
// 注意事项：
//   - 成功返回时，选中节点的资源已在账本中扣减；调用方若后续下发失败，需调用账本 Release 回滚。
func (s *Scheduler) Schedule(_ context.Context, task *domain.Task, nodes []*domain.Node) (*Decision, error) {
	// Filter + Score：只读判断可行性并按剩余显存打分，不修改账本。
	// 由于同一任务对各候选节点的资源需求相同，用"当前剩余显存"打分与
	// 用"分配后剩余显存"打分的排序完全一致，故可只读打分。
	bestNode := ""
	bestScore := 0.0
	found := false
	for _, n := range nodes {
		if n.Status != domain.NodeOnline {
			continue
		}
		if !s.ledger.CanAllocate(n.ID, task.Resource) {
			continue // 该节点资源不足，跳过
		}
		score := s.score(s.ledger.NodeFreeMemory(n.ID))
		if !found || score > bestScore {
			found = true
			bestScore = score
			bestNode = n.ID
		}
	}

	if !found {
		return nil, apperr.Exhausted("没有满足任务资源需求的可用节点")
	}

	// 仅对选中的最优节点做一次真实扣减，避免多次写+回滚。
	alloc, err := s.ledger.TryAllocate(bestNode, task.Resource)
	if err != nil {
		// 打分与扣减之间资源可能被并发占用，视为本轮无可用节点。
		return nil, apperr.Exhausted("选中节点资源已被占用，稍后重试")
	}

	return &Decision{
		NodeID:        bestNode,
		AllocatedGPUs: alloc.AllocatedIDs(),
		ShareMode:     alloc.ShareMode,
	}, nil
}

// RebuildLedger 以存储层为真相源重建资源账本。
//
// 调度器进程重启或周期性巡检时调用：先清空账本，再登记全部节点的 GPU 容量，
// 最后重放全部活动任务（scheduled/running）此前分配到的资源占用。这样可避免
// "进程重启即认为所有卡空闲"造成的超卖，并让运行期新增/变更的节点进入账本。
//
// 参数：
//   - nodes:       当前全部节点（含 GPU 容量）。
//   - activeTasks: 当前处于活动态、仍占用资源的任务。
func (s *Scheduler) RebuildLedger(nodes []*domain.Node, activeTasks []*domain.Task) {
	s.ledger.Reset()
	for _, n := range nodes {
		s.ledger.SyncNode(n)
	}
	for _, t := range activeTasks {
		if len(t.AllocatedGPUs) == 0 {
			continue
		}
		s.ledger.Occupy(t.AllocatedGPUs, t.Resource.MemoryMB, t.Resource.EffectiveShareMode())
	}
}

// LoadLedgerSnapshot 从共享存储的 JSON 快照恢复账本状态，返回快照中的版本号。
//
// 用于多调度器实例共享同一份占用视图：本实例成为 leader 后先加载最新快照，
// 再在其上做调度决策，决策完成后调用 DumpLedgerSnapshot 写回，实现跨实例一致。
func (s *Scheduler) LoadLedgerSnapshot(data []byte) (int64, error) {
	return s.ledger.LoadSnapshot(data)
}

// DumpLedgerSnapshot 将当前账本状态序列化为 JSON 快照（附带指定版本号），供写回共享存储。
func (s *Scheduler) DumpLedgerSnapshot(version int64) ([]byte, error) {
	return s.ledger.MarshalSnapshot(version)
}

// SyncNodes 将全部节点的 GPU 容量登记进账本（保留既有占用）。
//
// 与 RebuildLedger 不同，SyncNodes 不清空占用，仅登记/刷新容量，用于在
// 已加载共享快照的基础上纳入运行期新增或变更的节点。
func (s *Scheduler) SyncNodes(nodes []*domain.Node) {
	for _, n := range nodes {
		s.ledger.SyncNode(n)
	}
}

// ReleaseTask 归还一个任务已占用的资源到账本。
//
// 该方法在任务被超时终止、抢占或异常清理时调用，把此前 Schedule/ScheduleGang
// 扣减的显存与实例占用释放回账本，避免"任务已死、资源仍被占用"的泄漏。
//
// 参数：
//   - allocatedIDs: 任务分配到的资源标识（GPU UUID 或 MIG 实例 ID）。
//   - memoryMB:     每块 GPU 上占用的显存（应与分配时一致）。
func (s *Scheduler) ReleaseTask(allocatedIDs []string, memoryMB int64) {
	if len(allocatedIDs) == 0 {
		return
	}
	s.ledger.Release(allocatedIDs, memoryMB)
}

// CanFit 只读地判断某节点当前是否能容纳指定资源需求（不修改账本）。
//
// 供抢占逻辑在归还资源后判断"是否已腾出足够资源"，避免盲抢。
func (s *Scheduler) CanFit(nodeID string, req domain.ResourceRequest) bool {
	return s.ledger.CanAllocate(nodeID, req)
}

// score 根据策略将节点剩余显存转换为分值。
//
// 参数：
//   - freeMemory: 分配后节点剩余显存（MB）。
//
// 返回值：
//   - float64: 分值，越大越优先。
//
// 策略说明：
//   - BinPack：剩余越少分越高（取负值），从而优先填满节点、减少碎片。
//   - Spread： 剩余越多分越高，从而均衡负载。
func (s *Scheduler) score(freeMemory int64) float64 {
	switch s.strategy {
	case StrategySpread:
		return float64(freeMemory)
	default: // BinPack
		return -float64(freeMemory)
	}
}

// ScheduleGang 为分布式训练任务做成组（Gang）调度。
//
// Gang Scheduling 的语义是"全部成功或全部失败"（All-or-Nothing）：
// 分布式训练必须一次性拿到所需的全部份数，否则已分到的资源会白白占用、
// 甚至造成多任务互相死锁。该方法把任务拆成 GangSize 份分别调度，
// 只要有一份分不到，就回滚全部，避免资源碎片与死锁。
//
// 参数：
//   - ctx:   上下文。
//   - task:  待调度任务，其 Resource.GangSize 指定所需份数。
//   - nodes: 当前所有节点列表。
//
// 返回值：
//   - []*Decision: 成组决策列表，长度等于 GangSize。
//   - error:       任一份无法分配时回滚全部并返回 CodeExhausted。
//
// 注意事项：
//   - GangSize <= 1 时退化为普通单次调度。
func (s *Scheduler) ScheduleGang(ctx context.Context, task *domain.Task, nodes []*domain.Node) ([]*Decision, error) {
	size := task.Resource.GangSize
	if size <= 1 {
		d, err := s.Schedule(ctx, task, nodes)
		if err != nil {
			return nil, err
		}
		return []*Decision{d}, nil
	}

	decisions := make([]*Decision, 0, size)
	usedNodes := make(map[string]int) // 记录各节点已承载的份数，用于拓扑分散
	for i := 0; i < size; i++ {
		// 拓扑感知：优先把每一份分散到尚未使用的节点，降低单点故障对分布式
		// 训练的影响；仅当无空闲新节点可用时，才回退到已承载份数的节点。
		d, err := s.scheduleGangMember(ctx, task, nodes, usedNodes)
		if err != nil {
			// 有一份分不到：回滚已成功的份，保证 All-or-Nothing。
			for _, done := range decisions {
				s.ledger.Release(done.AllocatedGPUs, task.Resource.MemoryMB)
			}
			return nil, apperr.Exhausted("Gang 调度失败：无法一次性满足全部份数")
		}
		usedNodes[d.NodeID]++
		decisions = append(decisions, d)
	}
	return decisions, nil
}

// scheduleGangMember 为成组任务的单份选节点，优先分散到未使用的节点。
//
// 先在"尚未被本组占用的节点"集合中调度以实现拓扑分散；若该集合无可用节点，
// 再回退到全部节点，保证在小集群/单节点场景下仍能凑齐份数。
func (s *Scheduler) scheduleGangMember(ctx context.Context, task *domain.Task, nodes []*domain.Node, usedNodes map[string]int) (*Decision, error) {
	if len(usedNodes) > 0 {
		fresh := make([]*domain.Node, 0, len(nodes))
		for _, n := range nodes {
			if _, ok := usedNodes[n.ID]; !ok {
				fresh = append(fresh, n)
			}
		}
		if len(fresh) > 0 {
			if d, err := s.Schedule(ctx, task, fresh); err == nil {
				return d, nil
			}
		}
	}
	return s.Schedule(ctx, task, nodes)
}
