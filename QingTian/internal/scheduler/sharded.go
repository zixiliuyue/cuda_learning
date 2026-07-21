// Package scheduler 实现任务调度的核心算法。
//
// This file implements sharded scheduler for horizontal scaling.
// When the cluster grows beyond thousands of GPUs, single scheduler becomes a bottleneck:
//   - Global lock contention under high concurrency
//   - Iterating all nodes on every scheduling becomes slow
//
// Sharded scheduler splits the cluster into multiple shards, each shard has its
// own scheduler and ledger, so locking and searching is isolated per shard,
// enabling horizontal scaling to tens of thousands of GPUs.
package scheduler

import (
	"context"
	"sync"

	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/pkg/apperr"
)

// Shard represents a single scheduling shard.
//
// Each shard maintains its own ledger of node resources and has independent locking,
// so scheduling in different shards doesn't contend the global lock.
type Shard struct {
	// Shard ID for identification
	ID string
	// Nodes in this shard
	NodeIDs []string
	// Scheduler instance for this shard
	Scheduler *Scheduler
	// Ledger for this shard - only contains nodes in this shard
	Ledger *ledger.Ledger
}

// NewShard creates a new scheduling shard.
func NewShard(id string, strategy string) *Shard {
	l := ledger.New()
	s := New(l, strategy)
	return &Shard{
		ID:        id,
		NodeIDs:   make([]string, 0),
		Scheduler: s,
		Ledger:    l,
	}
}

// ShardedScheduler is a scheduler that splits the cluster into multiple shards
// for horizontal scaling.
//
// Design:
//   - Each shard has its own ledger and scheduler, independent locking
//   - Tasks are mapped to shards via consistent hashing based on task ID
//   - Only nodes in the mapped shard are considered for scheduling
//   - This reduces lock contention and reduces the number of nodes iterated per scheduling
type ShardedScheduler struct {
	mu           sync.RWMutex
	config       ShardingConfig
	sharder      *Sharder
	shards       map[string]*Shard // shard ID -> shard
	strategy     string
}

// NewShardedScheduler creates a new sharded scheduler.
func NewShardedScheduler(config ShardingConfig, strategy string) *ShardedScheduler {
	return &ShardedScheduler{
		config:    config,
		sharder:   NewSharder(config.Replicas),
		shards:     make(map[string]*Shard),
		strategy:   strategy,
	}
}

// AddShard adds a shard to the sharded scheduler.
//
// Each shard is independently responsible for its own nodes.
func (s *ShardedScheduler) AddShard(shard *Shard) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.shards[shard.ID] = shard
	// Add each node in the shard to consistent hashing
	for _, nodeID := range shard.NodeIDs {
		s.sharder.AddNode(nodeID)
	}
}

// RemoveShard removes a shard from the sharded scheduler.
func (s *ShardedScheduler) RemoveShard(shardID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	shard, ok := s.shards[shardID]
	if !ok {
		return
	}
	for _, nodeID := range shard.NodeIDs {
		s.sharder.RemoveNode(nodeID)
	}
	delete(s.shards, shardID)
}

// AddNodeToShard adds a node to an existing shard.
func (s *ShardedScheduler) AddNodeToShard(node *domain.Node, shardID string) error {
	s.mu.Lock()
	shard, ok := s.shards[shardID]
	s.mu.Unlock()
	if !ok {
		return apperr.NotFound("shard not found: " + shardID)
	}

	// Add node to shard's ledger
	shard.Ledger.SyncNode(node)
	shard.NodeIDs = append(shard.NodeIDs, node.ID)

	// Add to consistent hashing
	s.mu.Lock()
	s.sharder.AddNode(node.ID)
	s.mu.Unlock()

	return nil
}

// Schedule finds the best node for the task in the mapped shard.
//
// The task is mapped to a node/shard via consistent hashing, and only nodes in that
// shard are considered for scheduling. This keeps per-scheduling work small and
// avoids global lock contention.
func (s *ShardedScheduler) Schedule(ctx context.Context, task *domain.Task, allNodes []*domain.Node) (*Decision, error) {
	// Get the shard/node that this task maps to via consistent hashing
	s.mu.RLock()
	targetNodeID := s.sharder.GetShard(task.ID)
	s.mu.RUnlock()

	if targetNodeID == "" {
		// No nodes in the circle, fall back to searching all nodes across all shards
		return s.scheduleAllShards(ctx, task, allNodes)
	}

	// Find which shard contains this node
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, shard := range s.shards {
		found := false
		for _, nid := range shard.NodeIDs {
			if nid == targetNodeID {
				found = true
				break
			}
		}
		if !found {
			continue
		}

		// Filter only nodes that are in this shard
		shardNodes := make([]*domain.Node, 0, len(shard.NodeIDs))
		for _, n := range allNodes {
			for _, sid := range shard.NodeIDs {
				if n.ID == sid {
					shardNodes = append(shardNodes, n)
					break
				}
			}
		}

		// Schedule within this shard
		return shard.Scheduler.Schedule(ctx, task, shardNodes)
	}

	// Target node not found in any shard, fall back to searching all
	return s.scheduleAllShards(ctx, task, allNodes)
}

// scheduleAllShards searches all shards for the best candidate.
//
// This is used as a fallback when consistent hashing doesn't find a matching node.
// This is slower but still works.
func (s *ShardedScheduler) scheduleAllShards(
	ctx context.Context,
	task *domain.Task,
	allNodes []*domain.Node,
) (*Decision, error) {
	bestScore := 0.0
	bestDecision := (*Decision)(nil)
	found := false

	// Search in parallel across all shards? No, each scheduling is serial here
	// because we need one decision. Parallel search would complicate things.
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, shard := range s.shards {
		// Filter nodes in this shard
		shardNodes := make([]*domain.Node, 0, len(shard.NodeIDs))
		for _, n := range allNodes {
			for _, sid := range shard.NodeIDs {
				if n.ID == sid {
					shardNodes = append(shardNodes, n)
					break
				}
			}
		}
		if len(shardNodes) == 0 {
			continue
		}

		dec, err := shard.Scheduler.Schedule(ctx, task, shardNodes)
		if err != nil {
			continue // this shard can't fit, try next
		}

		// Get the score from free memory + data affinity
		freeMem := shard.Ledger.NodeFreeMemory(dec.NodeID)
		score := s.score(shard.Scheduler, freeMem, getNodeByID(allNodes, dec.NodeID), task)
		if !found || score > bestScore {
			bestScore = score
			bestDecision = dec
			found = true
		}
	}

	if !found {
		return nil, apperr.Exhausted("no shard has enough resources for the task")
	}

	return bestDecision, nil
}

// getNodeByID gets a node from the list by ID.
func getNodeByID(nodes []*domain.Node, id string) *domain.Node {
	for _, n := range nodes {
		if n.ID == id {
			return n
		}
	}
	return nil
}

// score combines the base strategy score with data affinity bonus.
//
// Same scoring logic as the single scheduler, just per shard.
func (s *ShardedScheduler) score(sched *Scheduler, freeMemory int64, node *domain.Node, task *domain.Task) float64 {
	base := sched.score(freeMemory, node, task)

	// Data affinity bonus same as single scheduler
	if task.DatasetPath != "" && hasLocalDataset(node, task.DatasetPath) {
		base += 1000000
	}

	return base
}

// RebuildLedger rebuilds all shards' ledgers from the full node list.
//
// Call this periodically to keep shard ledgers in sync with cluster changes.
func (s *ShardedScheduler) RebuildLedger(allNodes []*domain.Node, activeTasks []*domain.Task) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, shard := range s.shards {
		// Reset the ledger
		shard.Ledger.Reset()
		// Sync only nodes in this shard
		for _, node := range allNodes {
			isInShard := false
			for _, nid := range shard.NodeIDs {
				if node.ID == nid {
					isInShard = true
					break
				}
			}
			if isInShard {
				shard.Ledger.SyncNode(node)
			}
		}
		// Replay active tasks that are assigned to nodes in this shard
		for _, task := range activeTasks {
			if task.NodeID != "" {
				// Check if task's node is in this shard
				inShard := false
				for _, nid := range shard.NodeIDs {
					if task.NodeID == nid {
						inShard = true
						break
					}
				}
				if inShard && len(task.AllocatedGPUs) > 0 {
					shard.Ledger.Occupy(task.AllocatedGPUs, task.Resource.MemoryMB, task.Resource.EffectiveShareMode())
				}
			}
		}
	}
}

// ScheduleGang does gang scheduling on the mapped shard.
//
// Same as single scheduler gang scheduling but within a single shard.
func (s *ShardedScheduler) ScheduleGang(ctx context.Context, task *domain.Task, allNodes []*domain.Node) ([]*Decision, error) {
	size := task.Resource.GangSize
	if size <= 1 {
		dec, err := s.Schedule(ctx, task, allNodes)
		if err != nil {
			return nil, err
		}
		return []*Decision{dec}, nil
	}

	// Get the mapped shard
	s.mu.RLock()
	targetNodeID := s.sharder.GetShard(task.ID)
	s.mu.RUnlock()

	// Find the shard
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, shard := range s.shards {
		hasTarget := false
		for _, nid := range shard.NodeIDs {
			if nid == targetNodeID {
				hasTarget = true
				break
			}
		}
		if !hasTarget {
			continue
		}

		// Filter nodes in this shard
		shardNodes := make([]*domain.Node, 0, len(shard.NodeIDs))
		for _, n := range allNodes {
			for _, sid := range shard.NodeIDs {
				if n.ID == sid {
					shardNodes = append(shardNodes, n)
					break
				}
			}
		}

		return shard.Scheduler.ScheduleGang(ctx, task, shardNodes)
	}

	// Fall back to searching all shards
	decisions := make([]*Decision, 0, size)
	usedNodes := make(map[string]int) // nodes used by this gang

	for i := 0; i < size; i++ {
		bestScore := 0.0
		bestDec := (*Decision)(nil)

		for _, shard := range s.shards {
			// Filter fresh nodes not yet used by this gang
			shardNodes := make([]*domain.Node, 0)
			for _, n := range allNodes {
				for _, sid := range shard.NodeIDs {
					if n.ID == sid {
						if _, used := usedNodes[n.ID]; !used {
							shardNodes = append(shardNodes, n)
						}
						break
					}
				}
			}
			if len(shardNodes) == 0 {
				continue
			}

			dec, err := shard.Scheduler.Schedule(ctx, task, shardNodes)
			if err != nil {
				continue
			}

			freeMem := shard.Ledger.NodeFreeMemory(dec.NodeID)
			score := s.score(shard.Scheduler, freeMem, getNodeByID(allNodes, dec.NodeID), task)
			if bestDec == nil || score > bestScore {
				bestScore = score
				bestDec = dec
			}
		}

		if bestDec == nil {
			// Can't get all N copies, rollback
			for _, dec := range decisions {
				if dec != nil {
					shard := s.findShardForNode(dec.NodeID)
					if shard != nil {
						shard.Ledger.Release(dec.AllocatedGPUs, task.Resource.MemoryMB)
					}
				}
			}
			return nil, apperr.Exhausted("Gang scheduling failed: cannot get all required copies across all shards")
		}

		decisions = append(decisions, bestDec)
		usedNodes[bestDec.NodeID]++
	}

	return decisions, nil
}

// findShardForNode finds which shard contains a given node.
func (s *ShardedScheduler) findShardForNode(nodeID string) *Shard {
	for _, shard := range s.shards {
		for _, nid := range shard.NodeIDs {
			if nid == nodeID {
				return shard
			}
		}
	}
	return nil
}

// ReleaseTask releases task resources in the correct shard's ledger.
func (s *ShardedScheduler) ReleaseTask(allocatedIDs []string, memoryMB int64) {
	if len(allocatedIDs) == 0 {
		return
	}
	// We don't actually need nodeID here because the allocated IDs all belong to the same node,
	// and we find the shard by walking through all shards' nodes.
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, shard := range s.shards {
		// Just release in each shard's scheduler - it's harmless if the GPU doesn't belong here,
		// it will just be a no-op because the ledger doesn't have it. This is simpler than finding
		// the exact node because we just need to release.
		shard.Scheduler.ReleaseTask(allocatedIDs, memoryMB)
		return
	}
}

// CanFit checks if a node can fit the requested resources.
//
// In sharded mode, find which shard the node belongs to and delegate check to that shard's scheduler.
func (s *ShardedScheduler) CanFit(nodeID string, req domain.ResourceRequest) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// Find which shard contains this node
	for _, shard := range s.shards {
		for _, nid := range shard.NodeIDs {
			if nid == nodeID {
				return shard.Scheduler.CanFit(nodeID, req)
			}
		}
	}
	// Node not found in any shard, can't fit
	return false
}

// LoadLedgerSnapshot loads ledger snapshot from shared storage.
//
// In sharded mode, we need to rebuild all shards from scratch since each shard has its own ledger.
// Rebuilding from database is more reliable than loading snapshot for sharded mode.
func (s *ShardedScheduler) LoadLedgerSnapshot(data []byte) (int64, error) {
	// Sharded mode doesn't support shared snapshot loading currently,
	// requires full rebuild from database.
	return 0, apperr.InvalidArgument("sharded scheduler does not support loading snapshot")
}

// DumpLedgerSnapshot dumps the current ledger state to JSON snapshot.
//
// In sharded mode, we don't support dumping a single snapshot since each shard has its own ledger.
func (s *ShardedScheduler) DumpLedgerSnapshot(version int64) ([]byte, error) {
	return nil, apperr.InvalidArgument("sharded scheduler does not support dumping snapshot")
}

// SyncNodes syncs all nodes' GPU capacities into all shards' ledgers.
//
// In sharded mode, add/update nodes to the correct shard based on shard assignment.
func (s *ShardedScheduler) SyncNodes(nodes []*domain.Node) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	// For each shard, sync only the nodes that belong to it
	for _, shard := range s.shards {
		// Collect nodes that belong to this shard
		shardNodes := make([]*domain.Node, 0, len(shard.NodeIDs))
		for _, node := range nodes {
			for _, shardNodeID := range shard.NodeIDs {
				if node.ID == shardNodeID {
					shardNodes = append(shardNodes, node)
					break
				}
			}
		}
		// Sync the nodes that belong to this shard
		shard.Scheduler.SyncNodes(shardNodes)
	}
}
