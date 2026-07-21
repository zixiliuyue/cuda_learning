// Package scheduler 实现任务调度的核心算法。
//
// This file implements consistent hashing sharding for scheduling horizontal scaling.
package scheduler

import (
	"hash/fnv"
	"strconv"
)

// Sharder implements consistent hashing for node sharding.
//
// Consistent hashing minimizes the reshuffling when nodes are added or removed,
// which keeps most assignments unchanged and maintains stable scheduling.
type Sharder struct {
	// virtual replicas per physical node for better distribution
	replicas int
	// circle is the sorted hash circle: hash -> node ID
	circle map[uint32]string
	// sorted hashes for binary search
	sorted []uint32
}

// NewSharder creates a new consistent hashing sharder.
//
// replicas: number of virtual replicas per node, typically 100-200.
func NewSharder(replicas int) *Sharder {
	if replicas <= 0 {
		replicas = 150 // default
	}
	return &Sharder{
		replicas: replicas,
		circle:   make(map[uint32]string),
		sorted:   make([]uint32, 0),
	}
}

// AddNode adds a node to the hash circle.
func (s *Sharder) AddNode(nodeID string) {
	for i := 0; i < s.replicas; i++ {
		hash := s.hash(nodeID + strconv.Itoa(i))
		s.circle[hash] = nodeID
		s.sorted = append(s.sorted, hash)
	}
	// sort keeps sorted for binary search
	// sort is incremental - we don't sort every time, but when nodes change is infrequent
	// it's acceptable to re-sort the entire list
	sorted := make([]uint32, 0, len(s.sorted))
	for h := range s.circle {
		sorted = append(sorted, h)
	}
	// sort in increasing order
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	s.sorted = sorted
}

// RemoveNode removes a node from the hash circle.
func (s *Sharder) RemoveNode(nodeID string) {
	for i := 0; i < s.replicas; i++ {
		hash := s.hash(nodeID + strconv.Itoa(i))
		delete(s.circle, hash)
	}
	// rebuild sorted list
	sorted := make([]uint32, 0, len(s.circle))
	for h := range s.circle {
		sorted = append(sorted, h)
	}
	// sort
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	s.sorted = sorted
}

// GetShard maps a task to the shard (node) that should schedule it.
//
// For scheduling horizontal scaling, we map the task to a shard based on task ID,
// and the shard is selected from available nodes via consistent hashing.
//
// Actually, in the multi-scheduler sharding design, each scheduler owns multiple nodes
// and the task is routed to the scheduler by consistent hashing of the task ID.
// This method returns the node that "owns" the task, which maps to which scheduler
// should handle it (since each scheduler only has its shard nodes).
func (s *Sharder) GetShard(taskID string) string {
	if len(s.sorted) == 0 {
		return ""
	}
	hash := s.hash(taskID)
	// binary search for the first hash >= task hash
	idx := search(s.sorted, hash)
	if idx == len(s.sorted) {
		idx = 0 // wrap around
	}
	return s.circle[s.sorted[idx]]
}

// hash computes the 32-bit hash of a string.
func (s *Sharder) hash(key string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(key))
	return h.Sum32()
}

// search finds the first element >= target using binary search.
func search(sorted []uint32, target uint32) int {
	low := 0
	high := len(sorted)
	for low < high {
		mid := (low + high) / 2
		if sorted[mid] < target {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}

// ShardingConfig configures the sharding behavior.
type ShardingConfig struct {
	// Enabled enables sharding horizontal scaling.
	Enabled bool `json:"enabled" yaml:"enabled"`
	// Replicas is the number of virtual replicas per node in consistent hashing.
	Replicas int `json:"replicas" yaml:"replicas"`
	// ShardCount is the number of shards (if using fixed shard counting instead of per-node).
	// When ShardCount > 0, we fix N shards instead of sharding by node.
	ShardCount int `json:"shard_count" yaml:"shard_count"`
}

// DefaultShardingConfig returns the default sharding config.
func DefaultShardingConfig() ShardingConfig {
	return ShardingConfig{
		Enabled:  false,
		Replicas: 150,
		ShardCount: 0,
	}
}
