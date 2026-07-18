// 本文件实现节点服务：节点注册、心跳、查询与离线检测。
package service

import (
	"context"
	"time"

	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
)

// heartbeatKeyPrefix 是节点心跳在缓存中的键前缀。
const heartbeatKeyPrefix = "heartbeat:"

// NodeService 提供节点纳管能力。
//
// 字段：
//   - repo:        节点仓储。
//   - cache:       缓存，用于存储带 TTL 的心跳。
//   - ledger:      资源账本，节点上报 GPU 时同步容量。
//   - nodeTimeout: 心跳超时时间。
type NodeService struct {
	repo        repository.NodeRepository
	cache       cache.Cache
	ledger      *ledger.Ledger
	nodeTimeout time.Duration
	now         func() time.Time
}

// NewNodeService 创建节点服务。
func NewNodeService(repo repository.NodeRepository, c cache.Cache, l *ledger.Ledger, nodeTimeout time.Duration) *NodeService {
	return &NodeService{repo: repo, cache: c, ledger: l, nodeTimeout: nodeTimeout, now: time.Now}
}

// RegisterRequest 是节点注册请求。
type RegisterRequest struct {
	Hostname      string       `json:"hostname"`
	IP            string       `json:"ip"`
	GPUs          []domain.GPU `json:"gpus"`
	DriverVersion string       `json:"driver_version"`
	CUDAVersion   string       `json:"cuda_version"`
}

// Register 注册（或重新注册）一个节点。
//
// 参数：
//   - ctx: 上下文。
//   - req: 注册请求，包含主机信息与 GPU 列表。
//
// 返回值：
//   - *domain.Node: 注册成功的节点（含生成的 ID）。
//   - error:        持久化失败时返回。
//
// 注意事项：
//   - 节点 ID 基于主机名生成，保证同一主机重启后仍是同一节点。
func (s *NodeService) Register(ctx context.Context, req RegisterRequest) (*domain.Node, error) {
	if req.Hostname == "" {
		return nil, apperr.InvalidArgument("主机名不能为空")
	}
	now := s.now()
	node := &domain.Node{
		ID:            "node-" + req.Hostname,
		Hostname:      req.Hostname,
		IP:            req.IP,
		Status:        domain.NodeOnline,
		GPUs:          req.GPUs,
		DriverVersion: req.DriverVersion,
		CUDAVersion:   req.CUDAVersion,
		LastHeartbeat: now,
		RegisteredAt:  now,
	}
	if err := s.repo.Upsert(ctx, node); err != nil {
		return nil, err
	}
	// 同步 GPU 容量到账本。
	s.ledger.SyncNode(node)
	// 写入心跳（带 TTL）。
	_ = s.writeHeartbeat(ctx, node.ID)
	return node, nil
}

// Heartbeat 处理节点心跳，刷新在线状态与实时 GPU 信息。
//
// 参数：
//   - ctx:    上下文。
//   - nodeID: 节点 ID。
//   - gpus:   实时 GPU 信息（利用率、显存等）。
func (s *NodeService) Heartbeat(ctx context.Context, nodeID string, gpus []domain.GPU) error {
	if err := s.repo.UpdateHeartbeat(ctx, nodeID, gpus); err != nil {
		return err
	}
	if len(gpus) > 0 {
		// 心跳携带的 GPU 信息用于账本容量刷新。
		node, err := s.repo.GetByID(ctx, nodeID)
		if err == nil {
			s.ledger.SyncNode(node)
		}
	}
	return s.writeHeartbeat(ctx, nodeID)
}

// writeHeartbeat 在缓存中写入带 TTL 的心跳标记。
func (s *NodeService) writeHeartbeat(ctx context.Context, nodeID string) error {
	return s.cache.Set(ctx, heartbeatKeyPrefix+nodeID, s.now().Format(time.RFC3339Nano), s.nodeTimeout)
}

// IsAlive 判断节点心跳是否仍在有效期内。
//
// 参数：
//   - ctx:    上下文。
//   - nodeID: 节点 ID。
//
// 返回值：
//   - bool: true 表示心跳有效（在线）。
func (s *NodeService) IsAlive(ctx context.Context, nodeID string) (bool, error) {
	_, ok, err := s.cache.Get(ctx, heartbeatKeyPrefix+nodeID)
	return ok, err
}

// Get 按 ID 查询节点。
func (s *NodeService) Get(ctx context.Context, id string) (*domain.Node, error) {
	return s.repo.GetByID(ctx, id)
}

// List 返回全部节点，并结合心跳修正在线状态。
func (s *NodeService) List(ctx context.Context) ([]*domain.Node, error) {
	nodes, err := s.repo.List(ctx)
	if err != nil {
		return nil, err
	}
	for _, n := range nodes {
		alive, _ := s.IsAlive(ctx, n.ID)
		if !alive {
			n.Status = domain.NodeOffline
		}
	}
	return nodes, nil
}

// SetOffline 将节点标记为离线（如运维手动下线或心跳超时检测）。
func (s *NodeService) SetOffline(ctx context.Context, id string) error {
	if err := s.repo.UpdateStatus(ctx, id, domain.NodeOffline); err != nil {
		return err
	}
	return s.cache.Del(ctx, heartbeatKeyPrefix+id)
}
