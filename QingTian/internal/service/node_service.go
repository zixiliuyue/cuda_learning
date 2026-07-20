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
//   - repo:                 节点仓储。
//   - cache:                缓存，用于存储带 TTL 的心跳。
//   - ledger:               资源账本，节点上报 GPU 时同步容量。
//   - nodeTimeout:          心跳超时时间。
//   - idleThresholdUtil:    空闲利用率阈值（百分比），低于此值认为可能空闲。
//   - idleThresholdDuration: 持续空闲多久才触发自动回收。
//   - stopChan:             用于停止定时扫描器。
//   - now:                  时间函数，便于测试。
type NodeService struct {
	repo                 repository.NodeRepository
	cache                cache.Cache
	ledger               *ledger.Ledger
	nodeTimeout          time.Duration
	idleThresholdUtil    int
	idleThresholdDuration time.Duration
	stopChan             chan struct{}
	now                  func() time.Time
}

// NewNodeService 创建节点服务。
func NewNodeService(repo repository.NodeRepository, c cache.Cache, l *ledger.Ledger, nodeTimeout time.Duration) *NodeService {
	return &NodeService{
		repo:                 repo,
		cache:                c,
		ledger:               l,
		nodeTimeout:          nodeTimeout,
		idleThresholdUtil:    10, // 默认值
		idleThresholdDuration: 30 * time.Minute, // 默认值
		stopChan:             make(chan struct{}),
		now:                  time.Now,
	}
}

// WithIdleConfig 设置空闲回收配置（用于链式构建，从配置读取自定义值）。
func (s *NodeService) WithIdleConfig(utilThreshold int, durationThreshold time.Duration) *NodeService {
	s.idleThresholdUtil = utilThreshold
	s.idleThresholdDuration = durationThreshold
	return s
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

// IdleNode 表示一个被判定为空闲的节点及其空闲信息，供空闲卡回收使用。
type IdleNode struct {
	NodeID     string  `json:"node_id"`     // 节点 ID
	Hostname   string  `json:"hostname"`    // 主机名
	AllocRatio float64 `json:"alloc_ratio"` // 账本显存分配率（0-1）
	MaxUtil    int     `json:"max_util"`    // 节点内 GPU 的最高实时利用率
}

// DetectIdleNodes 检测空闲节点，用于自动缩容/降功耗、降低运营成本。
//
// 判定标准：账本显存分配率为 0（无任务占用）且所有 GPU 的实时利用率
// 均不超过 utilThreshold。空闲卡长期空转是能耗浪费，识别后可自动降功耗，
// 强化平台"省钱"叙事。
//
// 参数：
//   - ctx:           上下文。
//   - utilThreshold: 判定空闲的利用率阈值（0-100）。
//
// 返回值：
//   - []IdleNode: 空闲节点列表（仅统计在线节点）。
//   - error:      读取节点失败时返回。
func (s *NodeService) DetectIdleNodes(ctx context.Context, utilThreshold int) ([]IdleNode, error) {
	nodes, err := s.repo.List(ctx)
	if err != nil {
		return nil, err
	}
	var idle []IdleNode
	for _, n := range nodes {
		alive, _ := s.IsAlive(ctx, n.ID)
		if !alive {
			continue // 离线节点不参与空闲回收
		}
		if s.ledger.NodeUtilization(n.ID) > 0 {
			continue // 账本上仍有占用，非空闲
		}
		maxUtil := 0
		for _, g := range n.GPUs {
			if g.Utilization > maxUtil {
				maxUtil = g.Utilization
			}
		}
		if maxUtil <= utilThreshold {
			idle = append(idle, IdleNode{
				NodeID:     n.ID,
				Hostname:   n.Hostname,
				AllocRatio: 0,
				MaxUtil:    maxUtil,
			})
		}
	}
	return idle, nil
}

// StartIdleScanner 启动闲置节点自动回收定时扫描器。
// 扫描间隔由调用方传入，通常设置为数倍调度间隔即可。
// 当节点在线、显存分配率为 0、且所有 GPU 利用率持续低于阈值超过配置时长时，
// 自动将节点标记为离线，从而触发资源回收（降低功耗或集群缩容）。
func (s *NodeService) StartIdleScanner(ctx context.Context, scanInterval time.Duration) {
	ticker := time.NewTicker(scanInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// 执行一次闲置检测与回收
			s.runIdleCheck(ctx)
		case <-s.stopChan:
			return
		case <-ctx.Done():
			return
		}
	}
}

// StopIdleScanner 停止闲置扫描器。
func (s *NodeService) StopIdleScanner() {
	select {
	case s.stopChan <- struct{}{}:
	default:
	}
}

// runIdleCheck 执行一轮闲置检测并回收满足条件的节点。
func (s *NodeService) runIdleCheck(ctx context.Context) {
	// 检测当前空闲节点
	idleNodes, err := s.DetectIdleNodes(ctx, s.idleThresholdUtil)
	if err != nil {
		return // 检测失败静默跳过，下轮重试
	}

	if len(idleNodes) == 0 {
		return // 无空闲节点
	}

	// 对每个空闲节点，检查是否持续空闲超过配置时长
	for _, idle := range idleNodes {
		node, err := s.Get(ctx, idle.NodeID)
		if err != nil {
			continue
		}

		// 获取最后心跳时间，判断是否已持续空闲足够久
		if s.now().Sub(node.LastHeartbeat) >= s.idleThresholdDuration {
			// 满足条件，自动标记为离线
			_ = s.SetOffline(ctx, idle.NodeID)
		}
	}
}
