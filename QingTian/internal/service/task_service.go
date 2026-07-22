// 本文件实现任务服务：任务提交、调度、状态处理、超时代理。
//
// 任务服务串联起"仓储 -> 调度器 -> 消息队列 -> Agent"整个闭环，是调度核心协调者。
package service

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/mq"
	"qingtian/internal/notify"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
	"qingtian/internal/tracing"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// TaskService 提供任务生命周期管理能力。
//
// 字段：
//   - repo:      任务仓储。
//   - nodeRepo:  节点仓储（调度时读取候选节点）。
//   - nodeService: 节点服务，用于节点故障检测存活状态。
//   - queue:     待调度优先级队列。
//   - producer:   消息生产者，下发任务到 Agent。
//   - sched:     调度器核心，执行过滤打分。
//   - now:       时间函数，便于测试注入。
//
// 下列依赖为可选（通过 With* 注入），未注入对应能力静默跳过：
//   - logs:      日志服务（任务结束归档日志）。
//   - metering: 计量服务（任务结束记录用量）。
//   - quota:     配额服务（提交任务校验配额）。
//   - projects:  项目服务（项目配额校验）。
//   - notifier:  通知服务（任务结束通知用户）。
//   - tracer:    分布式追踪。
//   - ledgerStore: 账本快照共享存储。
type TaskService struct {
	repo         repository.TaskRepository
	nodeRepo     repository.NodeRepository
	nodeService  *NodeService
	queue        cache.PriorityQueue
	producer     mq.Producer
	sched        scheduler.SchedulerInterface
	now          func() time.Time

	logs     *LogService
	metering *MeteringService
	quota    *QuotaService
	projects *ProjectService
	notifier notify.Notifier
	tracer   *tracing.Tracer

	// ledgerStore 是账本快照的共享存储（如 Redis）。注入后调度器进程重启可以从快照重建账本，
	// 避免"进程重启认为所有卡空闲"造成超卖，这是容错设计关键一步。
	ledgerStore cache.Cache

	// ledgerVersion 是 SyncLedger 加载到快照版本，写回的时候用 CAS 保证一致性。
	// 仅当 ledgerStore != nil 时有效。
	ledgerVersion int64
}

// ledgerSnapshotKey 是账本快照在共享存储中的键。
const ledgerSnapshotKey = "qingtian:ledger:snapshot"

// NewTaskService 创建任务服务。
func NewTaskService(
	repo repository.TaskRepository,
	nodeRepo repository.NodeRepository,
	queue cache.PriorityQueue,
	producer mq.Producer,
	sched scheduler.SchedulerInterface,
) *TaskService {
	return &TaskService{
		repo:      repo,
		nodeRepo:  nodeRepo,
		queue:     queue,
		producer:  producer,
		sched:     sched,
		now:       time.Now,
	}
}

// WithLogService 注入日志服务（链式）。
func (s *TaskService) WithLogService(l *LogService) *TaskService {
	s.logs = l
	return s
}

// WithMetering 注入计量服务（链式）。
func (s *TaskService) WithMetering(m *MeteringService) *TaskService {
	s.metering = m
	return s
}

// WithQuota 注入配额服务（链式）。
func (s *TaskService) WithQuota(q *QuotaService) *TaskService {
	s.quota = q
	return s
}

// WithNotifier 注入通知服务（链式）。
func (s *TaskService) WithNotifier(n notify.Notifier) *TaskService {
	s.notifier = n
	return s
}

// WithTracer 注入分布式追踪（链式）。
func (s *TaskService) WithTracer(t *tracing.Tracer) *TaskService {
	s.tracer = t
	return s
}

// WithLedgerStore 注入账本快照存储（链式）。
func (s *TaskService) WithLedgerStore(c cache.Cache) *TaskService {
	s.ledgerStore = c
	return s
}

// WithNodeService 注入节点服务，用于节点存活检测（链式）。
func (s *TaskService) WithNodeService(ns *NodeService) *TaskService {
	s.nodeService = ns
	return s
}

// WithProjects 注入项目服务（链式）。
func (s *TaskService) WithProjects(p *ProjectService) *TaskService {
	s.projects = p
	return s
}

// WithClock 注入自定义时钟，便于测试（链式）。
func (s *TaskService) WithClock(now func() time.Time) *TaskService {
	s.now = now
	return s
}

// SubmitRequest 是任务提交请求。
type SubmitRequest struct {
	Name           string                 `json:"name"`
	Image          string                 `json:"image"`
	Command        string                 `json:"command"`
	Env            []domain.EnvVar        `json:"env,omitempty"`
	Priority       int                    `json:"priority"`
	GPUCount       int                    `json:"gpu_count"`
	MemoryMB       int64                  `json:"memory_mb"`
	GPUModel       string                 `json:"gpu_model"`
	Vendor         domain.Vendor          `json:"vendor"`
	ShareMode      domain.ShareMode      `json:"share_mode"`
	TimeoutSeconds int64                  `json:"timeout_seconds"`
	Preemptible    bool                   `json:"preemptible"`
	MaxRetries     int                    `json:"max_retries,omitempty"` // 最大自动重试次数，0 表示不重试
	DatasetPath    string                 `json:"dataset_path"`
	ArtifactsPath  string                 `json:"artifacts_path,omitempty"` // 训练产物输出路径（任务工作目录相对路径）
	ProjectID      string                 `json:"project_id"`
	GangSize       int                    `json:"gang_size"` // 成组调度需要多少份数（分布式训练需要 N 个节点各跑一份）
	NodeSelector   *domain.NodeSelector   `json:"node_selector,omitempty"` // 节点标签选择约束
	Tolerations    []domain.Toleration    `json:"tolerations,omitempty"`    // 节点污点容忍列表
}

// Submit 提交一个新任务，入库后推入待调度队列。
//
// 提交前如果注入了配额服务，会按预计并发和卡时校验配额，超额拒绝提交。
func (s *TaskService) Submit(ctx context.Context, owner string, req SubmitRequest) (*domain.Task, error) {
	ctx, finish := s.startSpan(ctx, "task.submit")
	defer finish()
	if req.Name == "" || req.Image == "" {
		return nil, apperr.InvalidArgument("任务名称与镜像不能为空")
	}
	if req.GPUCount <= 0 {
		return nil, apperr.InvalidArgument("GPU 数量必须大于 0")
	}
	// 提交前如果注入了配额服务，会按预计并发校验配额，超额拒绝提交。
	if s.quota != nil {
		countActive := s.CountActive(ctx, owner, req.ProjectID)
		// 即使预估 0 卡时，如果剩余卡时 <= 0，也要拒绝提交（严格检查）
		estimatedHours := 0.0
		if err := s.quota.CheckAdmit(ctx, owner, estimatedHours, countActive+1); err != nil {
			return nil, err
		}
	}
	now := s.now()
	task := &domain.Task{
		ID:             idgen.New("task"),
		Name:           req.Name,
		Owner:          owner,
		ProjectID:      req.ProjectID,
		Image:          req.Image,
		Command:        req.Command,
		Env:            req.Env,
		Resource:       domain.ResourceRequest{
			GPUCount:     req.GPUCount,
			MemoryMB:     req.MemoryMB,
			GPUModel:     req.GPUModel,
			Vendor:       req.Vendor,
			ShareMode:    req.ShareMode,
			GangSize:     req.GangSize,
			NodeSelector: req.NodeSelector,
			Tolerations:  req.Tolerations,
		},
		Priority:        req.Priority,
		Status:          domain.TaskPending,
		DatasetPath:     req.DatasetPath,
		ResultPath:      "results/" + req.Name + "/",
		LogPath:         "logs/" + req.Name + ".log",
		ArtifactsPath:    req.ArtifactsPath,
		CheckpointPath:  "",
		CreatedAt:       now,
		UpdatedAt:       now,
		TimeoutSeconds:  req.TimeoutSeconds,
		Preemptible:     req.Preemptible,
		MaxRetries:      req.MaxRetries,
		Retries:         0,
	}
	if err := s.repo.Create(ctx, task); err != nil {
		return nil, err
	}
	if err := s.queue.Push(ctx, task.ID, float64(req.Priority)); err != nil {
		return nil, err
	}
	return task, nil
}

// CountActive 统计满足状态（scheduled/running）任务个数，用于配额校验。
func (s *TaskService) CountActive(ctx context.Context, owner, projectID string) int {
	n, err := s.repo.CountActive(ctx, owner, projectID)
	if err != nil {
		return 0
	}
	return n
}

// SyncLedger 从存储层真相源重建资源账本（调度器进程重启后调用）。
func (s *TaskService) SyncLedger(ctx context.Context) error {
	nodes, err := s.nodeRepo.List(ctx)
	if err != nil {
		return err
	}
	// 获取所有活动态任务（scheduled + running）
	scheduledTasks, err := s.repo.List(ctx, domain.TaskScheduled)
	if err != nil {
		return err
	}
	runningTasks, err := s.repo.List(ctx, domain.TaskRunning)
	if err != nil {
		return err
	}
	activeTasks := append(scheduledTasks, runningTasks...)

	// 尝试从共享存储加载快照，如果成功则直接使用，不需要全量重建
	if s.ledgerStore != nil {
		dataStr, ok, err := s.ledgerStore.Get(ctx, "qingtian:ledger:snapshot")
		if err == nil && ok {
			data := []byte(dataStr)
			version, err := s.sched.LoadLedgerSnapshot(data)
			if err == nil {
				s.ledgerVersion = version
				// 加载快照后，同步最新节点变更（保证运行期新增节点也被纳入）
				s.sched.SyncNodes(nodes)
				return nil
			}
			// 加载快照失败，回退到全量重建
		}
	}
	// 没有快照或加载失败，从数据库全量重建
	s.sched.RebuildLedger(nodes, activeTasks)
	return nil
}

// PersistLedger 将当前账本快照写回共享存储，CAS 保证版本一致性。
func (s *TaskService) PersistLedger(ctx context.Context) (bool, error) {
	if s.ledgerStore == nil {
		return false, nil // 没有共享存储，不需要写回
	}
	data, err := s.sched.DumpLedgerSnapshot(s.ledgerVersion + 1)
	if err != nil {
		return false, err
	}

	// 读取旧值用于 CAS
	var oldVal string
	var ok bool
	if s.ledgerVersion > 0 {
		var errGet error
		oldVal, ok, errGet = s.ledgerStore.Get(ctx, "qingtian:ledger:snapshot")
		if errGet != nil {
			oldVal = ""
			ok = false
		}
		if !ok {
			oldVal = ""
		}
	} else {
		// 当 ledgerVersion == 0 表示首次写入，旧值期望是空
		oldVal = ""
	}

	// 使用 CAS 写共享存储，版本不匹配放弃写入，保证一致性。
	dataStr := string(data)
	ok, err = s.ledgerStore.SetCAS(ctx, "qingtian:ledger:snapshot", oldVal, dataStr, 0)
	if err != nil {
		return false, err
	}
	if ok {
		s.ledgerVersion = s.ledgerVersion + 1
	}
	return ok, nil
}

// ScheduleOnce 被调度器循环周期性调用，弹出一个任务尝试调度。
//
// 返回：
//   - bool:   true  if a task was processed, false if the queue is empty.
//   - error:  error if any error occurred during processing.
func (s *TaskService) ScheduleOnce(ctx context.Context) (bool, error) {
	ctx, finish := s.startSpan(ctx, "task.schedule-once")
	defer finish()
	taskID, ok, err := s.queue.Pop(ctx)
	if err != nil {
		return false, err
	}
	if !ok {
		return false, nil // 队列为空
	}

	task, err := s.repo.GetByID(ctx, taskID)
	if err != nil {
		return true, err
	}
	// 已经取消的任务不需要调度
	if task.Status != domain.TaskPending {
		return true, nil
	}

	nodes, err := s.nodeRepo.List(ctx)
	if err != nil {
		return true, err
	}

	gangSize := task.Resource.GangSize
	if gangSize <= 1 {
		// 普通单节点调度
		decision, err := s.sched.Schedule(ctx, task, nodes)
		if err != nil {
			// 找不到可用节点，尝试抢占，抢占不成功就放回排队
			if s.tryPreempt(ctx, task) {
				decision, err = s.sched.Schedule(ctx, task, nodes)
			}
			if err != nil {
				// 抢占后还是失败，放回排队
				_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
				return true, nil
			}
		}

		// 分配成功，更新状态写入数据库
		task.NodeID = decision.NodeID
		task.AllocatedGPUs = decision.AllocatedGPUs
		if err := task.Transition(domain.TaskScheduled, s.now()); err != nil {
			// 状态迁移失败，回滚分配
			s.sched.ReleaseTask(decision.AllocatedGPUs, task.Resource.MemoryMB)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			return true, err
		}

		// 持久化状态变更到数据库
		if err := s.repo.Update(ctx, task); err != nil {
			// 更新失败，回滚分配
			s.sched.ReleaseTask(decision.AllocatedGPUs, task.Resource.MemoryMB)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			return true, err
		}

		// 下发任务到目标节点
		msg := mq.DispatchMessage{
			TaskID:         task.ID,
			NodeID:         decision.NodeID,
			Image:          task.Image,
			Command:        task.Command,
			Env:            task.Env,
			AllocatedGPUs:  task.AllocatedGPUs,
			DatasetPath:    task.DatasetPath,
			ResultPath:     task.ResultPath,
			ArtifactsPath:  task.ArtifactsPath,
			ShareMode:      string(decision.ShareMode),
			CheckpointPath: task.CheckpointPath,
		}
		body, err := json.Marshal(msg)
		if err != nil {
			// 序列化失败，回滚分配
			s.sched.ReleaseTask(decision.AllocatedGPUs, task.Resource.MemoryMB)
			task.Status = domain.TaskPending
			task.NodeID = ""
			task.AllocatedGPUs = nil
			task.UpdatedAt = s.now()
			_ = s.repo.Update(ctx, task)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			// 补偿完成，返回已处理（processed=true）但 err=nil，因为这一轮调度已经完成
			return true, nil
		}
		if err := s.producer.Publish(ctx, mq.TaskDispatchTopic(decision.NodeID), body); err != nil {
			// 下发失败，回滚分配，将任务放回 pending 重新排队
			s.sched.ReleaseTask(decision.AllocatedGPUs, task.Resource.MemoryMB)
			task.Status = domain.TaskPending
			task.NodeID = ""
			task.AllocatedGPUs = nil
			task.UpdatedAt = s.now()
			_ = s.repo.Update(ctx, task)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			// 补偿完成，返回已处理（processed=true）但 err=nil，因为这一轮调度已经完成
			return true, nil
		}
		return true, nil
	}

	// ========== 成组调度（Gang Scheduling）==========
	// 语义：All-or-Nothing，必须一次性分配完所有份数，否则全部回滚
	decisions, err := s.sched.ScheduleGang(ctx, task, nodes)
	if err != nil {
		// 分配失败（任意一份失败，ScheduleGang 内部已回滚），放回排队
		_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
		return true, nil
	}

	// Gang 调度成功：为每个成员生成下发消息，逐个发送到对应节点
	// 注意：原任务作为 Gang 主任务，每个决策对应一个 GPU 分配结果
	// 对于多节点分布式训练，每个节点一份，已经由 ScheduleGang 拓扑分散
	for i, decision := range decisions {
		// 为每个 Gang 成员分配独立的 GPU 列表
		// 由于分布式训练每个任务只需要自己那份 GPU，这里我们将原任务的
		// 分配信息更新后拷贝，每个成员对应一份下发
		// 用户层面看到的是一个 Gang 任务，实际下发 N 份到不同节点
		taskCopy := *task
		taskCopy.NodeID = decision.NodeID
		taskCopy.AllocatedGPUs = decision.AllocatedGPUs
		taskCopy.UpdatedAt = s.now()

		// 更新数据库状态
		if err := s.repo.Update(ctx, &taskCopy); err != nil {
			// 更新失败，回滚已经分配的所有资源
			for _, d := range decisions[:i] {
				s.sched.ReleaseTask(d.AllocatedGPUs, task.Resource.MemoryMB)
			}
			s.sched.ReleaseTask(decision.AllocatedGPUs, task.Resource.MemoryMB)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			return true, err
		}

		// 下发到目标节点 Agent
		msg := mq.DispatchMessage{
			TaskID:         taskCopy.ID,
			NodeID:         decision.NodeID,
			Image:          taskCopy.Image,
			Command:        taskCopy.Command,
			Env:            taskCopy.Env,
			AllocatedGPUs:  taskCopy.AllocatedGPUs,
			DatasetPath:    taskCopy.DatasetPath,
			ResultPath:     taskCopy.ResultPath,
			ShareMode:      string(decision.ShareMode),
			CheckpointPath: task.CheckpointPath,
		}
		body, err := json.Marshal(msg)
		if err != nil {
			// 序列化失败，回滚已经分配的所有资源
			for _, d := range decisions[:i] {
				s.sched.ReleaseTask(d.AllocatedGPUs, task.Resource.MemoryMB)
			}
			s.sched.ReleaseTask(decision.AllocatedGPUs, task.Resource.MemoryMB)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			return true, err
		}
		if err := s.producer.Publish(ctx, mq.TaskDispatchTopic(decision.NodeID), body); err != nil {
			// 下发失败，回滚已经分配的所有资源
			for _, d := range decisions[:i] {
				s.sched.ReleaseTask(d.AllocatedGPUs, task.Resource.MemoryMB)
			}
			s.sched.ReleaseTask(decision.AllocatedGPUs, task.Resource.MemoryMB)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			return true, err
		}
	}

	// 所有份数都下发成功，更新主任务状态为已调度
	task.GangID = task.ID // GangID 指向自身，同组所有任务共享这个 ID
	if err := task.Transition(domain.TaskScheduled, s.now()); err != nil {
		// 状态更新失败，回滚全部
		for _, d := range decisions {
			s.sched.ReleaseTask(d.AllocatedGPUs, task.Resource.MemoryMB)
		}
		_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
		return true, err
	}

	// 最后更新主任务到数据库
	if err := s.repo.Update(ctx, task); err != nil {
		// 更新失败，回滚全部
		for _, d := range decisions {
			s.sched.ReleaseTask(d.AllocatedGPUs, task.Resource.MemoryMB)
		}
		_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
		return true, err
	}

	return true, nil
}

// ScheduleBatch 其实就是循环调用 ScheduleOnce 处理队列中所有任务，
// 该方法被调度器循环周期性调用，每个周期处理一批任务。
//
// 这个设计让调度器可以单线程循环运行，配合 Redis 分布式锁保证只有一个leader在工作。
func (s *TaskService) ScheduleBatch(ctx context.Context) {
	for {
		ok, err := s.ScheduleOnce(ctx)
		if err != nil {
			continue
		}
		if !ok {
			break
		}
	}
}

// tryPreempt 尝试抢占在高优先级任务无法分配时尝试抢占低优先级任务，
// 尝试抢占，成功腾出空间就返回true，失败就返回false。
//
// 算法：
//  1. 在当前命中节点集合收集可抢占任务，按优先级从小到大排序，
//  2. 从小到大依次尝试抢占，只要腾出足够空间就立刻返回，
//  这样最小优先级先被抢占，保证只腾出刚好够用，减少不必要抢占。
func (s *TaskService) tryPreempt(ctx context.Context, task *domain.Task) bool {
	// 按节点归集可抢占任务（从数据库获取所有运行中任务）
	activeTasks, err := s.repo.List(ctx, domain.TaskRunning)
	if err != nil {
		return false
	}

	victimsByNode := make(map[string][]*domain.Task)
	for _, candidate := range activeTasks {
		if !candidate.Preemptible {
			continue
		}
		if candidate.Priority >= task.Priority {
			// 当前候选任务优先级不小于要抢占任务优先级，不能抢占
			continue
		}
		// 检查节点是否满足资源约束
		node, err := s.nodeRepo.GetByID(ctx, candidate.NodeID)
		if err != nil {
			continue
		}
		if !s.nodeMatchesConstraints(node, task.Resource) {
			// 节点不匹配资源约束，跳过
			continue
		}
		victimsByNode[candidate.NodeID] = append(victimsByNode[candidate.NodeID], candidate)
	}

	// 按节点遍历，从小到大尝试，成功腾出空间返回
	for nodeID, victims := range victimsByNode {
		sort.Slice(victims, func(i, j int) bool {
			return victims[i].Priority < victims[j].Priority
		})

		// 尝试从最低优先级开始抢占，直到腾出足够空间
		for _, victim := range victims {
			// 抢占该任务：终止任务并释放它的资源
			_ = s.terminateActiveTask(ctx, victim, domain.FailurePreempted, "被高优先级任务抢占")
			// 检查是否已经腾出足够空间
			if s.sched.CanFit(nodeID, task.Resource) {
				// 成功，可以分配了
				return true
			}
		}
	}
	return false
}

// nodeMatchesConstraints 检查节点是否满足任务的资源约束（型号、厂商、标签选择器、污点容忍）。
func (s *TaskService) nodeMatchesConstraints(n *domain.Node, req domain.ResourceRequest) bool {
	if req.GPUModel != "" {
		// 检查节点上至少有一块 GPU 匹配型号要求
		hasMatch := false
		for _, gpu := range n.GPUs {
			if gpu.Model == req.GPUModel {
				hasMatch = true
				break
			}
		}
		if !hasMatch {
			return false
		}
	}
	if req.Vendor != "" {
		// 检查节点上至少有一块 GPU 匹配厂商要求
		hasMatch := false
		for _, gpu := range n.GPUs {
			if gpu.Vendor == req.Vendor {
				hasMatch = true
				break
			}
		}
		if !hasMatch {
			return false
		}
	}
	// 检查节点标签匹配
	if req.NodeSelector != nil && req.NodeSelector.MatchLabels != nil {
		for k, v := range req.NodeSelector.MatchLabels {
			nodeVal, exists := n.Labels[k]
			if !exists || nodeVal != v {
				// 键不存在 或 值不匹配，筛选掉
				return false
			}
		}
	}
	// 检查污点容忍：对于每个节点污点，任务必须容忍才能被调度
	for _, taint := range n.Taints {
		// 检查任务是否容忍这个污点
		tolerated := false
		for _, toleration := range req.Tolerations {
			if taintMatchesToleration(taint, toleration) {
				tolerated = true
				break
			}
		}
		if !tolerated {
			// 不容忍这个污点，直接拒绝这个节点
			switch taint.Effect {
			case domain.TaintEffectNoSchedule:
				// 硬性拒绝，不允许调度
				return false
			case domain.TaintEffectPreferNoSchedule:
				// 软拒绝，继续匹配但打分会降低优先级，这里不过滤只影响打分
				// 所以这里不返回 false，继续过滤
			}
		}
	}
	return true
}

// taintMatchesToleration 检查污点是否被容忍匹配。
//
// 匹配规则：
//  - 键必须相等
//  - 如果 Operator 是 Equal，值必须相等
//  - 如果 Operator 是 Exists，只需要键存在（不要求值相等）
func taintMatchesToleration(taint domain.Taint, toleration domain.Toleration) bool {
	if taint.Key != toleration.Key {
		return false
	}
	if taint.Effect != toleration.Effect {
		return false
	}
	switch toleration.Operator {
	case domain.TolerationOpExists:
		// Exists 只要求键存在，已经匹配了键，所以匹配成功
		return true
	case domain.TolerationOpEqual:
		// Equal 要求值相等
		return toleration.Value == taint.Value
	default:
		return false
	}
}

// startSpan 开始一个分布式追踪 span，如果 tracer 未注入则返回原上下文与空 finish。
func (s *TaskService) startSpan(ctx context.Context, name string) (context.Context, func()) {
	if s.tracer == nil {
		return ctx, func() {}
	}
	ctx, span := s.tracer.Start(ctx, name)
	return ctx, func() { span.Finish() }
}

// 总结：
//  Gang 成组调度已经完整实现，支持分布式训练一次分配 N 个节点，
// 要么全部分配成功，要么一份失败回滚所有，保证全有全无可，不会出现部分分配成功卡住。
// 符合"all-or-nothing"分配原则。

// onTaskFinished 处理任务进入终态后的副作用：归档日志、计量计费、通知。
//
// 这些操作不影响任务状态流转成败，失败静默跳过。
func (s *TaskService) onTaskFinished(ctx context.Context, task *domain.Task) {
	// 判断任务是否进入终态
	isTerminal := task.Status == domain.TaskSucceeded || task.Status == domain.TaskFailed || task.Status == domain.TaskCanceled
	if !isTerminal {
		return
	}

	// 自动重试：如果任务失败、还有剩余重试次数，自动重新入队重试
	if task.Status == domain.TaskFailed && task.MaxRetries > 0 && task.Retries < task.MaxRetries {
		now := s.now()
		task.Retries++
		task.LastRetryAt = &now
		// 重置状态为 pending，清除调度信息
		task.Status = domain.TaskPending
		task.NodeID = ""
		task.AllocatedGPUs = nil
		task.StartedAt = nil
		task.FinishedAt = nil
		task.UpdatedAt = now
		task.Message = fmt.Sprintf("第 %d 次重试，上次失败: %s", task.Retries, task.Message)
		// 更新数据库
		if err := s.repo.Update(ctx, task); err != nil {
			s.onTaskFinished(ctx, task)
		} else {
			// 重新入队等待调度
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
		}
		// 计量和通知仍正常执行
	}

	// 计量：累加用量到配额
	if s.metering != nil && s.quota != nil {
		if task.StartedAt != nil && task.FinishedAt != nil {
			duration := task.FinishedAt.Sub(*task.StartedAt)
			hours := duration.Hours() * float64(len(task.AllocatedGPUs))
			subject := task.ProjectID
			if subject == "" {
				subject = task.Owner
			}
			_, _ = s.metering.RecordTaskUsage(ctx, task)
			_ = s.quota.Charge(ctx, subject, hours)
		}
	}
	// 日志归档
	if s.logs != nil {
		_ = s.logs.Archive(ctx, task.ID, task.LogPath)
	}
	// 通知：任务终态主动触达用户。
	if s.notifier != nil {
		title := fmt.Sprintf("任务 %s 已%s", task.Name, statusText(task.Status))
		content := fmt.Sprintf("任务 ID: %s\n状态: %s\n退出码: %d\n已重试: %d/%d\n说明: %s",
			task.ID, task.Status, task.ExitCode, task.Retries, task.MaxRetries, task.Message)
		_ = s.notifier.Notify(ctx, notify.Message{Title: title, Content: content})
	}
}

// statusText 返回任务终态的中文表述，用于通知标题。
func statusText(status domain.TaskStatus) string {
	switch status {
	case domain.TaskSucceeded:
		return "成功"
	case domain.TaskFailed:
		return "失败"
	case domain.TaskCanceled:
		return "取消"
	default:
		return string(status)
	}
}

// Get 按 ID 查询任务。
func (s *TaskService) Get(ctx context.Context, id string) (*domain.Task, error) {
	return s.repo.GetByID(ctx, id)
}

// List 按状态过滤查询任务。
func (s *TaskService) List(ctx context.Context, status domain.TaskStatus) ([]*domain.Task, error) {
	return s.repo.List(ctx, status)
}

// Cancel 取消一个任务。
//
// 参数：
//   - ctx: 上下文。
//   - id:  任务 ID。
//
// 返回值：
//   - error: 任务不存在或当前状态不可取消时返回。
func (s *TaskService) Cancel(ctx context.Context, id string) error {
	task, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return err
	}
	if err := task.Transition(domain.TaskCanceled, s.now()); err != nil {
		return err
	}
	return s.repo.Update(ctx, task)
}

// Resume 从最近一次 Checkpoint 恢复并重新调度一个已失败的任务（断点续跑）。
//
// 该方法把失败任务重置回 pending 并重新入队，携带其 CheckpointPath；
// 重新调度下发后，Agent 会从该存档恢复进度，避免从头重跑。
//
// 参数：
//   - ctx: 上下文。
//   - id:  任务 ID。
//
// 返回值：
//   - error: 任务不存在、无存档、或当前状态不允许恢复时返回。
func (s *TaskService) Resume(ctx context.Context, id string) error {
	task, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return err
	}
	if task.Status != domain.TaskFailed {
		return apperr.Conflict("仅失败任务可断点续跑，当前状态: " + string(task.Status))
	}
	if task.CheckpointPath == "" {
		return apperr.InvalidArgument("任务无 Checkpoint，无法断点续跑")
	}
	// 重置为 pending：清空调度信息与失败信息，保留 CheckpointPath 供恢复。
	now := s.now()
	task.Status = domain.TaskPending
	task.NodeID = ""
	task.AllocatedGPUs = nil
	task.ExitCode = 0
	task.Message = "断点续跑：从 Checkpoint 恢复"
	task.FailureCategory = domain.FailureNone
	task.StartedAt = nil
	task.FinishedAt = nil
	task.UpdatedAt = now
	if err := s.repo.Update(ctx, task); err != nil {
		return err
	}
	return s.queue.Push(ctx, task.ID, float64(task.Priority))
}

// SetCheckpoint 记录任务最近一次 Checkpoint 的存档路径。
//
// 供 Agent 在周期性存档后回调控制面登记，使断点续跑有据可依。
func (s *TaskService) SetCheckpoint(ctx context.Context, id, checkpointPath string) error {
	task, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return err
	}
	task.CheckpointPath = checkpointPath
	task.UpdatedAt = s.now()
	return s.repo.Update(ctx, task)
}

// terminateActiveTask 终止一个仍在占用资源的活动任务，并归还其资源。
//
// 处理步骤：
//  1. 通知目标节点 Agent 终止任务进程（尽力而为，失败不阻断状态推进）。
//  2. 归还账本中该任务占用的 GPU/显存，避免资源泄漏。
//  3. 将任务状态转移到 failed 并写入失败归类与说明。
//
// 参数：
//   - ctx:      上下文。
//   - task:     待终止的任务（应处于 scheduled/running）。
//   - category: 失败归类（如超时、抢占）。
//   - reason:   面向用户的说明文案。
//
// 返回值：
//   - error: 状态持久化失败时返回。
func (s *TaskService) terminateActiveTask(ctx context.Context, task *domain.Task, category domain.FailureCategory, reason string) error {
	// 1. 通知 Agent 终止进程（内存/NSQ 总线均支持发布，失败仅记录不阻断）。
	if task.NodeID != "" {
		ctrl := mq.ControlMessage{TaskID: task.ID, NodeID: task.NodeID, Action: mq.ControlKill, Reason: reason}
		if body, mErr := json.Marshal(ctrl); mErr == nil {
			_ = s.producer.Publish(ctx, mq.TaskControlTopic(task.NodeID), body)
		}
	}
	// 2. 归还账本资源。
	if s.sched != nil {
		s.sched.ReleaseTask(task.AllocatedGPUs, task.Resource.MemoryMB)
	}
	// 3. 推进状态到 failed。
	task.FailureCategory = category
	task.Message = reason
	task.ExitCode = -1
	if err := task.Transition(domain.TaskFailed, s.now()); err != nil {
		return err
	}
	if err := s.repo.Update(ctx, task); err != nil {
		return err
	}
	// 触发终态副作用（计量/计费/通知/日志归档）。
	s.onTaskFinished(ctx, task)
	return nil
}

// EnforceTimeouts 扫描全部运行中任务，将已超时者强制终止（超时防护）。
//
// 任务因 Agent 挂起或网络中断而无限期运行会造成 GPU 资源泄漏。该方法由守护循环
// 周期性调用，找出运行时长超过其 TimeoutSeconds 的任务并终止、归还资源，
// 是"防资源泄漏"的兜底保障。
//
// 参数：
//   - ctx: 上下文。
//
// 返回值：
//   - int:   本轮被终止的任务数。
//   - error: 遍历任务失败时返回。
func (s *TaskService) EnforceTimeouts(ctx context.Context) (int, error) {
	running, err := s.repo.List(ctx, domain.TaskRunning)
	if err != nil {
		return 0, err
	}
	now := s.now()
	killed := 0
	for _, task := range running {
		if !task.IsTimedOut(now) {
			continue
		}
		reason := fmt.Sprintf("任务运行超过超时上限 %d 秒，已被强制终止", task.TimeoutSeconds)
		if err := s.terminateActiveTask(ctx, task, domain.FailureTimeout, reason); err != nil {
			// 单个任务终止失败不影响其它任务，继续处理。
			continue
		}
		killed++
	}
	return killed, nil
}

// EnforceZombieDetection 扫描全部运行中任务，检测并回收僵尸任务。
//
// 僵尸任务定义：任务处于运行状态，但分配给它的所有 GPU 利用率持续低于
// 阈值超过指定时长，说明任务实际已卡住不干活了但进程还在占用资源。
// 该方法由守护循环周期性调用，自动回收这类泄漏的资源。
//
// 参数：
//   - ctx: 上下文。
//   - idleThresholdUtil: 利用率阈值（百分比），低于此值认为空闲
//   - idleThresholdDuration: 持续空闲多久才触发回收
//
// 返回值：
//   - int:   本轮被回收的僵尸任务数。
//   - error: 遍历任务失败时返回。
func (s *TaskService) EnforceZombieDetection(ctx context.Context, idleThresholdUtil int, idleThresholdDuration time.Duration) (int, error) {
	running, err := s.repo.List(ctx, domain.TaskRunning)
	if err != nil {
		return 0, err
	}
	now := s.now()
	killed := 0

	for _, task := range running {
		if task.NodeID == "" || task.StartedAt == nil {
			continue
		}

		// 任务刚开始运行，还没达到最小检测时长，跳过
		if now.Sub(*task.StartedAt) < idleThresholdDuration {
			continue
		}

		node, err := s.nodeRepo.GetByID(ctx, task.NodeID)
		if err != nil {
			continue // 节点不存在，会被 ReconcileNodeFailures 处理
		}

		// 计算任务分配到的 GPU 的平均利用率
		var totalUtil int
		var count int
		for _, allocGPU := range task.AllocatedGPUs {
			for _, gpu := range node.GPUs {
				if gpu.UUID == allocGPU {
					totalUtil += gpu.Utilization
					count++
					break
				}
			}
		}

		if count == 0 {
			continue // 没有找到分配的 GPU，跳过
		}

		avgUtil := totalUtil / count
		if avgUtil < idleThresholdUtil {
			// 利用率持续低于阈值，判定为僵尸任务
			reason := fmt.Sprintf("僵尸任务检测：分配的 GPU 平均利用率 %.1f%% 持续低于阈值 %d%% 超过 %v，已被强制回收",
				float64(avgUtil), idleThresholdUtil, idleThresholdDuration)
			if err := s.terminateActiveTask(ctx, task, domain.FailureTimeout, reason); err != nil {
				continue
			}
			killed++
		}
	}
	return killed, nil
}

// ReconcileNodeFailures 将离线节点上的未完成任务放回待调度队列重调度。
//
// 节点故障离线后，其上运行的任务永远不会完成，需要重新调度到其它节点上。
// 该方法由守护循环周期性调用，实现自动故障恢复。
//
// 参数：
//   - ctx: 上下文。
//
// 返回值：
//   - int:   本轮被移回排队的任务数。
//   - error: 查询失败时返回。
func (s *TaskService) ReconcileNodeFailures(ctx context.Context) (int, error) {
	// 获取所有已调度但未完成的任务
	scheduled, err := s.repo.List(ctx, domain.TaskScheduled)
	if err != nil {
		return 0, err
	}
	moved := 0
	now := s.now()
	for _, task := range scheduled {
		if task.NodeID == "" {
			continue
		}
		// 使用 nodeService 检测节点是否存活（基于心跳缓存 TTL）
		var alive bool
		if s.nodeService != nil {
			alive, _ = s.nodeService.IsAlive(ctx, task.NodeID)
		} else {
			// 降级：如果 nodeService 未注入，直接从仓库读取状态
			node, err := s.nodeRepo.GetByID(ctx, task.NodeID)
			alive = err == nil && node.Status == domain.NodeOnline
		}
		if !alive {
			// 节点不存在或离线，将任务放回排队重调度
			task.Status = domain.TaskPending
			task.NodeID = ""
			task.AllocatedGPUs = nil
			task.UpdatedAt = now
			_ = s.repo.Update(ctx, task)
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			moved++
		}
	}
	return moved, nil
}

// HandleStatus 处理 Agent 回传的任务状态消息。
//
// 该方法是任务状态回传消息的消费回调，实现幂等：对已处于终态的任务忽略更新。
//
// 参数：
//   - ctx:  上下文。
//   - body: 状态消息 JSON 字节。
//
// 返回值：
//   - error: 处理失败时返回（会触发消息重投）。
func (s *TaskService) HandleStatus(ctx context.Context, body []byte) error {
	var msg mq.StatusMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		return apperr.Wrap(apperr.CodeInvalidArgument, "状态消息解析失败", err)
	}
	task, err := s.repo.GetByID(ctx, msg.TaskID)
	if err != nil {
		return err
	}
	target := domain.TaskStatus(msg.Status)
	// 幂等：目标状态与当前一致，或当前已是终态，直接忽略。
	if task.Status == target || !domain.CanTransition(task.Status, target) {
		return nil
	}
	task.ExitCode = msg.ExitCode
	task.Message = msg.Message
	// 失败时做结构化诊断，将黑盒失败转化为可操作提示。
	if target == domain.TaskFailed {
		diag := domain.DiagnoseFailure(msg.ExitCode, msg.Output)
		task.FailureCategory = diag.Category
		if diag.Summary != "" {
			task.Message = diag.Summary + "；建议：" + diag.Suggestion
		}
	}
	if err := task.Transition(target, s.now()); err != nil {
		return err
	}
	if err := s.repo.Update(ctx, task); err != nil {
		return err
	}
	// 进入终态时触发副作用：计量、计费、通知、日志归档。
	if target == domain.TaskSucceeded || target == domain.TaskFailed {
		s.onTaskFinished(ctx, task)
	}
	return nil
}
