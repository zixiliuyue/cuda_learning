// 本文件实现任务服务：任务提交、调度、下发、状态回传处理与查询。
//
// 任务服务串联调度器、账本、仓储与消息队列，是"提交->调度->执行->回传"闭环的核心。
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
//   - queue:     待调度优先级队列。
//   - producer:  消息生产者，用于下发任务。
//   - sched:     调度器。
//   - now:       时间函数，便于测试注入。
//
// 下列为可选依赖（通过 With* 注入），未注入时对应能力静默跳过，
// 从而保持核心闭环的最小可用与既有测试的兼容：
//   - logs:     日志服务（任务结束时归档日志）。
//   - metering: 计量服务（任务结束时记录用量）。
//   - quota:    配额服务（提交时校验、结束时计费）。
//   - notifier: 通知器（任务终态主动触达）。
type TaskService struct {
	repo     repository.TaskRepository
	nodeRepo repository.NodeRepository
	queue    cache.PriorityQueue
	producer mq.Producer
	sched    *scheduler.Scheduler
	now      func() time.Time

	logs     *LogService
	metering *MeteringService
	quota    *QuotaService
	notifier notify.Notifier
	tracer   *tracing.Tracer
}

// NewTaskService 创建任务服务。
func NewTaskService(
	repo repository.TaskRepository,
	nodeRepo repository.NodeRepository,
	queue cache.PriorityQueue,
	producer mq.Producer,
	sched *scheduler.Scheduler,
) *TaskService {
	return &TaskService{
		repo:     repo,
		nodeRepo: nodeRepo,
		queue:    queue,
		producer: producer,
		sched:    sched,
		now:      time.Now,
	}
}

// WithLogService 注入日志服务，返回自身便于链式调用。
func (s *TaskService) WithLogService(l *LogService) *TaskService {
	s.logs = l
	return s
}

// WithMetering 注入计量服务。
func (s *TaskService) WithMetering(m *MeteringService) *TaskService {
	s.metering = m
	return s
}

// WithQuota 注入配额服务。
func (s *TaskService) WithQuota(q *QuotaService) *TaskService {
	s.quota = q
	return s
}

// WithNotifier 注入通知器。
func (s *TaskService) WithNotifier(n notify.Notifier) *TaskService {
	s.notifier = n
	return s
}

// WithClock 注入自定义时间函数，仅用于测试（如验证超时判定）。
func (s *TaskService) WithClock(now func() time.Time) *TaskService {
	if now != nil {
		s.now = now
	}
	return s
}

// WithTracer 注入分布式追踪器，串联提交/调度/执行链路。
func (s *TaskService) WithTracer(t *tracing.Tracer) *TaskService {
	s.tracer = t
	return s
}

// startSpan 安全地开启一个 Span：未注入 tracer 时返回原 ctx 与 nil。
//
// 返回的 finish 函数总是可安全调用（nil-safe），便于在调用处 defer。
func (s *TaskService) startSpan(ctx context.Context, name string) (context.Context, func()) {
	if s.tracer == nil {
		return ctx, func() {}
	}
	spanCtx, span := s.tracer.Start(ctx, name)
	return spanCtx, span.Finish
}

// SubmitRequest 是任务提交请求。
type SubmitRequest struct {
	Name           string                 `json:"name"`
	Image          string                 `json:"image"`
	Command        string                 `json:"command"`
	Resource       domain.ResourceRequest `json:"resource"`
	Priority       int                    `json:"priority"`
	DatasetPath    string                 `json:"dataset_path"`
	ProjectID      string                 `json:"project_id"`
	TimeoutSeconds int64                  `json:"timeout_seconds"`
	Preemptible    bool                   `json:"preemptible"`
}

// Submit 提交一个新任务，落库并进入待调度队列。
//
// 提交前若注入了配额服务，会按预计卡时与并发数做配额校验，超额则拒绝，
// 从而支撑"按用量计费 + 配额套餐"的商业化基础。
//
// 参数：
//   - ctx:   上下文。
//   - owner: 提交者用户 ID。
//   - req:   提交请求。
//
// 返回值：
//   - *domain.Task: 创建成功的任务（状态为 pending）。
//   - error:        参数非法、配额不足或持久化失败时返回。
func (s *TaskService) Submit(ctx context.Context, owner string, req SubmitRequest) (*domain.Task, error) {
	ctx, finish := s.startSpan(ctx, "task.submit")
	defer finish()
	if req.Name == "" || req.Image == "" {
		return nil, apperr.InvalidArgument("任务名称与镜像不能为空")
	}
	if req.Resource.GPUCount <= 0 {
		return nil, apperr.InvalidArgument("GPU 数量必须大于 0")
	}
	// 配额校验：以计费主体（项目优先，否则用户）为准。
	if s.quota != nil {
		subject := owner
		if req.ProjectID != "" {
			subject = req.ProjectID
		}
		running := s.countRunning(ctx, owner, req.ProjectID)
		// 预计卡时无法预知时长，这里以 GPU 份数作为并发/额度的近似门槛（估 1 卡时）。
		est := float64(req.Resource.GPUCount)
		if err := s.quota.CheckAdmit(ctx, subject, est, running); err != nil {
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
		Resource:       req.Resource,
		Priority:       req.Priority,
		Status:         domain.TaskPending,
		DatasetPath:    req.DatasetPath,
		TimeoutSeconds: req.TimeoutSeconds,
		Preemptible:    req.Preemptible,
		ResultPath:     "results/" + req.Name + "/",
		LogPath:        "logs/" + req.Name + ".log",
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	if err := s.repo.Create(ctx, task); err != nil {
		return nil, err
	}
	// 入待调度队列，score 使用优先级。
	if err := s.queue.Push(ctx, task.ID, float64(task.Priority)); err != nil {
		return nil, err
	}
	return task, nil
}

// countRunning 统计某主体当前处于运行中的任务数，用于并发配额校验。
func (s *TaskService) countRunning(ctx context.Context, owner, projectID string) int {
	tasks, err := s.repo.ListByOwner(ctx, owner)
	if err != nil {
		return 0
	}
	n := 0
	for _, t := range tasks {
		if projectID != "" && t.ProjectID != projectID {
			continue
		}
		if t.Status == domain.TaskRunning || t.Status == domain.TaskScheduled {
			n++
		}
	}
	return n
}

// ScheduleOnce 从队列取出一个任务并尝试调度、下发。
//
// 该方法由调度器循环周期性调用，实现步骤：
//  1. 从优先级队列弹出一个任务 ID。
//  2. 读取任务与在线节点，调用调度器决策。
//  3. 决策成功则更新任务状态为 scheduled，并通过 NSQ 下发到目标节点。
//  4. 决策失败（无可用节点）则将任务重新入队，等待下次调度。
//
// 参数：
//   - ctx: 上下文。
//
// 返回值：
//   - bool:  是否处理了一个任务（false 表示队列为空）。
//   - error: 处理过程中的错误。
func (s *TaskService) ScheduleOnce(ctx context.Context) (bool, error) {
	ctx, finish := s.startSpan(ctx, "task.schedule")
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
	// 已被取消的任务无需调度。
	if task.Status != domain.TaskPending {
		return true, nil
	}

	nodes, err := s.nodeRepo.List(ctx)
	if err != nil {
		return true, err
	}

	decision, err := s.sched.Schedule(ctx, task, nodes)
	if err != nil {
		// 无可用节点：尝试抢占低优先级的可抢占任务以腾出资源。
		if s.tryPreempt(ctx, task, nodes) {
			// 抢占成功，资源已释放，立即重试一次调度。
			decision, err = s.sched.Schedule(ctx, task, nodes)
		}
		if err != nil {
			// 仍无可用节点：重新入队，稍后重试。
			_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
			return true, nil
		}
	}

	// 更新任务状态为 scheduled。
	task.NodeID = decision.NodeID
	task.AllocatedGPUs = decision.AllocatedGPUs
	if err := task.Transition(domain.TaskScheduled, s.now()); err != nil {
		return true, err
	}
	if err := s.repo.Update(ctx, task); err != nil {
		return true, err
	}

	// 通过 NSQ 下发到目标节点。
	msg := mq.DispatchMessage{
		TaskID:         task.ID,
		NodeID:         task.NodeID,
		Image:          task.Image,
		Command:        task.Command,
		AllocatedGPUs:  task.AllocatedGPUs,
		DatasetPath:    task.DatasetPath,
		ResultPath:     task.ResultPath,
		ShareMode:      string(decision.ShareMode),
		CheckpointPath: task.CheckpointPath,
	}
	body, _ := json.Marshal(msg)
	if err := s.producer.Publish(ctx, mq.TaskDispatchTopic(task.NodeID), body); err != nil {
		return true, err
	}
	return true, nil
}

// tryPreempt 尝试抢占低优先级的可抢占任务，为高优先级任务腾出资源。
//
// 抢占是提升关键任务响应速度的核心手段：当高优先级任务无节点可调度时，
// 选出优先级更低且标记为可抢占（Preemptible）的运行/已调度任务，将其终止
// 并归还资源。为避免误伤，仅抢占严格低于本任务优先级的可抢占任务。
//
// 参数：
//   - ctx:   上下文。
//   - task:  触发抢占的高优先级待调度任务。
//   - nodes: 当前节点列表（此处未直接使用，保留以便未来按节点定向抢占）。
//
// 返回值：
//   - bool: 是否成功抢占了至少一个任务（成功后调用方可重试调度）。
func (s *TaskService) tryPreempt(ctx context.Context, task *domain.Task, _ []*domain.Node) bool {
	all, err := s.repo.List(ctx, "")
	if err != nil {
		return false
	}
	// 收集候选：可抢占、活动中、且优先级严格低于本任务的任务。
	var victims []*domain.Task
	for _, t := range all {
		if t.ID == task.ID || !t.Preemptible || !t.IsActive() {
			continue
		}
		if t.Priority >= task.Priority {
			continue
		}
		victims = append(victims, t)
	}
	if len(victims) == 0 {
		return false
	}
	// 优先抢占优先级最低者，尽量以最小代价腾出资源。
	sort.Slice(victims, func(i, j int) bool {
		return victims[i].Priority < victims[j].Priority
	})
	preempted := false
	for _, v := range victims {
		reason := fmt.Sprintf("被高优先级任务 %s 抢占", task.Name)
		if err := s.terminateActiveTask(ctx, v, domain.FailurePreempted, reason); err != nil {
			continue
		}
		preempted = true
		// 抢占一个通常即可释放足够资源；释放后交由调用方重试调度。
		break
	}
	return preempted
}

// ScheduleBatch 一次性调度队列中至多 maxBatch 个任务（批量调度）。
//
// 调度器单实例逐个决策在集群规模变大时会成为吞吐瓶颈。批量调度在一次
// 循环内连续弹出并调度多个任务，减少调度间隔空转，是支撑更大集群的
// "调度域分片 + 批量调度"能力的基础。
//
// 参数：
//   - ctx:      上下文。
//   - maxBatch: 单批最多处理的任务数；<=0 时取 1。
//
// 返回值：
//   - int:   实际处理的任务数（队列提前为空则小于 maxBatch）。
//   - error: 处理过程中的首个错误。
func (s *TaskService) ScheduleBatch(ctx context.Context, maxBatch int) (int, error) {
	if maxBatch <= 0 {
		maxBatch = 1
	}
	processed := 0
	for i := 0; i < maxBatch; i++ {
		ok, err := s.ScheduleOnce(ctx)
		if err != nil {
			return processed, err
		}
		if !ok {
			break // 队列已空
		}
		processed++
	}
	return processed, nil
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

// onTaskFinished 处理任务终态后的副作用（尽力而为，失败不影响主流程）。
//
// 依次执行：记录用量 -> 按卡时计费 -> 归档日志 -> 主动通知。
// 各步骤所需服务未注入时静默跳过。
func (s *TaskService) onTaskFinished(ctx context.Context, task *domain.Task) {
	var gpuHours float64
	// 计量：记录用量并得到本次卡时。
	if s.metering != nil {
		if rec, err := s.metering.RecordTaskUsage(ctx, task); err == nil && rec != nil {
			gpuHours = rec.GPUHours
		}
	}
	// 计费：把卡时累加到计费主体（项目优先，否则用户）配额。
	if s.quota != nil && gpuHours > 0 {
		subject := task.Owner
		if task.ProjectID != "" {
			subject = task.ProjectID
		}
		_ = s.quota.Charge(ctx, subject, gpuHours)
	}
	// 日志归档：把实时日志落到对象存储长期留存，并关闭日志流。
	if s.logs != nil && task.LogPath != "" {
		_ = s.logs.Archive(ctx, task.ID, task.LogPath)
	}
	// 通知：任务终态主动触达用户。
	if s.notifier != nil {
		title := fmt.Sprintf("任务 %s 已%s", task.Name, statusText(task.Status))
		content := fmt.Sprintf("任务 ID: %s\n状态: %s\n退出码: %d\n说明: %s",
			task.ID, task.Status, task.ExitCode, task.Message)
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
