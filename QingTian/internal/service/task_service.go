// 本文件实现任务服务：任务提交、调度、下发、状态回传处理与查询。
//
// 任务服务串联调度器、账本、仓储与消息队列，是"提交->调度->执行->回传"闭环的核心。
package service

import (
	"context"
	"encoding/json"
	"time"

	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/mq"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
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
type TaskService struct {
	repo     repository.TaskRepository
	nodeRepo repository.NodeRepository
	queue    cache.PriorityQueue
	producer mq.Producer
	sched    *scheduler.Scheduler
	now      func() time.Time
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

// SubmitRequest 是任务提交请求。
type SubmitRequest struct {
	Name        string                 `json:"name"`
	Image       string                 `json:"image"`
	Command     string                 `json:"command"`
	Resource    domain.ResourceRequest `json:"resource"`
	Priority    int                    `json:"priority"`
	DatasetPath string                 `json:"dataset_path"`
}

// Submit 提交一个新任务，落库并进入待调度队列。
//
// 参数：
//   - ctx:   上下文。
//   - owner: 提交者用户 ID。
//   - req:   提交请求。
//
// 返回值：
//   - *domain.Task: 创建成功的任务（状态为 pending）。
//   - error:        参数非法或持久化失败时返回。
func (s *TaskService) Submit(ctx context.Context, owner string, req SubmitRequest) (*domain.Task, error) {
	if req.Name == "" || req.Image == "" {
		return nil, apperr.InvalidArgument("任务名称与镜像不能为空")
	}
	if req.Resource.GPUCount <= 0 {
		return nil, apperr.InvalidArgument("GPU 数量必须大于 0")
	}
	now := s.now()
	task := &domain.Task{
		ID:          idgen.New("task"),
		Name:        req.Name,
		Owner:       owner,
		Image:       req.Image,
		Command:     req.Command,
		Resource:    req.Resource,
		Priority:    req.Priority,
		Status:      domain.TaskPending,
		DatasetPath: req.DatasetPath,
		ResultPath:  "results/" + req.Name + "/",
		CreatedAt:   now,
		UpdatedAt:   now,
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
		// 无可用节点：重新入队，稍后重试。
		_ = s.queue.Push(ctx, task.ID, float64(task.Priority))
		return true, nil
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
		TaskID:        task.ID,
		NodeID:        task.NodeID,
		Image:         task.Image,
		Command:       task.Command,
		AllocatedGPUs: task.AllocatedGPUs,
		DatasetPath:   task.DatasetPath,
		ResultPath:    task.ResultPath,
	}
	body, _ := json.Marshal(msg)
	if err := s.producer.Publish(ctx, mq.TaskDispatchTopic(task.NodeID), body); err != nil {
		return true, err
	}
	return true, nil
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
	return s.repo.Update(ctx, task)
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
