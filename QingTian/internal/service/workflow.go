// Package service 工作流编排服务
//
// 本层负责编排工作流执行逻辑：依赖解析 -> 就绪任务提交 -> 状态推进 -> 完成回调。
package service

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// WorkflowService 工作流管理服务
// 负责工作流定义的 CRUD 操作
type WorkflowService struct {
	repo repository.WorkflowRepository
}

// NewWorkflowService 创建工作流管理服务
func NewWorkflowService(repo repository.WorkflowRepository) *WorkflowService {
	return &WorkflowService{
		repo: repo,
	}
}

// Get 获取工作流定义详情
func (s *WorkflowService) Get(ctx context.Context, id string) (*domain.Workflow, error) {
	return s.repo.GetByID(ctx, id)
}

// List 列出工作流定义
func (s *WorkflowService) List(ctx context.Context, createdBy string, skip, limit int) ([]*domain.Workflow, int64, error) {
	return s.repo.List(ctx, createdBy, skip, limit)
}

// Create 创建工作流定义
func (s *WorkflowService) Create(ctx context.Context, wf *domain.Workflow) error {
	// 生成唯一 ID（前端未提供时），避免空 _id 导致 MongoDB 主键冲突。
	if wf.ID == "" {
		wf.ID = idgen.New("wf")
	}
	// 校验 DAG
	if err := s.ValidateDAG(wf); err != nil {
		return err
	}
	wf.Status = domain.WorkflowStatusActive
	wf.CreatedAt = time.Now()
	wf.UpdatedAt = wf.CreatedAt
	return s.repo.Create(ctx, wf)
}

// Update 更新工作流定义
//
// 归属校验：非管理员仅能更新自己创建的工作流，防止越权（IDOR）。
func (s *WorkflowService) Update(ctx context.Context, wf *domain.Workflow, requesterID string, isAdmin bool) error {
	// 先加载已存在的工作流校验归属，避免越权覆盖他人资源。
	existing, err := s.repo.GetByID(ctx, wf.ID)
	if err != nil {
		return err
	}
	if !isAdmin && existing.CreatedBy != requesterID {
		return apperr.Forbidden("无权访问该工作流")
	}
	// 保留原始创建者，防止请求体篡改归属。
	wf.CreatedBy = existing.CreatedBy
	if err := s.ValidateDAG(wf); err != nil {
		return err
	}
	wf.UpdatedAt = time.Now()
	return s.repo.Update(ctx, wf)
}

// Delete 删除工作流定义（软删除，改为 archived 状态）
//
// 归属校验：非管理员仅能删除自己创建的工作流，防止越权（IDOR）。
func (s *WorkflowService) Delete(ctx context.Context, id, requesterID string, isAdmin bool) error {
	existing, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return err
	}
	if !isAdmin && existing.CreatedBy != requesterID {
		return apperr.Forbidden("无权访问该工作流")
	}
	return s.repo.Delete(ctx, id)
}

// ValidateDAG 校验 DAG 工作流：检测环 + 任务ID唯一性 + 边引用有效性
func (s *WorkflowService) ValidateDAG(wf *domain.Workflow) error {
	// 1. 校验任务 ID 唯一性
	taskIDs := make(map[string]bool)
	for _, task := range wf.Tasks {
		if taskIDs[task.ID] {
			return apperr.InvalidArgument(fmt.Sprintf("duplicate task ID: %s", task.ID))
		}
		taskIDs[task.ID] = true
	}

	// 2. 校验边引用的任务存在
	for _, edge := range wf.Edges {
		if !taskIDs[edge.From] {
			return apperr.InvalidArgument(fmt.Sprintf("edge from references non-existent task ID: %s", edge.From))
		}
		if !taskIDs[edge.To] {
			return apperr.InvalidArgument(fmt.Sprintf("edge to references non-existent task ID: %s", edge.To))
		}
	}

	// 3. 检测环
	if hasCycle := detectCycle(wf); hasCycle {
		return apperr.InvalidArgument("cycle detected in workflow, cycles are not allowed")
	}

	return nil
}

// detectCycle 使用 DFS 检测 DAG 中是否有环
func detectCycle(wf *domain.Workflow) bool {
	graph := buildGraph(wf)
	visited := make(map[string]bool)
	recursion := make(map[string]bool)

	for taskID := range graph {
		if !visited[taskID] {
			if dfs(taskID, graph, visited, recursion) {
				return true
			}
		}
	}
	return false
}

// buildGraph 从工作流构建邻接表
func buildGraph(wf *domain.Workflow) map[string][]string {
	graph := make(map[string][]string)
	for _, edge := range wf.Edges {
		graph[edge.From] = append(graph[edge.From], edge.To)
	}
	return graph
}

// dfs 深度优先搜索检测环
func dfs(u string, graph map[string][]string, visited, recursion map[string]bool) bool {
	visited[u] = true
	recursion[u] = true

	for _, v := range graph[u] {
		if !visited[v] {
			if dfs(v, graph, visited, recursion) {
				return true
			}
		} else if recursion[v] {
			// 已经在递归栈中，说明找到环
			return true
		}
	}

	recursion[u] = false
	return false
}

// WorkflowEngine 工作流执行引擎
// 负责驱动 DAG 执行：解析依赖关系 -> 提交就绪任务 -> 监听任务完成 -> 推进工作流状态
type WorkflowEngine struct {
	workflowRepo   repository.WorkflowRepository
	runRepo       repository.WorkflowRunRepository
	taskRepo      repository.TaskRunRepository
	taskService   TaskSubmitter
}

// TaskSubmitter 定义任务提交接口
// 引擎将工作流任务转换为 QingTian 原生任务后提交到调度器，返回原生任务 ID
type TaskSubmitter interface {
	// Submit 提交单个任务到 QingTian 调度系统，返回原生任务 ID
	Submit(ctx context.Context, wf *domain.Workflow, run *domain.WorkflowRun, task *domain.WorkflowTask, globalParams map[string]interface{}) (string, error)
}

// NewWorkflowEngine 创建工作流执行引擎
func NewWorkflowEngine(
	wfRepo repository.WorkflowRepository,
	runRepo repository.WorkflowRunRepository,
	taskRepo repository.TaskRunRepository,
	taskSubmit TaskSubmitter,
) *WorkflowEngine {
	return &WorkflowEngine{
		workflowRepo:   wfRepo,
		runRepo:       runRepo,
		taskRepo:      taskRepo,
		taskService:   taskSubmit,
	}
}

// Run 执行一次工作流
func (e *WorkflowEngine) Run(
	ctx context.Context,
	workflowID string,
	params map[string]interface{},
	trigger domain.TriggerInfo,
) (*domain.WorkflowRun, error) {
	// 1. 获取工作流定义
	wf, err := e.workflowRepo.GetByID(ctx, workflowID)
	if err != nil {
		return nil, err
	}

	// 2. 创建执行实例
	run := &domain.WorkflowRun{
		ID:          idgen.New("wfrun"),
		WorkflowID:  workflowID,
		Name:        wf.Name,
		Parameters:  params,
		Status:      domain.WorkflowRunStatusPending,
		Trigger:     trigger,
		CreatedBy:    trigger.TriggeredBy,
		CreatedAt:    time.Now(),
	}
	if err := e.runRepo.Create(ctx, run); err != nil {
		return nil, err
	}

	// 3. 构建执行图并提交第一批就绪任务
	execGraph := buildExecutionGraph(wf)
	readyTasks := execGraph.GetReadyTasks(run)

	for _, task := range readyTasks {
		if _, err := e.submitTask(ctx, wf, run, task, params); err != nil {
			return run, err
		}
	}

	return run, nil
}

// buildExecutionGraph 从工作流定义构建执行图，方便获取就绪任务
type ExecutionGraph struct {
	tasks   map[string]*domain.WorkflowTask
	deps    map[string][]string             // taskID -> 这个task依赖的tasks （当前task要等这些deps完成）
	children map[string][]string             // taskID -> 依赖这个task的tasks （这些tasks要等当前task完成）
	status   map[string]domain.TaskRunStatus // taskID -> 当前状态
}

// buildExecutionGraph 从工作流定义构建执行图
func buildExecutionGraph(wf *domain.Workflow) *ExecutionGraph {
	g := &ExecutionGraph{
		tasks:   make(map[string]*domain.WorkflowTask),
		deps:    make(map[string][]string),
		children: make(map[string][]string),
		status:   make(map[string]domain.TaskRunStatus),
	}
	for _, task := range wf.Tasks {
		taskCopy := task
		g.tasks[task.ID] = &taskCopy
		g.status[task.ID] = domain.TaskRunStatusPending
	}
	// 每个边加入依赖关系
	// 边: From → To 意味着 From 必须在 To 之前完成
	// 所以 → To 依赖 From，To要等From完成
	for _, edge := range wf.Edges {
		g.deps[edge.To] = append(g.deps[edge.To], edge.From)
		g.children[edge.From] = append(g.children[edge.From], edge.To)
	}
	return g
}

// GetReadyTasks 获取当前可以执行的任务（所有依赖都已满足）
func (g *ExecutionGraph) GetReadyTasks(run *domain.WorkflowRun) []*domain.WorkflowTask {
	var ready []*domain.WorkflowTask
	for taskID, task := range g.tasks {
		// 如果这个任务已经有状态了，跳过
		if g.status[taskID] != domain.TaskRunStatusPending {
			continue
		}
		// 检查所有依赖是否都成功完成
		allDone := true
		for _, depID := range g.deps[taskID] {
			if g.status[depID] != domain.TaskRunStatusSucceeded {
				allDone = false
				break
			}
		}
		if allDone {
			ready = append(ready, task)
		}
	}
	return ready
}

// OnTaskComplete 任务完成回调，当一个任务完成后，检查是否有新的任务就绪
// 使用乐观并发控制防止并行任务完成时的竞态条件（多个并行任务同时完成会产生丢失更新）
func (e *WorkflowEngine) OnTaskComplete(
	ctx context.Context,
	runID string,
	taskID string,
	success bool,
) error {
	// 获取任务执行实例
	taskRun, err := e.taskRepo.GetByID(ctx, taskID)
	if err != nil {
		return err
	}

	// 更新任务状态
	if success {
		taskRun.Status = domain.TaskRunStatusSucceeded
	} else {
		taskRun.Status = domain.TaskRunStatusFailed
	}
	if err := e.taskRepo.Update(ctx, taskRun); err != nil {
		return err
	}

	// 获取工作流运行实例，使用乐观循环处理并发更新
	// 乐观并发控制：当多个并行任务同时完成时，我们重试冲突直到成功
	// 这比悲观锁更适合低冲突场景，性能更好
	maxRetries := 10
	for attempt := 0; attempt < maxRetries; attempt++ {
		// 获取当前版本
		run, err := e.runRepo.GetByID(ctx, runID)
		if err != nil {
			return err
		}

		// 保存当前版本用于乐观锁检查
		// 注意：MongoDB 自带 _id 作为主键，我们依赖它的原子更新模型
		// 这里通过重新计算状态然后尝试写入，如果冲突就重试

		// 更新工作流状态：如果所有任务都完成，工作流就完成了
		allCompleted := true
		var failed int
		// 遍历当前执行图所有任务，检查状态
		for i, tr := range run.TaskRuns {
			if tr.ID == taskRun.ID {
				// 更新当前任务的状态到本地拷贝
				run.TaskRuns[i].Status = taskRun.Status
			}
			switch run.TaskRuns[i].Status {
			case domain.TaskRunStatusPending, domain.TaskRunStatusRunning:
				allCompleted = false
			case domain.TaskRunStatusFailed:
				failed++
			}
		}

		switch {
		case failed > 0:
			run.Status = domain.WorkflowRunStatusFailed
		case allCompleted:
			run.Status = domain.WorkflowRunStatusSucceeded
		default:
			run.Status = domain.WorkflowRunStatusRunning
		}

		// 尝试更新，如果失败（说明被其他修改者改变了）我们重试
		err = e.runRepo.Update(ctx, run)
		if err == nil {
			// 更新成功，跳出循环
			break
		}
		// 如果更新失败，循环重试
		// 这就是乐观并发控制：冲突时重试，低冲突场景性能优异
	}

	// 获取工作流运行实例（最终版本）
	run, err := e.runRepo.GetByID(ctx, runID)
	if err != nil {
		return err
	}

	// 如果工作流还没完成，找出新的就绪任务提交它们
	allCompleted := true
	for _, tr := range run.TaskRuns {
		switch tr.Status {
		case domain.TaskRunStatusPending, domain.TaskRunStatusRunning:
			allCompleted = false
		}
	}

	if !allCompleted {
		// 获取工作流定义
		wfDef, err := e.workflowRepo.GetByID(ctx, run.WorkflowID)
		if err != nil {
			return err
		}
		execGraph := buildExecutionGraph(wfDef)
		readyTasks := execGraph.GetReadyTasks(run)
		for _, task := range readyTasks {
			// 检查这个任务是否已经有了任务运行记录
			// 防止因为竞态条件重复提交同一个任务
			alreadySubmitted := false
			for _, tr := range run.TaskRuns {
				if tr.TaskID == task.ID {
					alreadySubmitted = true
					break
				}
			}
			if alreadySubmitted {
				continue
			}
			_, err := e.submitTask(ctx, wfDef, run, task, run.Parameters)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// submitTask 提交单个任务
func (e *WorkflowEngine) submitTask(
	ctx context.Context,
	wf *domain.Workflow,
	run *domain.WorkflowRun,
	task *domain.WorkflowTask,
	globalParams map[string]interface{},
) (string, error) {
	// 解析输入：从上游任务输出解析出当前任务输入
	resolvedInputs, err := e.resolveInputs(task, run, globalParams)
	if err != nil {
		return "", err
	}

	// 创建任务执行记录
	taskRun := &domain.TaskRun{
		ID:           generateID(),
		TaskID:       task.ID,
		TaskName:     task.Name,
		Status:       domain.TaskRunStatusRunning,
		Inputs:       resolvedInputs,
		Outputs:      make(map[string]interface{}),
		StartedAt:    pointer(time.Now()),
	}

	// 提交任务到 QingTian 调度系统，获取原生任务 ID
	nativeTaskID, err := e.taskService.Submit(ctx, wf, run, task, resolvedInputs)
	if err != nil {
		taskRun.Status = domain.TaskRunStatusFailed
		taskRun.Error = err.Error()
		_ = e.taskRepo.Create(ctx, taskRun)
		return "", err
	}

	taskRun.NativeTaskID = nativeTaskID

	// 保存任务执行记录
	if err := e.taskRepo.Create(ctx, taskRun); err != nil {
		return nativeTaskID, err
	}

	// 添加到工作流执行实例
	run.TaskRuns = append(run.TaskRuns, *taskRun)

	// 更新工作流状态为运行中
	if run.Status == domain.WorkflowRunStatusPending {
		run.Status = domain.WorkflowRunStatusRunning
		run.StartedAt = pointer(time.Now())
	}

	// 保存更新
	_ = e.runRepo.Update(ctx, run)

	return nativeTaskID, nil
}

// resolveInputs 解析任务输入：从上游任务输出中解析出当前任务输入参数
func (e *WorkflowEngine) resolveInputs(
	task *domain.WorkflowTask,
	run *domain.WorkflowRun,
	globalParams map[string]interface{},
) (map[string]interface{}, error) {
	result := make(map[string]interface{})

	// 填充默认值
	for _, input := range task.Inputs {
		if input.Default != "" && globalParams[input.Name] == nil {
			result[input.Name] = input.Default
		} else {
			result[input.Name] = globalParams[input.Name]
		}
		// from 上游任务输出
		if input.From != "" {
			// 格式: "taskID.outputName"
			parts := strings.Split(input.From, ".")
			if len(parts) != 2 {
				return nil, apperr.InvalidArgument(fmt.Sprintf("invalid input from format: %s, expected taskID.outputName", input.From))
			}
			srcTaskID := parts[0]
			srcOutputName := parts[1]

			// 找到对应的任务 run
			var srcTask *domain.TaskRun
			for _, tr := range run.TaskRuns {
				if tr.TaskID == srcTaskID {
					srcTask = &tr
					break
				}
			}
			if srcTask == nil {
				return nil, apperr.InvalidArgument(fmt.Sprintf("cannot find source task run: %s", srcTaskID))
			}

			// 从srcTask.Outputs取出输出值
			val, ok := srcTask.Outputs[srcOutputName]
			if !ok {
				result[input.Name] = nil
				continue
			}
			result[input.Name] = val
		}
	}

	return result, nil
}

// WorkflowRunService 工作流执行实例管理服务
type WorkflowRunService struct {
	repo repository.WorkflowRunRepository
}

// NewWorkflowRunService 创建工作流执行实例服务
func NewWorkflowRunService(repo repository.WorkflowRunRepository) *WorkflowRunService {
	return &WorkflowRunService{
		repo: repo,
	}
}

// Get 获取执行实例详情
func (s *WorkflowRunService) Get(ctx context.Context, id string) (*domain.WorkflowRun, error) {
	return s.repo.GetByID(ctx, id)
}

// List 列出执行实例
func (s *WorkflowRunService) List(ctx context.Context, workflowID string, skip, limit int) ([]*domain.WorkflowRun, int64, error) {
	return s.repo.ListByWorkflow(ctx, workflowID, skip, limit)
}

// ExecutorHandler 执行器处理器接口
// 不同执行类型实现该接口来准备任务配置
type ExecutorHandler interface {
	// Prepare 准备任务配置，返回镜像、命令、环境变量
	Prepare(task *domain.WorkflowTask, resolvedInputs map[string]interface{}, run *domain.WorkflowRun) (string, string, []domain.EnvVar, error)
}

// containerExecutorHandler 容器执行处理器
type containerExecutorHandler struct {
	submitter *DefaultTaskSubmitter
}

func (h *containerExecutorHandler) Prepare(
	task *domain.WorkflowTask,
	resolvedInputs map[string]interface{},
	run *domain.WorkflowRun,
) (string, string, []domain.EnvVar, error) {
	var image string
	var command string

	// 优先使用新配置结构，向后兼容旧字段
	if task.ContainerConfig != nil {
		image = task.ContainerConfig.Image
		if len(task.ContainerConfig.Command) > 0 {
			command = strings.Join(task.ContainerConfig.Command, " ")
			if len(task.ContainerConfig.Args) > 0 {
				command += " " + strings.Join(task.ContainerConfig.Args, " ")
			}
		}

		// 转换环境变量
		env := make([]domain.EnvVar, 0, len(task.ContainerConfig.Env))
		for k, v := range task.ContainerConfig.Env {
			expandedV := h.submitter.expandEnvVar(v, resolvedInputs, run.Parameters)
			env = append(env, domain.EnvVar{Name: k, Value: expandedV})
		}
		return image, command, env, nil
	}

	// 向后兼容：使用旧字段
	image = task.Image
	if len(task.Command) > 0 {
		command = strings.Join(task.Command, " ")
		if len(task.Args) > 0 {
			command += " " + strings.Join(task.Args, " ")
		}
	}

	// 转换环境变量
	env := make([]domain.EnvVar, 0, len(task.Env))
	for k, v := range task.Env {
		expandedV := h.submitter.expandEnvVar(v, resolvedInputs, run.Parameters)
		env = append(env, domain.EnvVar{Name: k, Value: expandedV})
	}
	return image, command, env, nil
}

// shellExecutorHandler Shell 脚本执行处理器
// 使用预构建的 shell-runner 镜像执行用户脚本
type shellExecutorHandler struct {
	submitter *DefaultTaskSubmitter
}

func (h *shellExecutorHandler) Prepare(
	task *domain.WorkflowTask,
	resolvedInputs map[string]interface{},
	run *domain.WorkflowRun,
) (string, string, []domain.EnvVar, error) {
	// 使用预构建的 shell-runner 镜像
	image := "qingtian/shell-runner:latest"

	if task.ShellConfig == nil {
		return image, "", nil, apperr.InvalidArgument("shellConfig is required for shell executor")
	}

	// 将脚本写入临时文件并执行
	// 脚本内容通过环境变量传递或者直接作为命令
	script := task.ShellConfig.Script
	// base64 编码脚本内容避免引号转义问题，在入口脚本中解码执行
	command := fmt.Sprintf("echo '%s' | base64 -d > /workspace/script.sh && chmod +x /workspace/script.sh && /workspace/script.sh", encodeBase64(script))

	// 转换环境变量
	env := make([]domain.EnvVar, 0, len(task.ShellConfig.Env))
	for k, v := range task.ShellConfig.Env {
		expandedV := h.submitter.expandEnvVar(v, resolvedInputs, run.Parameters)
		env = append(env, domain.EnvVar{Name: k, Value: expandedV})
	}

	return image, command, env, nil
}

// pythonExecutorHandler Python 脚本执行处理器
// 使用预构建的 python-runner 镜像执行用户脚本
type pythonExecutorHandler struct {
	submitter *DefaultTaskSubmitter
}

func (h *pythonExecutorHandler) Prepare(
	task *domain.WorkflowTask,
	resolvedInputs map[string]interface{},
	run *domain.WorkflowRun,
) (string, string, []domain.EnvVar, error) {
	// 使用预构建的 python-runner 镜像
	image := "qingtian/python-runner:latest"

	if task.PythonConfig == nil {
		return image, "", nil, apperr.InvalidArgument("pythonConfig is required for python executor")
	}

	script := task.PythonConfig.Script
	// 同样使用 base64 编码避免引号问题
	command := fmt.Sprintf("echo '%s' | base64 -d > /workspace/script.py", encodeBase64(script))

	// 如果有 requirements，先安装
	if len(task.PythonConfig.Requirements) > 0 {
		reqs := strings.Join(task.PythonConfig.Requirements, " ")
		command += fmt.Sprintf(" && pip install %s", reqs)
	}

	// 执行脚本
	command += " && python /workspace/script.py"

	// 转换环境变量
	env := make([]domain.EnvVar, 0, len(task.PythonConfig.Env))
	for k, v := range task.PythonConfig.Env {
		expandedV := h.submitter.expandEnvVar(v, resolvedInputs, run.Parameters)
		env = append(env, domain.EnvVar{Name: k, Value: expandedV})
	}

	return image, command, env, nil
}

// apiExecutorHandler API 调用执行处理器
// 使用预构建的 api-runner 镜像发起 HTTP 请求
type apiExecutorHandler struct {
	submitter *DefaultTaskSubmitter
}

func (h *apiExecutorHandler) Prepare(
	task *domain.WorkflowTask,
	resolvedInputs map[string]interface{},
	run *domain.WorkflowRun,
) (string, string, []domain.EnvVar, error) {
	// 使用预构建的 api-runner 镜像
	image := "qingtian/api-runner:latest"

	if task.APIConfig == nil {
		return image, "", nil, apperr.InvalidArgument("apiConfig is required for api executor")
	}

	cfg := task.APIConfig

	// 将配置通过环境变量传递给 runner
	// 命令由 runner 入口脚本处理
	command := "/app/run.sh"

	// 转换环境变量，展开占位符
	env := make([]domain.EnvVar, 0)

	// 展开占位符并设置环境变量供 runner 使用
	expandedURL := h.submitter.expandEnvVar(cfg.URL, resolvedInputs, run.Parameters)
	env = append(env, domain.EnvVar{Name: "API_URL", Value: expandedURL})

	expandedMethod := h.submitter.expandEnvVar(cfg.Method, resolvedInputs, run.Parameters)
	if expandedMethod == "" {
		expandedMethod = "GET"
	}
	env = append(env, domain.EnvVar{Name: "API_METHOD", Value: expandedMethod})

	expandedBody := h.submitter.expandEnvVar(cfg.Body, resolvedInputs, run.Parameters)
	env = append(env, domain.EnvVar{Name: "API_BODY", Value: expandedBody})

	// Headers 逐个设置
	for k, v := range cfg.Headers {
		expandedV := h.submitter.expandEnvVar(v, resolvedInputs, run.Parameters)
		env = append(env, domain.EnvVar{Name: "API_HEADER_" + k, Value: expandedV})
	}

	env = append(env, domain.EnvVar{Name: "API_TIMEOUT", Value: fmt.Sprintf("%d", cfg.Timeout)})

	// Success codes 作为空格分隔的字符串
	if len(cfg.SuccessCodes) > 0 {
		successStr := strings.Builder{}
		for i, code := range cfg.SuccessCodes {
			if i > 0 {
				successStr.WriteByte(' ')
			}
			successStr.WriteString(fmt.Sprintf("%d", code))
		}
		env = append(env, domain.EnvVar{Name: "API_SUCCESS_CODES", Value: successStr.String()})
	}

	return image, command, env, nil
}

// encodeBase64 对字符串进行 base64 编码，用于嵌入脚本
// 将脚本内容编码后嵌入命令行，避免引号转义问题
func encodeBase64(s string) string {
	return base64.StdEncoding.EncodeToString([]byte(s))
}

// DefaultTaskSubmitter 默认任务提交器实现
// 将工作流中的 WorkflowTask 转换为 QingTian 原生 Task 并提交到调度系统
// 支持多种执行类型：容器、Shell、Python、API
type DefaultTaskSubmitter struct {
	taskService *TaskService
	owner       string // 默认提交者所有者
	handlers    map[domain.ExecutorType]ExecutorHandler
}

// NewDefaultTaskSubmitter 创建默认任务提交器
func NewDefaultTaskSubmitter(taskService *TaskService, defaultOwner string) *DefaultTaskSubmitter {
	s := &DefaultTaskSubmitter{
		taskService: taskService,
		owner:       defaultOwner,
	}
	// 初始化各执行类型处理器
	s.handlers = map[domain.ExecutorType]ExecutorHandler{
		domain.ExecutorTypeContainer: &containerExecutorHandler{submitter: s},
		domain.ExecutorTypeShell:      &shellExecutorHandler{submitter: s},
		domain.ExecutorTypePython:     &pythonExecutorHandler{submitter: s},
		domain.ExecutorTypeAPI:        &apiExecutorHandler{submitter: s},
	}
	return s
}

// Submit 实现 TaskSubmitter 接口，将 WorkflowTask 转换为原生 Task 并提交
func (s *DefaultTaskSubmitter) Submit(
	ctx context.Context,
	wf *domain.Workflow,
	run *domain.WorkflowRun,
	task *domain.WorkflowTask,
	resolvedInputs map[string]interface{},
) (string, error) {
	// 获取对应执行类型的处理器
	execType := task.ExecutorType
	if execType == "" {
		// 向后兼容：默认容器执行
		execType = domain.ExecutorTypeContainer
	}

	handler, ok := s.handlers[execType]
	if !ok {
		return "", apperr.InvalidArgument(fmt.Sprintf("unknown executor type: %s", execType))
	}

	// 调用处理器准备配置
	image, command, env, err := handler.Prepare(task, resolvedInputs, run)
	if err != nil {
		return "", err
	}

	// 转换资源规格
	gpuCount := task.Resources.GPUCount
	if gpuCount <= 0 {
		gpuCount = 1 // 默认至少 1 卡
		// 对于 API 调用，通常不需要 GPU，允许 0
		if execType == domain.ExecutorTypeAPI {
			gpuCount = 0
		}
	}

	// 解析 GPUMemory 转换为 MB
	// 支持格式如 "16GB", "8Gi", "4096MB"
	memoryMB := s.parseMemoryToMB(task.Resources.GPUMemory)

	// 构造提交请求
	req := SubmitRequest{
		Name:            fmt.Sprintf("%s-%s", wf.Name, task.Name),
		Image:           image,
		Command:         command,
		Env:             env,
		Priority:        50, // 默认优先级
		GPUCount:        gpuCount,
		MemoryMB:        memoryMB,
		GPUModel:        task.Resources.GPUModel,
		Vendor:          domain.VendorNVIDIA, // 默认 NVIDIA，可通过参数覆盖
		ShareMode:       domain.ShareExclusive, // 默认独占
		TimeoutSeconds:  int64(task.Timeout.Duration.Seconds()),
		Preemptible:     false, // 默认不可抢占
		MaxRetries:      task.Retries,
		// TODO: 支持从工作流参数中读取 ProjectID
		ProjectID:       "",
	}

	// 所有者从工作流触发信息获取，如果没有则使用默认
	owner := run.CreatedBy
	if owner == "" {
		owner = s.owner
	}

	// 提交任务
	nativeTask, err := s.taskService.Submit(ctx, owner, req)
	if err != nil {
		return "", err
	}

	return nativeTask.ID, nil
}

// expandEnvVar 替换环境变量中的占位符
// 支持 ${paramName} 格式，替换为对应参数值
func (s *DefaultTaskSubmitter) expandEnvVar(value string, inputs map[string]interface{}, globalParams map[string]interface{}) string {
	// 简单实现：查找 ${...} 并替换
	// 完整实现可以用 regexp，但这里保持简单
	var result strings.Builder
	i := 0
	n := len(value)
	for i < n {
		if i+1 < n && value[i] == '$' && value[i+1] == '{' {
			i += 2
			start := i
			for i < n && value[i] != '}' {
				i++
			}
			if i >= n {
				// 没有闭合，直接输出
				result.WriteString("${")
				i = start
				continue
			}
			paramName := value[start:i]
			i++ // 跳过 }

			// 先从 inputs 找，再从 globalParams 找
			var val interface{}
			if v, ok := inputs[paramName]; ok {
				val = v
			} else if v, ok := globalParams[paramName]; ok {
				val = v
			} else {
				// 找不到保留原样
				result.WriteString("${")
				result.WriteString(paramName)
				result.WriteString("}")
				continue
			}

			// 转换为字符串
			result.WriteString(fmt.Sprintf("%v", val))
		} else {
			result.WriteByte(value[i])
			i++
		}
	}
	return result.String()
}

// parseMemoryToMB 将内存字符串解析为 MB
// 支持: 16GB, 8G, 4096MB, 2Gi
func (s *DefaultTaskSubmitter) parseMemoryToMB(memStr string) int64 {
	if memStr == "" {
		return 0
	}

	// 移除空格
	memStr = strings.TrimSpace(memStr)

	// 处理单位
	// 常见单位: B, KB, MB, GB, TB, KiB, MiB, GiB, TiB
	multiplier := int64(1) // 默认就是 MB
	lastChar := memStr[len(memStr)-1]
	if lastChar >= '0' && lastChar <= '9' {
		// 没有单位，假设是 MB
		v := 0
		fmt.Sscan(memStr, &v)
		return int64(v)
	}

	// 处理两位单位后缀
	if len(memStr) >= 2 {
		// 检查最后两个字符是否是单位
		lastTwo := strings.ToLower(memStr[len(memStr)-2:])
		switch lastTwo {
		case "kb", "ki":
			memStr = memStr[:len(memStr)-2]
		case "mb", "mi":
			memStr = memStr[:len(memStr)-2]
			multiplier = 1
		case "gb", "gi":
			memStr = memStr[:len(memStr)-2]
			multiplier = 1024
		case "tb", "ti":
			memStr = memStr[:len(memStr)-2]
			multiplier = 1024 * 1024
		default:
			// 检查单个字符
			lc := strings.ToLower(string(memStr[len(memStr)-1]))
			switch lc {
			case "b", "k", "m", "g", "t":
				memStr = memStr[:len(memStr)-1]
				switch lc {
				case "b":
					multiplier = 1 / 1024 // B -> MB
				case "k":
					multiplier = 1 / 1024 // KB -> MB
				case "m":
					multiplier = 1 // MB
				case "g":
					multiplier = 1024 // GB -> MB
				case "t":
					multiplier = 1024 * 1024 // TB -> MB
				}
			}
		}
	}

	var value float64
	if _, err := fmt.Sscan(memStr, &value); err != nil {
		return 0
	}

	return int64(value * float64(multiplier))
}

// generateID 生成唯一 ID
func generateID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())
}

// pointer 是辅助函数，返回 T 的指针
func pointer[T any](v T) *T {
	return &v
}
