# QingTian 工作流编排与调度系统设计文档

> 版本：v1.0  
> 日期：2026-07-21  
> 状态：设计评审中

---

## 1. 功能概述

### 1.1 目标

为 QingTian 平台增加**工作流编排、任务执行、定时调度**能力，支持：

- DAG（有向无环图）任务编排
- 任务依赖管理与自动触发
- 定时任务调度（Cron 表达式）
- 工作流模板复用
- 执行历史追踪与重试

### 1.2 核心概念

| 概念 | 说明 |
|------|------|
| **Workflow（工作流）** | 由多个 Task 组成的 DAG，定义任务执行顺序和依赖关系 |
| **WorkflowTemplate（工作流模板）** | 可复用的工作流定义，支持参数化 |
| **WorkflowRun（工作流实例）** | 工作流的一次具体执行实例 |
| **Task（任务）** | 工作流中的单个执行单元（对应原 Task 模型）|
| **TaskRun（任务实例）** | 任务的一次具体执行 |
| **Schedule（定时计划）** | 定义工作流的定时触发规则 |

---

## 2. 数据模型设计

### 2.1 Workflow（工作流定义）

```go
// Workflow 工作流定义
type Workflow struct {
    ID          string            `json:"id" bson:"_id"`
    Name        string            `json:"name" bson:"name"`
    Description string            `json:"description" bson:"description"`
    Version     string            `json:"version" bson:"version"`
    
    // 工作流定义
    Tasks       []WorkflowTask    `json:"tasks" bson:"tasks"`       // 任务列表
    Edges       []Edge            `json:"edges" bson:"edges"`       // 依赖边
    
    // 参数定义
    Parameters  []Parameter       `json:"parameters" bson:"parameters"` // 可传入参数
    
    // 元数据
    Labels      map[string]string `json:"labels" bson:"labels"`
    CreatedBy   string            `json:"createdBy" bson:"createdBy"`
    CreatedAt   time.Time         `json:"createdAt" bson:"createdAt"`
    UpdatedAt   time.Time         `json:"updatedAt" bson:"updatedAt"`
    
    // 状态
    Status      WorkflowStatus    `json:"status" bson:"status"` // draft/active/archived
}
```

### 2.2 WorkflowTask（任务节点）

```go
// WorkflowTask 工作流中的任务节点
type WorkflowTask struct {
    ID          string            `json:"id" bson:"id"`
    Name        string            `json:"name" bson:"name"`
    Description string            `json:"description" bson:"description"`
    
    // 任务配置
    Image       string            `json:"image" bson:"image"`
    Command     []string          `json:"command" bson:"command"`
    Args        []string          `json:"args" bson:"args"`
    Env         map[string]string `json:"env" bson:"env"`
    
    // 资源需求
    Resources   ResourceSpec      `json:"resources" bson:"resources"`
    
    // 执行配置
    Timeout     Duration          `json:"timeout" bson:"timeout"`
    Retries     int               `json:"retries" bson:"retries"`
    RetryDelay  Duration          `json:"retryDelay" bson:"retryDelay"`
    
    // 条件执行
    Condition   string            `json:"condition" bson:"condition"`
    
    // 输入输出
    Inputs      []Input           `json:"inputs" bson:"inputs"`
    Outputs     []Output          `json:"outputs" bson:"outputs"`
}
```

### 2.3 Edge & Parameter

```go
// Edge 依赖边
type Edge struct {
    From      string `json:"from" bson:"from"`
    To        string `json:"to" bson:"to"`
    Condition string `json:"condition,omitempty" bson:"condition,omitempty"`
}

// Parameter 参数定义
type Parameter struct {
    Name        string      `json:"name" bson:"name"`
    Description string      `json:"description" bson:"description"`
    Type        string      `json:"type" bson:"type"` // string/int/float/bool
    Default     interface{} `json:"default" bson:"default"`
    Required    bool        `json:"required" bson:"required"`
}

// ResourceSpec 资源规格
type ResourceSpec struct {
    GPUCount  int    `json:"gpuCount" bson:"gpuCount"`
    GPUMemory string `json:"gpuMemory" bson:"gpuMemory"`
    GPUModel  string `json:"gpuModel" bson:"gpuModel"`
    CPU       string `json:"cpu" bson:"cpu"`
    Memory    string `json:"memory" bson:"memory"`
}
```

### 2.4 WorkflowRun（工作流执行实例）

```go
// WorkflowRun 工作流执行实例
type WorkflowRun struct {
    ID         string                  `json:"id" bson:"_id"`
    WorkflowID string                  `json:"workflowId" bson:"workflowId"`
    Name       string                  `json:"name" bson:"name"`
    
    // 参数值
    Parameters map[string]interface{}  `json:"parameters" bson:"parameters"`
    
    // 状态
    Status     WorkflowRunStatus       `json:"status" bson:"status"`
    Phase      string                  `json:"phase" bson:"phase"`
    
    // 执行信息
    StartedAt  *time.Time              `json:"startedAt" bson:"startedAt"`
    FinishedAt *time.Time              `json:"finishedAt" bson:"finishedAt"`
    Duration   time.Duration           `json:"duration" bson:"duration"`
    
    // 任务执行实例
    TaskRuns   []TaskRun               `json:"taskRuns" bson:"taskRuns"`
    
    // 触发信息
    Trigger    TriggerInfo             `json:"trigger" bson:"trigger"`
    
    // 元数据
    CreatedBy  string                  `json:"createdBy" bson:"createdBy"`
    CreatedAt  time.Time               `json:"createdAt" bson:"createdAt"`
    
    // 结果
    Outputs    map[string]interface{}  `json:"outputs" bson:"outputs"`
    Message    string                  `json:"message" bson:"message"`
}
```

### 2.5 TaskRun（任务执行实例）

```go
// TaskRun 任务执行实例
type TaskRun struct {
    ID           string                 `json:"id" bson:"id"`
    TaskID       string                 `json:"taskId" bson:"taskId"`
    TaskName     string                 `json:"taskName" bson:"taskName"`
    
    // 关联的原生任务 ID（QingTian Task）
    NativeTaskID string                 `json:"nativeTaskId" bson:"nativeTaskId"`
    
    // 状态
    Status       TaskRunStatus          `json:"status" bson:"status"`
    Phase        string                 `json:"phase" bson:"phase"`
    
    // 执行信息
    StartedAt    *time.Time             `json:"startedAt" bson:"startedAt"`
    FinishedAt   *time.Time             `json:"finishedAt" bson:"finishedAt"`
    Duration     time.Duration          `json:"duration" bson:"duration"`
    
    // 重试信息
    RetryCount   int                    `json:"retryCount" bson:"retryCount"`
    
    // 节点信息
    NodeID       string                 `json:"nodeId" bson:"nodeId"`
    
    // 输入输出
    Inputs       map[string]interface{} `json:"inputs" bson:"inputs"`
    Outputs      map[string]interface{} `json:"outputs" bson:"outputs"`
    
    // 日志和错误
    LogPath      string                 `json:"logPath" bson:"logPath"`
    Error        string                 `json:"error" bson:"error"`
}

// WorkflowRunStatus 工作流执行状态
type WorkflowRunStatus string

const (
    WorkflowRunStatusPending   WorkflowRunStatus = "Pending"
    WorkflowRunStatusRunning   WorkflowRunStatus = "Running"
    WorkflowRunStatusSucceeded WorkflowRunStatus = "Succeeded"
    WorkflowRunStatusFailed    WorkflowRunStatus = "Failed"
    WorkflowRunStatusCanceled  WorkflowRunStatus = "Canceled"
)

// TaskRunStatus 任务执行状态
type TaskRunStatus string

const (
    TaskRunStatusPending   TaskRunStatus = "Pending"
    TaskRunStatusRunning   TaskRunStatus = "Running"
    TaskRunStatusSucceeded TaskRunStatus = "Succeeded"
    TaskRunStatusFailed    TaskRunStatus = "Failed"
    TaskRunStatusSkipped   TaskRunStatus = "Skipped"
)
```

### 2.6 Schedule（定时计划）

```go
// Schedule 定时计划
type Schedule struct {
    ID          string                `json:"id" bson:"_id"`
    Name        string                `json:"name" bson:"name"`
    Description string                `json:"description" bson:"description"`
    
    // 关联的工作流
    WorkflowID  string                `json:"workflowId" bson:"workflowId"`
    
    // Cron 表达式
    CronExpr    string                `json:"cronExpr" bson:"cronExpr"`
    
    // 时区
    Timezone    string                `json:"timezone" bson:"timezone"`
    
    // 参数
    Parameters  map[string]interface{} `json:"parameters" bson:"parameters"`
    
    // 执行配置
    ConcurrencyPolicy ConcurrencyPolicy `json:"concurrencyPolicy" bson:"concurrencyPolicy"`
    SuccessfulJobsHistoryLimit int      `json:"successfulJobsHistoryLimit" bson:"successfulJobsHistoryLimit"`
    FailedJobsHistoryLimit int          `json:"failedJobsHistoryLimit" bson:"failedJobsHistoryLimit"`
    
    // 状态
    Enabled     bool                  `json:"enabled" bson:"enabled"`
    Status      ScheduleStatus        `json:"status" bson:"status"`
    
    // 上次执行
    LastRunAt   *time.Time            `json:"lastRunAt" bson:"lastRunAt"`
    LastRunID   string                `json:"lastRunId" bson:"lastRunId"`
    
    // 下次执行
    NextRunAt   *time.Time            `json:"nextRunAt" bson:"nextRunAt"`
    
    // 元数据
    CreatedBy   string                `json:"createdBy" bson:"createdBy"`
    CreatedAt   time.Time             `json:"createdAt" bson:"createdAt"`
    UpdatedAt   time.Time             `json:"updatedAt" bson:"updatedAt"`
}

// ConcurrencyPolicy 并发策略
type ConcurrencyPolicy string

const (
    ConcurrencyPolicyAllow   ConcurrencyPolicy = "Allow"   // 允许并发
    ConcurrencyPolicyForbid  ConcurrencyPolicy = "Forbid"  // 禁止并发，跳过新执行
    ConcurrencyPolicyReplace ConcurrencyPolicy = "Replace" // 替换正在执行的
)

// ScheduleStatus 定时计划状态
type ScheduleStatus string

const (
    ScheduleStatusActive   ScheduleStatus = "active"
    ScheduleStatusPaused   ScheduleStatus = "paused"
    ScheduleStatusError    ScheduleStatus = "error"
)

// TriggerInfo 触发信息
type TriggerInfo struct {
    Type        string    `json:"type" bson:"type"` // manual/schedule/webhook
    ScheduleID  string    `json:"scheduleId,omitempty" bson:"scheduleId,omitempty"`
    TriggeredBy string    `json:"triggeredBy" bson:"triggeredBy"`
    TriggeredAt time.Time `json:"triggeredAt" bson:"triggeredAt"`
}
```

---

## 3. 系统架构

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         API Server                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Workflow API │  │ Schedule API │  │ Run API      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Engine                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ DAG Executor │  │ Task Router  │  │ State Machine│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Input Resolver│ │ Output Router│  │ Retry Manager│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Schedule Service                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Cron Parser  │  │ Next Run Calc│  │ Job Trigger  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Concurrency  │  │ History Clean│  │ Timezone Supp│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Original QingTian Scheduler                     │
│            (Filter + Score + Ledger + Dispatch)              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                      NSQ / Agent
```

### 3.2 核心组件

| 组件 | 职责 |
|------|------|
| **WorkflowService** | 工作流 CRUD、模板管理 |
| **WorkflowEngine** | DAG 解析、任务调度、状态推进 |
| **ScheduleService** | 定时计划管理、Cron 解析、触发执行 |
| **WorkflowExecutor** | 工作流执行器，管理 TaskRun 生命周期 |
| **TaskRunController** | TaskRun 状态监听，驱动工作流状态推进 |

---

## 4. 核心功能设计

### 4.1 DAG 任务编排

#### 4.1.1 DAG 校验器

```go
// WorkflowValidator 工作流校验器
type WorkflowValidator struct{}

func (v *WorkflowValidator) Validate(wf *Workflow) error {
    // 1. 校验任务 ID 唯一性
    taskIDs := make(map[string]bool)
    for _, task := range wf.Tasks {
        if taskIDs[task.ID] {
            return fmt.Errorf("duplicate task ID: %s", task.ID)
        }
        taskIDs[task.ID] = true
    }
    
    // 2. 校验边引用的任务存在
    for _, edge := range wf.Edges {
        if !taskIDs[edge.From] {
            return fmt.Errorf("edge references non-existent task: %s", edge.From)
        }
        if !taskIDs[edge.To] {
            return fmt.Errorf("edge references non-existent task: %s", edge.To)
        }
    }
    
    // 3. 校验 DAG 无环
    if err := v.detectCycle(wf); err != nil {
        return err
    }
    
    return nil
}

// detectCycle 使用 DFS 检测环
func (v *WorkflowValidator) detectCycle(wf *Workflow) error {
    graph := buildGraph(wf)
    visited := make(map[string]bool)
    recStack := make(map[string]bool)
    
    for taskID := range graph {
        if !visited[taskID] {
            if v.hasCycleDFS(taskID, graph, visited, recStack) {
                return fmt.Errorf("cycle detected in workflow")
            }
        }
    }
    return nil
}

func (v *WorkflowValidator) hasCycleDFS(node string, graph map[string][]string, 
    visited, recStack map[string]bool) bool {
    visited[node] = true
    recStack[node] = true
    
    for _, neighbor := range graph[node] {
        if !visited[neighbor] {
            if v.hasCycleDFS(neighbor, graph, visited, recStack) {
                return true
            }
        } else if recStack[neighbor] {
            return true
        }
    }
    
    recStack[node] = false
    return false
}
```

#### 4.1.2 DAG 执行器

```go
// WorkflowExecutor 工作流执行器
type WorkflowExecutor struct {
    workflowRepo  repository.WorkflowRepository
    runRepo       repository.WorkflowRunRepository
    taskService   *service.TaskService
    taskRunQueue  cache.PriorityQueue
    producer      mq.Producer
}

// Execute 执行工作流
func (e *WorkflowExecutor) Execute(ctx context.Context, run *WorkflowRun) error {
    // 1. 获取工作流定义
    wf, err := e.workflowRepo.Get(ctx, run.WorkflowID)
    if err != nil {
        return err
    }
    
    // 2. 构建执行图
    execGraph := e.buildExecutionGraph(wf)
    
    // 3. 查找可执行任务（无依赖或依赖已满足）
    readyTasks := execGraph.GetReadyTasks(run)
    
    // 4. 提交就绪任务
    for _, task := range readyTasks {
        if err := e.submitTask(ctx, wf, run, task); err != nil {
            return err
        }
    }
    
    // 5. 更新状态
    run.Status = WorkflowRunStatusRunning
    return e.runRepo.Update(ctx, run)
}

// ExecutionGraph 执行图
type ExecutionGraph struct {
    tasks    map[string]*WorkflowTask
    deps     map[string][]string        // taskID -> 依赖的任务ID列表
    children map[string][]string        // taskID -> 被依赖的任务ID列表
    status   map[string]TaskRunStatus   // 任务执行状态
}

// GetReadyTasks 获取就绪任务
func (g *ExecutionGraph) GetReadyTasks(run *WorkflowRun) []*WorkflowTask {
    var ready []*WorkflowTask
    
    for taskID, task := range g.tasks {
        // 已提交或已完成的不处理
        if g.status[taskID] != "" {
            continue
        }
        
        // 检查依赖是否全部成功
        allDepsSucceeded := true
        for _, depID := range g.deps[taskID] {
            if g.status[depID] != TaskRunStatusSucceeded {
                allDepsSucceeded = false
                break
            }
        }
        
        if allDepsSucceeded {
            ready = append(ready, task)
        }
    }
    
    return ready
}

// submitTask 提交单个任务
func (e *WorkflowExecutor) submitTask(ctx context.Context, wf *Workflow, 
    run *WorkflowRun, task *WorkflowTask) error {
    // 1. 解析输入参数
    inputs, err := e.resolveInputs(wf, run, task)
    if err != nil {
        return err
    }
    
    // 2. 创建 TaskRun
    taskRun := &TaskRun{
        ID:         idgen.NewID(),
        TaskID:     task.ID,
        TaskName:   task.Name,
        Status:     TaskRunStatusPending,
        Inputs:     inputs,
        RetryCount: 0,
    }
    
    // 3. 转换为 QingTian 原生任务并提交
    nativeTask := e.convertToNativeTask(wf, run, task, taskRun)
    if err := e.taskService.Submit(ctx, nativeTask); err != nil {
        return err
    }
    
    // 4. 关联原生任务 ID
    taskRun.NativeTaskID = nativeTask.ID
    run.TaskRuns = append(run.TaskRuns, *taskRun)
    
    return nil
}
```
