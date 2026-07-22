// 本文件实现 HTTP 处理器与路由注册。
//
// Server 聚合各 service 与令牌管理器，向外暴露 REST 风格的 API。
package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"qingtian/internal/auth"
	"qingtian/internal/domain"
	"qingtian/internal/metrics"
	"qingtian/internal/service"
	"qingtian/pkg/apperr"
	"qingtian/pkg/httpx"
)

// 权限常量定义，便于统一修改
const (
	PermUserManage      auth.Permission = "user:manage"
	PermNodeView       auth.Permission = "node:view"
	PermNodeManage    auth.Permission = "node:manage"
	PermTaskView       auth.Permission = "task:view"
	PermTaskCreate    auth.Permission = "task:create"
	PermTaskCancel    auth.Permission = "task:cancel"
	PermTemplateView  auth.Permission = "template:view"
	PermTemplateCreate auth.Permission = "template:create"
	PermProjectView  auth.Permission = "project:view"
	PermProjectCreate auth.Permission = "project:create"
	PermQuotaView     auth.Permission = "quota:view"
	PermQuotaEdit     auth.Permission = "quota:edit"
	PermMeteringView  auth.Permission = "metering:view"
	PermWorkflowView  auth.Permission = "workflow:view"
	PermWorkflowCreate auth.Permission = "workflow:create"
)

// Server 是控制面 HTTP 服务，持有各业务服务依赖。
type Server struct {
	users      *service.UserService
	nodes      *service.NodeService
	tasks      *service.TaskService
	templates  *service.TaskTemplateService
	terminals  *service.TerminalSessionManager
	sso        *service.SSOService
	tokens     *auth.TokenManager
	logs       *service.LogService
	storage    *service.StorageService
	metering   *service.MeteringService
	quota      *service.QuotaService
	projects   *service.ProjectService
	exporter   *metrics.Exporter
	// 工作流编排与定时调度
	workflowService     *service.WorkflowService
	workflowEngine      *service.WorkflowEngine
	workflowRunService  *service.WorkflowRunService
	scheduleService     *service.ScheduleService
	edition             domain.Edition
}

// NewServer 创建 HTTP 服务实例（仅核心依赖）。
//
// 扩展能力（日志/存储/计量/配额/项目）通过 With* 方法可选注入，
// 未注入时对应接口返回未启用错误，保证核心闭环独立可用。
func NewServer(
	users *service.UserService,
	nodes *service.NodeService,
	tasks *service.TaskService,
	tokens *auth.TokenManager,
) *Server {
	return &Server{users: users, nodes: nodes, tasks: tasks, tokens: tokens, edition: domain.EditionCommunity}
}

// WithLogService 注入日志服务。
func (s *Server) WithLogService(l *service.LogService) *Server { s.logs = l; return s }

// WithStorageService 注入存储服务。
func (s *Server) WithStorageService(st *service.StorageService) *Server { s.storage = st; return s }

// WithMetering 注入计量服务。
func (s *Server) WithMetering(m *service.MeteringService) *Server { s.metering = m; return s }

// WithQuota 注入配额服务。
func (s *Server) WithQuota(q *service.QuotaService) *Server { s.quota = q; return s }

// WithProjects 注入项目服务。
func (s *Server) WithProjects(p *service.ProjectService) *Server { s.projects = p; return s }

// WithTemplates 注入任务模板服务。
func (s *Server) WithTemplates(t *service.TaskTemplateService) *Server { s.templates = t; return s }

// WithTerminals 注入终端会话管理器。
func (s *Server) WithTerminals(t *service.TerminalSessionManager) *Server { s.terminals = t; return s }

// WithSSO 注入 SSO 单点登录服务。
func (s *Server) WithSSO(sso *service.SSOService) *Server { s.sso = sso; return s }

// WithEdition 设置平台功能版本，用于企业版能力门控。
func (s *Server) WithEdition(e domain.Edition) *Server { s.edition = e; return s }

// WithMetricsExporter 注入 Prometheus 指标导出器。
func (s *Server) WithMetricsExporter(e *metrics.Exporter) *Server { s.exporter = e; return s }

// WithWorkflow 注入工作流服务。
func (s *Server) WithWorkflow(
	wf *service.WorkflowService,
	wfEngine *service.WorkflowEngine,
	wfRun *service.WorkflowRunService,
	sch *service.ScheduleService,
) *Server {
	s.workflowService = wf
	s.workflowEngine = wfEngine
	s.workflowRunService = wfRun
	s.scheduleService = sch
	return s
}

// Routes 构建并返回路由处理器。
//
// 参数：
//   - authMW: 认证中间件，对需要登录的接口逐个包裹。
//
// 返回：
//   - http.Handler: 完整路由 mux。
func (s *Server) Routes(authMW Middleware) http.Handler {
	mux := http.NewServeMux()

	// 健康检查（无需认证）
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	// 获取平台信息（无需认证）：返回 edition 和功能开启状态
	mux.HandleFunc("GET /api/v1/info", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"edition":         string(s.edition),
			"project_enabled": s.projects != nil,
			"metering_enabled": s.metering != nil && s.quota != nil,
		})
	})

	// Prometheus 指标抓取端点（无需认证，遵循 Prometheus 惯例）
	mux.HandleFunc("GET /metrics", s.handleMetrics)

	// 认证相关（无需登录）
	mux.HandleFunc("POST /api/v1/auth/register", s.handleRegister)
	mux.HandleFunc("POST /api/v1/auth/login", s.handleLogin)

	// 节点注册与心跳（Agent 调用，这里不强制用户 JWT，生产应改为 mTLS/Agent 令牌）
	mux.HandleFunc("POST /api/v1/nodes/register", s.handleNodeRegister)
	mux.HandleFunc("POST /api/v1/nodes/{id}/heartbeat", s.handleNodeHeartbeat)

	// 需要认证的路由
	authed := func(h http.HandlerFunc) http.Handler {
		return authMW(h)
	}

	mux.Handle("GET /api/v1/nodes", authed(s.handleListNodes))
	mux.Handle("GET /api/v1/tasks", authed(s.handleListTasks))
	mux.Handle("POST /api/v1/tasks", authed(s.handleSubmitTask))
	mux.Handle("GET /api/v1/tasks/{id}", authed(s.handleGetTask))
	mux.Handle("POST /api/v1/tasks/{id}/cancel", authed(s.handleCancelTask))
	mux.Handle("POST /api/v1/tasks/{id}/resume", authed(s.handleResumeTask))
	mux.Handle("GET /api/v1/tasks/{id}/logs", authed(s.handleTaskLogs))
	mux.Handle("GET /api/v1/tasks/{id}/logs/stream", authed(s.handleTaskLogStream))
	mux.Handle("GET /api/v1/tasks/{id}/logs/download", authed(s.handleTaskLogDownload))
	mux.Handle("GET /api/v1/tasks/{id}/result", authed(s.handleTaskResultURL))
	mux.Handle("GET /api/v1/users", authed(s.handleListUsers))

	// 数据闭环：数据集上传预签名
	mux.Handle("POST /api/v1/datasets/presign-upload", authed(s.handlePresignUpload))

	// 空闲节点回收（需节点管理权限）
	mux.Handle("GET /api/v1/nodes/idle", authed(s.handleIdleNodes))

	// 成本计量看板
	mux.Handle("GET /api/v1/metering/dashboard", authed(s.handleMeteringDashboard))

	// 计费配额
	mux.Handle("GET /api/v1/quotas/{subject}", authed(s.handleGetQuota))
	mux.Handle("PUT /api/v1/quotas/{subject}", authed(s.handleSetQuota))

	// 团队工作区
	mux.Handle("POST /api/v1/projects", authed(s.handleCreateProject))
	mux.Handle("GET /api/v1/projects", authed(s.handleListProjects))
	mux.Handle("POST /api/v1/projects/{id}/members", authed(s.handleAddProjectMember))

	// 任务模板
	mux.Handle("POST /api/v1/templates", authed(s.handleCreateTemplate))
	mux.Handle("GET /api/v1/templates", authed(s.handleListMyTemplates))
	mux.Handle("GET /api/v1/templates/{id}", authed(s.handleGetTemplate))
	mux.Handle("PUT /api/v1/templates/{id}", authed(s.handleUpdateTemplate))
	mux.Handle("DELETE /api/v1/templates/{id}", authed(s.handleDeleteTemplate))
	mux.Handle("GET /api/v1/projects/{project_id}/templates", authed(s.handleListProjectTemplates))

	// Web 终端
	mux.Handle("POST /api/v1/terminals", authed(s.handleCreateTerminal))
	mux.Handle("GET /api/v1/terminals/my", authed(s.handleListMyTerminals))
	mux.Handle("GET /api/v1/terminals/{id}", authed(s.handleGetTerminal))
	mux.Handle("DELETE /api/v1/terminals/{id}", authed(s.handleCloseTerminal))

	// SSO OIDC 端点（无需认证）
	if s.sso != nil {
		mux.HandleFunc("GET /api/v1/auth/sso/login", s.handleSSOLogin)
		mux.HandleFunc("GET /api/v1/auth/sso/callback", s.handleSSOCallback)
	}

	return mux
}

// decodeJSON 解析请求体 JSON 到目标结构，失败返回业务错误。
func decodeJSON(r *http.Request, dst interface{}) error {
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return apperr.Wrap("InvalidArgument", "请求体 JSON 解析失败", err)
	}
	return nil
}

// requirePermission 校验当前用户是否具备指定权限，不具备则写错误回 Response。
func requirePermission(r *http.Request, perm auth.Permission) error {
	claims, ok := claimsFromContext(r.Context())
	if !ok {
		return apperr.Unauthorized("未认证")
	}
	if !auth.HasPermission(claims.Role, perm) {
		return apperr.Forbidden("无权限执行该操作")
	}
	return nil
}

// handleRegister 处理用户注册。
func (s *Server) handleRegister(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Username string      `json:"username"`
		Password string      `json:"password"`
		Role     domain.Role `json:"role"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if req.Role == "" {
		req.Role = domain.RoleDeveloper // 默认注册为开发者
	}
	u, err := s.users.Register(r.Context(), req.Username, req.Password, req.Role)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, u)
}

// handleLogin 处理登录，返回 JWT 令牌与用户信息。
func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Username string `json:"username"`
		Password string `json:"password"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	token, u, err := s.users.Login(r.Context(), req.Username, req.Password)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]interface{}{"token": token, "user": u})
}

// handleListUsers 返回用户列表（需 user:manage 权限）。
func (s *Server) handleListUsers(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermUserManage); err != nil {
		httpx.WriteError(w, err)
		return
	}
	users, err := s.users.List(r.Context())
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, users)
}

// handleNodeRegister 处理节点注册（Agent 调用）。
func (s *Server) handleNodeRegister(w http.ResponseWriter, r *http.Request) {
	var req service.RegisterRequest
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	node, err := s.nodes.Register(r.Context(), req)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, node)
}

// handleNodeHeartbeat 处理节点心跳（Agent 调用）。
func (s *Server) handleNodeHeartbeat(w http.ResponseWriter, r *http.Request) {
	nodeID := r.PathValue("id")
	var req struct {
		GPUs          []domain.GPU `json:"gpus"`
		LocalDatasets []string    `json:"local_datasets,omitempty"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.nodes.Heartbeat(r.Context(), nodeID, req.GPUs); err != nil {
		httpx.WriteError(w, err)
		return
	}
	// 如果心跳携带了本地数据集列表，更新节点信息
	if len(req.LocalDatasets) > 0 {
		if node, err := s.nodes.Get(r.Context(), nodeID); err == nil {
			node.LocalDatasets = req.LocalDatasets
			_ = s.nodes.UpdateLocalDatasets(r.Context(), nodeID, req.LocalDatasets)
		}
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "ok"})
}

// handleListNodes 返回节点列表（需 node:view 权限）。
func (s *Server) handleListNodes(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermNodeView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	nodes, err := s.nodes.List(r.Context())
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, nodes)
}

// handleSubmitTask 处理任务提交（需 task:create 权限）。
func (s *Server) handleSubmitTask(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	var req service.SubmitRequest
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	task, err := s.tasks.Submit(r.Context(), claims.UserID, req)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, task)
}

// handleListTasks 返回任务列表，支持 ?status= 过滤（需 task:view 权限）。
func (s *Server) handleListTasks(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	status := domain.TaskStatus(r.URL.Query().Get("status"))
	tasks, err := s.tasks.List(r.Context(), status)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, tasks)
}

// handleGetTask 返回单个任务详情（需 task:view 权限）。
func (s *Server) handleGetTask(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	task, err := s.tasks.Get(r.Context(), id)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, task)
}

// handleCancelTask 取消任务（需 task:cancel 权限）。
func (s *Server) handleCancelTask(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCancel); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	if err := s.tasks.Cancel(r.Context(), id); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "canceled"})
}

// handleCreateTemplate 创建任务模板（需 task:create 权限）。
func (s *Server) handleCreateTemplate(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	var req service.CreateRequest
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	template, err := s.templates.Create(r.Context(), claims.UserID, req)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, template)
}

// handleListMyTemplates 列出当前用户的所有模板（需 task:view 权限）。
func (s *Server) handleListMyTemplates(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	templates, err := s.templates.ListByOwner(r.Context(), claims.UserID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, templates)
}

// handleListProjectTemplates 列出指定项目的所有模板（需 task:view 权限）。
func (s *Server) handleListProjectTemplates(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	projectID := r.PathValue("project_id")
	templates, err := s.templates.ListByProject(r.Context(), projectID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, templates)
}

// handleGetTemplate 获取单个模板详情（需 task:view 权限）。
func (s *Server) handleGetTemplate(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	template, err := s.templates.Get(r.Context(), id)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, template)
}

// handleUpdateTemplate 更新模板（需 task:create 权限且所有者）。
func (s *Server) handleUpdateTemplate(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	id := r.PathValue("id")
	var req service.UpdateRequest
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	updated, err := s.templates.Update(r.Context(), id, claims.UserID, req)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, updated)
}

// handleDeleteTemplate 删除模板（需 task:create 权限且所有者）。
func (s *Server) handleDeleteTemplate(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	id := r.PathValue("id")
	err := s.templates.Delete(r.Context(), id, claims.UserID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "deleted"})
}

// handleCreateTerminal 创建 Web Terminal 终端会话（需 task:create 权限）。
func (s *Server) handleCreateTerminal(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	var req service.CreateTerminalRequest
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	session, err := s.terminals.Create(r.Context(), claims.UserID, req)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, session)
}

// handleListMyTerminals 列出当前用户所有活跃终端会话（需 task:view 权限）。
func (s *Server) handleListMyTerminals(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	sessions, err := s.terminals.ListByOwner(r.Context(), claims.UserID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, sessions)
}

// handleGetTerminal 获取终端会话信息（需 task:view 权限且所有者）。
func (s *Server) handleGetTerminal(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	id := r.PathValue("id")
	session, err := s.terminals.Get(r.Context(), id, claims.UserID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, session)
}

// handleCloseTerminal 关闭终端会话（需 task:create 权限且所有者）。
func (s *Server) handleCloseTerminal(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	_, _ = claimsFromContext(r.Context())
	id := r.PathValue("id")
	s.terminals.Close(id)
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "closed"})
}

// handleSSOLogin 处理 SSO 登录发起，重定向到 OIDC Provider。
//
// 生成随机 state，重定向用户到 OIDC Provider 进行认证。
// 前端可以跳转到此端点开始 SSO 登录流程。
func (s *Server) handleSSOLogin(w http.ResponseWriter, r *http.Request) {
	if s.sso == nil {
		httpx.WriteError(w, apperr.NotFound("SSO not enabled"))
		return
	}
	// Generate random state for CSRF protection
	// In production, you should store state in session or cookie for validation
	state := "state" // TODO: In real implementation, generate random and store in cookie
	url := s.sso.GetAuthorizationURL(state)
	http.Redirect(w, r, url, http.StatusTemporaryRedirect)
}

// handleSSOCallback 处理 OIDC Provider 回调，完成认证并返回 JWT。
//
// 流程：
//  1. 校验 state（TODO）
//  2. 用 code 换 token，验证 ID token
//  3. 查找或创建用户，自动 provisioning
//  4. 签发本平台 JWT 令牌返回给前端
func (s *Server) handleSSOCallback(w http.ResponseWriter, r *http.Request) {
	if s.sso == nil {
		httpx.WriteError(w, apperr.NotFound("SSO not enabled"))
		return
	}

	code := r.URL.Query().Get("code")
	state := r.URL.Query().Get("state")
	_ = state // TODO: Validate state against stored value to prevent CSRF

	if code == "" {
		httpx.WriteError(w, apperr.InvalidArgument("code parameter is required"))
		return
	}

	user, tokenPair, err := s.sso.Callback(r.Context(), code)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}

	// Return the same format as normal login
	httpx.WriteJSON(w, http.StatusOK, map[string]interface{}{
		"token": tokenPair,
		"user":  user,
	})
}

// ========== Workflow API ==========

// handleCreateWorkflow 创建工作流定义
func (s *Server) handleCreateWorkflow(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	var req domain.Workflow
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	req.CreatedBy = claims.UserID
	if err := s.workflowService.Create(r.Context(), &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, req)
}

// handleListWorkflows 列出工作流定义
func (s *Server) handleListWorkflows(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	skip, _ := strconv.Atoi(r.URL.Query().Get("skip"))
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 {
		limit = 20
	}
	workflows, total, err := s.workflowService.List(r.Context(), claims.UserID, skip, limit)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]interface{}{
		"items": workflows,
		"total": total,
	})
}

// handleGetWorkflow 获取工作流详情
func (s *Server) handleGetWorkflow(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	wf, err := s.workflowService.Get(r.Context(), id)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, wf)
}

// handleUpdateWorkflow 更新工作流定义
func (s *Server) handleUpdateWorkflow(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	var wf domain.Workflow
	if err := decodeJSON(r, &wf); err != nil {
		httpx.WriteError(w, err)
		return
	}
	wf.ID = id
	if err := s.workflowService.Update(r.Context(), &wf); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, wf)
}

// handleDeleteWorkflow 删除工作流
func (s *Server) handleDeleteWorkflow(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	if err := s.workflowService.Delete(r.Context(), id); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "deleted"})
}

// handleRunWorkflow 手动触发一次工作流执行
func (s *Server) handleRunWorkflow(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	id := r.PathValue("id")
	var req struct {
		Parameters map[string]interface{} `json:"parameters"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	trigger := domain.TriggerInfo{
		Type:        "manual",
		TriggeredBy: claims.UserID,
		TriggeredAt: time.Now(),
	}
	run, err := s.workflowEngine.Run(r.Context(), id, req.Parameters, trigger)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, run)
}

// handleListWorkflowRuns 列出工作流的执行历史
func (s *Server) handleListWorkflowRuns(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	skip, _ := strconv.Atoi(r.URL.Query().Get("skip"))
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 {
		limit = 20
	}
	runs, total, err := s.workflowRunService.List(r.Context(), id, skip, limit)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]interface{}{
		"items": runs,
		"total": total,
	})
}

// handleGetWorkflowRun 获取工作流执行实例详情
func (s *Server) handleGetWorkflowRun(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	run, err := s.workflowRunService.Get(r.Context(), id)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, run)
}

// ========== Schedule API ==========

// handleCreateSchedule 创建定时计划
func (s *Server) handleCreateSchedule(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	claims, _ := claimsFromContext(r.Context())
	var req domain.Schedule
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	req.CreatedBy = claims.UserID
	if err := s.scheduleService.Create(r.Context(), &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, req)
}

// handleListSchedules 列出定时计划
func (s *Server) handleListSchedules(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	skip, _ := strconv.Atoi(r.URL.Query().Get("skip"))
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 {
		limit = 20
	}
	schedules, total, err := s.scheduleService.List(r.Context(), skip, limit)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]interface{}{
		"items": schedules,
		"total": total,
	})
}

// handleGetSchedule 获取定时计划详情
func (s *Server) handleGetSchedule(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	sch, err := s.scheduleService.GetByID(r.Context(), id)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, sch)
}

// handleUpdateSchedule 更新定时计划
func (s *Server) handleUpdateSchedule(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	var sch domain.Schedule
	if err := decodeJSON(r, &sch); err != nil {
		httpx.WriteError(w, err)
		return
	}
	sch.ID = id
	if err := s.scheduleService.Update(r.Context(), &sch); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, sch)
}

// handleDeleteSchedule 删除定时计划
func (s *Server) handleDeleteSchedule(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	if err := s.scheduleService.Delete(r.Context(), id); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "deleted"})
}

// handleTriggerSchedule 手动触发一次定时计划执行
func (s *Server) handleTriggerSchedule(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermWorkflowCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	var req struct {
		Parameters map[string]interface{} `json:"parameters"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.scheduleService.TriggerRun(r.Context(), id, req.Parameters); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "triggered"})
}
