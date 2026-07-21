// 本文件实现 HTTP 处理器与路由注册。
//
// Server 聚合各 service 与令牌管理器，向外暴露 REST 风格的 API。
package api

import (
	"encoding/json"
	"net/http"

	"qingtian/internal/auth"
	"qingtian/internal/domain"
	"qingtian/internal/metrics"
	"qingtian/internal/service"
	"qingtian/pkg/apperr"
	"qingtian/pkg/httpx"
)

// Server 是控制面 HTTP 服务，持有各业务服务依赖。
type Server struct {
	users      *service.UserService
	nodes      *service.NodeService
	tasks      *service.TaskService
	templates  *service.TaskTemplateService
	terminals  *service.TerminalSessionManager
	tokens     *auth.TokenManager
	logs       *service.LogService
	storage    *service.StorageService
	metering   *service.MeteringService
	quota      *service.QuotaService
	projects   *service.ProjectService
	exporter   *metrics.Exporter
	edition    domain.Edition
}

// NewServer 创建 HTTP 服务实例（仅核心依赖）。
//
// 扩展能力（日志/存储/计量/配额/项目/版本）通过 With* 方法可选注入，
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

// WithEdition 设置平台功能版本，用于企业版能力门控。
func (s *Server) WithEdition(e domain.Edition) *Server { s.edition = e; return s }

// WithMetricsExporter 注入 Prometheus 指标导出器。
func (s *Server) WithMetricsExporter(e *metrics.Exporter) *Server { s.exporter = e; return s }

// Routes 构建并返回路由处理器。
//
// 参数：
//   - authMW: 认证中间件，对需要登录的接口逐个包裹。
//
// 返回值：
//   - http.Handler: 完整路由。
//
// 说明：
//   - 采用 Go 1.22+ ServeMux 的"方法+路径"模式；对需要登录的接口用 authMW 单独包裹，
//     避免嵌套 mux 带来的路由冲突。
func (s *Server) Routes(authMW Middleware) http.Handler {
	mux := http.NewServeMux()

	// 健康检查（无需认证）。
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	// Prometheus 指标抓取端点（无需认证，遵循监控采集惯例）。
	mux.HandleFunc("GET /metrics", s.handleMetrics)

	// 认证相关（无需登录）。
	mux.HandleFunc("POST /api/v1/auth/register", s.handleRegister)
	mux.HandleFunc("POST /api/v1/auth/login", s.handleLogin)

	// 节点注册与心跳（供 Agent 调用，此处不强制用户 JWT，生产应改为 mTLS/Agent 令牌）。
	mux.HandleFunc("POST /api/v1/nodes/register", s.handleNodeRegister)
	mux.HandleFunc("POST /api/v1/nodes/{id}/heartbeat", s.handleNodeHeartbeat)

	// authed 将普通处理函数包裹认证中间件后再注册。
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
	mux.Handle("GET /api/v1/tasks/{id}/result", authed(s.handleTaskResultURL))
	mux.Handle("GET /api/v1/users", authed(s.handleListUsers))

	// 数据闭环：数据集上传预签名。
	mux.Handle("POST /api/v1/datasets/presign-upload", authed(s.handlePresignUpload))

	// 空闲卡回收（需节点管理权限）。
	mux.Handle("GET /api/v1/nodes/idle", authed(s.handleIdleNodes))

	// 成本计量看板。
	mux.Handle("GET /api/v1/metering/dashboard", authed(s.handleMeteringDashboard))

	// 计费与配额。
	mux.Handle("GET /api/v1/quotas/{subject}", authed(s.handleGetQuota))
	mux.Handle("PUT /api/v1/quotas/{subject}", authed(s.handleSetQuota))

	// 团队工作区（项目）。
	mux.Handle("POST /api/v1/projects", authed(s.handleCreateProject))
	mux.Handle("GET /api/v1/projects", authed(s.handleListProjects))
	mux.Handle("POST /api/v1/projects/{id}/members", authed(s.handleAddProjectMember))

	// 任务模板。
	mux.Handle("POST /api/v1/templates", authed(s.handleCreateTemplate))
	mux.Handle("GET /api/v1/templates", authed(s.handleListMyTemplates))
	mux.Handle("GET /api/v1/templates/{id}", authed(s.handleGetTemplate))
	mux.Handle("PUT /api/v1/templates/{id}", authed(s.handleUpdateTemplate))
	mux.Handle("DELETE /api/v1/templates/{id}", authed(s.handleDeleteTemplate))
	mux.Handle("GET /api/v1/projects/{project_id}/templates", authed(s.handleListProjectTemplates))

	return mux
}

// decodeJSON 解析请求体 JSON 到目标结构，失败返回业务错误。
func decodeJSON(r *http.Request, dst interface{}) error {
	if err := json.NewDecoder(r.Body).Decode(dst); err != nil {
		return apperr.InvalidArgument("请求体 JSON 解析失败")
	}
	return nil
}

// requirePermission 校验当前用户是否具备指定权限。
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
	if err := requirePermission(r, auth.PermUserManage); err != nil {
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
		GPUs []domain.GPU `json:"gpus"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.nodes.Heartbeat(r.Context(), nodeID, req.GPUs); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "ok"})
}

// handleListNodes 返回节点列表（需 node:view 权限）。
func (s *Server) handleListNodes(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermNodeView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
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
	if err := requirePermission(r, auth.PermTaskView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskCancel); err != nil {
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
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
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
	if err := requirePermission(r, auth.PermTaskView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
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
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
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
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
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
	if err := requirePermission(r, auth.PermTaskView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskView); err != nil {
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
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	_, _ = claimsFromContext(r.Context())
	id := r.PathValue("id")
	s.terminals.Close(id)
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "closed"})
}
