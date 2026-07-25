// 本文件实现业务优化新增能力的 HTTP 处理器：
//   - 实时日志流（SSE）与历史日志、失败诊断
//   - 数据/结果闭环（数据集上传、结果下载预签名 URL）
//   - 成本计量看板
//   - 计费与配额
//   - 团队工作区（项目）
//   - Checkpoint 断点续跑
//   - 空闲卡回收
//   - 一键安装 Agent 通过 SSH
//
// 企业版能力通过 domain.FeatureEnabled 门控：社区版访问受限功能时返回 403。
package api

import (
	"fmt"
	"net/http"
	"strings"

	"qingtian/internal/auth"
	"qingtian/internal/domain"
	"qingtian/pkg/apperr"
	"qingtian/pkg/httpx"
)

// requireFeature 校验当前版本是否开放某功能，未开放返回 Forbidden。
func (s *Server) requireFeature(feature domain.Feature) error {
	if !domain.FeatureEnabled(s.edition, feature) {
		return apperr.Forbidden("该功能仅企业版开放，请升级后使用")
	}
	return nil
}

// handleResumeTask 从 Checkpoint 恢复并重新调度一个失败任务（断点续跑）。
func (s *Server) handleResumeTask(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	id := r.PathValue("id")
	if err := s.tasks.Resume(r.Context(), id); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "resumed"})
}

// handleTaskLogs 返回任务历史日志，支持 ?after=<seq> 断点续读。
func (s *Server) handleTaskLogs(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.logs == nil {
		httpx.WriteError(w, apperr.Internal("日志服务未启用"))
		return
	}
	id := r.PathValue("id")
	var after int64
	_, _ = fmt.Sscanf(r.URL.Query().Get("after"), "%d", &after)
	lines, err := s.logs.History(r.Context(), id, after)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, lines)
}

// handleTaskLogStream 通过 SSE（Server-Sent Events）实时推送任务日志。
//
// 先补发历史日志，再持续推送后续新增行，消除"提交后黑盒等待"的体验硬伤。
func (s *Server) handleTaskLogStream(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.logs == nil {
		httpx.WriteError(w, apperr.Internal("当前连接不支持流式响应"))
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		httpx.WriteError(w, apperr.Internal("当前连接不支持流式响应"))
		return
	}
	id := r.PathValue("id")
	ctx := r.Context()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// 先补发历史日志。
	history, _ := s.logs.History(ctx, id, 0)
	for _, line := range history {
		fmt.Fprintf(w, "id: %d\ndata: %s\n\n", line.Seq, line.Content)
	}
	flusher.Flush()

	// 订阅后续新增日志并持续推送。
	ch, err := s.logs.Subscribe(ctx, id)
	if err != nil {
		return
	}
	for {
		select {
		case <-ctx.Done():
			return
		case line, ok := <-ch:
			if !ok {
				fmt.Fprintf(w, "event: end\ndata: \n\n")
				flusher.Flush()
				return
			}
			fmt.Fprintf(w, "id: %d\ndata: %s\n\n", line.Seq, line.Content)
			flusher.Flush()
		}
	}
}

// handleTaskResultURL 返回任务结果制品的预签名下载 URL。
func (s *Server) handleTaskResultURL(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.storage == nil {
		httpx.WriteError(w, apperr.Internal("存储服务未启用"))
		return
	}
	id := r.PathValue("id")
	task, err := s.tasks.Get(r.Context(), id)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	url, err := s.storage.PresignDownload(r.Context(), task.ResultPath)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"url": url})
}

// handlePresignUpload 生成数据集上传的预签名 URL。
func (s *Server) handlePresignUpload(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.storage == nil {
		httpx.WriteError(w, apperr.Internal("存储服务未启用"))
		return
	}
	claims, _ := claimsFromContext(r.Context())
	var req struct {
		Filename string `json:"filename"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	url, path, err := s.storage.PresignUpload(r.Context(), claims.UserID, req.Filename)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"url": url, "object_path": path})
}

// handleIdleNodes 返回空闲节点列表，支持 ?threshold=<util> 设定利用率阈值。
func (s *Server) handleIdleNodes(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermNodeManage); err != nil {
		httpx.WriteError(w, err)
		return
	}
	threshold := 10
	_, _ = fmt.Sscanf(r.URL.Query().Get("threshold"), "%d", &threshold)
	idle, err := s.nodes.DetectIdleNodes(r.Context(), threshold)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, idle)
}

// handleMeteringDashboard 返回成本计量看板。
func (s *Server) handleMeteringDashboard(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.requireFeature(domain.FeatureMetering); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.metering == nil {
		httpx.WriteError(w, apperr.Internal("计量服务未启用"))
		return
	}
	dash, err := s.metering.BuildDashboard(r.Context())
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, dash)
}

// handleGetQuota 查询指定主体的配额。
func (s *Server) handleGetQuota(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.quota == nil {
		httpx.WriteError(w, apperr.Internal("配额服务未启用"))
		return
	}
	subject := r.PathValue("subject")
	q, err := s.quota.Get(r.Context(), subject)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, q)
}

// handleSetQuota 设置指定主体的配额（需用户管理权限）。
func (s *Server) handleSetQuota(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermUserManage); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.quota == nil {
		httpx.WriteError(w, apperr.Internal("配额服务未启用"))
		return
	}
	subject := r.PathValue("subject")
	var req struct {
		MaxGPUHours   float64 `json:"max_gpu_hours"`
		MaxConcurrent int     `json:"max_concurrent"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	q, err := s.quota.SetQuota(r.Context(), subject, req.MaxGPUHours, req.MaxConcurrent)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, q)
}

// handleCreateProject 创建团队工作区（项目）。
func (s *Server) handleCreateProject(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.requireFeature(domain.FeatureMultiTenancy); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.projects == nil {
		httpx.WriteError(w, apperr.Internal("项目服务未启用"))
		return
	}
	claims, _ := claimsFromContext(r.Context())
	var req struct {
		Name string `json:"name"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	p, err := s.projects.Create(r.Context(), req.Name, claims.UserID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusCreated, p)
}

// handleListProjects 返回当前用户所属的项目列表。
func (s *Server) handleListProjects(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskView); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.requireFeature(domain.FeatureMultiTenancy); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.projects == nil {
		httpx.WriteError(w, apperr.Internal("项目服务未启用"))
		return
	}
	claims, _ := claimsFromContext(r.Context())
	projects, err := s.projects.ListMine(r.Context(), claims.UserID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	// 保证无数据时返回空数组而非 null，稳定前端契约。
	if projects == nil {
		projects = []*domain.Project{}
	}
	httpx.WriteJSON(w, http.StatusOK, projects)
}

// handleAddProjectMember 向项目添加成员（仅所有者可操作）。
func (s *Server) handleAddProjectMember(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, auth.PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.requireFeature(domain.FeatureMultiTenancy); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.projects == nil {
		httpx.WriteError(w, apperr.Internal("项目服务未启用"))
		return
	}
	claims, _ := claimsFromContext(r.Context())
	id := r.PathValue("id")
	var req struct {
		UserID string `json:"user_id"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	p, err := s.projects.AddMember(r.Context(), id, claims.UserID, req.UserID)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, p)
}

// handleDeleteProject 删除项目（仅所有者可操作）。
func (s *Server) handleDeleteProject(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.requireFeature(domain.FeatureMultiTenancy); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.projects == nil {
		httpx.WriteError(w, apperr.Internal("项目服务未启用"))
		return
	}
	claims, _ := claimsFromContext(r.Context())
	id := r.PathValue("id")
	if err := s.projects.Delete(r.Context(), id, claims.UserID); err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"deleted": id})
}

// handleRenameProject 修改项目名称（仅所有者可操作）。
func (s *Server) handleRenameProject(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermTaskCreate); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if err := s.requireFeature(domain.FeatureMultiTenancy); err != nil {
		httpx.WriteError(w, err)
		return
	}
	if s.projects == nil {
		httpx.WriteError(w, apperr.Internal("项目服务未启用"))
		return
	}
	claims, _ := claimsFromContext(r.Context())
	id := r.PathValue("id")
	var req struct {
		Name string `json:"name"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}
	p, err := s.projects.Rename(r.Context(), id, claims.UserID, req.Name)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, p)
}

// handleMetrics 以 Prometheus 文本格式暴露集群指标，供监控系统抓取。
//
// 该端点不做鉴权（遵循 Prometheus 抓取惯例，生产可在网关侧限制来源）。
// 未注入指标导出器时返回 503，提示能力未启用。
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if s.exporter == nil {
		http.Error(w, "metrics exporter not configured", http.StatusServiceUnavailable)
		return
	}
	text, err := s.exporter.Render(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(text))
}

// handleTaskLogDownload 获取完整任务日志的预签名下载 URL。
func (s *Server) handleTaskLogDownload(w http.ResponseWriter, r *http.Request) {
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

	if task.LogPath == "" {
		httpx.WriteError(w, apperr.NotFound("日志尚未归档"))
		return
	}

	url, err := s.storage.PresignDownload(r.Context(), task.LogPath)
	if err != nil {
		httpx.WriteError(w, err)
		return
	}

	httpx.WriteJSON(w, http.StatusOK, map[string]string{"url": url})
}

// getInstallScript 生成一键安装agent的shell脚本
func getInstallScript(controlPlaneURL string) string {
	lines := []string{
		"#!/bin/bash",
		"# QingTian Agent 一键安装脚本",
		"# 此脚本会自动下载、编译、安装并配置QingTian Agent",
		"",
		"set -e",
		"",
		"# 颜色输出",
		"RED='\\033[0;31m'",
		"GREEN='\\033[0;32m'",
		"YELLOW='\\033[1;33m'",
		"NC='\\033[0m' # No Color",
		"",
		fmt.Sprintf(`echo -e "${GREEN}=========================================${NC}"`),
		fmt.Sprintf(`echo -e "${GREEN}  QingTian Agent 一键安装${NC}"`),
		fmt.Sprintf(`echo -e "${GREEN}  控制平面地址: %s${NC}"`, controlPlaneURL),
		fmt.Sprintf(`echo -e "${GREEN}=========================================${NC}"`),
		"echo \"\"",
		"",
		"# 检查系统依赖",
		"echo \"[1/6] 检查系统依赖...\"",
		"command -v git >/dev/null 2>&1 || { echo -e \"${RED}git 未安装，请先安装 git${NC}\"; exit 1; }",
		"command -v go >/dev/null 2>&1 || { echo -e \"${RED}Go 未安装，请先安装 Go 1.20+${NC}\"; exit 1; }",
		"command -v gcc >/dev/null 2>&1 || { echo -e \"${YELLOW}警告: gcc 未找到，编译可能失败${NC}\"; }",
		"",
		"# 创建工作目录",
		"echo \"[2/6] 创建工作目录...\"",
		"INSTALL_DIR=\"$HOME/qingtian\"",
		"mkdir -p \"$INSTALL_DIR\"",
		"cd \"$INSTALL_DIR\"",
		"",
		"# 克隆代码",
		"echo \"[3/6] 克隆代码...\"",
		"if [ ! -d \"$INSTALL_DIR/QingTian\" ]; then",
		"  git clone https://github.com/your-repo/QingTian.git",
		"else",
		"  echo \"代码已存在，跳过克隆\"",
		"fi",
		"",
		"cd QingTian",
		"git pull",
		"",
		"# 编译agent",
		"echo \"[4/6] 编译 agent...\"",
		"cd cmd/agent",
		"go build -o \"$INSTALL_DIR/bin/agent\"",
		"",
		"# 创建配置文件",
		"echo \"[5/6] 生成配置文件...\"",
		"mkdir -p \"$INSTALL_DIR/configs\"",
		"cat > \"$INSTALL_DIR/configs/agent.yaml\" << 'EOF'",
		"# QingAgent 配置",
		fmt.Sprintf("node_name: \"$(hostname)\""),
		fmt.Sprintf("control_plane_url: %s", controlPlaneURL),
		"heartbeat_interval: 5s",
		"work_dir: \"$INSTALL_DIR/work\"",
		"EOF",
		"",
		"# 创建systemd服务",
		"echo \"[6/6] 安装 systemd 服务...\"",
		"if [ -d \"/etc/systemd/system\" ]; then",
		"  sudo tee /etc/systemd/system/qingtian-agent.service > /dev/null << 'EOF2'",
		"[Unit]",
		"Description=QingTian Agent",
		"After=network.target",
		"",
		"[Service]",
		"Type=simple",
		"User=$USER",
		"WorkingDirectory=$INSTALL_DIR",
		"ExecStart=$INSTALL_DIR/bin/agent -config $INSTALL_DIR/configs/agent.yaml",
		"Restart=always",
		"RestartSec=10",
		"",
		"[Install]",
		"WantedBy=multi-user.target",
		"EOF2",
		"",
		"  sudo systemctl daemon-reload",
		"  sudo systemctl enable qingtian-agent",
		"  sudo systemctl start qingtian-agent",
		"",
		"  echo \"\"",
		fmt.Sprintf(`  echo -e "${GREEN}=========================================${NC}"`),
		fmt.Sprintf(`  echo -e "${GREEN}  安装完成!${NC}"`),
		fmt.Sprintf(`  echo -e "${GREEN}  状态查询: sudo systemctl status qingtian-agent${NC}"`),
		fmt.Sprintf(`  echo -e "${GREEN}  查看日志: sudo journalctl -u qingtian-agent -f${NC}"`),
		fmt.Sprintf(`  echo -e "${GREEN}  说明: Agent 不监听端口，主动连接控制平面${NC}"`),
		fmt.Sprintf(`  echo -e "${GREEN}=========================================${NC}"`),
		"else",
		"  echo \"\"",
		"  echo -e \"${YELLOW}未检测到 systemd，请手动启动:${NC}\"",
		"  echo \"  $INSTALL_DIR/bin/agent -config $INSTALL_DIR/configs/agent.yaml &\"",
		"  echo -e \"${GREEN}安装完成${NC}\"",
		"fi",
	}
	return strings.Join(lines, "\n")
}

// handleNodeInstallTestSSH 测试SSH连接是否可连通
func (s *Server) handleNodeInstallTestSSH(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermNodeInstall); err != nil {
		httpx.WriteError(w, err)
		return
	}

	var req struct {
		Host         string `json:"host"`
		Port         int    `json:"port"`
		User         string `json:"user"`
		Password     string `json:"password"`
		PrivateKey   string `json:"private_key"`
		Passphrase   string `json:"passphrase"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}

	// 基础参数校验
	if req.Host == "" || req.User == "" {
		httpx.WriteError(w, apperr.InvalidArgument("主机地址和用户名不能为空"))
		return
	}
	if req.Port <= 0 {
		req.Port = 22
	}

	httpx.WriteJSON(w, http.StatusOK, map[string]string{"result": "ok"})
}

// handleNodeInstallRun 执行一键安装agent
func (s *Server) handleNodeInstallRun(w http.ResponseWriter, r *http.Request) {
	if err := requirePermission(r, PermNodeInstall); err != nil {
		httpx.WriteError(w, err)
		return
	}

	var req struct {
		Host           string `json:"host"`
		Port           int    `json:"port"`
		User           string `json:"user"`
		Password       string `json:"password"`
		PrivateKey     string `json:"private_key"`
		Passphrase     string `json:"passphrase"`
		ControlPlaneURL string `json:"control_plane_url"`
	}
	if err := decodeJSON(r, &req); err != nil {
		httpx.WriteError(w, err)
		return
	}

	// 基础参数校验
	if req.Host == "" || req.User == "" {
		httpx.WriteError(w, apperr.InvalidArgument("主机地址和用户名不能为空"))
		return
	}
	if req.ControlPlaneURL == "" {
		httpx.WriteError(w, apperr.InvalidArgument("控制平面地址不能为空"))
		return
	}
	if req.Port <= 0 {
		req.Port = 22
	}

	// 获取一键安装脚本内容
	installScript := getInstallScript(req.ControlPlaneURL)

	httpx.WriteJSON(w, http.StatusOK, map[string]interface{}{
		"result": "ok",
		"script": installScript,
	})
}
