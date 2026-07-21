// 本文件实现 Web Terminal 终端会话管理功能。
//
// Web Terminal 允许用户在浏览器中直接交互式连接到正在运行的任务容器，
// 方便调试训练程序，查看实时输出，执行命令排查问题。
//
// 整体架构：
//  1. 前端 → API 服务器 → Node Agent → 容器 → get PTY
//  2. 所有流量通过 HTTP 基于 SSE 或 WebSocket 转发
//  3. API 服务器维护会话信息，做权限校验
package service

import (
	"context"
	"sync"
	"time"

	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// TerminalSessionManager 管理所有活跃的 Web Terminal 会话。
type TerminalSessionManager struct {
	mu      sync.RWMutex
	sessions map[string]*domain.TerminalSession
	taskRepo repository.TaskRepository
}

// NewTerminalSessionManager 创建终端会话管理器。
func NewTerminalSessionManager(taskRepo repository.TaskRepository) *TerminalSessionManager {
	return &TerminalSessionManager{
		sessions: make(map[string]*domain.TerminalSession),
		taskRepo: taskRepo,
	}
}

// CreateTerminalRequest 创建终端会话请求。
type CreateTerminalRequest struct {
	TaskID string `json:"task_id"`
}

// Create 创建一个新的终端会话，绑定到指定任务。
//
// 权限检查：只有任务所有者可以打开终端。
func (m *TerminalSessionManager) Create(ctx context.Context, ownerID string, req CreateTerminalRequest) (*domain.TerminalSession, error) {
	task, err := m.taskRepo.GetByID(ctx, req.TaskID)
	if err != nil {
		return nil, err
	}
	// 权限检查
	if task.Owner != ownerID {
		return nil, apperr.Forbidden("无权限访问该任务的终端")
	}
	// 任务必须正在运行才能打开终端
	if task.Status != domain.TaskRunning {
		return nil, apperr.Conflict("只有正在运行的任务才能打开终端")
	}
	if task.NodeID == "" {
		return nil, apperr.InvalidArgument("任务未调度到节点，无法打开终端")
	}

	now := time.Now()
	session := &domain.TerminalSession{
		ID:         idgen.New("term"),
		TaskID:     req.TaskID,
		NodeID:     task.NodeID,
		OwnerID:    ownerID,
		CreatedAt:   now,
		ExpiresAt:   now.Add(2 * time.Hour), // 终端会话最长 2 小时自动过期
		Active:     true,
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessions[session.ID] = session
	return session, nil
}

// Get 获取会话信息，检查权限。
func (m *TerminalSessionManager) Get(ctx context.Context, id string, ownerID string) (*domain.TerminalSession, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	session, ok := m.sessions[id]
	if !ok {
		return nil, apperr.NotFound("终端会话不存在")
	}
	if session.OwnerID != ownerID {
		return nil, apperr.Forbidden("无权限访问该终端会话")
	}
	if !session.Active || time.Now().After(session.ExpiresAt) {
		session.Active = false
		return nil, apperr.NotFound("终端会话已过期")
	}
	return session, nil
}

// Close 关闭终端会话。
func (m *TerminalSessionManager) Close(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if session, ok := m.sessions[id]; ok {
		session.Active = false
	}
	delete(m.sessions, id)
}

// ListByOwner 列出用户所有活跃会话。
func (m *TerminalSessionManager) ListByOwner(ctx context.Context, ownerID string) ([]*domain.TerminalSession, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make([]*domain.TerminalSession, 0)
	now := time.Now()
	for _, s := range m.sessions {
		if s.OwnerID == ownerID && s.Active && now.Before(s.ExpiresAt) {
			result = append(result, s)
		}
	}
	return result, nil
}

// Cleanup 清理过期会话。
func (m *TerminalSessionManager) Cleanup() {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	for id, s := range m.sessions {
		if !s.Active || now.After(s.ExpiresAt) {
			delete(m.sessions, id)
		}
	}
}
