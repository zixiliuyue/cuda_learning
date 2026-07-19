package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"qingtian/internal/auth"
	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/mq"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
	"qingtian/internal/service"
)

// setupTestServer 组装一个使用内存实现的测试 HTTP 服务。
func setupTestServer() (http.Handler, *auth.TokenManager) {
	tm := auth.NewTokenManager("test-secret", time.Hour)
	resLedger := ledger.New()
	sched := scheduler.New(resLedger, "binpack")
	memCache := cache.NewMemoryCache()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()

	userSvc := service.NewUserService(repository.NewMemoryUserRepo(), tm)
	nodeRepo := repository.NewMemoryNodeRepo()
	nodeSvc := service.NewNodeService(nodeRepo, memCache, resLedger, 30*time.Second)
	taskSvc := service.NewTaskService(repository.NewMemoryTaskRepo(), nodeRepo, queue, bus, sched)

	srv := NewServer(userSvc, nodeSvc, taskSvc, tm)
	return srv.Routes(AuthMiddleware(tm)), tm
}

// doJSON 发送 JSON 请求并返回响应记录器。
func doJSON(h http.Handler, method, path, token string, body interface{}) *httptest.ResponseRecorder {
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest(method, path, &buf)
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// TestHealthz 验证健康检查接口。
func TestHealthz(t *testing.T) {
	h, _ := setupTestServer()
	rec := doJSON(h, http.MethodGet, "/healthz", "", nil)
	if rec.Code != http.StatusOK {
		t.Errorf("健康检查应返回 200，实际 %d", rec.Code)
	}
}

// TestAuthRequired 验证未携带令牌访问受保护接口返回 401。
func TestAuthRequired(t *testing.T) {
	h, _ := setupTestServer()
	rec := doJSON(h, http.MethodGet, "/api/v1/tasks", "", nil)
	if rec.Code != http.StatusUnauthorized {
		t.Errorf("未认证应返回 401，实际 %d", rec.Code)
	}
}

// TestRegisterLoginAndSubmitTask 验证注册->登录->提交任务的 HTTP 全流程。
func TestRegisterLoginAndSubmitTask(t *testing.T) {
	h, _ := setupTestServer()

	// 注册。
	rec := doJSON(h, http.MethodPost, "/api/v1/auth/register", "", map[string]interface{}{
		"username": "dev", "password": "pass123", "role": domain.RoleDeveloper,
	})
	if rec.Code != http.StatusCreated {
		t.Fatalf("注册应返回 201，实际 %d，body=%s", rec.Code, rec.Body.String())
	}

	// 登录，取令牌。
	rec = doJSON(h, http.MethodPost, "/api/v1/auth/login", "", map[string]string{
		"username": "dev", "password": "pass123",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("登录应返回 200，实际 %d", rec.Code)
	}
	var loginResp struct {
		Data struct {
			Token string `json:"token"`
		} `json:"data"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &loginResp)
	token := loginResp.Data.Token
	if token == "" {
		t.Fatal("登录应返回令牌")
	}

	// 提交任务。
	rec = doJSON(h, http.MethodPost, "/api/v1/tasks", token, map[string]interface{}{
		"name":  "job1",
		"image": "cuda:13.1",
		"resource": map[string]interface{}{
			"gpu_count": 1, "memory_mb": 4096,
		},
	})
	if rec.Code != http.StatusCreated {
		t.Fatalf("提交任务应返回 201，实际 %d，body=%s", rec.Code, rec.Body.String())
	}

	// 列表查询任务。
	rec = doJSON(h, http.MethodGet, "/api/v1/tasks", token, nil)
	if rec.Code != http.StatusOK {
		t.Errorf("查询任务应返回 200，实际 %d", rec.Code)
	}
}

// TestViewerForbiddenToCreateTask 验证 viewer 角色无权创建任务。
func TestViewerForbiddenToCreateTask(t *testing.T) {
	h, tm := setupTestServer()
	// 直接签发一个 viewer 令牌。
	token, _ := tm.Generate(&domain.User{ID: "v1", Username: "viewer", Role: domain.RoleViewer})
	rec := doJSON(h, http.MethodPost, "/api/v1/tasks", token, map[string]interface{}{
		"name": "job", "image": "img",
		"resource": map[string]interface{}{"gpu_count": 1, "memory_mb": 1024},
	})
	if rec.Code != http.StatusForbidden {
		t.Errorf("viewer 创建任务应返回 403，实际 %d", rec.Code)
	}
}
