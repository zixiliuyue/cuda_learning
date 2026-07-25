package api

import (
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"qingtian/internal/auth"
	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/logstream"
	"qingtian/internal/mq"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
	"qingtian/internal/service"
	"qingtian/internal/storage"
)

// setupFullServer 组装注入了全部扩展能力（企业版）的测试服务。
func setupFullServer(edition domain.Edition) (http.Handler, *auth.TokenManager) {
	tm := auth.NewTokenManager("test-secret", time.Hour, 24*time.Hour)
	resLedger := ledger.New()
	sched := scheduler.New(resLedger, "binpack")
	memCache := cache.NewMemoryCache()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	store := storage.NewMemoryStore()

	userSvc := service.NewUserService(repository.NewMemoryUserRepo(), tm)
	nodeRepo := repository.NewMemoryNodeRepo()
	nodeSvc := service.NewNodeService(nodeRepo, memCache, resLedger, 30*time.Second)
	taskSvc := service.NewTaskService(repository.NewMemoryTaskRepo(), nodeRepo, queue, bus, sched)

	logSvc := service.NewLogService(logstream.NewMemoryStreamer(), store)
	storageSvc := service.NewStorageService(store, 600)
	meteringSvc := service.NewMeteringService(repository.NewMemoryUsageRepo(), 2)
	quotaSvc := service.NewQuotaService(repository.NewMemoryQuotaRepo())
	projectSvc := service.NewProjectService(repository.NewMemoryProjectRepo())

	srv := NewServer(userSvc, nodeSvc, taskSvc, tm).
		WithLogService(logSvc).
		WithStorageService(storageSvc).
		WithMetering(meteringSvc).
		WithQuota(quotaSvc).
		WithProjects(projectSvc).
		WithEdition(edition)
	return srv.Routes(AuthMiddleware(tm)), tm
}

// devToken 签发一个 developer 令牌。
func devToken(tm *auth.TokenManager) string {
	token, _ := tm.Generate(&domain.User{ID: "dev1", Username: "dev", Role: domain.RoleDeveloper})
	return token
}

// adminToken 签发一个 admin 令牌。
func adminToken(tm *auth.TokenManager) string {
	token, _ := tm.Generate(&domain.User{ID: "admin1", Username: "admin", Role: domain.RoleAdmin})
	return token
}

// TestPresignUpload 验证数据集上传预签名接口。
func TestPresignUpload(t *testing.T) {
	h, tm := setupFullServer(domain.EditionEnterprise)
	rec := doJSON(h, http.MethodPost, "/api/v1/datasets/presign-upload", devToken(tm),
		map[string]string{"filename": "train.zip"})
	if rec.Code != http.StatusOK {
		t.Fatalf("预签名上传应返回 200，实际 %d，body=%s", rec.Code, rec.Body.String())
	}
	var resp struct {
		Data struct {
			URL        string `json:"url"`
			ObjectPath string `json:"object_path"`
		} `json:"data"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)
	if resp.Data.URL == "" || resp.Data.ObjectPath == "" {
		t.Errorf("应返回上传 URL 与对象路径，实际 %+v", resp.Data)
	}
}

// TestMeteringDashboardEnterprise 验证企业版可访问计量看板。
func TestMeteringDashboardEnterprise(t *testing.T) {
	h, tm := setupFullServer(domain.EditionEnterprise)
	rec := doJSON(h, http.MethodGet, "/api/v1/metering/dashboard", devToken(tm), nil)
	if rec.Code != http.StatusOK {
		t.Errorf("企业版计量看板应返回 200，实际 %d", rec.Code)
	}
}

// TestMultiTenancyGatedInCommunity 验证社区版访问团队工作区被门控为 403。
func TestMultiTenancyGatedInCommunity(t *testing.T) {
	h, tm := setupFullServer(domain.EditionCommunity)
	rec := doJSON(h, http.MethodPost, "/api/v1/projects", devToken(tm),
		map[string]string{"name": "team-a"})
	if rec.Code != http.StatusForbidden {
		t.Errorf("社区版创建项目应返回 403，实际 %d", rec.Code)
	}
}

// TestProjectLifecycleEnterprise 验证企业版可创建项目并列出。
func TestProjectLifecycleEnterprise(t *testing.T) {
	h, tm := setupFullServer(domain.EditionEnterprise)
	token := devToken(tm)
	rec := doJSON(h, http.MethodPost, "/api/v1/projects", token, map[string]string{"name": "team-a"})
	if rec.Code != http.StatusCreated {
		t.Fatalf("企业版创建项目应返回 201，实际 %d，body=%s", rec.Code, rec.Body.String())
	}
	rec = doJSON(h, http.MethodGet, "/api/v1/projects", token, nil)
	if rec.Code != http.StatusOK {
		t.Errorf("列出项目应返回 200，实际 %d", rec.Code)
	}
}

// TestSetQuotaRequiresAdmin 验证设置配额需要管理员权限。
func TestSetQuotaRequiresAdmin(t *testing.T) {
	h, tm := setupFullServer(domain.EditionEnterprise)
	// developer 无 user:manage 权限。
	rec := doJSON(h, http.MethodPut, "/api/v1/quotas/dev1", devToken(tm),
		map[string]interface{}{"max_gpu_hours": 100})
	if rec.Code != http.StatusForbidden {
		t.Errorf("非管理员设置配额应返回 403，实际 %d", rec.Code)
	}
	// admin 可设置。
	rec = doJSON(h, http.MethodPut, "/api/v1/quotas/dev1", adminToken(tm),
		map[string]interface{}{"max_gpu_hours": 100, "max_concurrent": 2})
	if rec.Code != http.StatusOK {
		t.Errorf("管理员设置配额应返回 200，实际 %d，body=%s", rec.Code, rec.Body.String())
	}
}

// TestIdleNodesRequiresNodeManage 验证空闲节点查询需要节点管理权限。
func TestIdleNodesRequiresNodeManage(t *testing.T) {
	h, tm := setupFullServer(domain.EditionEnterprise)
	// developer 无 node:manage 权限。
	rec := doJSON(h, http.MethodGet, "/api/v1/nodes/idle", devToken(tm), nil)
	if rec.Code != http.StatusForbidden {
		t.Errorf("非节点管理员查询空闲节点应返回 403，实际 %d", rec.Code)
	}
	rec = doJSON(h, http.MethodGet, "/api/v1/nodes/idle", adminToken(tm), nil)
	if rec.Code != http.StatusOK {
		t.Errorf("管理员查询空闲节点应返回 200，实际 %d", rec.Code)
	}
}

// TestTaskLogsHistory 验证任务历史日志查询。
func TestTaskLogsHistory(t *testing.T) {
	h, tm := setupFullServer(domain.EditionEnterprise)
	token := devToken(tm)
	// 先提交一个归属当前用户的任务，日志查询需通过归属校验（IDOR 防护）。
	rec := doJSON(h, http.MethodPost, "/api/v1/tasks", token, map[string]interface{}{
		"name": "logjob", "image": "img", "gpu_count": 1, "memory_mb": 1024,
	})
	if rec.Code != http.StatusCreated {
		t.Fatalf("提交任务应返回 201，实际 %d，body=%s", rec.Code, rec.Body.String())
	}
	var submitResp struct {
		Data struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &submitResp)
	// 无日志时也应正常返回 200（空数组）。
	rec = doJSON(h, http.MethodGet, "/api/v1/tasks/"+submitResp.Data.ID+"/logs", token, nil)
	if rec.Code != http.StatusOK {
		t.Errorf("查询任务日志应返回 200，实际 %d", rec.Code)
	}
}
