package service

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"qingtian/internal/auth"
	"qingtian/internal/cache"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/mq"
	"qingtian/internal/repository"
	"qingtian/internal/scheduler"
	"qingtian/pkg/apperr"
)

// jsonMarshal 是测试用的 JSON 序列化辅助函数。
func jsonMarshal(v any) []byte {
	b, _ := json.Marshal(v)
	return b
}

// jsonUnmarshal 是测试用的 JSON 反序列化辅助函数。
func jsonUnmarshal(b []byte, v any) error {
	return json.Unmarshal(b, v)
}

// TestUserRegisterLogin 验证注册与登录流程。
func TestUserRegisterLogin(t *testing.T) {
	repo := repository.NewMemoryUserRepo()
	tm := auth.NewTokenManager("secret", time.Hour, 7*24*time.Hour)
	svc := NewUserService(repo, tm)
	ctx := context.Background()

	u, err := svc.Register(ctx, "alice", "pass1234", domain.RoleDeveloper)
	if err != nil {
		t.Fatalf("注册应成功: %v", err)
	}
	if u.PasswordHash == "pass1234" {
		t.Error("不应存储明文密码")
	}
	// 重复用户名应冲突。
	if _, err := svc.Register(ctx, "alice", "x", domain.RoleDeveloper); err == nil {
		t.Error("重复用户名应报错")
	}
	// 登录成功。
	token, _, err := svc.Login(ctx, "alice", "pass1234")
	if err != nil || token == "" {
		t.Fatalf("登录应成功并返回令牌: %v", err)
	}
	// 错误密码。
	if _, _, err := svc.Login(ctx, "alice", "wrong"); err == nil {
		t.Error("错误密码应登录失败")
	}
}

// TestRegisterPasswordStrength 验证密码强度校验：少于 8 位或超过 72 字节应拒绝。
func TestRegisterPasswordStrength(t *testing.T) {
	repo := repository.NewMemoryUserRepo()
	tm := auth.NewTokenManager("secret", time.Hour, 7*24*time.Hour)
	svc := NewUserService(repo, tm)
	ctx := context.Background()

	cases := []struct {
		name     string
		password string
		wantErr  bool
	}{
		{"空密码", "", true},
		{"7位过短", "pass123", true},
		{"8位合规", "pass1234", false},
		{"超72字节", string(make([]byte, 73)), true},
	}
	for i, c := range cases {
		// 每个用例用不同用户名，避免用户名冲突干扰密码校验断言。
		username := "user" + string(rune('a'+i))
		_, err := svc.Register(ctx, username, c.password, domain.RoleViewer)
		if (err != nil) != c.wantErr {
			t.Errorf("%s: Register 错误状态=%v, 期望 wantErr=%v", c.name, err, c.wantErr)
		}
	}
}

// noDedupUserRepo 是一个 Create 永不去重的假仓储，用于模拟
// MongoDB 缺少 username 唯一索引的场景：Create 总是成功。
// 借此验证 UserService.Register 的查重逻辑不依赖仓储去重，
// 而是服务层自己拦住重复用户名（这正是之前重复 admin 的根因所在）。
type noDedupUserRepo struct {
	users map[string]*domain.User // key: username
}

func newNoDedupUserRepo() *noDedupUserRepo {
	return &noDedupUserRepo{users: make(map[string]*domain.User)}
}

// Create 永远成功，不做任何去重（模拟无唯一索引）。
func (r *noDedupUserRepo) Create(_ context.Context, u *domain.User) error {
	r.users[u.Username] = u
	return nil
}

func (r *noDedupUserRepo) GetByID(_ context.Context, id string) (*domain.User, error) {
	for _, u := range r.users {
		if u.ID == id {
			return u, nil
		}
	}
	return nil, apperr.NotFound("用户不存在: " + id)
}

func (r *noDedupUserRepo) GetByUsername(_ context.Context, username string) (*domain.User, error) {
	if u, ok := r.users[username]; ok {
		return u, nil
	}
	return nil, apperr.NotFound("用户不存在: " + username)
}

func (r *noDedupUserRepo) List(_ context.Context) ([]*domain.User, error) {
	out := make([]*domain.User, 0, len(r.users))
	for _, u := range r.users {
		out = append(out, u)
	}
	return out, nil
}

func (r *noDedupUserRepo) FindByExternalID(_ context.Context, externalID string) (*domain.User, error) {
	for _, u := range r.users {
		if u.ExternalID == externalID {
			return u, nil
		}
	}
	return nil, apperr.NotFound("用户不存在: external_id=" + externalID)
}

func (r *noDedupUserRepo) Update(_ context.Context, u *domain.User) error {
	r.users[u.Username] = u
	return nil
}

// TestRegisterRejectsDuplicateWithoutRepoDedup 验证：即使仓储层不做去重，
// 服务层也必须拦住重复用户名，返回冲突错误。
func TestRegisterRejectsDuplicateWithoutRepoDedup(t *testing.T) {
	repo := newNoDedupUserRepo()
	tm := auth.NewTokenManager("secret", time.Hour, 7*24*time.Hour)
	svc := NewUserService(repo, tm)
	ctx := context.Background()

	if _, err := svc.Register(ctx, "admin", "pass1234", domain.RoleAdmin); err != nil {
		t.Fatalf("首次注册应成功: %v", err)
	}
	// 第二次注册同名用户，服务层应返回冲突，而非再次成功。
	if _, err := svc.Register(ctx, "admin", "other", domain.RoleAdmin); err == nil {
		t.Fatal("重复用户名应被服务层拦截，返回冲突错误")
	}
	// 确认库中只存在一个 admin。
	users, _ := repo.List(ctx)
	if len(users) != 1 {
		t.Fatalf("应只存在 1 个用户，实际 %d 个", len(users))
	}
}

// TestClosedLoop 是核心的端到端闭环测试：
// 注册节点 -> 提交任务 -> 调度下发 -> 模拟 Agent 执行并回传 -> 任务终态。
func TestClosedLoop(t *testing.T) {
	ctx := context.Background()

	// 组装内存基础设施。
	nodeRepo := repository.NewMemoryNodeRepo()
	taskRepo := repository.NewMemoryTaskRepo()
	memCache := cache.NewMemoryCache()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	resLedger := ledger.New()
	sched := scheduler.New(resLedger, "binpack")

	nodeSvc := NewNodeService(nodeRepo, memCache, resLedger, 30*time.Second)
	taskSvc := NewTaskService(taskRepo, nodeRepo, queue, bus, sched)

	// 1. 注册一个 GPU 节点。
	node, err := nodeSvc.Register(ctx, RegisterRequest{
		Hostname: "gpu-01",
		GPUs: []domain.GPU{
			{UUID: "gpu-uuid-0", Model: "RTX 5060 Ti", MemoryTotal: 16384, Allocatable: true},
		},
	})
	if err != nil {
		t.Fatalf("节点注册应成功: %v", err)
	}

	// 2. 模拟 Agent：订阅本节点的下发主题，收到后回传 succeeded 状态。
	var executedTaskID string
	_ = bus.Subscribe(mq.TaskDispatchTopic(node.ID), "agent", func(ctx context.Context, body []byte) error {
		var msg mq.DispatchMessage
		_ = jsonUnmarshal(body, &msg)
		executedTaskID = msg.TaskID
		// 模拟真实 Agent：先回传 running，再回传 succeeded（符合状态机顺序）。
		running := mq.StatusMessage{TaskID: msg.TaskID, NodeID: msg.NodeID, Status: string(domain.TaskRunning)}
		if err := bus.Publish(ctx, mq.TopicTaskStatus, jsonMarshal(running)); err != nil {
			return err
		}
		done := mq.StatusMessage{TaskID: msg.TaskID, NodeID: msg.NodeID, Status: string(domain.TaskSucceeded), ExitCode: 0}
		return bus.Publish(ctx, mq.TopicTaskStatus, jsonMarshal(done))
	})
	// 控制面订阅状态回传主题。
	_ = bus.Subscribe(mq.TopicTaskStatus, "apiserver", taskSvc.HandleStatus)

	// 3. 提交任务。
	task, err := taskSvc.Submit(ctx, "user-1", SubmitRequest{
		Name:     "train-job",
		Image:    "cuda:13.1",
		Command:  "echo hello",
		GPUCount: 1,
		MemoryMB: 8192,
		Priority: 5,
	})
	if err != nil {
		t.Fatalf("任务提交应成功: %v", err)
	}
	if task.Status != domain.TaskPending {
		t.Errorf("新任务状态应为 pending，实际 %s", task.Status)
	}

	// 4. 触发一次调度。因内存总线是同步的，调度下发会同步触发 Agent 执行与状态回传。
	processed, err := taskSvc.ScheduleOnce(ctx)
	if err != nil {
		t.Fatalf("调度应成功: %v", err)
	}
	if !processed {
		t.Fatal("应处理了一个任务")
	}

	// 5. 校验闭环结果。
	if executedTaskID != task.ID {
		t.Errorf("Agent 应执行提交的任务，期望 %s，实际 %s", task.ID, executedTaskID)
	}
	final, _ := taskSvc.Get(ctx, task.ID)
	if final.Status != domain.TaskSucceeded {
		t.Errorf("闭环结束任务状态应为 succeeded，实际 %s", final.Status)
	}
	if final.NodeID != node.ID {
		t.Errorf("任务应被调度到 %s，实际 %s", node.ID, final.NodeID)
	}
}

// TestScheduleNoNodeRequeues 验证无可用节点时任务重新入队且保持 pending。
func TestScheduleNoNodeRequeues(t *testing.T) {
	ctx := context.Background()
	nodeRepo := repository.NewMemoryNodeRepo() // 无任何节点
	taskRepo := repository.NewMemoryTaskRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	sched := scheduler.New(ledger.New(), "binpack")
	taskSvc := NewTaskService(taskRepo, nodeRepo, queue, bus, sched)

	task, _ := taskSvc.Submit(ctx, "u1", SubmitRequest{
		Name: "job", Image: "img", GPUCount: 1, MemoryMB: 1024,
	})
	_, _ = taskSvc.ScheduleOnce(ctx)

	got, _ := taskSvc.Get(ctx, task.ID)
	if got.Status != domain.TaskPending {
		t.Errorf("无可用节点时任务应保持 pending，实际 %s", got.Status)
	}
	if n, _ := queue.Len(ctx); n != 1 {
		t.Errorf("任务应被重新入队，队列长度期望 1，实际 %d", n)
	}
}

// TestCancelTask 验证任务取消。
func TestCancelTask(t *testing.T) {
	ctx := context.Background()
	taskRepo := repository.NewMemoryTaskRepo()
	queue := cache.NewMemoryPriorityQueue()
	bus := mq.NewMemoryBus()
	sched := scheduler.New(ledger.New(), "binpack")
	taskSvc := NewTaskService(taskRepo, repository.NewMemoryNodeRepo(), queue, bus, sched)

	task, _ := taskSvc.Submit(ctx, "u1", SubmitRequest{
		Name: "job", Image: "img", GPUCount: 1, MemoryMB: 1024,
	})
	if err := taskSvc.Cancel(ctx, task.ID, "u1", false); err != nil {
		t.Fatalf("取消应成功: %v", err)
	}
	got, _ := taskSvc.Get(ctx, task.ID)
	if got.Status != domain.TaskCanceled {
		t.Errorf("任务应为 canceled，实际 %s", got.Status)
	}
}
