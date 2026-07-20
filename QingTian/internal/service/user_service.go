// Package service 实现平台的核心业务逻辑，编排领域对象、仓储、账本与消息队列。
//
// 服务层是控制面的业务中枢，被 API 层（HTTP handler）与调度器共同调用，
// 保证业务规则集中、可复用、可测试。
package service

import (
	"context"
	"time"

	"qingtian/internal/auth"
	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// UserService 提供用户注册、登录、查询等能力。
type UserService struct {
	repo   repository.UserRepository
	tokens *auth.TokenManager
	now    func() time.Time
}

// NewUserService 创建用户服务。
//
// 参数：
//   - repo:   用户仓储。
//   - tokens: JWT 令牌管理器。
func NewUserService(repo repository.UserRepository, tokens *auth.TokenManager) *UserService {
	return &UserService{repo: repo, tokens: tokens, now: time.Now}
}

// Register 注册新用户。
//
// 参数：
//   - ctx:      上下文。
//   - username: 用户名。
//   - password: 明文密码（内部会做 bcrypt 加密）。
//   - role:     角色。
//
// 返回值：
//   - *domain.User: 创建成功的用户。
//   - error:        参数非法或用户名冲突时返回。
func (s *UserService) Register(ctx context.Context, username, password string, role domain.Role) (*domain.User, error) {
	if username == "" || password == "" {
		return nil, apperr.InvalidArgument("用户名和密码不能为空")
	}
	// 先查重：用户名已存在则直接返回冲突。
	// 这是确定性业务规则，不依赖数据库唯一索引，避免因缺索引导致重复账号。
	// 仓储层的唯一索引作为并发场景下的最后防线（见 mongo.go EnsureIndexes）。
	if _, err := s.repo.GetByUsername(ctx, username); err == nil {
		return nil, apperr.Conflict("用户名已存在: " + username)
	}
	hash, err := auth.HashPassword(password)
	if err != nil {
		return nil, err
	}
	u := &domain.User{
		ID:           idgen.New("user"),
		Username:     username,
		PasswordHash: hash,
		Role:         role,
		CreatedAt:    s.now(),
	}
	if err := s.repo.Create(ctx, u); err != nil {
		return nil, err
	}
	return u, nil
}

// Login 校验用户名密码并签发令牌。
//
// 参数：
//   - ctx:      上下文。
//   - username: 用户名。
//   - password: 明文密码。
//
// 返回值：
//   - string:       JWT 令牌。
//   - *domain.User: 登录用户。
//   - error:        凭证错误时返回 CodeUnauthorized。
func (s *UserService) Login(ctx context.Context, username, password string) (string, *domain.User, error) {
	u, err := s.repo.GetByUsername(ctx, username)
	if err != nil {
		// 统一返回未授权，避免暴露用户名是否存在。
		return "", nil, apperr.Unauthorized("用户名或密码错误")
	}
	if !auth.CheckPassword(u.PasswordHash, password) {
		return "", nil, apperr.Unauthorized("用户名或密码错误")
	}
	token, err := s.tokens.Generate(u)
	if err != nil {
		return "", nil, err
	}
	return token, u, nil
}

// List 返回全部用户。
func (s *UserService) List(ctx context.Context) ([]*domain.User, error) {
	return s.repo.List(ctx)
}
