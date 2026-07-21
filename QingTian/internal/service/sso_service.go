// Package service 实现业务服务层。
//
// 本文件实现 SSO 单点登录服务，对接 OIDC OAuth 2.0。
package service

import (
	"context"
	"errors"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	"golang.org/x/oauth2"

	"qingtian/internal/auth"
	"qingtian/internal/domain"
	"qingtian/internal/repository"
	"qingtian/pkg/apperr"
	"qingtian/pkg/idgen"
)

// SSOService handles OIDC SSO authentication flow and user provisioning.
//
// 流程：
//  1. 用户重定向到 OIDC Provider 认证
//  2. Provider 回调带回 code，本服务换 token 并验证
//  3. 根据 OIDC sub (external ID) 查找用户：存在直接登录，不存在自动创建
//  4. 签发本平台 JWT 令牌返回给前端
type SSOService struct {
	oidc          *auth.OIDCAuthenticator
	userRepo      repository.UserRepository
	tokenManager  *auth.TokenManager
	defaultRole   domain.Role
}

// NewSSOService creates a new SSO service.
//
// If defaultRole is empty, defaults to domain.RoleDeveloper.
func NewSSOService(
	oidc *auth.OIDCAuthenticator,
	userRepo repository.UserRepository,
	tm *auth.TokenManager,
	defaultRole string,
) *SSOService {
	role := domain.RoleDeveloper
	if defaultRole != "" {
		role = domain.Role(defaultRole)
	}
	return &SSOService{
		oidc:         oidc,
		userRepo:     userRepo,
		tokenManager: tm,
		defaultRole:  role,
	}
}

// GetAuthorizationURL returns the URL to redirect the user for login.
//
// The state parameter should be a random string that will be validated in the callback.
func (s *SSOService) GetAuthorizationURL(state string) string {
	return s.oidc.GetAuthURL(state)
}

// Callback handles the OIDC callback with authorization code, completes authentication,
// and issues JWT tokens for the user (creates the user if it doesn't exist).
//
// Flow:
//  1. Exchange code for ID token
//  2. Verify token signature and claims
//  3. Extract user info from claims
//  4. Find or create user in local database
//  5. Issue JWT token pair (access + refresh)
func (s *SSOService) Callback(
	ctx context.Context,
	code string,
) (*domain.User, *auth.TokenPair, error) {
	// Exchange code for tokens and verify ID token
	idToken, token, err := s.oidc.ExchangeCode(ctx, code)
	if err != nil {
		return nil, nil, err
	}

	// Extract user info from claims
	userInfo, err := s.oidc.ExtractUserInfo(idToken)
	if err != nil {
		return nil, nil, apperr.Wrap(apperr.CodeInternal, "failed to extract user claims", err)
	}

	// Try to find user by external ID first
	user, err := s.findOrCreateUser(ctx, userInfo)
	if err != nil {
		return nil, nil, err
	}

	// Issue JWT tokens
	tokenPair, err := s.tokenManager.GeneratePair(user)
	if err != nil {
		return nil, nil, err
	}

	// Optional: we could store access/refresh token from OIDC provider
	// for API calls to the provider, but most use cases don't need it.
	_ = token

	return user, tokenPair, nil
}

// ExchangeAndVerify just exchanges code and verifies, doesn't issue local JWT.
// This can be used for custom login flows.
func (s *SSOService) ExchangeAndVerify(
	ctx context.Context,
	code string,
) (*auth.UserInfo, *oidc.IDToken, *oauth2.Token, error) {
	idToken, token, err := s.oidc.ExchangeCode(ctx, code)
	if err != nil {
		return nil, nil, nil, err
	}
	userInfo, err := s.oidc.ExtractUserInfo(idToken)
	return userInfo, idToken, token, err
}

// findOrCreateUser looks up user by external ID first, then by email/username.
// If not found, automatically creates a new user account.
func (s *SSOService) findOrCreateUser(
	ctx context.Context,
	userInfo *auth.UserInfo,
) (*domain.User, error) {
	// First try by external ID (OIDC sub) - this is the stable identifier
	if userInfo.Subject != "" {
		user, err := s.userRepo.FindByExternalID(ctx, userInfo.Subject)
		if err == nil {
			// Found existing user - update email if changed
			if user.Email != userInfo.Email {
				user.Email = userInfo.Email
				_ = s.userRepo.Update(ctx, user)
			}
			return user, nil
		}
		// Not an error if just not found - we try next
		var e *apperr.Error
		if !(errors.As(err, &e) && e.Code == apperr.CodeNotFound) {
			return nil, err
		}
	}

	// Try by username (email) - if user already exists with same email from SSO,
	// link the external ID to existing account
	if userInfo.Username != "" {
		existing, err := s.userRepo.GetByUsername(ctx, userInfo.Username)
		if err == nil {
			// User exists - update external ID and email
			existing.ExternalID = userInfo.Subject
			if existing.Email == "" {
				existing.Email = userInfo.Email
			}
			if err := s.userRepo.Update(ctx, existing); err != nil {
				return nil, err
			}
			return existing, nil
		}
		var e *apperr.Error
		if !(errors.As(err, &e) && e.Code == apperr.CodeNotFound) {
			return nil, err
		}
	}

	// User not found - create new automatically
	now := time.Now()
	var user *domain.User = &domain.User{
		ID:         idgen.New("usr"),
		Username:   userInfo.Username,
		ExternalID: userInfo.Subject,
		Email:      userInfo.Email,
		Role:       s.defaultRole,
		CreatedAt:  now,
	}
	// Password hash is empty for SSO users - they can't login with password anyway
	user.PasswordHash = ""

	if err := s.userRepo.Create(ctx, user); err != nil {
		return nil, apperr.Wrap(apperr.CodeInternal, "failed to create user from SSO", err)
	}

	return user, nil
}
