// Package auth 实现平台的认证（JWT）与授权（RBAC）能力。
//
// 认证：用户名+密码校验，签发 JWT 令牌。
// 授权：基于角色的访问控制，将角色映射到权限集合。
package auth

import (
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"

	"qingtian/internal/domain"
	"qingtian/pkg/apperr"
)

// HashPassword 使用 bcrypt 对明文密码加密。
//
// 参数：
//   - password: 明文密码。
//
// 返回值：
//   - string: bcrypt 哈希字符串。
//   - error:  加密失败时返回。
func HashPassword(password string) (string, error) {
	b, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return "", apperr.Wrap(apperr.CodeInternal, "密码加密失败", err)
	}
	return string(b), nil
}

// CheckPassword 校验明文密码与哈希是否匹配。
//
// 参数：
//   - hash:     存储的 bcrypt 哈希。
//   - password: 待校验明文密码。
//
// 返回值：
//   - bool: true 表示匹配。
func CheckPassword(hash, password string) bool {
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

// Claims 是 JWT 访问令牌中携带的自定义声明。
//
// 字段：
//   - UserID:   用户 ID。
//   - Username: 用户名。
//   - Role:     用户角色，用于授权判断。
type Claims struct {
	UserID   string      `json:"uid"`
	Username string      `json:"username"`
	Role     domain.Role `json:"role"`
	jwt.RegisteredClaims
}

// RefreshClaims 是 JWT 刷新令牌中携带的自定义声明。
//
// 刷新令牌有效期更长，用于获取新的访问令牌而无需重新登录。
type RefreshClaims struct {
	UserID   string      `json:"uid"`
	Username string      `json:"username"`
	jwt.RegisteredClaims
}

// TokenPair 包含一对 access token + refresh token，用于登录/刷新响应。
type TokenPair struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresAt    int64  `json:"expires_at"` // access token 过期时间戳（Unix 秒）
}

// TokenManager 负责 JWT 令牌的签发与校验。
type TokenManager struct {
	secret         []byte
	accessTokenTTL  time.Duration
	refreshTokenTTL time.Duration
}

// NewTokenManager 创建令牌管理器。
//
// 参数：
//   - secret: JWT 签名密钥。
//   - accessTokenTTL:    访问令牌有效期。
//   - refreshTokenTTL: 刷新令牌有效期。
func NewTokenManager(secret string, accessTokenTTL, refreshTokenTTL time.Duration) *TokenManager {
	return &TokenManager{
		secret:         []byte(secret),
		accessTokenTTL:  accessTokenTTL,
		refreshTokenTTL: refreshTokenTTL,
	}
}

// Generate 为指定用户签发 JWT 访问令牌。
//
// 参数：
//   - u: 用户对象。
//
// 返回值：
//   - string: 签名后的 JWT 字符串。
//   - error:  签名失败时返回。
func (m *TokenManager) Generate(u *domain.User) (string, error) {
	now := time.Now()
	claims := Claims{
		UserID:   u.ID,
		Username: u.Username,
		Role:     u.Role,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(m.accessTokenTTL)),
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString(m.secret)
	if err != nil {
		return "", apperr.Wrap(apperr.CodeInternal, "访问令牌签发失败", err)
	}
	return signed, nil
}

// GeneratePair 为指定用户签发一对 access token + refresh token。
//
// 用于登录成功后返回一对令牌：
//   - access token:  短期有效，用于 API 访问认证
//   - refresh token: 长期有效，用于获取新的 access token 不需要重新登录
//
// 参数：
//   - u: 用户对象。
//
// 返回值：
//   - *TokenPair: 包含两个令牌和过期时间戳。
//   - error:       签发失败时返回。
func (m *TokenManager) GeneratePair(u *domain.User) (*TokenPair, error) {
	now := time.Now()

	// 生成 access token
	accessClaims := Claims{
		UserID:   u.ID,
		Username: u.Username,
		Role:     u.Role,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(m.accessTokenTTL)),
		},
	}
	accessToken := jwt.NewWithClaims(jwt.SigningMethodHS256, accessClaims)
	accessSigned, err := accessToken.SignedString(m.secret)
	if err != nil {
		return nil, apperr.Wrap(apperr.CodeInternal, "访问令牌签发失败", err)
	}

	// 生成 refresh token
	refreshClaims := RefreshClaims{
		UserID:   u.ID,
		Username: u.Username,
		RegisteredClaims: jwt.RegisteredClaims{
			IssuedAt:  jwt.NewNumericDate(now),
			ExpiresAt: jwt.NewNumericDate(now.Add(m.refreshTokenTTL)),
		},
	}
	refreshToken := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
	refreshSigned, err := refreshToken.SignedString(m.secret)
	if err != nil {
		return nil, apperr.Wrap(apperr.CodeInternal, "刷新令牌签发失败", err)
	}

	return &TokenPair{
		AccessToken:  accessSigned,
		RefreshToken: refreshSigned,
		ExpiresAt:    now.Add(m.accessTokenTTL).Unix(),
	}, nil
}

// Parse 校验并解析 JWT 访问令牌。
//
// 参数：
//   - tokenString: JWT 字符串。
//
// 返回值：
//   - *Claims: 解析出的声明。
//   - error:   令牌无效或过期时返回 CodeUnauthorized 业务错误。
func (m *TokenManager) Parse(tokenString string) (*Claims, error) {
	claims := &Claims{}
	token, err := jwt.ParseWithClaims(tokenString, claims, func(t *jwt.Token) (interface{}, error) {
		// 校验签名算法，防止算法混淆攻击。
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, apperr.Unauthorized("不支持的签名算法")
		}
		return m.secret, nil
	})
	if err != nil || !token.Valid {
		return nil, apperr.Unauthorized("令牌无效或已过期")
	}
	return claims, nil
}

// ParseRefresh 校验并解析 JWT 刷新令牌。
//
// 参数：
//   - tokenString: JWT 字符串。
//
// 返回值：
//   - *RefreshClaims: 解析出的声明。
//   - error:   令牌无效或过期时返回 CodeUnauthorized 业务错误。
func (m *TokenManager) ParseRefresh(tokenString string) (*RefreshClaims, error) {
	claims := &RefreshClaims{}
	token, err := jwt.ParseWithClaims(tokenString, claims, func(t *jwt.Token) (interface{}, error) {
		// 校验签名算法，防止算法混淆攻击。
		if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, apperr.Unauthorized("不支持的签名算法")
		}
		return m.secret, nil
	})
	if err != nil || !token.Valid {
		return nil, apperr.Unauthorized("刷新令牌无效或已过期")
	}
	return claims, nil
}
