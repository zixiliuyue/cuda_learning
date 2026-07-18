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

// Claims 是 JWT 令牌中携带的自定义声明。
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

// TokenManager 负责 JWT 令牌的签发与校验。
type TokenManager struct {
	secret []byte
	ttl    time.Duration
}

// NewTokenManager 创建令牌管理器。
//
// 参数：
//   - secret: JWT 签名密钥。
//   - ttl:    令牌有效期。
func NewTokenManager(secret string, ttl time.Duration) *TokenManager {
	return &TokenManager{secret: []byte(secret), ttl: ttl}
}

// Generate 为指定用户签发 JWT 令牌。
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
			ExpiresAt: jwt.NewNumericDate(now.Add(m.ttl)),
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString(m.secret)
	if err != nil {
		return "", apperr.Wrap(apperr.CodeInternal, "令牌签发失败", err)
	}
	return signed, nil
}

// Parse 校验并解析 JWT 令牌。
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
