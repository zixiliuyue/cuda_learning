package auth

import (
	"testing"
	"time"

	"qingtian/internal/domain"
)

// TestHashAndCheckPassword 验证密码加密与校验。
func TestHashAndCheckPassword(t *testing.T) {
	hash, err := HashPassword("secret123")
	if err != nil {
		t.Fatalf("加密不应报错: %v", err)
	}
	if hash == "secret123" {
		t.Fatal("哈希不应等于明文")
	}
	if !CheckPassword(hash, "secret123") {
		t.Error("正确密码应校验通过")
	}
	if CheckPassword(hash, "wrong") {
		t.Error("错误密码应校验失败")
	}
}

// TestTokenGenerateAndParse 验证 JWT 签发与解析。
func TestTokenGenerateAndParse(t *testing.T) {
	tm := NewTokenManager("test-secret", time.Hour)
	u := &domain.User{ID: "u1", Username: "alice", Role: domain.RoleAdmin}
	token, err := tm.Generate(u)
	if err != nil {
		t.Fatalf("签发不应报错: %v", err)
	}
	claims, err := tm.Parse(token)
	if err != nil {
		t.Fatalf("解析不应报错: %v", err)
	}
	if claims.UserID != "u1" || claims.Username != "alice" || claims.Role != domain.RoleAdmin {
		t.Errorf("解析出的声明不正确: %+v", claims)
	}
}

// TestTokenExpired 验证过期令牌被拒绝。
func TestTokenExpired(t *testing.T) {
	tm := NewTokenManager("test-secret", -time.Hour) // 立即过期
	u := &domain.User{ID: "u1", Username: "bob", Role: domain.RoleViewer}
	token, _ := tm.Generate(u)
	if _, err := tm.Parse(token); err == nil {
		t.Fatal("过期令牌应解析失败")
	}
}

// TestTokenWrongSecret 验证密钥不匹配时校验失败。
func TestTokenWrongSecret(t *testing.T) {
	tm1 := NewTokenManager("secret-a", time.Hour)
	tm2 := NewTokenManager("secret-b", time.Hour)
	token, _ := tm1.Generate(&domain.User{ID: "u1", Role: domain.RoleViewer})
	if _, err := tm2.Parse(token); err == nil {
		t.Fatal("不同密钥应校验失败")
	}
}

// TestRBACPermissions 验证角色权限映射。
func TestRBACPermissions(t *testing.T) {
	cases := []struct {
		role domain.Role
		perm Permission
		want bool
	}{
		{domain.RoleAdmin, PermUserManage, true},
		{domain.RoleAdmin, PermNodeManage, true},
		{domain.RoleDeveloper, PermTaskCreate, true},
		{domain.RoleDeveloper, PermUserManage, false},
		{domain.RoleViewer, PermTaskView, true},
		{domain.RoleViewer, PermTaskCreate, false},
	}
	for _, c := range cases {
		if got := HasPermission(c.role, c.perm); got != c.want {
			t.Errorf("HasPermission(%s, %s) = %v, 期望 %v", c.role, c.perm, got, c.want)
		}
	}
}
