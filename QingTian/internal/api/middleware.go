// Package api 实现控制面的 HTTP 接口层，包括路由、中间件与各资源的处理器。
//
// API 层负责：解析请求、鉴权授权、调用 service 层、统一响应。
package api

import (
	"context"
	"log/slog"
	"net/http"
	"strings"

	"qingtian/internal/auth"
	"qingtian/pkg/apperr"
	"qingtian/pkg/httpx"
)

// ctxKey 是 context 中存放认证信息的键类型，避免键冲突。
type ctxKey string

const claimsKey ctxKey = "claims"

// Middleware 是 HTTP 中间件类型。
type Middleware func(http.Handler) http.Handler

// Chain 按顺序将多个中间件应用到 handler 上。
//
// 参数：
//   - h:  最终处理器。
//   - ms: 中间件列表，按传入顺序由外到内包裹。
func Chain(h http.Handler, ms ...Middleware) http.Handler {
	for i := len(ms) - 1; i >= 0; i-- {
		h = ms[i](h)
	}
	return h
}

// RecoverMiddleware 捕获处理过程中的 panic，返回 500，避免进程崩溃。
func RecoverMiddleware(log *slog.Logger) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					log.Error("请求处理发生 panic", "error", rec, "path", r.URL.Path)
					httpx.WriteError(w, http.ErrAbortHandler)
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// LogMiddleware 记录每个请求的方法、路径与耗时。
func LogMiddleware(log *slog.Logger) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			log.Info("收到请求", "method", r.Method, "path", r.URL.Path)
			next.ServeHTTP(w, r)
		})
	}
}

// CORSMiddleware 处理跨域，允许前端（Vue 开发服务器）访问 API。
func CORSMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// AuthMiddleware 校验请求头中的 JWT 令牌，并将解析出的声明放入 context。
//
// 参数：
//   - tm: 令牌管理器。
func AuthMiddleware(tm *auth.TokenManager) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			header := r.Header.Get("Authorization")
			if !strings.HasPrefix(header, "Bearer ") {
				httpx.WriteError(w, apperr.Unauthorized("缺少或格式错误的认证令牌"))
				return
			}
			token := strings.TrimPrefix(header, "Bearer ")
			claims, err := tm.Parse(token)
			if err != nil {
				httpx.WriteError(w, err)
				return
			}
			ctx := context.WithValue(r.Context(), claimsKey, claims)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// claimsFromContext 从 context 中取出认证声明。
//
// 返回值：
//   - *auth.Claims: 认证声明。
//   - bool:         是否存在（未认证时为 false）。
func claimsFromContext(ctx context.Context) (*auth.Claims, bool) {
	c, ok := ctx.Value(claimsKey).(*auth.Claims)
	return c, ok
}
