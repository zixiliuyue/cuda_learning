// Package api 实现控制面的 HTTP 接口层，包括路由、中间件与各资源的处理器。
//
// API 层负责：解析请求、鉴权授权、调用 service 层、统一响应。
package api

import (
	"context"
	"crypto/subtle"
	"log/slog"
	"net/http"
	"strings"

	"qingtian/internal/auth"
	"qingtian/internal/tracing"
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

// TraceMiddleware 为每个请求开启一个根 Span，串联该请求下游调用的链路。
//
// 该中间件把 tracing.Span 注入请求 context，service 层通过 context 派生子 Span，
// 从而形成 "HTTP 请求 -> 提交/调度" 的完整调用链。tracer 为 nil 时直接透传。
//
// 参数：
//   - tracer: 追踪器；为 nil 时中间件不产生副作用。
func TraceMiddleware(tracer *tracing.Tracer) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if tracer == nil {
				next.ServeHTTP(w, r)
				return
			}
			ctx, span := tracer.Start(r.Context(), "http "+r.Method+" "+r.URL.Path)
			span.SetAttr("http.method", r.Method).SetAttr("http.path", r.URL.Path)
			defer span.Finish()
			next.ServeHTTP(w, r.WithContext(ctx))
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

// AgentTokenMiddleware 校验 Agent 接入令牌，用于保护节点注册/心跳等无用户 JWT 的接口。
//
// Agent 请求需携带 Authorization: Bearer <token> 或 X-Agent-Token: <token>。
// 若配置的 token 为空，则打印一次警告并放行，保持对既有部署的向后兼容。
//
// 参数：
//   - token: 期望的 Agent 令牌，取自配置 Auth.AgentToken。
//   - log:   日志器，用于在未配置令牌时输出警告。
func AgentTokenMiddleware(token string, log *slog.Logger) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// 未配置令牌：向后兼容放行，但提示存在安全风险。
			if token == "" {
				if log != nil {
					log.Warn("Agent 接口未配置接入令牌（QT_AGENT_TOKEN），当前放行，生产环境请务必配置")
				}
				next.ServeHTTP(w, r)
				return
			}
			if provided := agentTokenFromRequest(r); subtle.ConstantTimeCompare([]byte(provided), []byte(token)) == 1 {
				next.ServeHTTP(w, r)
				return
			}
			httpx.WriteError(w, apperr.Unauthorized("Agent 接入令牌无效或缺失"))
		})
	}
}

// agentTokenFromRequest 依次从 Authorization Bearer 头与 X-Agent-Token 头提取 Agent 令牌。
func agentTokenFromRequest(r *http.Request) string {
	if header := r.Header.Get("Authorization"); strings.HasPrefix(header, "Bearer ") {
		return strings.TrimPrefix(header, "Bearer ")
	}
	return r.Header.Get("X-Agent-Token")
}
