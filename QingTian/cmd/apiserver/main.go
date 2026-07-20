// Command apiserver 是 QingTian 控制面 API 服务的入口。
//
// 它连接外部中间件（MongoDB/Redis/NSQ/MinIO），组装各业务服务，
// 并启动 HTTP 服务对外提供 REST API；同时消费任务状态回传消息。
package main

import (
	"context"
	"flag"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"qingtian/internal/api"
	"qingtian/internal/auth"
	"qingtian/internal/bootstrap"
	"qingtian/internal/config"
	"qingtian/internal/domain"
	"qingtian/internal/ledger"
	"qingtian/internal/logger"
	"qingtian/internal/metrics"
	"qingtian/internal/mq"
	"qingtian/internal/notify"
	"qingtian/internal/scheduler"
	"qingtian/internal/service"
	"qingtian/internal/tracing"
)

func main() {
	// 解析命令行参数：配置文件路径。
	configPath := flag.String("config", "configs/apiserver.yaml", "配置文件路径")
	flag.Parse()

	// 加载配置（文件不存在时使用默认配置）。
	cfg := config.Default()
	if _, err := os.Stat(*configPath); err == nil {
		loaded, lerr := config.Load(*configPath)
		if lerr != nil {
			panic("加载配置失败: " + lerr.Error())
		}
		cfg = loaded
	}

	log := logger.New(cfg.LogLevel, "apiserver")
	log.Info("正在启动 API 服务", "addr", cfg.Server.Addr)

	// 连接外部中间件。
	ctx := context.Background()
	infra, err := bootstrap.Connect(ctx, cfg)
	if err != nil {
		log.Error("连接基础设施失败", "error", err)
		os.Exit(1)
	}
	defer infra.Close(ctx)

	// 创建数据库索引（如 users.username 唯一）。
	// 失败不阻断启动：常见原因是历史重复数据导致唯一索引建立失败，
	// 仅告警提示，避免因脏数据无法启动；服务层已有查重兜底。
	if err := infra.Store.EnsureIndexes(ctx); err != nil {
		log.Error("创建数据库索引失败（可能存在重复用户名，请清理后重启）", "error", err)
	}

	// 组装依赖。
	tokens := auth.NewTokenManager(cfg.Auth.JWTSecret, cfg.Auth.TokenTTL)
	resLedger := ledger.New()
	sched := scheduler.New(resLedger, cfg.Scheduler.Strategy)

	// 扩展服务：日志、存储、计量、配额、项目、通知。
	logSvc := service.NewLogService(infra.LogStream, infra.Object)
	storageSvc := service.NewStorageService(infra.Object, int(cfg.Auth.TokenTTL.Seconds()))
	meteringSvc := service.NewMeteringService(infra.Store.Usages(), cfg.Metering.PricePerGPUHour)
	quotaSvc := service.NewQuotaService(infra.Store.Quotas())
	projectSvc := service.NewProjectService(infra.Store.Projects())
	var notifier notify.Notifier
	if cfg.Notify.WebhookURL != "" {
		notifier = notify.NewWebhookNotifier(cfg.Notify.WebhookURL, cfg.Notify.Platform)
	}

	userSvc := service.NewUserService(infra.Store.Users(), tokens)
	nodeSvc := service.NewNodeService(infra.Store.Nodes(), infra.Cache, resLedger, cfg.Scheduler.NodeTimeout).
		WithIdleConfig(cfg.Scheduler.IdleThresholdUtil, cfg.Scheduler.IdleThresholdDuration)
	tracer := tracing.NewTracer(tracing.NewSlogExporter(log))
	taskSvc := service.NewTaskService(infra.Store.Tasks(), infra.Store.Nodes(), infra.Cache, infra.Producer, sched).
		WithLogService(logSvc).WithMetering(meteringSvc).WithQuota(quotaSvc).WithTracer(tracer)
	if notifier != nil {
		taskSvc = taskSvc.WithNotifier(notifier)
	}

	// 消费任务状态回传消息，更新任务状态。
	if err := infra.Consumer.Subscribe(mq.TopicTaskStatus, "apiserver", taskSvc.HandleStatus); err != nil {
		log.Error("订阅任务状态主题失败", "error", err)
	}
	// 消费任务日志回传消息，写入日志流供前端实时订阅。
	if err := infra.Consumer.Subscribe(mq.TopicTaskLog, "apiserver", logSvc.HandleLog); err != nil {
		log.Error("订阅任务日志主题失败", "error", err)
	}

	// 构建路由。
	metricsExporter := metrics.NewExporter(infra.Store.Nodes())
	srv := api.NewServer(userSvc, nodeSvc, taskSvc, tokens).
		WithLogService(logSvc).
		WithStorageService(storageSvc).
		WithMetering(meteringSvc).
		WithQuota(quotaSvc).
		WithProjects(projectSvc).
		WithMetricsExporter(metricsExporter).
		WithEdition(domain.Edition(cfg.Edition))
	authMW := api.AuthMiddleware(tokens)
	handler := api.Chain(
		srv.Routes(authMW),
		api.RecoverMiddleware(log),
		api.LogMiddleware(log),
		api.TraceMiddleware(tracer),
		api.CORSMiddleware,
	)

	httpServer := &http.Server{Addr: cfg.Server.Addr, Handler: handler}

	// 优雅关闭。
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("HTTP 服务异常退出", "error", err)
		}
	}()
	log.Info("API 服务已启动")

	// 启动闲置节点自动回收定时扫描器
	// 扫描间隔设为 1 分钟，平衡检测及时性与性能开销
	go nodeSvc.StartIdleScanner(ctx, time.Minute)

	// 等待退出信号。
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("正在关闭 API 服务...")
	nodeSvc.StopIdleScanner()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)
}
