// Command apiserver is the entry point for the QingTian control-plane API server.
//
// It connects to external infrastructure (MongoDB/Redis/NSQ/MinIO), assembles the business services,
// and starts the HTTP service to provide REST API to the outside world; it also consumes task status update messages to update task status.
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
	// Parse command line flags: config file path.
	configPath := flag.String("config", "configs/apiserver.yaml", "config file path")
	flag.Parse()

	// Load config (use default if no config file exists).
	cfg, err := config.Load(*configPath)
	if err != nil {
		panic("failed to load config: " + err.Error())
	}

	log := logger.New(cfg.LogLevel, "apiserver")
	log.Info("starting API server", "addr", cfg.Server.Addr)

	// Connect to external infrastructure.
	ctx := context.Background()
	infra, err := bootstrap.Connect(ctx, cfg)
	if err != nil {
		log.Error("failed to connect infrastructure", "error", err)
		os.Exit(1)
	}
	defer infra.Close(ctx)

	// Ensure database indexes (like users.username unique).
	// Fail does not block startup: common reason is historical duplicate data caused unique index creation fail,
	// just warn, avoid blocking startup by dirty data; service layer already has duplicate check fallbacks.
	if err := infra.Store.EnsureIndexes(ctx); err != nil {
		log.Error("failed to create database indexes (may exist duplicate usernames, clean up restart)", "error", err)
	}

	// Assemble dependencies.
	tokens := auth.NewTokenManager(cfg.Auth.JWTSecret, cfg.Auth.AccessTokenTTL, cfg.Auth.RefreshTokenTTL)
	resLedger := ledger.New()
	sched := scheduler.New(resLedger, cfg.Scheduler.Strategy)

	// Extend services: log, storage, metering, quota, project, notify.
	logSvc := service.NewLogService(infra.LogStream, infra.Object)
	storageSvc := service.NewStorageService(infra.Object, int(cfg.Auth.TokenTTL.Seconds()))
	meteringSvc := service.NewMeteringService(infra.Store.Usages(), cfg.Metering.PricePerGPUHour)
	quotaSvc := service.NewQuotaService(infra.Store.Quotas())
	projectSvc := service.NewProjectService(infra.Store.Projects())
	var notifier notify.Notifier
	if cfg.Notify.WebhookURL != "" {
		notifier = notify.NewWebhookNotifier(cfg.Notify.Platform, cfg.Notify.WebhookURL)
	}

	nodeSvc := service.NewNodeService(infra.Store.Nodes(), infra.Cache, resLedger, cfg.Scheduler.NodeTimeout).
		WithIdleConfig(cfg.Scheduler.IdleThresholdUtil, cfg.Scheduler.IdleThresholdDuration)
	taskSvc := service.NewTaskService(infra.Store.Tasks(), infra.Cache, infra.Producer, sched).
		WithLogService(logSvc).
		WithStorageService(storageSvc).
		WithMetering(meteringSvc).
		WithQuota(quotaSvc).
		WithProjects(projectSvc)
	if notifier != nil {
		taskSvc = taskSvc.WithNotifier(notifier)
	}
	tracer := tracing.NewTracer(tracing.NewSlogExporter(log))
	taskSvc = taskSvc.WithTracer(tracer)

	// Build routes.
	metricsExporter := metrics.NewExporter(infra.Nodes())
	router := api.NewServer(userSvc, nodeSvc, taskSvc, metricsExporter)
	httpServer := &http.Server{Addr: cfg.Server.Addr, Handler: router}

	// Start idle node auto-reclamation scanner with scan interval 1min to balance timeliness and performance overhead.
	go nodeSvc.StartIdleScanner(ctx, time.Minute)

	// Wait for quit signal.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("shutting down API server...")
	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Error("failed to shutdown API server", "error", err)
	}
}
