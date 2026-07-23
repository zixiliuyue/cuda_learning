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
	var resLedger *ledger.Ledger
	var sched scheduler.SchedulerInterface
	if cfg.Sharding.Enabled {
		// Sharded scheduler for horizontal scaling across multiple shards
		sharded := scheduler.NewShardedScheduler(scheduler.ShardingConfig{
			Enabled:  cfg.Sharding.Enabled,
			Replicas: cfg.Sharding.Replicas,
			ShardCount: cfg.Sharding.ShardCount,
		}, cfg.Scheduler.Strategy)
		// Note: Caller needs to add shards and nodes to the sharded scheduler after creation
		// Typically this is done when starting up from database by reading all nodes and grouping
		// them into configured shards.
		sched = sharded
		log.Info("Sharded scheduler enabled", "replicas", cfg.Sharding.Replicas, "shard_count", cfg.Sharding.ShardCount)
		// For sharded mode, node service still needs a ledger for heartbeat caching.
		// We use a single global ledger for node info cache since it's only tracking current node status,
		// not the resource allocation which is handled per shard.
		resLedger = ledger.New()
	} else {
		// Single scheduler for normal deployment
		resLedger = ledger.New()
		sched = scheduler.New(resLedger, cfg.Scheduler.Strategy)
	}


	// Extend services: log, storage, metering, quota, project, notify.
	logSvc := service.NewLogService(infra.LogStream, infra.Object)
	_ = service.NewStorageService(infra.Object, int(cfg.Auth.AccessTokenTTL.Seconds()))
	meteringSvc := service.NewMeteringService(infra.Store.Usages(), cfg.Metering.PricePerGPUHour)
	quotaSvc := service.NewQuotaService(infra.Store.Quotas())
	projectSvc := service.NewProjectService(infra.Store.Projects())
	templateSvc := service.NewTaskTemplateService(infra.Store.TaskTemplates())
	terminalMgr := service.NewTerminalSessionManager(infra.Store.Tasks())
	var notifier notify.Notifier
	if cfg.Notify.WebhookURL != "" {
		notifier = notify.NewWebhookNotifier(cfg.Notify.Platform, cfg.Notify.WebhookURL)
	}

	userSvc := service.NewUserService(infra.Store.Users(), tokens)

	nodeSvc := service.NewNodeService(infra.Store.Nodes(), infra.Cache, resLedger, cfg.Scheduler.NodeTimeout).
		WithIdleConfig(cfg.Scheduler.IdleThresholdUtil, cfg.Scheduler.IdleThresholdDuration)
	taskSvc := service.NewTaskService(infra.Store.Tasks(), infra.Store.Nodes(), infra.Cache, infra.Producer, sched).
		WithLogService(logSvc).
		WithMetering(meteringSvc).
		WithQuota(quotaSvc).
		WithProjects(projectSvc).
		WithNodeService(nodeSvc)
	if notifier != nil {
		taskSvc = taskSvc.WithNotifier(notifier)
	}
	tracer := tracing.NewTracer(tracing.NewSlogExporter(log))
	taskSvc = taskSvc.WithTracer(tracer)

	// Initialize SSO OIDC service if enabled
	var ssoSvc *service.SSOService
	if cfg.OIDC.Enabled {
		oidcCfg := auth.OIDCConfig{
			Issuer:        cfg.OIDC.Issuer,
			ClientID:      cfg.OIDC.ClientID,
			ClientSecret:  cfg.OIDC.ClientSecret,
			RedirectURL:   cfg.OIDC.RedirectURL,
			Scopes:        cfg.OIDC.Scopes,
			UserNameClaim: cfg.OIDC.UserNameClaim,
		}
		oidcAuth, err := auth.NewOIDCAuthenticator(ctx, oidcCfg)
		if err != nil {
			log.Error("failed to initialize OIDC authenticator", "error", err)
			os.Exit(1)
		}
		ssoSvc = service.NewSSOService(oidcAuth, infra.Store.Users(), tokens, cfg.OIDC.DefaultRole)
		log.Info("OIDC SSO enabled", "issuer", cfg.OIDC.Issuer)
	}

	// Build routes.
	metricsExporter := metrics.NewExporter(infra.Store.Nodes())

	// 初始化工作流服务
	workflowRepo := infra.Store.Workflows()
	workflowRunRepo := infra.Store.WorkflowRuns()
	taskRunRepo := infra.Store.TaskRuns()
	scheduleRepo := infra.Store.Schedules()

	workflowSvc := service.NewWorkflowService(workflowRepo)
	workflowRunSvc := service.NewWorkflowRunService(workflowRunRepo)
	defaultSubmitter := service.NewDefaultTaskSubmitter(taskSvc, "")
	wfEngine := service.NewWorkflowEngine(workflowRepo, workflowRunRepo, taskRunRepo, defaultSubmitter)
	scheduleSvc := service.NewScheduleService(scheduleRepo, wfEngine)

	server := api.NewServer(userSvc, nodeSvc, taskSvc, tokens).
		WithTemplates(templateSvc).
		WithTerminals(terminalMgr).
		WithSSO(ssoSvc).
		WithMetering(meteringSvc).
		WithQuota(quotaSvc).
		WithProjects(projectSvc).
		WithEdition(domain.Edition(cfg.Edition)).
		WithWorkflow(workflowSvc, wfEngine, workflowRunSvc, scheduleSvc).
		WithMetricsExporter(metricsExporter)
	mux := server.Routes(api.AuthMiddleware(tokens))
	httpServer := &http.Server{Addr: cfg.Server.Addr, Handler: mux}

	// Start idle node auto-reclamation scanner with scan interval 1min to balance timeliness and performance overhead.
	go nodeSvc.StartIdleScanner(ctx, time.Minute)

	// Start HTTP server in a goroutine
	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("failed to start HTTP server", "error", err)
			os.Exit(1)
		}
	}()

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
