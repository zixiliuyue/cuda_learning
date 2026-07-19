// Command scheduler 是 QingTian 调度器服务的入口。
//
// 它周期性地从待调度队列取出任务，执行调度决策并通过 NSQ 下发到目标节点。
// 调度器与 API 服务共享 MongoDB/Redis/NSQ，通过分布式锁保证多实例互斥。
package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"
	"time"

	"qingtian/internal/bootstrap"
	"qingtian/internal/config"
	"qingtian/internal/ledger"
	"qingtian/internal/logger"
	"qingtian/internal/scheduler"
	"qingtian/internal/service"
	"qingtian/internal/tracing"
)

func main() {
	configPath := flag.String("config", "configs/scheduler.yaml", "配置文件路径")
	flag.Parse()

	cfg := config.Default()
	if _, err := os.Stat(*configPath); err == nil {
		loaded, lerr := config.Load(*configPath)
		if lerr != nil {
			panic("加载配置失败: " + lerr.Error())
		}
		cfg = loaded
	}

	log := logger.New(cfg.LogLevel, "scheduler")
	log.Info("正在启动调度器服务", "strategy", cfg.Scheduler.Strategy, "interval", cfg.Scheduler.Interval)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	infra, err := bootstrap.Connect(ctx, cfg)
	if err != nil {
		log.Error("连接基础设施失败", "error", err)
		os.Exit(1)
	}
	defer infra.Close(ctx)

	// 组装调度所需依赖。账本需要从节点仓储加载已注册节点的 GPU 容量。
	resLedger := ledger.New()
	nodes, err := infra.Store.Nodes().List(ctx)
	if err == nil {
		for _, n := range nodes {
			resLedger.SyncNode(n)
		}
	}
	sched := scheduler.New(resLedger, cfg.Scheduler.Strategy)
	tracer := tracing.NewTracer(tracing.NewSlogExporter(log))
	taskSvc := service.NewTaskService(infra.Store.Tasks(), infra.Store.Nodes(), infra.Cache, infra.Producer, sched).WithTracer(tracer)

	// 启动调度循环。
	go runSchedulerLoop(ctx, log, cfg, infra, taskSvc)

	log.Info("调度器服务已启动")
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("正在关闭调度器服务...")
	cancel()
	time.Sleep(200 * time.Millisecond) // 给循环退出留出时间
}

// runSchedulerLoop 是调度器主循环。
//
// 每个调度周期尝试获取分布式锁（保证多实例互斥），然后持续处理队列中的任务，
// 直到队列为空或本轮处理数达到上限，随后释放锁并等待下一周期。
func runSchedulerLoop(ctx context.Context, log logInterface, cfg *config.Config, infra *bootstrap.Infra, taskSvc *service.TaskService) {
	ticker := time.NewTicker(cfg.Scheduler.Interval)
	defer ticker.Stop()
	const lockKey = "scheduler:leader"

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// 获取调度锁，避免多个调度器实例同时调度导致资源竞争。
			acquired, err := infra.Cache.Acquire(ctx, lockKey, cfg.Scheduler.Interval)
			if err != nil || !acquired {
				continue
			}
			// 批量调度：单轮最多处理 100 个任务，防止长期占用锁。
			if _, err := taskSvc.ScheduleBatch(ctx, 100); err != nil {
				log.Error("批量调度任务失败", "error", err)
			}
			// 超时防护：终止运行超时的任务并归还资源，防止 GPU 泄漏。
			if killed, err := taskSvc.EnforceTimeouts(ctx); err != nil {
				log.Error("超时防护巡检失败", "error", err)
			} else if killed > 0 {
				log.Info("已终止超时任务", "count", killed)
			}
			_ = infra.Cache.Release(ctx, lockKey)
		}
	}
}

// logInterface 是对日志器的最小抽象，便于本文件解耦具体日志实现。
type logInterface interface {
	Error(msg string, args ...any)
	Info(msg string, args ...any)
}
