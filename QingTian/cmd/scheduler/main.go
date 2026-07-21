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
	"sync"
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

	// 组装调度所需依赖。账本状态存放于 Redis 共享快照，多个调度器实例
	// 在 leader 锁保护下"加载->调度->写回"，因此此处只需创建空账本。
	resLedger := ledger.New()
	sched := scheduler.New(resLedger, cfg.Scheduler.Strategy)
	nodeSvc := service.NewNodeService(infra.Store.Nodes(), infra.Cache, resLedger, cfg.Scheduler.NodeTimeout).
		WithIdleConfig(cfg.Scheduler.IdleThresholdUtil, cfg.Scheduler.IdleThresholdDuration)
	tracer := tracing.NewTracer(tracing.NewSlogExporter(log))
	taskSvc := service.NewTaskService(infra.Store.Tasks(), infra.Store.Nodes(), infra.Cache, infra.Producer, sched).
		WithTracer(tracer).WithLedgerStore(infra.Cache).
		WithNodeService(nodeSvc)

	// 启动调度循环。
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		runSchedulerLoop(ctx, log, cfg, infra, taskSvc)
	}()

	log.Info("调度器服务已启动")
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("正在关闭调度器服务...")
	cancel()
	wg.Wait() // 等待调度循环真正退出，确保锁释放与在途处理完成
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
			// 准备账本：优先加载 Redis 共享快照并纳入运行期节点变更；共享存储
			// 尚无快照时回退为从 MongoDB 全量重建。失败则跳过本轮，避免在错误
			// 账本上调度导致超卖。
			if err := taskSvc.SyncLedger(ctx); err != nil {
				log.Error("准备资源账本失败，跳过本轮调度", "error", err)
				_ = infra.Cache.Release(ctx, lockKey)
				continue
			}
			// 批量调度：处理队列中所有任务，配合 Redis 分布式锁保证只有一个 leader 在工作。
			taskSvc.ScheduleBatch(ctx)
			// 超时防护：终止运行超时的任务并归还资源，防止 GPU 泄漏。
			if killed, err := taskSvc.EnforceTimeouts(ctx); err != nil {
				log.Error("超时防护巡检失败", "error", err)
			} else if killed > 0 {
				log.Info("已终止超时任务", "count", killed)
			}
			// 节点故障自动 reconciliation：将离线节点上的未完成任务放回排队重调度。
			if moved, err := taskSvc.ReconcileNodeFailures(ctx); err != nil {
				log.Error("节点故障重调度巡检失败", "error", err)
			} else if moved > 0 {
				log.Info("已将离线节点任务放回重调度", "count", moved)
			}
			// CAS 写回账本快照：若快照在调度期间被其它实例修改，
			// 则放弃写入（安全降级：下一轮从最新快照重建即可）。
			ok, err := taskSvc.PersistLedger(ctx)
			if err != nil {
				log.Error("写回资源账本快照失败", "error", err)
			} else if !ok {
				log.Info("账本快照已被其它实例更新，本轮放弃写回")
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
