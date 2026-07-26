// Command agent 是 QingTian 节点 Agent 的入口。
//
// Agent 部署在每个 GPU 节点上，负责向控制面注册、上报心跳、
// 消费任务下发消息并执行任务、回传状态。
package main

import (
	"context"
	"encoding/json"
	"flag"
	"os"
	"os/signal"
	"syscall"

	"qingtian/internal/agent"
	"qingtian/internal/config"
	"qingtian/internal/logger"
	"qingtian/internal/mq"
)

func main() {
	configPath := flag.String("config", "configs/agent.yaml", "配置文件路径")
	flag.Parse()

	cfg := config.Default()
	if _, err := os.Stat(*configPath); err == nil {
		loaded, lerr := config.Load(*configPath)
		if lerr != nil {
			panic("加载配置失败: " + lerr.Error())
		}
		cfg = loaded
	}

	log := logger.New(cfg.LogLevel, "agent")

	// 确定节点主机名。
	hostname := cfg.Agent.NodeName
	if hostname == "" {
		hostname, _ = os.Hostname()
	}
	log.Info("正在启动 Agent", "hostname", hostname, "control_plane", cfg.Agent.ControlPlaneURL)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 组装 Agent 依赖。
	// 采集器：无 NVML 环境使用模拟采集器（对应 RTX 5060 Ti 16G）。
	collector := agent.DefaultMockCollector()
	// 消息队列：NSQ 生产者与消费者。
	producer, err := mq.NewNSQProducer(cfg.NSQ)
	if err != nil {
		log.Error("创建 NSQ 生产者失败", "error", err)
		os.Exit(1)
	}
	defer producer.Close()
	consumer := mq.NewNSQConsumer(cfg.NSQ)
	defer consumer.Close()
	// 执行器：流式命令执行，逐行把日志经 NSQ 回传控制面，支撑实时日志流。
	logSink := func(taskID, line string) {
		body, _ := json.Marshal(mq.LogMessage{TaskID: taskID, Line: line})
		_ = producer.Publish(ctx, mq.TopicTaskLog, body)
	}
	executor := agent.NewStreamingExecutor(cfg.Agent.WorkDir, logSink)
	// 安全警示：本地执行器直接在宿主机 shell 中运行任务命令，未做容器隔离。
	// 一旦被授权用户提交恶意命令，将影响宿主机。生产环境必须通过容器
	// （nvidia-container-runtime）隔离执行，详见 README「安全说明」。
	log.Warn("任务执行采用本地 shell（无容器隔离），生产环境请改用容器执行器以隔离宿主机")
	// 控制面客户端：HTTP。
	cp := agent.NewHTTPControlPlane(cfg.Agent.ControlPlaneURL)

	ag := agent.New(agent.Config{
		Hostname:          hostname,
		IP:                cfg.Agent.NodeName,
		HeartbeatInterval: cfg.Agent.HeartbeatInterval,
	}, collector, executor, cp, producer, consumer, log)

	// 注册节点。
	if err := ag.Register(ctx); err != nil {
		log.Error("节点注册失败", "error", err)
		os.Exit(1)
	}
	// 订阅任务下发。
	if err := ag.SubscribeDispatch(); err != nil {
		log.Error("订阅任务下发失败", "error", err)
		os.Exit(1)
	}
	// 订阅任务控制指令（超时/抢占终止）。
	if err := ag.SubscribeControl(); err != nil {
		log.Error("订阅任务控制指令失败", "error", err)
		os.Exit(1)
	}
	// 启动心跳循环。
	go ag.RunHeartbeat(ctx)

	log.Info("Agent 已启动")
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("正在关闭 Agent...")
	cancel()
}
