// Command agent 是 QingTian 节点 Agent 的入口。
//
// Agent 部署在每个 GPU 节点上，负责向控制面注册、上报心跳、
// 消费任务下发消息并执行任务、回传状态。
package main

import (
	"context"
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
	// 执行器：本地命令执行。
	executor := agent.NewCommandExecutor(cfg.Agent.WorkDir)
	// 控制面客户端：HTTP。
	cp := agent.NewHTTPControlPlane(cfg.Agent.ControlPlaneURL)
	// 消息队列：NSQ 生产者与消费者。
	producer, err := mq.NewNSQProducer(cfg.NSQ)
	if err != nil {
		log.Error("创建 NSQ 生产者失败", "error", err)
		os.Exit(1)
	}
	defer producer.Close()
	consumer := mq.NewNSQConsumer(cfg.NSQ)
	defer consumer.Close()

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
	// 启动心跳循环。
	go ag.RunHeartbeat(ctx)

	log.Info("Agent 已启动")
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Info("正在关闭 Agent...")
	cancel()
}
