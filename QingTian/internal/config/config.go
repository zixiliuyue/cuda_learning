// Package config 负责加载与管理平台各服务的配置。
//
// 配置采用 YAML 文件描述，支持通过环境变量覆盖关键连接串，
// 便于在容器化部署时注入敏感信息（如数据库密码）。
package config

import (
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// MongoConfig 描述 MongoDB 连接配置。
type MongoConfig struct {
	URI      string `yaml:"uri"`      // 连接串，例如 mongodb://localhost:27017
	Database string `yaml:"database"` // 数据库名
}

// RedisConfig 描述 Redis 连接配置。
type RedisConfig struct {
	Addr     string `yaml:"addr"`     // 地址，例如 localhost:6379
	Password string `yaml:"password"` // 密码，无则留空
	DB       int    `yaml:"db"`       // 使用的数据库编号
}

// NSQConfig 描述 NSQ 连接配置。
type NSQConfig struct {
	NSQDAddr    string `yaml:"nsqd_addr"`     // 生产者连接的 nsqd 地址，例如 localhost:4150
	LookupdAddr string `yaml:"lookupd_addr"`  // 消费者发现用的 nsqlookupd 地址，例如 localhost:4161
	MaxInFlight int    `yaml:"max_in_flight"` // 消费者最大并发处理消息数
}

// MinIOConfig 描述 MinIO 对象存储配置。
type MinIOConfig struct {
	Endpoint  string `yaml:"endpoint"`   // 服务地址，例如 localhost:9000
	AccessKey string `yaml:"access_key"` // 访问密钥
	SecretKey string `yaml:"secret_key"` // 密钥
	UseSSL    bool   `yaml:"use_ssl"`    // 是否启用 HTTPS
	Bucket    string `yaml:"bucket"`     // 默认存储桶
}

// AuthConfig 描述认证相关配置。
type AuthConfig struct {
	JWTSecret string        `yaml:"jwt_secret"` // JWT 签名密钥，生产环境务必使用强随机值
	TokenTTL  time.Duration `yaml:"token_ttl"`  // 令牌有效期
}

// ServerConfig 描述 HTTP 服务监听配置。
type ServerConfig struct {
	Addr string `yaml:"addr"` // 监听地址，例如 :8080
}

// SchedulerConfig 描述调度器行为配置。
type SchedulerConfig struct {
	Interval    time.Duration `yaml:"interval"`     // 调度轮询间隔
	Strategy    string        `yaml:"strategy"`     // 调度策略：binpack / spread
	NodeTimeout time.Duration `yaml:"node_timeout"` // 节点心跳超时时间，超时判离线
}

// AgentConfig 描述 Agent 配置。
type AgentConfig struct {
	NodeName          string        `yaml:"node_name"`          // 节点名称
	ControlPlaneURL   string        `yaml:"control_plane_url"`  // 控制面 API 地址
	HeartbeatInterval time.Duration `yaml:"heartbeat_interval"` // 心跳上报间隔
	WorkDir           string        `yaml:"work_dir"`           // 任务工作目录
}

// Config 是平台的总配置，聚合各子系统配置。
//
// 不同服务只会用到其中一部分字段，共用同一结构可简化配置文件管理。
type Config struct {
	LogLevel  string          `yaml:"log_level"`
	Edition   string          `yaml:"edition"` // 功能版本：community / enterprise
	Server    ServerConfig    `yaml:"server"`
	Mongo     MongoConfig     `yaml:"mongo"`
	Redis     RedisConfig     `yaml:"redis"`
	NSQ       NSQConfig       `yaml:"nsq"`
	MinIO     MinIOConfig     `yaml:"minio"`
	Auth      AuthConfig      `yaml:"auth"`
	Scheduler SchedulerConfig `yaml:"scheduler"`
	Agent     AgentConfig     `yaml:"agent"`
	Metering  MeteringConfig  `yaml:"metering"`
	Notify    NotifyConfig    `yaml:"notify"`
}

// MeteringConfig 描述计量与计费配置。
type MeteringConfig struct {
	PricePerGPUHour float64 `yaml:"price_per_gpu_hour"` // 单位卡时价格（元/GPU·小时）
}

// NotifyConfig 描述任务通知配置。
type NotifyConfig struct {
	Platform   string `yaml:"platform"`    // 渠道：feishu / wecom
	WebhookURL string `yaml:"webhook_url"` // 群机器人 Webhook 地址
}

// Load 从指定 YAML 文件加载配置，并应用默认值与环境变量覆盖。
//
// 参数：
//   - path: 配置文件路径。
//
// 返回值：
//   - *Config: 解析后的配置。
//   - error:   文件读取或解析失败时返回。
//
// 环境变量覆盖规则（存在时优先）：
//   - QT_MONGO_URI、QT_REDIS_ADDR、QT_NSQD_ADDR、QT_MINIO_ENDPOINT、QT_JWT_SECRET。
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	cfg := Default()
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}
	applyEnvOverrides(cfg)
	return cfg, nil
}

// Default 返回带合理默认值的配置，供未提供配置文件时使用。
func Default() *Config {
	return &Config{
		LogLevel: "info",
		Edition:  "community",
		Server:   ServerConfig{Addr: ":8080"},
		Mongo:    MongoConfig{URI: "mongodb://localhost:27017", Database: "qingtian"},
		Redis:    RedisConfig{Addr: "localhost:6379", DB: 0},
		NSQ:      NSQConfig{NSQDAddr: "localhost:4150", LookupdAddr: "localhost:4161", MaxInFlight: 16},
		MinIO:    MinIOConfig{Endpoint: "localhost:9000", AccessKey: "minioadmin", SecretKey: "minioadmin", Bucket: "qingtian"},
		Auth:     AuthConfig{JWTSecret: "change-me-in-production", TokenTTL: 24 * time.Hour},
		Scheduler: SchedulerConfig{
			Interval:    2 * time.Second,
			Strategy:    "binpack",
			NodeTimeout: 30 * time.Second,
		},
		Agent: AgentConfig{
			HeartbeatInterval: 5 * time.Second,
			WorkDir:           "./agent-work",
		},
		Metering: MeteringConfig{PricePerGPUHour: 1.0},
		Notify:   NotifyConfig{Platform: "feishu"},
	}
}

// applyEnvOverrides 用环境变量覆盖敏感或部署相关配置。
func applyEnvOverrides(cfg *Config) {
	if v := os.Getenv("QT_MONGO_URI"); v != "" {
		cfg.Mongo.URI = v
	}
	if v := os.Getenv("QT_REDIS_ADDR"); v != "" {
		cfg.Redis.Addr = v
	}
	if v := os.Getenv("QT_NSQD_ADDR"); v != "" {
		cfg.NSQ.NSQDAddr = v
	}
	if v := os.Getenv("QT_MINIO_ENDPOINT"); v != "" {
		cfg.MinIO.Endpoint = v
	}
	if v := os.Getenv("QT_JWT_SECRET"); v != "" {
		cfg.Auth.JWTSecret = v
	}
}
