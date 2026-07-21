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
	NSQDAddr    string `yaml:"nsqd_addr"`    // 生产者连接的 nsqd 地址，例如 localhost:4150
	LookupdAddr string `yaml:"lookupd_addr"` // 消费者发现用的 nsqlookupd 地址，例如 localhost:4161
	MaxInFlight int    `yaml:"max_in_flight"` // 消费者最大并发处理消息数
}

// MinIOConfig 描述 MinIO 对象存储配置。
type MinIOConfig struct {
	Endpoint  string `yaml:"endpoint"`  // 服务地址，例如 localhost:9000
	AccessKey string `yaml:"access_key"`  // 访问密钥
	SecretKey string `yaml:"secret_key"`  // 密钥
	Bucket  string `yaml:"bucket"`    // 默认存储桶
	UseSSL  bool   `yaml:"use_ssl"`    // 是否启用 HTTPS
}

// AuthConfig 描述认证相关配置。
type AuthConfig struct {
	JWTSecret         string        `yaml:"jwt_secret"`         // JWT 签名密钥，生产环境务必使用强随机值
	AccessTokenTTL  time.Duration `yaml:"access_token_ttl"`  // 访问令牌有效期
	RefreshTokenTTL time.Duration `yaml:"refresh_token_ttl"` // 刷新令牌有效期
}

// ServerConfig 描述 HTTP 服务监听配置。
type ServerConfig struct {
	Addr string `yaml:"addr"` // 监听地址，例如 :8080
}

// SchedulerConfig 描述调度器行为配置。
type SchedulerConfig struct {
	Interval             time.Duration     `yaml:"interval"`             // 调度轮询间隔
	Strategy             string          `yaml:"strategy"`           // 调度策略：binpack / spread
	NodeTimeout          time.Duration     `yaml:"node_timeout"`        // 节点心跳超时，超时判离线
	IdleThresholdUtil    int             `yaml:"idle_threshold_util"`    // 利用率阈值（百分比），低于此值认为空闲
	IdleThresholdDuration time.Duration `yaml:"idle_threshold_duration"` // 持续空闲多久才触发自动回收
}

// ShardingConfig 描述调度分片水平扩展配置。
type ShardingConfig struct {
	Enabled  bool `yaml:"enabled"`  // 是否启用分片水平扩展
	Replicas int  `yaml:"replicas"` // 一致性哈希虚拟节点副本数，默认 150
	ShardCount int `yaml:"shard_count"` // 固定分片数量，0 表示按节点分片
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
	LogLevel   string          `yaml:"log_level"`
	Edition    string          `yaml:"edition"`
	Server     ServerConfig    `yaml:"server"`
	Mongo      MongoConfig    `yaml:"mongo"`
	Redis      RedisConfig    `yaml:"redis"`
	NSQ        NSQConfig     `yaml:"nsq"`
	MinIO      MinIOConfig    `yaml:"minio"`
	Auth       AuthConfig    `yaml:"auth"`
	OIDC       OIDCConfig     `yaml:"oidc"`
	Scheduler  SchedulerConfig `yaml:"scheduler"`
	Sharding   ShardingConfig  `yaml:"sharding"`
	Agent      AgentConfig    `yaml:"agent"`
	Metering   MeteringConfig  `yaml:"metering"`
	Notify     NotifyConfig    `yaml:"notify"`
}

// MeteringConfig 描述计量与计费配置。
type MeteringConfig struct {
	PricePerGPUHour float64 `yaml:"price_per_gpu_hour"` // 单位卡时价格（元/GPU·小时）
}

// NotifyConfig 描述任务通知配置。
type NotifyConfig struct {
	Platform   string `yaml:"platform"`   // 渠道：feishu / wecom
	WebhookURL string `yaml:"webhook_url"` // 群机器人 Webhook 地址
}

// OIDCConfig 描述 OIDC SSO 认证配置。
type OIDCConfig struct {
	Enabled        bool   `yaml:"enabled"`         // 是否启用 OIDC SSO
	Issuer         string `yaml:"issuer"`          // OIDC Provider 地址，例如 https://keycloak.example.com/realms/myrealm
	ClientID       string `yaml:"client_id"`       // OAuth Client ID
	ClientSecret   string `yaml:"client_secret"`   // OAuth Client Secret
	RedirectURL    string `yaml:"redirect_url"`    // 回调地址，例如 https://qingtian.example.com/api/v1/auth/sso/callback
	Scopes         string `yaml:"scopes"`          // 请求的 scope，默认 openid email profile
	UserNameClaim  string `yaml:"username_claim"`  // 哪个 claim 用作用户名，默认 email
	DefaultRole    string `yaml:"default_role"`    // 新用户默认角色，默认 developer
}

// Load 从指定 YAML 文件加载配置，并应用默认值与环境变量覆盖。
//
// 参数：
//
//	path: 配置文件路径。
//
// 返回值：
//
//	- *Config: 解析后的配置。
//	- error:  loading 错误时返回。
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
		Redis:    RedisConfig{Addr: "localhost:6379", Password: "", DB: 0},
		NSQ:      NSQConfig{NSQDAddr: "localhost:4150", LookupdAddr: "localhost:4161", MaxInFlight: 16},
		MinIO:     MinIOConfig{Endpoint: "localhost:9000", AccessKey: "minioadmin", SecretKey: "minioadmin", Bucket: "qingtian", UseSSL: false},
		Auth:     AuthConfig{JWTSecret: "change-me-in-production", AccessTokenTTL: 24 * time.Hour, RefreshTokenTTL: 7 * 24 * time.Hour},
		OIDC: OIDCConfig{
			Enabled:       false,
			Scopes:        "openid email profile",
			UserNameClaim: "email",
			DefaultRole:   "developer",
		},
		Scheduler: SchedulerConfig{
			Interval:             2 * time.Second,
			Strategy:             "binpack",
			NodeTimeout:          30 * time.Second,
			IdleThresholdUtil:    10,
			IdleThresholdDuration: 30 * time.Minute,
		},
		Sharding: ShardingConfig{
			Enabled:  false,
			Replicas: 150,
		},
		Agent: AgentConfig{
			NodeName:          "",
			ControlPlaneURL:   "",
			HeartbeatInterval: 5 * time.Second,
			WorkDir:           "./agent-work",
		},
		Metering: MeteringConfig{
			PricePerGPUHour: 1.0,
		},
		Notify:   NotifyConfig{
			Platform: "feishu",
			WebhookURL: "",
		},
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
	if v := os.Getenv("QT_NSQ_LOOKUPD_ADDR"); v != "" {
		cfg.NSQ.LookupdAddr = v
	}
	if v := os.Getenv("QT_MINIO_ENDPOINT"); v != "" {
		cfg.MinIO.Endpoint = v
	}
	if v := os.Getenv("QT_JWT_SECRET"); v != "" {
		cfg.Auth.JWTSecret = v
	}
	if v := os.Getenv("QT_ACCESS_TOKEN_TTL"); v != "" {
		d, err := time.ParseDuration(v)
		if err == nil {
			cfg.Auth.AccessTokenTTL = d
		}
	}
	if v := os.Getenv("QT_REFRESH_TOKEN_TTL"); v != "" {
		d, err := time.ParseDuration(v)
		if err == nil {
			cfg.Auth.RefreshTokenTTL = d
		}
	}
	if v := os.Getenv("QT_CONTROL_PLANE_URL"); v != "" {
		cfg.Agent.ControlPlaneURL = v
	}
	if v := os.Getenv("QT_OIDC_ENABLED"); v != "" {
		cfg.OIDC.Enabled = v == "true" || v == "1"
	}
	if v := os.Getenv("QT_OIDC_ISSUER"); v != "" {
		cfg.OIDC.Issuer = v
	}
	if v := os.Getenv("QT_OIDC_CLIENT_ID"); v != "" {
		cfg.OIDC.ClientID = v
	}
	if v := os.Getenv("QT_OIDC_CLIENT_SECRET"); v != "" {
		cfg.OIDC.ClientSecret = v
	}
	if v := os.Getenv("QT_OIDC_REDIRECT_URL"); v != "" {
		cfg.OIDC.RedirectURL = v
	}
	if v := os.Getenv("QT_OIDC_DEFAULT_ROLE"); v != "" {
		cfg.OIDC.DefaultRole = v
	}
}
