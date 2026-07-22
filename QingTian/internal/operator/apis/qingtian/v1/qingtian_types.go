// Package v1 defines the API types for QingTian Kubernetes Operator.
//
// QingTian Operator manages QingTian clusters on Kubernetes:
//   - Deploys and manages apiserver deployment
//   - Deploys and manages scheduler deployment
//   - Manages node-agent DaemonSet on GPU nodes
//   - Manages MongoDB, Redis, NSQ dependencies (or uses external)
//   - Handles upgrades and scaling
package v1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// QingTianCluster is the Schema for the qingtianclusters API
type QingTianCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   QingTianClusterSpec   `json:"spec"`
	Status QingTianClusterStatus `json:"status,omitempty"`
}

// QingTianClusterSpec defines the desired state of QingTianCluster
type QingTianClusterSpec struct {
	// Version is the QingTian version to deploy
	Version string `json:"version"`

	// Image is the apiserver image (optional, uses default if empty)
	Image string `json:"image,omitempty"`

	// Replicas is the number of apiserver replicas
	Replicas int32 `json:"replicas,omitempty"`

	// EnableSharding enables scheduler sharding for horizontal scaling
	EnableSharding bool `json:"enableSharding,omitempty"`

	// ShardCount is the number of scheduler shards when sharding is enabled
	ShardCount int `json:"shardCount,omitempty"`

	// NodeSelector is the node selector for GPU nodes that run qingtian-agent
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Database configuration
	Database DatabaseSpec `json:"database"`

	// Cache configuration (Redis)
	Cache CacheSpec `json:"cache,omitempty"`

	// Messaging configuration (NSQ) for task dispatch
	Messaging MessagingSpec `json:"messaging,omitempty"`

	// Storage configuration (MinIO) for artifacts
	Storage StorageSpec `json:"storage,omitempty"`

	// Auth configuration for OIDC SSO
	Auth AuthSpec `json:"auth,omitempty"`

	// Resource requirements for apiserver
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`

	// Ingress enables and configures ingress for apiserver
	Ingress IngressSpec `json:"ingress,omitempty"`
}

// DatabaseSpec defines database connection info
type DatabaseSpec struct {
	// Internal enables deploying MongoDB as a Kubernetes StatefulSet (use false for external)
	Internal bool   `json:"internal"`
	// URI is the MongoDB connection URI when using external database
	URI      string `json:"uri,omitempty"`
	// Database is the database name
	Database string `json:"database,omitempty"`
}

// CacheSpec defines Redis cache configuration
type CacheSpec struct {
	// Internal enables deploying Redis as a Kubernetes Deployment
	Internal bool   `json:"internal"`
	// Address is the Redis address when using external
	Address string `json:"address,omitempty"`
	// Password is the Redis password when using external
	Password string `json:"password,omitempty"`
	// DB is the Redis database number
	DB int `json:"db,omitempty"`
}

// MessagingSpec defines NSQ messaging configuration
type MessagingSpec struct {
	// Internal enables deploying NSQ as a Kubernetes Deployment
	Internal  bool   `json:"internal"`
	// NSQDAddr is the nsqd address when using external
	NSQDAddr string `json:"nsqdAddr,omitempty"`
	// LookupdAddr is the nsqlookupd address when using external
	LookupdAddr string `json:"lookupdAddr,omitempty"`
}

// StorageSpec defines MinIO storage configuration
type StorageSpec struct {
	// Internal enables deploying MinIO as a Kubernetes StatefulSet
	Internal bool   `json:"internal"`
	// Endpoint is the MinIO endpoint when using external
	Endpoint string `json:"endpoint,omitempty"`
	// AccessKey is the MinIO access key
	AccessKey string `json:"accessKey,omitempty"`
	// SecretKey is the MinIO secret key
	SecretKey string `json:"secretKey,omitempty"`
	// Bucket is the default bucket name
	Bucket string `json:"bucket,omitempty"`
	// UseSSL enables HTTPS
	UseSSL bool `json:"useSSL,omitempty"`
}

// AuthSpec defines authentication configuration
type AuthSpec struct {
	// JWTSecret is the JWT signing secret (stored in Secret)
	JWTSecret string `json:"jwtSecret,omitempty"`
	// OIDC configures OIDC SSO login
	OIDC *OIDCSpec `json:"oidc,omitempty"`
}

// OIDCSpec defines OIDC configuration
type OIDCSpec struct {
	Enabled        bool   `json:"enabled"`
	Issuer         string `json:"issuer"`
	ClientID       string `json:"clientID"`
	ClientSecret   string `json:"clientSecret"`
	RedirectURL    string `json:"redirectURL"`
	DefaultRole    string `json:"defaultRole,omitempty"`
	UserNameClaim  string `json:"userNameClaim,omitempty"`
}

// IngressSpec defines ingress configuration
type IngressSpec struct {
	Enabled                bool              `json:"enabled"`
	Host                   string              `json:"host,omitempty"`
	Annotations            map[string]string `json:"annotations,omitempty"`
	TLS                    []IngressTLS       `json:"tls,omitempty"`
	IngressClassName       *string           `json:"ingressClassName,omitempty"`
}

// IngressTLS defines TLS configuration for ingress
type IngressTLS struct {
	Hosts      []string `json:"hosts,omitempty"`
	SecretName string  `json:"secretName,omitempty"`
}

// QingTianClusterStatus defines the observed state of QingTianCluster
type QingTianClusterStatus struct {
	// Phase is the current phase of the cluster
	Phase ClusterPhase `json:"phase"`

	// Version is the currently running version
	Version string `json:"version"`

	// ReadyReplicas is the number of ready apiserver replicas
	ReadyReplicas int32 `json:"readyReplicas"`

	// AgentCount is the number of running agent pods
	AgentCount int `json:"agentCount"`

	// Conditions represents the latest available observations
	Conditions []ClusterCondition `json:"conditions,omitempty"`

	// LastUpdated is the last time the status was updated
	LastUpdated metav1.Time `json:"lastUpdated,omitempty"`
}

// ClusterPhase defines the phase of the cluster
type ClusterPhase string

const (
	// ClusterPhasePending means the cluster is being created
	ClusterPhasePending ClusterPhase = "Pending"
	// ClusterPhaseRunning means the cluster is running and ready
	ClusterPhaseRunning ClusterPhase = "Running"
	// ClusterPhaseFailed means the cluster creation failed
	ClusterPhaseFailed ClusterPhase = "Failed"
	// ClusterPhaseUpgrading means the cluster is being upgraded
	ClusterPhaseUpgrading ClusterPhase = "Upgrading"
)

// ClusterCondition describes cluster condition
type ClusterCondition struct {
	Type           ClusterConditionType `json:"type"`
	Status         corev1.ConditionStatus `json:"status"`
	LastUpdateTime metav1.Time          `json:"lastUpdateTime"`
	Reason         string                `json:"reason"`
	Message        string                `json:"message"`
}

// ClusterConditionType defines condition type
type ClusterConditionType string

const (
	// ClusterConditionAvailable means the cluster is available
	ClusterConditionAvailable ClusterConditionType = "Available"
	// ClusterConditionDatabaseReady means database is ready
	ClusterConditionDatabaseReady ClusterConditionType = "DatabaseReady"
	// ClusterConditionCacheReady means cache is ready
	ClusterConditionCacheReady ClusterConditionType = "CacheReady"
	// ClusterConditionApiserverReady means apiserver is ready
	ClusterConditionApiserverReady ClusterConditionType = "ApiserverReady"
	// ClusterConditionAgentReady means agent DaemonSet is ready
	ClusterConditionAgentReady ClusterConditionType = "AgentReady"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// QingTianClusterList contains a list of QingTianCluster
type QingTianClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []QingTianCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&QingTianCluster{}, &QingTianClusterList{})
}
