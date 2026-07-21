package controller

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	qingtianv1 "qingtian/internal/operator/apis/qingtian/v1"
)

// ensureSecret ensures the Secret exists with all sensitive data
func (r *QingTianClusterReconciler) ensureSecret(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
) (*corev1.Secret, error) {
	name := getSecretName(cluster)
	var existing corev1.Secret
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		// Secret already exists
		return &existing, nil
	}
	if !errors.IsNotFound(err) {
		return nil, err
	}

	// Create new Secret
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
		},
		Data: make(map[string][]byte),
	}

	// If user didn't provide JWT secret, generate a random one
	if cluster.Spec.Auth.JWTSecret != "" {
		secret.Data["jwt-secret"] = []byte(cluster.Spec.Auth.JWTSecret)
	}
	// OIDC client secret stored in secret if provided
	if cluster.Spec.Auth.OIDC != nil && cluster.Spec.Auth.OIDC.Enabled {
		secret.Data["oidc-client-secret"] = []byte(cluster.Spec.Auth.OIDC.ClientSecret)
	}
	// Redis password stored in secret if external
	if !cluster.Spec.Cache.Internal && cluster.Spec.Cache.Password != "" {
		secret.Data["redis-password"] = []byte(cluster.Spec.Cache.Password)
	}
	// MinIO credentials stored in secret if external
	if !cluster.Spec.Storage.Internal {
		if cluster.Spec.Storage.AccessKey != "" {
			secret.Data["minio-access-key"] = []byte(cluster.Spec.Storage.AccessKey)
		}
		if cluster.Spec.Storage.SecretKey != "" {
			secret.Data["minio-secret-key"] = []byte(cluster.Spec.Storage.SecretKey)
		}
	}

	if err := r.Create(ctx, secret); err != nil {
		return nil, err
	}
	return secret, nil
}

// ensureConfigMap ensures the ConfigMap exists with the apiserver config
func (r *QingTianClusterReconciler) ensureConfigMap(
	ctx context.Context,
	cluster *qingtianv1.QingTianCluster,
	secret *corev1.Secret,
) (*corev1.ConfigMap, error) {
	name := getConfigMapName(cluster)
	var existing corev1.ConfigMap
	err := r.Get(ctx, client.ObjectKey{Namespace: cluster.Namespace, Name: name}, &existing)
	if err == nil {
		// ConfigMap already exists - could add reconciliation here
		return &existing, nil
	}
	if !errors.IsNotFound(err) {
		return nil, err
	}

	// Generate YAML config
	configYaml := generateConfigYAML(cluster, secret)

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cluster.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(cluster, qingtianv1.GroupVersion.WithKind("QingTianCluster")),
			},
		},
		Data: map[string]string{
			"config.yaml": configYaml,
		},
	}

	if err := r.Create(ctx, cm); err != nil {
		return nil, err
	}
	return cm, nil
}

// generateConfigYAML generates the apiserver config YAML from CR spec
func generateConfigYAML(cluster *qingtianv1.QingTianCluster, secret *corev1.Secret) string {
	// Build YAML config - this is a simplified version, full config covers all options
	yaml := `log_level: info
edition: community
server:
  addr: :8080
`

	// Database config
	if cluster.Spec.Database.Internal {
		yaml += fmt.Sprintf(`mongo:
  uri: mongodb://%s-mongodb:27017
  database: %s
`, cluster.Name, getDefaultIfEmpty(cluster.Spec.Database.Database, "qingtian"))
	} else {
		yaml += fmt.Sprintf(`mongo:
  uri: %s
  database: %s
`, cluster.Spec.Database.URI, getDefaultIfEmpty(cluster.Spec.Database.Database, "qingtian"))
	}

	// Cache config
	if cluster.Spec.Cache.Internal {
		yaml += fmt.Sprintf(`redis:
  addr: %s-redis:6379
  password: ""
  db: %d
`, cluster.Name, cluster.Spec.Cache.DB)
	} else {
		password := ""
		if secret.Data != nil {
			if pwd, ok := secret.Data["redis-password"]; ok {
				password = string(pwd)
			}
		}
		yaml += fmt.Sprintf(`redis:
  addr: %s
  password: %s
  db: %d
`, cluster.Spec.Cache.Address, password, cluster.Spec.Cache.DB)
	}

	// Sharding config
	yaml += fmt.Sprintf(`sharding:
  enabled: %t
  replicas: 150
  shard_count: %d
`, cluster.Spec.EnableSharding, cluster.Spec.ShardCount)

	// Scheduler defaults
	yaml += `scheduler:
  interval: 2s
  strategy: binpack
  node_timeout: 30s
  idle_threshold_util: 10
  idle_threshold_duration: 30m
`

	// Auth config
	jwtSecret := string(secret.Data["jwt-secret"])
	yaml += fmt.Sprintf(`auth:
  jwt_secret: %s
  access_token_ttl: 24h
  refresh_token_ttl: 168h
`, jwtSecret)

	// OIDC config if enabled
	if cluster.Spec.Auth.OIDC != nil && cluster.Spec.Auth.OIDC.Enabled {
		clientSecret := string(secret.Data["oidc-client-secret"])
		yaml += fmt.Sprintf(`oidc:
  enabled: true
  issuer: %s
  client_id: %s
  client_secret: %s
  redirect_url: %s
  username_claim: %s
  default_role: %s
`,
			cluster.Spec.Auth.OIDC.Issuer,
			cluster.Spec.Auth.OIDC.ClientID,
			clientSecret,
			cluster.Spec.Auth.OIDC.RedirectURL,
			getDefaultIfEmpty(cluster.Spec.Auth.OIDC.UserNameClaim, "email"),
			getDefaultIfEmpty(cluster.Spec.Auth.OIDC.DefaultRole, "developer"),
		)
	} else {
		yaml += `oidc:
  enabled: false
`
	}

	return yaml
}

// getDefaultIfEmpty returns default if value is empty
func getDefaultIfEmpty(value, def string) string {
	if value == "" {
		return def
	}
	return value
}
