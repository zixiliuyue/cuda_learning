// Package controller implements the QingTianCluster controller.
//
// The controller watches QingTianCluster CR and creates all required Kubernetes resources
// to run a complete QingTian cluster on your Kubernetes cluster.
package controller

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"

	qingtianv1 "qingtian/internal/operator/apis/qingtian/v1"
)

// QingTianClusterReconciler reconciles a QingTianCluster object
type QingTianClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// New creates a new controller instance
func New(mgr manager.Manager) *QingTianClusterReconciler {
	return &QingTianClusterReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}
}

// SetupWithManager sets up the controller with the Manager
func (r *QingTianClusterReconciler) SetupWithManager(mgr manager.Manager) error {
	c, err := controller.New("qingtiancluster", mgr, controller.Options{Reconciler: r})
	if err != nil {
		return err
	}

	// Watch for changes to QingTianCluster
	err = c.Watch(
		&source.Kind{Type: &qingtianv1.QingTianCluster{}},
		&handler.EnqueueRequestForObject{},
	)
	if err != nil {
		return err
	}

	// Watch for changes to Deployment owned by QingTianCluster
	err = c.Watch(
		&source.Kind{Type: &appsv1.Deployment{}},
		&handler.EnqueueRequestForOwner{
			IsController: true,
			OwnerType:    &qingtianv1.QingTianCluster{},
		},
	)
	if err != nil {
		return err
	}

	// Watch for changes to DaemonSet owned by QingTianCluster
	err = c.Watch(
		&source.Kind{Type: &appsv1.DaemonSet{}},
		&handler.EnqueueRequestForOwner{
			IsController:   true,
			OwnerType: &qingtianv1.QingTianCluster{},
		},
	)
	if err != nil {
		return err
	}

	return nil
}

// Reconcile is called when a QingTianCluster CR is created/updated/deleted
func (r *QingTianClusterReconciler) Reconcile(ctx context.Context, req reconcile.Request) (reconcile.Result, error) {
	log := logf.FromContext(ctx)

	var cluster qingtianv1.QingTianCluster
	if err := r.Get(ctx, req.NamespacedName, &cluster); err != nil {
		if errors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			// Owned objects are automatically garbage collected. Return and don't requeue.
			return reconcile.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return reconcile.Result{}, err
	}

	// 1. Ensure Secret exists with all sensitive data
	if err := r.ensureSecret(ctx, &cluster); err != nil {
		log.Error(err, "Failed to ensure Secret exists")
		return reconcile.Result{}, err
	}

	// 2. Ensure ConfigMap exists with the apiserver configuration
	if err := r.ensureConfigMap(ctx, &cluster); err != nil {
		log.Error(err, "Failed to ensure ConfigMap exists")
		return reconcile.Result{}, err
	}

	// 3. If internal database is enabled, ensure MongoDB StatefulSet
	if cluster.Spec.Database.Internal {
		if err := r.ensureMongoDB(ctx, &cluster); err != nil {
			log.Error(err, "Failed to ensure MongoDB StatefulSet")
			return reconcile.Result{}, err
		}
	}

	// 4. If internal cache is enabled, ensure Redis Deployment
	if cluster.Spec.Cache.Internal {
		if err := r.ensureRedis(ctx, &cluster); err != nil {
			log.Error(err, "Failed to ensure Redis Deployment")
			return reconcile.Result{}, err
		}
	}

	// 5. Ensure apiserver Deployment
	if err := r.ensureApiserver(ctx, &cluster); err != nil {
		log.Error(err, "Failed to ensure apiserver Deployment")
		return reconcile.Result{}, err
	}

	// 6. Ensure apiserver Service
	if err := r.ensureApiserverService(ctx, &cluster); err != nil {
		log.Error(err, "Failed to ensure apiserver Service")
		return reconcile.Result{}, err
	}

	// 7. Ensure agent DaemonSet that runs on GPU nodes
	if err := r.ensureAgentDaemonSet(ctx, &cluster); err != nil {
		log.Error(err, "Failed to ensure agent DaemonSet")
		return reconcile.Result{}, err
	}

	// 8. If ingress enabled, ensure Ingress
	if cluster.Spec.Ingress.Enabled {
		if err := r.ensureIngress(ctx, &cluster); err != nil {
			log.Error(err, "Failed to ensure Ingress")
			return reconcile.Result{}, err
		}
	}

	log.Info("Successfully reconciled QingTianCluster", "name", cluster.Name)
	return reconcile.Result{RequeueAfter: 5 * 60}, nil
}

// getConfigMapName returns the name of the ConfigMap for a given cluster
func getConfigMapName(cluster *qingtianv1.QingTianCluster) string {
	return fmt.Sprintf("%s-config", cluster.Name)
}

// getSecretName returns the name of the Secret for a given cluster
func getSecretName(cluster *qingtianv1.QingTianCluster) string {
	return fmt.Sprintf("%s-secret", cluster.Name)
}

// getApiserverName returns the name of the apiserver Deployment
func getApiserverName(cluster *qingtianv1.QingTianCluster) string {
	return fmt.Sprintf("%s-apiserver", cluster.Name)
}

// getServiceName returns the name of the apiserver Service
func getServiceName(cluster *qingtianv1.QingTianCluster) string {
	return fmt.Sprintf("%s-apiserver", cluster.Name)
}

// getAgentName returns the name of the agent DaemonSet
func getAgentName(cluster *qingtianv1.QingTianCluster) string {
	return fmt.Sprintf("%s-agent", cluster.Name)
}

// getMongoDBName returns the name of the MongoDB StatefulSet
func getMongoDBName(cluster *qingtianv1.QingTianCluster) string {
	return fmt.Sprintf("%s-mongodb", cluster.Name)
}

// getRedisName returns the name of the redis Deployment
func getRedisName(cluster *qingtianv1.QingTianCluster) string {
	return fmt.Sprintf("%s-redis", cluster.Name)
}
