// Package main is the entry point for the QingTian Kubernetes Operator.
//
// The Operator runs inside Kubernetes as a Deployment that watches QingTianCluster
// custom resources and manages the corresponding infrastructure.
package main

import (
	"flag"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	ctrl "sigs.k8s.io/controller-runtime"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	qingtianv1 "qingtian/internal/operator/apis/qingtian/v1"
	"qingtian/internal/operator/controller"
)

func main() {
	var metricsAddr string
	var enableLeaderElection bool
	var probeAddr string
	var qingtianEndpoint string
	var qingtianToken string
	flag.StringVar(&metricsAddr, "metrics-addr", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-addr", "", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "enable-leader-election", false,
		"Enable leader election for controller manager. Enabling this will ensure there is only one active controller manager.")
	flag.StringVar(&qingtianEndpoint, "qingtian-endpoint", "http://qingtian-apiserver:8080", "QingTian API server endpoint")
	flag.StringVar(&qingtianToken, "qingtian-token", "", "Authentication token for QingTian API")

	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:             runtime.NewScheme(),
		Metrics:                metricsserver.Options{
			BindAddress: metricsAddr,
		},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "bfd46a65.qingtian.ai",
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	// Add our types to the scheme
	if err := qingtianv1.AddToScheme(mgr.GetScheme()); err != nil {
		setupLog.Error(err, "unable to add QingTian API to scheme")
		os.Exit(1)
	}

	// Setup the QingTianCluster controller
	if err = (&controller.QingTianClusterReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "QingTianCluster")
		os.Exit(1)
	}

	// Setup the TrainingJob controller
	if err = (&controller.TrainingJobReconciler{
		Client:           mgr.GetClient(),
		Scheme:           mgr.GetScheme(),
		QingTianEndpoint: qingtianEndpoint,
		QingTianToken:    qingtianToken,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "TrainingJob")
		os.Exit(1)
	}

	// +kubebuilder:scaffold:builder

	// Add health and ready checks
	if err := mgr.AddHealthzCheck("/healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("/readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

var setupLog = ctrl.Log.WithName("setup")
