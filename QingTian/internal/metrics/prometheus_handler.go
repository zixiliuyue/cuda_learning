// Package metrics implements Prometheus metrics exposure.
//
// This provides an HTTP handler for Prometheus to scrape metrics.
package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Handler returns the Prometheus metrics handler.
func Handler() http.Handler {
	return promhttp.Handler()
}

// InstrumentHandler adds instrumentation around an HTTP handler.
func InstrumentHandler(name string, handler http.Handler) http.Handler {
	// TODO: Add request latency metrics if needed
	return handler
}
