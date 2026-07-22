package autoscaling

import (
	"context"
	"fmt"
	"time"
)

// MetricPoint represents a single metric data point.
type MetricPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
}

// Prediction represents a predicted value.
type Prediction struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Lower     float64   `json:"lower"`
	Upper     float64   `json:"upper"`
}

// TimeSeriesModel defines the interface for time series forecasting.
type TimeSeriesModel interface {
	// Forecast predicts future values given historical data.
	Forecast(ctx context.Context, history []MetricPoint, horizon int) ([]Prediction, error)
}

// PredictiveScaler uses time series forecasting to predict future load and scale ahead of time.
type PredictiveScaler struct {
	model  TimeSeriesModel
	config PredictiveConfig
}

// PredictiveConfig configures predictive scaling.
type PredictiveConfig struct {
	// Enabled enables predictive scaling.
	Enabled bool
	// HorizonMinutes is how far ahead to predict.
	HorizonMinutes int
	// TrainingDays is how many days of historical data to use for training.
	TrainingDays int
	// MinNodes minimum nodes.
	MinNodes int
	// MaxNodes maximum nodes.
	MaxNodes int
}

// NewPredictiveScaler creates a new PredictiveScaler.
func NewPredictiveScaler(model TimeSeriesModel, config PredictiveConfig) *PredictiveScaler {
	return &PredictiveScaler{
		model:  model,
		config: config,
	}
}

// Predict predicts future queue length and returns suggested node count.
func (s *PredictiveScaler) Predict(ctx context.Context, history []MetricPoint) ([]Prediction, error) {
	if !s.config.Enabled {
		return nil, fmt.Errorf("predictive scaling is disabled")
	}

	// Filter history to only include training days
	cutoff := time.Now().AddDate(0, 0, -s.config.TrainingDays)
	filtered := make([]MetricPoint, 0, len(history))
	for _, p := range history {
		if p.Timestamp.After(cutoff) {
			filtered = append(filtered, p)
		}
	}

	// Predict the next horizon minutes
	predictions, err := s.model.Forecast(ctx, filtered, s.config.HorizonMinutes)
	if err != nil {
		return nil, fmt.Errorf("forecast failed: %w", err)
	}

	return predictions, nil
}

// SuggestNodes suggests the number of nodes based on predicted queue length.
func (s *PredictiveScaler) SuggestNodes(predictions []Prediction, queueLengthTarget int) int {
	// Find the maximum predicted queue length in the horizon
	maxQueue := 0.0
	for _, p := range predictions {
		if p.Value > maxQueue {
			maxQueue = p.Value
		}
	}

	// Simple linear scaling: nodes = ceil(max_queue / queueLengthTarget)
	suggested := int((maxQueue + float64(queueLengthTarget) - 1) / float64(queueLengthTarget))

	// Clamp to min/max
	if suggested < s.config.MinNodes {
		suggested = s.config.MinNodes
	}
	if suggested > s.config.MaxNodes {
		suggested = s.config.MaxNodes
	}

	return suggested
}
