// Package testhelpers provides shared fixtures for tests.
package testhelpers

import (
	"github.com/mixaill76/auto_ai_router/internal/config"
)

// NewTestMonitoringConfig creates a test monitoring configuration.
// Commonly used in router and integration tests.
func NewTestMonitoringConfig(healthPath string, logErrors bool, errorsLogPath string) *config.MonitoringConfig {
	return &config.MonitoringConfig{
		PrometheusEnabled: false,
		HealthCheckPath:   healthPath,
		LogErrors:         logErrors,
		ErrorsLogPath:     errorsLogPath,
	}
}
