package testhelpers

import (
	"github.com/mixaill76/auto_ai_router/internal/config"
)

// NewTestModelManager creates a test model manager instance.
// Useful for tests that need model manager but don't need specific configuration.
func NewTestModelManagerConfig() []config.ModelRPMConfig {
	return []config.ModelRPMConfig{}
}

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

// NewTestCredentialConfig creates a test credential configuration.
// Useful for building test credentials for proxy/balancer testing.
func NewTestCredentialConfig(name string, credType config.ProviderType, baseURL, apiKey string) config.CredentialConfig {
	return config.CredentialConfig{
		Name:    name,
		Type:    credType,
		BaseURL: baseURL,
		APIKey:  apiKey,
		RPM:     100,
		TPM:     10000,
	}
}
