package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestKeyMetricsConfig_DefaultsWhenAbsent(t *testing.T) {
	var m MonitoringConfig
	require.NoError(t, yaml.Unmarshal([]byte("prometheus_enabled: true\n"), &m))
	assert.False(t, m.KeyMetrics.Enabled)
	assert.Nil(t, m.KeyMetrics.InfoLabels)
	assert.Equal(t, 5000, m.KeyMetrics.MaxKeys)
	assert.Equal(t, 24*time.Hour, m.KeyMetrics.IdleTTL)

	assert.Equal(t, defaultKeyMetricsConfig(), defaultMonitoringConfig().KeyMetrics)
}

func TestKeyMetricsConfig_Parse(t *testing.T) {
	t.Setenv("AIR_KEY_METRICS", "true")
	var m MonitoringConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
prometheus_enabled: true
key_metrics:
  enabled: os.environ/AIR_KEY_METRICS
  info_labels: [team_alias, " user_id ", team_alias]
  max_keys: 0
  idle_ttl: 0s
`), &m))
	assert.True(t, m.KeyMetrics.Enabled)
	assert.Equal(t, []string{"team_alias", "user_id"}, m.KeyMetrics.InfoLabels)
	assert.Equal(t, 0, m.KeyMetrics.MaxKeys)
	assert.Equal(t, time.Duration(0), m.KeyMetrics.IdleTTL)
}

func TestKeyMetricsConfig_Invalid(t *testing.T) {
	for name, body := range map[string]string{
		"unknown label":    "info_labels: [key_hash]",
		"negative max":     "max_keys: -1",
		"negative ttl":     "idle_ttl: -1m",
		"bad duration":     "idle_ttl: soon",
		"bad enabled bool": "enabled: maybe",
	} {
		t.Run(name, func(t *testing.T) {
			var m MonitoringConfig
			err := yaml.Unmarshal([]byte("key_metrics:\n  "+body+"\n"), &m)
			assert.Error(t, err)
		})
	}
}

func TestConfig_KeyMetricsEnabled(t *testing.T) {
	cfg := &Config{}
	cfg.Monitoring.KeyMetrics.Enabled = true
	cfg.Monitoring.PrometheusEnabled = true
	assert.False(t, cfg.KeyMetricsEnabled(), "requires litellm_db")

	cfg.LiteLLMDB.Enabled = true
	assert.True(t, cfg.KeyMetricsEnabled())

	cfg.Monitoring.PrometheusEnabled = false
	assert.False(t, cfg.KeyMetricsEnabled(), "requires a metrics sink")
	cfg.OTEL.Enabled = true
	assert.True(t, cfg.KeyMetricsEnabled())

	cfg.Monitoring.KeyMetrics.Enabled = false
	assert.False(t, cfg.KeyMetricsEnabled())
}
