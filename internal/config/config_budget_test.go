package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestLiteLLMDBEnforcementDefaultsToOptIn(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("{}"), &cfg))

	assert.False(t, cfg.EnforceBudgetReservation)
	assert.False(t, cfg.EnforceKeyRateLimits)
}

func TestLiteLLMDBEnforcementCanBeEnabledExplicitly(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
enforce_budget_reservation: true
enforce_key_rate_limits: true
`), &cfg))

	assert.True(t, cfg.EnforceBudgetReservation)
	assert.True(t, cfg.EnforceKeyRateLimits)
}
