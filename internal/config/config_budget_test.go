package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestLiteLLMDBBudgetEnforcementIsOptIn(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("enabled: false\n"), &cfg))
	assert.False(t, cfg.EnforceBudgetReservation)
	assert.False(t, cfg.EnforceKeyRateLimits)
	assert.Equal(t, 15*time.Minute, cfg.BudgetReservationTTL)
	assert.Equal(t, 1000, cfg.DefaultEstimatedCompletionTokens)
}

func TestLiteLLMDBBudgetEnforcementParsesExplicitSettings(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
enabled: false
enforce_budget_reservation: true
budget_reservation_ttl: 30m
enforce_key_rate_limits: true
default_estimated_completion_tokens: 2048
`), &cfg))
	assert.True(t, cfg.EnforceBudgetReservation)
	assert.True(t, cfg.EnforceKeyRateLimits)
	assert.Equal(t, 30*time.Minute, cfg.BudgetReservationTTL)
	assert.Equal(t, 2048, cfg.DefaultEstimatedCompletionTokens)
}
