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
	assert.True(t, cfg.IncludeTeamSpendInUserSpend)
}

func TestLiteLLMDBTeamSpendProjectionParsesExplicitSetting(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("include_team_spend_in_user_spend: false\n"), &cfg))
	assert.False(t, cfg.IncludeTeamSpendInUserSpend)
}

func TestServerCredentialNameAsTeamIDParsesExplicitSetting(t *testing.T) {
	var cfg ServerConfig
	require.NoError(t, yaml.Unmarshal([]byte("credential_name_as_team_id: true\n"), &cfg))
	assert.True(t, cfg.CredentialNameAsTeamID)
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

func TestLiteLLMDBEnableCostMarginParsesExplicitSetting(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("enabled: false\n"), &cfg))
	assert.False(t, cfg.EnableCostMargin)

	require.NoError(t, yaml.Unmarshal([]byte("enable_cost_margin: true\n"), &cfg))
	assert.True(t, cfg.EnableCostMargin)
}

func TestLiteLLMDBBudgetTimezoneParsesExplicitSetting(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("enabled: false\n"), &cfg))
	assert.Equal(t, time.UTC, cfg.BudgetTimezone)

	require.NoError(t, yaml.Unmarshal([]byte("budget_timezone: Europe/Moscow\n"), &cfg))
	assert.Equal(t, "Europe/Moscow", cfg.BudgetTimezone.String())

	assert.Error(t, yaml.Unmarshal([]byte("budget_timezone: Mars/Olympus\n"), &cfg))
}

func TestLiteLLMDBBudgetResetTimeParsesLiteLLMFormats(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("enabled: false\n"), &cfg))
	assert.Zero(t, cfg.BudgetResetTime)

	require.NoError(t, yaml.Unmarshal([]byte(`budget_reset_time: "12:30"`), &cfg))
	assert.Equal(t, 12*time.Hour+30*time.Minute, cfg.BudgetResetTime)
	require.NoError(t, yaml.Unmarshal([]byte(`budget_reset_time: "07:00:15"`), &cfg))
	assert.Equal(t, 7*time.Hour+15*time.Second, cfg.BudgetResetTime)

	for _, value := range []string{`"24:00"`, `"noon"`, `"12"`} {
		assert.Error(t, yaml.Unmarshal([]byte("budget_reset_time: "+value), &cfg), value)
	}
}
