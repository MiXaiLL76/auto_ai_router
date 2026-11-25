package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestLiteLLMDBSpendUpdatesSingleRoundTripIsOptIn(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("enabled: false\n"), &cfg))
	assert.False(t, cfg.SpendUpdatesSingleRoundTrip)

	require.NoError(t, yaml.Unmarshal([]byte("spend_updates_single_round_trip: true\n"), &cfg))
	assert.True(t, cfg.SpendUpdatesSingleRoundTrip)
	assert.Error(t, yaml.Unmarshal([]byte("spend_updates_single_round_trip: perhaps\n"), &cfg))
}
