package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestLiteLLMDBLogQueueFullTimeout(t *testing.T) {
	var cfg LiteLLMDBConfig
	require.NoError(t, yaml.Unmarshal([]byte("enabled: false\n"), &cfg))
	assert.Equal(t, 5*time.Second, cfg.LogQueueFullTimeout, "today's wait by default")

	require.NoError(t, yaml.Unmarshal([]byte("log_queue_full_timeout: 0s\n"), &cfg))
	assert.Zero(t, cfg.LogQueueFullTimeout)
	require.NoError(t, yaml.Unmarshal([]byte("log_queue_full_timeout: 250ms\n"), &cfg))
	assert.Equal(t, 250*time.Millisecond, cfg.LogQueueFullTimeout)
	assert.Error(t, yaml.Unmarshal([]byte("log_queue_full_timeout: soon\n"), &cfg))
}

func TestLiteLLMDBLogQueueFullTimeoutMustNotBeNegative(t *testing.T) {
	c := baseValidConfigForKafkaTests()
	c.LiteLLMDB.Enabled = true
	c.LiteLLMDB.DatabaseURL = "postgresql://localhost/test"
	c.LiteLLMDB.LogWorkers = 4
	require.NoError(t, c.Validate(), "0 is valid")
	c.LiteLLMDB.LogQueueFullTimeout = -time.Second
	err := c.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "litellm_db.log_queue_full_timeout must not be negative")
}
