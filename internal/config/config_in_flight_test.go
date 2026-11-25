package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestServerMaxInFlightRequests(t *testing.T) {
	var cfg ServerConfig
	require.NoError(t, yaml.Unmarshal([]byte("port: 8080\n"), &cfg))
	assert.Zero(t, cfg.MaxInFlightRequests, "no cap by default")
	require.NoError(t, yaml.Unmarshal([]byte("max_in_flight_requests: 5000\n"), &cfg))
	assert.Equal(t, 5000, cfg.MaxInFlightRequests)
	assert.Error(t, yaml.Unmarshal([]byte("max_in_flight_requests: many\n"), &cfg))

	c := baseValidConfigForKafkaTests()
	c.Server.MaxInFlightRequests = 5000
	require.NoError(t, c.Validate())
	c.Server.MaxInFlightRequests = -1
	err := c.Validate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_in_flight_requests")
}
