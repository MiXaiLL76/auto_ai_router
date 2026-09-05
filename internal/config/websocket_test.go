package config

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
	"testing"
)

func TestModelWebSocketResponsesConfig(t *testing.T) {
	t.Setenv("TEST_NATIVE_WS", "true")
	for _, tt := range []struct {
		yaml string
		want bool
	}{
		{"name: gpt-6-astra", false},
		{"name: gpt-6-astra\nwebsocket_responses: false", false},
		{"name: gpt-6-astra\nwebsocket_responses: true", true},
		{"name: gpt-6-astra\nwebsocket_responses: os.environ/TEST_NATIVE_WS", true},
	} {
		var cfg ModelRPMConfig
		require.NoError(t, yaml.Unmarshal([]byte(tt.yaml), &cfg))
		assert.Equal(t, tt.want, cfg.WebSocketResponses)
	}
	var cfg ModelRPMConfig
	require.Error(t, yaml.Unmarshal([]byte("name: gpt-6-astra\nwebsocket_responses: invalid"), &cfg))
}
