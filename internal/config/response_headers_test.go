package config

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestCredentialResponseHeaders(t *testing.T) {
	var cred CredentialConfig
	t.Setenv("LABEL", "private")
	require.NoError(t, yaml.Unmarshal([]byte("name: west\ntype: openai\nresponse_headers:\n  X-Pool: os.environ/LABEL\n  X-Route: '${credential.name}'\n"), &cred))
	require.Equal(t, map[string]string{"X-Pool": "os.environ/LABEL", "X-Route": "${credential.name}"}, cred.ResponseHeaders)
	for _, fields := range []string{"Bad Name: x", "X-Pool: \"a\\nb\"", "X-Test: a\n  x-test: b", "Content-Length: 4", "tRaNsFeR-EnCoDiNg: test", "X-Total-Cost: 0", "X-Credential-Name: private"} {
		require.Error(t, yaml.Unmarshal([]byte("name: west\ntype: openai\nresponse_headers:\n  "+fields+"\n"), &cred))
	}
	for _, provider := range []ProviderType{ProviderTypeAIR, ProviderTypeProxy} {
		require.Error(t, validateCredentialResponseHeaders("west", provider, map[string]string{"X-Pool": "a"}))
	}
}
