package config

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestLiteLLMDBLogCredentialName(t *testing.T) {
	t.Setenv("LOG_CREDENTIAL_NAME", "true")
	for _, tc := range []struct {
		name string
		yaml string
		want bool
	}{
		{name: "omitted", yaml: "enabled: true"},
		{name: "disabled", yaml: "log_credential_name: false"},
		{name: "enabled", yaml: "log_credential_name: true", want: true},
		{name: "environment", yaml: "log_credential_name: os.environ/LOG_CREDENTIAL_NAME", want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var cfg LiteLLMDBConfig
			require.NoError(t, yaml.Unmarshal([]byte(tc.yaml), &cfg))
			assert.Equal(t, tc.want, cfg.LogCredentialName)
			encoded, err := yaml.Marshal(cfg)
			require.NoError(t, err)
			var restored LiteLLMDBConfig
			require.NoError(t, yaml.Unmarshal(encoded, &restored))
			assert.Equal(t, tc.want, restored.LogCredentialName)
		})
	}

	var cfg LiteLLMDBConfig
	err := yaml.Unmarshal([]byte("log_credential_name: invalid"), &cfg)
	require.ErrorContains(t, err, "litellm_db.log_credential_name")
}
