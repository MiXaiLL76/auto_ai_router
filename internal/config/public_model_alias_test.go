package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadPublicModelAliasesSeparatelyFromProviderAliases(t *testing.T) {
	t.Setenv("CANONICAL_PUBLIC_MODEL", "openai/gpt-4.1")
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(`
server:
  port: 8080
  master_key: sk-test
credentials:
  - name: provider
    type: openai
    api_key: sk-provider
    base_url: https://provider.invalid
monitoring: {}
model_alias:
  openai/gpt-4.1: gpt-4.1
client_model_ids:
  - os.environ/CANONICAL_PUBLIC_MODEL
public_model_alias:
  gpt-4.1: os.environ/CANONICAL_PUBLIC_MODEL
`), 0o600))

	cfg, err := Load(path)
	require.NoError(t, err)
	assert.Equal(t, map[string]string{"openai/gpt-4.1": "gpt-4.1"}, cfg.ModelAlias)
	assert.Equal(t, []string{"openai/gpt-4.1"}, cfg.ClientModelIDs)
	assert.Equal(t, map[string]string{"gpt-4.1": "openai/gpt-4.1"}, cfg.PublicModelAlias)
}

func TestClientModelIDsFailClosedOnDuplicateAndOutOfSurfaceAlias(t *testing.T) {
	tests := []struct {
		name      string
		boundary  string
		aliases   string
		wantError string
	}{
		{
			name:      "duplicate canonical ID",
			boundary:  "  - openai/gpt-4.1\n  - openai/gpt-4.1",
			wantError: `duplicate model ID "openai/gpt-4.1"`,
		},
		{
			name:     "public alias target outside boundary",
			boundary: "  - openai/gpt-4.1",
			aliases: `public_model_alias:
  gpt-5: openai/gpt-5`,
			wantError: `public_model_alias "gpt-5" targets "openai/gpt-5" outside client_model_ids`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			source := `
server:
  port: 8080
  master_key: sk-test
credentials:
  - name: provider
    type: openai
    api_key: sk-provider
    base_url: https://provider.invalid
monitoring: {}
client_model_ids:
` + tt.boundary + "\n" + tt.aliases + "\n"
			require.NoError(t, os.WriteFile(path, []byte(source), 0o600))
			_, err := Load(path)
			require.ErrorContains(t, err, tt.wantError)
		})
	}
}
