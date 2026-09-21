package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func vllmTestConfig(cred CredentialConfig) *Config {
	return &Config{
		Server: ServerConfig{
			Port:           8080,
			MaxBodySizeMB:  10,
			MasterKey:      "test-key",
			RequestTimeout: 30 * time.Second,
		},
		Credentials: []CredentialConfig{cred},
		Fail2Ban:    Fail2BanConfig{MaxAttempts: 3},
	}
}

func TestVLLMProviderType(t *testing.T) {
	assert.True(t, ProviderTypeVLLM.IsValid())
	assert.Equal(t, ProviderType("vllm"), ProviderTypeVLLM)

	// vLLM keeps its own identity for accounting but speaks the OpenAI wire protocol.
	cred := CredentialConfig{Type: ProviderTypeVLLM}
	assert.Equal(t, ProviderTypeOpenAI, cred.EffectiveProviderType())
	assert.False(t, cred.IsProxyLike())
}

func TestNormalizeProviderType_HostedVLLM(t *testing.T) {
	for _, raw := range []string{"hosted_vllm", "Hosted-VLLM", " hosted_vllm ", "vllm", "VLLM"} {
		assert.Equal(t, ProviderTypeVLLM, normalizeProviderType(raw), raw)
	}
	assert.Equal(t, ProviderTypeOpenAI, normalizeProviderType("openai"))
}

func TestCredentialConfig_UnmarshalVLLMFromYAML(t *testing.T) {
	var cred CredentialConfig
	require.NoError(t, yaml.Unmarshal([]byte("name: ray\ntype: hosted_vllm\nbase_url: http://vllm:8000\nrpm: -1\n"), &cred))
	assert.Equal(t, ProviderTypeVLLM, cred.Type)
}

func TestConfig_Validate_VLLM(t *testing.T) {
	t.Run("api key is optional", func(t *testing.T) {
		cfg := vllmTestConfig(CredentialConfig{Name: "ray", Type: ProviderTypeVLLM, BaseURL: "http://vllm:8000", RPM: -1, TPM: -1})
		require.NoError(t, cfg.Validate())
	})
	t.Run("base url is required", func(t *testing.T) {
		cfg := vllmTestConfig(CredentialConfig{Name: "ray", Type: ProviderTypeVLLM, RPM: -1, TPM: -1})
		err := cfg.Validate()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "base_url is required for vllm")
	})
	t.Run("base url must be a valid url", func(t *testing.T) {
		cfg := vllmTestConfig(CredentialConfig{Name: "ray", Type: ProviderTypeVLLM, BaseURL: "not a url", RPM: -1, TPM: -1})
		require.Error(t, cfg.Validate())
	})
}

func TestIsVLLMProviderName(t *testing.T) {
	for _, name := range []string{"vllm", "hosted_vllm", "hosted-vllm", "VLLM", " Hosted_VLLM "} {
		assert.True(t, IsVLLMProviderName(name), name)
	}
	for _, name := range []string{"", "openai", "vllm-cluster", "hosted"} {
		assert.False(t, IsVLLMProviderName(name), name)
	}
	assert.Equal(t, "hosted_vllm", VLLMLiteLLMProvider)
}

func TestTrimVLLMProviderPrefix(t *testing.T) {
	assert.Equal(t, "m", TrimVLLMProviderPrefix("hosted_vllm/m"))
	assert.Equal(t, "m", TrimVLLMProviderPrefix("vllm/m"))
	assert.Equal(t, "m", TrimVLLMProviderPrefix("hosted-vllm/m"))
	assert.Equal(t, "m", TrimVLLMProviderPrefix("m"))
	assert.Equal(t, "Qwen/Qwen3-8B", TrimVLLMProviderPrefix("Qwen/Qwen3-8B"))
	assert.Equal(t, "hosted_vllm", TrimVLLMProviderPrefix("hosted_vllm"))
	assert.Equal(t, "openai/gpt", TrimVLLMProviderPrefix("openai/gpt"))
}
