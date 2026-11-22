package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestConfigValidateFirstErrorAndDefaults(t *testing.T) {
	cfg := Config{
		Server: ServerConfig{
			MaxBodySizeMB:  10,
			RequestTimeout: time.Second,
		},
		Fail2Ban: Fail2BanConfig{MaxAttempts: 3},
		Credentials: []CredentialConfig{
			{Name: "fallback", Type: ProviderTypeVLLM, RPM: 10, IsFallback: true},
		},
		Models: []ModelRPMConfig{{Name: "test", Weight: -1}},
		Kafka:  KafkaConfig{Enabled: true},
	}

	require.EqualError(t, cfg.Validate(), "invalid port: 0")
	require.Zero(t, cfg.Server.ResponseBodyMultiplier)

	cfg.Server.Port = 8080
	require.EqualError(t, cfg.Validate(), "master_key is required")
	require.Equal(t, 10, cfg.Server.ResponseBodyMultiplier)
	require.Zero(t, cfg.Server.ReadTimeout)

	cfg.Server.MasterKey = "test-key"
	require.EqualError(t, cfg.Validate(), "credential fallback: base_url is required for vllm type")
	require.Equal(t, cfg.Server.RequestTimeout, cfg.Server.ReadTimeout)
	require.Zero(t, cfg.Credentials[0].Priority)

	cfg.Credentials[0].BaseURL = "https://example.com"
	require.EqualError(t, cfg.Validate(), "model test: invalid weight: -1 (must be 0 for default or positive number)")
	require.Equal(t, FallbackPriorityGroup, cfg.Credentials[0].Priority)

	cfg.Models[0].Weight = 0
	require.EqualError(t, cfg.Validate(), "kafka.brokers is required when kafka is enabled")
}

func TestConfigValidateFail2BanDuplicateErrorCodes(t *testing.T) {
	cfg := Config{
		Server: ServerConfig{Port: 8080, MaxBodySizeMB: 10, MasterKey: "test-key"},
		Credentials: []CredentialConfig{
			{Name: "provider", Type: ProviderTypeVLLM, BaseURL: "https://example.com", RPM: 10},
		},
		Fail2Ban: Fail2BanConfig{
			MaxAttempts:    3,
			ErrorCodeRules: []ErrorCodeRuleConfig{{Code: 429}, {Code: 500}, {Code: 429}, {Code: 500}},
			CredentialOverrides: map[string]ErrorCodeOverrideConfig{
				"provider": {ErrorCodeRules: []ErrorCodeRuleConfig{{Code: 429}, {Code: 429}}},
			},
		},
	}

	require.EqualError(t, cfg.Validate(), "fail2ban: duplicate error_code_rules for code 429")
	cfg.Fail2Ban.ErrorCodeRules = cfg.Fail2Ban.ErrorCodeRules[:2]
	require.EqualError(t, cfg.Validate(), "fail2ban: duplicate error_code_rules for code 429 in credential_overrides[provider]")
	cfg.Fail2Ban.CredentialOverrides["provider"] = ErrorCodeOverrideConfig{
		ErrorCodeRules: []ErrorCodeRuleConfig{{Code: 429}},
	}
	require.NoError(t, cfg.Validate())
}
