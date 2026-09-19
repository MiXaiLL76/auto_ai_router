package models

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
)

func TestManager_DefaultParamsPerCredentialAndAlias(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	creds := []config.CredentialConfig{
		{Name: "ray", Type: config.ProviderTypeVLLM, BaseURL: "http://ray", RPM: -1, TPM: -1},
		{Name: "h200", Type: config.ProviderTypeVLLM, BaseURL: "http://h200", RPM: -1, TPM: -1},
	}
	flash := map[string]any{"temperature": 0.7, "chat_template_kwargs": map[string]any{"enable_thinking": false}}

	m := New(logger, 100, nil)
	m.SetCredentials(creds)
	m.UpdateDBModels([]config.ModelRPMConfig{
		{Name: "qwen-flash", Model: "qwen-36-35b-fp8", Credential: "ray", DefaultParams: flash},
		{Name: "kimi-k26", Credential: "h200"},
		// Same alias on another credential keeps its own defaults.
		{Name: "qwen-flash", Model: "qwen-36-35b-fp8", Credential: "h200", DefaultParams: map[string]any{"top_k": 5.0}},
	}, nil, creds)

	assert.Equal(t, flash, m.GetDefaultParamsForCredential("qwen-flash", "ray"))
	assert.Equal(t, map[string]any{"top_k": 5.0}, m.GetDefaultParamsForCredential("qwen-flash", "h200"))
	assert.Nil(t, m.GetDefaultParamsForCredential("kimi-k26", "h200"), "a deployment without defaults has none")
	assert.Nil(t, m.GetDefaultParamsForCredential("qwen-flash", "unknown"))
	assert.Nil(t, m.GetDefaultParamsForCredential("unknown", "ray"))

	// The next sync replaces the whole DB-sourced set: defaults that vanished from the
	// database must vanish here too.
	m.UpdateDBModels([]config.ModelRPMConfig{
		{Name: "qwen-flash", Model: "qwen-36-35b-fp8", Credential: "ray"},
	}, nil, creds)
	assert.Nil(t, m.GetDefaultParamsForCredential("qwen-flash", "ray"))
	assert.Nil(t, m.GetDefaultParamsForCredential("qwen-flash", "h200"))
}

func TestManager_VLLMUsesLiteLLMHostedPrefix(t *testing.T) {
	assert.Equal(t, "hosted_vllm", providerTypeLiteLLMPrefix[config.ProviderTypeVLLM])
}
