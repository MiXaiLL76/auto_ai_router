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

func TestManager_StaticDefaultParams(t *testing.T) {
	logger := testhelpers.NewTestLogger()
	creds := []config.CredentialConfig{
		{Name: "n13", Type: config.ProviderTypeVLLM, BaseURL: "http://n13", RPM: -1, TPM: -1},
		{Name: "dev", Type: config.ProviderTypeVLLM, BaseURL: "http://dev", RPM: -1, TPM: -1},
	}
	ocr := map[string]any{
		"skip_special_tokens": false,
		"vllm_xargs":          map[string]any{"ngram_size": 35, "window_size": 128},
	}

	m := New(logger, 100, []config.ModelRPMConfig{
		{Name: "unlimited-ocr", Credential: "n13", DefaultParams: ocr},
		// Without credential: applies on every credential serving the alias.
		{Name: "qwen", DefaultParams: map[string]any{"top_k": 20}},
		// Credential-specific entry is merged over the any-credential one.
		{Name: "qwen", Credential: "dev", DefaultParams: map[string]any{"vllm_xargs": map[string]any{"x": 1}}},
	})
	m.SetCredentials(creds)

	assert.Equal(t, ocr, m.GetDefaultParamsForCredential("unlimited-ocr", "n13"))
	assert.Nil(t, m.GetDefaultParamsForCredential("unlimited-ocr", "dev"), "bound to its credential")
	assert.Equal(t, map[string]any{"top_k": 20}, m.GetDefaultParamsForCredential("qwen", "n13"))
	assert.Equal(t, map[string]any{"top_k": 20, "vllm_xargs": map[string]any{"x": 1}},
		m.GetDefaultParamsForCredential("qwen", "dev"))

	t.Run("static defaults survive a DB sync", func(t *testing.T) {
		m.UpdateDBModels(nil, nil, creds)
		assert.Equal(t, ocr, m.GetDefaultParamsForCredential("unlimited-ocr", "n13"))
		assert.Equal(t, map[string]any{"top_k": 20}, m.GetDefaultParamsForCredential("qwen", "n13"))
	})

	t.Run("config wins over the database, objects are merged", func(t *testing.T) {
		m.UpdateDBModels([]config.ModelRPMConfig{{
			Name: "unlimited-ocr", Credential: "n13",
			DefaultParams: map[string]any{
				"temperature":         0.3,
				"skip_special_tokens": true,
				"vllm_xargs":          map[string]any{"window_size": 1024, "whitelist_token_ids": []any{1}},
			},
		}}, nil, creds)
		assert.Equal(t, map[string]any{
			"temperature":         0.3,
			"skip_special_tokens": false,
			"vllm_xargs":          map[string]any{"ngram_size": 35, "window_size": 128, "whitelist_token_ids": []any{1}},
		}, m.GetDefaultParamsForCredential("unlimited-ocr", "n13"))
		assert.Equal(t, map[string]any{"ngram_size": 35, "window_size": 128}, ocr["vllm_xargs"], "inputs are not modified")
	})
}

func TestMergeDefaultParams(t *testing.T) {
	base := map[string]any{"a": 1, "obj": map[string]any{"x": 1, "y": 2}, "list": []any{1}}
	over := map[string]any{"b": 2, "obj": map[string]any{"y": 20}, "list": []any{9, 9}}
	assert.Equal(t, map[string]any{
		"a": 1, "b": 2,
		"obj":  map[string]any{"x": 1, "y": 20},
		"list": []any{9, 9},
	}, mergeDefaultParams(base, over))
	assert.Equal(t, map[string]any{"x": 1, "y": 2}, base["obj"], "base is not modified")

	assert.Equal(t, map[string]any{"obj": "scalar"},
		mergeDefaultParams(map[string]any{"obj": map[string]any{"x": 1}}, map[string]any{"obj": "scalar"}))
}
