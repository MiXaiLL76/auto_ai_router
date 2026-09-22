package modeltable

import (
	"encoding/json"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	cryptoutils "github.com/mixaill76/auto_ai_router/internal/litellmdb/crypto_utils"
	"github.com/mixaill76/auto_ai_router/internal/litellmdb/queries"
	"github.com/mixaill76/auto_ai_router/internal/testhelpers"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The fixtures below reproduce the shape of a real LiteLLM database that fronts a vLLM
// fleet: every deployment is hosted_vllm, secrets and the credential binding are stored
// encrypted with the master key, several deployments share one model name, one
// deployment sits behind a provider prefix, and router_settings carries the aliases
// clients actually call.

const fixtureSigningKey = "fixture-master-key"

func encrypted(t *testing.T, value string) string {
	t.Helper()
	out, err := cryptoutils.EncryptValueHelper(value, fixtureSigningKey)
	require.NoError(t, err)
	return out
}

// modelRow builds a LiteLLM_ProxyModelTable row through JSON, exactly as it is read from
// the database.
func modelRow(t *testing.T, id, name string, params, info map[string]any, blocked bool) queries.ModelTable {
	t.Helper()
	raw, err := json.Marshal(map[string]any{
		"model_id": id, "model_name": name, "litellm_params": params, "model_info": info,
	})
	require.NoError(t, err)
	var row queries.ModelTable
	require.NoError(t, json.Unmarshal(raw, &row))
	row.Blocked = blocked
	return row
}

// vllmParams are the litellm_params of a hosted_vllm deployment bound to a named
// credential, with the encrypted fields LiteLLM encrypts.
func vllmParams(t *testing.T, model, credential string, extra map[string]any) map[string]any {
	t.Helper()
	params := map[string]any{
		"model":                   encrypted(t, model),
		"custom_llm_provider":     encrypted(t, "hosted_vllm"),
		"litellm_credential_name": encrypted(t, credential),
		"use_in_pass_through":     false,
		"use_litellm_proxy":       false,
	}
	for k, v := range extra {
		params[k] = v
	}
	return params
}

func chatInfo() map[string]any { return map[string]any{"mode": "chat", "db_model": false} }

func credentialRow(t *testing.T, name, provider, apiBase string) queries.CredentialTable {
	t.Helper()
	values := map[string]any{"api_base": encrypted(t, apiBase)}
	info := map[string]any{}
	if provider != "" {
		info["custom_llm_provider"] = provider
	}
	raw, err := json.Marshal(map[string]any{
		"credential_id": "id-" + name, "credential_name": name, "credential_values": values, "credential_info": info,
	})
	require.NoError(t, err)
	var row queries.CredentialTable
	require.NoError(t, json.Unmarshal(raw, &row))
	return row
}

func fixtureRouterSettings() queries.RouterSettings {
	return queries.RouterSettings{
		ModelGroupAlias: map[string]string{
			"gpt-oss":        "gpt-oss-120b",
			"qwen-flash":     "qwen-36-35b-fast",
			"qwen-ultra":     "qwen3.5-397b-a17b-fp8",
			"coder-ultra":    "kimi-k26",
			"qwen-36-35b":    "qwen-36-35b-fp8",
			"qwen3-coder":    "qwen-ultra-flash",
			"gemma-3-27b-it": "qwen-36-35b-fp8", // clashes with a real group of the same name
			"ghost":          "does-not-exist",
			"self":           "self",
		},
		Fallbacks: map[string][]string{"coder-ultra": {"qwen-ultra"}},
	}
}

func fixtureCredentials(t *testing.T) []queries.CredentialTable {
	t.Helper()
	return []queries.CredentialTable{
		credentialRow(t, "ray-service-prod", "hosted_vllm", "http://ray:8000/v1"),
		credentialRow(t, "vllm-only-h200x8-deployment-nodeport", "hosted_vllm", "http://h200:8000/v1"),
		credentialRow(t, "rgm-s-dsapp01-frida-1", "hosted_vllm", "http://frida-1:8000/v1"),
		credentialRow(t, "rgm-s-dsapp01-frida-2", "hosted_vllm", "http://frida-2:8000/v1"),
		credentialRow(t, "rgm-s-dsapp01-rerank-1", "hosted_vllm", "http://rerank:8000/v1"),
		credentialRow(t, "openai-cred", "openai", "https://api.openai.example/v1"),
		credentialRow(t, "no-provider-cred", "", "http://noinfo:8000/v1"),
		credentialRow(t, "plain-cred", "hosted_vllm", "http://plain:8000/v1"),
	}
}

func fixtureModels(t *testing.T) []queries.ModelTable {
	t.Helper()
	priced := func(in, out float64) map[string]any {
		return map[string]any{"input_cost_per_token": in, "output_cost_per_token": out}
	}
	merge := func(base map[string]any, more map[string]any) map[string]any {
		for k, v := range more {
			base[k] = v
		}
		return base
	}
	embedInfo := map[string]any{"mode": "embedding"}

	return []queries.ModelTable{
		modelRow(t, "id-gpt-oss", "gpt-oss-120b", vllmParams(t, "gpt-oss-120b", "ray-service-prod", priced(6e-8, 3e-7)), chatInfo(), false),
		modelRow(t, "id-qwen-ultra", "qwen3.5-397b-a17b-fp8", vllmParams(t, "qwen397b-int4", "ray-service-prod", priced(4e-7, 2.4e-6)), chatInfo(), false),
		modelRow(t, "id-ultra-flash", "qwen-ultra-flash", vllmParams(t, "qwen397b-int4", "ray-service-prod",
			map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}), chatInfo(), false),
		modelRow(t, "id-fp8", "qwen-36-35b-fp8", vllmParams(t, "qwen-36-35b-fp8", "ray-service-prod",
			merge(priced(2e-7, 1e-6), map[string]any{"presence_penalty": 1.5})), chatInfo(), false),
		modelRow(t, "id-fast", "qwen-36-35b-fast", vllmParams(t, "hosted_vllm/qwen-36-35b-fp8", "ray-service-prod",
			merge(priced(2e-7, 1e-6), map[string]any{
				"chat_template_kwargs": map[string]any{"enable_thinking": false},
				"temperature":          0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
				"presence_penalty": 1.5, "repetition_penalty": 1,
			})), chatInfo(), false),
		modelRow(t, "id-kimi", "kimi-k26", vllmParams(t, "kimi-k26", "vllm-only-h200x8-deployment-nodeport", priced(1e-6, 4e-6)), chatInfo(), false),
		modelRow(t, "id-frida-1", "frida", vllmParams(t, "frida", "rgm-s-dsapp01-frida-1", map[string]any{"input_cost_per_token": 1e-8}), embedInfo, false),
		modelRow(t, "id-frida-2", "frida", vllmParams(t, "frida", "rgm-s-dsapp01-frida-2", map[string]any{"input_cost_per_token": 1e-8}), embedInfo, false),
		modelRow(t, "id-rerank", "bge-reranker-v2-m3", vllmParams(t, "bge-reranker-v2-m3", "rgm-s-dsapp01-rerank-1", nil), map[string]any{"mode": "rerank"}, false),
		modelRow(t, "id-blocked", "deepseek-v4-flash", vllmParams(t, "deepseek-v4-flash", "ray-service-prod", priced(4e-7, 2.4e-6)), chatInfo(), true),
		// Defaults are a vLLM feature: an OpenAI-typed deployment must not get them.
		modelRow(t, "id-openai", "gpt-x", map[string]any{
			"model":                   encrypted(t, "gpt-x"),
			"custom_llm_provider":     encrypted(t, "openai"),
			"litellm_credential_name": encrypted(t, "openai-cred"),
			"temperature":             0.3,
		}, chatInfo(), false),
		// A slash inside a real model id is not a provider prefix.
		modelRow(t, "id-hf", "hf-model", vllmParams(t, "Qwen/Qwen3-8B", "ray-service-prod", nil), chatInfo(), false),
		// No named credential: connection details are inline, without any API key.
		modelRow(t, "id-inline", "inline-model", map[string]any{
			"model":               encrypted(t, "hosted_vllm/inline-model-real"),
			"custom_llm_provider": encrypted(t, "hosted_vllm"),
			"api_base":            encrypted(t, "http://inline:8000/v1"),
		}, chatInfo(), false),
		// The bound credential carries no provider; it comes from this deployment.
		modelRow(t, "id-noinfo", "noinfo-model", vllmParams(t, "noinfo-real", "no-provider-cred", nil), chatInfo(), false),
		// A credential name written in plaintext by another tool still binds.
		modelRow(t, "id-plain", "plain-model", map[string]any{
			"model":                   encrypted(t, "plain-real"),
			"custom_llm_provider":     encrypted(t, "hosted_vllm"),
			"litellm_credential_name": "plain-cred",
		}, chatInfo(), false),
		// A real model group named like an alias wins over the alias.
		modelRow(t, "id-gemma", "gemma-3-27b-it", vllmParams(t, "gemma-3-27b-it", "ray-service-prod", nil), chatInfo(), false),
	}
}

type builtFixture struct {
	creds   []config.CredentialConfig
	models  []config.ModelRPMConfig
	aliases map[string]string
	prices  map[string]bool
	priced  map[string]float64 // model name -> input cost
}

func buildFixture(t *testing.T) builtFixture {
	t.Helper()
	creds, models, prices, aliases := buildAIRModels(testhelpers.NewTestLogger(),
		fixtureCredentials(t), fixtureModels(t), fixtureRouterSettings(), fixtureSigningKey)
	out := builtFixture{creds: creds, models: models, aliases: aliases, prices: map[string]bool{}, priced: map[string]float64{}}
	for name, price := range prices {
		out.prices[name] = true
		out.priced[name] = price.InputCostPerToken
	}
	return out
}

func (f builtFixture) modelsNamed(name string) []config.ModelRPMConfig {
	var out []config.ModelRPMConfig
	for _, m := range f.models {
		if m.Name == name {
			out = append(out, m)
		}
	}
	return out
}

func (f builtFixture) only(t *testing.T, name string) config.ModelRPMConfig {
	t.Helper()
	found := f.modelsNamed(name)
	require.Len(t, found, 1, "model %q", name)
	return found[0]
}

func (f builtFixture) credential(t *testing.T, name string) config.CredentialConfig {
	t.Helper()
	for _, c := range f.creds {
		if c.Name == name {
			return c
		}
	}
	require.Failf(t, "credential missing", "%q", name)
	return config.CredentialConfig{}
}

func TestBuildAIRModels_VLLMCredentials(t *testing.T) {
	f := buildFixture(t)

	for _, name := range []string{"ray-service-prod", "vllm-only-h200x8-deployment-nodeport", "rgm-s-dsapp01-frida-1", "plain-cred"} {
		assert.Equal(t, config.ProviderTypeVLLM, f.credential(t, name).Type, name)
	}
	ray := f.credential(t, "ray-service-prod")
	assert.Equal(t, "http://ray:8000/v1", ray.BaseURL, "the encrypted api_base is decrypted")
	assert.Empty(t, ray.APIKey, "vLLM is used without an API key")
	assert.Equal(t, config.ProviderTypeOpenAI, f.credential(t, "openai-cred").Type)

	// The credential has no provider of its own: it is taken from the deployment.
	assert.Equal(t, config.ProviderTypeVLLM, f.credential(t, "no-provider-cred").Type)
}

func TestBuildAIRModels_RealModelNames(t *testing.T) {
	f := buildFixture(t)

	// Model behind a different real name.
	ultra := f.only(t, "qwen3.5-397b-a17b-fp8")
	assert.Equal(t, "qwen397b-int4", ultra.Model)
	assert.Equal(t, "ray-service-prod", ultra.Credential)

	// The provider prefix vLLM does not know is stripped.
	assert.Equal(t, "qwen-36-35b-fp8", f.only(t, "qwen-36-35b-fast").Model)

	// Same real name as the group: nothing to map.
	assert.Empty(t, f.only(t, "gpt-oss-120b").Model)

	// A slash that belongs to the model id is kept.
	assert.Equal(t, "Qwen/Qwen3-8B", f.only(t, "hf-model").Model)
}

func TestBuildAIRModels_SkipsBlockedAndRerank(t *testing.T) {
	f := buildFixture(t)
	assert.Empty(t, f.modelsNamed("deepseek-v4-flash"), "blocked deployments must not serve traffic")
	assert.Empty(t, f.modelsNamed("bge-reranker-v2-m3"), "/rerank is not supported")
}

func TestBuildAIRModels_SeveralDeploymentsPerName(t *testing.T) {
	f := buildFixture(t)
	frida := f.modelsNamed("frida")
	require.Len(t, frida, 2)
	assert.ElementsMatch(t, []string{"rgm-s-dsapp01-frida-1", "rgm-s-dsapp01-frida-2"},
		[]string{frida[0].Credential, frida[1].Credential})
}

func TestBuildAIRModels_CredentialBinding(t *testing.T) {
	f := buildFixture(t)

	// The plaintext name and the provider-less credential both bind.
	assert.Equal(t, "plain-cred", f.only(t, "plain-model").Credential)
	assert.Equal(t, "plain-real", f.only(t, "plain-model").Model)
	assert.Equal(t, "no-provider-cred", f.only(t, "noinfo-model").Credential)

	// No credential at all: a synthetic one is made from api_base alone.
	inline := f.only(t, "inline-model")
	assert.Equal(t, "db-model-id-inline", inline.Credential)
	assert.Equal(t, "inline-model-real", inline.Model)
	synthetic := f.credential(t, "db-model-id-inline")
	assert.Equal(t, config.ProviderTypeVLLM, synthetic.Type)
	assert.Equal(t, "http://inline:8000/v1", synthetic.BaseURL)

	known := map[string]bool{}
	for _, c := range f.creds {
		known[c.Name] = true
	}
	for _, m := range f.models {
		assert.True(t, known[m.Credential], "model %q references unknown credential %q", m.Name, m.Credential)
	}
}

func TestBuildAIRModels_DefaultParamsOnlyForVLLM(t *testing.T) {
	f := buildFixture(t)

	assert.Equal(t, map[string]any{
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
		"temperature":          0.7,
		"top_p":                0.8,
		"top_k":                20.0,
		"min_p":                0.0,
		"presence_penalty":     1.5,
		"repetition_penalty":   1.0,
	}, f.only(t, "qwen-36-35b-fast").DefaultParams)
	assert.Equal(t, map[string]any{"presence_penalty": 1.5}, f.only(t, "qwen-36-35b-fp8").DefaultParams)
	assert.Equal(t, map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}},
		f.only(t, "qwen-ultra-flash").DefaultParams)
	assert.Nil(t, f.only(t, "gpt-oss-120b").DefaultParams)

	// An OpenAI-typed deployment with the same kind of param gets none.
	assert.Nil(t, f.only(t, "gpt-x").DefaultParams)
}

func TestBuildAIRModels_ModelGroupAliases(t *testing.T) {
	f := buildFixture(t)

	// A usable alias resolves to its target group at routing time.
	assert.Equal(t, map[string]string{
		"gpt-oss":     "gpt-oss-120b",
		"qwen-flash":  "qwen-36-35b-fast",
		"qwen-ultra":  "qwen3.5-397b-a17b-fp8",
		"coder-ultra": "kimi-k26",
		"qwen-36-35b": "qwen-36-35b-fp8",
		"qwen3-coder": "qwen-ultra-flash",
	}, f.aliases)

	// It is not a model of its own: a copy of the deployments would get its own rate
	// limiter, balancer state and bans, so the two names would not share the target's limits.
	for alias, target := range f.aliases {
		assert.Empty(t, f.modelsNamed(alias), "alias %s must not become a model", alias)
		assert.NotEmpty(t, f.modelsNamed(target), "alias %s target must be served", alias)
	}

	// Deployments stay under the target group, with the provider-facing name and the
	// default params the requests routed there will use.
	assert.Equal(t, "qwen397b-int4", f.only(t, "qwen3.5-397b-a17b-fp8").Model)
	assert.Equal(t, "ray-service-prod", f.only(t, "qwen3.5-397b-a17b-fp8").Credential)
	assert.Equal(t, "vllm-only-h200x8-deployment-nodeport", f.only(t, "kimi-k26").Credential)
	assert.Equal(t, "qwen-36-35b-fp8", f.only(t, "qwen-36-35b-fast").Model)
	assert.NotNil(t, f.only(t, "qwen-36-35b-fast").DefaultParams)

	// An alias never overrides a real group of the same name.
	assert.NotContains(t, f.aliases, "gemma-3-27b-it")
	gemma := f.only(t, "gemma-3-27b-it")
	assert.Empty(t, gemma.Model)

	// Unusable aliases are dropped quietly.
	assert.NotContains(t, f.aliases, "ghost")
	assert.NotContains(t, f.aliases, "self")

	// router_settings.fallbacks is not turned into models.
	assert.Len(t, f.modelsNamed("kimi-k26"), 1)
}

func TestBuildAIRModels_AliasesAreBilledAsTheirTarget(t *testing.T) {
	f := buildFixture(t)

	// The alias is resolved to its target before billing, so prices live under the
	// target's name only.
	assert.InDelta(t, 4e-7, f.priced["qwen3.5-397b-a17b-fp8"], 1e-15)
	assert.InDelta(t, 2e-7, f.priced["qwen-36-35b-fast"], 1e-15)
	assert.InDelta(t, 1e-6, f.priced["kimi-k26"], 1e-15)
	assert.True(t, f.prices["gpt-oss-120b"])
	for alias := range f.aliases {
		assert.False(t, f.prices[alias], "alias %s has no price of its own", alias)
	}
	assert.False(t, f.prices["ghost"])
}

func TestBuildAIRModels_IsDeterministic(t *testing.T) {
	first := buildFixture(t)
	second := buildFixture(t)
	assert.Equal(t, first.models, second.models)
	assert.Equal(t, first.creds, second.creds)
}

func TestBuildAIRModels_NoRouterSettings(t *testing.T) {
	_, models, _, aliases := buildAIRModels(testhelpers.NewTestLogger(),
		fixtureCredentials(t), fixtureModels(t), queries.RouterSettings{}, fixtureSigningKey)
	assert.Empty(t, aliases)
	assert.NotEmpty(t, models)
}

func TestMapProviderType_VLLM(t *testing.T) {
	for _, in := range []string{"hosted_vllm", "vllm", "HOSTED_VLLM", "hosted-vllm"} {
		assert.Equal(t, config.ProviderTypeVLLM, mapProviderType(in), in)
	}
}
