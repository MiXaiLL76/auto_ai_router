package queries

import "encoding/json"

const QueryProxyModelTable = `SELECT model_id, model_name, litellm_params, model_info FROM public."LiteLLM_ProxyModelTable"`

// QueryProxyModelTableWithBlocked also reads the blocked flag that newer LiteLLM
// schemas add to the table. Databases from before that migration lack the column;
// the loader falls back to QueryProxyModelTable when this query fails with an
// undefined_column error.
const QueryProxyModelTableWithBlocked = `SELECT model_id, model_name, litellm_params, model_info, blocked FROM public."LiteLLM_ProxyModelTable"`

// QueryRouterSettings reads the router_settings row of LiteLLM_Config, where the
// proxy keeps model_group_alias and fallbacks.
const QueryRouterSettings = `SELECT param_value FROM public."LiteLLM_Config" WHERE param_name = 'router_settings'`

// https://github.com/BerriAI/litellm/blob/v1.80.13.rc.1/litellm/types/router.py

// CustomPricingLiteLLMParams содержит настройки стоимости для токенов, времени и медиафайлов
type CustomPricingLiteLLMParams struct {
	InputCostPerToken                 *float64 `json:"input_cost_per_token,omitempty"`
	OutputCostPerToken                *float64 `json:"output_cost_per_token,omitempty"`
	OutputCostPerTokenAbove32kTokens  *float64 `json:"output_cost_per_token_above_32k_tokens,omitempty"`
	OutputCostPerTokenAbove128kTokens *float64 `json:"output_cost_per_token_above_128k_tokens,omitempty"`
	OutputCostPerTokenAbove200kTokens *float64 `json:"output_cost_per_token_above_200k_tokens,omitempty"`
	OutputCostPerTokenAbove256kTokens *float64 `json:"output_cost_per_token_above_256k_tokens,omitempty"`
	OutputCostPerTokenAbove272kTokens *float64 `json:"output_cost_per_token_above_272k_tokens,omitempty"`
	OutputCostPerTokenAbove512kTokens *float64 `json:"output_cost_per_token_above_512k_tokens,omitempty"`

	InputCostPerSecond  *float64 `json:"input_cost_per_second,omitempty"`
	OutputCostPerSecond *float64 `json:"output_cost_per_second,omitempty"`

	// Гибкие настройки стоимости (Flex/Priority/Cache)
	CacheReadInputTokenCost                            *float64 `json:"cache_read_input_token_cost,omitempty"`
	CacheCreationInputTokenCost                        *float64 `json:"cache_creation_input_token_cost,omitempty"`
	CacheReadInputTokenCostAbove32kTokens              *float64 `json:"cache_read_input_token_cost_above_32k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove32kTokens          *float64 `json:"cache_creation_input_token_cost_above_32k_tokens,omitempty"`
	CacheReadInputTokenCostAbove128kTokens             *float64 `json:"cache_read_input_token_cost_above_128k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove128kTokens         *float64 `json:"cache_creation_input_token_cost_above_128k_tokens,omitempty"`
	CacheReadInputTokenCostAbove200kTokens             *float64 `json:"cache_read_input_token_cost_above_200k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove200kTokens         *float64 `json:"cache_creation_input_token_cost_above_200k_tokens,omitempty"`
	CacheReadInputTokenCostAbove256kTokens             *float64 `json:"cache_read_input_token_cost_above_256k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove256kTokens         *float64 `json:"cache_creation_input_token_cost_above_256k_tokens,omitempty"`
	CacheReadInputTokenCostAbove512kTokens             *float64 `json:"cache_read_input_token_cost_above_512k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove512kTokens         *float64 `json:"cache_creation_input_token_cost_above_512k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove1hr                *float64 `json:"cache_creation_input_token_cost_above_1hr,omitempty"`
	CacheCreationInputTokenCostAbove1hrAbove200kTokens *float64 `json:"cache_creation_input_token_cost_above_1hr_above_200k_tokens,omitempty"`
	CacheReadInputTokenCostAbove272kTokens             *float64 `json:"cache_read_input_token_cost_above_272k_tokens,omitempty"`
	CacheCreationInputTokenCostAbove272kTokens         *float64 `json:"cache_creation_input_token_cost_above_272k_tokens,omitempty"`
	CacheReadInputAudioTokenCost                       *float64 `json:"cache_read_input_audio_token_cost,omitempty"`

	InputCostPerTokenAbove32kTokens  *float64 `json:"input_cost_per_token_above_32k_tokens,omitempty"`
	InputCostPerTokenAbove128kTokens *float64 `json:"input_cost_per_token_above_128k_tokens,omitempty"`
	InputCostPerTokenAbove200kTokens *float64 `json:"input_cost_per_token_above_200k_tokens,omitempty"`
	InputCostPerTokenAbove256kTokens *float64 `json:"input_cost_per_token_above_256k_tokens,omitempty"`
	InputCostPerTokenAbove272kTokens *float64 `json:"input_cost_per_token_above_272k_tokens,omitempty"`
	InputCostPerTokenAbove512kTokens *float64 `json:"input_cost_per_token_above_512k_tokens,omitempty"`

	InputCostPerAudioToken                    *float64 `json:"input_cost_per_audio_token,omitempty"`
	InputCostPerAudioPerSecond                *float64 `json:"input_cost_per_audio_per_second,omitempty"`
	InputCostPerAudioPerSecondAbove128kTokens *float64 `json:"input_cost_per_audio_per_second_above_128k_tokens,omitempty"`
	OutputCostPerAudioToken                   *float64 `json:"output_cost_per_audio_token,omitempty"`
	OutputCostPerAudioPerSecond               *float64 `json:"output_cost_per_audio_per_second,omitempty"`

	InputCostPerVideoPerSecond                 *float64 `json:"input_cost_per_video_per_second,omitempty"`
	InputCostPerVideoPerSecondAbove128kTokens  *float64 `json:"input_cost_per_video_per_second_above_128k_tokens,omitempty"`
	InputCostPerVideoPerSecondAbove15sInterval *float64 `json:"input_cost_per_video_per_second_above_15s_interval,omitempty"`
	InputCostPerVideoPerSecondAbove8sInterval  *float64 `json:"input_cost_per_video_per_second_above_8s_interval,omitempty"`
	OutputCostPerVideoPerSecond                *float64 `json:"output_cost_per_video_per_second,omitempty"`

	InputCostPerImage                *float64           `json:"input_cost_per_image,omitempty"`
	InputCostPerImageAbove128kTokens *float64           `json:"input_cost_per_image_above_128k_tokens,omitempty"`
	OutputCostPerImage               *float64           `json:"output_cost_per_image,omitempty"`
	OutputCostPerImageToken          *float64           `json:"output_cost_per_image_token,omitempty"`
	OutputCostPerReasoningToken      *float64           `json:"output_cost_per_reasoning_token,omitempty"`
	SearchContextCostPerQuery        map[string]float64 `json:"search_context_cost_per_query,omitempty"`
	WebSearchBillingUnit             *string            `json:"web_search_billing_unit,omitempty"`
}

// GenericLiteLLMParams
type GenericLiteLLMParams struct {
	// Встраивание родительских структур (предполагается, что они определены)
	CredentialLiteLLMParams
	CustomPricingLiteLLMParams

	CustomLLMProvider *string `json:"custom_llm_provider,omitempty"`
	CredentialName    *string `json:"credential_name,omitempty"`
	// LiteLLMCredentialName is the key LiteLLM actually stores in litellm_params
	// to bind a deployment to a LiteLLM_CredentialsTable row (types/router.py
	// LiteLLM_Params.litellm_credential_name). It is encrypted like the other
	// secrets. CredentialName is kept as a legacy spelling.
	LiteLLMCredentialName *string `json:"litellm_credential_name,omitempty"`
	TPM                   *int    `json:"tpm,omitempty"`
	RPM                   *int    `json:"rpm,omitempty"`

	// Per-deployment default request params. LiteLLM merges litellm_params under
	// the request kwargs, so a client-supplied value always wins. AIR applies
	// them to vLLM deployments only (see DefaultRequestParams).
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
	Temperature        *float64       `json:"temperature,omitempty"`
	TopP               *float64       `json:"top_p,omitempty"`
	TopK               *float64       `json:"top_k,omitempty"`
	MinP               *float64       `json:"min_p,omitempty"`
	PresencePenalty    *float64       `json:"presence_penalty,omitempty"`
	FrequencyPenalty   *float64       `json:"frequency_penalty,omitempty"`
	RepetitionPenalty  *float64       `json:"repetition_penalty,omitempty"`
	// Integer params keep their literal: a float64 round trip corrupts a seed above 2^53.
	MaxTokens *json.Number `json:"max_tokens,omitempty"`
	Seed      *json.Number `json:"seed,omitempty"`

	ModelInfo map[string]interface{} `json:"model_info,omitempty"`
}

// EffectiveCredentialName returns the credential this deployment is bound to,
// preferring LiteLLM's litellm_credential_name over the legacy credential_name.
func (p *GenericLiteLLMParams) EffectiveCredentialName() string {
	if p == nil {
		return ""
	}
	if p.LiteLLMCredentialName != nil && *p.LiteLLMCredentialName != "" {
		return *p.LiteLLMCredentialName
	}
	if p.CredentialName != nil {
		return *p.CredentialName
	}
	return ""
}

// DefaultRequestParams returns the deployment's default request-body params as a
// JSON-ready map (only the keys that are set), or nil when there are none.
func (p *GenericLiteLLMParams) DefaultRequestParams() map[string]any {
	if p == nil {
		return nil
	}
	out := make(map[string]any)
	if len(p.ChatTemplateKwargs) > 0 {
		out["chat_template_kwargs"] = p.ChatTemplateKwargs
	}
	for key, val := range map[string]*float64{
		"temperature":        p.Temperature,
		"top_p":              p.TopP,
		"top_k":              p.TopK,
		"min_p":              p.MinP,
		"presence_penalty":   p.PresencePenalty,
		"frequency_penalty":  p.FrequencyPenalty,
		"repetition_penalty": p.RepetitionPenalty,
	} {
		if val != nil {
			out[key] = *val
		}
	}
	for key, val := range map[string]*json.Number{
		"max_tokens": p.MaxTokens,
		"seed":       p.Seed,
	} {
		if val != nil {
			out[key] = *val
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

type ModelTable struct {
	ModelID   *string                `json:"model_id,omitempty"`
	ModelName *string                `json:"model_name,omitempty"`
	LlmParams *GenericLiteLLMParams  `json:"litellm_params,omitempty"`
	ModelInfo map[string]interface{} `json:"model_info,omitempty"`
	// Blocked mirrors LiteLLM_ProxyModelTable.blocked; a blocked deployment must
	// not serve traffic. False when the column does not exist.
	Blocked bool `json:"blocked,omitempty"`
}

// Mode returns model_info.mode ("chat", "embedding", "rerank", ...) or "".
func (m ModelTable) Mode() string {
	mode, _ := m.ModelInfo["mode"].(string)
	return mode
}

// RouterSettings is the subset of LiteLLM_Config.router_settings AIR imports.
type RouterSettings struct {
	// ModelGroupAlias maps a client-facing name to the model group it resolves to.
	ModelGroupAlias map[string]string
	// Fallbacks maps a model group to the groups LiteLLM falls back to, in order.
	Fallbacks map[string][]string
}
