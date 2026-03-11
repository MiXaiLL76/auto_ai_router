package queries

const QueryProxyModelTable = `SELECT model_id, model_name, litellm_params, model_info FROM public."LiteLLM_ProxyModelTable"`

// https://github.com/BerriAI/litellm/blob/v1.80.13.rc.1/litellm/types/router.py

// CustomPricingLiteLLMParams содержит настройки стоимости для токенов, времени и медиафайлов
type CustomPricingLiteLLMParams struct {
	InputCostPerToken                 *float64 `json:"input_cost_per_token,omitempty"`
	OutputCostPerToken                *float64 `json:"output_cost_per_token,omitempty"`
	OutputCostPerTokenAbove128kTokens *float64 `json:"output_cost_per_token_above_128k_tokens,omitempty"`
	OutputCostPerTokenAbove200kTokens *float64 `json:"output_cost_per_token_above_200k_tokens,omitempty"`

	InputCostPerSecond  *float64 `json:"input_cost_per_second,omitempty"`
	OutputCostPerSecond *float64 `json:"output_cost_per_second,omitempty"`

	// Гибкие настройки стоимости (Flex/Priority/Cache)
	CacheReadInputTokenCost                *float64 `json:"cache_read_input_token_cost,omitempty"`
	CacheReadInputTokenCostAbove200kTokens *float64 `json:"cache_read_input_token_cost_above_200k_tokens,omitempty"`
	CacheReadInputAudioTokenCost           *float64 `json:"cache_read_input_audio_token_cost,omitempty"`

	InputCostPerTokenAbove128kTokens *float64 `json:"input_cost_per_token_above_128k_tokens,omitempty"`
	InputCostPerTokenAbove200kTokens *float64 `json:"input_cost_per_token_above_200k_tokens,omitempty"`

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

	InputCostPerImage                *float64 `json:"input_cost_per_image,omitempty"`
	InputCostPerImageAbove128kTokens *float64 `json:"input_cost_per_image_above_128k_tokens,omitempty"`
	OutputCostPerImage               *float64 `json:"output_cost_per_image,omitempty"`
	OutputCostPerImageToken          *float64 `json:"output_cost_per_image_token,omitempty"`
	OutputCostPerReasoningToken      *float64 `json:"output_cost_per_reasoning_token,omitempty"`
}

// GenericLiteLLMParams
type GenericLiteLLMParams struct {
	// Встраивание родительских структур (предполагается, что они определены)
	CredentialLiteLLMParams
	CustomPricingLiteLLMParams

	CustomLLMProvider *string `json:"custom_llm_provider,omitempty"`
	CredentialName    *string `json:"credential_name,omitempty"`
	TPM               *int    `json:"tpm,omitempty"`
	RPM               *int    `json:"rpm,omitempty"`

	ModelInfo map[string]interface{} `json:"model_info,omitempty"`
}

type ModelTable struct {
	ModelId   *string                `json:"model_id,omitempty"`
	ModelName *string                `json:"model_name,omitempty"`
	LlmParams *GenericLiteLLMParams  `json:"litellm_params,omitempty"`
	ModelInfo map[string]interface{} `json:"model_info,omitempty"`
}
