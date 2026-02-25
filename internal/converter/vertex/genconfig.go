package vertex

import (
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// buildGenerationConfig constructs Vertex GenerationConfig from OpenAI request params.
// Returns nil if no config parameters are set.
func buildGenerationConfig(req *openai.OpenAIRequest, model string) *genai.GenerationConfig {
	// Check if any config params are present
	hasParams := req.Temperature != nil || req.MaxTokens != nil || req.MaxCompletionTokens != nil ||
		req.TopP != nil || req.ExtraBody != nil || req.N != nil || req.Seed != nil ||
		req.FrequencyPenalty != nil || req.PresencePenalty != nil || req.Stop != nil ||
		len(req.Modalities) > 0 || req.ReasoningEffort != "" || req.ResponseFormat != nil ||
		req.Logprobs != nil || req.TopLogprobs != nil

	if !hasParams {
		return nil
	}

	// Parameters not supported by Vertex AI (ignored):
	//   - LogitBias: Vertex AI does not support token bias
	//   - User: no user tracking field in GenerationConfig
	//   - Store: Vertex AI does not store completions
	//   - ServiceTier: no latency tier configuration in Vertex AI
	//   - Metadata: Vertex AI request does not accept user metadata
	//   - PromptCacheKey: Vertex AI handles caching automatically
	//   - PromptCacheRetention: managed by Vertex AI automatically
	//   - Verbosity: not supported in Vertex AI
	//   - Prediction: speculative decoding not available in Vertex AI
	//   - ParallelToolCalls: Vertex AI always allows parallel tool calls (cannot be disabled)
	//   - StreamOptions: stream_options.include_usage is always enabled in this proxy

	cfg := &genai.GenerationConfig{}

	// Direct scalar params
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	if req.MaxTokens != nil {
		cfg.MaxOutputTokens = int32(*req.MaxTokens)
	}
	// max_completion_tokens takes precedence over max_tokens
	if req.MaxCompletionTokens != nil {
		cfg.MaxOutputTokens = int32(*req.MaxCompletionTokens)
	}
	if req.TopP != nil {
		v := float32(*req.TopP)
		cfg.TopP = &v
	}
	if req.N != nil {
		cfg.CandidateCount = int32(*req.N)
	}
	if req.Seed != nil {
		v := int32(*req.Seed)
		cfg.Seed = &v
	}
	if req.FrequencyPenalty != nil {
		v := float32(*req.FrequencyPenalty)
		cfg.FrequencyPenalty = &v
	}
	if req.PresencePenalty != nil {
		v := float32(*req.PresencePenalty)
		cfg.PresencePenalty = &v
	}

	// Phase 5: LogProbs
	if req.Logprobs != nil && *req.Logprobs {
		cfg.ResponseLogprobs = true
	}
	if req.TopLogprobs != nil && *req.TopLogprobs > 0 {
		v := int32(*req.TopLogprobs)
		cfg.Logprobs = &v
	}

	// Stop sequences
	if req.Stop != nil {
		switch stop := req.Stop.(type) {
		case string:
			cfg.StopSequences = []string{stop}
		case []interface{}:
			for _, s := range stop {
				if str, ok := s.(string); ok {
					cfg.StopSequences = append(cfg.StopSequences, str)
				}
			}
		}
	}

	// Modalities (direct field)
	for _, mod := range req.Modalities {
		cfg.ResponseModalities = append(cfg.ResponseModalities, genai.Modality(strings.ToUpper(mod)))
	}

	// ExtraBody overrides
	if req.ExtraBody != nil {
		applyExtraBodyToConfig(cfg, req.ExtraBody, req.Model)
	}

	// Deduplicate ResponseModalities (modalities may be added from multiple sources:
	// req.Modalities, extra_body.generation_config.response_modalities, extra_body.modalities)
	if len(cfg.ResponseModalities) > 1 {
		seen := make(map[genai.Modality]struct{})
		unique := cfg.ResponseModalities[:0]
		for _, m := range cfg.ResponseModalities {
			if _, ok := seen[m]; !ok {
				seen[m] = struct{}{}
				unique = append(unique, m)
			}
		}
		cfg.ResponseModalities = unique
	}

	// Response format / JSON schema
	if req.ResponseFormat != nil {
		if schema := convertOpenAIResponseFormatToGenaiSchema(req.ResponseFormat); schema != nil {
			cfg.ResponseSchema = schema
		}
		if rfMap, ok := req.ResponseFormat.(map[string]interface{}); ok {
			if rfType, ok := rfMap["type"].(string); ok && (rfType == "json_schema" || rfType == "json_object") {
				cfg.ResponseMIMEType = "application/json"
			}
		}
	}

	// Phase 3: Thinking / Reasoning
	var thinkingParam interface{}
	if req.ExtraBody != nil {
		thinkingParam = req.ExtraBody["thinking"]
	}
	if tc := mapReasoningToThinkingConfig(thinkingParam, req.ReasoningEffort, model); tc != nil {
		cfg.ThinkingConfig = tc
	}

	// Phase 4: Audio output (SpeechConfig)
	if req.ExtraBody != nil {
		if audioParam, ok := req.ExtraBody["audio"]; ok {
			if sc := mapAudioParam(audioParam); sc != nil {
				cfg.SpeechConfig = sc
			}
		}
	}

	return cfg
}

// applyExtraBodyToConfig applies extra_body generation_config overrides to genai.GenerationConfig.
func applyExtraBodyToConfig(cfg *genai.GenerationConfig, extraBody map[string]interface{}, model string) {
	// generation_config nested object
	if gcMap, ok := extraBody["generation_config"].(map[string]interface{}); ok {
		if mimeType, ok := gcMap["response_mime_type"].(string); ok {
			// Skip response_mime_type for image generation models
			if !strings.Contains(strings.ToLower(model), "image") {
				cfg.ResponseMIMEType = mimeType
			}
		}
		if modalities, ok := gcMap["response_modalities"].([]interface{}); ok {
			for _, m := range modalities {
				if mod, ok := m.(string); ok {
					cfg.ResponseModalities = append(cfg.ResponseModalities, genai.Modality(mod))
				}
			}
		}
		if topK, ok := gcMap["top_k"].(float64); ok {
			v := float32(topK)
			cfg.TopK = &v
		}
		if seed, ok := gcMap["seed"].(float64); ok {
			v := int32(seed)
			cfg.Seed = &v
		}
		if temp, ok := gcMap["temperature"].(float64); ok {
			v := float32(temp)
			cfg.Temperature = &v
		}
	}

	// top-level modalities in extra_body
	if modalities, ok := extraBody["modalities"].([]interface{}); ok {
		for _, m := range modalities {
			if mod, ok := m.(string); ok {
				cfg.ResponseModalities = append(cfg.ResponseModalities, genai.Modality(strings.ToUpper(mod)))
			}
		}
	}
}

// mapAudioParam converts OpenAI audio param to genai.SpeechConfig (Phase 4).
// Format: {"voice": "alloy", "format": "wav"}
func mapAudioParam(audioParam interface{}) *genai.SpeechConfig {
	audioMap, ok := audioParam.(map[string]interface{})
	if !ok {
		return nil
	}

	speechConfig := &genai.SpeechConfig{}
	if voice, ok := audioMap["voice"].(string); ok && voice != "" {
		speechConfig.VoiceConfig = &genai.VoiceConfig{
			PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
				VoiceName: voice,
			},
		}
	}

	return speechConfig
}
