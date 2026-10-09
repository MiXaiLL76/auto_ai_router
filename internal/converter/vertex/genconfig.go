package vertex

import (
	"cmp"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// buildGenerationConfig constructs Vertex GenerationConfig from OpenAI request params.
// Returns nil if no config parameters are set.
func buildGenerationConfig(req *openai.OpenAIRequest, model string) *VertexGenerationConfig {
	// Check if any config params are present
	hasParams := req.Temperature != nil || req.MaxTokens != nil || req.MaxCompletionTokens != nil ||
		req.TopP != nil || req.ExtraBody != nil || req.N != nil || req.Seed != nil ||
		req.FrequencyPenalty != nil || req.PresencePenalty != nil || req.Stop != nil ||
		len(req.Modalities) > 0 || req.ReasoningEffort != "" || req.ResponseFormat != nil ||
		req.Logprobs != nil || req.TopLogprobs != nil ||
		req.Thinking != nil || req.ThinkingBudget != nil || req.ThinkingLevel != "" || req.ThinkingLevelCamel != "" ||
		req.ThinkingConfig != nil || req.ThinkingConfigCamel != nil ||
		hasTopLevelImageConfig(req)

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

	cfg := &VertexGenerationConfig{GenerationConfig: &genai.GenerationConfig{}}

	// Direct scalar params
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	if req.MaxTokens != nil {
		cfg.MaxOutputTokens = ClampInt32(*req.MaxTokens)
	}
	// max_completion_tokens takes precedence over max_tokens
	if req.MaxCompletionTokens != nil {
		cfg.MaxOutputTokens = ClampInt32(*req.MaxCompletionTokens)
	}
	if req.TopP != nil {
		v := float32(*req.TopP)
		cfg.TopP = &v
	}
	if req.N != nil {
		cfg.CandidateCount = ClampInt32(*req.N)
	}
	if req.Seed != nil {
		v := ClampInt32(*req.Seed)
		cfg.Seed = &v
	}
	// Dropped for models that reject them outright — see supportsPenalty.
	if supportsPenalty(model) {
		if req.FrequencyPenalty != nil {
			v := float32(*req.FrequencyPenalty)
			cfg.FrequencyPenalty = &v
		}
		if req.PresencePenalty != nil {
			v := float32(*req.PresencePenalty)
			cfg.PresencePenalty = &v
		}
	}

	// Phase 5: LogProbs
	if req.Logprobs != nil && *req.Logprobs {
		cfg.ResponseLogprobs = true
	}
	if req.TopLogprobs != nil && *req.TopLogprobs > 0 {
		v := ClampInt32(*req.TopLogprobs)
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
		cfg.ImageConfig = applyExtraBodyToConfig(cfg.GenerationConfig, req.ExtraBody, model)
	}
	if topLevelImageConfig := parseTopLevelImageConfig(req); topLevelImageConfig != nil {
		if cfg.ImageConfig == nil {
			cfg.ImageConfig = topLevelImageConfig
		} else {
			if topLevelImageConfig.AspectRatio != "" {
				cfg.ImageConfig.AspectRatio = topLevelImageConfig.AspectRatio
			}
			if topLevelImageConfig.ImageSize != "" {
				cfg.ImageConfig.ImageSize = topLevelImageConfig.ImageSize
			}
		}
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
	// Priority (highest first); each source is read at the top level, where the
	// OpenAI SDKs put extra_body keys, then inside a literal extra_body object:
	//   1. thinking_config (Gemini-native format): see nativeThinkingConfig
	//   2. thinking_budget / thinking_level shorthands
	//   3. thinking (Anthropic-style)
	//   4. reasoning_effort
	//   5. Default: the model's own level for a profiled model, otherwise disable
	//      autonomous thinking for predictable latency
	// A source counts only when it sets something (see mapNativeThinkingConfig): an
	// empty or malformed one must not skip the default.
	var thinkingResolved bool
	for _, tcMap := range nativeThinkingConfigs(req) {
		if tc := mapNativeThinkingConfig(tcMap, model); tc != nil {
			cfg.ThinkingConfig = tc
			thinkingResolved = true
			break
		}
	}
	thinkingLevel := cmp.Or(req.ThinkingLevel, req.ThinkingLevelCamel,
		extraBodyString(req.ExtraBody, "thinking_level"), extraBodyString(req.ExtraBody, "thinkingLevel"))
	thinkingBudget := req.ThinkingBudget
	if thinkingBudget == nil {
		thinkingBudget = req.ExtraBody["thinking_budget"]
	}
	reasoningEffort := cmp.Or(req.ReasoningEffort, extraBodyString(req.ExtraBody, "reasoning_effort"))
	if !thinkingResolved && (thinkingBudget != nil || thinkingLevel != "") {
		// Top-level thinking_budget / thinking_level (Gemini-style)
		tcMap := make(map[string]interface{})
		if thinkingBudget != nil {
			tcMap["thinking_budget"] = thinkingBudget
		}
		if thinkingLevel != "" {
			tcMap["thinking_level"] = thinkingLevel
		}
		if tc := mapNativeThinkingConfig(tcMap, model); tc != nil {
			cfg.ThinkingConfig = tc
			thinkingResolved = true
		}
	}
	if !thinkingResolved {
		// Use top-level thinking field first, then extra_body.thinking
		var thinkingParam interface{}
		if req.Thinking != nil {
			thinkingParam = req.Thinking
		} else if req.ExtraBody != nil {
			thinkingParam = req.ExtraBody["thinking"]
		}
		if tc := mapReasoningToThinkingConfig(thinkingParam, reasoningEffort, model); tc != nil {
			cfg.ThinkingConfig = tc
		} else if tc := DefaultThinkingConfig(model); tc != nil {
			// No thinking params specified: explicitly disable dynamic thinking for
			// predictable latency. Without this, Gemini 2.5/3 models autonomously
			// decide whether to reason, causing unpredictable latency spikes.
			cfg.ThinkingConfig = tc
		}
	}

	// For gemini-2.5-pro with dynamic thinking (budget=-1) and a small MaxOutputTokens,
	// cap the thinking budget so that actual output tokens are not starved.
	// budget=0 is not supported for 2.5-pro, so dynamic (-1) is used by default,
	// but dynamic can consume the entire MaxOutputTokens budget for thinking.
	// Only apply the cap when the result is >= 128 (the model's minimum fixed budget).
	if isGemini25ProModel(model) && cfg.MaxOutputTokens > 0 &&
		cfg.ThinkingConfig != nil && cfg.ThinkingConfig.ThinkingBudget != nil &&
		*cfg.ThinkingConfig.ThinkingBudget == -1 {
		const minBudget = int32(128)
		budget := cfg.MaxOutputTokens / 2
		if budget >= minBudget {
			cfg.ThinkingConfig.ThinkingBudget = &budget
		}
		// budget < 128: keep dynamic (-1) — fixed budgets below minimum are rejected by the API
	}

	// Phase 4: Audio output (SpeechConfig)
	if req.ExtraBody != nil {
		if audioParam, ok := req.ExtraBody["audio"]; ok {
			if sc := mapAudioParam(audioParam); sc != nil {
				cfg.SpeechConfig = sc
			}
		}
	}

	ApplyGenerationConstraints(cfg.GenerationConfig, model)

	return cfg
}

// nativeThinkingConfigs returns the request's thinking_config / thinkingConfig objects
// by priority: top level first, then extra_body, then extra_body.generation_config.
func nativeThinkingConfigs(req *openai.OpenAIRequest) []map[string]interface{} {
	var configs []map[string]interface{}
	for _, tcMap := range []map[string]interface{}{req.ThinkingConfig, req.ThinkingConfigCamel} {
		if tcMap != nil {
			configs = append(configs, tcMap)
		}
	}
	gcMap, _ := req.ExtraBody["generation_config"].(map[string]interface{})
	for _, fields := range []map[string]interface{}{req.ExtraBody, gcMap} {
		for _, key := range []string{"thinking_config", "thinkingConfig"} {
			if tcMap, ok := fields[key].(map[string]interface{}); ok {
				configs = append(configs, tcMap)
			}
		}
	}
	return configs
}

// extraBodyString returns the trimmed string extra_body[key], or "".
func extraBodyString(extraBody map[string]interface{}, key string) string {
	value, _ := extraBody[key].(string)
	return strings.TrimSpace(value)
}

// applyExtraBodyToConfig applies extra_body generation_config overrides to genai.GenerationConfig.
func applyExtraBodyToConfig(cfg *genai.GenerationConfig, extraBody map[string]interface{}, model string) *genai.ImageConfig {
	var imageConfig *genai.ImageConfig

	// generation_config nested object
	if gcMap, ok := extraBody["generation_config"].(map[string]interface{}); ok {
		if mimeType, ok := gcMap["response_mime_type"].(string); ok {
			// Skip response_mime_type for image generation models
			if !isImageModel(model) {
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
		imageConfig = parseImageConfig(gcMap["image_config"])
		if imageConfig == nil {
			imageConfig = parseImageConfig(gcMap["imageConfig"])
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

	return imageConfig
}

func parseImageConfig(value interface{}) *genai.ImageConfig {
	params, ok := value.(map[string]interface{})
	if !ok {
		return nil
	}

	aspectRatio, _ := params["aspectRatio"].(string)
	if aspectRatio == "" {
		aspectRatio, _ = params["aspect_ratio"].(string)
	}
	imageSize, _ := params["imageSize"].(string)
	if imageSize == "" {
		imageSize, _ = params["image_size"].(string)
	}
	if aspectRatio == "" && imageSize == "" {
		return nil
	}

	return &genai.ImageConfig{
		AspectRatio: aspectRatio,
		ImageSize:   imageSize,
	}
}

func hasTopLevelImageConfig(req *openai.OpenAIRequest) bool {
	return len(req.ImageConfig) > 0 || len(req.ImageConfigSnake) > 0 ||
		req.AspectRatio != "" || req.AspectRatioSnake != "" ||
		req.ImageSize != "" || req.ImageSizeSnake != ""
}

func parseTopLevelImageConfig(req *openai.OpenAIRequest) *genai.ImageConfig {
	params := make(map[string]interface{})
	for key, value := range req.ImageConfig {
		params[key] = value
	}
	for key, value := range req.ImageConfigSnake {
		params[key] = value
	}
	if req.AspectRatio != "" {
		params["aspectRatio"] = req.AspectRatio
		delete(params, "aspect_ratio")
	}
	if req.AspectRatioSnake != "" {
		params["aspectRatio"] = req.AspectRatioSnake
		delete(params, "aspect_ratio")
	}
	if req.ImageSize != "" {
		params["imageSize"] = req.ImageSize
		delete(params, "image_size")
	}
	if req.ImageSizeSnake != "" {
		params["imageSize"] = req.ImageSizeSnake
		delete(params, "image_size")
	}
	return parseImageConfig(params)
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
