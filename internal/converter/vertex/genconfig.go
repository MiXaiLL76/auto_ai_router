package vertex

import (
	"cmp"
	"strings"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"google.golang.org/genai"
)

// buildGenerationConfig constructs Vertex GenerationConfig from OpenAI request params.
// Returns nil if no config parameters are set, except for a model whose profile pins
// its thinking levels: it always takes its default level, as on the images and
// Responses routes.
func buildGenerationConfig(req *openai.OpenAIRequest, model string) *VertexGenerationConfig {
	if !hasGenerationParams(req) && !lookupGeminiModelProfile(model).hasThinkingLevels() {
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

	applyChatGenerationParams(cfg.GenerationConfig, req, model)
	cfg.ImageConfig = resolveChatImageConfig(cfg.GenerationConfig, req, model)
	cfg.ResponseModalities = uniqueModalities(cfg.ResponseModalities)
	applyChatResponseFormat(cfg.GenerationConfig, req.ResponseFormat)
	cfg.ThinkingConfig = resolveChatThinkingConfig(req, model)
	capDynamicThinkingBudget(cfg.GenerationConfig, model)
	if audio, ok := req.ExtraBody["audio"]; ok {
		cfg.SpeechConfig = mapAudioParam(audio)
	}
	ApplyGenerationConstraints(cfg.GenerationConfig, model)
	return cfg
}

func hasGenerationParams(req *openai.OpenAIRequest) bool {
	return req.Temperature != nil || req.MaxTokens != nil || req.MaxCompletionTokens != nil ||
		req.TopP != nil || req.ExtraBody != nil || req.N != nil || req.Seed != nil ||
		req.FrequencyPenalty != nil || req.PresencePenalty != nil || req.Stop != nil ||
		len(req.Modalities) > 0 || req.ReasoningEffort != "" || req.ResponseFormat != nil ||
		req.Logprobs != nil || req.TopLogprobs != nil ||
		req.Thinking != nil || req.ThinkingBudget != nil || req.ThinkingLevel != "" || req.ThinkingLevelCamel != "" ||
		req.ThinkingConfig != nil || req.ThinkingConfigCamel != nil ||
		hasTopLevelImageConfig(req)
}

func applyChatGenerationParams(cfg *genai.GenerationConfig, req *openai.OpenAIRequest, model string) {
	// Direct scalar params
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	// A supplied max_completion_tokens owns the limit even when it is zero.
	if limit := cmp.Or(req.MaxCompletionTokens, req.MaxTokens); limit != nil {
		cfg.MaxOutputTokens = ClampInt32(*limit)
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

	// Modalities (direct field)
	for _, mod := range req.Modalities {
		cfg.ResponseModalities = append(cfg.ResponseModalities, genai.Modality(strings.ToUpper(mod)))
	}
}

func resolveChatImageConfig(cfg *genai.GenerationConfig, req *openai.OpenAIRequest, model string) *genai.ImageConfig {
	image := applyExtraBodyToConfig(cfg, req.ExtraBody, model)
	top := parseTopLevelImageConfig(req)
	if image == nil {
		return top
	}
	if top != nil {
		// Merge field by field so a top-level aspect ratio can coexist with
		// an image size specified under extra_body.generation_config.
		image.AspectRatio = cmp.Or(top.AspectRatio, image.AspectRatio)
		image.ImageSize = cmp.Or(top.ImageSize, image.ImageSize)
	}
	return image
}

func uniqueModalities(modalities []genai.Modality) []genai.Modality {
	if len(modalities) < 2 {
		return modalities
	}
	seen := make(map[genai.Modality]struct{}, len(modalities))
	unique := modalities[:0]
	for _, modality := range modalities {
		if _, ok := seen[modality]; !ok {
			seen[modality] = struct{}{}
			unique = append(unique, modality)
		}
	}
	return unique
}

func applyChatResponseFormat(cfg *genai.GenerationConfig, format interface{}) {
	if format == nil {
		return
	}
	if schema := convertOpenAIResponseFormatToGenaiSchema(format); schema != nil {
		cfg.ResponseSchema = schema
	}
	if fields, ok := format.(map[string]interface{}); ok {
		kind, _ := fields["type"].(string)
		switch kind {
		case "json_schema", "json_object":
			cfg.ResponseMIMEType = "application/json"
		}
	}
}

// Resolve the first usable source: native config, Gemini shorthands,
// Anthropic thinking / reasoning effort, then the model default.
func resolveChatThinkingConfig(req *openai.OpenAIRequest, model string) *genai.ThinkingConfig {
	for _, fields := range nativeThinkingConfigs(req) {
		if tc := mapNativeThinkingConfig(fields, model); tc != nil {
			return tc
		}
	}
	level := cmp.Or(req.ThinkingLevel, req.ThinkingLevelCamel,
		extraBodyString(req.ExtraBody, "thinking_level"), extraBodyString(req.ExtraBody, "thinkingLevel"))
	budget := cmp.Or(req.ThinkingBudget, req.ExtraBody["thinking_budget"])
	if budget != nil || level != "" {
		if tc := mapNativeThinkingConfig(map[string]interface{}{
			"thinking_budget": budget,
			"thinking_level":  level,
		}, model); tc != nil {
			return tc
		}
	}
	thinking := cmp.Or(req.Thinking, req.ExtraBody["thinking"])
	effort := cmp.Or(req.ReasoningEffort, extraBodyString(req.ExtraBody, "reasoning_effort"))
	if tc := mapReasoningToThinkingConfig(thinking, effort, model); tc != nil {
		return tc
	}
	return DefaultThinkingConfig(model)
}

// Gemini 2.5 Pro's dynamic thinking can consume the entire output budget.
// Cap it at half the budget when the result meets the minimum fixed budget.
func capDynamicThinkingBudget(cfg *genai.GenerationConfig, model string) {
	if !isGemini25ProModel(model) || cfg.ThinkingConfig == nil || cfg.ThinkingConfig.ThinkingBudget == nil ||
		*cfg.ThinkingConfig.ThinkingBudget != -1 {
		return
	}
	if budget := cfg.MaxOutputTokens / 2; budget >= 128 {
		cfg.ThinkingConfig.ThinkingBudget = &budget
	}
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
	if !hasTopLevelImageConfig(req) {
		return nil
	}
	params := make(map[string]interface{}, len(req.ImageConfig)+len(req.ImageConfigSnake))
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
