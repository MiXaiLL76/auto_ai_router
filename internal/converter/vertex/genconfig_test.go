package vertex

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter/openai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestBuildGenerationConfig_DefaultThinkingDisable(t *testing.T) {
	t.Run("gemini25_no_params_disables_thinking", func(t *testing.T) {
		req := &openai.OpenAIRequest{Model: "gemini-2.5-flash"}
		// Need at least one param to trigger buildGenerationConfig (hasParams check)
		temp := float64(0.7)
		req.Temperature = &temp
		cfg := buildGenerationConfig(req, "gemini-2.5-flash")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.ThinkingConfig)
		assert.False(t, cfg.ThinkingConfig.IncludeThoughts)
		require.NotNil(t, cfg.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, int32(0), *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini3_no_params_disables_thinking", func(t *testing.T) {
		req := &openai.OpenAIRequest{Model: "gemini-3-flash-preview"}
		temp := float64(0.7)
		req.Temperature = &temp
		cfg := buildGenerationConfig(req, "gemini-3-flash-preview")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.ThinkingConfig)
		assert.False(t, cfg.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelMinimal, cfg.ThinkingConfig.ThinkingLevel)
	})

	t.Run("non_thinking_model_no_thinking_config", func(t *testing.T) {
		req := &openai.OpenAIRequest{Model: "gemini-2.0-flash"}
		temp := float64(0.7)
		req.Temperature = &temp
		cfg := buildGenerationConfig(req, "gemini-2.0-flash")
		require.NotNil(t, cfg)
		assert.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("gemini_image_model_no_thinking_config", func(t *testing.T) {
		req := &openai.OpenAIRequest{Model: "gemini-2.5-flash-image"}
		temp := float64(0.7)
		req.Temperature = &temp
		cfg := buildGenerationConfig(req, "gemini-2.5-flash-image")
		require.NotNil(t, cfg)
		assert.Nil(t, cfg.ThinkingConfig)
	})

	t.Run("explicit_reasoning_effort_overrides_default", func(t *testing.T) {
		req := &openai.OpenAIRequest{
			Model:           "gemini-2.5-flash",
			ReasoningEffort: "high",
		}
		cfg := buildGenerationConfig(req, "gemini-2.5-flash")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.ThinkingConfig)
		assert.False(t, cfg.ThinkingConfig.IncludeThoughts)
		require.NotNil(t, cfg.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, int32(24576), *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("gemini25pro_no_params_uses_dynamic_thinking", func(t *testing.T) {
		// gemini-2.5-pro cannot have budget=0; uses dynamic (-1) as default
		req := &openai.OpenAIRequest{Model: "gemini-2.5-pro"}
		temp := float64(0.7)
		req.Temperature = &temp
		cfg := buildGenerationConfig(req, "gemini-2.5-pro")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.ThinkingConfig, "gemini-2.5-pro: must set dynamic ThinkingConfig")
		require.NotNil(t, cfg.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, int32(-1), *cfg.ThinkingConfig.ThinkingBudget, "must use dynamic (-1) budget")
	})

	t.Run("extra_body_thinking_config_takes_priority", func(t *testing.T) {
		// extra_body.thinking_config has highest priority over reasoning_effort
		req := &openai.OpenAIRequest{
			Model:           "gemini-2.5-flash",
			ReasoningEffort: "low",
			ExtraBody: map[string]interface{}{
				"thinking_config": map[string]interface{}{
					"thinking_budget":  float64(20000),
					"include_thoughts": true,
				},
			},
		}
		cfg := buildGenerationConfig(req, "gemini-2.5-flash")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.ThinkingConfig)
		assert.True(t, cfg.ThinkingConfig.IncludeThoughts)
		require.NotNil(t, cfg.ThinkingConfig.ThinkingBudget)
		assert.Equal(t, int32(20000), *cfg.ThinkingConfig.ThinkingBudget)
	})

	t.Run("extra_body_thinking_config_gemini3_level", func(t *testing.T) {
		req := &openai.OpenAIRequest{
			Model: "gemini-3.1-pro-preview",
			ExtraBody: map[string]interface{}{
				"thinking_config": map[string]interface{}{
					"thinking_level":   "high",
					"include_thoughts": true,
				},
			},
		}
		cfg := buildGenerationConfig(req, "gemini-3.1-pro-preview")
		require.NotNil(t, cfg)
		require.NotNil(t, cfg.ThinkingConfig)
		assert.True(t, cfg.ThinkingConfig.IncludeThoughts)
		assert.Equal(t, genai.ThinkingLevelHigh, cfg.ThinkingConfig.ThinkingLevel)
	})
}

func TestApplyExtraBodyToConfig(t *testing.T) {
	t.Run("nil extra_body does nothing", func(t *testing.T) {
		cfg := &genai.GenerationConfig{}
		applyExtraBodyToConfig(cfg, nil, "gemini-2.5-flash")
		// Config should remain empty
		assert.Empty(t, cfg.ResponseModalities)
		assert.Empty(t, cfg.ResponseMIMEType)
	})

	t.Run("with response_modalities in generation_config", func(t *testing.T) {
		cfg := &genai.GenerationConfig{}
		extraBody := map[string]interface{}{
			"generation_config": map[string]interface{}{
				"response_modalities": []interface{}{"TEXT", "IMAGE"},
			},
		}
		applyExtraBodyToConfig(cfg, extraBody, "gemini-2.5-flash")
		assert.Contains(t, cfg.ResponseModalities, genai.Modality("TEXT"))
		assert.Contains(t, cfg.ResponseModalities, genai.Modality("IMAGE"))
	})

	t.Run("with image_config in generation_config", func(t *testing.T) {
		cfg := &genai.GenerationConfig{}
		extraBody := map[string]interface{}{
			"generation_config": map[string]interface{}{
				"image_config": map[string]interface{}{
					"aspectRatio": "1:1",
					"imageSize":   "1K",
				},
			},
		}
		imageConfig := applyExtraBodyToConfig(cfg, extraBody, "gemini-3.1-flash-image-preview")
		require.NotNil(t, imageConfig)
		assert.Equal(t, "1:1", imageConfig.AspectRatio)
		assert.Equal(t, "1K", imageConfig.ImageSize)
	})

	t.Run("with top-level modalities uppercased", func(t *testing.T) {
		cfg := &genai.GenerationConfig{}
		extraBody := map[string]interface{}{
			"modalities": []interface{}{"text", "audio"},
		}
		applyExtraBodyToConfig(cfg, extraBody, "gemini-2.5-flash")
		assert.Contains(t, cfg.ResponseModalities, genai.Modality("TEXT"))
		assert.Contains(t, cfg.ResponseModalities, genai.Modality("AUDIO"))
	})

	t.Run("with generation_config fields", func(t *testing.T) {
		cfg := &genai.GenerationConfig{}
		extraBody := map[string]interface{}{
			"generation_config": map[string]interface{}{
				"top_k":              float64(40),
				"seed":               float64(42),
				"temperature":        float64(0.7),
				"response_mime_type": "application/json",
			},
		}
		applyExtraBodyToConfig(cfg, extraBody, "gemini-2.5-flash")
		require.NotNil(t, cfg.TopK)
		assert.Equal(t, float32(40), *cfg.TopK)
		require.NotNil(t, cfg.Seed)
		assert.Equal(t, int32(42), *cfg.Seed)
		require.NotNil(t, cfg.Temperature)
		assert.Equal(t, float32(0.7), *cfg.Temperature)
		assert.Equal(t, "application/json", cfg.ResponseMIMEType)
	})

	t.Run("response_mime_type skipped for image models", func(t *testing.T) {
		cfg := &genai.GenerationConfig{}
		extraBody := map[string]interface{}{
			"generation_config": map[string]interface{}{
				"response_mime_type": "application/json",
			},
		}
		applyExtraBodyToConfig(cfg, extraBody, "imagen-3.0-generate-001")
		assert.Empty(t, cfg.ResponseMIMEType)
	})
}

func TestMapAudioParam(t *testing.T) {
	t.Run("nil param returns nil", func(t *testing.T) {
		result := mapAudioParam(nil)
		assert.Nil(t, result)
	})

	t.Run("non-map param returns nil", func(t *testing.T) {
		result := mapAudioParam("invalid")
		assert.Nil(t, result)
	})

	t.Run("valid param with voice", func(t *testing.T) {
		param := map[string]interface{}{
			"voice":  "alloy",
			"format": "wav",
		}
		result := mapAudioParam(param)
		require.NotNil(t, result)
		require.NotNil(t, result.VoiceConfig)
		require.NotNil(t, result.VoiceConfig.PrebuiltVoiceConfig)
		assert.Equal(t, "alloy", result.VoiceConfig.PrebuiltVoiceConfig.VoiceName)
	})

	t.Run("empty voice does not set VoiceConfig", func(t *testing.T) {
		param := map[string]interface{}{
			"voice":  "",
			"format": "wav",
		}
		result := mapAudioParam(param)
		require.NotNil(t, result)
		assert.Nil(t, result.VoiceConfig)
	})

	t.Run("no voice key does not set VoiceConfig", func(t *testing.T) {
		param := map[string]interface{}{
			"format": "wav",
		}
		result := mapAudioParam(param)
		require.NotNil(t, result)
		assert.Nil(t, result.VoiceConfig)
	})
}
