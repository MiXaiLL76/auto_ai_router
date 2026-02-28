package vertex

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

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
