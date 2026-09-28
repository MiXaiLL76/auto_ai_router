package config

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestVisionFallbackConfig_UnmarshalYAML(t *testing.T) {
	t.Setenv("TEST_VISION_MODE", "STRIP")
	var cfg VisionFallbackConfig
	require.NoError(t, yaml.Unmarshal([]byte(`
mode: os.environ/TEST_VISION_MODE
describe_model: qwen-vl
describe_prompt: describe it
max_images: "3"
max_tokens: 256
timeout: 45s
`), &cfg))
	assert.Equal(t, VisionFallbackConfig{
		Mode: VisionFallbackStrip, DescribeModel: "qwen-vl", DescribePrompt: "describe it",
		MaxImages: 3, MaxTokens: 256, Timeout: 45 * time.Second,
	}, cfg)
}

func TestVisionFallbackConfig_UnmarshalYAMLErrors(t *testing.T) {
	for name, doc := range map[string]string{
		"not a mapping":  `[1, 2]`,
		"bad max_images": `max_images: many`,
		"bad max_tokens": `max_tokens: lots`,
		"bad timeout":    `timeout: soon`,
	} {
		t.Run(name, func(t *testing.T) {
			var cfg VisionFallbackConfig
			assert.Error(t, yaml.Unmarshal([]byte(doc), &cfg))
		})
	}
}

func TestVisionFallbackConfig_Defaults(t *testing.T) {
	cfg := VisionFallbackConfig{}
	cfg.ApplyDefaults()
	assert.Equal(t, VisionFallbackConfig{
		Mode: VisionFallbackReject, DescribePrompt: DefaultVisionDescribePrompt,
		MaxImages: DefaultVisionMaxImages, MaxTokens: DefaultVisionMaxTokens, Timeout: DefaultVisionTimeout,
	}, cfg)

	cfg = VisionFallbackConfig{DescribeModel: "qwen-vl"}
	cfg.ApplyDefaults()
	assert.Equal(t, VisionFallbackDescribe, cfg.Mode)

	cfg = VisionFallbackConfig{Mode: VisionFallbackStrip, DescribePrompt: "p", MaxImages: 1, MaxTokens: 2, Timeout: time.Second}
	cfg.ApplyDefaults()
	assert.Equal(t, VisionFallbackConfig{Mode: VisionFallbackStrip, DescribePrompt: "p", MaxImages: 1, MaxTokens: 2, Timeout: time.Second}, cfg,
		"explicit values are kept")
}

func TestVisionFallbackConfig_Validate(t *testing.T) {
	valid := []VisionFallbackConfig{
		{},
		{Mode: VisionFallbackReject},
		{Mode: VisionFallbackStrip},
		{Mode: VisionFallbackDescribe, DescribeModel: "qwen-vl"},
	}
	for _, cfg := range valid {
		assert.NoError(t, cfg.Validate(), "%+v", cfg)
	}
	invalid := []VisionFallbackConfig{
		{Mode: VisionFallbackDescribe},
		{Mode: "bogus"},
		{MaxImages: -1},
		{MaxTokens: -1},
		{Timeout: -time.Second},
	}
	for _, cfg := range invalid {
		assert.Error(t, cfg.Validate(), "%+v", cfg)
	}
}

func TestConfigValidate_VisionFallback(t *testing.T) {
	cfg := &Config{VisionFallback: VisionFallbackConfig{Mode: "bogus"}}
	assert.ErrorContains(t, cfg.Validate(), "vision_fallback.mode")
}

func TestModelRPMConfig_SupportsVision(t *testing.T) {
	t.Setenv("TEST_SUPPORTS_VISION", "false")
	var m ModelRPMConfig
	require.NoError(t, yaml.Unmarshal([]byte("name: glm\nsupports_vision: os.environ/TEST_SUPPORTS_VISION\n"), &m))
	require.NotNil(t, m.SupportsVision)
	assert.False(t, *m.SupportsVision)

	m = ModelRPMConfig{}
	require.NoError(t, yaml.Unmarshal([]byte("name: glm\n"), &m))
	assert.Nil(t, m.SupportsVision)

	assert.ErrorContains(t, yaml.Unmarshal([]byte("name: glm\nsupports_vision: maybe\n"), &m), "supports_vision")
}
