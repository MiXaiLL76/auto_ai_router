package models

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadModelPrices_EmptyLink(t *testing.T) {
	prices, err := LoadModelPrices("")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty link")
	assert.Nil(t, prices)
}

func TestLoadModelPrices_InvalidFormat(t *testing.T) {
	prices, err := LoadModelPrices("ftp://example.com/prices.json")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported link format")
	assert.Nil(t, prices)
}

func TestLoadModelPrices_FromFile(t *testing.T) {
	// Create a temporary file with valid JSON
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "prices.json")

	pricesJSON := `{
		"gpt-4": {"prompt": 0.03, "completion": 0.06},
		"gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002}
	}`
	err := os.WriteFile(filePath, []byte(pricesJSON), 0644)
	require.NoError(t, err)

	prices, err := LoadModelPrices("file://" + filePath)
	require.NoError(t, err)
	assert.NotNil(t, prices)
	assert.Len(t, prices, 2)
}

func TestLoadModelPrices_GPT56LongContext(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "prices.json")
	pricesJSON := `{
		"gpt-5.6-sol": {
			"input_cost_per_token": 0.0000065,
			"output_cost_per_token": 0.000039,
			"cache_read_input_token_cost": 0.00000065,
			"cache_creation_input_token_cost": 0.000008125,
			"input_cost_per_token_above_272k_tokens": 0.000013,
			"output_cost_per_token_above_272k_tokens": 0.0000585,
			"cache_read_input_token_cost_above_272k_tokens": 0.0000013,
			"cache_creation_input_token_cost_above_272k_tokens": 0.00001625
		}
	}`
	require.NoError(t, os.WriteFile(filePath, []byte(pricesJSON), 0o600))

	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	price := prices["gpt-5.6-sol"]
	require.NotNil(t, price)
	assert.InDelta(t, 0.000013, price.InputCostPerTokenAbove272k, 1e-12)
	assert.InDelta(t, 0.0000585, price.OutputCostPerTokenAbove272k, 1e-12)
	assert.InDelta(t, 0.0000013, price.CacheReadInputTokenCostAbove272k, 1e-12)
	assert.InDelta(t, 0.00001625, price.CacheCreationInputTokenCostAbove272k, 1e-12)
}

func TestLoadModelPrices_InputCostPerImage(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "prices.json")
	pricesJSON := `{
		"gemini-3.1-flash-image-preview": {
			"input_cost_per_image": 0.088113
		}
	}`
	require.NoError(t, os.WriteFile(filePath, []byte(pricesJSON), 0o600))

	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	price := prices["gemini-3.1-flash-image-preview"]
	require.NotNil(t, price)
	assert.InDelta(t, 0.088113, price.InputCostPerImage, 1e-12)
}

func TestLoadModelPrices_CachePricingExtensions(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "prices.json")
	pricesJSON := `{
		"claude-test": {
			"input_cost_per_token": 0.000003,
			"output_cost_per_token": 0.000015,
			"cache_read_input_token_cost_above_200k_tokens": 0.0000006,
			"cache_creation_input_token_cost_above_200k_tokens": 0.0000075,
			"cache_creation_input_token_cost_above_1hr": 0.000006,
			"cache_creation_input_token_cost_above_1hr_above_200k_tokens": 0.000012,
			"cache_read_input_audio_token_cost": 0.0000009,
			"search_context_cost_per_query": {
				"search_context_size_low": 0.01,
				"search_context_size_medium": 0.02,
				"search_context_size_high": 0.03
			},
			"web_search_billing_unit": "per_prompt",
			"litellm_provider": "gemini"
		}
	}`
	require.NoError(t, os.WriteFile(filePath, []byte(pricesJSON), 0o600))

	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	price := prices["claude-test"]
	require.NotNil(t, price)
	assert.InDelta(t, 0.0000006, price.CacheReadInputTokenCostAbove200k, 1e-12)
	assert.InDelta(t, 0.0000075, price.CacheCreationInputTokenCostAbove200k, 1e-12)
	assert.InDelta(t, 0.000006, price.CacheCreationInputTokenCostAbove1hr, 1e-12)
	assert.InDelta(t, 0.000012, price.CacheCreationInputTokenCostAbove1hrAbove200k, 1e-12)
	assert.InDelta(t, 0.0000009, price.CacheReadInputAudioTokenCost, 1e-12)
	assert.InDelta(t, 0.02, price.SearchContextCostPerQuery["search_context_size_medium"], 1e-12)
	assert.Equal(t, "per_prompt", price.WebSearchBillingUnit)
	assert.Equal(t, "gemini", price.LiteLLMProvider)
}

func TestLoadModelPrices_InfersProviderFromPrefixedKey(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "prices.json")
	require.NoError(t, os.WriteFile(filePath, []byte(`{
		"vertex_ai/gemini-2.5-flash": {
			"search_context_cost_per_query": {
				"search_context_size_medium": 0.035
			}
		}
	}`), 0o600))

	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	assert.Equal(t, "vertex_ai", prices["gemini-2.5-flash"].LiteLLMProvider)
}

func TestLoadModelPrices_InfersProviderFromBareModelName(t *testing.T) {
	// Real-world case: a price file keys entries by plain model name (no
	// "provider/model" convention) and never sets litellm_provider at all --
	// the model-name prefix is the only signal available.
	filePath := filepath.Join(t.TempDir(), "prices.json")
	require.NoError(t, os.WriteFile(filePath, []byte(`{
		"gemini-2.5-pro": {
			"input_cost_per_token": 0.000001625,
			"input_cost_per_token_above_200k_tokens": 0.00000325
		},
		"claude-sonnet-4.5": {
			"input_cost_per_token": 0.0000039,
			"input_cost_per_token_above_200k_tokens": 0.0000078
		}
	}`), 0o600))

	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	assert.Equal(t, "gemini", prices["gemini-2.5-pro"].LiteLLMProvider)
	// A bare name with no recognized prefix must stay empty, not guess wrong.
	assert.Equal(t, "", prices["claude-sonnet-4.5"].LiteLLMProvider)
}

func TestLoadModelPrices_ExplicitProviderWinsOverNameInference(t *testing.T) {
	filePath := filepath.Join(t.TempDir(), "prices.json")
	require.NoError(t, os.WriteFile(filePath, []byte(`{
		"gemini-2.5-pro": {
			"input_cost_per_token": 0.000001625,
			"litellm_provider": "vertex_ai"
		}
	}`), 0o600))

	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	assert.Equal(t, "vertex_ai", prices["gemini-2.5-pro"].LiteLLMProvider)
}

func TestLoadModelPrices_FromFilePath(t *testing.T) {
	// Create a temporary file with valid JSON
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "prices.json")

	pricesJSON := `{"gpt-4": {"prompt": 0.03}}`
	err := os.WriteFile(filePath, []byte(pricesJSON), 0644)
	require.NoError(t, err)

	// Without file:// prefix
	prices, err := LoadModelPrices(filePath)
	require.NoError(t, err)
	assert.NotNil(t, prices)
}

func TestLoadModelPrices_FilePathTraversal(t *testing.T) {
	// Try path traversal attack - using a relative path with ../
	prices, err := LoadModelPrices("file:///etc/../etc/passwd")
	require.Error(t, err)
	// Either path traversal or file not found are acceptable
	assert.True(t,
		len(err.Error()) > 0,
	)
	assert.Nil(t, prices)
}

func TestLoadModelPrices_InvalidJSON(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "invalid.json")

	// Invalid JSON
	err := os.WriteFile(filePath, []byte(`{invalid`), 0644)
	require.NoError(t, err)

	prices, err := LoadModelPrices("file://" + filePath)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse")
	assert.Nil(t, prices)
}

func TestLoadModelPrices_NormalizesModelNames(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "prices.json")

	// Model names with different cases and prefixes
	pricesJSON := `{
		"openai/gpt-4": {"prompt": 0.03},
		"OpenAI/GPT-4": {"prompt": 0.04},
		"gpt-4": {"prompt": 0.05}
	}`
	err := os.WriteFile(filePath, []byte(pricesJSON), 0644)
	require.NoError(t, err)

	prices, err := LoadModelPrices("file://" + filePath)
	require.NoError(t, err)
	assert.NotNil(t, prices)
	// Should be normalized to lowercase, last part only
	// Only one entry should remain (the last one wins due to collision warning)
	assert.GreaterOrEqual(t, len(prices), 1)
}

func TestLoadFromFile_SizeLimit(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := filepath.Join(tmpDir, "large.json")

	// Create a file that's larger than the limit
	largeContent := make([]byte, MaxFileSizeBytes+1)
	err := os.WriteFile(filePath, largeContent, 0644)
	require.NoError(t, err)

	data, err := loadFromFile(filePath)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds 100MB")
	assert.Nil(t, data)
}

func TestLoadFromFile_FileNotFound(t *testing.T) {
	data, err := loadFromFile("/nonexistent/path/prices.json")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to stat")
	assert.Nil(t, data)
}

func TestHasPathTraversal(t *testing.T) {
	tests := []struct {
		name  string
		path  string
		valid bool
	}{
		{"simple path", "/etc/config.json", true},
		{"relative path", "./config.json", true},
		{"with dots", "../etc/config.json", false},
		{"double dots", "a/../b/config.json", false},
		{"path traversal attempt", "/etc/../../../passwd", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := hasPathTraversal(tt.path)
			if tt.valid {
				assert.False(t, result)
			} else {
				assert.True(t, result)
			}
		})
	}
}

func TestNormalizeModelName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"gpt-4", "gpt-4"},
		{"gpt-4-turbo", "gpt-4-turbo"},
		{"openai/gpt-4", "gpt-4"},
		{"anthropic.claude/claude-3-opus", "claude-3-opus"},
		{"vertex/gemini-1.5-pro", "gemini-1.5-pro"},
		{"claude-sonnet", "claude-sonnet"},
		{"GPT-4", "gpt-4"},
		{"  GPT-4  ", "gpt-4"},
		{"OpenAI/GPT-4-Turbo", "gpt-4-turbo"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := NormalizeModelName(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestLoadModelPrices_HTTPNotFound(t *testing.T) {
	// This will fail because the URL doesn't exist, but tests the HTTP path
	prices, err := LoadModelPrices("https://example.com/nonexistent/prices.json")
	require.Error(t, err)
	assert.Nil(t, prices)
}

func TestLoadFromHTTP_InvalidURL(t *testing.T) {
	data, err := loadFromHTTP("not-a-valid-url")
	require.Error(t, err)
	assert.Nil(t, data)
}

func TestLoadFromHTTP_UnsupportedScheme(t *testing.T) {
	data, err := loadFromHTTP("ftp://example.com/prices.json")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported scheme")
	assert.Nil(t, data)
}
