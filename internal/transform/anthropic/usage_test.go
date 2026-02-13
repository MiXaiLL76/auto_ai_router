package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAnthropicUsage_BasicParsing(t *testing.T) {
	jsonData := `{
		"input_tokens": 100,
		"output_tokens": 50
	}`

	var usage AnthropicUsage
	err := json.Unmarshal([]byte(jsonData), &usage)
	assert.NoError(t, err)
	assert.Equal(t, 100, usage.InputTokens)
	assert.Equal(t, 50, usage.OutputTokens)
	assert.Equal(t, 0, usage.CacheCreationInputTokens)
	assert.Equal(t, 0, usage.CacheReadInputTokens)
}

func TestAnthropicUsage_WithCacheTokens(t *testing.T) {
	jsonData := `{
		"input_tokens": 100,
		"output_tokens": 50,
		"cache_creation_input_tokens": 20,
		"cache_read_input_tokens": 30
	}`

	var usage AnthropicUsage
	err := json.Unmarshal([]byte(jsonData), &usage)
	assert.NoError(t, err)
	assert.Equal(t, 100, usage.InputTokens)
	assert.Equal(t, 50, usage.OutputTokens)
	assert.Equal(t, 20, usage.CacheCreationInputTokens)
	assert.Equal(t, 30, usage.CacheReadInputTokens)
}

func TestAnthropicUsage_WithCacheCreationDetails(t *testing.T) {
	jsonData := `{
		"input_tokens": 100,
		"output_tokens": 50,
		"cache_creation": {
			"ephemeral_5m_input_tokens": 15,
			"ephemeral_1h_input_tokens": 5
		}
	}`

	var usage AnthropicUsage
	err := json.Unmarshal([]byte(jsonData), &usage)
	assert.NoError(t, err)
	assert.NotNil(t, usage.CacheCreation)
	assert.Equal(t, 15, usage.CacheCreation.Ephemeral5mInputTokens)
	assert.Equal(t, 5, usage.CacheCreation.Ephemeral1hInputTokens)
}

func TestAnthropicUsage_WithServerToolUse(t *testing.T) {
	jsonData := `{
		"input_tokens": 100,
		"output_tokens": 50,
		"server_tool_use": {
			"web_search_requests": 3
		}
	}`

	var usage AnthropicUsage
	err := json.Unmarshal([]byte(jsonData), &usage)
	assert.NoError(t, err)
	assert.NotNil(t, usage.ServerToolUse)
	assert.Equal(t, 3, usage.ServerToolUse.WebSearchRequests)
}

func TestAnthropicUsage_WithServiceTierAndGeo(t *testing.T) {
	jsonData := `{
		"input_tokens": 100,
		"output_tokens": 50,
		"service_tier": "priority",
		"inference_geo": "us-east-1"
	}`

	var usage AnthropicUsage
	err := json.Unmarshal([]byte(jsonData), &usage)
	assert.NoError(t, err)
	assert.Equal(t, "priority", usage.ServiceTier)
	assert.Equal(t, "us-east-1", usage.InferenceGeo)
}

func TestAnthropicUsage_CompleteResponse(t *testing.T) {
	jsonData := `{
		"input_tokens": 100,
		"output_tokens": 50,
		"cache_creation_input_tokens": 20,
		"cache_read_input_tokens": 30,
		"cache_creation": {
			"ephemeral_5m_input_tokens": 15,
			"ephemeral_1h_input_tokens": 5
		},
		"server_tool_use": {
			"web_search_requests": 3
		},
		"service_tier": "priority",
		"inference_geo": "eu-west-1"
	}`

	var usage AnthropicUsage
	err := json.Unmarshal([]byte(jsonData), &usage)
	assert.NoError(t, err)

	// Verify all fields are properly set
	assert.Equal(t, 100, usage.InputTokens)
	assert.Equal(t, 50, usage.OutputTokens)
	assert.Equal(t, 20, usage.CacheCreationInputTokens)
	assert.Equal(t, 30, usage.CacheReadInputTokens)
	assert.NotNil(t, usage.CacheCreation)
	assert.Equal(t, 15, usage.CacheCreation.Ephemeral5mInputTokens)
	assert.NotNil(t, usage.ServerToolUse)
	assert.Equal(t, 3, usage.ServerToolUse.WebSearchRequests)
	assert.Equal(t, "priority", usage.ServiceTier)
	assert.Equal(t, "eu-west-1", usage.InferenceGeo)
}

func TestAnthropicUsage_ToTokenUsage_Basic(t *testing.T) {
	usage := &AnthropicUsage{
		InputTokens:  100,
		OutputTokens: 50,
	}

	tokenUsage := usage.ToTokenUsage()
	assert.NotNil(t, tokenUsage)
	assert.Equal(t, 100, tokenUsage.PromptTokens)
	assert.Equal(t, 50, tokenUsage.CompletionTokens)
	assert.Equal(t, 0, tokenUsage.CachedInputTokens)
}

func TestAnthropicUsage_ToTokenUsage_WithCache(t *testing.T) {
	usage := &AnthropicUsage{
		InputTokens:              100,
		OutputTokens:             50,
		CacheCreationInputTokens: 20,
		CacheReadInputTokens:     30,
	}

	tokenUsage := usage.ToTokenUsage()
	assert.NotNil(t, tokenUsage)
	assert.Equal(t, 100, tokenUsage.PromptTokens)
	assert.Equal(t, 50, tokenUsage.CompletionTokens)
	// Total cached = creation + read
	assert.Equal(t, 50, tokenUsage.CachedInputTokens)
}

func TestAnthropicUsage_ToTokenUsage_Nil(t *testing.T) {
	var usage *AnthropicUsage
	tokenUsage := usage.ToTokenUsage()
	assert.Nil(t, tokenUsage)
}

func TestCacheCreationDetails_JSON(t *testing.T) {
	details := &CacheCreationDetails{
		Ephemeral5mInputTokens: 10,
		Ephemeral1hInputTokens: 20,
	}

	data, err := json.Marshal(details)
	assert.NoError(t, err)

	var unmarshaled CacheCreationDetails
	err = json.Unmarshal(data, &unmarshaled)
	assert.NoError(t, err)
	assert.Equal(t, 10, unmarshaled.Ephemeral5mInputTokens)
	assert.Equal(t, 20, unmarshaled.Ephemeral1hInputTokens)
}

func TestServerToolUsageDetails_JSON(t *testing.T) {
	details := &ServerToolUsageDetails{
		WebSearchRequests: 5,
	}

	data, err := json.Marshal(details)
	assert.NoError(t, err)

	var unmarshaled ServerToolUsageDetails
	err = json.Unmarshal(data, &unmarshaled)
	assert.NoError(t, err)
	assert.Equal(t, 5, unmarshaled.WebSearchRequests)
}

func TestAnthropicUsage_ServiceTiers(t *testing.T) {
	tests := []string{
		"standard",
		"priority",
		"batch",
	}

	for _, tier := range tests {
		jsonData := `{
			"input_tokens": 100,
			"output_tokens": 50,
			"service_tier": "` + tier + `"
		}`

		var usage AnthropicUsage
		err := json.Unmarshal([]byte(jsonData), &usage)
		assert.NoError(t, err)
		assert.Equal(t, tier, usage.ServiceTier)
	}
}

func TestAnthropicUsage_PartialCacheCreation(t *testing.T) {
	// Only 5m cache
	jsonData := `{
		"input_tokens": 100,
		"output_tokens": 50,
		"cache_creation": {
			"ephemeral_5m_input_tokens": 15
		}
	}`

	var usage AnthropicUsage
	err := json.Unmarshal([]byte(jsonData), &usage)
	assert.NoError(t, err)
	assert.NotNil(t, usage.CacheCreation)
	assert.Equal(t, 15, usage.CacheCreation.Ephemeral5mInputTokens)
	assert.Equal(t, 0, usage.CacheCreation.Ephemeral1hInputTokens)
}
