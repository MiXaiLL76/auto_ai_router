package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAnthropicToOpenAI_PreservesCacheCreationTTLDetails(t *testing.T) {
	body := []byte(`{
		"id":"msg_cache_ttl","type":"message","role":"assistant","model":"claude-test",
		"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
		"usage":{
			"input_tokens":100,"output_tokens":10,"cache_read_input_tokens":80,
			"cache_creation_input_tokens":20,
			"cache_creation":{"ephemeral_5m_input_tokens":5,"ephemeral_1h_input_tokens":15}
		}
	}`)

	converted, err := AnthropicToOpenAI(body, "claude-test")
	require.NoError(t, err)

	var response struct {
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			PromptTokensDetails struct {
				CacheCreationTokens       int `json:"cache_creation_tokens"`
				CacheCreationTokenDetails struct {
					Ephemeral5mInputTokens int `json:"ephemeral_5m_input_tokens"`
					Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
				} `json:"cache_creation_token_details"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(converted, &response))
	assert.Equal(t, 200, response.Usage.PromptTokens)
	assert.Equal(t, 20, response.Usage.PromptTokensDetails.CacheCreationTokens)
	assert.Equal(t, 5, response.Usage.PromptTokensDetails.CacheCreationTokenDetails.Ephemeral5mInputTokens)
	assert.Equal(t, 15, response.Usage.PromptTokensDetails.CacheCreationTokenDetails.Ephemeral1hInputTokens)
}

func TestAnthropicToOpenAI_DerivesMissingCacheCreationAggregateFromTTLDetails(t *testing.T) {
	usage := convertAnthropicUsageToOpenAI(&AnthropicUsage{
		InputTokens: 10,
		CacheCreation: &CacheCreationDetails{
			Ephemeral5mInputTokens: 3,
			Ephemeral1hInputTokens: 7,
		},
	})

	require.NotNil(t, usage)
	assert.Equal(t, 20, usage.PromptTokens)
	require.NotNil(t, usage.PromptTokensDetails)
	assert.Equal(t, 10, usage.PromptTokensDetails.CacheCreationTokens)
}

func TestNormalizeCacheCreationUsage_PreservesNonzeroAggregateWhenDetailsAreMalformed(t *testing.T) {
	total, fiveMinutes, oneHour := NormalizeCacheCreationUsage(10, &CacheCreationDetails{
		Ephemeral5mInputTokens: 8,
		Ephemeral1hInputTokens: 8,
	})

	assert.Equal(t, 10, total)
	assert.Equal(t, 8, fiveMinutes)
	assert.Equal(t, 8, oneHour)
}
