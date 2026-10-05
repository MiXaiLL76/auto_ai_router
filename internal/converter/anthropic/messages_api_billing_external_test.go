// Package anthropic_test holds full-path billing regression tests that need
// both the anthropic converter and internal/models — an external test
// package (distinct from the internal "anthropic" test package) is required
// here because internal/converter imports internal/converter/anthropic, so
// a package-"anthropic" test file importing internal/converter directly
// would form an import cycle.
package anthropic_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	anthropicconv "github.com/mixaill76/auto_ai_router/internal/converter/anthropic"
	"github.com/mixaill76/auto_ai_router/internal/models"
)

// explicitCacheModelPrice is the shared tariff used by both full-path tests
// below: implicit and explicit cache-read rates are intentionally different
// so a request that bills at the wrong one is caught immediately.
func explicitCacheModelPrice() *models.ModelPrice {
	return &models.ModelPrice{
		InputCostPerToken:               0.0000003,
		OutputCostPerToken:              0.0000012,
		CacheReadInputTokenCost:         0.00000015, // implicit — must NOT be used
		ExplicitCacheReadInputTokenCost: 0.000000075,
		CacheCreationInputTokenCost:     0.0000009,
	}
}

func TestChatToMessages_AlibabaExplicitCacheBillsAtExplicitTariff(t *testing.T) {
	// Full path: Alibaba Chat Completions response -> Messages API response ->
	// the same extraction/costing billing actually runs
	// (converter.ExtractTokenUsage -> models.CalculateTokenCosts).
	// Regression test for non-streaming /v1/messages explicit-cache requests
	// silently billing at the implicit tariff.
	body := []byte(`{
		"id":"chatcmpl-1",
		"model":"qwen3.7-flash",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"hi"},
			"finish_reason":"stop"
		}],
		"usage":{
			"prompt_tokens":1827,
			"completion_tokens":511,
			"total_tokens":2338,
			"prompt_tokens_details":{"cached_tokens":1486,"cache_type":"ephemeral","cache_creation_input_tokens":335,"cache_write_tokens":335,"cache_creation":{"ephemeral_5m_input_tokens":335}}
		}
	}`)

	converted, err := anthropicconv.ChatToMessages(body, anthropicconv.MessagesAdapterMetadata{})
	require.NoError(t, err)

	// This is what proxy billing actually does with the converted body
	// (see extractOpenAITokensAndUsage -> converter.ExtractTokenUsageWithOptions).
	usage := converter.ExtractTokenUsage(converted)
	require.NotNil(t, usage)
	require.Equal(t, "ephemeral", usage.CacheType)
	require.Equal(t, 1486, usage.CachedInputTokens)
	require.Equal(t, 335, usage.CacheCreationTokens)
	require.Equal(t, 335, usage.CacheCreation5mTokens)

	costs := models.CalculateTokenCosts(usage, explicitCacheModelPrice())
	require.NotNil(t, costs)
	assert.InDelta(t, 1486*0.000000075, costs.ExplicitCachedInputCost, 1e-15)
	assert.Zero(t, costs.CachedInputCost, "explicit-cache /v1/messages request must not bill the implicit tariff")
}

func TestTransformChatStreamToMessages_AlibabaExplicitCacheBillsAtExplicitTariff(t *testing.T) {
	// Full path: Alibaba Chat Completions SSE -> Messages API SSE -> the same
	// extraction/costing billing actually runs on the final message_delta
	// event (converter.ExtractTokenUsage -> models.CalculateTokenCosts).
	// Regression test for streaming /v1/messages explicit-cache requests
	// silently billing at the implicit tariff.
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","model":"qwen3.7-flash","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"qwen3.7-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		"",
		`data: {"id":"chatcmpl-1","model":"qwen3.7-flash","choices":[],"usage":{"prompt_tokens":1827,"completion_tokens":511,"total_tokens":2338,"prompt_tokens_details":{"cached_tokens":1486,"cache_type":"ephemeral","cache_creation_input_tokens":335,"cache_write_tokens":335,"cache_creation":{"ephemeral_5m_input_tokens":335}}}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	var output bytes.Buffer
	require.NoError(t, anthropicconv.TransformChatStreamToMessages(
		strings.NewReader(stream), &output, "qwen3.7-flash", anthropicconv.MessagesAdapterMetadata{},
	))

	got := output.String()
	deltaIdx := strings.Index(got, "event: message_delta\n")
	require.NotEqual(t, -1, deltaIdx)
	afterEvent := got[deltaIdx:]
	dataIdx := strings.Index(afterEvent, "data: ")
	require.NotEqual(t, -1, dataIdx)
	dataLine := afterEvent[dataIdx+6:]
	if endIdx := strings.Index(dataLine, "\n"); endIdx > 0 {
		dataLine = dataLine[:endIdx]
	}
	// Sanity-check we actually sliced a JSON object before handing it to the extractor.
	var sanityCheck map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(dataLine), &sanityCheck))

	// This is what proxy billing actually does with the emitted SSE event
	// (see extractTokenUsageFromPayloads -> converter.ExtractTokenUsageWithOptions).
	usage := converter.ExtractTokenUsage([]byte(dataLine))
	require.NotNil(t, usage)
	require.Equal(t, "ephemeral", usage.CacheType)
	require.Equal(t, 1486, usage.CachedInputTokens)
	require.Equal(t, 335, usage.CacheCreationTokens)
	require.Equal(t, 335, usage.CacheCreation5mTokens)

	costs := models.CalculateTokenCosts(usage, explicitCacheModelPrice())
	require.NotNil(t, costs)
	assert.InDelta(t, 1486*0.000000075, costs.ExplicitCachedInputCost, 1e-15)
	assert.Zero(t, costs.CachedInputCost, "streaming explicit-cache /v1/messages request must not bill the implicit tariff")
}
