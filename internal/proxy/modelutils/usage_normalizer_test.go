package modelutils

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNormalizeCompletionUsage(t *testing.T) {
	body := []byte(`{"usage":{"completion_tokens":134,"completion_tokens_details":{"text_tokens":134,"reasoning_tokens":127,"provider_detail":"kept"}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "qwen/qwen3.7-plus")

	require.True(t, changed)
	var response map[string]any
	require.NoError(t, json.Unmarshal(normalized, &response))
	details := response["usage"].(map[string]any)["completion_tokens_details"].(map[string]any)
	assert.Equal(t, float64(7), details["text_tokens"])
	assert.Equal(t, float64(127), details["reasoning_tokens"])
	assert.Equal(t, "kept", details["provider_detail"])
}

func TestNormalizeCompletionUsageLeavesValidPartitionUnchanged(t *testing.T) {
	body := []byte(`{"usage":{"completion_tokens":134,"completion_tokens_details":{"text_tokens":7,"reasoning_tokens":127}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "qwen3.7-plus")

	assert.False(t, changed)
	assert.Equal(t, body, normalized)
}

func TestNormalizeCompletionUsageAddsMissingTextTokens(t *testing.T) {
	body := []byte(`{"usage":{"completion_tokens":202,"completion_tokens_details":{"reasoning_tokens":194}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "qwen3.7-plus")

	require.True(t, changed)
	var response map[string]any
	require.NoError(t, json.Unmarshal(normalized, &response))
	details := response["usage"].(map[string]any)["completion_tokens_details"].(map[string]any)
	assert.Equal(t, float64(8), details["text_tokens"])
	assert.Equal(t, float64(194), details["reasoning_tokens"])
}

func TestNormalizeCompletionUsageReplacesNullTextTokens(t *testing.T) {
	body := []byte(`{"usage":{"completion_tokens":202,"completion_tokens_details":{"text_tokens":null,"reasoning_tokens":194}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "qwen3.7-plus")

	require.True(t, changed)
	var response map[string]any
	require.NoError(t, json.Unmarshal(normalized, &response))
	details := response["usage"].(map[string]any)["completion_tokens_details"].(map[string]any)
	assert.Equal(t, float64(8), details["text_tokens"])
}

func TestNormalizeCompletionUsageReplacesZeroTextTokens(t *testing.T) {
	body := []byte(`{"usage":{"completion_tokens":202.0,"completion_tokens_details":{"text_tokens":0,"reasoning_tokens":"194"}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "qwen3.7-plus")

	require.True(t, changed)
	var response map[string]any
	require.NoError(t, json.Unmarshal(normalized, &response))
	details := response["usage"].(map[string]any)["completion_tokens_details"].(map[string]any)
	assert.Equal(t, float64(8), details["text_tokens"])
}

func TestNormalizeCompletionUsageUsesResponseModel(t *testing.T) {
	body := []byte(`{"model":"qwen3.7-plus","usage":{"completion_tokens":202,"completion_tokens_details":{"text_tokens":202,"reasoning_tokens":194}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "provider-backend-alias")

	require.True(t, changed)
	assert.Contains(t, string(normalized), `"text_tokens":8`)
}

func TestUsageNormalizingReadCloser(t *testing.T) {
	stream := "event: message\r\n" +
		`data: {"usage":{"completion_tokens":134,"completion_tokens_details":{"text_tokens":134,"reasoning_tokens":127}}}` + "\r\n\r\n" +
		"data: [DONE]\r\n\r\n"

	reader, wrapped := NewUsageNormalizingReadCloser(io.NopCloser(strings.NewReader(stream)), "gateway/qwen3.7-plus")
	require.True(t, wrapped)
	output, err := io.ReadAll(reader)

	require.NoError(t, err)
	assert.Contains(t, string(output), `"text_tokens":7`)
	assert.Contains(t, string(output), `"reasoning_tokens":127`)
	assert.Contains(t, string(output), "data: [DONE]\r\n\r\n")
}

func TestUsageNormalizingReadCloserNormalizesCacheCreationForNonQwen(t *testing.T) {
	// Claude via a Requesty upstream, streamed: the final usage chunk carries the
	// cache write only under caching_tokens. The stream must be normalized even
	// though the model is not Qwen.
	stream := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\r\n" +
		`data: {"id":"rqsty-cmpl-abc","usage":{"prompt_tokens":3010,"completion_tokens":50,"total_tokens":3060,"prompt_tokens_details":{"cached_tokens":0,"caching_tokens":2755,"caching_token_details":{"caching_5m_tokens":2755,"caching_1h_tokens":0}}}}` + "\r\n\r\n" +
		"data: [DONE]\r\n\r\n"

	reader, wrapped := NewUsageNormalizingReadCloser(io.NopCloser(strings.NewReader(stream)), "anthropic/claude-sonnet-4.6")
	require.True(t, wrapped)
	output, err := io.ReadAll(reader)

	require.NoError(t, err)
	assert.Contains(t, string(output), `"cache_creation_tokens":2755`)
	assert.Contains(t, string(output), `"caching_tokens":2755`)
	assert.Contains(t, string(output), `"content":"hi"`)
	assert.Contains(t, string(output), "data: [DONE]\r\n\r\n")
}

func TestUsageNormalizerLeavesOtherModelsUntouched(t *testing.T) {
	body := []byte(`{"usage":{"completion_tokens":134,"completion_tokens_details":{"text_tokens":134,"reasoning_tokens":127}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "gpt-5.6")

	assert.False(t, changed)
	assert.Equal(t, body, normalized)
}

func TestNormalizeCacheCreationUsageFromRequestyNaming(t *testing.T) {
	// OpenAI→OpenAI passthrough where the upstream (Requesty/AIR) reported the
	// cache write only under caching_tokens. The standard cache_creation_tokens
	// must be synthesized so pricing jsonpaths match it.
	body := []byte(`{"id":"rqsty-cmpl-abc","usage":{"prompt_tokens":3010,"completion_tokens":50,"total_tokens":3060,"prompt_tokens_details":{"cached_tokens":0,"caching_tokens":2755,"caching_token_details":{"caching_5m_tokens":2755,"caching_1h_tokens":0}}}}`)

	normalized, changed := NormalizeCacheCreationUsage(body)

	require.True(t, changed)
	var response map[string]any
	require.NoError(t, json.Unmarshal(normalized, &response))
	details := response["usage"].(map[string]any)["prompt_tokens_details"].(map[string]any)
	// Standard naming added.
	assert.Equal(t, float64(2755), details["cache_creation_tokens"])
	ctd := details["cache_creation_token_details"].(map[string]any)
	assert.Equal(t, float64(2755), ctd["ephemeral_5m_input_tokens"])
	// Alternate naming preserved.
	assert.Equal(t, float64(2755), details["caching_tokens"])
}

func TestNormalizeCacheCreationUsageLeavesStandardNamingUnchanged(t *testing.T) {
	body := []byte(`{"usage":{"prompt_tokens_details":{"cache_creation_tokens":2755}}}`)

	normalized, changed := NormalizeCacheCreationUsage(body)

	assert.False(t, changed)
	assert.Equal(t, body, normalized)
}

func TestNormalizeCacheCreationUsageNoCacheWriteUnchanged(t *testing.T) {
	// Only a cache READ is present — no cache write to synthesize.
	body := []byte(`{"usage":{"prompt_tokens_details":{"cached_tokens":192317}}}`)

	normalized, changed := NormalizeCacheCreationUsage(body)

	assert.False(t, changed)
	assert.Equal(t, body, normalized)
}

func TestNormalizeCacheCreationUsageLeavesCacheWriteTokensUnchanged(t *testing.T) {
	// cache_write_tokens is matched directly by the pricing jsonpaths, so it
	// must NOT be rewritten (would risk double-counting).
	body := []byte(`{"usage":{"prompt_tokens_details":{"cache_write_tokens":1200}}}`)

	normalized, changed := NormalizeCacheCreationUsage(body)

	assert.False(t, changed)
	assert.Equal(t, body, normalized)
}

func TestNormalizeCacheCreationUsageLeavesAlibabaNamingUnchanged(t *testing.T) {
	// Alibaba's cache_creation.ephemeral_* is matched directly by the pricing
	// jsonpaths, so it must NOT be rewritten.
	body := []byte(`{"usage":{"prompt_tokens_details":{"cache_creation":{"ephemeral_5m_input_tokens":900,"ephemeral_1h_input_tokens":100}}}}`)

	normalized, changed := NormalizeCacheCreationUsage(body)

	assert.False(t, changed)
	assert.Equal(t, body, normalized)
}
