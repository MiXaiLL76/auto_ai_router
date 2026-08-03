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

func TestUsageNormalizerLeavesOtherModelsUntouched(t *testing.T) {
	body := []byte(`{"usage":{"completion_tokens":134,"completion_tokens_details":{"text_tokens":134,"reasoning_tokens":127}}}`)

	normalized, changed := NormalizeCompletionUsage(body, "gpt-5.6")

	assert.False(t, changed)
	assert.Equal(t, body, normalized)
}
