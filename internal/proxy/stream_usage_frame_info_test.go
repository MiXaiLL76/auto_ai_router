package proxy

import (
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/converter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// countingStreamUsageExtractor counts the JSON decodes so a test can pin that a
// full usage frame is decoded once, not twice (see streamUsageFrameInfo).
type countingStreamUsageExtractor struct {
	inner StreamUsageExtractor
	calls int
}

func (c *countingStreamUsageExtractor) ExtractUsage(chunk []byte) *StreamUsageInfo {
	c.calls++
	return c.inner.ExtractUsage(chunk)
}

// TestStreamUsageFrameInfoSkipsExtractorForFullCounts pins the single-parse
// path: when one payload reported a positive prompt and completion count, both
// fields are present by construction, so the extractor must not re-decode the
// bytes the converter already decoded.
func TestStreamUsageFrameInfoSkipsExtractorForFullCounts(t *testing.T) {
	extractor := &countingStreamUsageExtractor{inner: &openAIStreamUsageExtractor{}}
	usage := &converter.TokenUsage{PromptTokens: 900, CompletionTokens: 42}

	info := streamUsageFrameInfo(extractor, usage, true,
		[]byte(`data: {"usage":{"prompt_tokens":900,"completion_tokens":42}}`))

	require.NotNil(t, info)
	assert.Zero(t, extractor.calls, "the converter's positive counts already imply presence")
	assert.Equal(t, 900, info.PromptTokens)
	assert.True(t, info.PromptTokensPresent)
	assert.Equal(t, 42, info.CompletionTokens)
	assert.True(t, info.CompletionTokensPresent)
}

// TestStreamUsageFrameInfoParsesWhenPresenceMatters covers the shapes where the
// converter's values cannot stand in for presence: a reported zero, an absent
// field, an all-zero object (the converter returns no TokenUsage at all), and
// several payloads whose values the merge combines.
func TestStreamUsageFrameInfoParsesWhenPresenceMatters(t *testing.T) {
	tests := []struct {
		name          string
		usage         *converter.TokenUsage
		singlePayload bool
		chunk         string
		wantNil       bool
		wantPrompt    int
		wantCompl     int
	}{
		{
			name:          "reported zero completion",
			usage:         &converter.TokenUsage{PromptTokens: 900},
			singlePayload: true,
			chunk:         `data: {"usage":{"prompt_tokens":900,"completion_tokens":0}}`,
			wantPrompt:    900,
		},
		{
			name:          "all-zero usage object",
			usage:         nil,
			singlePayload: true,
			chunk:         `data: {"usage":{"prompt_tokens":0,"completion_tokens":0}}`,
		},
		{
			name:          "several payloads",
			usage:         &converter.TokenUsage{PromptTokens: 900, CompletionTokens: 42},
			singlePayload: false,
			chunk:         `data: {"usage":{"prompt_tokens":900,"completion_tokens":42}}`,
			wantPrompt:    900,
			wantCompl:     42,
		},
		{
			name:          "no usage object at all",
			usage:         &converter.TokenUsage{WebSearchRequests: 1},
			singlePayload: true,
			chunk:         `data: {"web_search_call":{"id":"ws_1"}}`,
			wantNil:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			extractor := &countingStreamUsageExtractor{inner: &openAIStreamUsageExtractor{}}

			info := streamUsageFrameInfo(extractor, tt.usage, tt.singlePayload, []byte(tt.chunk))

			assert.Equal(t, 1, extractor.calls, "the extractor is the only decoder that knows presence")
			if tt.wantNil {
				assert.Nil(t, info)
				return
			}
			require.NotNil(t, info)
			assert.Equal(t, tt.wantPrompt, info.PromptTokens)
			assert.Equal(t, tt.wantCompl, info.CompletionTokens)
			assert.True(t, info.PromptTokensPresent)
			assert.True(t, info.CompletionTokensPresent)
		})
	}
}
