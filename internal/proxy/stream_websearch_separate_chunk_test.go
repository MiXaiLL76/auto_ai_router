package proxy

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestHandleStreamingWithTokens_WebSearchSignalInSeparateChunkFromUsage is a
// regression test for the SSE usage-prefilter gap found in review: a chunk
// carrying only web-search evidence (choices[].message.annotations, no
// "usage" key) must not be silently skipped just because it doesn't look
// like a usage chunk (see chunkMayCarryTokenUsage).
func TestHandleStreamingWithTokens_WebSearchSignalInSeparateChunkFromUsage(t *testing.T) {
	upstreamServer := newIPv4Server(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Two SEPARATE frames: one carries only the web-search annotation (no
		// "usage" key at all), the other carries only usage (no annotations) —
		// the exact split-frame shape the review found the old "usage"-only
		// prefilter would silently drop.
		chunks := []string{
			`data: {"choices":[{"message":{"annotations":[{"type":"url_citation"}]}}]}` + "\n\n",
			`data: {"choices":[{"delta":{"content":"hello"}}]}` + "\n\n",
			`data: {"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n",
			"data: [DONE]\n\n",
		}

		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		for _, chunk := range chunks {
			_, _ = fmt.Fprint(w, chunk)
			flusher.Flush()
			time.Sleep(time.Millisecond)
		}
	}))
	defer upstreamServer.Close()

	prx := NewTestProxyBuilder().
		WithSingleCredential("test", config.ProviderTypeProxy, upstreamServer.URL, "upstream-key-1").
		WithRequestTimeout(5 * time.Second).
		Build()

	resp, err := http.Get(upstreamServer.URL)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()

	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{
		RequestID:  "test-req-websearch-split",
		Credential: &config.CredentialConfig{Name: "test-cred", Type: config.ProviderTypeOpenAI},
	}

	err = prx.handleStreamingWithTokens(w, resp, "test-cred", "gpt-4o-mini", logCtx)
	require.NoError(t, err)

	require.NotNil(t, logCtx.TokenUsage)
	assert.Equal(t, 10, logCtx.TokenUsage.PromptTokens)
	assert.Equal(t, 5, logCtx.TokenUsage.CompletionTokens)
	assert.Equal(t, 1, logCtx.TokenUsage.WebSearchRequests,
		"WebSearchRequests from the annotations-only chunk must survive even though a later chunk's usage overwrote other fields")
}
