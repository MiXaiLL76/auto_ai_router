package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mixaill76/auto_ai_router/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestMarkAudioUsageContractForClient_HidesFromUntrustedClient(t *testing.T) {
	w := httptest.NewRecorder()

	NewTestProxyBuilder().Build().markAudioUsageContractForClient(w, nil, true)

	assert.Empty(t, w.Header().Get(HeaderAIRUsageAudioTokens))
}

func TestMarkAudioUsageContractForClient_AllowlistHidesFromUntrustedClient(t *testing.T) {
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{IsProxyRequest: false}

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		markAudioUsageContractForClient(w, logCtx, true)

	assert.Empty(t, w.Header().Get(HeaderAIRUsageAudioTokens))
}

func TestMarkAudioUsageContractForClient_AllowlistKeepsForTrustedProxyPeer(t *testing.T) {
	w := httptest.NewRecorder()
	logCtx := &RequestLogContext{IsProxyRequest: true}

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		markAudioUsageContractForClient(w, logCtx, false)

	assert.Equal(t, "exclude-cached", w.Header().Get(HeaderAIRUsageAudioTokens))
}

func TestMarkAudioUsageContractForClient_AllowlistClearsStaleValueForUntrustedClient(t *testing.T) {
	w := httptest.NewRecorder()
	w.Header().Set(HeaderAIRUsageAudioTokens, "include-cached")

	NewTestProxyBuilder().
		WithResponseHeaderMode(config.ResponseHeaderModeAllowlist).
		Build().
		markAudioUsageContractForClient(w, nil, true)

	assert.Empty(t, w.Header().Get(HeaderAIRUsageAudioTokens))
}

func TestKimiCacheWriteTTLFromHeaders(t *testing.T) {
	t.Run("parses both TTL headers", func(t *testing.T) {
		h := http.Header{}
		h.Set(HeaderKimiCacheWriteTokens5m, "200")
		h.Set(HeaderKimiCacheWriteTokens1h, "800")

		fiveMin, oneHour := kimiCacheWriteTTLFromHeaders(h)

		assert.Equal(t, 200, fiveMin)
		assert.Equal(t, 800, oneHour)
	})

	t.Run("default cache write only sets 5m header", func(t *testing.T) {
		h := http.Header{}
		h.Set(HeaderKimiCacheWriteTokens5m, "1234")
		h.Set(HeaderKimiCacheWriteTokens1h, "0")

		fiveMin, oneHour := kimiCacheWriteTTLFromHeaders(h)

		assert.Equal(t, 1234, fiveMin)
		assert.Equal(t, 0, oneHour)
	})

	t.Run("missing headers yield zero", func(t *testing.T) {
		fiveMin, oneHour := kimiCacheWriteTTLFromHeaders(http.Header{})
		assert.Equal(t, 0, fiveMin)
		assert.Equal(t, 0, oneHour)
	})

	t.Run("nil headers yield zero", func(t *testing.T) {
		fiveMin, oneHour := kimiCacheWriteTTLFromHeaders(nil)
		assert.Equal(t, 0, fiveMin)
		assert.Equal(t, 0, oneHour)
	})

	t.Run("malformed header value ignored", func(t *testing.T) {
		h := http.Header{}
		h.Set(HeaderKimiCacheWriteTokens5m, "not-a-number")

		fiveMin, oneHour := kimiCacheWriteTTLFromHeaders(h)

		assert.Equal(t, 0, fiveMin)
		assert.Equal(t, 0, oneHour)
	})

	t.Run("tokenUsageExtractionOptionsForResponse folds headers in regardless of credential type", func(t *testing.T) {
		h := http.Header{}
		h.Set(HeaderKimiCacheWriteTokens5m, "200")
		h.Set(HeaderKimiCacheWriteTokens1h, "800")

		opts := tokenUsageExtractionOptionsForResponse(nil, h)

		assert.Equal(t, 200, opts.CacheWriteTTLHeader5mTokens)
		assert.Equal(t, 800, opts.CacheWriteTTLHeader1hTokens)
	})
}
