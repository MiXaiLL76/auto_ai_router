package proxy

import (
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
