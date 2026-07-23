package proxy

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetHopByHopHeaders(t *testing.T) {
	headers := GetHopByHopHeaders()

	// Should contain all 8 RFC 7230 hop-by-hop headers
	expectedHeaders := []string{
		"Connection",
		"Keep-Alive",
		"Proxy-Authenticate",
		"Proxy-Authorization",
		"TE",
		"Trailer",
		"Transfer-Encoding",
		"Upgrade",
	}

	assert.Len(t, headers, len(expectedHeaders))
	for _, h := range expectedHeaders {
		assert.True(t, headers[h], "should contain %s", h)
	}

	// Verify it returns a copy (modifying it doesn't affect the original)
	headers["X-Custom"] = true
	original := GetHopByHopHeaders()
	_, hasCustom := original["X-Custom"]
	assert.False(t, hasCustom, "modifying returned map should not affect the original")
}

func TestCopyRequestHeadersStripsInternalUsageContractHeader(t *testing.T) {
	src := httptestRequestWithHeaders(map[string]string{
		HeaderAIRUsageAudioTokens: "exclude-cached",
		HeaderAIRProxyClient:      "1",
		"X-Regular":               "ok",
	})
	dst, err := http.NewRequest(http.MethodPost, "http://upstream.example/v1/chat/completions", nil)
	assert.NoError(t, err)

	copyRequestHeaders(dst, src, "")

	assert.Empty(t, dst.Header.Get(HeaderAIRUsageAudioTokens))
	assert.Empty(t, dst.Header.Get(HeaderAIRProxyClient))
	assert.Equal(t, "ok", dst.Header.Get("X-Regular"))
}

func httptestRequestWithHeaders(headers map[string]string) *http.Request {
	req, _ := http.NewRequest(http.MethodPost, "http://router.example/v1/chat/completions", nil)
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	return req
}
